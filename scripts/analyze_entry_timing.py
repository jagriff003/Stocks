"""
Track C #1 — is the model late to its own winners, and is being earlier worth it?

    python scripts/analyze_entry_timing.py
    python scripts/analyze_entry_timing.py --early-rank 8 --sustain 3

The finding this tests: 60% of new-entrant selections had already spent 10+ of
the prior 40 sessions ranked in the top 8 before finally cracking the top 4 and
being bought.  Average entrant had been top-4 ranked for 7.3 of the prior 40
days before purchase.  Only 4.7% were genuinely fresh.

The hindsight trap, which this script is built to avoid: of ALL stock-days
ranked 5-8 — not just the ones that eventually won — only 49.5% reached the top
4 within 20 sessions.  A rule that buys earlier on rank alone roughly doubles
false-positive entries.  So the question is not "would earlier entry have caught
more of the winners" (it would, trivially) but:

    Across EVERY stock that triggers an early-entry rule — the ones that go on
    to be selected AND the ~50% that stall or reverse — what is the return
    difference against actual entry timing?

Measuring only the eventual winners answers a question nobody can trade.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.backtest import build_target_portfolios
from momentum.data import load_data
from momentum.exits import daily_ranks
from momentum.experiments import production_config
from momentum.strategy import compute_scores
from momentum.universe import current_symbols


def forward_return(close: pd.DataFrame, symbol: str, start, horizon: int):
    """Return of `symbol` over `horizon` sessions from `start`."""
    if symbol not in close.columns or start not in close.index:
        return np.nan
    i = close.index.get_loc(start)
    j = i + horizon
    if j >= len(close.index):
        return np.nan
    p0, p1 = close.iat[i, close.columns.get_loc(symbol)], \
             close.iat[j, close.columns.get_loc(symbol)]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return np.nan
    return p1 / p0 - 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Entry timing analysis")
    parser.add_argument("--early-rank", type=int, default=8,
                        help="rank that constitutes an early-entry signal")
    parser.add_argument("--sustain", type=int, default=3,
                        help="consecutive days at that rank to trigger")
    parser.add_argument("--lookahead", type=int, default=20,
                        help="sessions allowed to reach the top N")
    parser.add_argument("--horizon", type=int, default=21,
                        help="forward return horizon for the comparison")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config = production_config()
    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache, verbose=False)

    print("=" * 84)
    print("ENTRY TIMING — is earlier entry worth its false positives?")
    print(f"  early signal: rank <= {args.early_rank} for {args.sustain} "
          f"consecutive sessions")
    print(f"  success test: reaches top {config.top_n} within {args.lookahead} "
          f"sessions")
    print("=" * 84)

    ranking_scores, base_scores, _ = compute_scores(prices, config)
    _, history = build_target_portfolios(
        ranking_scores, price_columns=list(prices.close.columns),
        top_n=config.top_n, min_data_days=config.min_data_days,
        hold_days=config.hold_days, vix_data=prices.vix, vix_config=config.vix,
        base_composite_scores=base_scores, velocity_config=config.velocity,
        correlation_config=config.correlation, close=prices.close,
    )

    history_counts = ranking_scores.notna().cumsum()
    ranks = daily_ranks(
        ranking_scores, history_counts, config.min_data_days,
        base_scores=base_scores,
        min_level=config.velocity.min_level_threshold if config.velocity else None,
    )
    ranks = ranks.loc[ranks.notna().any(axis=1)]

    # --- when was each symbol actually bought? ---
    actual_entries: dict = {}
    held_before: set = set()
    for record in history:
        date = pd.Timestamp(record["Date"])
        picks = set(record["Selected_Stocks"])
        for sym in picks - held_before:
            actual_entries.setdefault(sym, []).append(date)
        held_before = picks

    # --- every early-entry trigger, winners and losers alike ---
    top_n = config.top_n
    in_top_n = ranks <= top_n
    early = ranks <= args.early_rank

    triggers = []
    for symbol in ranks.columns:
        series = early[symbol].astype(int)
        # Consecutive-day streak of the early condition.
        streak = series.groupby((series != series.shift()).cumsum()).cumsum()
        fires = streak[streak == args.sustain].index

        top_series = in_top_n[symbol]
        for date in fires:
            # Already a top-N name — not an early entry, just a holding.
            if bool(top_series.get(date, False)):
                continue

            i = ranks.index.get_loc(date)
            window = top_series.iloc[i + 1: i + 1 + args.lookahead]
            arrived = bool(window.any())
            arrival_date = window[window].index[0] if arrived else None

            triggers.append({
                "Symbol": symbol,
                "Signal Date": date,
                "Reached Top N": arrived,
                "Sessions To Arrival": (ranks.index.get_loc(arrival_date) - i
                                        if arrived else np.nan),
                "Early Return": forward_return(prices.close, symbol, date,
                                               args.horizon),
                "Arrival Return": (forward_return(prices.close, symbol,
                                                  arrival_date, args.horizon)
                                   if arrived else np.nan),
            })

    frame = pd.DataFrame(triggers)
    if frame.empty:
        print("\nNo early-entry signals fired.")
        return 0

    precision = frame["Reached Top N"].mean()
    print(f"\n--- Signal quality ---")
    print(f"  Early signals fired               {len(frame):,}")
    print(f"  Reached top {top_n} within {args.lookahead}      {precision:.1%}"
          f"   <- precision")
    print(f"  Median sessions to arrival        "
          f"{frame['Sessions To Arrival'].median():.0f}")
    print(f"  Distinct symbols                  {frame['Symbol'].nunique()}")

    # --- the honest comparison ---
    winners = frame[frame["Reached Top N"]]
    losers = frame[~frame["Reached Top N"]]

    print(f"\n--- Return over {args.horizon} sessions from the EARLY signal ---")
    print(f"  All triggers        {frame['Early Return'].mean():+.2%}  "
          f"(n={frame['Early Return'].notna().sum():,})")
    print(f"    of which winners  {winners['Early Return'].mean():+.2%}  "
          f"(n={winners['Early Return'].notna().sum():,})")
    print(f"    of which stalled  {losers['Early Return'].mean():+.2%}  "
          f"(n={losers['Early Return'].notna().sum():,})   <- the false-positive cost")

    print(f"\n--- Return over {args.horizon} sessions from ACTUAL arrival "
          f"(winners only) ---")
    print(f"  Buying on arrival   {winners['Arrival Return'].mean():+.2%}")

    early_all = frame["Early Return"].mean()
    arrival_win = winners["Arrival Return"].mean()
    delta = early_all - arrival_win

    print(f"\n--- Verdict ---")
    print(f"  Buying EVERY early signal:      {early_all:+.2%}")
    print(f"  Buying only on actual arrival:  {arrival_win:+.2%}")
    print(f"  Difference:                     {delta:+.2%}")

    edge = frame["Early Return"].dropna()
    t_stat = edge.mean() / (edge.std() / np.sqrt(len(edge))) if len(edge) > 1 else np.nan

    if delta > 0.005:
        print(f"\n  Earlier entry wins by {delta:.2%} even after paying for the")
        print(f"  {1 - precision:.0%} of signals that never arrive. Worth building.")
    elif delta < -0.005:
        print(f"\n  Earlier entry LOSES {abs(delta):.2%} once false positives are")
        print(f"  included. The {precision:.0%} precision is not high enough to carry")
        print(f"  the {1 - precision:.0%} that stall — which is exactly the trap the")
        print(f"  hindsight check warned about. Measuring winners alone would have")
        print(f"  shown a gain that cannot be traded.")
    else:
        print(f"\n  The difference is {delta:+.2%} — a wash. Earlier entry neither")
        print(f"  helps nor hurts once false positives are counted.")

    out = REPO_ROOT / "rsi_ma_entry_timing.csv"
    frame.to_csv(out, index=False)
    print(f"\nPer-signal detail exported to: {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
