"""
Quarterly universe screening report.

    python scripts/screen_universe.py
    python scripts/screen_universe.py --window 500 --threshold 0.75
    python scripts/screen_universe.py --as-of 2026-01-15

Run this alongside the Schwab screen when revising the universe.  It answers a
question the screener cannot: of the names that pass on their own merits, which
ones are already covered by something else in the universe?

Two names correlating at 0.85 over a year occupy two slots and deliver one
bet's worth of diversification.  Swapping one for an independent exposure costs
nothing in expected return — they rank similarly by construction — and widens
the set of genuinely different portfolios the ranker can build.

This report is advisory.  It changes nothing at runtime; the medium-term filter
in `momentum/correlation.py` is the part that acts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.config import CorrelationConfig
from momentum.correlation import screen_universe
from momentum.data import load_data
from momentum.universe import current_symbols, sector_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Universe correlation screen")
    parser.add_argument("--window", type=int, default=200,
                        help="trading days of history for the estimate")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="flag pairs at or above this correlation")
    parser.add_argument("--as-of", default=None,
                        help="screen as of this date (YYYY-MM-DD)")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config = CorrelationConfig(long_term_window=args.window,
                               redundant_threshold=args.threshold)

    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache, verbose=False)
    sectors = sector_map()

    report = screen_universe(prices.close, sectors, config, as_of=args.as_of)

    as_of_label = args.as_of or f"{prices.index[-1]:%Y-%m-%d}"
    print("=" * 88)
    print(f"UNIVERSE CORRELATION SCREEN — as of {as_of_label}")
    print(f"  {args.window}-day window, flagging pairs at or above {args.threshold:.2f}")
    print("=" * 88)

    # --- how much diversification the universe actually contains ---
    eb = report["effective_bets"]
    print(f"\n--- Diversification of the universe ---")
    print(f"  Tickers                  {eb['n_assets']}")
    print(f"  Effective independent bets {eb['effective_bets']:.1f}")
    print(f"  Largest common factor      {eb['top_eigenvalue_share']:.1%} of variance")
    print(f"\n  {eb['n_assets']} tickers behaving like {eb['effective_bets']:.1f} "
          f"independent bets. The gap is what the ranker cannot see: it can pick")
    print(f"  four different tickers that are substantially the same position.")

    # --- distribution first, so the threshold is judged in context ---
    pairs = report["redundant_pairs"]
    corr_values = pairs["Correlation"]
    flagged = pairs[pairs["Flagged"]]

    print(f"\n--- Pairwise correlation distribution ---")
    print(f"  {len(pairs)} pairs   mean {corr_values.mean():.3f}   "
          f"median {corr_values.median():.3f}   max {corr_values.max():.3f}")
    for level in (0.60, 0.70, 0.80, 0.90):
        print(f"  at or above {level:.2f}: {(corr_values >= level).sum():>4} pairs")

    print(f"\n--- Most correlated pairs (top 12, regardless of threshold) ---")
    print(f"  {'A':<7}{'B':<7}{'Corr':>7}  {'Same sector':<12} Sectors")
    print("  " + "-" * 74)
    for _, row in pairs.head(12).iterrows():
        same = "yes" if row.get("Same Sector") else "no"
        print(f"  {row['Symbol A']:<7}{row['Symbol B']:<7}"
              f"{row['Correlation']:>7.3f}  {same:<12} "
              f"{row.get('Sector A', '?')} / {row.get('Sector B', '?')}")

    print(f"\n--- Flagged at your threshold ({args.threshold:.2f}) ---")
    if flagged.empty:
        print(f"  None. Nothing in the universe is a near-duplicate at "
              f"{args.threshold:.2f}.")
        print(f"  The screening process is already doing this job — treat this "
              f"report as")
        print(f"  confirmation that nothing has drifted, not as a list of "
              f"problems to fix.")
        if corr_values.max() < args.threshold:
            print(f"\n  For a threshold that would actually bind on this universe, "
                  f"try")
            print(f"  --threshold {corr_values.quantile(0.99):.2f} "
                  f"(the 99th percentile of current pairs).")
    else:
        print(f"  {len(flagged)} pair(s). Consider replacing one of each with an "
              f"independent exposure.")
        cross = flagged[~flagged.get("Same Sector",
                                     pd.Series(dtype=bool)).fillna(False)]
        if not cross.empty:
            print(f"\n  {len(cross)} are CROSS-sector — the ones sector labels "
                  f"would miss entirely.")

    # --- most redundant individual names ---
    by_symbol = report["by_symbol"]
    print(f"\n--- Most redundant tickers (highest mean correlation to the rest) ---")
    print(f"  A name can clear every pairwise threshold and still add nothing the")
    print(f"  universe does not already cover. These are the best swap candidates.\n")
    print(f"  {'Symbol':<8}{'Mean':>8}{'Max':>8}  {'Closest peer':<14}{'Pairs over':>11}")
    print("  " + "-" * 60)
    for _, row in by_symbol.head(12).iterrows():
        print(f"  {row['Symbol']:<8}{row['Mean Correlation']:>8.3f}"
              f"{row['Max Correlation']:>8.3f}  {row['Closest Peer']:<14}"
              f"{row['Above Threshold']:>11}")

    print(f"\n--- Least redundant tickers (most diversifying) ---")
    print(f"  Keep these. They are what makes the other {eb['n_assets'] - 5} usable.\n")
    print(f"  {'Symbol':<8}{'Mean':>8}{'Max':>8}  {'Closest peer':<14}")
    print("  " + "-" * 48)
    for _, row in by_symbol.tail(8).iloc[::-1].iterrows():
        print(f"  {row['Symbol']:<8}{row['Mean Correlation']:>8.3f}"
              f"{row['Max Correlation']:>8.3f}  {row['Closest Peer']:<14}")

    # --- export ---
    pairs_path = REPO_ROOT / "rsi_ma_universe_redundant_pairs.csv"
    symbol_path = REPO_ROOT / "rsi_ma_universe_redundancy.csv"
    pairs.to_csv(pairs_path, index=False)
    by_symbol.to_csv(symbol_path, index=False)
    print(f"\nExported:\n  {pairs_path.name}\n  {symbol_path.name}")

    print("\nTo act on this: edit universe.csv — set Active=N on a redundant name")
    print("and add its replacement, or just swap the Symbol. Re-run to confirm the")
    print("effective-bets count went up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
