"""
Did the incoming name actually beat the one it replaced?

    python scripts/analyze_swap_quality.py
    python scripts/analyze_swap_quality.py --gap 0.5

A swap rule can only add value if the signal arrives before the opportunity is
gone.  The CAGR column cannot tell you whether it does — a rule can break even
overall while being systematically late, with the lateness hidden by the cost
drag.  This measures the two legs of every swap directly: what the challenger
went on to earn, and what the incumbent it displaced went on to earn.

If challengers underperform the names they replaced, the rule is buying after
the move, which is the same lateness documented on the entry side: 60% of new
entrants had already spent 10+ of the prior 40 sessions in the top 8 before
being bought.  That would mean no gap threshold can rescue the rule, because the
problem is when the signal fires, not how big it has to be.
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
from momentum.exits import swap_quality_analysis
from momentum.experiments import production_config, _apply_overrides
from momentum.strategy import compute_scores
from momentum.universe import current_symbols


def main() -> int:
    parser = argparse.ArgumentParser(description="Swap quality analysis")
    parser.add_argument("--gap", type=float, default=1.0)
    parser.add_argument("--consecutive", type=int, default=2)
    parser.add_argument("--max-per-month", type=int, default=2)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config = _apply_overrides(production_config(), {
        "exits__enabled": True,
        "exits__mode": "score_gap",
        "exits__min_score_gap": args.gap,
        "exits__consecutive_days": args.consecutive,
        "exits__max_exits_per_month": args.max_per_month,
    })

    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache, verbose=False)

    print("=" * 80)
    print("SWAP QUALITY — does the challenger beat the incumbent?")
    print(f"  gap {args.gap:g}z, {args.consecutive}d confirmation, "
          f"max {args.max_per_month}/month")
    print("=" * 80)

    ranking_scores, base_scores, _ = compute_scores(prices, config)
    _, history = build_target_portfolios(
        ranking_scores, price_columns=list(prices.close.columns),
        top_n=config.top_n, min_data_days=config.min_data_days,
        hold_days=config.hold_days, vix_data=prices.vix, vix_config=config.vix,
        base_composite_scores=base_scores, velocity_config=config.velocity,
        correlation_config=config.correlation, exit_config=config.exits,
        close=prices.close,
    )

    frame = swap_quality_analysis(history, prices.close, horizons=(5, 10, 21))
    if frame.empty:
        print("\nNo swaps triggered at this configuration.")
        return 0

    n_swaps = frame["Date"].nunique()
    print(f"\n  {len(frame) // 3:,} swaps over the backtest "
          f"({n_swaps:,} distinct dates)\n")

    print(f"{'Horizon':<10}{'Incoming':>11}{'Outgoing':>11}{'Edge':>11}"
          f"{'Win rate':>11}{'t-stat':>9}")
    print("-" * 63)

    for horizon in (5, 10, 21):
        sub = frame[frame["Horizon"] == horizon]
        if sub.empty:
            continue
        edge = sub["Edge"]
        t_stat = (edge.mean() / (edge.std() / np.sqrt(len(edge)))
                  if edge.std() > 0 else np.nan)
        print(f"{horizon:>3}d      "
              f"{sub['In Return'].mean():>10.2%} "
              f"{sub['Out Return'].mean():>10.2%} "
              f"{edge.mean():>10.2%} "
              f"{(edge > 0).mean():>10.1%} "
              f"{t_stat:>8.2f}")

    print("\n--- Read ---")
    edge21 = frame[frame["Horizon"] == 21]["Edge"]
    mean_edge = edge21.mean()
    t_stat = (edge21.mean() / (edge21.std() / np.sqrt(len(edge21)))
              if edge21.std() > 0 else np.nan)

    if pd.notna(t_stat) and abs(t_stat) < 2:
        print(f"  The 21-day edge is {mean_edge:+.2%} with t = {t_stat:.2f} —")
        print(f"  statistically indistinguishable from zero. The challenger is")
        print(f"  no better than the name it displaced, so the swap is paying")
        print(f"  ~15 bps round trip to exchange one position for an equivalent")
        print(f"  one. The signal arrives too late to carry information.")
    elif mean_edge > 0:
        print(f"  Challengers beat incumbents by {mean_edge:+.2%} over 21 days")
        print(f"  (t = {t_stat:.2f}). The signal does arrive in time; the rule")
        print(f"  is worth running if the edge exceeds transaction cost.")
    else:
        print(f"  Challengers UNDERPERFORM the names they replaced by")
        print(f"  {abs(mean_edge):.2%} over 21 days (t = {t_stat:.2f}). The rule")
        print(f"  is systematically buying after the move — the same lateness")
        print(f"  documented on the entry side. No gap threshold fixes this,")
        print(f"  because the problem is WHEN the signal fires, not how big it is.")

    out = REPO_ROOT / "rsi_ma_swap_quality.csv"
    frame.to_csv(out, index=False)
    print(f"\nPer-swap detail exported to: {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
