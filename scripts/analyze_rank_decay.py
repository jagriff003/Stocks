"""
Track B evidence — what does holding a rank-decayed position actually cost?

    python scripts/analyze_rank_decay.py
    python scripts/analyze_rank_decay.py --exit-rank 5 --consecutive 3

Prior analysis established the FREQUENCY of rank dropout (68% of holdings fall
out of the top 4 before their hold ends, median day 3, only 17% recover) but not
its RETURN impact, because it had rank data without prices.  Frequency alone
cannot justify an exit rule: a pattern can be perfectly real and still contain
no money.

This attaches prices and answers the question that decides it — from the moment
the exit would have triggered, what did the position go on to earn?  If that is
reliably negative, exiting is worth its transaction cost.  If it is noise around
zero, the 68% was describing drift, not decay.

The result is committed and re-runnable, so this evidence does not have to be
reconstructed from memory next time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.data import load_data
from momentum.exits import daily_ranks, rank_decay_analysis
from momentum.experiments import production_config
from momentum.strategy import compute_scores
from momentum.backtest import build_target_portfolios
from momentum.universe import current_symbols


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank-decay analysis")
    parser.add_argument("--exit-rank", type=int, default=6,
                        help="rank a holding must reach to trigger an exit")
    parser.add_argument("--consecutive", type=int, default=2,
                        help="consecutive days at that rank before triggering")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config = production_config()
    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache, verbose=False)

    print("=" * 84)
    print("RANK DECAY ANALYSIS")
    print(f"  exit trigger: rank >= {args.exit_rank} for "
          f"{args.consecutive} consecutive sessions")
    print("=" * 84)

    ranking_scores, base_scores, _ = compute_scores(prices, config)

    targets, rebalance_history = build_target_portfolios(
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

    result = rank_decay_analysis(
        rebalance_history, ranks, prices.close,
        top_n=config.top_n, hold_days=config.hold_days,
        exit_rank_threshold=args.exit_rank,
        consecutive_days=args.consecutive,
    )

    frame, summary = result["instances"], result["summary"]

    print(f"\n--- Frequency (does the pattern exist?) ---")
    print(f"  Holding instances analysed        {summary['n_instances']:,}")
    print(f"  Dropped out of top {config.top_n} during hold  "
          f"{summary['pct_dropped_out']:.1%}")
    print(f"  Median day of first dropout       "
          f"{summary['median_dropout_day']:.0f}")
    print(f"  Instances hitting the exit trigger {summary['n_triggered']:,} "
          f"({summary['pct_triggered']:.1%})")
    print(f"  Of those, rank recovered          {summary['pct_recovered']:.1%}")

    print(f"\n--- Return impact (is there money in it?) ---")
    print(f"  Mean return after trigger         "
          f"{summary['mean_return_after_trigger']:+.2%}")
    print(f"  Median return after trigger       "
          f"{summary['median_return_after_trigger']:+.2%}")
    print(f"  Share of triggers that lost money {summary['pct_negative_after_trigger']:.1%}")
    print(f"  Mean days held after trigger      "
          f"{summary['mean_days_after_trigger']:.1f}")

    mean_after = summary["mean_return_after_trigger"]
    print(f"\n--- Read ---")
    if pd.isna(mean_after):
        print("  Not enough triggered instances to judge.")
    elif mean_after < -0.005:
        print(f"  Positions that trigger go on to LOSE {abs(mean_after):.2%} on")
        print(f"  average over the remaining {summary['mean_days_after_trigger']:.0f} days.")
        print(f"  Exiting is worth paying for if the cost is below that.")
    elif mean_after > 0.005:
        print(f"  Positions that trigger go on to GAIN {mean_after:.2%} on average.")
        print(f"  The dropout pattern is real but exiting on it would destroy value —")
        print(f"  the model is de-ranking names that keep working.")
    else:
        print(f"  Return after trigger is {mean_after:+.2%} — indistinguishable from")
        print(f"  zero. The 68% dropout figure describes drift, not decay: there is")
        print(f"  no edge to harvest here, and any exit rule would pay slippage for")
        print(f"  a coin flip.")

    # A rank exit only helps if the replacement does better than what it
    # replaced. Frame the bar explicitly.
    if pd.notna(mean_after):
        print(f"\n  Bar for an exit rule to add value: the replacement must beat")
        print(f"  {mean_after:+.2%} over ~{summary['mean_days_after_trigger']:.0f} days,")
        print(f"  by more than the ~15 bps round-trip cost of the swap.")

    out = REPO_ROOT / "rsi_ma_rank_decay_instances.csv"
    frame.to_csv(out, index=False)
    print(f"\nPer-instance detail exported to: {out.name}")

    # Distribution of outcomes, so a skewed mean does not mislead.
    after = frame.loc[frame["Triggered"], "Return After Trigger"].dropna()
    if len(after) > 20:
        print(f"\n--- Distribution of return after trigger ({len(after):,} instances) ---")
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            print(f"  p{int(q * 100):<3} {after.quantile(q):+.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
