"""
Walk-forward validation of the velocity-blend candidates.

    python scripts/run_walkforward.py
    python scripts/run_walkforward.py --train-years 5 --test-months 12
    python scripts/run_walkforward.py --objective cagr

Settles the question the full-sample sweep cannot: is w5_L70V30's advantage real
and forward-usable, or an artifact of choosing the parameter with knowledge of
the whole history?

Candidate set is deliberately kept to the plausible configs rather than the full
grid.  Walk-forward over a large candidate set has its own selection problem —
with enough candidates, one of them wins any given training window by chance.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.data import load_data
from momentum.experiments import (export_comparison, print_comparison,
                                  production_config, run_experiments, variant)
from momentum.universe import current_symbols
from momentum.validation import print_walk_forward, walk_forward


def candidates():
    """
    Plausible velocity settings, plus the no-velocity control.

    Kept small on purpose.  Every extra candidate raises the chance that one of
    them tops a training window by luck, which is the same overfitting problem
    one level up.
    """
    base = production_config()
    exps = [variant(base, "level_only", "no velocity", velocity=None)]

    for window in (3, 5, 7, 10, 15):
        for lw in (0.7, 0.5):
            exps.append(
                variant(base, f"w{window}_L{int(lw * 100)}V{int((1 - lw) * 100)}", "",
                        velocity__velocity_window=window,
                        velocity__level_weight=lw,
                        velocity__velocity_weight=round(1 - lw, 4),
                        velocity__blend_normalization="cross_sectional")
            )
    return exps


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--train-years", type=float, default=3.0)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--objective", default="sharpe_ratio",
                        choices=["sharpe_ratio", "cagr", "calmar_ratio"])
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("WALK-FORWARD VALIDATION — velocity blend")
    print(f"  train {args.train_years}y / test {args.test_months}m / "
          f"objective {args.objective}")
    print(f"  Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache)

    exps = candidates()
    print(f"\nRunning {len(exps)} candidate configs over full history...")
    results = run_experiments(exps, prices)

    print_comparison(results, baseline="level_only")

    wf = walk_forward(results,
                      train_years=args.train_years,
                      test_months=args.test_months,
                      objective=args.objective)
    print_walk_forward(wf, objective=args.objective)

    tag = f"t{args.train_years:g}y_{args.test_months}m_{args.objective}"
    wf.selections.to_csv(REPO_ROOT / f"rsi_ma_walkforward_selections_{tag}.csv",
                         index=False)
    wf.fixed_oos.to_csv(REPO_ROOT / f"rsi_ma_walkforward_fixed_{tag}.csv", index=False)
    export_comparison(results, str(REPO_ROOT / "rsi_ma_walkforward_candidates.csv"))

    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
