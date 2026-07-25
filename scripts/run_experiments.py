"""
Parameter sweeps.

    python scripts/run_experiments.py execution   how much did the unachievable
                                                  same-close fill flatter us?
    python scripts/run_experiments.py velocity    level/velocity weight sweep
    python scripts/run_experiments.py vix         legacy VIX regime thresholds
    python scripts/run_experiments.py zscore      z-score method and window
    python scripts/run_experiments.py all

Every suite runs over one shared price panel so differences are attributable to
the parameter under test.  Read the deltas, not the levels: absolute figures
carry survivorship bias from applying today's screened universe backwards.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.config import VelocityConfig, VixRegimeConfig
from momentum.data import load_data
from momentum.experiments import (Experiment, export_comparison, legacy_config,
                                  print_comparison, print_subperiods,
                                  production_config, run_experiments, variant)
from momentum.universe import current_symbols


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------

def suite_execution():
    """
    What the execution assumption is worth.

    The pre-refactor backtest filled at the close that generated the signal.
    This isolates that assumption from everything else: same scores, same
    selections, only the fill convention and slippage change.

    If the strategy's edge largely evaporates between 'same_close' and
    'next_open', that is the single most important thing to know before tuning
    anything else — because every historical parameter choice was made against
    the flattering version.
    """
    base = legacy_config()   # legacy blend, so ONLY execution varies here
    return [
        variant(base, "same_close_0bps", "pre-refactor assumption",
                execution__execute_at="same_close", execution__slippage_bps=0.0),
        variant(base, "same_close_7.5bps", "same fill, realistic slippage",
                execution__execute_at="same_close", execution__slippage_bps=7.5),
        variant(base, "next_open_0bps", "realistic fill, no slippage",
                execution__execute_at="next_open", execution__slippage_bps=0.0),
        variant(base, "next_open_7.5bps", "realistic fill and slippage",
                execution__execute_at="next_open", execution__slippage_bps=7.5),
        variant(base, "next_close_7.5bps", "full day of action lag",
                execution__execute_at="next_close", execution__slippage_bps=7.5),
    ]


def suite_velocity():
    """
    Level vs velocity weighting, on a corrected scale.

    The previous sweep that selected 0.7/0.3 ran against a normalization
    mismatch — the level was a rolling z-score, the velocity was
    cross-sectional, so the nominal weights did not correspond to actual
    influence.  This re-runs the question with both components on the same
    scale, which makes 'invert the weights' a meaningful test rather than a
    confounded one.

    L0/V100 is included because it *is* Track C's "rank on the rate of change of
    the z-score instead of its level" — no new machinery needed.
    """
    base = production_config()
    experiments = [
        variant(base, "legacy_scale_L70V30", "old confounded scale (reference)",
                velocity__blend_normalization="legacy"),
        variant(base, "level_only", "no velocity component",
                velocity=None),
    ]

    for window in (5, 10, 20):
        for lw in (1.0, 0.7, 0.5, 0.3, 0.0):
            name = f"w{window}_L{int(lw*100)}V{int((1-lw)*100)}"
            experiments.append(
                variant(base, name,
                        "velocity-only ranking (Track C #2)" if lw == 0.0 else "",
                        velocity__velocity_window=window,
                        velocity__level_weight=lw,
                        velocity__velocity_weight=round(1.0 - lw, 4))
            )

    # Rank-based normalization at the current weighting, as an outlier-robustness
    # check on the whole approach.
    experiments.append(
        variant(base, "w10_L70V30_rank", "rank-normalized blend",
                velocity__blend_normalization="rank")
    )
    return experiments


def suite_robustness():
    """
    Is the velocity-window optimum a plateau or a spike?

    The main sweep hands back w5_L70V30 winning CAGR, Sharpe, drawdown and
    Calmar simultaneously, while its immediate neighbour w10_L70V30 gives up
    six points of CAGR.  A parameter surface that sharp is the signature of
    fitting to history, and the strategy's own design philosophy flags exactly
    this: weights calibrated to historical relationships decay as markets
    evolve.

    Two questions this answers:
      1. Sweeping the window finely (3/5/7/10/15) — does performance fall off a
         cliff either side of 5, or degrade gracefully?
      2. Does rank normalization, which is immune to velocity outliers, flatten
         the surface?  If the z-scored version is spiky and the rank version is
         flat at a similar level, the spike was outlier luck and rank is the
         safer choice even where it scores marginally lower.
    """
    base = production_config()
    experiments = []

    for window in (3, 5, 7, 10, 15):
        for norm in ("cross_sectional", "rank"):
            tag = "z" if norm == "cross_sectional" else "rank"
            experiments.append(
                variant(base, f"w{window}_L70V30_{tag}", "",
                        velocity__velocity_window=window,
                        velocity__level_weight=0.7,
                        velocity__velocity_weight=0.3,
                        velocity__blend_normalization=norm)
            )

    # True no-velocity control.  Note this is NOT the same as L100/V0: with
    # velocity_weight=0.0 the term 0.0 * NaN is still NaN, so a L100V0 config
    # silently drops stocks that lack velocity history.  Only velocity=None
    # isolates the level cleanly.
    experiments.append(variant(base, "level_only", "true no-velocity control",
                               velocity=None))
    return experiments


def suite_vix():
    """Legacy VIX regime thresholds. Superseded by the Track A ladder."""
    base = production_config()
    return [
        variant(base, "no_vix_filter", "baseline", vix=None),
        variant(base, "vix_z1.0_2.0", "",
                vix=VixRegimeConfig(elevated_zscore=1.0, crisis_zscore=2.0)),
        variant(base, "vix_z0.75_1.5", "reacts sooner",
                vix=VixRegimeConfig(elevated_zscore=0.75, crisis_zscore=1.5)),
        variant(base, "vix_z1.5_2.5", "current production",
                vix=VixRegimeConfig(elevated_zscore=1.5, crisis_zscore=2.5)),
        variant(base, "vix_z1.0_2.0_win30", "faster baseline",
                vix=VixRegimeConfig(elevated_zscore=1.0, crisis_zscore=2.0,
                                    zscore_window=30)),
        variant(base, "vix_z1.0_2.0_roc30", "with spike trigger",
                vix=VixRegimeConfig(elevated_zscore=1.0, crisis_zscore=2.0,
                                    use_roc_trigger=True, roc_threshold=0.30)),
    ]


def suite_zscore():
    """Z-score normalization method and window."""
    base = production_config()
    return [
        variant(base, "cross_sectional", "",
                scoring__zscore_method="cross_sectional",
                scoring__zscore_window=252, min_data_days=200),
        variant(base, "rolling_63d", "",
                scoring__zscore_method="rolling",
                scoring__zscore_window=63, min_data_days=200),
        variant(base, "rolling_126d", "current production",
                scoring__zscore_method="rolling",
                scoring__zscore_window=126, min_data_days=200),
        variant(base, "rolling_252d", "",
                scoring__zscore_method="rolling",
                scoring__zscore_window=252, min_data_days=252),
        variant(base, "rolling_504d", "",
                scoring__zscore_method="rolling",
                scoring__zscore_window=504, min_data_days=504),
    ]


SUITES = {
    "execution": (suite_execution, "rsi_ma_execution_comparison.csv",
                  "same_close_0bps"),
    "velocity":  (suite_velocity,  "rsi_ma_velocity_comparison.csv",
                  "legacy_scale_L70V30"),
    "robustness": (suite_robustness, "rsi_ma_velocity_robustness.csv",
                   "level_only"),
    "vix":       (suite_vix,       "rsi_ma_vix_regime_comparison.csv",
                  "no_vix_filter"),
    "zscore":    (suite_zscore,    "rsi_ma_zscore_comparison.csv",
                  "rolling_126d"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run parameter sweeps")
    parser.add_argument("suite", choices=list(SUITES) + ["all"])
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("MOMENTUM PARAMETER SWEEPS")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache)

    names = list(SUITES) if args.suite == "all" else [args.suite]

    for name in names:
        builder, out_file, baseline = SUITES[name]
        experiments = builder()

        print(f"\n{'-' * 78}")
        print(f"SUITE: {name}  ({len(experiments)} configs)")
        print(f"{'-' * 78}")
        print((builder.__doc__ or "").rstrip())

        results = run_experiments(experiments, prices)
        print_comparison(results, baseline=baseline)
        print_subperiods(results, n_periods=3)
        export_comparison(results, str(REPO_ROOT / out_file))

    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
