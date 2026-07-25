"""
Refactor parity check.

Runs the pre-refactor code path (functions still living in
`stock_alloc_rsi_ma_legacy.py`) and the new `momentum` package over the *same*
price panel, and asserts they produce the same numbers.

To make the comparison meaningful the new package is deliberately put back into
legacy mode for this test:

  blend_normalization = 'legacy'   the old level/velocity scale mismatch
  execute_at          = 'same_close'  the old unachievable fill assumption
  slippage_bps        = 0             the old zero-cost assumption

Those three are exactly the things the refactor changes on purpose.  Everything
else must match to floating-point tolerance.  A failure here means the refactor
altered behaviour by accident, which is the only kind of change we don't want.

Run:  python tests/test_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.config import (ExecutionConfig, ModelConfig, ScoringConfig,
                             VelocityConfig, VixRegimeConfig)
from momentum.data import load_data
from momentum.backtest import build_target_portfolios, simulate_portfolio
from momentum.signals import apply_velocity_blend, calculate_composite_scores
from momentum.universe import current_symbols

import stock_alloc_rsi_ma_legacy as legacy


TOL = 1e-9

# The production configuration as of the refactor — the thing being reproduced.
PROD = dict(
    rsi_window=14, ma_short=50, ma_long=200, derivative_window=5,
    zscore_window=126, zscore_method="rolling", rel_strength_window=20,
    velocity_window=10, level_weight=0.7, velocity_weight=0.3,
    min_level_threshold=-3.0,
    vix_zscore_window=60, elevated_zscore=1.5, crisis_zscore=2.5, elevated_top_n=2,
    top_n=4, min_data_days=200, hold_days=14,
)


class Report:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, detail=""):
        self.checks.append((name, passed, detail))
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
        return passed

    @property
    def ok(self) -> bool:
        return all(p for _, p, _ in self.checks)


def compare_frames(a: pd.DataFrame, b: pd.DataFrame, name: str, rep: Report):
    a = a.astype(float)
    b = b.astype(float)

    if not a.index.equals(b.index):
        return rep.check(name, False, f"index mismatch: {len(a)} vs {len(b)} rows")
    if list(a.columns) != list(b.columns):
        return rep.check(name, False, "column mismatch")

    both_nan = a.isna() & b.isna()
    diff = (a - b).abs()
    diff = diff.where(~both_nan)

    mismatched_nan = (a.isna() != b.isna()).sum().sum()
    max_diff = np.nanmax(diff.values) if diff.notna().any().any() else 0.0

    passed = mismatched_nan == 0 and (np.isnan(max_diff) or max_diff < TOL)
    return rep.check(name, passed,
                     f"max abs diff {max_diff:.3e}, NaN-pattern mismatches {mismatched_nan}")


def main() -> int:
    print("=" * 78)
    print("REFACTOR PARITY CHECK")
    print("=" * 78)

    symbols = current_symbols()
    print(f"\nLoading shared price panel ({len(symbols)} symbols)...")
    prices = load_data(symbols, start_date="2010-01-01", verbose=True)

    # The legacy code receives the exact same panel, including the
    # single-column-DataFrame SPY shape it was originally written against.
    legacy_spy = prices.spy.to_frame(name="SPY")

    rep = Report()

    # ---------------------------------------------------------------
    print("\n--- 1. Composite scores ---")
    legacy_composite, _, _, _, _ = legacy.calculate_composite_scores(
        prices.close, legacy_spy,
        rsi_window=PROD["rsi_window"], ma_short=PROD["ma_short"],
        ma_long=PROD["ma_long"], derivative_window=PROD["derivative_window"],
        zscore_window=PROD["zscore_window"], zscore_method=PROD["zscore_method"],
        rel_strength_window=PROD["rel_strength_window"], save_underlying=False,
    )

    scoring = ScoringConfig(
        rsi_window=PROD["rsi_window"], ma_short=PROD["ma_short"],
        ma_long=PROD["ma_long"], derivative_window=PROD["derivative_window"],
        zscore_window=PROD["zscore_window"], zscore_method=PROD["zscore_method"],
        rel_strength_window=PROD["rel_strength_window"],
    )
    new_detail = calculate_composite_scores(prices.close, prices.spy, scoring)
    new_composite = new_detail["composite"]

    compare_frames(legacy_composite, new_composite, "composite scores", rep)

    # ---------------------------------------------------------------
    print("\n--- 2. Velocity blend (legacy normalization) ---")
    legacy_vel_cfg = legacy.VelocityConfig(
        velocity_window=PROD["velocity_window"],
        level_weight=PROD["level_weight"],
        velocity_weight=PROD["velocity_weight"],
        min_level_threshold=PROD["min_level_threshold"],
    )
    legacy_blended = legacy.apply_velocity_blend(legacy_composite, legacy_vel_cfg)

    new_vel_cfg = VelocityConfig(
        velocity_window=PROD["velocity_window"],
        level_weight=PROD["level_weight"],
        velocity_weight=PROD["velocity_weight"],
        min_level_threshold=PROD["min_level_threshold"],
        blend_normalization="legacy",
    )
    new_blended = apply_velocity_blend(new_composite, new_vel_cfg)

    compare_frames(legacy_blended, new_blended, "blended scores", rep)

    # ---------------------------------------------------------------
    print("\n--- 3. Rebalance history ---")
    legacy_vix_cfg = legacy.VixRegimeConfig(
        zscore_window=PROD["vix_zscore_window"],
        elevated_zscore=PROD["elevated_zscore"],
        crisis_zscore=PROD["crisis_zscore"],
        elevated_top_n=PROD["elevated_top_n"],
    )

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        legacy_perf, legacy_rebal = legacy.select_top_stocks_biweekly(
            legacy_blended, prices.close,
            top_n=PROD["top_n"], min_data_days=PROD["min_data_days"],
            hold_days=PROD["hold_days"],
            vix_data=prices.vix, vix_config=legacy_vix_cfg,
            base_composite_scores=legacy_composite, velocity_config=legacy_vel_cfg,
        )

    new_vix_cfg = VixRegimeConfig(
        zscore_window=PROD["vix_zscore_window"],
        elevated_zscore=PROD["elevated_zscore"],
        crisis_zscore=PROD["crisis_zscore"],
        elevated_top_n=PROD["elevated_top_n"],
    )
    targets, new_rebal = build_target_portfolios(
        new_blended, price_columns=list(prices.close.columns),
        top_n=PROD["top_n"], min_data_days=PROD["min_data_days"],
        hold_days=PROD["hold_days"],
        vix_data=prices.vix, vix_config=new_vix_cfg,
        base_composite_scores=new_composite, velocity_config=new_vel_cfg,
    )

    rep.check("rebalance count", len(legacy_rebal) == len(new_rebal),
              f"{len(legacy_rebal)} legacy vs {len(new_rebal)} new")

    if len(legacy_rebal) == len(new_rebal):
        date_match = all(a["Date"] == b["Date"] for a, b in zip(legacy_rebal, new_rebal))
        pick_match = all(a["Selected_Stocks"] == b["Selected_Stocks"]
                         for a, b in zip(legacy_rebal, new_rebal))
        regime_match = all(a["Regime"] == b["Regime"]
                           for a, b in zip(legacy_rebal, new_rebal))
        rep.check("rebalance dates", date_match)
        rep.check("rebalance selections", pick_match)
        rep.check("rebalance regimes", regime_match)

        if not pick_match:
            for a, b in zip(legacy_rebal, new_rebal):
                if a["Selected_Stocks"] != b["Selected_Stocks"]:
                    print(f"      first divergence {a['Date']:%Y-%m-%d}: "
                          f"legacy={a['Selected_Stocks']} new={b['Selected_Stocks']}")
                    break

    # ---------------------------------------------------------------
    print("\n--- 4. Daily returns (same_close, zero cost) ---")
    new_result = simulate_portfolio(
        targets, prices.close, prices.open_,
        execution=ExecutionConfig(execute_at="same_close", slippage_bps=0.0),
    )

    legacy_returns = (legacy_perf.set_index("Date")["Portfolio_Return"]
                      .astype(float).sort_index())
    new_returns = new_result.returns.astype(float).sort_index()

    rep.check("return series length",
              len(legacy_returns) == len(new_returns),
              f"{len(legacy_returns)} legacy vs {len(new_returns)} new")

    common = legacy_returns.index.intersection(new_returns.index)
    if len(common):
        diff = (legacy_returns.loc[common] - new_returns.loc[common]).abs()
        rep.check("daily returns", diff.max() < TOL,
                  f"max abs diff {diff.max():.3e} over {len(common):,} days")

    # ---------------------------------------------------------------
    print("\n--- 5. Headline metrics ---")
    legacy_metrics = legacy.calculate_performance_metrics(legacy_returns, is_daily=True)
    print(f"    legacy: CAGR {legacy_metrics['CAGR']}, "
          f"Sharpe {legacy_metrics['Sharpe Ratio']}, "
          f"MaxDD {legacy_metrics['Max Drawdown']}")
    m = new_result.metrics
    print(f"    new:    CAGR {m['cagr']:.2%}, "
          f"Sharpe {m['sharpe_ratio']:.2f}, "
          f"MaxDD {m['max_drawdown']:.2%}")

    legacy_cagr = float(legacy_metrics["CAGR"].rstrip("%")) / 100
    rep.check("CAGR agreement (2dp)", abs(legacy_cagr - m["cagr"]) < 5e-5,
              f"{legacy_cagr:.4%} vs {m['cagr']:.4%}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    n_pass = sum(1 for _, p, _ in rep.checks if p)
    print(f"{n_pass}/{len(rep.checks)} checks passed")
    print("=" * 78)

    if rep.ok:
        print("\nPARITY CONFIRMED — the refactor reproduces the existing model.")
    else:
        print("\nPARITY FAILED — investigate before building on the refactor.")

    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
