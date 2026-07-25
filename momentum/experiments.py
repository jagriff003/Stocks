"""
Parameterized experiment runner.

Every experiment is a named `ModelConfig` run over one shared price panel, so
differences in the comparison table are attributable to the parameter that
changed and nothing else.

Comparison tables carry turnover next to return.  A variant that lifts CAGR by
two points while tripling the trade count has not obviously improved anything,
and that trade-off should be visible in the same row rather than requiring a
second query.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest import PortfolioResult
from .metrics import calculate_performance_metrics
from .config import (CorrelationConfig, ExecutionConfig, ExitConfig,
                     ModelConfig, ScoringConfig, VelocityConfig,
                     VixRegimeConfig)
from .data import PriceData
from .strategy import run_strategy


@dataclass
class Experiment:
    """A named configuration to run."""
    name: str
    config: ModelConfig
    note: str = ""


@dataclass
class ExperimentResult:
    name: str
    config: ModelConfig = field(repr=False)
    result: PortfolioResult = field(repr=False)
    note: str = ""

    @property
    def metrics(self) -> Dict[str, float]:
        return self.result.metrics

    @property
    def turnover(self) -> Dict[str, float]:
        return self.result.turnover


def run_experiments(experiments: List[Experiment], prices: PriceData,
                    verbose: bool = True) -> List[ExperimentResult]:
    """Run each experiment over the shared panel, reporting as it goes."""
    results = []
    for exp in experiments:
        if verbose:
            suffix = f"  ({exp.note})" if exp.note else ""
            print(f"\n  Running {exp.name}{suffix}...")

        result = run_strategy(prices, exp.config, verbose=False)
        results.append(ExperimentResult(exp.name, exp.config, result, exp.note))

        if verbose:
            m, t = result.metrics, result.turnover
            print(f"    CAGR {m['cagr']:>7.2%}  Sharpe {m['sharpe_ratio']:>5.2f}  "
                  f"MaxDD {m['max_drawdown']:>7.2%}  "
                  f"Trades/yr {t['trades_per_year']:>5.1f}  "
                  f"AvgHold {t['avg_hold_days']:>5.1f}d")

    return results


def comparison_table(results: List[ExperimentResult]) -> pd.DataFrame:
    """Tabulate results, return metrics and turnover side by side."""
    rows = []
    for r in results:
        m, t = r.metrics, r.turnover
        cfg = r.config

        vel = cfg.velocity
        vel_label = "none" if vel is None else (
            f"w{vel.velocity_window} L{vel.level_weight:.0%}/V{vel.velocity_weight:.0%}"
            f" [{vel.blend_normalization}]"
        )
        vix = cfg.vix
        vix_label = "none" if vix is None else (
            f"z>{vix.elevated_zscore}/{vix.crisis_zscore} w{vix.zscore_window}"
        )

        rows.append({
            "Experiment": r.name,
            "Execution": cfg.execution.execute_at,
            "Slippage (bps)": cfg.execution.slippage_bps,
            "Velocity": vel_label,
            "VIX": vix_label,
            "Hold Days": cfg.hold_days,
            "Top N": cfg.top_n,
            "CAGR": m["cagr"],
            "Sharpe": m["sharpe_ratio"],
            "Sortino": m["sortino_ratio"],
            "Volatility": m["volatility"],
            "Max Drawdown": m["max_drawdown"],
            "Calmar": m["calmar_ratio"],
            "Total Return": m["total_return"],
            "Trades/Year": t["trades_per_year"],
            "Changes/Year": t["changes_per_year"],
            "Avg Hold (days)": t["avg_hold_days"],
            "Median Hold (days)": t["median_hold_days"],
            "Annual Turnover": t["annual_turnover"],
            "Total Cost": r.result.total_cost,
            "Trading Days": m["num_periods"],
            "Note": r.note,
        })

    return pd.DataFrame(rows)


def print_comparison(results: List[ExperimentResult],
                     baseline: Optional[str] = None) -> None:
    """
    Print a readable comparison, with deltas against `baseline` if named.

    Deltas matter more than levels here.  The absolute figures carry
    survivorship bias from applying today's screened universe backwards; the
    difference between two variants over the same universe does not.
    """
    table = comparison_table(results)

    print("\n" + "=" * 108)
    print("EXPERIMENT COMPARISON")
    print("=" * 108)

    header = (f"{'Experiment':<24}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
              f"{'Calmar':>8}{'Trd/Yr':>8}{'Hold':>7}{'Turnover':>10}")
    print(header)
    print("-" * len(header))

    for _, row in table.iterrows():
        print(f"{row['Experiment']:<24}"
              f"{row['CAGR']:>8.2%} "
              f"{row['Sharpe']:>7.2f} "
              f"{row['Max Drawdown']:>8.2%} "
              f"{row['Calmar']:>7.2f} "
              f"{row['Trades/Year']:>7.1f} "
              f"{row['Avg Hold (days)']:>6.1f} "
              f"{row['Annual Turnover']:>9.1%}")

    print("=" * 108)

    if baseline and baseline in set(table["Experiment"]):
        base = table[table["Experiment"] == baseline].iloc[0]
        print(f"\nDeltas vs '{baseline}':")
        print(f"{'Experiment':<24}{'dCAGR':>10}{'dSharpe':>10}{'dMaxDD':>10}{'dTrd/Yr':>10}")
        print("-" * 64)
        for _, row in table.iterrows():
            if row["Experiment"] == baseline:
                continue
            print(f"{row['Experiment']:<24}"
                  f"{row['CAGR'] - base['CAGR']:>+9.2%} "
                  f"{row['Sharpe'] - base['Sharpe']:>+9.2f} "
                  f"{row['Max Drawdown'] - base['Max Drawdown']:>+9.2%} "
                  f"{row['Trades/Year'] - base['Trades/Year']:>+9.1f}")

    print("\nBest by metric:")
    for label, col, better in [("CAGR", "CAGR", max),
                               ("Sharpe", "Sharpe", max),
                               ("Max Drawdown", "Max Drawdown", max),
                               ("Calmar", "Calmar", max)]:
        idx = table[col].idxmax() if better is max else table[col].idxmin()
        row = table.loc[idx]
        fmt = f"{row[col]:.2%}" if col in ("CAGR", "Max Drawdown") else f"{row[col]:.2f}"
        print(f"  - {label:<14} {row['Experiment']} ({fmt})")


def subperiod_table(results: List[ExperimentResult],
                    n_periods: int = 3,
                    boundaries: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Per-subperiod metrics for each experiment.

    A full-sample sweep cannot distinguish a parameter that works from one that
    happened to fit.  Splitting the history and checking whether a config's
    ranking holds in each segment is the cheapest available guard: a setting
    that wins the full sample by winning one segment and losing the others is
    fitted, however good the headline looks.

    This is not a substitute for true walk-forward validation — the parameter is
    still chosen with knowledge of the whole sample — but it is enough to reject
    the obviously-fragile candidates.
    """
    rows = []
    for r in results:
        returns = r.result.returns
        if boundaries:
            edges = [returns.index[0]] + [pd.Timestamp(b) for b in boundaries] \
                    + [returns.index[-1]]
            segments = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        else:
            splits = np.array_split(np.arange(len(returns)), n_periods)
            segments = [(returns.index[s[0]], returns.index[s[-1]]) for s in splits]

        for i, (start, end) in enumerate(segments, 1):
            seg = returns.loc[start:end]
            if len(seg) < 30:
                continue
            m = calculate_performance_metrics(
                seg, risk_free_rate=r.config.execution.risk_free_rate)
            rows.append({
                "Experiment": r.name,
                "Period": f"P{i}",
                "Start": start.date(),
                "End": end.date(),
                "CAGR": m["cagr"],
                "Sharpe": m["sharpe_ratio"],
                "Max Drawdown": m["max_drawdown"],
            })

    return pd.DataFrame(rows)


def print_subperiods(results: List[ExperimentResult], n_periods: int = 3) -> None:
    """
    Print per-subperiod CAGR with a consistency verdict.

    'Consistency' here is how many segments a config beats the median config in.
    A setting that wins the full sample but only leads in one segment is
    reporting luck, not edge.
    """
    table = subperiod_table(results, n_periods=n_periods)
    if table.empty:
        return

    pivot = table.pivot(index="Experiment", columns="Period", values="CAGR")
    sharpe = table.pivot(index="Experiment", columns="Period", values="Sharpe")
    periods = list(pivot.columns)

    spans = (table.drop_duplicates("Period")
                  .set_index("Period")[["Start", "End"]])

    print("\n" + "=" * 96)
    print(f"SUBPERIOD STABILITY  ({n_periods} contiguous segments)")
    print("=" * 96)
    for p in periods:
        print(f"  {p}: {spans.loc[p, 'Start']} to {spans.loc[p, 'End']}")

    header = f"{'Experiment':<24}" + "".join(f"{p + ' CAGR':>13}" for p in periods) \
             + f"{'Wins':>7}"
    print("\n" + header)
    print("-" * len(header))

    medians = pivot.median()
    wins = (pivot > medians).sum(axis=1)

    for name in pivot.index:
        line = f"{name:<24}"
        for p in periods:
            line += f"{pivot.loc[name, p]:>12.2%} "
        line += f"{wins[name]:>5}/{len(periods)}"
        print(line)

    print("-" * len(header))
    print(f"{'median':<24}" + "".join(f"{medians[p]:>12.2%} " for p in periods))

    consistent = wins[wins == len(periods)].index.tolist()
    print(f"\nBeat the median in every segment: "
          f"{', '.join(consistent) if consistent else 'none'}")

    # Flag full-sample winners that are not segment-consistent.
    full = {r.name: r.metrics["cagr"] for r in results}
    best = max(full, key=full.get)
    if best not in consistent:
        print(f"\n  WARNING: '{best}' has the best full-sample CAGR "
              f"({full[best]:.2%}) but leads in only {wins[best]}/{len(periods)} "
              f"segments — treat it as fitted, not selected.")


def export_comparison(results: List[ExperimentResult], path: str) -> pd.DataFrame:
    """Write the comparison table to CSV and return it."""
    table = comparison_table(results)
    table.to_csv(path, index=False)
    print(f"\nComparison exported to: {path}")
    return table


# --------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------

def production_config(**overrides) -> ModelConfig:
    """
    The live model's configuration, as a starting point for variants.

    Three deliberate departures from the pre-refactor model, each documented
    where it lives.  `legacy_config()` reverts all of them.

      1. blend_normalization='cross_sectional' — the old code mixed a rolling
         z-score level with a cross-sectionally normalized velocity, so the
         nominal weights did not correspond to actual influence.

      2. execute_at='next_open' with 7.5 bps slippage — the old backtest filled
         at the close that generated the signal, which is unachievable, and
         charged nothing.

      3. velocity_window=5, was 10 — see below.

    On velocity_window=5 (changed 2026-07-25).  The old value of 10 was chosen
    by a sweep that ran against defect (1), so it was picking among confounded
    options.  Re-run on a corrected scale, window 10 is beaten by switching
    velocity off entirely, in every subperiod.  Window 5 leads on CAGR, Sharpe,
    drawdown and Calmar, and beats the median candidate in all three subperiods.

    Two caveats to keep in view, because this choice is not proven:

      - It is an in-sample selection.  Walk-forward validation established that
        RE-TUNING on a trailing window is actively harmful here (it beat the
        median fixed config only 13% of the time and lost 3pp of CAGR to it), so
        the right posture is set-and-forget — but "don't re-tune" is not the
        same as "this value is correct".
      - The surface around it is sharp: window 7 gives up ~9pp of first-period
        CAGR.  If live results drift materially from backtest, this parameter is
        the first place to look.

      The reassuring part is HOW it wins: it was the single best config in only
      1 of 23 walk-forward windows, yet has the best full-period return and the
      shallowest drawdown.  It compounds by avoiding disasters rather than by
      spiking, which is the more durable pattern.

      To revert: set velocity_window=10, or velocity=None for no blend at all.
    """
    config = ModelConfig(
        scoring=ScoringConfig(
            rsi_window=14, ma_short=50, ma_long=200, derivative_window=5,
            zscore_window=126, zscore_method="rolling", rel_strength_window=20,
        ),
        velocity=VelocityConfig(
            velocity_window=5, level_weight=0.7, velocity_weight=0.3,
            min_level_threshold=-3.0, blend_normalization="cross_sectional",
        ),
        execution=ExecutionConfig(execute_at="next_open", slippage_bps=7.5),
        vix=VixRegimeConfig(
            zscore_window=60, elevated_zscore=1.5, crisis_zscore=2.5,
            elevated_top_n=2,
        ),
        top_n=4, hold_days=14, min_data_days=200,
    )
    return _apply_overrides(config, overrides)


def legacy_config(**overrides) -> ModelConfig:
    """
    The pre-refactor model exactly: legacy blend, same-close fill, no costs.

    Every pre-refactor value is pinned explicitly rather than inherited from
    `production_config()`.  Inheriting was a latent bug: when production moved
    to velocity_window=5, this function silently started describing a model
    that never existed, and anything using it as a baseline — the execution
    suite in particular — would have compared against the wrong reference
    without failing.  A frozen reference has to be frozen.
    """
    config = ModelConfig(
        scoring=ScoringConfig(
            rsi_window=14, ma_short=50, ma_long=200, derivative_window=5,
            zscore_window=126, zscore_method="rolling", rel_strength_window=20,
        ),
        velocity=VelocityConfig(
            velocity_window=10,           # pre-refactor value
            level_weight=0.7, velocity_weight=0.3,
            min_level_threshold=-3.0,
            blend_normalization="legacy",  # pre-refactor scale mismatch
        ),
        execution=ExecutionConfig(execute_at="same_close", slippage_bps=0.0),
        vix=VixRegimeConfig(
            zscore_window=60, elevated_zscore=1.5, crisis_zscore=2.5,
            elevated_top_n=2,
        ),
        correlation=CorrelationConfig(enabled=False),   # did not exist
        exits=ExitConfig(enabled=False),                # did not exist
        graduated_vix=None,                             # did not exist
        top_n=4, hold_days=14, min_data_days=200,
        notes="frozen pre-refactor reference — do not retune",
    )
    return _apply_overrides(config, overrides)


def _apply_overrides(config: ModelConfig, overrides: Dict) -> ModelConfig:
    """
    Apply dotted overrides, e.g. `velocity__level_weight=0.3`.

    Deep-copies first so a variant can never mutate the config another
    experiment is holding — a shared-mutable-default bug here would silently
    corrupt a whole comparison run.
    """
    config = copy.deepcopy(config)
    for key, value in overrides.items():
        if "__" in key:
            section, attr = key.split("__", 1)
            target = getattr(config, section)
            if target is None:
                raise ValueError(f"Cannot override {key}: {section} is None")
            setattr(target, attr, value)
        else:
            setattr(config, key, value)
    return config


def variant(base: ModelConfig, name: str, note: str = "", **overrides) -> Experiment:
    """Build a named `Experiment` as a modified copy of `base`."""
    return Experiment(name=name, config=_apply_overrides(base, overrides), note=note)
