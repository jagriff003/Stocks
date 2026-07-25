"""
Walk-forward validation.

The problem this solves: a full-sample sweep chooses parameters with knowledge
of the whole history, so its winner's margin is part real edge and part
in-sample fitting, in unknown proportion.  Subperiod consistency narrows that
gap but does not close it — the candidate set was still selected knowing how the
segments turned out.

Walk-forward closes it by only ever selecting on data that precedes the data it
scores on:

    |<-- train 3y -->|<- test 6m ->|
              |<-- train 3y -->|<- test 6m ->|
                        |<-- train 3y -->|<- test 6m ->|

Each test window is scored using the config that looked best over the *preceding*
train window only.  Stitching the test windows together gives an out-of-sample
record for the whole selection *procedure*, not for a parameter chosen in
hindsight.

Two questions it answers, which are different and both matter:

  1. Is the winning parameter stable, or does the optimum drift?  If the chosen
     config changes every window, no fixed setting is trustworthy and the
     apparent full-sample winner was noise.

  2. Does re-tuning periodically actually beat just fixing the parameter?
     Re-tuning has a cost — it chases recent noise.  Often a fixed, mediocre,
     stable setting wins.

Implementation note: candidate configs produce return series that do not depend
on the window they are evaluated over, so each config is run once over full
history and then sliced.  Walk-forward costs no extra backtests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .experiments import ExperimentResult
from .metrics import calculate_performance_metrics


@dataclass
class WalkForwardResult:
    """Outcome of one walk-forward run."""
    selections: pd.DataFrame              # per test window: chosen config + result
    oos_returns: pd.Series                # stitched out-of-sample return stream
    oos_metrics: Dict[str, float]
    fixed_oos: pd.DataFrame               # each config over the same OOS span
    stability: Dict[str, float] = field(default_factory=dict)


def walk_forward(results: List[ExperimentResult],
                 train_years: float = 3.0,
                 test_months: int = 6,
                 objective: str = "sharpe_ratio",
                 risk_free_rate: float = 0.045) -> WalkForwardResult:
    """
    Roll a train/test window across the history, selecting on train only.

    Parameters
    ----------
    results : list of ExperimentResult
        The candidate configs.  All must have been run over the same panel.
    train_years : float
        Trailing window used to pick the config for the next test window.
    test_months : int
        Length of each out-of-sample window.  Also the re-tuning cadence.
    objective : str
        Metric maximized on the training window.  'sharpe_ratio' balances
        return against risk; 'cagr' chases return alone and will pick
        higher-turnover configs.
    """
    if not results:
        raise ValueError("No candidate configs supplied")

    returns = pd.DataFrame({r.name: r.result.returns for r in results}).dropna(how="all")
    if returns.empty:
        raise ValueError("Candidate configs share no common return dates")

    names = list(returns.columns)
    start, end = returns.index[0], returns.index[-1]

    train_delta = pd.DateOffset(days=int(round(train_years * 365.25)))
    test_delta = pd.DateOffset(months=test_months)

    rows: List[Dict] = []
    oos_pieces: List[pd.Series] = []

    test_start = start + train_delta
    while test_start < end:
        test_end = min(test_start + test_delta, end)
        train_slice = returns.loc[test_start - train_delta:test_start]
        test_slice = returns.loc[test_start:test_end]

        if len(train_slice) < 120 or len(test_slice) < 20:
            test_start = test_end
            continue

        # --- select on training data only ---
        scores = {}
        for name in names:
            series = train_slice[name].dropna()
            if len(series) < 60:
                continue
            scores[name] = calculate_performance_metrics(
                series, risk_free_rate=risk_free_rate)[objective]

        if not scores:
            test_start = test_end
            continue

        chosen = max(scores, key=lambda n: (scores[n] if pd.notna(scores[n]) else -np.inf))

        # --- score it on the untouched test window ---
        chosen_test = test_slice[chosen].dropna()
        if chosen_test.empty:
            test_start = test_end
            continue

        test_metrics = calculate_performance_metrics(
            chosen_test, risk_free_rate=risk_free_rate)

        # What the best-in-hindsight choice would have been, for reference.
        hindsight_scores = {
            n: calculate_performance_metrics(
                test_slice[n].dropna(), risk_free_rate=risk_free_rate)[objective]
            for n in names if test_slice[n].dropna().shape[0] >= 20
        }
        best_hindsight = max(hindsight_scores, key=hindsight_scores.get) \
            if hindsight_scores else None

        rows.append({
            "Test Start": test_start.date(),
            "Test End": test_end.date(),
            "Chosen": chosen,
            f"Train {objective}": scores[chosen],
            f"Test {objective}": test_metrics[objective],
            "Test CAGR": test_metrics["cagr"],
            "Test MaxDD": test_metrics["max_drawdown"],
            "Best In Hindsight": best_hindsight,
            "Chose Best": chosen == best_hindsight,
        })

        oos_pieces.append(chosen_test)
        test_start = test_end

    if not rows:
        raise ValueError(
            "No usable walk-forward windows — shorten train_years or test_months"
        )

    selections = pd.DataFrame(rows)
    oos_returns = pd.concat(oos_pieces).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated()]
    oos_metrics = calculate_performance_metrics(oos_returns, risk_free_rate=risk_free_rate)

    # --- each fixed config over the same out-of-sample span, for comparison ---
    oos_start, oos_end = oos_returns.index[0], oos_returns.index[-1]
    fixed_rows = []
    for name in names:
        series = returns.loc[oos_start:oos_end, name].dropna()
        if len(series) < 60:
            continue
        m = calculate_performance_metrics(series, risk_free_rate=risk_free_rate)
        fixed_rows.append({
            "Config": name,
            "OOS CAGR": m["cagr"],
            "OOS Sharpe": m["sharpe_ratio"],
            "OOS MaxDD": m["max_drawdown"],
            "OOS Calmar": m["calmar_ratio"],
            "Windows Chosen": int((selections["Chosen"] == name).sum()),
            "Windows Best": int((selections["Best In Hindsight"] == name).sum()),
        })
    fixed_oos = pd.DataFrame(fixed_rows).sort_values("OOS Sharpe", ascending=False)

    # --- how stable was the selection? ---
    chosen_seq = selections["Chosen"].tolist()
    switches = sum(1 for a, b in zip(chosen_seq, chosen_seq[1:]) if a != b)
    stability = {
        "n_windows": len(chosen_seq),
        "n_distinct_configs": len(set(chosen_seq)),
        "switch_rate": switches / max(len(chosen_seq) - 1, 1),
        "hit_rate": float(selections["Chose Best"].mean()),
        "modal_config_share": max(chosen_seq.count(c) for c in set(chosen_seq))
                              / len(chosen_seq),
    }

    return WalkForwardResult(selections, oos_returns, oos_metrics, fixed_oos, stability)


def print_walk_forward(wf: WalkForwardResult, objective: str = "sharpe_ratio") -> None:
    """Print the walk-forward record and what it implies."""
    print("\n" + "=" * 100)
    print("WALK-FORWARD VALIDATION")
    print("=" * 100)

    print(f"\n{'Test window':<26}{'Chosen':<20}{'Test CAGR':>11}"
          f"{'Test MaxDD':>12}{'Best in hindsight':>22}")
    print("-" * 100)
    for _, row in wf.selections.iterrows():
        mark = " *" if row["Chose Best"] else "  "
        print(f"{str(row['Test Start']) + ' to ' + str(row['Test End']):<26}"
              f"{row['Chosen']:<20}"
              f"{row['Test CAGR']:>10.2%} "
              f"{row['Test MaxDD']:>11.2%} "
              f"{str(row['Best In Hindsight']):>20}{mark}")

    s = wf.stability
    print("\n--- Selection stability ---")
    print(f"  Windows                     {s['n_windows']}")
    print(f"  Distinct configs chosen     {s['n_distinct_configs']}")
    print(f"  Switch rate window-to-window{s['switch_rate']:>7.0%}")
    print(f"  Most-chosen config share    {s['modal_config_share']:>7.0%}")
    print(f"  Picked the truly-best config{s['hit_rate']:>7.0%} of the time")

    if s["switch_rate"] > 0.5:
        print("\n  READ: the optimum moves nearly every window. No fixed setting is")
        print("        reliable here, and the full-sample winner was noise.")
    elif s["modal_config_share"] > 0.6:
        print("\n  READ: one config dominates the selection. That is evidence for a")
        print("        genuinely stable optimum rather than a fitted one.")

    m = wf.oos_metrics
    print("\n--- Adaptive procedure, out of sample ---")
    print(f"  CAGR {m['cagr']:.2%}   Sharpe {m['sharpe_ratio']:.2f}   "
          f"MaxDD {m['max_drawdown']:.2%}   Calmar {m['calmar_ratio']:.2f}")

    print("\n--- Fixed configs over the same span ---")
    print("    CAUTION: this table is NOT out-of-sample for the fixed configs. It is")
    print("    their full-sample performance restricted to the walk-forward span, and")
    print("    the candidate set was chosen knowing that history. It is a fair")
    print("    benchmark for the ADAPTIVE procedure (which never saw the test windows)")
    print("    and nothing more. Do not read the top row as a validated winner.")
    print(f"{'Config':<22}{'OOS CAGR':>11}{'OOS Sharpe':>12}{'OOS MaxDD':>12}"
          f"{'Chosen':>9}{'Was best':>10}")
    print("-" * 76)
    for _, row in wf.fixed_oos.iterrows():
        print(f"{row['Config']:<22}"
              f"{row['OOS CAGR']:>10.2%} "
              f"{row['OOS Sharpe']:>11.2f} "
              f"{row['OOS MaxDD']:>11.2%} "
              f"{row['Windows Chosen']:>8} "
              f"{row['Windows Best']:>9}")

    best_fixed = wf.fixed_oos.iloc[0]
    median_fixed = wf.fixed_oos["OOS Sharpe"].median()

    print(f"\nVERDICT — on the one question this test actually settles")
    print(f"  Adaptive re-tuning every window : CAGR {m['cagr']:>7.2%}  "
          f"Sharpe {m['sharpe_ratio']:.2f}")
    print(f"  Median fixed config             : "
          f"Sharpe {median_fixed:.2f}")
    print(f"  Best fixed config ({best_fixed['Config']}) : "
          f"CAGR {best_fixed['OOS CAGR']:>7.2%}  Sharpe {best_fixed['OOS Sharpe']:.2f}")

    if m["sharpe_ratio"] < median_fixed:
        print("\n  Re-tuning LOST to the MEDIAN fixed setting out of sample. Selecting")
        print("  on a trailing window chases noise: it beat picking at random only")
        print(f"  {s['hit_rate']:.0%} of the time. Do not re-optimize on a schedule.")
    elif best_fixed["OOS Sharpe"] > m["sharpe_ratio"]:
        print("\n  Re-tuning beat the median fixed setting but lost to the best one.")
        print("  Weak evidence for re-optimization; a good fixed setting is simpler.")
    else:
        print("\n  Re-tuning BEAT every fixed setting out of sample — the optimum")
        print("  genuinely moves, and periodic re-validation earns its keep.")

    print("\n  This test does NOT establish which fixed config is best going forward.")
    print("  That choice remains an in-sample one; judge it on consistency across")
    print("  subperiods and on how it behaves in its worst window, not on its rank here.")
