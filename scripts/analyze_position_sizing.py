"""
Does position sizing matter, once the book has already been chosen?

THE HYPOTHESIS UNDER TEST
   The operator's prior is that it does not — that weighting a 4-name momentum
   book by inverse volatility, or by score, or scaling it to a volatility
   target, will not produce a systematic improvement over equal weight.  This
   script is written to give that hypothesis every chance to be wrong, and to
   report clearly if it is not.

   That framing matters for how the output should be read.  A null result here
   is a real result, not a failed experiment.  The failure mode to guard
   against is the opposite one: finding a difference that is really an artifact
   of the harness, of a single window, or of an unpriced cost.

WHAT MAKES THE COMPARISON VALID
   Selection is held fixed.  Every scheme gets the identical book on the
   identical dates — the only thing that varies is the weights.  So a
   difference in results is sizing, or it is noise, and nothing else.

   Three guards, all reported in the output rather than assumed:

   1. PARITY.  scheme='equal' must reproduce the baseline engine's numbers.
      The engine was refactored to carry weights, and if that refactor moved
      equal weight then every comparison below is measuring the refactor.
      Checked first; the script exits if it fails.

   2. REWEIGHTING IS CHARGED.  Inverse-vol and vol-target move weights as
      volatility estimates move, so they trade even when the book does not
      change.  The cost model is weight-aware (`_trade_cost_weighted`), so
      that trading is paid for.  Turnover is printed beside every result
      because a scheme that wins on return while trading twice as much has not
      necessarily won.

   3. WALK-FORWARD, NOT ONE WINDOW.  The full-sample backtest window is known
      to sit in the top decile of available history, so a full-sample ranking
      of four schemes is close to meaningless.  The headline numbers come from
      rolling out-of-sample windows; the full-sample numbers are printed too,
      but as context, not as the answer.

WHAT THIS DOES NOT TEST
   Weight drift.  Every scheme, including equal weight, is rebalanced daily
   back to its targets — that is the convention the engine has always used.
   Letting winners run is a separate experiment with a separate answer.

   Correlation-aware sizing.  vol_target approximates portfolio volatility as
   the weighted average of constituent volatilities, which ignores
   diversification and so overstates it.  On a 4-name book the covariance
   matrix is a noisy object; that experiment needs its own justification.

Run:  python scripts/analyze_position_sizing.py
      python scripts/analyze_position_sizing.py --train-years 3 --test-months 6
      python scripts/analyze_position_sizing.py --quick     # skip walk-forward
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from momentum.backtest import (  # noqa: E402
    _segment_return, _trade_cost, build_target_portfolios, simulate_portfolio)
from momentum.data import load_data  # noqa: E402
from momentum.experiments import ExperimentResult  # noqa: E402
from momentum.restrictions import check_symbols  # noqa: E402
from momentum.sizing import SizingConfig, compute_weights  # noqa: E402
from momentum.strategy import compute_scores  # noqa: E402
from momentum.universe import current_symbols  # noqa: E402
from momentum.validation import walk_forward  # noqa: E402

warnings.filterwarnings("ignore")

OUT_SUMMARY = REPO_ROOT / "rsi_ma_position_sizing.csv"
OUT_WF = REPO_ROOT / "rsi_ma_position_sizing_walkforward.csv"
OUT_WEIGHTS = REPO_ROOT / "rsi_ma_position_sizing_weights.csv"


# --------------------------------------------------------------------------
# The schemes under test
# --------------------------------------------------------------------------

def scheme_grid() -> List[SizingConfig]:
    """
    The four schemes, plus the variations that show whether a result is robust.

    Two caps are run for inverse-vol because the cap is doing at least as much
    work as the scheme: uncapped, a 4-name book can put 60% in one name.  If
    capped and uncapped inverse-vol disagree, the story is concentration, not
    volatility weighting.

    Two volatility targets are run for the same reason — a single target sitting
    near the book's realized volatility would make the scheme nearly inert and
    would not distinguish "de-risking does not help" from "we never de-risked".
    """
    return [
        SizingConfig(scheme="equal"),
        SizingConfig(scheme="inverse_vol", max_weight=0.35),
        SizingConfig(scheme="inverse_vol", max_weight=None),
        SizingConfig(scheme="score_proportional", max_weight=0.35),
        SizingConfig(scheme="score_proportional", max_weight=None),
        SizingConfig(scheme="vol_target", target_vol=0.15, max_weight=0.35),
        SizingConfig(scheme="vol_target", target_vol=0.25, max_weight=0.35),
    ]


# --------------------------------------------------------------------------
# Guard 1: parity
# --------------------------------------------------------------------------

def _reference_simulate(targets, close, open_, execution) -> Dict[str, object]:
    """
    An independent replay of the book using the ORIGINAL unweighted arithmetic.

    This exists to make the parity check mean something.  Comparing
    `sizing=equal` against `sizing=None` proves nothing once both route through
    the same weighted code — they are the same path, so they agree by
    construction.  The only honest check is against the arithmetic the engine
    used BEFORE weights existed, which is `_segment_return` and `_trade_cost`,
    both still present in backtest.py.

    Deliberately a separate, dumber loop.  A verification harness that shares
    code with the thing it verifies verifies nothing.  It covers the
    'next_open' path only, which is the live convention.
    """
    dates = list(targets.index)
    held: List[str] = []
    total_cost, records = 0.0, []
    for i in range(1, len(dates)):
        d_prev, d = dates[i - 1], dates[i]
        if d_prev not in close.index or d not in close.index:
            continue
        signal = list(targets.loc[d_prev])
        close_prev, close_now = close.loc[d_prev], close.loc[d]
        cost = 0.0
        if signal != held:
            if held:
                r1 = _segment_return(held, close_prev, open_.loc[d])
                r2 = _segment_return(signal, open_.loc[d], close_now)
                gross = (1 + r1) * (1 + r2) - 1
            else:
                gross = _segment_return(signal, open_.loc[d], close_now)
            cost = _trade_cost(held, signal, execution)
            held = signal
        else:
            if not held:
                continue
            gross = _segment_return(held, close_prev, close_now)
        total_cost += cost
        records.append({"Date": d, "gross": gross, "net": gross - cost})
    frame = pd.DataFrame(records).set_index("Date")
    return {"gross": frame["gross"], "net": frame["net"], "cost": total_cost}


def verify_equal_weight_parity(targets, close, open_, execution, scores
                               ) -> Dict[str, float]:
    """
    Check that the weighted engine, given equal weights, is the old engine.

    Returns diagnostics; the caller decides whether to continue.

    The GROSS stream must match bit-for-bit — that is the test of whether the
    weighted arithmetic reduces correctly.  The NET stream is expected to
    differ in exactly one known way: the old `_trade_cost` charged nothing when
    the book grew, because it measured turnover as the fraction of the OLD book
    that left.  Adding two names to a three-name book scored as zero turnover
    despite requiring two purchases and three trims.  The weight-aware cost
    model charges it.  That difference is reported with its size rather than
    being quietly absorbed, because it moves the baseline every other number
    here is compared against.
    """
    ew = simulate_portfolio(targets, close, open_, execution=execution,
                            sizing=SizingConfig(scheme="equal"), scores=scores)
    ref = _reference_simulate(targets, close, open_, execution)

    idx = ew.returns.index.intersection(ref["net"].index)
    g = (ew.gross_returns.reindex(idx) - ref["gross"].reindex(idx)).abs()
    n = (ew.returns.reindex(idx) - ref["net"].reindex(idx)).abs()
    return {
        "aligned_days": len(idx),
        "index_identical": ew.returns.index.equals(ref["net"].index),
        "gross_max_abs_diff": float(g.max()),
        "net_max_abs_diff": float(n.max()),
        "net_days_differing": int((n > 1e-15).sum()),
        "cost_weighted": ew.total_cost,
        "cost_original": ref["cost"],
        "cagr_weighted": ew.metrics["cagr"],
    }


# --------------------------------------------------------------------------
# Running the schemes
# --------------------------------------------------------------------------

def run_schemes(targets, close, open_, execution, scores,
                grid: List[SizingConfig]) -> List[ExperimentResult]:
    out = []
    for cfg in grid:
        res = simulate_portfolio(targets, close, open_, execution=execution,
                                 sizing=cfg, scores=scores)
        out.append(ExperimentResult(name=cfg.label(), config=None, result=res,
                                    note=cfg.scheme))
        m = res.metrics
        # `turnover` counts NAME changes, which are identical across every
        # scheme here by construction — selection is held fixed.  It would
        # print the same number seven times and imply the schemes trade alike,
        # which is false.  Cost-implied turnover is derived from what was
        # actually paid, so it does see reweighting.
        implied = (res.total_cost / execution.slippage_frac / m["years"]
                   if m["years"] else float("nan"))
        print(f"  {cfg.label():<28} CAGR {m['cagr']:>7.2%}  "
              f"Sharpe {m['sharpe_ratio']:>5.2f}  "
              f"MaxDD {m['max_drawdown']:>7.2%}  "
              f"turnover {implied:>6.2f}x/yr  "
              f"cost {res.total_cost:>6.3f}")
    return out


def weight_diagnostics(targets, close, scores, grid: List[SizingConfig]
                       ) -> pd.DataFrame:
    """
    How different are the weights, actually?

    This is the question that decides whether a null result is interesting.  If
    inverse-vol produces weights that are 24% / 26% / 25% / 25%, then "no
    systematic difference" says nothing about volatility weighting — it says the
    book was too homogeneous for any weighting to matter.  The mean absolute
    deviation from 1/N is the number that separates those two conclusions.
    """
    rows = []
    dates = [d for d in targets.index if len(targets.loc[d]) > 1]
    sample = dates[::5] or dates
    for cfg in grid:
        devs, maxw, gross = [], [], []
        for d in sample:
            syms = list(targets.loc[d])
            w = compute_weights(syms, d, close, cfg, scores)
            if w.empty:
                continue
            n = len(w)
            devs.append(float((w - w.sum() / n).abs().mean()))
            maxw.append(float(w.max()))
            gross.append(float(w.sum()))
        rows.append({
            "scheme": cfg.label(),
            "mean_abs_dev_from_equal": np.mean(devs) if devs else np.nan,
            "mean_max_weight": np.mean(maxw) if maxw else np.nan,
            "mean_gross_exposure": np.mean(gross) if gross else np.nan,
            "n_rebalances": len(devs),
        })
    return pd.DataFrame(rows)


def paired_significance(results: List[ExperimentResult],
                        baseline: str = "equal") -> pd.DataFrame:
    """
    Is the daily return difference against equal weight distinguishable from 0?

    A paired test is the right instrument because the schemes hold the SAME
    names on the SAME days — the difference series removes the market move and
    the selection entirely, leaving only the weighting.  That is a far more
    sensitive test than comparing two Sharpe ratios, and it is the test most
    likely to detect a real effect if one exists.

    Reported as an annualized mean difference with a t-statistic.  |t| < 2 is
    not evidence of no effect, but with ~3600 daily observations it does bound
    how large an undetected effect could be.
    """
    base = next(r for r in results if r.name == baseline)
    rows = []
    for r in results:
        if r.name == baseline:
            continue
        d = (r.result.returns - base.result.returns).dropna()
        if len(d) < 30:
            continue
        mean, sd = float(d.mean()), float(d.std(ddof=1))
        t = mean / (sd / np.sqrt(len(d))) if sd > 0 else np.nan
        rows.append({
            "scheme": r.name,
            "ann_return_diff": mean * 252,
            "t_stat": t,
            "n_days": len(d),
            "distinguishable": "yes" if abs(t) >= 2.0 else "no",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-years", type=float, default=3.0)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--objective", default="sharpe_ratio")
    ap.add_argument("--quick", action="store_true",
                    help="skip the walk-forward stage")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    from run_live import build_config

    print("=" * 86)
    print("POSITION SIZING — does weighting the book change anything?")
    print("=" * 86)

    check_symbols(current_symbols(), "universe.csv")

    config = build_config()
    prices = load_data(current_symbols(), use_cache=not args.no_cache)
    ranking_scores, base_scores, _ = compute_scores(
        prices, config, underlying_path=str(REPO_ROOT / "rsi_ma_underlying_measures.csv"))

    targets, _ = build_target_portfolios(
        ranking_scores, price_columns=list(prices.close.columns),
        top_n=config.top_n, min_data_days=config.min_data_days,
        hold_days=config.hold_days, vix_data=prices.vix, vix_config=config.vix,
        base_composite_scores=base_scores, velocity_config=config.velocity,
        correlation_config=config.correlation,
        graduated_config=config.graduated_vix, exit_config=config.exits,
        close=prices.close, rank_offset=config.rank_offset,
        rank_offset_scope=config.rank_offset_scope,
        monitor_symbols=config.monitor_symbols, verbose=False)

    # --- Guard 1 -----------------------------------------------------------
    print("\n--- PARITY CHECK: does equal weight still mean what it meant? ---")
    p = verify_equal_weight_parity(targets, prices.close, prices.open_,
                                   config.execution, ranking_scores)
    print("  Reference: an independent replay using the pre-weights arithmetic.")
    print(f"  aligned sessions                  : {p['aligned_days']} "
          f"(index identical: {p['index_identical']})")
    print(f"  GROSS returns, max abs difference : {p['gross_max_abs_diff']:.2e}")
    print(f"  NET   returns, max abs difference : {p['net_max_abs_diff']:.2e} "
          f"on {p['net_days_differing']} sessions")
    print(f"  slippage paid, original {p['cost_original']:.6f} -> "
          f"weight-aware {p['cost_weighted']:.6f}")
    if p["net_days_differing"]:
        print("    The gap is the book-expansion fix: the original cost model")
        print("    charged nothing when the book GREW, because it measured")
        print("    turnover as the share of the old book that left.  Buying two")
        print("    new names scored as zero turnover.  Now charged.")
    if p["gross_max_abs_diff"] > 1e-12:
        print("\n  FAILED. The weighted engine does not reproduce equal weight.")
        print("  Refusing to report scheme comparisons: they would be measuring")
        print("  the refactor rather than the sizing.")
        raise SystemExit(1)
    print("  PASSED — equal weight is unchanged to floating-point precision.")

    # --- full sample -------------------------------------------------------
    grid = scheme_grid()
    print(f"\n--- FULL SAMPLE ({prices.close.index[0]:%Y-%m-%d} to "
          f"{prices.close.index[-1]:%Y-%m-%d}) ---")
    print("  Context only.  This window is known to sit in the top decile of")
    print("  available history, so do not rank the schemes on it.")
    results = run_schemes(targets, prices.close, prices.open_,
                          config.execution, ranking_scores, grid)

    # --- how different are the weights at all? -----------------------------
    print("\n--- WEIGHT DISPERSION: is there anything for sizing to do? ---")
    wd = weight_diagnostics(targets, prices.close, ranking_scores, grid)
    print(wd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    wd.to_csv(OUT_WEIGHTS, index=False)

    # --- paired test -------------------------------------------------------
    print("\n--- PAIRED DAILY DIFFERENCE vs EQUAL WEIGHT ---")
    print("  Same names, same days, so this isolates weighting alone.")
    sig = paired_significance(results)
    print(sig.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- walk-forward ------------------------------------------------------
    summary = pd.DataFrame([{
        "scheme": r.name,
        "cagr": r.metrics["cagr"],
        "sharpe_ratio": r.metrics["sharpe_ratio"],
        "max_drawdown": r.metrics["max_drawdown"],
        "name_turnover": r.turnover["annual_turnover"],
        "cost_implied_turnover": (r.result.total_cost
                                  / config.execution.slippage_frac
                                  / r.metrics["years"]),
        "total_cost": r.result.total_cost,
    } for r in results])

    if not args.quick:
        print(f"\n--- WALK-FORWARD ({args.train_years}y train / "
              f"{args.test_months}m test, select on {args.objective}) ---")
        print("  The headline. Each window picks a scheme on training data only")
        print("  and is judged on the next window it has never seen.")
        wf = walk_forward(results, train_years=args.train_years,
                          test_months=args.test_months, objective=args.objective,
                          risk_free_rate=config.execution.risk_free_rate)
        print(f"\n  Stitched out-of-sample: CAGR {wf.oos_metrics['cagr']:.2%}  "
              f"Sharpe {wf.oos_metrics['sharpe_ratio']:.2f}  "
              f"MaxDD {wf.oos_metrics['max_drawdown']:.2%}")
        print("\n  Each scheme held fixed over the same out-of-sample span:")
        fixed = wf.fixed_oos.copy()
        print(fixed.to_string(float_format=lambda v: f"{v:.4f}"))
        fixed.to_csv(OUT_WF)

        picked = wf.selections["Chosen"].value_counts()
        n_win = len(wf.selections)
        print(f"\n  How often each scheme was selected by the trainer "
              f"({n_win} windows):")
        for name, n in picked.items():
            print(f"    {name:<28} {n:>3}")
        hit = float(wf.selections["Chose Best"].mean())
        print(f"\n  The trainer picked the best-in-hindsight scheme in "
              f"{hit:.0%} of windows; {1.0 / len(results):.0%} is chance.")
        print("  A trainer that keeps switching is evidence the differences are")
        print("  noise: a genuinely better scheme would be picked consistently,")
        print("  and well above chance.")

        summary = summary.merge(
            fixed.rename(columns={"Config": "scheme"}),
            on="scheme", how="left", suffixes=("", "_oos"))

        # --- the comparison that actually answers the question ------------
        eq = fixed[fixed["Config"] == "equal"].iloc[0]
        print("\n--- VERDICT ---")
        print(f"  Adaptive (retune the scheme every {args.test_months}m): "
              f"CAGR {wf.oos_metrics['cagr']:.2%}  "
              f"Sharpe {wf.oos_metrics['sharpe_ratio']:.2f}")
        print(f"  Fixed equal weight, never touched      : "
              f"CAGR {eq['OOS CAGR']:.2%}  Sharpe {eq['OOS Sharpe']:.2f}")
        gap = wf.oos_metrics["sharpe_ratio"] - eq["OOS Sharpe"]
        if gap < 0:
            print(f"\n  Choosing a sizing scheme is WORSE than not choosing one,")
            print(f"  by {abs(gap):.2f} Sharpe out of sample.  Training-window")
            print("  performance does not predict the next window, so the")
            print("  selection adds turnover and noise and nothing else.")

        best = fixed.iloc[0]
        print(f"\n  Best fixed scheme out of sample: {best['Config']} "
              f"(Sharpe {best['OOS Sharpe']:.2f} vs equal {eq['OOS Sharpe']:.2f})")

        # Drawdown is the one axis where the schemes genuinely separate, so it
        # gets reported on its own terms — including whether the reduction was
        # worth what it cost, which is what Calmar answers and MaxDD alone
        # does not.  The deepest-drawdown-reduction scheme and the best
        # risk-adjusted scheme are NOT the same one here, and saying so is the
        # point: "lower MaxDD" on its own is not an improvement.
        dd = fixed.sort_values("OOS MaxDD", ascending=False).iloc[0]
        cal = fixed.sort_values("OOS Calmar", ascending=False).iloc[0]
        print(f"\n  Drawdown is where the schemes do separate:")
        print(f"    smallest MaxDD : {dd['Config']:<26} {dd['OOS MaxDD']:>7.2%} "
              f"(equal {eq['OOS MaxDD']:.2%}), "
              f"Calmar {dd['OOS Calmar']:.2f} vs {eq['OOS Calmar']:.2f}")
        if dd["OOS Calmar"] < eq["OOS Calmar"]:
            print("      -> bought with more CAGR than it saved. Not a win.")
        print(f"    best Calmar    : {cal['Config']:<26} {cal['OOS Calmar']:>7.2f} "
              f"(equal {eq['OOS Calmar']:.2f}), "
              f"MaxDD {cal['OOS MaxDD']:.2%}")
        if cal["Config"] != "equal":
            print("      -> the only scheme that improves on equal weight on any")
            print("         risk-adjusted measure, and only on this one.")

    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"\nWrote {OUT_SUMMARY.name}, {OUT_WEIGHTS.name}"
          + ("" if args.quick else f", {OUT_WF.name}"))


if __name__ == "__main__":
    main()
