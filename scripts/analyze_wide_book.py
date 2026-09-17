"""
The wide-book redesign: does holding more of the universe survive the overlay?

Track F found that the universe produces the return and the ranker produces
almost none of it (+0.84pp gross, 61st percentile of random draws), while the
overlay produces the entire drawdown advantage (-19.23% against the universe's
-35.22%).  The obvious response is to hold more of the universe and stop paying
attention to the ranking.  There is one serious obstacle, and this script exists
to test it rather than assume past it.

THE OBSTACLE
   Track F also found the overlay is worth +3.03pp gross on RANKED picks and
   -1.18pp on RANDOM ones.  Its mechanism is concentration: in an elevated
   regime it cuts the book to `elevated_top_n` momentum names, and the top-K cut
   shows the ranker's edge lives at exactly ranks 1-2.  The overlay works by
   forcing the book into the only part of the ranking that carries information.

   A wide book has no such mechanism available.  Worse, the de-risking itself
   does not scale: only three defensive tickers exist, so a 25-name book can be
   at most 12% defensive no matter what the rule says.  **The drawdown control -
   the single thing the model demonstrably provides - is the part most at risk
   from going wide.**  That is the hypothesis under test.

THREE SWEEPS

   1. BOOK SIZE x OVERLAY, slot-based.  The live overlay, applied to books of
      4 to 40.  Expect it to degrade: this measures how fast.

   2. WEIGHT-BASED OVERLAY.  The same intent expressed as a defensive WEIGHT
      instead of a slot count (`momentum.drift.defensive_weight_targets`), which
      is the only form that scales.  A 25-name book can hold 30% defensive
      across three tickers; it cannot hold 30% of its SLOTS in three tickers.

   3. ROTATION AND REBALANCING.  How much of either a wide book actually needs.
      Uses the weight-aware simulator throughout, so "rebalancing" is a real
      question rather than an assumption baked into the return calculation.

Everything runs through `momentum.drift.simulate_with_drift`, whose gross stream
is asserted equal to `simulate_portfolio`'s under policy='daily' before any of
this is reported.  Book sizes stop at 40: at 49 the eligibility gate binds on
19% of dates and the run starts 18 months later, which would not be comparable.

Every threshold is a flag.

Run:  python scripts/analyze_wide_book.py
      python scripts/analyze_wide_book.py --sizes 4 10 20 30 --sweep 1 2
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.backtest import build_target_portfolios
from momentum.data import load_data
from momentum.drift import (RebalanceConfig, defensive_weight_targets,
                            simulate_with_drift, validate_against_legacy)
from momentum.experiments import production_config
from momentum.metrics import calculate_performance_metrics
from momentum.regime import calculate_vix_regime
from momentum.strategy import compute_scores
from momentum.universe import current_symbols, defensive_symbols

OUT = "rsi_ma_wide_book.csv"


def targets_for(cfg, prices, ranking, base):
    """Target book series under `cfg`, via the production builder."""
    return build_target_portfolios(
        ranking, price_columns=list(prices.close.columns), top_n=cfg.top_n,
        min_data_days=cfg.min_data_days, hold_days=cfg.hold_days,
        vix_data=prices.vix, vix_config=cfg.vix, base_composite_scores=base,
        velocity_config=cfg.velocity, correlation_config=cfg.correlation,
        graduated_config=cfg.graduated_vix, exit_config=cfg.exits,
        close=prices.close, rank_offset=cfg.rank_offset,
        rank_offset_scope=cfg.rank_offset_scope)[0]


def run(label, cfg, prices, ranking, base, rebalance, weight_targets=None,
        note=""):
    targets = targets_for(cfg, prices, ranking, base)
    r = simulate_with_drift(targets, prices.close, prices.open_,
                            execution=cfg.execution, rebalance=rebalance,
                            weight_targets=weight_targets)
    g = calculate_performance_metrics(
        r.gross_returns, risk_free_rate=cfg.execution.risk_free_rate)
    defensive = set(defensive_symbols())
    dw = r.weights.reindex(columns=[c for c in r.weights.columns
                                    if c in defensive]).sum(axis=1)
    return {
        "Variant": label,
        "Top N": cfg.top_n,
        "Hold Days": cfg.hold_days,
        "Rebalance": rebalance.label,
        "CAGR": r.metrics["cagr"],
        "Gross CAGR": g["cagr"],
        "Sharpe": r.metrics["sharpe_ratio"],
        "MaxDD": r.metrics["max_drawdown"],
        "Calmar": r.metrics["calmar_ratio"],
        "Vol": r.metrics["volatility"],
        "Trades/Year": r.turnover["trades_per_year"],
        "Total Cost": r.total_cost,
        "Rebalances": r.n_rebalances,
        "Max Weight": float(r.max_weight.max()),
        "Mean Defensive Wt": float(dw.mean()),
        "Note": note,
    }


def show(rows, title, subtitle=""):
    df = pd.DataFrame(rows)
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)
    if subtitle:
        print(subtitle)
    header = (f"{'Variant':<30}{'CAGR':>9}{'Gross':>9}{'Sharpe':>8}{'MaxDD':>9}"
              f"{'Calmar':>8}{'Vol':>8}{'Trd/Yr':>8}{'MaxWt':>8}{'DefWt':>8}")
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        print(f"{r['Variant']:<30}{r['CAGR']:>8.2%} {r['Gross CAGR']:>8.2%} "
              f"{r['Sharpe']:>7.2f} {r['MaxDD']:>8.2%} {r['Calmar']:>7.2f} "
              f"{r['Vol']:>7.2%} {r['Trades/Year']:>7.1f} "
              f"{r['Max Weight']:>7.1%} {r['Mean Defensive Wt']:>7.1%}")
    return df


def sweep_book_size(prices, ranking, base, sizes, rebal):
    """Sweep 1 - the live slot-based overlay, applied to wider and wider books."""
    rows = []
    for n in sizes:
        rows.append(run(f"top_n={n}, overlay on", production_config(top_n=n),
                        prices, ranking, base, rebal))
        rows.append(run(f"top_n={n}, overlay OFF",
                        production_config(top_n=n, vix=None),
                        prices, ranking, base, rebal))
    return show(rows, "SWEEP 1 - BOOK SIZE x SLOT-BASED OVERLAY (the live mechanism)",
                "  DefWt is the realized defensive weight. Watch it collapse as the\n"
                "  book widens: three tickers cannot fill a large book's slots.")


def sweep_weight_overlay(prices, ranking, base, sizes, weights, rebal, cfg0):
    """Sweep 2 - the overlay as a defensive weight, which scales."""
    regimes, _ = calculate_vix_regime(prices.vix, cfg0.vix)
    defensive = defensive_symbols()
    rows = []
    for n in sizes:
        cfg = production_config(top_n=n, vix=None)   # no slot-based overlay
        rows.append(run(f"top_n={n}, def wt 0%", cfg, prices, ranking, base,
                        rebal, note="no overlay"))
        for w in weights:
            wt = defensive_weight_targets(
                regimes, defensive,
                {"elevated": w, "crisis": min(1.0, w * 2)})
            rows.append(run(f"top_n={n}, def wt {w:.0%}/{min(1.0, w*2):.0%}",
                            cfg, prices, ranking, base, rebal,
                            weight_targets=wt,
                            note=f"elevated {w:.0%}, crisis {min(1.0,w*2):.0%}"))
    return show(rows, "SWEEP 2 - WEIGHT-BASED OVERLAY (elevated / crisis defensive weight)",
                "  The defensive sleeve is held by WEIGHT, so it is not capped by\n"
                "  having only three tickers. This is the only version that scales.")


def sweep_rotation(prices, ranking, base, n, holds, policies, cfg0, def_weight):
    """
    Sweep 3 - how much rotation and rebalancing a wide book needs.

    Deliberately runs the WEIGHT-based overlay, not the slot-based one.  The
    slot overlay can only re-select on a rotation date, so at a 126-day hold it
    sits through an entire crisis switched off; a sweep using it reports that
    long holds have terrible drawdown when what it measured was an overlay
    unable to engage.  The weight overlay engages daily, which keeps the risk
    dial off the selection clock and leaves rotation frequency as the only
    thing varying.
    """
    regimes, _ = calculate_vix_regime(prices.vix, cfg0.vix)
    crisis_w = min(1.0, def_weight * 2)
    wt = defensive_weight_targets(
        regimes, defensive_symbols(),
        {"elevated": def_weight, "crisis": crisis_w})
    rows = []
    for h in holds:
        cfg = production_config(top_n=n, hold_days=h, vix=None)
        for pol in policies:
            rows.append(run(f"hold {h}d, {pol.label}", cfg, prices, ranking,
                            base, pol, weight_targets=wt))
    return show(rows,
                f"SWEEP 3 - ROTATION AND REBALANCING (top_n={n}, "
                f"weight overlay {def_weight:.0%}/{crisis_w:.0%})",
                "  Rotation changes WHICH names are held; rebalancing"
                " changes how much of each.\n"
                "  The weight overlay engages daily, so it is not"
                " confounded with the rotation clock.")


def main() -> int:
    p = argparse.ArgumentParser(description="Wide-book redesign tests")
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 10, 15, 20, 25, 30, 40])
    p.add_argument("--def-weights", type=float, nargs="+", default=[0.15, 0.30, 0.50])
    p.add_argument("--wide-n", type=int, default=20,
                   help="book size used for the rotation sweep")
    p.add_argument("--holds", type=int, nargs="+", default=[14, 30, 63, 126, 252])
    p.add_argument("--rotation-def-weight", type=float, default=0.30,
                   help="defensive weight used for the rotation sweep's overlay")
    p.add_argument("--sweep", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--keep-unsettled", action="store_true")
    args = p.parse_args()

    print("=" * 112)
    print("WIDE-BOOK REDESIGN")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 112)

    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache,
                       drop_unsettled=not args.keep_unsettled)
    cfg0 = production_config()
    ranking, base, _ = compute_scores(prices, cfg0)

    print("\nValidating the weight-aware simulator against simulate_portfolio:")
    t0 = targets_for(cfg0, prices, ranking, base)
    gap = validate_against_legacy(t0, prices.close, prices.open_, cfg0.execution)
    print(f"  gross streams agree to {gap:.2e}")

    rebal = RebalanceConfig(policy="on_rotation")
    frames = []
    if 1 in args.sweep:
        frames.append(sweep_book_size(prices, ranking, base, args.sizes, rebal))
    if 2 in args.sweep:
        frames.append(sweep_weight_overlay(prices, ranking, base,
                                           [n for n in args.sizes if n >= 10],
                                           args.def_weights, rebal, cfg0))
    if 3 in args.sweep:
        policies = [RebalanceConfig(policy="on_rotation"),
                    RebalanceConfig(policy="never"),
                    RebalanceConfig(policy="periodic", period_days=63),
                    RebalanceConfig(policy="band", band=0.25)]
        frames.append(sweep_rotation(prices, ranking, base, args.wide_n,
                                     args.holds, policies, cfg0,
                                     args.rotation_def_weight))

    if frames:
        pd.concat(frames).to_csv(REPO_ROOT / OUT, index=False)
        print(f"\nExported to: {REPO_ROOT / OUT}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
