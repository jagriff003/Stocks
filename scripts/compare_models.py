"""
The baseline change: the production model against Track J, as each is configured today.

Not a parameter study.  Each model is run exactly as it would be run tomorrow —
its own universe, its own book size, its own rotation clock, its own overlay —
and the difference between them is the change being contemplated.

WHY THE UNIVERSES DIFFER, AND WHY THAT IS NOT A CONFOUND

Every earlier comparison in this work deliberately held the universe constant so
that a ranker could be compared to a ranker.  This one does the opposite on
purpose.  Moving from 46 curated names to ~745 screened ones IS the change; it
is not a nuisance variable to be divided out.  Track F established that the
universe carries most of the return, so a comparison that equalised it would be
measuring the smaller half of the decision.

WHAT TO DISTRUST

Both configurations were selected with knowledge of this sample, and in
different ways that are worth naming separately:

  production  Its scoring defects were found and fixed in response to 2026's
              poor results, so backtesting it over 2026 measures a config chosen
              with knowledge of the outcome. Its YTD is circular.
  Track J     Its score, book size, rotation and correlation cap were all chosen
              on the full sample during a single day's work. Its levels are
              in-sample; only the out-of-sample window test escapes that, and it
              is reported separately in FINDINGS.

Neither number is a forecast.  The delta is the least contaminated thing here,
and even it inherits survivorship from both pools.

Run:  python scripts/compare_models.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.backtest import _PortfolioBuilder, simulate_portfolio
from momentum.data import PriceData, load_data
from momentum.experiments import production_config
from momentum.liquidity import (LiquidityConfig, apply_tradable,
                                slippage_panel, tradable_mask)
from momentum.metrics import calculate_performance_metrics
from momentum.reversal import ReversalConfig, build_terms, composite
from momentum.schedule import TUESDAY, rotation_dates, sleeve_assignment
from momentum.strategy import run_strategy
from momentum.universe import current_symbols, defensive_symbols

from scripts.analyze_reversal_backtest import load_volume
from scripts.analyze_reversal_ic import NON_EQUITY, load_pool_panel, pool_symbols

OUT = "rsi_ma_model_comparison.csv"
POOL_FILE = "random_pool.csv"
METRICS = ["cagr", "volatility", "sharpe_ratio", "sortino_ratio",
           "max_drawdown", "calmar_ratio"]


def windows(r: pd.Series) -> dict:
    r = r.dropna()
    def seg(s):
        x = r.loc[s:]
        return (1 + x).prod() - 1 if len(x) else np.nan
    thirds = np.array_split(np.arange(len(r)), 3)
    return {
        "YTD 2026": seg("2026-01-01"),
        "last 126d": (1 + r.iloc[-126:]).prod() - 1,
        "last 63d": (1 + r.iloc[-63:]).prod() - 1,
        **{f"P{i+1}": calculate_performance_metrics(
            r.iloc[t[0]:t[-1] + 1])["cagr"] for i, t in enumerate(thirds)},
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Production vs Track J, as configured")
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--tranches", type=int, default=4)
    p.add_argument("--every-weeks", type=int, default=2)
    p.add_argument("--anchor", default="2026-09-15")
    p.add_argument("--hold", type=int, default=40)
    p.add_argument("--max-corr", type=float, default=0.70)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--min-adv", type=float, default=10e6)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--account", type=float, default=100_000.0)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    print("=" * 100)
    print("BASELINE CHANGE — production model vs Track J, each as configured today")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 100)

    streams = {}

    # --- production, untouched ---
    print("\n  running the production model (46 names, top_n=4, hold=14, "
          "VIX overlay on)...", flush=True)
    prod_cfg = production_config()
    p46 = load_data(current_symbols(), start_date=args.start,
                    use_cache=not args.no_cache, cache_max_age_hours=1e9,
                    verbose=False)
    prod = run_strategy(p46, prod_cfg)
    streams["production (46 names)"] = (prod.returns, prod.turnover,
                                        prod.total_cost)

    # --- Track J, as the live runner builds it ---
    print("  running Track J (pullback, ~745 pool, 4 sleeves, Tuesday "
          "calendar)...", flush=True)
    cfg = replace(production_config(), top_n=args.top_n, hold_days=args.hold,
                  vix=None)
    cfg = replace(cfg, correlation=replace(
        cfg.correlation, enabled=True, apply_above_vix=None,
        method="absolute", max_correlation=args.max_corr))
    rcfg = ReversalConfig()
    lcfg = LiquidityConfig(account_notional=args.account)
    defensive = set(defensive_symbols())

    raw = sorted(set(pd.read_csv(REPO_ROOT / POOL_FILE)["symbol"].dropna())
                 - NON_EQUITY)
    full = load_pool_panel(raw + sorted(defensive), start=args.start,
                           use_cache=not args.no_cache)
    allowed = set(pool_symbols(POOL_FILE)) | defensive
    keep = [c for c in full.close.columns if c in allowed]
    full = PriceData(close=full.close[keep], open_=full.open_[keep],
                     spy=full.spy, vix=full.vix)
    vol = load_volume(raw + sorted(defensive), start=args.start,
                      use_cache=not args.no_cache)
    vol = vol.reindex(index=full.close.index, columns=full.close.columns)

    tmask = tradable_mask(full.close, vol, args.min_adv, args.min_price, lcfg)
    terms = build_terms(full.close, rcfg)
    scored = apply_tradable(
        composite(terms, replace(rcfg, turn_weight=-1.0, strength_weight=0.0,
                                 room_weight=1.0), "flip"), tmask, defensive)

    rot = rotation_dates(full.close.index, weekday=TUESDAY,
                         every_weeks=args.every_weeks,
                         anchor=pd.Timestamp(args.anchor))
    assign = sleeve_assignment(rot, args.tranches)
    builder = _PortfolioBuilder(
        scored, price_columns=list(full.close.columns), top_n=cfg.top_n,
        min_data_days=cfg.min_data_days, hold_days=cfg.hold_days,
        vix_data=full.vix, vix_config=None, base_composite_scores=None,
        velocity_config=cfg.velocity, correlation_config=cfg.correlation,
        graduated_config=None, exit_config=None, close=full.close,
        rank_offset=0, rank_offset_scope="all", monitor_symbols=[])
    slip = slippage_panel(full.close, vol, lcfg,
                          top_n=cfg.top_n * args.tranches)

    rets, turns, cost = [], [], 0.0
    for j in range(args.tranches):
        picks = {}
        for d in [d for d in rot if assign[d] == j]:
            rec = builder.select_on(d)
            if rec:
                picks[d] = list(rec["Selected_Stocks"])
        if not picks:
            continue
        tgt = pd.Series({d: picks[d] for d in sorted(picks)}, dtype=object)
        tgt = tgt.reindex(full.close.index).ffill().dropna()
        r = simulate_portfolio(tgt, full.close, full.open_,
                               execution=cfg.execution, sizing=cfg.sizing,
                               scores=scored, slippage_by_symbol=slip)
        rets.append(r.returns.rename(f"s{j}"))
        turns.append(r.turnover)
        cost += r.total_cost
    combined = pd.concat(rets, axis=1).mean(axis=1).dropna()
    tj_turn = {"annual_turnover": np.mean([t["annual_turnover"] for t in turns]),
               "trades_per_year": np.sum([t["trades_per_year"] for t in turns])}
    streams["Track J (4 sleeves)"] = (combined, tj_turn, cost)

    # --- align to a common window, or the comparison is a date comparison ---
    common = None
    for r, _, _ in streams.values():
        idx = r.dropna().index
        common = idx if common is None else common.intersection(idx)
    print(f"\n  Common window: {common[0]:%Y-%m-%d} to {common[-1]:%Y-%m-%d} "
          f"({len(common):,} sessions)")

    rows = []
    for name, (r, turn, c) in streams.items():
        rr = r.reindex(common).dropna()
        m = calculate_performance_metrics(
            rr, risk_free_rate=production_config().execution.risk_free_rate)
        rows.append({"Model": name, **{k: m[k] for k in METRICS},
                     "turnover": turn.get("annual_turnover", np.nan),
                     "trades_yr": turn.get("trades_per_year", np.nan),
                     **windows(rr)})

    df = pd.DataFrame(rows).set_index("Model")
    df.to_csv(REPO_ROOT / OUT)

    print("\n" + "=" * 100)
    print("HEADLINE")
    print("=" * 100)
    hdr = (f"    {'model':<26}{'CAGR':>9}{'Sharpe':>8}{'Sortino':>9}"
           f"{'MaxDD':>9}{'Calmar':>8}{'Vol':>8}{'turn':>8}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name, r in df.iterrows():
        print(f"    {name:<26}{r['cagr']:>9.2%}{r['sharpe_ratio']:>8.2f}"
              f"{r['sortino_ratio']:>9.2f}{r['max_drawdown']:>9.2%}"
              f"{r['calmar_ratio']:>8.2f}{r['volatility']:>8.2%}"
              f"{r['turnover']:>8.0%}")

    a, b = df.index[0], df.index[1]
    print(f"\n    CHANGE, {b} minus {a}:")
    for k, lab, fmt in (("cagr", "CAGR", "{:+.2%}"),
                        ("sharpe_ratio", "Sharpe", "{:+.2f}"),
                        ("sortino_ratio", "Sortino", "{:+.2f}"),
                        ("max_drawdown", "Max drawdown", "{:+.2%}"),
                        ("calmar_ratio", "Calmar", "{:+.2f}"),
                        ("volatility", "Volatility", "{:+.2%}"),
                        ("turnover", "Turnover", "{:+.0%}")):
        d = df.loc[b, k] - df.loc[a, k]
        note = ""
        if k == "max_drawdown":
            note = "  (deeper)" if d < 0 else "  (shallower)"
        print(f"      {lab:<16}{fmt.format(d):>10}{note}")

    print("\n" + "=" * 100)
    print("BY PERIOD")
    print("=" * 100)
    cols = ["P1", "P2", "P3", "YTD 2026", "last 126d", "last 63d"]
    hdr2 = f"    {'model':<26}" + "".join(f"{c:>12}" for c in cols)
    print(hdr2)
    print("    " + "-" * (len(hdr2) - 4))
    for name, r in df.iterrows():
        print(f"    {name:<26}" + "".join(f"{r[c]:>11.2%} " for c in cols))
    print(f"    {'change':<26}" +
          "".join(f"{df.loc[b, c] - df.loc[a, c]:>+11.2%} " for c in cols))
    print("\n    P1/P2/P3 are equal thirds of the common window.")
    print("    YTD 2026 is CIRCULAR for the production model: its config was")
    print("    chosen in response to this year's results. Do not read it as a")
    print("    forecast, and do not read the change in it as an improvement.")

    print(f"\n  Exported to {REPO_ROOT / OUT}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
