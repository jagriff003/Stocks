"""
Track K option (b): put direct real-asset ETFs INTO the Track J pool.

TODO 0l.  Tier 1 found the hedge layer fails on Track J: Track J already
rotates into energy and commodity PRODUCERS on its own, and its returns
mean-revert at the horizon the layer compares on.  What Tier 3 says it still
lacks is the direct instrument: in 1973-74 commodities and bullion tripled
while energy equities fell 30%.  So instead of a layer that overrides the
model, give the model's own score the instruments and let it choose.

THE COMPONENTS, AND WHY THESE

One ETF per component, picked so they do not crowd each other out of the
correlation cap (daily correlations 2011-2026 in brackets):

  oil          DBO   optimum-yield roll; USO lost ~75% in April 2020 on contango
  natural gas  UNG   (0.15 with DBO)
  agriculture  DBA   (0.29 with DBO)
  base metals  DBB   (0.34 with DBO; CPER is thinner and 0.54 with DBB)
  silver       SLV   (0.42 with DBB at most)
  gold         IAU   already in the pool; ranks 469th on 2026-09-21

  dollar       UUP   not a real asset, but on the agreed hedge menu (James asked)

DBC is left out: it is 0.90 with DBO, a second copy of the oil bet.

THE SCREENS

The $10M dollar-volume floor exists for stocks — delisting and impact.  DBO and
DBB fail it most years ($1-8M), yet an order here is ~$8k (account / 32
positions).  They bypass it the way IAU/SHY/TLT already do; the slippage model
still charges them on their real point-in-time volume.

ARMS

  A0  Track J exactly as the live runner builds it
  A1  + components, correlation filter applied to them as to any name
  A2  + components, EXEMPT from the correlation filter
      (`CorrelationConfig.exempt_symbols`, added for this; empty by default)

Each at both biweekly phases (the anchor Tuesday and the Tuesday after): the
repo has seen double-digit CAGR spreads from phase alone (TODO 0i).

CHECKS (the script refuses to print results if any fails)

  A  A0 reproduces the runner's stored stream, day by day
  B  the augmented panel with the components made unselectable reproduces A0
     exactly — the join, the correlation matrix and the cost panel change
     nothing by themselves
  C  nothing restricted reaches either panel

Run:  python scripts/analyze_trackj_real_assets.py
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings("ignore")

from momentum.backtest import _PortfolioBuilder, simulate_portfolio  # noqa: E402
from momentum.data import PriceData  # noqa: E402
from momentum.liquidity import (LiquidityConfig, apply_tradable, slippage_panel,  # noqa: E402
                                tradable_mask)
from momentum.metrics import calculate_performance_metrics  # noqa: E402
from momentum.restrictions import check_symbols  # noqa: E402
from momentum.reversal import ReversalConfig, build_terms, composite  # noqa: E402
from momentum.schedule import TUESDAY, rotation_dates, sleeve_assignment  # noqa: E402
from momentum.strategy import compute_scores  # noqa: E402
from momentum.universe import defensive_symbols  # noqa: E402
from scripts.analyze_reversal_backtest import load_volume  # noqa: E402
from scripts.analyze_reversal_ic import NON_EQUITY, load_pool_panel, pool_symbols  # noqa: E402
from scripts.run_live_trackj import POOL_FILE, build_config  # noqa: E402

COMPONENTS = {"DBO": "oil", "UNG": "natural gas", "DBA": "agriculture",
              "DBB": "base metals", "SLV": "silver", "IAU": "gold", "UUP": "dollar"}
NEW = [s for s in COMPONENTS if s != "IAU"]
REAL = list(COMPONENTS)

# the live runner's defaults
START, TOP_N, HOLD, TRANCHES, EVERY_WEEKS = "2010-01-01", 8, 40, 4, 2
ANCHORS = {"phase A (anchor 2026-09-15)": "2026-09-15", "phase B (+1 week)": "2026-09-22"}
MIN_ADV, MIN_PRICE, ACCOUNT = 10e6, 5.0, 100_000.0
RF = 0.045   # the runner's metrics convention

EPISODES = [("2015-16 commodity crash", "2015-06-01", "2016-02-29"),
            ("2018 Q4 selloff", "2018-10-01", "2018-12-24"),
            ("2020 COVID crash", "2020-02-19", "2020-03-23"),
            ("2020 rebound", "2020-03-24", "2020-08-31"),
            ("2021-22 inflation", "2021-01-01", "2022-10-31"),
            ("2022 alone", "2022-01-01", "2022-12-31"),
            ("2023-24", "2023-01-01", "2024-12-31"),
            ("2025-26 metals run", "2025-01-01", "2026-07-31"),
            ("2026 YTD", "2026-01-01", None)]


def load_panels():
    defensive = set(defensive_symbols())
    raw = sorted(set(pd.read_csv(REPO_ROOT / POOL_FILE)["symbol"].dropna()) - NON_EQUITY)
    base = load_pool_panel(raw + sorted(defensive), start=START, verbose=False)
    allowed = set(pool_symbols(POOL_FILE, verbose=False)) | defensive
    keep = [c for c in base.close.columns if c in allowed]
    base = PriceData(close=base.close[keep], open_=base.open_[keep], spy=base.spy, vix=base.vix)
    vol = load_volume(raw + sorted(defensive), start=START, verbose=False)
    vol = vol.reindex(index=base.close.index, columns=base.close.columns)

    etf = load_pool_panel(NEW, start=START, verbose=False)
    evol = load_volume(NEW, start=START, verbose=False)
    idx = base.close.index
    aug = PriceData(close=pd.concat([base.close, etf.close.reindex(idx)[NEW]], axis=1),
                    open_=pd.concat([base.open_, etf.open_.reindex(idx)[NEW]], axis=1),
                    spy=base.spy, vix=base.vix)
    avol = pd.concat([vol, evol.reindex(index=idx)[NEW]], axis=1)
    return base, vol, aug, avol, defensive


def score(prices, volume, cfg, always, lcfg):
    """Exactly the runner's scoring path."""
    rcfg = ReversalConfig()
    tmask = tradable_mask(prices.close, volume, MIN_ADV, MIN_PRICE, lcfg)
    ranking, _, _ = compute_scores(prices, cfg)
    terms = build_terms(prices.close, rcfg)
    pull = composite(terms, replace(rcfg, turn_weight=-1.0, strength_weight=0.0,
                                    room_weight=1.0), "flip").reindex_like(ranking)
    return apply_tradable(pull, tmask, always)


def simulate(prices, volume, scored, cfg, anchor, lcfg):
    """The runner's sleeve simulation: k sleeves on the Tuesday calendar."""
    rot = rotation_dates(prices.close.index, weekday=TUESDAY, every_weeks=EVERY_WEEKS,
                         anchor=pd.Timestamp(anchor))
    assign = sleeve_assignment(rot, TRANCHES)
    builder = _PortfolioBuilder(
        scored, price_columns=list(prices.close.columns), top_n=cfg.top_n,
        min_data_days=cfg.min_data_days, hold_days=cfg.hold_days, vix_data=prices.vix,
        vix_config=cfg.vix, base_composite_scores=None, velocity_config=cfg.velocity,
        correlation_config=cfg.correlation, graduated_config=cfg.graduated_vix,
        exit_config=None, close=prices.close, rank_offset=cfg.rank_offset,
        rank_offset_scope=cfg.rank_offset_scope, monitor_symbols=cfg.monitor_symbols)
    slip = slippage_panel(prices.close, volume, lcfg, top_n=cfg.top_n * TRANCHES)
    rets, weights = [], []
    for j in range(TRANCHES):
        picks = {}
        for d in [d for d in rot if assign[d] == j]:
            rec = builder.select_on(d)
            if rec:
                picks[d] = list(rec["Selected_Stocks"])
        if not picks:
            continue
        tgt = pd.Series({d: picks[d] for d in sorted(picks)}, dtype=object)
        tgt = tgt.reindex(prices.close.index).ffill().dropna()
        r = simulate_portfolio(tgt, prices.close, prices.open_, execution=cfg.execution,
                               sizing=cfg.sizing, scores=scored, slippage_by_symbol=slip)
        rets.append(r.returns.rename(f"s{j}"))
        # each sleeve is 1/TRANCHES of capital, equal weight inside it
        w = pd.DataFrame(0.0, index=list(r.holdings.keys()), columns=REAL)
        for d, names in r.holdings.items():
            n = len(names)
            for s in names:
                if s in REAL and n:
                    w.at[d, s] += 1.0 / n / TRANCHES
        weights.append(w)
    combined = pd.concat(rets, axis=1).mean(axis=1).dropna()
    wt = sum(w.reindex(combined.index).fillna(0.0) for w in weights)
    return combined, wt


def rank_diagnostic(scored, rot):
    """
    Where each component sits in the cross-section on rotation dates.  Track J
    takes the top 8 of ~610 tradable names — the top ~1.3% — so a name has to
    be extreme on the pullback score to be held at all.
    """
    rows = []
    s = scored.loc[scored.index.intersection(rot)]
    rk = s.rank(axis=1, ascending=False)
    n = s.notna().sum(axis=1)
    for c in REAL:
        r = rk[c].dropna()
        pct = (r / n.loc[r.index])
        rows.append({"component": f"{c} ({COMPONENTS[c]})", "rotations scored": len(r),
                     "median rank": r.median(), "best rank": r.min(),
                     "median percentile": pct.median(),
                     "in top 8": (r <= 8).mean(), "in top 32": (r <= 32).mean()})
    return pd.DataFrame(rows).set_index("component")


def metrics(r):
    m = calculate_performance_metrics(r.dropna(), risk_free_rate=RF)
    return {k: m[k] for k in ("cagr", "sharpe_ratio", "sortino_ratio", "max_drawdown",
                              "calmar_ratio", "volatility")}


def main():
    pd.set_option("display.width", 250)
    lcfg = LiquidityConfig(account_notional=ACCOUNT)
    cfg = build_config(TOP_N, HOLD)
    base, vol, aug, avol, defensive = load_panels()
    check_symbols(list(base.close.columns), "Track J panel")
    check_symbols(list(aug.close.columns), "Track J + real assets panel")
    print(f"Panels: base {base.close.shape[1]} names, augmented {aug.close.shape[1]}; "
          f"{base.close.index[0]:%Y-%m-%d} .. {base.close.index[-1]:%Y-%m-%d}")
    print("CHECK C  restricted list applied to both panels: OK")

    anchor = ANCHORS["phase A (anchor 2026-09-15)"]
    s0 = score(base, vol, cfg, defensive, lcfg)
    a0, _ = simulate(base, vol, s0, cfg, anchor, lcfg)

    stored = pd.read_csv(REPO_ROOT / "trackj_portfolio_performance.csv", index_col=0,
                         parse_dates=True)["Portfolio_Return"]
    common = stored.index.intersection(a0.index)
    gap = (stored.loc[common] - a0.loc[common]).abs().max()
    print(f"CHECK A  A0 vs the runner's stored stream: {len(common)} common days, max daily gap {gap:.1e}")
    assert len(common) > 3500 and gap < 1e-10, gap

    # B: augmented panel, components unselectable (base scores, NaN for the new columns)
    s_blank = s0.reindex(columns=aug.close.columns)
    b_chk, _ = simulate(aug, avol, s_blank, cfg, anchor, lcfg)
    gap_b = (b_chk - a0.reindex(b_chk.index)).abs().max()
    print(f"CHECK B  augmented panel, components unselectable, vs A0: max daily gap {gap_b:.1e}")
    assert gap_b < 1e-12, gap_b
    print()

    always = defensive | set(REAL)
    s_aug = score(aug, avol, cfg, always, lcfg)
    cfg_ex = replace(cfg, correlation=replace(cfg.correlation, exempt_symbols=tuple(REAL)))
    arms = {"A0 Track J": (base, vol, s0, cfg),
            "A1 + components": (aug, avol, s_aug, cfg),
            "A2 + components, corr-exempt": (aug, avol, s_aug, cfg_ex)}

    streams, wts, rows = {}, {}, []
    for pname, anc in ANCHORS.items():
        for aname, (p, v, sc, c) in arms.items():
            if aname.startswith("A0") and anc == anchor:
                r, w = a0, pd.DataFrame(0.0, index=a0.index, columns=REAL)
            else:
                print(f"  simulating {aname}, {pname}...", flush=True)
                r, w = simulate(p, v, sc, c, anc, lcfg)
            streams[(aname, pname)] = r
            wts[(aname, pname)] = w
            rows.append({"arm": aname, "phase": pname, **metrics(r),
                         "real-asset weight": w.sum(axis=1).mean(),
                         "days holding any": (w.sum(axis=1) > 0).mean()})
    tab = pd.DataFrame(rows).set_index(["arm", "phase"])

    print("\n" + "=" * 110)
    print("HEADLINE — runner convention (Sharpe over a fixed 4.5%)")
    print("=" * 110)
    show = tab.copy()
    for c in show.columns:
        show[c] = show[c].map(lambda x: f"{x:.2f}" if c in ("sharpe_ratio", "sortino_ratio", "calmar_ratio")
                              else f"{x:+.2%}")
    print(show.to_string())

    print("\nDELTA vs A0, same phase")
    drows = []
    for (aname, pname), r in streams.items():
        if aname.startswith("A0"):
            continue
        b = tab.loc[("A0 Track J", pname)]
        a = tab.loc[(aname, pname)]
        drows.append({"arm": aname, "phase": pname, "dCAGR": a["cagr"] - b["cagr"],
                      "dSharpe": a["sharpe_ratio"] - b["sharpe_ratio"],
                      "dMaxDD": a["max_drawdown"] - b["max_drawdown"]})
    dt = pd.DataFrame(drows).set_index(["arm", "phase"])
    print(dt.assign(dCAGR=dt["dCAGR"].map("{:+.2%}".format),
                    dSharpe=dt["dSharpe"].map("{:+.2f}".format),
                    dMaxDD=dt["dMaxDD"].map("{:+.2%}".format)).to_string())

    print("\nEPISODES — cumulative return (phase A)")
    erows = []
    for name, a, b in EPISODES:
        row = {"episode": name}
        for aname in arms:
            r = streams[(aname, "phase A (anchor 2026-09-15)")].loc[a:b]
            row[aname] = (1 + r).prod() - 1
        row["A2 real-asset weight"] = wts[("A2 + components, corr-exempt", "phase A (anchor 2026-09-15)")].loc[a:b].sum(axis=1).mean()
        erows.append(row)
    ep = pd.DataFrame(erows).set_index("episode")
    print(ep.map(lambda x: f"{x:+.1%}").to_string())

    print("\nWHICH COMPONENTS THE SCORE USED — mean book weight by year (A2, phase A)")
    w2 = wts[("A2 + components, corr-exempt", "phase A (anchor 2026-09-15)")]
    by_year = w2.resample("YE").mean()
    by_year.index = by_year.index.year
    by_year.columns = [f"{s} ({COMPONENTS[s]})" for s in by_year.columns]
    print(by_year.map(lambda x: f"{x:.1%}" if x else "-").to_string())
    w1 = wts[("A1 + components", "phase A (anchor 2026-09-15)")]
    print(f"\nA1 (filtered) mean real-asset weight {w1.sum(axis=1).mean():.1%} vs A2 (exempt) "
          f"{w2.sum(axis=1).mean():.1%}: what the correlation cap was blocking")

    rot = rotation_dates(aug.close.index, weekday=TUESDAY, every_weeks=EVERY_WEEKS,
                         anchor=pd.Timestamp(anchor))
    rd = rank_diagnostic(s_aug, rot)
    print("\nWHY SO LITTLE: where each component ranks on the pullback score, all rotation dates")
    print(rd.assign(**{"median percentile": rd["median percentile"].map("{:.0%}".format),
                       "in top 8": rd["in top 8"].map("{:.1%}".format),
                       "in top 32": rd["in top 32"].map("{:.1%}".format)}).to_string())

    rd.to_csv(REPO_ROOT / "trackk_option_b_ranks.csv")
    tab.to_csv(REPO_ROOT / "trackk_option_b_headline.csv")
    ep.to_csv(REPO_ROOT / "trackk_option_b_episodes.csv")
    by_year.to_csv(REPO_ROOT / "trackk_option_b_usage.csv")


if __name__ == "__main__":
    main()
