"""
Track K option (a): does the regime trigger fire when it should, and rarely otherwise?

TODO 0l.  Track K is now a PREPARATION model run alongside Track J: it should
speak only when the environment looks like the one it prepares for.  The
trigger, fixed before this script first ran (2026-09-23):

    FIRE when at least 2 of {gold, silver, commodities, energy} beat the MARKET
    by >= 10% over ~3 months AND beat cash,
    AND the stock-bond correlation (market vs 10y Treasuries) over ~1 year is
    positive.

Judged against the market (SPY / the French market), not Track J: Tier 1
found Track J mean-reverts at this horizon, so a competition against its own
trailing return hedges just before its recoveries.

PASS CRITERIA, stated before the run

  fires in 1973-74, 1977-81 and 2021-22 (and 1946-48 where commodity data
  allows); fires in under ~15% of months outside those episodes.

THREE READINGS

  1. monthly, 1927-2026, French market as the book (Tier 3 data)
  2. daily, 2006-2026, SPY as the book, ETFs from the store (Tier 2 data)
  3. followed mechanically on Track J, decided on its rotation Tuesdays,
     2012-2026 — what it would have cost to obey it (Tier 1 data)

Run:  python scripts/analyze_hedge_trigger.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.hedge import HedgeConfig, apply_weights, hedge_weights, regime_open  # noqa: E402
from momentum.longhistory import HEDGE_ASSETS, load_long_history  # noqa: E402
from momentum.schedule import TUESDAY, rotation_dates  # noqa: E402
from scripts.analyze_hedge_etfs import load as load_etfs  # noqa: E402
from scripts.analyze_hedge_trackj import CANDIDATE as TIER1  # noqa: E402
from scripts.analyze_hedge_trackj import load as load_trackj  # noqa: E402
from scripts.analyze_hedge_trackj import stats  # noqa: E402

TRIGGER_MONTHLY = HedgeConfig(
    lookback=3, enter_margin=0.10, slot=0.25, max_hedge=0.75, corr_window=12,
    corr_gate=0.0, duration_assets=("UST10",),
    regime_assets=("GOLD", "SILVER", "CMDTY", "ENERGY"), regime_min=2,
    regime_corr_asset="UST10", regime_corr_above=0.0)

# The live configuration, imported rather than restated, so what reports live
# is exactly what this study validated.
from momentum.trackk import LIVE_CONFIG as TRIGGER_DAILY  # noqa: E402
assert TRIGGER_DAILY == replace(
    TIER1, corr_window=252, regime_assets=("GOLD", "SILVER", "CMDTY", "ENERGY"),
    regime_min=2, regime_corr_asset="UST10", regime_corr_above=0.0)

TARGET = {"1946-48": ("1946-01", "1948-08"), "1973-74": ("1973-01", "1974-09"),
          "1977-81": ("1977-01", "1981-09"), "2021-22": ("2021-01", "2022-10")}
OTHER = {"1929-32": ("1929-09", "1932-06"), "1968-70": ("1968-12", "1970-06"),
         "1987": ("1987-09", "1987-11"), "2000-02": ("2000-09", "2002-09"),
         "2008 H1": ("2008-01", "2008-06"), "2008 H2": ("2008-07", "2008-12"),
         "2015-16": ("2015-06", "2016-02"), "2020": ("2020-02", "2020-08"),
         "2025-26": ("2025-01", "2026-07")}


def episode_rows(fire: pd.Series, eps: dict, kind: str) -> list:
    rows = []
    for name, (a, b) in eps.items():
        f = fire.loc[a:b]
        if f.empty:
            continue
        rows.append({"episode": name, "kind": kind, "periods": len(f), "% firing": f.mean(),
                     "first fired": f.index[f.values][0].strftime("%Y-%m-%d") if f.any() else "-"})
    return rows


def outside(fire: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=fire.index)
    for a, b in TARGET.values():
        mask.loc[a:b] = False
    return fire[mask]


def monthly():
    print("=" * 110)
    print("1. MONTHLY 1927-2026 — French market as the book")
    print("=" * 110)
    out = {}
    for label, backfill in (("as measured", False), ("CMDTY backfilled 1926-59 (PPI)", True)):
        p = load_long_history(verbose=False, cmdty_backfill=backfill)
        mkt = p["MARKET"].dropna()
        a = p[HEDGE_ASSETS].loc[mkt.index]
        fire = pd.Series(regime_open(mkt, a, TRIGGER_MONTHLY), index=mkt.index)
        out[label] = fire
        rows = episode_rows(fire, TARGET, "TARGET") + episode_rows(fire, OTHER, "other")
        t = pd.DataFrame(rows).set_index("episode")
        o = outside(fire)
        print(f"\n  [{label}]  firing {fire.mean():.1%} of all months; {o.mean():.1%} outside the target "
              f"episodes (criterion < ~15%)")
        print(t.assign(**{"% firing": t["% firing"].map("{:.0%}".format)}).to_string())
    fire = out["as measured"]
    dec = fire.groupby((fire.index.year // 10) * 10).mean()
    print("\n  % of months firing by decade (as measured):")
    print("  " + "  ".join(f"{d}s {v:.0%}" for d, v in dec.items()))
    return out


def daily():
    print("\n" + "=" * 110)
    print("2. DAILY 2006-2026 — SPY as the book, ETFs from the store")
    print("=" * 110)
    spy_book, assets, _, _ = load_etfs("spy")
    fire = pd.Series(regime_open(spy_book, assets, TRIGGER_DAILY), index=spy_book.index).loc["2006-07":]
    eps = {k: v for k, v in {**TARGET, **OTHER}.items() if v[0] >= "2006"}
    rows = episode_rows(fire, {k: eps[k] for k in eps if k in TARGET}, "TARGET") + \
        episode_rows(fire, {k: eps[k] for k in eps if k not in TARGET}, "other")
    t = pd.DataFrame(rows).set_index("episode")
    o = outside(fire)
    print(f"  firing {fire.mean():.1%} of days; {o.mean():.1%} outside 2021-22")
    print(t.assign(**{"% firing": t["% firing"].map("{:.0%}".format)}).to_string())
    by_year = fire.groupby(fire.index.year).mean()
    print("\n  % of days firing by year:")
    print("  " + "  ".join(f"{y} {v:.0%}" for y, v in by_year.items()))
    return fire


def on_trackj():
    print("\n" + "=" * 110)
    print("3. OBEYED ON TRACK J — signal on SPY, Track J scaled, decided on its rotation Tuesdays")
    print("=" * 110)
    tj, assets, store = load_trackj()
    for c in assets:
        f = assets[c].first_valid_index()
        assets.loc[f:, c] = assets.loc[f:, c].fillna(0.0)
    spy = store["SPY"].reindex(tj.index)
    rot = rotation_dates(tj.index, weekday=TUESDAY, every_weeks=2, anchor=pd.Timestamp("2026-09-15"))
    rows = []
    for label, cfg, dates in (("trigger, rotation Tuesdays", TRIGGER_DAILY, rot),
                              ("trigger, the other Tuesdays",
                               TRIGGER_DAILY,
                               rotation_dates(tj.index, weekday=TUESDAY, every_weeks=2,
                                              anchor=pd.Timestamp("2026-09-22"))),
                              ("no trigger (Tier 1 candidate on SPY)", TIER1, rot)):
        w = hedge_weights(spy, assets, cfg, decide_at=dates)
        res = apply_weights(tj, assets, w, cfg.cost_bps, cfg.exec_lag).loc["2012-01-01":]
        h, u = stats(res["net"], assets["CASH"]), stats(tj.reindex(res.index), assets["CASH"])
        held = w.shift(cfg.exec_lag).reindex(res.index).drop(columns="STOCKS")
        r2122 = res["net"].loc["2021-01":"2022-10"]
        rows.append({"variant": label, "dCAGR": h["CAGR"] - u["CAGR"], "dSharpe": h["Sharpe"] - u["Sharpe"],
                     "dMaxDD": h["MaxDD"] - u["MaxDD"], "% days hedged": (res["hedge_share"] > 0).mean(),
                     "mean hedge share": res["hedge_share"].mean(),
                     "2021-22 delta": (1 + r2122).prod() - (1 + tj.loc[r2122.index]).prod(),
                     "top assets": ", ".join(f"{k} {v:.1%}" for k, v in
                                             held.mean().sort_values(ascending=False).head(3).items())})
    t = pd.DataFrame(rows).set_index("variant")
    fmt = t.copy()
    for c in ("dCAGR", "dMaxDD", "% days hedged", "mean hedge share", "2021-22 delta"):
        fmt[c] = fmt[c].map("{:+.2%}".format)
    fmt["dSharpe"] = fmt["dSharpe"].map("{:+.2f}".format)
    print(fmt.to_string())
    return t


def main():
    pd.set_option("display.width", 220)
    m = monthly()
    d = daily()
    t = on_trackj()
    pd.DataFrame({k: v for k, v in m.items()}).to_csv(REPO_ROOT / "trackk_trigger_monthly.csv")
    d.to_frame("firing").to_csv(REPO_ROOT / "trackk_trigger_daily.csv")
    t.to_csv(REPO_ROOT / "trackk_trigger_trackj.csv")


if __name__ == "__main__":
    main()
