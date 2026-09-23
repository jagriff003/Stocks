"""
Track K, Tier 2: the hedge layer on the real instruments, daily, 2006-2026.

TODO 0l.  Tier 3 found the shape at index level, monthly, over a century: the
bare competition is a permanent allocation; the filter is an entry margin (the
hedge must beat the stock book decisively), not a stock-weakness gate; and
fixed harvest targets sell the fat tail.  This tier asks whether that survives
on the ETFs that would actually be traded, at the agreed weekly cadence, with
execution a session after the signal — and adds what monthly data could not
test: 2008's commodity spike and crash within six months (exit speed), the
2020 crash and rebound, and the 2025-26 metals run including silver's -28.5%
day (the trailing-stop harvest, agreed 2026-09-23).

STOCK BOOK

Ken French's daily top prior-return decile, as in Tier 3, to its last month
(2026-07).  Still NOT Track J; Tier 1 is.  SPY as the book is a sensitivity row.

INSTRUMENTS (from the daily store, `data/market/daily_returns.csv`)

  GOLD IAU  SILVER SLV  CMDTY DBC  ENERGY XLE  REALEST VNQ  TIPS TIP
  UST_S SHY  UST10 IEF  UST_L TLT  DOLLAR UUP  CASH BIL (French RF before 2007-05)
  Duration gate on IEF and TLT.  TIP is not gated: it is the inflation hedge,
  and the competition will drop it if its duration hurts.

CANDIDATE — Tier 3's entry-margin shape translated to sessions, fixed before
this run (2026-09-23): lookback 63, enter_margin +10%, slot 25%, cap 75%,
correlation gate 0 over 756 sessions, weekly decisions (every 5 sessions),
exec_lag 2 (trade at the NEXT session's close), 10bps per unit of turnover.

CHECKS (the script refuses to print results if any fails)

  A  French daily MARKET vs SPY from the store: daily correlation > 0.95 and
     the calendars agree; French RF vs BIL within 0.5%/yr
  B  zero cap reproduces the stock book exactly
  C  weights unchanged when the panel is truncated at the decision date
  D  portfolio return recomputed by hand on 25 random days

Run:  python scripts/analyze_hedge_etfs.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.hedge import HedgeConfig, apply_weights, hedge_weights, run_hedge  # noqa: E402
from momentum.longhistory import french_daily  # noqa: E402
from momentum.marketstore import load_daily_returns  # noqa: E402

PPY = 252
ROLES = {"GOLD": "IAU", "SILVER": "SLV", "CMDTY": "DBC", "ENERGY": "XLE", "REALEST": "VNQ",
         "TIPS": "TIP", "UST_S": "SHY", "UST10": "IEF", "UST_L": "TLT", "DOLLAR": "UUP",
         "CASH": "BIL"}
START = "2006-07-01"

CANDIDATE = HedgeConfig(lookback=63, enter_margin=0.10, slot=0.25, max_hedge=0.75,
                        corr_window=756, corr_gate=0.0, duration_assets=("UST10", "UST_L"),
                        decide_every=5, exec_lag=2, cost_bps=10.0)

EPISODES = [
    ("2008 H1 commodity spike", "2008-01-01", "2008-06-30"),
    ("2008 H2 crash", "2008-07-01", "2008-12-31"),
    ("GFC to the low", "2007-10-10", "2009-03-09"),
    ("2009 rebound", "2009-03-10", "2009-12-31"),
    ("2011 gold peak and after", "2011-07-01", "2011-12-31"),
    ("2013 taper tantrum", "2013-05-01", "2013-08-31"),
    ("2015-16 commodity crash", "2015-06-01", "2016-02-29"),
    ("2020 COVID crash", "2020-02-19", "2020-03-23"),
    ("2020 rebound", "2020-03-24", "2020-08-31"),
    ("2021-22 inflation", "2021-01-01", "2022-10-31"),
    ("2022 alone", "2022-01-01", "2022-12-31"),
    ("2025-26 metals run", "2025-01-01", "2026-07-31"),
]
PERIODS = [("full 2006-07..2026-07", START, None), ("2011-2026 (the 5pp bar)", "2011-01-01", None),
           ("calm 2011-19", "2011-01-01", "2019-12-31"), ("2006-2010", START, "2010-12-31")]


def load(book: str = "french"):
    ff = french_daily(verbose=False)
    store = load_daily_returns(list(ROLES.values()) + ["SPY"])
    cal = ff.index if book == "french" else store["SPY"].dropna().index
    cal = cal[cal >= "2002-07-31"]
    assets = pd.DataFrame({role: store[sym] for role, sym in ROLES.items()}).reindex(cal)
    # cash: BIL where it exists, the T-bill before
    rf = ff["CASH"].reindex(cal)
    assets["CASH"] = assets["CASH"].where(assets["CASH"].notna() & (cal >= "2007-06-01"), rf)
    stock = (ff["STOCKS"] if book == "french" else store["SPY"]).reindex(cal)
    # an ETF priced before its first stored return is not yet listed; after it, a
    # missing session is a real gap and must not silently become a zero
    for c in assets.columns:
        first = assets[c].first_valid_index()
        if first is not None:
            gaps = assets.loc[first:, c].isna()
            assert gaps.sum() <= 3, f"{c}: {int(gaps.sum())} sessions missing against the book's calendar"
            assets.loc[first:, c] = assets.loc[first:, c].fillna(0.0)
    return stock.dropna(), assets.loc[stock.dropna().index], ff, store


def stats(r, cash):
    r = r.dropna()
    c = cash.reindex(r.index).fillna(0.0)
    yrs = len(r) / PPY
    w = (1 + r).cumprod()
    ex = r - c
    return {"CAGR": w.iloc[-1] ** (1 / yrs) - 1, "Sharpe": ex.mean() / ex.std() * np.sqrt(PPY),
            "Vol": r.std() * np.sqrt(PPY), "MaxDD": (w / w.cummax() - 1).min()}


def checks(stock, assets, ff, store, cfg):
    common = ff.index.intersection(store["SPY"].dropna().index)
    common = common[common >= "2006-01-01"]
    corr = ff["MARKET"].loc[common].corr(store["SPY"].loc[common])
    only_ff = ff.index[(ff.index >= "2006-01-01")].difference(store["SPY"].dropna().index)
    only_spy = store["SPY"].dropna().index[(store["SPY"].dropna().index >= "2006-01-01")
                                          & (store["SPY"].dropna().index <= ff.index[-1])].difference(ff.index)
    b = store["BIL"].dropna()
    b = b[b.index >= "2008-01-01"]
    b = b[b.index <= ff.index[-1]]
    gap = ((1 + b).prod() ** (PPY / len(b)) - 1) - ((1 + ff["CASH"].loc[b.index]).prod() ** (PPY / len(b)) - 1)
    print(f"CHECK A  French MARKET vs SPY daily corr {corr:.4f} (> 0.95); calendars differ on "
          f"{len(only_ff)} / {len(only_spy)} days; BIL - RF {gap:+.2%}/yr (|gap| < 0.5%)")
    assert corr > 0.95 and len(only_ff) <= 3 and len(only_spy) <= 3 and abs(gap) < 0.005

    _, res = run_hedge(stock, assets, replace(cfg, max_hedge=0.0))
    g = (res["net"] - stock.reindex(res.index)).abs().max()
    assert g == 0.0, g
    print(f"CHECK B  zero cap reproduces the stock book: max gap {g:.1e}: OK")

    full = hedge_weights(stock, assets, cfg)
    for cut in ("2008-07-15", "2020-03-20", "2022-06-16", "2026-01-30"):
        t = full.index[full.index <= cut][-1]
        part = hedge_weights(stock.loc[:t], assets.loc[:t], cfg)
        pd.testing.assert_series_equal(full.loc[t], part.iloc[-1], check_names=False)
    print("CHECK C  weights identical when truncated at 2008-07, 2020-03, 2022-06, 2026-01-30: OK")

    w, res = run_hedge(stock, assets, cfg)
    rets = pd.concat([stock.rename("STOCKS"), assets], axis=1)
    rng = np.random.default_rng(0)
    worst = 0.0
    for t in rng.choice(res.index[5:], 25, replace=False):
        dec = w.index[w.index.get_loc(t) - cfg.exec_lag]
        gross = sum(w.at[dec, a] * rets.at[t, a] for a in w.columns if w.at[dec, a] > 0)
        worst = max(worst, abs(gross - res.at[t, "gross"]))
    assert worst < 1e-12, worst
    print(f"CHECK D  portfolio return recomputed by hand on 25 random days: max gap {worst:.1e}: OK\n")


def summarize(stock, assets, cfg, label):
    w, res = run_hedge(stock, assets, cfg)
    res = res.loc[START:]
    s = stock.reindex(res.index)
    cash = assets["CASH"]
    row = {"variant": label}
    for name, a, b in PERIODS:
        rr, ss = res["net"].loc[a:b], s.loc[a:b]
        h, u = stats(rr, cash), stats(ss, cash)
        if name.startswith("full"):
            row.update({"CAGR": h["CAGR"], "dCAGR": h["CAGR"] - u["CAGR"], "Sharpe": h["Sharpe"],
                        "dSharpe": h["Sharpe"] - u["Sharpe"], "MaxDD": h["MaxDD"],
                        "dMaxDD": h["MaxDD"] - u["MaxDD"]})
        elif name.startswith("2011-2026"):
            row["d2011-26"] = h["CAGR"] - u["CAGR"]
        elif name.startswith("calm"):
            row["dcalm11-19"] = h["CAGR"] - u["CAGR"]
    row["% hedged"] = (res["hedge_share"] > 0).mean()
    row["turnover/yr"] = res["turnover"].sum() / (len(res) / PPY)
    for name, a, b in EPISODES:
        rr = res["net"].loc[a:b]
        row[name] = (1 + rr).prod() - (1 + s.loc[rr.index]).prod()
    return row, w, res


def episode_detail(stock, assets, w, res):
    rows = []
    held = w.shift(CANDIDATE.exec_lag).reindex(res.index).drop(columns="STOCKS")
    for name, a, b in EPISODES:
        rr = res.loc[a:b]
        s = stock.loc[rr.index]
        h = held.loc[rr.index].mean()
        top = ", ".join(f"{k} {v:.0%}" for k, v in h[h > 0.02].sort_values(ascending=False).items())
        on = rr["hedge_share"] > 0
        rows.append({"episode": name, "stocks": (1 + s).prod() - 1, "hedged": (1 + rr["net"]).prod() - 1,
                     "delta": (1 + rr["net"]).prod() - (1 + s).prod(),
                     "hedge share": rr["hedge_share"].mean(),
                     "first hedged": rr.index[on.values][0].strftime("%Y-%m-%d") if on.any() else "-",
                     "held": top or "-"})
    return pd.DataFrame(rows).set_index("episode")


def fmt(df, keep=("Sharpe", "dSharpe")):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda x: (f"{x:+.2f}" if c in keep else f"{x:+.1%}") if pd.notna(x) else "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-phases", action="store_true", help="skip the all-phases cadence sweep")
    args = ap.parse_args()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 70)

    stock, assets, ff, store = load("french")
    print(f"Book: French top momentum decile, daily, {stock.index[0]:%Y-%m-%d} .. {stock.index[-1]:%Y-%m-%d}; "
          f"results from {START}.\nCandidate: {CANDIDATE}\n")
    checks(stock, assets, ff, store, CANDIDATE)

    variants = [("CANDIDATE L=63 enter+10% weekly", CANDIDATE),
                ("bare competition (no margin)", replace(CANDIDATE, enter_margin=0.0)),
                ("L=21 enter+5%", replace(CANDIDATE, lookback=21, enter_margin=0.05)),
                ("L=63 enter+5%", replace(CANDIDATE, enter_margin=0.05)),
                ("L=63 enter+15%", replace(CANDIDATE, enter_margin=0.15)),
                ("L=126 enter+10%", replace(CANDIDATE, lookback=126, enter_margin=0.10)),
                ("L=126 enter+20%", replace(CANDIDATE, lookback=126, enter_margin=0.20)),
                ("candidate, exec at signal close", replace(CANDIDATE, exec_lag=1)),
                ("candidate, no correlation gate", replace(CANDIDATE, corr_gate=None)),
                ("candidate, corr window 126", replace(CANDIDATE, corr_window=126)),
                ("candidate, corr window 252", replace(CANDIDATE, corr_window=252))]
    for tr in (0.10, 0.15, 0.20):
        for bo in (21, 63):
            variants.append((f"candidate + trail {tr:.0%}, blackout {bo}",
                             replace(CANDIDATE, harvest_trail=tr, harvest_blackout=bo)))
    # post-hoc (after this script's first run): fast hand-back to stocks
    for rl in (10, 21):
        for bo in (10, 21, 63):
            variants.append((f"candidate + hand back {rl}d, blackout {bo}",
                             replace(CANDIDATE, harvest_rel_lookback=rl, harvest_blackout=bo)))
    variants.append(("candidate + hand back 21d/bo 21 + trail 10%",
                     replace(CANDIDATE, harvest_rel_lookback=21, harvest_blackout=21, harvest_trail=0.10)))
    rows = []
    for label, cfg in variants:
        row, w, res = summarize(stock, assets, cfg, label)
        rows.append(row)
        if label.startswith("CANDIDATE"):
            cw, cres = w, res
    tab = pd.DataFrame(rows).set_index("variant")

    head = ["CAGR", "dCAGR", "Sharpe", "dSharpe", "MaxDD", "dMaxDD", "d2011-26", "dcalm11-19",
            "% hedged", "turnover/yr"]
    print("=" * 120)
    print(f"VARIANTS, {START} .. 2026-07 — d* are hedged minus the stock book alone")
    print("=" * 120)
    print(fmt(tab[head]).to_string())
    print("\nEPISODES — hedged minus unhedged cumulative return")
    print(fmt(tab[[e[0] for e in EPISODES]]).T.to_string())

    det = episode_detail(stock, assets, cw, cres)
    print("\nCANDIDATE, episode by episode")
    print(fmt(det, keep=()).to_string())

    use = cw.shift(CANDIDATE.exec_lag).reindex(cres.index).drop(columns="STOCKS")
    print("\nCANDIDATE asset usage: % of days held / average weight")
    print(pd.DataFrame({"% days held": (use > 0).mean(), "avg weight": use.mean()})
          .map(lambda x: f"{x:.1%}").T.to_string())

    if not args.no_phases:
        print("\nCADENCE — every phase of each decision interval (TODO 0l sweep, TODO 0i lesson)")
        prow = []
        for every in (1, 5, 10, 20):
            d = []
            for ph in range(every):
                row, _, _ = summarize(stock, assets, replace(CANDIDATE, decide_every=every, decide_phase=ph), "")
                d.append(row)
            d = pd.DataFrame(d)
            prow.append({"every N sessions": every, "phases": every,
                         "dCAGR mean": d["dCAGR"].mean(), "dCAGR min": d["dCAGR"].min(),
                         "dCAGR max": d["dCAGR"].max(), "dSharpe mean": d["dSharpe"].mean(),
                         "d2011-26 mean": d["d2011-26"].mean(),
                         "turnover/yr": d["turnover/yr"].mean(),
                         "2008 H2 mean": d["2008 H2 crash"].mean(),
                         "2021-22 mean": d["2021-22 inflation"].mean()})
        pt = pd.DataFrame(prow).set_index("every N sessions")
        print(fmt(pt, keep=("dSharpe mean",)).to_string())
        pt.to_csv(REPO_ROOT / "trackk_tier2_cadence.csv")

    s2, a2, _, _ = load("spy")
    row, _, _ = summarize(s2, a2, CANDIDATE, "CANDIDATE on SPY as the book")
    print("\nSENSITIVITY — SPY as the stock book (through the latest stored session)")
    print(fmt(pd.DataFrame([row]).set_index("variant")[head + ["2021-22 inflation", "2008 H2 crash"]]).to_string())

    tab.to_csv(REPO_ROOT / "trackk_tier2_variants.csv")
    det.to_csv(REPO_ROOT / "trackk_tier2_episodes.csv")


if __name__ == "__main__":
    main()
