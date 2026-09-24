"""
Track K, Tier 1: the hedge layer over Track J itself.

TODO 0l.  Tiers 3 and 2 found the shape on proxy stock books.  Tier 2 ended on
the finding that the book decides the verdict: over a momentum decile the layer
cost -1.4pp (-4.8pp over 2011-26), over SPY it added +1.3pp and +0.20 Sharpe.
Track J is the book it would actually sit on.

THE STOCK BOOK

`trackj_portfolio_performance.csv`, written by `run_live_trackj.py`: four
sleeves on the Tuesday calendar, per-name point-in-time costs, next-open
execution — the configuration being graduated, not a neighbour of it.  Check A
asserts it reproduces the metrics recorded in TODO 0k, so a stale or
differently-configured file cannot slip through.  Survivorship inflates its
level (TODO 0d); read deltas.

WHAT ELSE THIS ASKS

  How much of the hedge does Track J already do?  Its score is free to rotate
  into energy and materials, and on 2026-09-20 its combined book was 7/29
  energy.  The rolling beta of Track J's returns to XLE, DBC and IAU, next to
  SPY, says whether it leaned into real assets on its own in 2021-22.

THE CANDIDATE — Tier 2's, unchanged except the correlation window

Lookback 63, entry margin +10%, 25% slots, 75% cap, weekly, trades at the next
session's close, 10bps.  Correlation window 252 rather than 756: Track J's
stream starts 2011-10, so a three-year window could not gate anything before
late 2014; Tier 2 measured 252 as equivalent (-1.1pp against -1.4pp).
Results start 2012-01, when the first 63-session lookback exists.

CHECKS (the script refuses to print results if any fails)

  A  the Track J stream, over TODO 0k's window (2011-10-19 .. 2026-09-18),
     reproduces 20.91% CAGR, 0.73 Sharpe, -42.93% max drawdown (repo
     convention, fixed 4.5% risk-free).  TODO 0k recorded 20.81% / -42.99% at
     the larger account notional used before 2026-09-24; see FINDINGS.
  B  the store covers every Track J session for every instrument once listed
  C  zero cap reproduces Track J exactly
  D  weights unchanged when truncated at the decision date
  E  portfolio return recomputed by hand on 25 random days

Run:  python scripts/analyze_hedge_trackj.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.hedge import HedgeConfig, apply_weights, hedge_weights, run_hedge  # noqa: E402
from momentum.marketstore import load_daily_returns  # noqa: E402
from momentum.metrics import calculate_performance_metrics  # noqa: E402
from scripts.analyze_hedge_etfs import CANDIDATE as TIER2, ROLES, fmt, stats  # noqa: E402

PPY = 252
START = "2012-01-01"
CANDIDATE = replace(TIER2, corr_window=252)

EPISODES = [
    ("2015-16 commodity crash", "2015-06-01", "2016-02-29"),
    ("2018 Q4 selloff", "2018-10-01", "2018-12-24"),
    ("2019 recovery", "2018-12-26", "2019-12-31"),
    ("2020 COVID crash", "2020-02-19", "2020-03-23"),
    ("2020 rebound", "2020-03-24", "2020-08-31"),
    ("2021-22 inflation", "2021-01-01", "2022-10-31"),
    ("2022 alone", "2022-01-01", "2022-12-31"),
    ("2023-24", "2023-01-01", "2024-12-31"),
    ("2025-26 metals run", "2025-01-01", "2026-07-31"),
    ("2026 YTD", "2026-01-01", "2026-09-18"),
]
PERIODS = [("full 2012..2026-09", START, None), ("calm 2012-19", START, "2019-12-31"),
           ("2020-2026", "2020-01-01", None)]


def load():
    tj = pd.read_csv(REPO_ROOT / "trackj_portfolio_performance.csv", index_col=0,
                     parse_dates=True)["Portfolio_Return"].dropna()
    store = load_daily_returns(list(ROLES.values()) + ["SPY"])
    assets = pd.DataFrame({role: store[sym] for role, sym in ROLES.items()}).reindex(tj.index)
    return tj, assets, store


def checks(tj, assets, store, cfg):
    # a fixed window, so the check does not drift as the runner adds sessions
    ref = tj.loc[:"2026-09-18"]
    m = calculate_performance_metrics(ref, risk_free_rate=0.045)
    got = (m["cagr"], m["sharpe_ratio"], m["max_drawdown"])
    print(f"CHECK A  Track J stream {ref.index[0]:%Y-%m-%d}..{ref.index[-1]:%Y-%m-%d}: CAGR {got[0]:.2%}, "
          f"Sharpe {got[1]:.2f}, MaxDD {got[2]:.2%} vs 20.91% / 0.73 / -42.93%")
    assert abs(got[0] - 0.2091) < 1e-4 and abs(got[1] - 0.731) < 5e-3 and abs(got[2] + 0.4293) < 1e-4, got
    missing_spy = tj.index.difference(store["SPY"].dropna().index)
    worst = 0
    for c in assets.columns:
        first = assets[c].first_valid_index()
        holes = int(assets.loc[first:, c].isna().sum())
        worst = max(worst, holes)
        assets.loc[first:, c] = assets.loc[first:, c].fillna(0.0)
    print(f"CHECK B  Track J sessions missing from the store's calendar: {len(missing_spy)}; "
          f"worst instrument hole count: {worst}")
    assert len(missing_spy) == 0 and worst <= 3

    _, res = run_hedge(tj, assets, replace(cfg, max_hedge=0.0))
    g = (res["net"] - tj.reindex(res.index)).abs().max()
    assert g == 0.0, g
    print(f"CHECK C  zero cap reproduces Track J: max gap {g:.1e}: OK")

    full = hedge_weights(tj, assets, cfg)
    for cut in ("2020-03-20", "2022-06-16", "2026-01-30"):
        t = full.index[full.index <= cut][-1]
        part = hedge_weights(tj.loc[:t], assets.loc[:t], cfg)
        pd.testing.assert_series_equal(full.loc[t], part.iloc[-1], check_names=False)
    print("CHECK D  weights identical when truncated at 2020-03, 2022-06, 2026-01-30: OK")

    w, res = run_hedge(tj, assets, cfg)
    rets = pd.concat([tj.rename("STOCKS"), assets], axis=1)
    rng = np.random.default_rng(0)
    worst = 0.0
    for t in rng.choice(res.index[5:], 25, replace=False):
        dec = w.index[w.index.get_loc(t) - cfg.exec_lag]
        gross = sum(w.at[dec, a] * rets.at[t, a] for a in w.columns if w.at[dec, a] > 0)
        worst = max(worst, abs(gross - res.at[t, "gross"]))
    assert worst < 1e-12, worst
    print(f"CHECK E  portfolio return recomputed by hand on 25 random days: max gap {worst:.1e}: OK\n")
    return assets


def summarize(book, assets, cfg, label):
    w, res = run_hedge(book, assets, cfg)
    res = res.loc[START:]
    s = book.reindex(res.index)
    cash = assets["CASH"]
    row = {"variant": label}
    for name, a, b in PERIODS:
        h, u = stats(res["net"].loc[a:b], cash), stats(s.loc[a:b], cash)
        if name.startswith("full"):
            row.update({"CAGR": h["CAGR"], "book CAGR": u["CAGR"], "dCAGR": h["CAGR"] - u["CAGR"],
                        "Sharpe": h["Sharpe"], "dSharpe": h["Sharpe"] - u["Sharpe"],
                        "MaxDD": h["MaxDD"], "dMaxDD": h["MaxDD"] - u["MaxDD"]})
        else:
            row[f"d {name}"] = h["CAGR"] - u["CAGR"]
    row["% hedged"] = (res["hedge_share"] > 0).mean()
    row["turnover/yr"] = res["turnover"].sum() / (len(res) / PPY)
    for name, a, b in EPISODES:
        rr = res["net"].loc[a:b]
        row[name] = (1 + rr).prod() - (1 + s.loc[rr.index]).prod()
    return row, w, res


def real_asset_tilt(tj, store):
    """Rolling 126-session betas of Track J on SPY plus one real asset at a time."""
    out = {}
    for sym in ("XLE", "DBC", "IAU"):
        d = pd.concat([tj, store["SPY"], store[sym]], axis=1, keys=["tj", "spy", "x"], sort=True).dropna()
        betas = []
        for end in range(126, len(d) + 1, 21):
            chunk = d.iloc[end - 126:end]
            X = np.column_stack([np.ones(126), chunk["spy"], chunk["x"]])
            coef, *_ = np.linalg.lstsq(X, chunk["tj"].values, rcond=None)
            betas.append((chunk.index[-1], coef[2]))
        out[sym] = pd.Series(dict(betas))
    b = pd.DataFrame(out)
    return b.resample("YE").mean()


def mean_reversion(series: dict) -> pd.DataFrame:
    """
    Non-overlapping 63-session blocks: next-block return by quintile of the
    past block.  A falling profile means the book's weak stretches precede its
    strong ones — which a competition against its OWN trailing return mis-times.
    """
    rows = []
    for name, r in series.items():
        c = np.log1p(r.dropna()).cumsum()
        x = pd.concat([c - c.shift(63), c.shift(-63) - c], axis=1).dropna().iloc[::63]
        q = pd.qcut(x.iloc[:, 0], 5, labels=False)
        prof = np.expm1(x.groupby(q).mean().iloc[:, 1])
        rows.append({"book": name, "blocks": len(x), "corr(past, next)": x.iloc[:, 0].corr(x.iloc[:, 1]),
                     **{f"next | past Q{k + 1}": v for k, v in prof.items()}})
    return pd.DataFrame(rows).set_index("book")


def spy_signal(tj, assets, spy):
    """Judge the competition against the MARKET, scale Track J.  All 5 phases."""
    rows = []
    for label, cfg in (("candidate", CANDIDATE), ("enter +5%", replace(CANDIDATE, enter_margin=0.05)),
                       ("enter +15%", replace(CANDIDATE, enter_margin=0.15)),
                       ("L=126 enter +10%", replace(CANDIDATE, lookback=126))):
        d = []
        for ph in range(5):
            c = replace(cfg, decide_phase=ph)
            w = hedge_weights(spy, assets, c)
            res = apply_weights(tj, assets, w, c.cost_bps, c.exec_lag).loc[START:]
            h, u = stats(res["net"], assets["CASH"]), stats(tj.reindex(res.index), assets["CASH"])
            d.append({"dCAGR": h["CAGR"] - u["CAGR"], "dSharpe": h["Sharpe"] - u["Sharpe"],
                      "dMaxDD": h["MaxDD"] - u["MaxDD"], "% hedged": (res["hedge_share"] > 0).mean()})
        d = pd.DataFrame(d)
        rows.append({"variant": f"{label}, signal = SPY", "dCAGR mean": d["dCAGR"].mean(),
                     "dCAGR min": d["dCAGR"].min(), "dCAGR max": d["dCAGR"].max(),
                     "dSharpe": d["dSharpe"].mean(), "dMaxDD": d["dMaxDD"].mean(),
                     "% hedged": d["% hedged"].mean()})
    return pd.DataFrame(rows).set_index("variant")


def main():
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 70)
    tj, assets, store = load()
    print(f"Book: Track J, 4 sleeves, from trackj_portfolio_performance.csv. Results from {START}.\n"
          f"Candidate: {CANDIDATE}\n")
    assets = checks(tj, assets, store, CANDIDATE)

    variants = [("CANDIDATE (Tier 2, corr 252)", CANDIDATE),
                ("bare competition", replace(CANDIDATE, enter_margin=0.0)),
                ("enter +5%", replace(CANDIDATE, enter_margin=0.05)),
                ("enter +15%", replace(CANDIDATE, enter_margin=0.15)),
                ("enter +20%", replace(CANDIDATE, enter_margin=0.20)),
                ("L=126 enter +10%", replace(CANDIDATE, lookback=126)),
                ("L=126 enter +20%", replace(CANDIDATE, lookback=126, enter_margin=0.20)),
                ("cap 50%", replace(CANDIDATE, max_hedge=0.50)),
                ("hand back 10d, blackout 21", replace(CANDIDATE, harvest_rel_lookback=10,
                                                       harvest_blackout=21)),
                ("trail 10%, blackout 21", replace(CANDIDATE, harvest_trail=0.10, harvest_blackout=21))]
    rows = []
    for label, cfg in variants:
        row, w, res = summarize(tj, assets, cfg, label)
        rows.append(row)
        if label.startswith("CANDIDATE"):
            cw, cres = w, res
    phase = [summarize(tj, assets, replace(CANDIDATE, decide_phase=p), "")[0] for p in range(5)]
    tab = pd.DataFrame(rows).set_index("variant")

    head = ["CAGR", "book CAGR", "dCAGR", "Sharpe", "dSharpe", "MaxDD", "dMaxDD",
            "d calm 2012-19", "d 2020-2026", "% hedged", "turnover/yr"]
    print("=" * 120)
    print(f"TRACK J WITH THE HEDGE LAYER, {START} .. {tj.index[-1]:%Y-%m-%d}. d* = hedged minus Track J alone. "
          "Sharpe over BIL.")
    print("=" * 120)
    print(fmt(tab[head]).to_string())
    ph = pd.DataFrame(phase)
    print(f"\nCandidate across all 5 weekly phases: dCAGR mean {ph['dCAGR'].mean():+.2%} "
          f"(range {ph['dCAGR'].min():+.2%} .. {ph['dCAGR'].max():+.2%}), dSharpe mean "
          f"{ph['dSharpe'].mean():+.2f}")

    print("\nEPISODES — hedged minus Track J cumulative return")
    print(fmt(tab[[e[0] for e in EPISODES]]).T.to_string())

    held = cw.shift(CANDIDATE.exec_lag).reindex(cres.index).drop(columns="STOCKS")
    det = []
    for name, a, b in EPISODES:
        rr = cres.loc[a:b]
        s = tj.loc[rr.index]
        h = held.loc[rr.index].mean()
        det.append({"episode": name, "Track J": (1 + s).prod() - 1, "hedged": (1 + rr["net"]).prod() - 1,
                    "hedge share": rr["hedge_share"].mean(),
                    "held": ", ".join(f"{k} {v:.0%}" for k, v in
                                      h[h > 0.02].sort_values(ascending=False).items()) or "-"})
    det = pd.DataFrame(det).set_index("episode")
    print("\nCANDIDATE, episode by episode")
    print(fmt(det, keep=()).to_string())

    tilt = real_asset_tilt(tj, store)
    print("\nDOES TRACK J HEDGE ITSELF? Mean rolling 126-session beta to each real asset, "
          "controlling for SPY, by year")
    print(tilt.map(lambda x: f"{x:+.2f}").to_string())

    from scripts.analyze_hedge_etfs import load as load_tier2
    french = load_tier2("french")[0]
    spy = store["SPY"].reindex(tj.index)
    mr = mean_reversion({"Track J": tj.loc[START:], "French top decile": french.loc[START:],
                         "SPY": spy.loc[START:]})
    print("\nWHY THE COMPETITION MIS-TIMES TRACK J: next 63 sessions by quintile of the past 63")
    print(fmt(mr, keep=("corr(past, next)",)).to_string())

    sp = spy_signal(tj, assets, spy)
    print("\nPOST-HOC: judge the competition against SPY, scale Track J (mean over 5 weekly phases)")
    print(fmt(sp, keep=("dSharpe",)).to_string())

    out = REPO_ROOT
    mr.to_csv(out / "trackk_tier1_mean_reversion.csv")
    sp.to_csv(out / "trackk_tier1_spy_signal.csv")
    tab.to_csv(out / "trackk_tier1_variants.csv")
    det.to_csv(out / "trackk_tier1_episodes.csv")
    tilt.to_csv(out / "trackk_tier1_trackj_tilt.csv")


if __name__ == "__main__":
    main()
