"""
Track K trigger: is it late, and does it flicker?

TODO 0l.  The first live run (2026-09-23) showed the trigger firing in short
spells (1-21 sessions since 2021) and, in 2022, mostly AFTER energy peaked —
which is why obeying it cost 8pp over 2021-22 at one phase.  James's framing:
by the time everyone piles into a successful trade, the money has been made.

TWO QUESTIONS, MEASURED DIRECTLY

  LATE?     At each decision date the trigger holds real assets, what do those
            assets do over the NEXT three months against the market?  If the
            trigger buys tops, that forward excess is zero or negative.  Also,
            per inflation episode: how much of each asset's run had already
            happened when the trigger first fired.
  FLICKER?  On/off changes per year on the decision calendar.

VARIANTS, fixed before the run (2026-09-23)

  base          the live trigger (63 sessions / 3 months, +10%, 2 of 4, corr > 0)
  persist 2     open only after 2 consecutive firing decisions, close on 1 quiet
  persist 2/2   open after 2 firing, close only after 2 quiet
  earlier: L42  2-month lookback
  earlier: +5%  5% margin instead of 10%
  slower: L126 +15%   6-month lookback, 15% margin

Read on two calendars: monthly 1927-2026 (French market as the book), and
Track J's rotation Tuesdays 2012-2026 (SPY as the signal, Track J scaled,
both biweekly phases).

WHAT WOULD CHANGE THE LIVE CONFIG

A variant that cuts flips by at least a third without costing more on Track J
at either phase and without firing later in 1973-74 or 1977-81.  Otherwise the
live trigger stays as it is and these are recorded as characteristics.

Run:  python scripts/analyze_trigger_timing.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.hedge import apply_weights, hedge_weights, regime_open  # noqa: E402
from momentum.longhistory import HEDGE_ASSETS, load_long_history  # noqa: E402
from momentum.schedule import TUESDAY, rotation_dates  # noqa: E402
from momentum.trackk import LIVE_CONFIG  # noqa: E402
from scripts.analyze_hedge_trackj import load as load_trackj  # noqa: E402
from scripts.analyze_hedge_trackj import stats  # noqa: E402
from scripts.analyze_hedge_trigger import TRIGGER_MONTHLY  # noqa: E402

EPISODES = {"1973-74": ("1973-01", "1974-09"), "1977-81": ("1977-01", "1981-09"),
            "2021-22": ("2021-01", "2022-10")}


def variants(base, monthly: bool):
    lb = (lambda m, d: m if monthly else d)
    return {
        "base": base,
        "persist 2": replace(base, regime_persist=2),
        "persist 2/2": replace(base, regime_persist=2, regime_release=2),
        "earlier: L42": replace(base, lookback=lb(2, 42)),
        "earlier: +5%": replace(base, enter_margin=0.05),
        "slower: L126 +15%": replace(base, lookback=lb(6, 126), enter_margin=0.15),
    }


def forward_excess(w, assets, market, dates, horizon):
    """At each decision date holding a hedge: next-`horizon` return of that basket minus the market."""
    lr_a = np.log1p(assets.fillna(0.0))
    lr_m = np.log1p(market)
    out = []
    idx = market.index
    for t in dates:
        h = w.loc[t].drop("STOCKS")
        h = h[h > 0]
        if h.empty:
            continue
        i = idx.get_loc(t)
        if i + horizon >= len(idx):
            continue
        win = idx[i + 1:i + 1 + horizon]
        basket = float(np.expm1(lr_a.loc[win, h.index].sum()).mul(h / h.sum()).sum())
        mkt = float(np.expm1(lr_m.loc[win].sum()))
        out.append((t, basket - mkt))
    return pd.Series(dict(out), dtype=float)


def summarize_forward(fx):
    if fx.empty:
        return {"n": 0, "fwd mean": np.nan, "fwd median": np.nan, "fwd hit": np.nan}
    return {"n": len(fx), "fwd mean": fx.mean(), "fwd median": fx.median(), "fwd hit": (fx > 0).mean()}


def on_state(w, dates):
    return (w.loc[dates].drop(columns="STOCKS").sum(axis=1) > 0)


def monthly():
    p = load_long_history(verbose=False, cmdty_backfill=True)
    mkt = p["MARKET"].dropna()
    a = p[HEDGE_ASSETS].loc[mkt.index]
    rows, first = [], []
    for name, cfg in variants(TRIGGER_MONTHLY, monthly=True).items():
        w = hedge_weights(mkt, a, cfg)
        on = on_state(w, mkt.index)
        outside = on.copy()
        for x, y in EPISODES.values():
            outside.loc[x:y] = False
        fx = forward_excess(w, a, mkt, mkt.index, 3)
        yrs = len(on) / 12
        rows.append({"variant": name, "% months on": on.mean(), "% on outside episodes": outside.mean(),
                     "flips / yr": on.astype(int).diff().abs().sum() / yrs, **summarize_forward(fx)})
        for ep, (x, y) in EPISODES.items():
            o = on.loc[x:y]
            first.append({"variant": name, "episode": ep,
                          "first on": o.index[o.values][0].strftime("%Y-%m") if o.any() else "-",
                          "% of episode on": o.mean()})
    return pd.DataFrame(rows).set_index("variant"), pd.DataFrame(first)


def lateness_by_episode():
    """How much of each trigger asset's in-episode run had happened at the base trigger's first fire."""
    p = load_long_history(verbose=False, cmdty_backfill=True)
    mkt = p["MARKET"].dropna()
    a = p[HEDGE_ASSETS].loc[mkt.index]
    fire = pd.Series(regime_open(mkt, a, TRIGGER_MONTHLY), index=mkt.index)
    rows = []
    for ep, (x, y) in EPISODES.items():
        f = fire.loc[x:y]
        if not f.any():
            continue
        t0 = f.index[f.values][0]
        for asset in TRIGGER_MONTHLY.regime_assets:
            r = a[asset].loc[pd.Timestamp(x) - pd.DateOffset(months=12):y].dropna()
            if r.empty:
                continue
            wv = (1 + r).cumprod()
            trough_t = wv.loc[:wv.idxmax()].idxmin()
            peak_t = wv.idxmax()
            run = wv.loc[peak_t] / wv.loc[trough_t] - 1
            done = (wv.loc[t0] / wv.loc[trough_t] - 1) / run if trough_t <= t0 else 0.0
            after = wv.loc[y:].iloc[0] / wv.loc[t0] - 1
            m = (1 + mkt.loc[t0:y].iloc[1:]).prod() - 1
            rows.append({"episode": ep, "first fire": t0.strftime("%Y-%m"), "asset": asset,
                         "trough": trough_t.strftime("%Y-%m"), "peak": peak_t.strftime("%Y-%m"),
                         "run": run, "done at first fire": min(done, 9.99),
                         "fire -> episode end": after, "market same span": m})
    return pd.DataFrame(rows).set_index(["episode", "asset"])


def trackj():
    tj, assets, store = load_trackj()
    for c in assets:
        f = assets[c].first_valid_index()
        assets.loc[f:, c] = assets.loc[f:, c].fillna(0.0)
    spy = store["SPY"].reindex(tj.index)
    rows = []
    for phase, anchor in (("A", "2026-09-15"), ("B", "2026-09-22")):
        rot = rotation_dates(tj.index, weekday=TUESDAY, every_weeks=2, anchor=pd.Timestamp(anchor))
        rot = [d for d in rot if d >= pd.Timestamp("2012-01-01")]
        for name, cfg in variants(LIVE_CONFIG, monthly=False).items():
            w = hedge_weights(spy, assets, cfg, decide_at=rot)
            res = apply_weights(tj, assets, w, cfg.cost_bps, cfg.exec_lag).loc["2012-01-01":]
            h, u = stats(res["net"], assets["CASH"]), stats(tj.reindex(res.index), assets["CASH"])
            on = on_state(w, rot)
            fx = forward_excess(w, assets, spy, rot, 63)
            r2122 = res["net"].loc["2021-01":"2022-10"]
            rows.append({"variant": name, "phase": phase, "dCAGR": h["CAGR"] - u["CAGR"],
                         "dSharpe": h["Sharpe"] - u["Sharpe"], "% rotations on": on.mean(),
                         "flips / yr": on.astype(int).diff().abs().sum() / (len(rot) / 26),
                         "2021-22 delta": (1 + r2122).prod() - (1 + tj.loc[r2122.index]).prod(),
                         **summarize_forward(fx)})
    return pd.DataFrame(rows).set_index(["variant", "phase"])


def pct(df, cols, digits=1):
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(lambda v: f"{v:+.{digits}%}" if pd.notna(v) else "")
    return out


def main():
    pd.set_option("display.width", 230)
    print("=" * 110)
    print("1. LATE? Each trigger asset's run in the inflation episodes, at the base trigger's first fire (monthly)")
    print("=" * 110)
    late = lateness_by_episode()
    print(pct(late, ["run", "done at first fire", "fire -> episode end", "market same span"], 0).to_string())

    print("\n" + "=" * 110)
    print("2. MONTHLY 1927-2026, French market as the book. fwd = held basket minus market over the next 3 months")
    print("=" * 110)
    m, first = monthly()
    print(pct(m, ["% months on", "% on outside episodes", "fwd mean", "fwd median", "fwd hit"]).assign(
        **{"flips / yr": m["flips / yr"].round(2)}).to_string())
    fp = first.pivot(index="variant", columns="episode", values="first on")
    print("\n  first month on, by episode:")
    print("  " + fp.to_string().replace("\n", "\n  "))

    print("\n" + "=" * 110)
    print("3. TRACK J'S ROTATION TUESDAYS 2012-2026, SPY as the signal, Track J scaled. fwd = held basket minus SPY, next 63 sessions")
    print("=" * 110)
    t = trackj()
    print(pct(t, ["dCAGR", "% rotations on", "2021-22 delta", "fwd mean", "fwd median", "fwd hit"]).assign(
        dSharpe=t["dSharpe"].map("{:+.2f}".format), **{"flips / yr": t["flips / yr"].round(2)}).to_string())

    late.to_csv(REPO_ROOT / "trackk_timing_lateness.csv")
    m.to_csv(REPO_ROOT / "trackk_timing_monthly.csv")
    t.to_csv(REPO_ROOT / "trackk_timing_trackj.csv")


if __name__ == "__main__":
    main()
