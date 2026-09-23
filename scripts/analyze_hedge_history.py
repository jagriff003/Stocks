"""
Track K, Tier 3: does the hedge layer protect a momentum stock book in the
inflation and crash episodes of 1927-2026, and what does it cost in between?

TODO 0l.  The price panel starts in 2010 and holds one inflation episode.  This
runs the layer at index level, monthly, over a century: Ken French's top
prior-return decile as the stock book, against gold, silver, commodities,
energy equities, REITs, 10-year Treasuries and cash (`momentum/longhistory.py`
documents each series and its limits).

WHAT THIS CAN AND CANNOT SAY

It can say whether the competition fires in the right episodes, which assets it
reaches for, and roughly what the insurance costs in calm decades.  It cannot
validate stock selection — the stock book here is 10% of the market, not eight
names — so its CAGR levels are not Track J's.  Deltas between hedged and
unhedged on the same book are the quantity to read.

PRE-REGISTERED DEFAULTS (fixed before the first run, 2026-09-23)

    lookback 3 months, slot 25%, cap 75%, no entry/exit margin, stock-bond
    correlation gate at 0 over 36 months, cash eligible, 10bps per unit of
    one-way turnover.  The sweep is a shape check around these, not a search.

CHECKS (the script refuses to print results if any fails)

  A  every series against its traded instrument: measured series correlate
     >0.95 with a gap under 1%/yr; pre-ETF constructions leak no more than the
     ETF's own autocorrelation (+/-0.10)
  B  zero cap reproduces the stock book exactly
  C  weights unchanged when the panel is truncated at the decision date
  D  portfolio return recomputed by hand on 25 random months

Run:  python scripts/analyze_hedge_history.py
      python scripts/analyze_hedge_history.py --no-sweep
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

from momentum.hedge import HedgeConfig, hedge_weights, run_hedge  # noqa: E402
from momentum.longhistory import HEDGE_ASSETS, load_long_history, validate_constructions  # noqa: E402

PPY = 12

EPISODES = [
    # name, start, end, kind
    ("Great Depression", "1929-09", "1932-06", "deflationary crash"),
    ("1937-38 recession", "1937-03", "1938-03", "deflationary crash"),
    ("Postwar inflation, yield peg", "1946-01", "1948-08", "INFLATION"),
    ("1968-70 bear", "1968-12", "1970-06", "inflation + bear"),
    ("1973-74 stagflation", "1973-01", "1974-09", "INFLATION"),
    ("1977-81 inflation surge", "1977-01", "1981-09", "INFLATION"),
    ("1987 crash", "1987-09", "1987-11", "crash"),
    ("2000-02 tech bust", "2000-09", "2002-09", "deflationary crash"),
    ("2008 GFC", "2007-11", "2009-02", "deflationary crash"),
    ("2009 momentum crash", "2009-03", "2009-05", "out of scope"),
    ("2015-16 commodity crash", "2015-06", "2016-02", "false-positive check"),
    ("2020 COVID", "2020-02", "2020-03", "crash"),
    ("2021-22 inflation", "2021-01", "2022-10", "INFLATION"),
]

PERIODS = [
    ("full 1927-2026", "1927-01", None),
    ("1927-1959", "1927-01", "1959-12"),
    ("1960-1989", "1960-01", "1989-12"),
    ("1990-2010", "1990-01", "2010-12"),
    ("2011-2026 (the 5pp bar)", "2011-01", None),
    ("calm 1950-65", "1950-01", "1965-12"),
    ("calm 1982-99", "1982-10", "1999-12"),
    ("calm 2011-19", "2011-01", "2019-12"),
]


def stats(r: pd.Series, cash: pd.Series, cpi: pd.Series) -> dict:
    r = r.dropna()
    c, p = cash.reindex(r.index), cpi.reindex(r.index)
    yrs = len(r) / PPY
    wealth = (1 + r).cumprod()
    ex = r - c
    return {
        "CAGR": wealth.iloc[-1] ** (1 / yrs) - 1,
        "real CAGR": ((1 + r) / (1 + p)).prod() ** (1 / yrs) - 1,
        "Sharpe": ex.mean() / ex.std() * np.sqrt(PPY),
        "Vol": r.std() * np.sqrt(PPY),
        "MaxDD": (wealth / wealth.cummax() - 1).min(),
    }


def check_constructions():
    v = validate_constructions(verbose=False)
    m = v.xs("measured")
    c = v.xs("construction")
    ok_m = (m["corr"] > 0.95) & (m["gap_ann"].abs() < 0.01)
    ok_c = (c["leak"] - c["ac1_etf"]).abs() < 0.10
    print("CHECK A  series against their traded instruments")
    print(v[["vs", "months", "corr", "leak", "ac1_etf", "gap_ann"]].round(3).to_string())
    assert ok_m.all(), f"measured series off: {m[~ok_m].index.tolist()}"
    assert ok_c.all(), f"construction leaks: {c[~ok_c].index.tolist()}"
    print("  measured: corr > 0.95, |gap| < 1%/yr: OK   constructions: |leak - ETF ac1| < 0.10: OK")
    print("  (the 'naive average' rows are shown for contrast and are expected to leak)\n")


def run_checks(stock, assets, cfg):
    w, res = run_hedge(stock, assets, replace(cfg, max_hedge=0.0))
    gap = (res["net"] - stock.reindex(res.index)).abs().max()
    assert gap == 0.0, gap
    print(f"CHECK B  zero cap reproduces the stock book: max gap {gap:.1e} (tol 0): OK")

    full = hedge_weights(stock, assets, cfg)
    for cut in ("1947-06-30", "1974-09-30", "2008-10-31", "2022-06-30"):
        part = hedge_weights(stock.loc[:cut], assets.loc[:cut], cfg)
        pd.testing.assert_series_equal(full.loc[cut], part.iloc[-1], check_names=False)
    print("CHECK C  weights identical when truncated at 1947-06, 1974-09, 2008-10, 2022-06: OK")

    w, res = run_hedge(stock, assets, cfg)
    rets = pd.concat([stock.rename("STOCKS"), assets], axis=1)
    rng = np.random.default_rng(0)
    worst = 0.0
    for t in rng.choice(res.index[1:], 25, replace=False):
        prev = w.index[w.index.get_loc(t) - 1]
        wt = w.loc[prev]
        gross = sum(wt[a] * rets.at[t, a] for a in wt.index if wt[a] > 0)
        worst = max(worst, abs(gross - res.at[t, "gross"]))
    assert worst < 1e-12, worst
    print(f"CHECK D  portfolio return recomputed by hand on 25 random months: max gap {worst:.1e} (tol 1e-12): OK\n")


def episode_table(res, stock, w, panel):
    rows = []
    for name, a, b, kind in EPISODES:
        r = res.loc[a:b]
        s = stock.loc[r.index]
        cpi = panel["CPI"].loc[r.index]
        infl = (1 + cpi).prod() ** (PPY / len(r)) - 1
        hedged = (1 + r["net"]).prod() - 1
        base = (1 + s).prod() - 1
        # which assets carried the hedge: mean weight over the DECISION dates
        dec = w.shift(1).loc[r.index].drop(columns="STOCKS").mean()
        top = ", ".join(f"{k} {v:.0%}" for k, v in dec[dec > 0.02].sort_values(ascending=False).items())
        rows.append({"episode": name, "kind": kind, "months": len(r),
                     "CPI ann": infl, "stocks": base, "hedged": hedged,
                     "delta": hedged - base,
                     "stocks real": (1 + base) / (1 + cpi).prod() - 1,
                     "hedged real": (1 + hedged) / (1 + cpi).prod() - 1,
                     "hedge share": r["hedge_share"].mean(), "held": top or "-"})
    return pd.DataFrame(rows).set_index("episode")


def period_table(res, stock, panel):
    rows = []
    for name, a, b in PERIODS:
        r = res.loc[a:b]
        s = stock.loc[r.index]
        mk = panel["MARKET"].loc[r.index]
        h, u, m = (stats(x, panel["CASH"], panel["CPI"]) for x in (r["net"], s, mk))
        rows.append({"period": name,
                     "stocks CAGR": u["CAGR"], "hedged CAGR": h["CAGR"],
                     "delta CAGR": h["CAGR"] - u["CAGR"],
                     "stocks Sharpe": u["Sharpe"], "hedged Sharpe": h["Sharpe"],
                     "stocks MaxDD": u["MaxDD"], "hedged MaxDD": h["MaxDD"],
                     "hedged real CAGR": h["real CAGR"], "market CAGR": m["CAGR"],
                     "hedge share": r["hedge_share"].mean(),
                     "% months hedged": (r["hedge_share"] > 0).mean(),
                     "turnover/yr": r["turnover"].sum() / (len(r) / PPY)})
    return pd.DataFrame(rows).set_index("period")


def asset_usage(res, w, assets):
    held = w.shift(1).loc[res.index].drop(columns="STOCKS")
    contrib = (held * assets.reindex(res.index)).sum() / (len(res) / PPY)
    return pd.DataFrame({"% months held": (held > 0).mean(),
                         "avg weight": held.mean(),
                         "contribution / yr": contrib})


KEY_EPISODES = {"1946-48": ("1946-01", "1948-08"), "1973-74": ("1973-01", "1974-09"),
                "1977-81": ("1977-01", "1981-09"), "2000-02": ("2000-09", "2002-09"),
                "2008": ("2007-11", "2009-02"), "2021-22": ("2021-01", "2022-10")}


def variants(base: HedgeConfig):
    """
    Blocks, in the order they were conceived.  A is the pre-registered default
    and its shape check.  B, C and D came AFTER seeing A and are post-hoc: B
    from the 74%-of-months finding, C from James's harvest hypothesis
    (2026-09-23), D to give 1946-48 a commodity candidate at all.
    """
    out = []
    for lb in (1, 3, 6, 12):
        for gate in (0.0, None):
            for cash in ("CASH", None):
                out.append(("A pre-registered", f"L={lb} gate={'on' if gate is not None else 'off'} "
                            f"cash={'on' if cash else 'off'}",
                            replace(base, lookback=lb, corr_gate=gate, cash_asset=cash), False))
    for em in (0.01, 0.03):
        out.append(("A pre-registered", f"L=3 exit_margin {em:.0%}", replace(base, exit_margin=em), False))
    out.append(("A pre-registered", "L=3 fast exit 1m", replace(base, fast_exit_lookback=1), False))

    for lb in (1, 3):
        for dl in (3, 6, 12):
            out.append(("B danger gate", f"L={lb} danger {dl}m",
                        replace(base, lookback=lb, danger_lookback=dl), False))

    for name, cfg in (("L=3", base), ("L=1", replace(base, lookback=1)),
                      ("L=1 danger 6m", replace(base, lookback=1, danger_lookback=6)),
                      ("L=3 danger 6m", replace(base, danger_lookback=6))):
        for hg in (0.25, 0.50):
            out.append(("C harvest", f"{name} harvest +{hg:.0%}",
                        replace(cfg, harvest_gain=hg), False))
        out.append(("C harvest", f"{name} harvest z>=2",
                    replace(cfg, harvest_z=2.0), False))

    # E: the danger gate enters late because inflation assets start running
    # while stocks are still rising (2021).  Keep entry keyed to hedge strength,
    # but demand DECISIVE strength over the stock book.
    for lb, margins in ((1, (0.02, 0.05)), (3, (0.05, 0.10)), (6, (0.10, 0.20))):
        for m in margins:
            out.append(("E entry margin", f"L={lb} enter +{m:.0%}",
                        replace(base, lookback=lb, enter_margin=m), False))

    for name, cfg in (("L=3 default", base), ("L=1", replace(base, lookback=1)),
                      ("L=1 danger 6m", replace(base, lookback=1, danger_lookback=6)),
                      ("L=1 danger 6m harvest +50%", replace(base, lookback=1, danger_lookback=6,
                                                             harvest_gain=0.50))):
        out.append(("D CMDTY backfill 1926-59", name, cfg, True))
    return out


def sweep(stock, assets, assets_bf, base, panel):
    rows = []
    for block, name, cfg, backfill in variants(base):
        a = assets_bf if backfill else assets
        if cfg.cash_asset is None:
            cfg = replace(cfg, require_beats_cash=False)
            a = a.drop(columns="CASH")
        _, res = run_hedge(stock, a, cfg)
        s = stock.reindex(res.index)
        h = stats(res["net"], panel["CASH"], panel["CPI"])
        u = stats(s, panel["CASH"], panel["CPI"])

        def d(x, y):
            return (stats(res["net"].loc[x:y], panel["CASH"], panel["CPI"])["CAGR"]
                    - stats(s.loc[x:y], panel["CASH"], panel["CPI"])["CAGR"])

        row = {"block": block, "variant": name,
               "CAGR": h["CAGR"], "dCAGR": h["CAGR"] - u["CAGR"], "Sharpe": h["Sharpe"],
               "dSharpe": h["Sharpe"] - u["Sharpe"], "MaxDD": h["MaxDD"],
               "d2011-26": d("2011-01", None), "dcalm11-19": d("2011-01", "2019-12"),
               "% hedged": (res["hedge_share"] > 0).mean(),
               "turnover/yr": res["turnover"].sum() / (len(res) / PPY)}
        for k, (x, y) in KEY_EPISODES.items():
            rr = res.loc[x:y]
            row[k] = (1 + rr["net"]).prod() - (1 + s.loc[rr.index]).prod()
        rows.append(row)
    return pd.DataFrame(rows).set_index(["block", "variant"])


def best_run(x: pd.Series) -> float:
    """Largest peak-over-prior-trough gain: the most any entry/exit timing could catch."""
    w = np.r_[1.0, (1 + x.dropna()).cumprod().values]
    return (w / np.minimum.accumulate(w)).max() - 1


def hypothesis_table(panel, assets_bf):
    """
    James's hypothesis, measured directly and without any rule: in each stress
    episode, how far did each asset run, against how far the stock book fell?
    Cell = buy-and-hold over the episode / best run inside it.
    """
    rows = []
    for name, a, b, kind in EPISODES:
        r = pd.concat([panel["STOCKS"], assets_bf.drop(columns="CASH")], axis=1).loc[a:b]
        row = {"episode": name}
        for c in r.columns:
            x = r[c].dropna()
            if len(x) < len(r) * 0.9:
                row[c] = ""
                continue
            row[c] = f"{(1 + x).prod() - 1:+.0%} / {best_run(x):+.0%}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("episode")


def capture_table(stock, assets, cfg, label):
    """
    For each key episode: what the hedge slots realised per unit of capital
    deployed, against the best run any single hedge asset offered.  Low means
    it arrived late or left early; high means it caught the move.
    """
    w, res = run_hedge(stock, assets, cfg)
    held = w.shift(1).reindex(res.index).drop(columns="STOCKS")
    rows = []
    for name, (a, b) in KEY_EPISODES.items():
        h = held.loc[a:b]
        r = assets.loc[h.index]
        share = h.sum(axis=1)
        on = share > 0
        slot_ret = (h * r.fillna(0.0)).sum(axis=1)
        per_unit = (1 + slot_ret[on] / share[on]).prod() - 1 if on.any() else 0.0
        best = {c: best_run(r[c]) for c in r.columns if c != "CASH" and r[c].notna().any()}
        top = max(best, key=best.get)
        rows.append({"config": label, "episode": name,
                     "first hedged": h.index[on.values][0].strftime("%Y-%m") if on.any() else "-",
                     "months hedged": f"{int(on.sum())}/{len(h)}",
                     "realised per unit hedged": per_unit,
                     "best single run": f"{top} {best[top]:+.0%}"})
    return pd.DataFrame(rows).set_index(["config", "episode"])


def pct(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--lookback", type=int, default=3)
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)

    panel = load_long_history(verbose=True)
    stock = panel["STOCKS"].dropna()
    assets = panel[HEDGE_ASSETS].loc[stock.index]
    assets_bf = load_long_history(verbose=False, cmdty_backfill=True)[HEDGE_ASSETS].loc[stock.index]
    cfg = HedgeConfig(lookback=args.lookback)
    print(f"\nPanel {stock.index[0]:%Y-%m} .. {stock.index[-1]:%Y-%m}, {len(stock)} months. "
          f"Config: {cfg}\n")

    check_constructions()
    run_checks(stock, assets, cfg)

    w, res = run_hedge(stock, assets, cfg)

    per = period_table(res, stock, panel)
    print("=" * 110)
    print("BY PERIOD — stock book (French top momentum decile) alone vs with the hedge layer. "
          "Sharpe is over the actual T-bill.")
    print("=" * 110)
    print(pct(per, [c for c in per.columns if c not in ("stocks Sharpe", "hedged Sharpe")])
          .assign(**{"stocks Sharpe": per["stocks Sharpe"].round(2),
                     "hedged Sharpe": per["hedged Sharpe"].round(2)}).T.to_string())

    ep = episode_table(res, stock, w, panel)
    print("\n" + "=" * 110)
    print("EPISODES — cumulative return over the episode")
    print("=" * 110)
    print(pct(ep, ["CPI ann", "stocks", "hedged", "delta", "stocks real", "hedged real",
                   "hedge share"]).to_string())

    use = asset_usage(res, w, assets)
    print("\nASSET USAGE, full period")
    print(pct(use, use.columns).to_string())

    out = REPO_ROOT
    per.to_csv(out / "trackk_hedge_history_periods.csv")
    ep.to_csv(out / "trackk_hedge_history_episodes.csv")
    w.to_csv(out / "trackk_hedge_history_weights.csv")

    hyp = hypothesis_table(panel.loc[stock.index], assets_bf)
    print("\n" + "=" * 110)
    print("THE HYPOTHESIS, NO RULE APPLIED — per episode: buy-and-hold / best run inside the episode. "
          "CMDTY before 1960 is the PPI backfill.")
    print("=" * 110)
    print(hyp.to_string())
    hyp.to_csv(out / "trackk_hedge_history_hypothesis.csv")

    if not args.no_sweep:
        sw = sweep(stock, assets, assets_bf, cfg, panel)
        print("\n" + "=" * 110)
        print("SWEEP — blocks B-D are post-hoc. Episode columns: hedged minus unhedged cumulative return; "
              "d* columns are CAGR deltas.")
        print("=" * 110)
        fmt = sw.copy()
        for c in fmt.columns:
            fmt[c] = (fmt[c].map(lambda x: f"{x:+.2f}") if c in ("Sharpe", "dSharpe")
                      else fmt[c].map(lambda x: f"{x:+.1%}"))
        print(fmt.to_string())
        sw.to_csv(out / "trackk_hedge_history_sweep.csv")

        cap = pd.concat([
            capture_table(stock, assets, cfg, "L=3 default"),
            capture_table(stock, assets, replace(cfg, lookback=1, danger_lookback=6), "L=1 danger 6m"),
            capture_table(stock, assets, replace(cfg, lookback=1, danger_lookback=6, harvest_gain=0.5),
                          "L=1 danger 6m harvest +50%"),
        ])
        print("\nCAPTURE — what the hedge slots realised per unit deployed, against the best run on offer")
        print(pct(cap, ["realised per unit hedged"]).to_string())
        cap.to_csv(out / "trackk_hedge_history_capture.csv")


if __name__ == "__main__":
    main()
