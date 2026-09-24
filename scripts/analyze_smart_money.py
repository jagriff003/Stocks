"""
TODO 0m, first pass: does "smart money" positioning predict Track J's next eight weeks?

James's hypothesis (2026-09-24): when informed money leaves stocks, the stock
book is getting lonely, and Track K should know.  The bar he set is the one
that matters for trading: a signal earns its place only if it predicts TRACK
J's forward return — the thing actually held — and adds to what Track J's own
trailing quarter already says (Track J mean-reverts at that horizon, FINDINGS
Tier 1).  The commodity side asks the Track K question directly: does
speculator crowding predict an asset's return MINUS Track J's, i.e. whether
moving into it would have paid?

PRE-REGISTERED (fixed 2026-09-24, before any result was seen)

  decision dates   every Tuesday, 2011-10 .. 2026-09 (Track J's history)
  point-in-time    CFTC used only 7+ days after its position date (i.e. after
                   the Friday release); DIX from the next session.  Insider
                   data deferred to a second pass: Form 4s are timely (filed
                   within two business days), but only the SEC's quarterly
                   bulk files are free to backfill, and a live signal needs
                   EDGAR's daily feed.  13F holdings are excluded outright:
                   45 days late, quarterly.
  signals (8)      S&P 500 futures, net % of open interest:
                     asset managers   13-week change ("exiting"), 3-year z ("crowded")
                     leveraged funds  13-week change, 3-year z
                   DIX: 20-session mean, z vs its trailing year
                   managed-money crowding (3-year z): crude -> DBC, gold -> IAU,
                   silver -> SLV, each against (asset - Track J) forward return
  primary horizon  42 sessions (one sleeve's eight-week hold); 21 and 63 shown
                   for shape only
  model            forward return ~ signal (standardized) + trailing 63-session
                   return (Track J's, or asset minus Track J's), OLS with
                   Newey-West standard errors (9 weekly lags, the overlap)
  PASS             |t| >= 2, same sign in both halves (2011-18, 2019-26)
  LEAD             1.5 <= |t| < 2, same sign in both halves
  otherwise        NULL
  Only a PASS goes on to the Track K timing test.  With 8 signals, about one
  will reach |t| >= 1.5 by chance; the output says so.

CHECKS (the script refuses to print results if any fails)

  A  every CFTC value used is >= 7 days old on its decision date; every DIX
     value >= 1 day old
  B  forward and trailing returns recomputed by hand on random dates
  C  the Newey-West t for one regression recomputed from first principles

Run:  python scripts/analyze_smart_money.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings("ignore")

import statsmodels.api as sm  # noqa: E402

from momentum import smartmoney as smm  # noqa: E402
from momentum.marketstore import load_daily_returns  # noqa: E402

H, HORIZONS, CTRL = 42, (21, 42, 63), 63
NW_LAGS = 9
HALVES = (("2011-10-01", "2018-12-31"), ("2019-01-01", "2026-12-31"))
EQUITY = ["asset_mgr_net_flow", "asset_mgr_net_crowd", "lev_money_net_flow",
          "lev_money_net_crowd", "dix_z"]
COMMOD = {"CRUDE_crowd": "DBC", "GOLD_crowd": "IAU", "SILVER_crowd": "SLV"}


def compound(r: pd.Series, fwd: int) -> pd.Series:
    """Forward (fwd>0): r[t+1..t+fwd].  Trailing (fwd<0): r[t-|fwd|+1..t]."""
    lr = np.log1p(r)
    if fwd > 0:
        return np.expm1(lr[::-1].rolling(fwd).sum()[::-1].shift(-1))
    return np.expm1(lr.rolling(-fwd).sum())


def fit(y, x, ctrl):
    d = pd.concat([y.rename("y"), x.rename("x"), ctrl.rename("c")], axis=1).dropna()
    d["x"] = (d["x"] - d["x"].mean()) / d["x"].std()
    X = sm.add_constant(d[["x", "c"]])
    r = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return r, d


def verdict(t_full, t1, t2):
    same = np.sign(t1) == np.sign(t2) == np.sign(t_full)
    if abs(t_full) >= 2 and same:
        return "PASS"
    if abs(t_full) >= 1.5 and same:
        return "LEAD"
    return "NULL"


def main():
    pd.set_option("display.width", 220)
    tj = pd.read_csv(REPO_ROOT / "trackj_portfolio_performance.csv", index_col=0,
                     parse_dates=True)["Portfolio_Return"].dropna()
    store = load_daily_returns(list(COMMOD.values()))
    assets = store.reindex(tj.index).fillna(0.0)

    dates = tj.index[(tj.index.weekday == 1)]
    fwd = {h: compound(tj, h) for h in HORIZONS}
    trail = compound(tj, -CTRL)

    pos = smm.sp500_positioning(verbose=False)
    com = smm.commodity_positioning(verbose=False)
    eq_sig = smm.as_of(smm.cftc_signals(pos), dates, smm.CFTC_LAG_DAYS)
    co_sig = smm.as_of(smm.cftc_signals(com), dates, smm.CFTC_LAG_DAYS)
    dx = smm.dix(verbose=False)
    dx_sig = smm.as_of(smm.dix_signal(dx).to_frame(), dates, 1)
    print(f"Track J {tj.index[0]:%Y-%m-%d}..{tj.index[-1]:%Y-%m-%d}; {len(dates)} Tuesday decision dates. "
          f"CFTC S&P positions to {pos.index[-1]:%Y-%m-%d}, commodities to {com.index[-1]:%Y-%m-%d}, "
          f"DIX to {dx.index[-1]:%Y-%m-%d}\n")

    # ---- CHECK A: point in time
    for name, s, lag in (("CFTC S&P", eq_sig, 7), ("CFTC commodities", co_sig, 7), ("DIX", dx_sig, 1)):
        ok = s["source_date"].dropna()
        age = (ok.index - ok.values).days
        assert (age >= lag).all(), f"{name}: a value younger than {lag} days was used"
        print(f"CHECK A  {name}: youngest value used is {age.min()} days old (>= {lag}), "
              f"oldest {age.max()} days: OK")
    assert (dx_sig["source_date"].dropna().index - dx_sig["source_date"].dropna().values).days.max() <= 5, \
        "DIX gaps longer than a long weekend"

    # ---- CHECK B: hand recompute
    rng = np.random.default_rng(0)
    worst = 0.0
    for t in rng.choice(dates[70:-70], 20, replace=False):
        i = tj.index.get_loc(t)
        f = float(np.prod(1 + tj.iloc[i + 1:i + 1 + H]) - 1)
        b = float(np.prod(1 + tj.iloc[i - CTRL + 1:i + 1]) - 1)
        worst = max(worst, abs(f - fwd[H].loc[t]), abs(b - trail.loc[t]))
    assert worst < 1e-12, worst
    print(f"CHECK B  forward/trailing returns recomputed by hand on 20 dates: max gap {worst:.1e}: OK")

    # ---- equity side
    rows = []
    y = fwd[H].reindex(dates)
    c = trail.reindex(dates)
    for s in EQUITY:
        x = (dx_sig if s == "dix_z" else eq_sig)[s]
        r, d = fit(y, x, c)
        halves = [fit(y.loc[a:b], x.loc[a:b], c.loc[a:b])[0].tvalues["x"] for a, b in HALVES]
        uni = sm.OLS(d["y"], sm.add_constant(d[["x"]])).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
        shape = {f"t @{h}": fit(fwd[h].reindex(dates), x, c)[0].tvalues["x"] for h in HORIZONS if h != H}
        rows.append({"signal": s, "target": "Track J", "n": int(r.nobs),
                     "per 1 sd": r.params["x"], "t": r.tvalues["x"], "t 2011-18": halves[0],
                     "t 2019-26": halves[1], "t no control": uni.tvalues["x"], **shape,
                     "verdict": verdict(r.tvalues["x"], *halves)})
        if s == "asset_mgr_net_flow":
            check_r, check_d = r, d

    # ---- CHECK C: Newey-West by hand for one regression
    X = sm.add_constant(check_d[["x", "c"]]).values
    e = check_r.resid.values
    n = len(e)
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, NW_LAGS + 1):
        w = 1 - L / (NW_LAGS + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    t_hand = check_r.params.values[1] / np.sqrt((XtXi @ S @ XtXi)[1, 1])
    gap = abs(t_hand - check_r.tvalues["x"])
    assert gap < 0.02 * abs(t_hand) + 1e-6, (t_hand, check_r.tvalues["x"])
    print(f"CHECK C  Newey-West t recomputed by hand: {t_hand:.4f} vs statsmodels "
          f"{check_r.tvalues['x']:.4f}: OK\n")

    # ---- commodity side: asset minus Track J
    for s, sym in COMMOD.items():
        a = assets[sym]
        yc = (compound(a, H) - fwd[H]).reindex(dates)
        cc = (compound(a, -CTRL) - trail).reindex(dates)
        x = co_sig[s]
        r, d = fit(yc, x, cc)
        halves = [fit(yc.loc[p:q], x.loc[p:q], cc.loc[p:q])[0].tvalues["x"] for p, q in HALVES]
        uni = sm.OLS(d["y"], sm.add_constant(d[["x"]])).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
        shape = {f"t @{h}": fit((compound(a, h) - fwd[h]).reindex(dates), x, cc)[0].tvalues["x"]
                 for h in HORIZONS if h != H}
        rows.append({"signal": s, "target": f"{sym} - Track J", "n": int(r.nobs),
                     "per 1 sd": r.params["x"], "t": r.tvalues["x"], "t 2011-18": halves[0],
                     "t 2019-26": halves[1], "t no control": uni.tvalues["x"], **shape,
                     "verdict": verdict(r.tvalues["x"], *halves)})

    tab = pd.DataFrame(rows).set_index("signal")
    show = tab.copy()
    show["per 1 sd"] = show["per 1 sd"].map("{:+.2%}".format)
    for col in [c for c in show.columns if c == "t" or c.startswith("t ")]:
        show[col] = show[col].map("{:+.2f}".format)
    print("=" * 120)
    print(f"FORWARD {H}-SESSION RETURN ~ signal + own trailing {CTRL}-session return   "
          f"(per 1 sd = effect of a one-standard-deviation signal)")
    print("=" * 120)
    print(show.to_string())
    k = (tab["t"].abs() >= 1.5).sum()
    print(f"\n{k} of {len(tab)} signals reach |t| >= 1.5; about {0.13 * len(tab):.1f} would by chance "
          f"(two-sided 13% each). {int((tab['verdict'] == 'PASS').sum())} PASS.")
    tab.to_csv(REPO_ROOT / "trackk_smartmoney_first_pass.csv")


if __name__ == "__main__":
    main()
