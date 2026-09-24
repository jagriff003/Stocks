"""
TODO 0m, second step: do the two S&P positioning flows improve Track K's timing?

James (2026-09-24): take both flows to the Track K timing test — the
asset-manager PASS even though it runs opposite to the smart-money idea, and
the leveraged-fund LEAD — "to see what changes".

THE "LONELY" SCORE

Oriented by what the first pass found, higher = lonelier for the stock book:
    combined = z(asset-manager 13-week flow) - z(leveraged-fund 13-week flow)
Each flow is z-scored against its own trailing three years only, and CFTC data
is used only 7+ days after its position date (after the Friday release).

PRE-REGISTERED VARIANTS (fixed 2026-09-24, before any result)

    baseline  the live trigger (2 of 4 real assets beat SPY by 10% and cash,
              stock-bond correlation positive)
    Confirm   fire only if the trigger fires AND lonely > 0
    Early     ALSO fire when only 1 of 4 assets qualifies, if lonely > 1
    Both      Confirm and Early together
    Both-AM, Both-LEV   Both, with each flow alone, to see which does the work

PRIMARY: obeyed on Track J (signal on SPY, Track J scaled, rotation Tuesdays),
2012-2026, both biweekly phases.  "Improves" = cost no worse than baseline at
BOTH phases, Sharpe no worse, and a better forward excess of the held basket.

SECONDARY — the out-of-sample check: the flows were selected on Track J's own
2011-2026 returns, so the primary is partly circular.  The same variants on the
Ken French top-momentum decile from 2008 include 2008-2011, which the selection
never saw.

With ~13 firing rotations since 2012, every conclusion here is weak.

CHECKS (the script refuses to print results if any fails)

  A  the baseline through the external gate reproduces the live path exactly
  B  Confirm fires on a subset of the baseline's rotations, Early on a superset
  C  every CFTC value used is at least 7 days old

Run:  python scripts/analyze_smart_money_trackk.py
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

from momentum import smartmoney as smm  # noqa: E402
from momentum.hedge import apply_weights, hedge_weights, regime_open  # noqa: E402
from momentum.marketstore import load_daily_returns  # noqa: E402
from momentum.schedule import TUESDAY, rotation_dates  # noqa: E402
from momentum.trackk import LIVE_CONFIG  # noqa: E402
from scripts.analyze_hedge_etfs import load as load_etfs  # noqa: E402
from scripts.analyze_hedge_trackj import load as load_trackj  # noqa: E402
from scripts.analyze_hedge_trackj import stats  # noqa: E402
from scripts.analyze_trigger_timing import forward_excess, on_state  # noqa: E402

RELAXED = replace(LIVE_CONFIG, regime_min=1)
ANCHORS = {"A": "2026-09-15", "B": "2026-09-22"}


def gates(trig, trig1, L):
    lon = L.to_numpy()
    pos = np.nan_to_num(lon, nan=-np.inf) > 0
    high = np.nan_to_num(lon, nan=-np.inf) > 1
    return {"Confirm": trig & pos, "Early": trig | (trig1 & high), "Both": (trig & pos) | (trig1 & high)}


def evaluate(book, spy, assets, idx, start, windows, label):
    trig = regime_open(spy, assets, LIVE_CONFIG)
    trig1 = regime_open(spy, assets, RELAXED)
    pos = smm.sp500_positioning(verbose=False)
    lon = smm.lonely_score(pos)
    L = smm.as_of(lon, idx, smm.CFTC_LAG_DAYS)
    ages = (L["source_date"].dropna().index - L["source_date"].dropna().values).days
    assert (ages >= smm.CFTC_LAG_DAYS).all()
    print(f"CHECK C  [{label}] youngest CFTC value used is {ages.min()} days old (>= 7): OK")

    variants = {"baseline": trig}
    g = gates(trig, trig1, L["combined"])
    variants.update({"Confirm": g["Confirm"], "Early": g["Early"], "Both": g["Both"]})
    variants["Both-AM"] = gates(trig, trig1, L["am"])["Both"]
    variants["Both-LEV"] = gates(trig, trig1, L["lev"])["Both"]

    rows = []
    for ph, anchor in ANCHORS.items():
        rot = [d for d in rotation_dates(idx, weekday=TUESDAY, every_weeks=2, anchor=pd.Timestamp(anchor))
               if d >= pd.Timestamp(start)]
        w_live = hedge_weights(spy, assets, LIVE_CONFIG, decide_at=rot)
        w_gate = hedge_weights(spy, assets, LIVE_CONFIG, decide_at=rot, regime_gate=trig)
        pd.testing.assert_frame_equal(w_live, w_gate)
        base_on = None
        for name, gate in variants.items():
            w = hedge_weights(spy, assets, LIVE_CONFIG, decide_at=rot, regime_gate=gate)
            on = on_state(w, rot)
            if name == "baseline":
                base_on = on
            elif name.startswith("Confirm"):
                assert not (on & ~base_on).any(), "Confirm fired where the baseline did not"
            elif name.startswith("Early"):
                assert not (base_on & ~on).any(), "Early missed a baseline firing"
            res = apply_weights(book, assets, w, LIVE_CONFIG.cost_bps, LIVE_CONFIG.exec_lag).loc[start:]
            h, u = stats(res["net"], assets["CASH"]), stats(book.reindex(res.index), assets["CASH"])
            fx = forward_excess(w, assets, spy, rot, 63)
            row = {"book": label, "variant": name, "phase": ph, "dCAGR": h["CAGR"] - u["CAGR"],
                   "dSharpe": h["Sharpe"] - u["Sharpe"], "% rotations on": on.mean(),
                   "flips/yr": on.astype(int).diff().abs().sum() / (len(rot) / 26),
                   "fired": int(on.sum()), "fwd basket - SPY": fx.mean() if len(fx) else np.nan}
            for wname, (a, b) in windows.items():
                rr = res["net"].loc[a:b]
                row[wname] = (1 + rr).prod() - (1 + book.loc[rr.index]).prod()
            rows.append(row)
    print(f"CHECK A  [{label}] baseline through the external gate == live path, both phases: OK")
    print(f"CHECK B  [{label}] Confirm fires on a subset, Early on a superset of the baseline: OK")
    return pd.DataFrame(rows)


def show(t):
    out = t.copy()
    for c in out.columns:
        if c in ("dSharpe", "flips/yr"):
            out[c] = out[c].map(lambda v: f"{v:+.2f}" if c == "dSharpe" else f"{v:.2f}")
        elif out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    return out.set_index(["variant", "phase"]).drop(columns="book").to_string()


def main():
    pd.set_option("display.width", 230)

    # ---- primary: Track J
    tj, assets, store = load_trackj()
    for c in assets:
        f = assets[c].first_valid_index()
        assets.loc[f:, c] = assets.loc[f:, c].fillna(0.0)
    spy = store["SPY"].reindex(tj.index)
    prim = evaluate(tj, spy, assets, tj.index, "2012-01-01",
                    {"2021-22": ("2021-01-01", "2022-10-31"), "2025-26": ("2025-01-01", "2026-09-30")},
                    "Track J")

    # ---- secondary: French top decile, including the years the selection never saw
    fr, fassets, _, fstore = load_etfs("french")
    fspy = fstore["SPY"].reindex(fr.index)
    sec = evaluate(fr, fspy, fassets, fr.index, "2008-07-01",
                   {"2008-11 (unseen)": ("2008-07-01", "2011-09-30"), "2021-22": ("2021-01-01", "2022-10-31")},
                   "French decile")

    print("\n" + "=" * 120)
    print("PRIMARY - obeyed on Track J, 2012-2026. d* = hedged minus Track J; fwd = held basket minus SPY, next 63 sessions")
    print("=" * 120)
    print(show(prim))
    print("\n" + "=" * 120)
    print("SECONDARY - French top-momentum decile, 2008-07..2026-07 (2008-11 was never seen by the signal selection)")
    print("=" * 120)
    print(show(sec))
    pd.concat([prim, sec]).to_csv(REPO_ROOT / "trackk_smartmoney_timing.csv", index=False)


if __name__ == "__main__":
    main()
