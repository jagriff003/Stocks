"""
Is there a depth of drawdown from which a position does not recover?

TODO 0g, the evidence half.  James's hypothesis: a trailing stop has to be sized
in units of the name's own daily volatility, because -8% is noise on a 5%/day
semiconductor and a thesis break on a 1%/day utility — and beyond some multiple
of that, recovery stops happening and bailing beats holding.

That is a sharp, falsifiable claim and it is not obviously true for THIS model.
`pullback` is a dip-buying score: it buys names that have fallen, on the premise
that they revert. A position drifting further below its peak might therefore be
a BETTER hold, not a worse one, and the excursion that looks like a stop signal
might be the entry signal arriving late. The two readings predict opposite signs
and only measurement separates them.

WHAT IT MEASURES

For every position the model held, on every day it held it:

    excursion = (price / peak-since-entry - 1) / daily volatility

i.e. how many of its own daily moves the name sits below its best level since
entry.  Then, conditional on that excursion, the return from here to the end of
the hold — which is exactly the quantity a stop decision trades away.

READ THE `hold from here` COLUMN

If deep excursions are followed by NEGATIVE returns to the end of the hold, a
stop there would have saved money and the hypothesis is supported.  If they are
followed by POSITIVE returns, the stop would have sold the bottom and the
reversion premise is doing its job.

WHAT THIS IS NOT

It is not a backtest of a stop rule.  It conditions on a state and reports what
followed, which tells you whether a threshold EXISTS before anything is built.
An actual stop changes the book — the freed slot buys something else — and that
second-order effect needs `momentum/exits.py` and a full simulation.  Doing the
cheap test first is deliberate: Track B rejected two exit rules, so the prior is
against, and there is no point building machinery to implement a threshold that
the data says is not there.

Run:  python scripts/analyze_stop_levels.py
      python scripts/analyze_stop_levels.py --top-n 8 --hold 40
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

from momentum.data import PriceData
from momentum.experiments import production_config
from momentum.liquidity import (LiquidityConfig, apply_tradable, realized_vol,
                                slippage_panel, tradable_mask)
from momentum.reversal import ReversalConfig, build_terms, composite
from momentum.universe import defensive_symbols

from scripts.analyze_reversal_backtest import load_volume, run_arm
from scripts.analyze_reversal_ic import NON_EQUITY, load_pool_panel, pool_symbols

OUT = "rsi_ma_stop_levels.csv"
POOL_FILE = "random_pool.csv"


def main() -> int:
    p = argparse.ArgumentParser(description="Where, if anywhere, to put a stop")
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--hold", type=int, default=40)
    p.add_argument("--vol-window", type=int, default=60)
    p.add_argument("--max-corr", type=float, default=0.70)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--min-adv", type=float, default=10e6)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    cfg = replace(production_config(), top_n=args.top_n, hold_days=args.hold,
                  vix=None)
    if args.max_corr:
        cfg = replace(cfg, correlation=replace(
            cfg.correlation, enabled=True, apply_above_vix=None,
            method="absolute", max_correlation=args.max_corr))
    rcfg = ReversalConfig()
    lcfg = LiquidityConfig()
    defensive = set(defensive_symbols())

    print("=" * 104)
    print("STOP LEVELS — is there a depth from which positions do not recover?")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"pullback   top_n {cfg.top_n}   hold {cfg.hold_days}d   "
          f"vol window {args.vol_window}d")
    print("=" * 104)

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
    pull = apply_tradable(
        composite(terms, replace(rcfg, turn_weight=-1.0, strength_weight=0.0,
                                 room_weight=1.0), "flip"), tmask, defensive)
    slip = slippage_panel(full.close, vol, lcfg, top_n=cfg.top_n)

    print("\n  running the model to collect its positions...", flush=True)
    m, res = run_arm(pull, None, full, cfg, slip, return_result=True)

    close = full.close
    dvol = realized_vol(close, args.vol_window)
    idx = close.index

    # --- walk every position, day by day ---
    rows = []
    # `holdings_history`, not `rebalance_history`: run_arm returns the
    # simulation result, and only `run_strategy` attaches the rebalance log to
    # it. Reading the empty default silently produced zero position-days.
    hist = res.holdings_history
    if not hist:
        print("  no holdings history on the result")
        return 1
    for i, rec in enumerate(hist):
        entry = pd.Timestamp(rec["Date"])
        exit_ = (pd.Timestamp(hist[i + 1]["Date"]) if i + 1 < len(hist)
                 else idx[-1])
        if entry not in idx or exit_ not in idx:
            continue
        window = idx[(idx >= entry) & (idx <= exit_)]
        if len(window) < 3:
            continue
        for n in rec["Holdings"]:
            if n not in close.columns:
                continue
            path = close[n].reindex(window)
            if path.isna().all():
                continue
            peak = path.cummax()
            v = dvol[n].reindex(window)
            final = path.iloc[-1]
            entry_px = path.dropna().iloc[0] if path.notna().any() else np.nan
            vals = path.to_numpy()
            n_days = len(window)
            for j, (d, px) in enumerate(path.items()):
                pk, vv = peak.get(d), v.get(d)
                if not (px == px and pk == pk and vv == vv and vv > 0):
                    continue
                fwd = vals[j + 1:]
                fwd = fwd[np.isfinite(fwd)]
                if fwd.size == 0:
                    continue
                best_ahead = float(fwd.max())
                rows.append({
                    "Excursion": (px / pk - 1) / vv,
                    "OffPeakPct": px / pk - 1,
                    # RECOVERY, three ways. "Positive return from here" is not
                    # recovery -- a position can drift up a little and never
                    # regain its high. These are the questions a stop decision
                    # actually turns on.
                    "RecoverPeak": best_ahead >= pk,
                    "RecoverEntry": (best_ahead >= entry_px
                                     if entry_px == entry_px else np.nan),
                    "EndsAbovePeak": final >= pk,
                    "EndsAboveEntry": (final >= entry_px
                                       if entry_px == entry_px else np.nan),
                    "ToHoldEnd": final / px - 1,
                    "DaysLeft": int(n_days - j - 1),
                    "HoldFrac": j / max(1, n_days - 1),
                    "Name": n, "Date": d,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  no position-days collected")
        return 1
    df = df[df["DaysLeft"] >= 3]

    print(f"  {len(df):,} position-days over {df['Name'].nunique()} names")

    edges = [-np.inf, -6, -5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0.001]
    labels = ["< -6", "-6 to -5", "-5 to -4", "-4 to -3", "-3 to -2.5",
              "-2.5 to -2", "-2 to -1.5", "-1.5 to -1", "-1 to -0.5",
              "-0.5 to 0"]
    df["Bucket"] = pd.cut(df["Excursion"], bins=edges, labels=labels)

    print("\n" + "=" * 104)
    print("PROBABILITY OF RECOVERY, BY DEPTH BELOW THE PEAK SINCE ENTRY")
    print("=" * 104)
    print("  'regains peak'  — does it trade back to its high-water mark before")
    print("                    the hold ends? This is recovery in the strict sense.")
    print("  'regains entry' — does it get back to what was paid for it?")
    print("  'ends > peak'   — is it AT or above the peak when the hold ends?")
    hdr = (f"    {'vols below peak':<16}{'days':>9}{'off peak':>10}"
           f"{'regains peak':>14}{'regains entry':>15}{'ends > peak':>13}"
           f"{'to hold end':>13}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    out = []
    for lab in labels:
        sub = df[df["Bucket"] == lab]
        if sub.empty:
            continue
        row = {"Bucket": lab, "N": len(sub),
               "OffPeakPct": sub["OffPeakPct"].mean(),
               "RecoverPeak": sub["RecoverPeak"].mean(),
               "RecoverEntry": sub["RecoverEntry"].mean(),
               "EndsAbovePeak": sub["EndsAbovePeak"].mean(),
               "MeanToHoldEnd": sub["ToHoldEnd"].mean()}
        print(f"    {lab:<16}{len(sub):>9,}{row['OffPeakPct']:>10.1%}"
              f"{row['RecoverPeak']:>14.0%}{row['RecoverEntry']:>15.0%}"
              f"{row['EndsAbovePeak']:>13.0%}{row['MeanToHoldEnd']:>13.2%}")
        out.append(row)

    print("\n" + "=" * 104)
    print("DOES IT DEPEND ON WHEN IN THE HOLD IT HAPPENS?")
    print("=" * 104)
    print("  Same question, split by how far through the hold the drawdown")
    print("  occurs. Early has time to recover; late does not, so any gradient")
    print("  here is about remaining time rather than about the signal.")
    thirds = {"early (first 1/3)": (0.0, 1 / 3),
              "middle": (1 / 3, 2 / 3),
              "late (last 1/3)": (2 / 3, 1.01)}
    hdr2 = (f"    {'vols below':<12}{'when':<20}{'days':>8}"
            f"{'regains peak':>14}{'to hold end':>13}")
    print(hdr2)
    print("    " + "-" * (len(hdr2) - 4))
    for k in (1, 2, 3, 4, 5):
        band = df[(df["Excursion"] <= -k) & (df["Excursion"] > -(k + 1))]
        if len(band) < 100:
            continue
        for wlab, (lo, hi) in thirds.items():
            sub = band[(band["HoldFrac"] >= lo) & (band["HoldFrac"] < hi)]
            if len(sub) < 40:
                continue
            print(f"    {('-%d to -%d' % (k, k + 1)):<12}{wlab:<20}"
                  f"{len(sub):>8,}{sub['RecoverPeak'].mean():>14.0%}"
                  f"{sub['ToHoldEnd'].mean():>13.2%}")
            out.append({"Bucket": f"-{k} to -{k+1} / {wlab}", "N": len(sub),
                        "RecoverPeak": sub["RecoverPeak"].mean(),
                        "MeanToHoldEnd": sub["ToHoldEnd"].mean()})

    print("\n" + "=" * 104)
    print("IS THERE AN ABSOLUTE THRESHOLD AS WELL AS A VOL-SCALED ONE?")
    print("=" * 104)
    print("  Recovery probability by raw percentage off the peak, so the two")
    print("  framings can be compared directly.")
    pedges = [-1.01, -0.40, -0.30, -0.20, -0.15, -0.10, -0.05, 0.001]
    plabels = ["< -40%", "-40 to -30%", "-30 to -20%", "-20 to -15%",
               "-15 to -10%", "-10 to -5%", "-5 to 0%"]
    df["PctBucket"] = pd.cut(df["OffPeakPct"], bins=pedges, labels=plabels)
    hdr3 = (f"    {'off peak':<14}{'days':>9}{'avg vols':>10}"
            f"{'regains peak':>14}{'to hold end':>13}")
    print(hdr3)
    print("    " + "-" * (len(hdr3) - 4))
    for lab in plabels:
        sub = df[df["PctBucket"] == lab]
        if len(sub) < 50:
            continue
        print(f"    {lab:<14}{len(sub):>9,}{sub['Excursion'].mean():>10.1f}"
              f"{sub['RecoverPeak'].mean():>14.0%}"
              f"{sub['ToHoldEnd'].mean():>13.2%}")
        out.append({"Bucket": lab, "N": len(sub),
                    "RecoverPeak": sub["RecoverPeak"].mean(),
                    "MeanToHoldEnd": sub["ToHoldEnd"].mean()})

    pd.DataFrame(out).to_csv(REPO_ROOT / OUT, index=False)

    print("\n" + "=" * 104)
    print("HOW TO READ IT")
    print("=" * 104)
    print("  The t-statistics are optimistic: position-days overlap heavily")
    print("  (the same position contributes many rows) so they are not")
    print("  independent observations. Read the SIGN and the monotonicity.")
    print("  A threshold worth acting on shows a mean that turns negative and")
    print("  STAYS negative as the threshold deepens. A single negative bucket")
    print("  surrounded by positive ones is noise.")
    print(f"\n  Exported to {REPO_ROOT / OUT}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
