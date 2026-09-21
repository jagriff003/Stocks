"""
Hold or swap? The decision version of the drawdown question.

`analyze_stop_levels.py` asked whether a position recovers.  That is the wrong
question for a decision, because a position does not have to recover to be worth
keeping — it only has to beat what the slot would otherwise hold.  James's
framing, which this implements:

    "Given a loss of X%, what are the odds of recovery to an amount greater than
     the current price based on the hold period remaining?"

    ...anchored against what the model could do with another stock.

TWO PROBABILITIES, AND ONLY THE SECOND IS A DECISION

  P(up from here)   Does the position end the hold above its CURRENT price?
                    This is "recovery to an amount greater than the current
                    price" exactly as asked — not recovery to the peak, and not
                    recovery to the entry, both of which are sunk.

  P(beats swap)     Does holding beat selling and buying the best available
                    name the model is not already holding, over the same
                    remaining days?  **This is the decision.** A position with
                    a 45% chance of rising is still worth keeping if the
                    alternative is worse, and a position with a 70% chance of
                    rising should be sold if the alternative is better.

CONDITIONED ON WHAT

Depth in the name's own daily volatilities, raw percentage below peak, and
**days remaining in the hold** — because a 20-day-old drawdown with 20 sessions
left is a different proposition from the same drawdown with three.

THE REPLACEMENT IS THE BEST NON-HELD NAME AT THAT MOMENT

Scored on the same panel, eligible under the same screens, excluding names
already in the book.  It ignores the correlation filter, which would sometimes
force a lower-ranked substitute — so the replacement modelled here is slightly
BETTER than the one actually available, which biases the test against holding.
That is the conservative direction for a question about whether to sell.

THE STANDING PRIOR

Track B tested rank-triggered exits and score-gap swaps and rejected both:
losses scaled monotonically with turnover, -1.2pp to -10.5pp. This is a
different trigger — drawdown state rather than rank — but the burden is the
same, and a positive result here would need the turnover cost netted off before
it meant anything.

Run:  python scripts/analyze_hold_vs_swap.py
      python scripts/analyze_hold_vs_swap.py --top-n 8 --hold 40
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

OUT = "rsi_ma_hold_vs_swap.csv"
POOL_FILE = "random_pool.csv"


def main() -> int:
    p = argparse.ArgumentParser(description="Hold or swap, conditioned on state")
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--hold", type=int, default=40)
    p.add_argument("--vol-window", type=int, default=60)
    p.add_argument("--max-corr", type=float, default=0.70)
    p.add_argument("--cost-bps", type=float, default=15.0,
                   help="round-trip cost of a swap, in bps, netted off the "
                        "replacement's return")
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

    print("=" * 108)
    print("HOLD OR SWAP — conditioned on depth, and on days left in the hold")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"pullback   top_n {cfg.top_n}   hold {cfg.hold_days}d   "
          f"swap cost {args.cost_bps:.0f} bps round trip")
    print("=" * 108)

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
    slip = slippage_panel(full.close, vol, lcfg, top_n=cfg.top_n)

    print("\n  running the model...", flush=True)
    m, res = run_arm(scored, None, full, cfg, slip, return_result=True)

    close = full.close
    dvol = realized_vol(close, args.vol_window)
    idx = close.index
    cost = args.cost_bps / 10_000.0

    rows = []
    hist = res.holdings_history
    for i, rec in enumerate(hist):
        entry = pd.Timestamp(rec["Date"])
        exit_ = (pd.Timestamp(hist[i + 1]["Date"]) if i + 1 < len(hist)
                 else idx[-1])
        if entry not in idx or exit_ not in idx:
            continue
        window = idx[(idx >= entry) & (idx <= exit_)]
        if len(window) < 4:
            continue
        book = set(rec["Holdings"])

        # The replacement, per day: best-scoring eligible name not in the book.
        # Computed once per window rather than per name, since it is the same
        # candidate whichever position is being considered for sale.
        sub_scores = scored.loc[window].drop(columns=[c for c in book
                                                      if c in scored.columns],
                                             errors="ignore")
        best = sub_scores.idxmax(axis=1)

        for n in rec["Holdings"]:
            if n not in close.columns:
                continue
            path = close[n].reindex(window)
            if path.isna().all():
                continue
            peak = path.cummax()
            v = dvol[n].reindex(window)
            exit_px = path.iloc[-1]
            for j, (d, px) in enumerate(path.items()):
                pk, vv, rep = peak.get(d), v.get(d), best.get(d)
                left = len(window) - j - 1
                if left < 3 or not (px == px and pk == pk and vv == vv and vv > 0):
                    continue
                if not isinstance(rep, str) or rep not in close.columns:
                    continue
                rp0, rp1 = close.at[d, rep], close.at[exit_, rep]
                if not (rp0 == rp0 and rp1 == rp1 and rp0 > 0):
                    continue
                held_ret = exit_px / px - 1
                swap_ret = (rp1 / rp0 - 1) - cost
                rows.append({
                    "Vols": (px / pk - 1) / vv,
                    "Pct": px / pk - 1,
                    "DaysLeft": left,
                    "HeldRet": held_ret,
                    "SwapRet": swap_ret,
                    "UpFromHere": held_ret > 0,
                    "BeatsSwap": held_ret > swap_ret,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  nothing collected")
        return 1
    print(f"  {len(df):,} position-days with a replacement available")

    vb = [-99, -4, -3, -2, -1, 0.001]
    vl = ["beyond -4", "-4 to -3", "-3 to -2", "-2 to -1", "-1 to 0"]
    db = [2, 8, 15, 25, 99]
    dl = ["3-8 left", "9-15", "16-25", "26+"]
    df["VB"] = pd.cut(df["Vols"], bins=vb, labels=vl)
    df["DB"] = pd.cut(df["DaysLeft"], bins=db, labels=dl)

    for metric, title, note in (
        ("UpFromHere",
         "P(ends the hold ABOVE the current price)",
         "James's question exactly: odds of recovery to more than it is worth now."),
        ("BeatsSwap",
         "P(HOLDING beats swapping into the best available name)",
         "The decision. Below 50% means the slot is better used elsewhere."),
    ):
        print("\n" + "=" * 108)
        print(title)
        print("=" * 108)
        print(f"  {note}")
        tab = df.pivot_table(index="VB", columns="DB", values=metric,
                             aggfunc="mean", observed=False)
        cnt = df.pivot_table(index="VB", columns="DB", values=metric,
                             aggfunc="size", observed=False)
        hdr = f"    {'vols below peak':<16}" + "".join(f"{c:>12}" for c in dl)
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for r in vl:
            cells = ""
            for c in dl:
                val = tab.loc[r, c] if (r in tab.index and c in tab.columns) else np.nan
                n = cnt.loc[r, c] if (r in cnt.index and c in cnt.columns) else 0
                cells += f"{'%.0f%%' % (100 * val) if val == val and n >= 40 else '-':>12}"
            print(f"    {r:<16}{cells}")
        print(f"    {'(observations)':<16}" +
              "".join(f"{int(cnt[c].sum()):>12,}" for c in dl if c in cnt.columns))

    # --- the payoff, not just the probability ---
    print("\n" + "=" * 108)
    print("MEAN RETURN: HOLDING minus SWAPPING")
    print("=" * 108)
    print("  Probability is not payoff. A slot can be worth keeping at 45% odds")
    print("  if the upside is larger. Negative here means the swap wins.")
    tab = df.pivot_table(index="VB", columns="DB",
                         values=df["HeldRet"] - df["SwapRet"], aggfunc="mean",
                         observed=False) if False else None
    df["Edge"] = df["HeldRet"] - df["SwapRet"]
    tab = df.pivot_table(index="VB", columns="DB", values="Edge",
                         aggfunc="mean", observed=False)
    cnt = df.pivot_table(index="VB", columns="DB", values="Edge",
                         aggfunc="size", observed=False)
    hdr = f"    {'vols below peak':<16}" + "".join(f"{c:>12}" for c in dl)
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for r in vl:
        cells = ""
        for c in dl:
            val = tab.loc[r, c] if (r in tab.index and c in tab.columns) else np.nan
            n = cnt.loc[r, c] if (r in cnt.index and c in cnt.columns) else 0
            cells += f"{'%+.2f%%' % (100 * val) if val == val and n >= 40 else '-':>12}"
        print(f"    {r:<16}{cells}")

    df.drop(columns=["VB", "DB"]).to_csv(REPO_ROOT / OUT, index=False)

    print("\n" + "=" * 108)
    print("HOW TO READ IT")
    print("=" * 108)
    print("  A swap rule is worth building only where P(beats swap) is clearly")
    print("  below 50% AND the mean edge is clearly negative AND the cell has")
    print("  enough observations to mean something. One cell out of twenty")
    print("  crossing the line is what noise looks like.")
    print("  Position-days overlap heavily, so treat the cells as descriptive.")
    print("  Track B rejected two exit rules; the burden here is the same.")
    print(f"\n  Exported to {REPO_ROOT / OUT}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
