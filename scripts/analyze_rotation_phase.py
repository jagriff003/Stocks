"""
How much of the result is luck about WHICH day the rotation clock started?

TODO 0h.1.  FINDINGS records that rotation phase is worth roughly 3.2pp of the
headline CAGR at `hold_days=14` — the model rebalances every 14 sessions, and
which of those 14 possible offsets the backtest happens to start on moves the
answer by that much.  It is not a small effect and it is pure artifact.

**Going to `hold=42` makes this worse, not better.**  Six rebalances a year
instead of eighteen means each rotation decision carries three times the
weight, and there are 42 possible phases instead of 14.  Reporting one of them
as "the" backtest result would be reporting one draw from a distribution
nobody has looked at.

WHAT IT DOES

Shifts the start of the panel by 0, 1, 2, ... sessions.  Everything else is
identical, so the only thing that changes is which calendar days the rebalances
land on.  The spread across offsets is the size of the artifact; the median is
the honest point estimate.

Both scores are run at every offset, because the question is not only "how
noisy is the new score" but "does it still beat the old one at most phases".
A score that wins at the reported phase and loses at the other forty-one has
not been shown to win.

Run:  python scripts/analyze_rotation_phase.py --hold 42
      python scripts/analyze_rotation_phase.py --hold 14 --step 1
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
from momentum.liquidity import (LiquidityConfig, apply_tradable,
                                slippage_panel, tradable_mask)
from momentum.reversal import ReversalConfig, build_terms, composite
from momentum.strategy import compute_scores
from momentum.universe import defensive_symbols

from scripts.analyze_reversal_backtest import (equal_weight_arm, load_volume,
                                               run_arm)
from scripts.analyze_reversal_ic import NON_EQUITY, load_pool_panel, pool_symbols

OUT = "rsi_ma_rotation_phase.csv"
POOL_FILE = "random_pool.csv"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Rotation-phase sensitivity of the Track J model")
    p.add_argument("--hold", type=int, default=42)
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--step", type=int, default=3,
                   help="sample every Nth phase offset; 1 is exhaustive")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--min-adv", type=float, default=10e6)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--account", type=float, default=100_000.0)
    p.add_argument("--level-floor", choices=["on", "off"], default="off")
    p.add_argument("--weekday", action="store_true",
                   help="day-of-week mode: use a hold that is a whole number "
                        "of weeks so the rebalance weekday is FIXED, and test "
                        "all five. Separate question from phase — see below.")
    p.add_argument("--weekday-hold", type=int, default=40,
                   help="whole-week hold for the weekday test (40 = 8 weeks)")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    # Phase and day-of-week are different questions and it takes one line of
    # arithmetic to see why.  At hold=42 the rebalance advances 42 mod 5 = 2
    # weekdays each cycle, so the weekday ROTATES and every phase run already
    # averages over all five.  That makes the phase test clean of weekday
    # effects — and also blind to them.  A hold that is a whole number of weeks
    # (40 = 8 weeks) pins the weekday instead, so five offsets give five
    # weekdays and the difference between them is the day-of-week effect.
    if args.weekday:
        args.hold = args.weekday_hold
        args.step = 1

    cfg = replace(production_config(), top_n=args.top_n, hold_days=args.hold,
                  vix=None)
    rcfg = ReversalConfig()
    lcfg = LiquidityConfig(account_notional=args.account)
    defensive = set(defensive_symbols())

    offsets = list(range(0, 5 if args.weekday else args.hold,
                         max(1, args.step)))

    print("=" * 96)
    print("ROTATION PHASE SENSITIVITY")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"hold {args.hold}d   top_n {args.top_n}   level floor "
          f"{args.level_floor}   {len(offsets)} of {args.hold} phases")
    print("=" * 96)
    print("  Each offset shifts the panel start by N sessions, so the rebalance")
    print("  dates move and nothing else does. Spread = the size of the artifact.")

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

    rows = []
    print(f"\n    {'offset':>7}{'live':>10}{'new':>10}{'delta':>9}"
          f"{'new Sh':>9}{'new DD':>9}{'equal-wt':>10}   {'weekday':<11}")
    print("    " + "-" * 74)

    for off in offsets:
        sub = PriceData(close=full.close.iloc[off:], open_=full.open_.iloc[off:],
                        spy=full.spy, vix=full.vix)
        v = vol.iloc[off:]

        ranking, base, _ = compute_scores(sub, cfg)
        terms = build_terms(sub.close, rcfg)
        pull = composite(terms, replace(rcfg, turn_weight=-1.0,
                                        strength_weight=0.0, room_weight=1.0),
                         "flip").reindex_like(ranking)

        tmask = tradable_mask(sub.close, v, args.min_adv, args.min_price, lcfg)
        ranking = apply_tradable(ranking, tmask, defensive)
        pull = apply_tradable(pull, tmask, defensive)
        slip = slippage_panel(sub.close, v, lcfg, top_n=cfg.top_n)
        base_arm = base if args.level_floor == "on" else None

        eq, _ = equal_weight_arm(
            sub, cfg, ranking,
            [x for x in ranking.columns if x not in defensive], slip)
        live = run_arm(ranking, base_arm, sub, cfg, slip)
        new, nres = run_arm(pull, base_arm, sub, cfg, slip, return_result=True)

        # What weekday did the rebalances actually land on?  Reported rather
        # than assumed, because holidays break the clean arithmetic.
        days = [pd.Timestamp(h["Date"]).day_name()
                for h in nres.holdings_history]
        modal = pd.Series(days).mode()
        modal = modal.iloc[0] if len(modal) else "-"
        share = (pd.Series(days) == modal).mean() if days else float("nan")

        rows.append({"Offset": off, "ModalWeekday": modal,
                     "ModalShare": share,
                     "Live": live["CAGR"], "New": new["CAGR"],
                     "Delta": new["CAGR"] - live["CAGR"],
                     "NewSharpe": new["Sharpe"], "NewMaxDD": new["MaxDD"],
                     "EqualWeight": eq["CAGR"],
                     "NewMinusEW": new["CAGR"] - eq["CAGR"]})
        print(f"    {off:>7}{live['CAGR']:>10.2%}{new['CAGR']:>10.2%}"
              f"{new['CAGR'] - live['CAGR']:>+9.2%}{new['Sharpe']:>9.2f}"
              f"{new['MaxDD']:>9.1%}{eq['CAGR']:>10.2%}"
              f"   {modal[:3]} {share:.0%}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(REPO_ROOT / OUT, index=False)

    print("\n" + "=" * 96)
    print("THE ARTIFACT")
    print("=" * 96)
    for col, label in (("New", "new score"), ("Live", "live composite"),
                       ("EqualWeight", "equal weight")):
        s = d[col]
        print(f"  {label:<18} median {s.median():>7.2%}   "
              f"range {s.min():>7.2%} to {s.max():>7.2%}   "
              f"spread {s.max() - s.min():>6.2%}   sd {s.std():>6.2%}")

    print(f"\n  Phase spread on the new score is {d['New'].max() - d['New'].min():.2%} "
          f"of CAGR.\n  FINDINGS records ~3.2pp at hold=14; anything much "
          f"larger here is the\n  cost of rebalancing six times a year instead "
          f"of eighteen.")

    if args.weekday:
        print("\n" + "=" * 96)
        print("DAY OF WEEK")
        print("=" * 96)
        print(f"  hold={args.hold} is {args.hold / 5:.0f} whole weeks, so each "
              f"offset pins the rebalance to one\n  weekday. 'share' is how "
              f"often it actually landed there — holidays break the tie.")
        print(f"\n    {'weekday':<12}{'share':>8}{'new':>10}{'live':>10}"
              f"{'delta':>9}")
        print("    " + "-" * 45)
        for _, r in d.iterrows():
            print(f"    {r['ModalWeekday']:<12}{r['ModalShare']:>8.0%}"
                  f"{r['New']:>10.2%}{r['Live']:>10.2%}{r['Delta']:>+9.2%}")
        spread = d["New"].max() - d["New"].min()
        print(f"\n  Weekday spread on the new score: {spread:.2%} of CAGR.")
        print(f"  Compare against the PHASE spread from the default run before "
              f"reading anything\n  into it — if they are the same size, this "
              f"is phase noise wearing a weekday label.")

    wins = (d["Delta"] > 0).mean()
    beats_ew = (d["NewMinusEW"] > 0).mean()
    print(f"\n  The new score beats the live composite at "
          f"{wins:.0%} of sampled phases.")
    print(f"  It beats owning the pool equal-weighted at {beats_ew:.0%}.")
    print(f"  Median advantage over live: {d['Delta'].median():+.2%} "
          f"(worst {d['Delta'].min():+.2%}, best {d['Delta'].max():+.2%})")

    if wins < 0.8:
        print("\n  *** The advantage is NOT robust to phase. The reported "
              "backtest number is\n      one lucky draw and should not be "
              "acted on.")
    else:
        print("\n  The advantage survives at most phases, so it is not an "
              "artifact of where\n  the rotation clock happened to start. The "
              "LEVEL is still one draw — quote\n  the median, not the best.")

    print(f"\n  Exported to {REPO_ROOT / OUT}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
