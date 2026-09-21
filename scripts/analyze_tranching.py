"""
Does splitting capital across staggered rotation clocks remove the phase bet?

TODO 0i.1.  At `hold=40` the model's CAGR spans 13.13% to 27.69% depending only
on which session the rotation clock started — a **14.56pp** spread, against
0.71pp for equal-weighting the pool.  None of that is information; it is which
days you happened to trade on.  In live trading you get exactly one phase and
cannot know in advance whether it is a good one.

THE IDEA, AND WHY IT IS DIFFERENT FROM EVERY REJECTED TRACK

Split the capital into `k` sleeves, each running the identical model on a clock
offset by `hold/k` sessions.  The book becomes the union of the sleeves and the
realized return approaches the phase *average* instead of a single draw.

**This makes no signal claim at all.**  Tracks A, B, C and D were all rejected
because they claimed a timing edge and paid for it in turnover.  Tranching
claims nothing about when to trade; it claims only that betting on one arbitrary
phase is avoidable.  That is why it is worth testing despite this repo's uniform
record of rejecting complexity.

THREE THINGS THAT HAVE TO BE TRUE FOR IT TO BE FREE

  1. **Phase spread must actually collapse.**  If k sleeves do not narrow the
     range, the whole exercise is pointless.
  2. **Turnover per dollar must not rise.**  Each sleeve trades 1/k of the
     capital, k times as often in wall-clock terms — it should wash.  Asserted
     rather than assumed.
  3. **Slippage should IMPROVE.**  A sleeve's order is `account / (k * top_n)`
     rather than `account / top_n`, and the impact term scales with the square
     root of participation.  Smaller orders are cheaper, so the cost model is
     given the true per-sleeve order size rather than the single-book one.

WHAT IT COSTS

Position count: k sleeves of `top_n` names, overlapping.  The union is reported,
because "8 positions" becoming "22 positions" is an operational fact even if the
economics are unchanged.

Run:  python scripts/analyze_tranching.py
      python scripts/analyze_tranching.py --tranches 1 2 4 5 --hold 40
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
from momentum.metrics import calculate_performance_metrics
from momentum.reversal import ReversalConfig, build_terms, composite
from momentum.strategy import compute_scores
from momentum.universe import defensive_symbols

from scripts.analyze_reversal_backtest import load_volume, run_arm
from scripts.analyze_reversal_ic import NON_EQUITY, load_pool_panel, pool_symbols

OUT = "rsi_ma_tranching.csv"
POOL_FILE = "random_pool.csv"


def combine(streams, rf=0.045):
    """
    Equal-capital combination of sleeve return streams.

    Daily mean of the sleeves, i.e. a costless daily rebalance back to equal
    capital across sleeves.  That is the same convention the rest of this repo
    uses for equal weight within a book (see `_segment_return`), and it is
    recorded there as a known approximation rather than an intended feature: it
    slightly understates a runaway sleeve.  Using it here keeps tranched and
    untranched results comparable, which matters more than the approximation.
    """
    df = pd.concat(streams, axis=1).dropna(how="all")
    r = df.mean(axis=1).dropna()
    return r, calculate_performance_metrics(r, risk_free_rate=rf)


def main() -> int:
    p = argparse.ArgumentParser(description="Staggered-tranche phase analysis")
    p.add_argument("--tranches", type=int, nargs="+", default=[1, 2, 4, 5, 8])
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--hold", type=int, default=40)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--min-adv", type=float, default=10e6)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--account", type=float, default=100_000.0)
    p.add_argument("--max-corr", type=float, default=0.70)
    p.add_argument("--max-phases", type=int, default=8,
                   help="starting phases sampled per tranche count")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    cfg = replace(production_config(), top_n=args.top_n, hold_days=args.hold,
                  vix=None)
    if args.max_corr:
        cfg = replace(cfg, correlation=replace(
            cfg.correlation, enabled=True, apply_above_vix=None,
            method="absolute", max_correlation=args.max_corr))
    rcfg = ReversalConfig()
    defensive = set(defensive_symbols())

    print("=" * 104)
    print("TRANCHING — does splitting capital across rotation clocks remove "
          "the phase bet?")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"top_n {cfg.top_n}   hold {cfg.hold_days}d   "
          f"account ${args.account:,.0f}   tranches {args.tranches}")
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

    def score_panel(offset: int):
        """
        Score panel for a sleeve starting `offset` sessions in.

        RECOMPUTED per offset, never sliced from a shared panel. Slicing does
        not move the rotation clock: the eligibility gate waits for 200 non-NaN
        scores, and the score panel already carries a ~315-day NaN prefix, so
        cutting 0-39 days off the front leaves the first eligible date exactly
        where it was. An earlier version of this script sliced, and reported a
        0.00% phase spread for a single untranched book -- which is to say it
        measured forty copies of the same phase and called it a result.

        Only the reversal terms are rebuilt, not the production composite: the
        terms are vectorized and cheap, and `base` is unused here because the
        level floor is off.
        """
        sub = PriceData(close=full.close.iloc[offset:],
                        open_=full.open_.iloc[offset:],
                        spy=full.spy, vix=full.vix)
        v = vol.iloc[offset:]
        tm = tradable_mask(sub.close, v, args.min_adv, args.min_price,
                           LiquidityConfig())
        tr = build_terms(sub.close, rcfg)
        sc = composite(tr, replace(rcfg, turn_weight=-1.0, strength_weight=0.0,
                                   room_weight=1.0), "flip")
        return sub, v, apply_tradable(sc, tm, defensive)

    rows = []
    print(f"\n  {'tranches':>9}{'phases':>8}{'CAGR med':>10}{'range':>18}"
          f"{'spread':>9}{'Sharpe':>8}{'MaxDD':>9}{'turn':>8}{'names':>7}")
    print("  " + "-" * 92)

    for k in args.tranches:
        step = max(1, args.hold // k)
        # A k-tranche book still has a phase: WHERE the whole set starts. There
        # are `step` distinct configurations, and the spread across them is what
        # is left of the phase bet after tranching.
        # Sampled, not exhaustive: estimating the spread needs a handful of
        # starting phases, not all of them, and each one is a full-history run
        # with the correlation filter engaged at every rebalance.
        all_starts = list(range(step))
        if args.max_phases and len(all_starts) > args.max_phases:
            idx = np.linspace(0, len(all_starts) - 1, args.max_phases)
            starts = [all_starts[int(round(i))] for i in idx]
        else:
            starts = all_starts
        lcfg_k = LiquidityConfig(account_notional=args.account)
        # Each sleeve's order is account/(k*top_n), so the cost model is given
        # the effective divisor rather than the single-book one.
        slip_k = slippage_panel(full.close, vol, lcfg_k,
                                top_n=args.top_n * k)

        cagrs, sharpes, dds, turns, names_held = [], [], [], [], []
        for s in starts:
            streams, turn_sum, uni = [], 0.0, set()
            for j in range(k):
                off = s + j * step
                if off >= len(full.close):
                    continue
                sub, _v, sc = score_panel(off)
                m, res = run_arm(sc, None, sub, cfg,
                                 slip_k.iloc[off:], return_result=True)
                streams.append(res.returns.rename(f"s{j}"))
                turn_sum += m["Annual Turnover"]
                if len(res.holdings):
                    uni |= set(res.holdings.iloc[-1])
            if not streams:
                continue
            r, m = combine(streams, cfg.execution.risk_free_rate)
            cagrs.append(m["cagr"])
            sharpes.append(m["sharpe_ratio"])
            dds.append(m["max_drawdown"])
            # Turnover per dollar: each sleeve trades 1/k of the capital.
            turns.append(turn_sum / k)
            names_held.append(len(uni))

        if not cagrs:
            continue
        lo, hi = min(cagrs), max(cagrs)
        print(f"  {k:>9}{len(starts):>8}{np.median(cagrs):>10.2%}"
              f"{f'{lo:.2%} to {hi:.2%}':>18}{hi - lo:>9.2%}"
              f"{np.median(sharpes):>8.2f}{np.median(dds):>9.1%}"
              f"{np.median(turns):>8.0%}{int(np.median(names_held)):>7}",
              flush=True)
        rows.append({"Tranches": k, "Phases": len(starts),
                     "CAGR_median": float(np.median(cagrs)),
                     "CAGR_min": lo, "CAGR_max": hi, "Spread": hi - lo,
                     "Sharpe_median": float(np.median(sharpes)),
                     "MaxDD_median": float(np.median(dds)),
                     "Turnover_median": float(np.median(turns)),
                     "Names_median": float(np.median(names_held))})

    out = pd.DataFrame(rows)
    out.to_csv(REPO_ROOT / OUT, index=False)

    # The check that would have caught the slicing bug: a single book MUST show
    # a real phase spread, because that is the whole premise of the exercise.
    if (out["Tranches"] == 1).any():
        s1 = float(out[out["Tranches"] == 1]["Spread"].iloc[0])
        if s1 < 0.02:
            print("\n  *** k=1 shows a phase spread of %.2f%%, which cannot be "
                  "right. ***" % (100 * s1))
            print("  *** The rotation clock is not moving between offsets. "
                  "Refusing to report. ***")
            return 3

    print("\n" + "=" * 104)
    print("WHAT IT BOUGHT, AND WHAT IT COST")
    print("=" * 104)
    if len(out) and (out["Tranches"] == 1).any():
        base_row = out[out["Tranches"] == 1].iloc[0]
        print(f"  Single book (k=1): spread {base_row['Spread']:.2%} of CAGR, "
              f"turnover {base_row['Turnover_median']:.0%}, "
              f"{int(base_row['Names_median'])} names")
        for _, r in out[out["Tranches"] > 1].iterrows():
            print(f"\n  k={int(r['Tranches'])}: spread "
                  f"{r['Spread']:.2%} "
                  f"({r['Spread'] / base_row['Spread']:.0%} of single-book), "
                  f"median CAGR {r['CAGR_median']:+.2%} "
                  f"({r['CAGR_median'] - base_row['CAGR_median']:+.2%} vs k=1)")
            print(f"       turnover {r['Turnover_median']:.0%} "
                  f"({r['Turnover_median'] - base_row['Turnover_median']:+.0%}), "
                  f"Sharpe {r['Sharpe_median']:.2f} "
                  f"({r['Sharpe_median'] - base_row['Sharpe_median']:+.2f}), "
                  f"{int(r['Names_median'])} names")

    print("\n  READ IT THIS WAY. Tranching is worth doing if the spread")
    print("  collapses while median CAGR, Sharpe and turnover per dollar are")
    print("  roughly unchanged. A median CAGR near the k=1 median is the POINT,")
    print("  not a disappointment: you are buying the average instead of")
    print("  gambling on the draw. The cost is position count.")
    print(f"\n  Exported to {REPO_ROOT / OUT}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
