"""
Out-of-sample: does the new score beat the live one in windows it was not chosen on?

The phase test asks whether the result survives an arbitrary implementation
choice.  This asks the harder question — whether the advantage shows up
repeatedly through time, in windows selected without reference to the outcome.

WHY THIS IS A CLEAN TEST HERE, WHICH IT USUALLY IS NOT

Walk-forward normally has to worry about leakage, because the thing being
validated is a parameter fitted on the training window.  **Neither score here
has a fitted parameter.**  The live composite's settings were fixed in 2026-07
and the new score's windows (252/21/63) are literature conventions that have
never been tuned to this data.  So splitting the record into windows leaks
nothing: both scores were fully specified before any window was looked at.

That makes the head-to-head about as close to out-of-sample as is possible
without waiting for new data.

TWO QUESTIONS, DELIBERATELY SEPARATED

  1. HEAD TO HEAD.  In each non-overlapping test window, did the new score beat
     the live one?  Reported as a fraction of windows and a t-statistic on the
     per-window excess.  This is the cross-validation analogue.

  2. SELECTION (TODO 0f).  Would picking between the two scores on trailing
     performance beat simply fixing one?  Run through the repo's existing
     `walk_forward`, which already found that trailing-window selection over
     PARAMETERS beat a fixed config only 13% of the time and cost 3pp.  If
     switching between scores fails the same way, 0f closes cheaply.

PHASE IS POOLED, NOT IGNORED

A single rotation phase would make every window inherit the same phase luck —
and phase is worth 10.19pp of CAGR at `hold=42`.  So the whole thing is run at
several offsets and the windows are pooled, which is also why the window count
is larger than the calendar suggests.

Run:  python scripts/analyze_walkforward_scores.py
      python scripts/analyze_walkforward_scores.py --phases 5 --window-months 12
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
from momentum.experiments import ExperimentResult, production_config
from momentum.liquidity import (LiquidityConfig, apply_tradable,
                                slippage_panel, tradable_mask)
from momentum.metrics import calculate_performance_metrics
from momentum.reversal import ReversalConfig, build_terms, composite
from momentum.strategy import compute_scores
from momentum.universe import defensive_symbols
from momentum.validation import print_walk_forward, walk_forward

from scripts.analyze_reversal_backtest import (equal_weight_arm, load_volume,
                                               run_arm)
from scripts.analyze_reversal_ic import NON_EQUITY, load_pool_panel, pool_symbols

OUT = "rsi_ma_walkforward_scores.csv"
OUT_WINDOWS = "rsi_ma_walkforward_windows.csv"
POOL_FILE = "random_pool.csv"


ORIGIN = pd.Timestamp("2010-01-01")


def window_returns(returns: pd.Series, months: int,
                   origin: pd.Timestamp = ORIGIN) -> pd.DataFrame:
    """
    Total return within each non-overlapping calendar window.

    Binned from a FIXED origin, not with `pd.Grouper(freq='6MS')`.  Grouper
    anchors its bins to each series' own first timestamp, and these arms do not
    start on the same date — the new score needs 252+63 sessions of history
    before it can rank, the live composite fewer.  So Grouper gave each arm a
    different set of bin edges, the window labels never matched between arms,
    and every head-to-head comparison silently had zero rows to compare.  It
    printed 'nan%' rather than failing, which is the expensive kind of wrong.

    A fixed origin makes the bins identical for every arm by construction.
    """
    if returns.empty:
        return pd.DataFrame()

    idx = pd.DatetimeIndex(returns.index)
    bucket = ((idx.year - origin.year) * 12 + (idx.month - origin.month)) // months
    rows = []
    for b, seg in returns.groupby(bucket):
        seg = seg.dropna()
        if len(seg) < 20:
            continue
        total = origin.month - 1 + int(b) * months
        label = pd.Timestamp(year=origin.year + total // 12,
                             month=total % 12 + 1, day=1)
        rows.append({"Start": label, "Days": len(seg),
                     "Return": float((1 + seg).prod() - 1)})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Out-of-sample comparison of the two scores")
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--hold", type=int, default=42)
    p.add_argument("--phases", type=int, default=3,
                   help="rotation offsets to pool over")
    p.add_argument("--window-months", type=int, default=6)
    p.add_argument("--train-years", type=float, default=3.0)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--min-adv", type=float, default=10e6)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--account", type=float, default=100_000.0)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    cfg = replace(production_config(), top_n=args.top_n, hold_days=args.hold,
                  vix=None)
    rcfg = ReversalConfig()
    lcfg = LiquidityConfig(account_notional=args.account)
    defensive = set(defensive_symbols())

    print("=" * 100)
    print("OUT-OF-SAMPLE: new score vs live composite")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"top_n {args.top_n}   hold {args.hold}d   {args.phases} phase(s) "
          f"pooled   {args.window_months}-month windows")
    print("=" * 100)
    print("  Neither score has a parameter fitted on this data, so splitting")
    print("  the record into windows leaks nothing. Both were fully specified")
    print("  before any window was looked at.")

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

    offsets = [int(round(i * args.hold / args.phases))
               for i in range(args.phases)]
    print(f"\n  phases sampled at offsets {offsets}")

    all_windows, streams = [], {}

    for off in offsets:
        sub = PriceData(close=full.close.iloc[off:], open_=full.open_.iloc[off:],
                        spy=full.spy, vix=full.vix)
        v = vol.iloc[off:]

        ranking, base, _ = compute_scores(sub, cfg)
        terms = build_terms(sub.close, rcfg)
        pull = composite(terms, replace(rcfg, turn_weight=-1.0,
                                        strength_weight=0.0, room_weight=1.0),
                         "flip").reindex_like(ranking)
        flip = (-terms["flip"]).reindex_like(ranking)

        tmask = tradable_mask(sub.close, v, args.min_adv, args.min_price, lcfg)
        ranking = apply_tradable(ranking, tmask, defensive)
        pull = apply_tradable(pull, tmask, defensive)
        flip = apply_tradable(flip, tmask, defensive)
        slip = slippage_panel(sub.close, v, lcfg, top_n=cfg.top_n)

        _, live_res = run_arm(ranking, None, sub, cfg, slip, return_result=True)
        _, flip_res = run_arm(flip, None, sub, cfg, slip, return_result=True)
        _, pull_res = run_arm(pull, None, sub, cfg, slip, return_result=True)
        eq, eq_res = equal_weight_arm(
            sub, cfg, ranking,
            [x for x in ranking.columns if x not in defensive], slip)

        streams[off] = {"live": live_res.returns, "flip_neg": flip_res.returns,
                        "pullback": pull_res.returns,
                        "equal_weight": eq_res.returns}

        w, counts = {}, {}
        for name, r in streams[off].items():
            wr = window_returns(r, args.window_months)
            counts[name] = len(wr)
            if wr.empty:
                continue
            w[name] = wr.set_index("Start")["Return"]

        # No global dropna.  The arms do not start on the same date — the new
        # score needs 252+63 sessions of history before it can rank, the
        # equal-weight arm needs only `min_data_days` — so a global dropna
        # silently discards every window where any single arm is missing, and
        # if the spans differ enough it discards all of them.  That is exactly
        # what happened on the first run: 0 windows, reported as 'nan%'.
        # Each comparison drops its own missing rows below instead.
        wdf = pd.DataFrame(w)
        wdf["Offset"] = off
        all_windows.append(wdf.reset_index(names="Start"))
        print(f"    offset {off:>3}: windows per arm {counts}", flush=True)

    W = pd.concat(all_windows, ignore_index=True)
    W.to_csv(REPO_ROOT / OUT_WINDOWS, index=False)

    print("\n" + "=" * 100)
    print(f"HEAD TO HEAD  -  {len(W)} pooled {args.window_months}-month windows")
    print("=" * 100)

    rows = []
    for challenger in ("flip_neg", "pullback"):
        for baseline in ("live", "equal_weight"):
            d = (W[challenger] - W[baseline]).dropna()
            n = len(d)
            if n == 0:
                continue
            t = d.mean() / (d.std() / np.sqrt(n)) if d.std() else np.nan
            rows.append({
                "Challenger": challenger, "Baseline": baseline,
                "Windows": n, "WinRate": float((d > 0).mean()),
                "MeanExcess": float(d.mean()), "MedianExcess": float(d.median()),
                "sd": float(d.std()), "t": float(t),
                "Worst": float(d.min()), "Best": float(d.max()),
            })

    hdr = (f"    {'challenger':<12}{'vs':<14}{'windows':>9}{'win rate':>10}"
           f"{'mean':>9}{'median':>9}{'t':>7}{'worst':>9}{'best':>9}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for r in rows:
        star = " *" if abs(r["t"]) >= 2 else ""
        print(f"    {r['Challenger']:<12}{r['Baseline']:<14}{r['Windows']:>9}"
              f"{r['WinRate']:>10.0%}{r['MeanExcess']:>9.2%}"
              f"{r['MedianExcess']:>9.2%}{r['t']:>7.2f}"
              f"{r['Worst']:>9.2%}{r['Best']:>9.2%}{star}")
    print("\n  Windows overlap across phases (the same calendar period appears")
    print("  once per offset), so the t-statistics are optimistic. Read the win")
    print("  rate and the worst window, which do not depend on that.")

    pd.DataFrame(rows).to_csv(REPO_ROOT / OUT, index=False)

    # --- window-by-window, one row per calendar period ---
    print("\n" + "=" * 100)
    print("BY CALENDAR PERIOD  (averaged over phases)")
    print("=" * 100)
    cols = [c for c in ("live", "flip_neg", "pullback", "equal_weight")
            if c in W.columns]
    byp = W.groupby("Start")[cols].mean()
    byp = byp.dropna(subset=[c for c in ("live", "flip_neg") if c in cols])
    byp["flip - live"] = byp["flip_neg"] - byp["live"]
    print(f"    {'window':<12}{'live':>9}{'flip_neg':>10}{'pullback':>10}"
          f"{'eq-wt':>9}{'flip-live':>11}")
    print("    " + "-" * 61)
    for start, r in byp.iterrows():
        print(f"    {start:%Y-%m}     {r['live']:>8.1%}{r['flip_neg']:>10.1%}"
              f"{r['pullback']:>10.1%}{r['equal_weight']:>9.1%}"
              f"{r['flip - live']:>+11.1%}")
    print(f"\n    flip_neg beat live in {int((byp['flip - live'] > 0).sum())} "
          f"of {len(byp)} calendar windows")

    # --- selection: TODO 0f ---
    print("\n" + "=" * 100)
    print("SELECTION BETWEEN SCORES  (TODO 0f)")
    print("=" * 100)
    print("  Would picking on trailing performance beat fixing one score?")
    print("  Walk-forward already found trailing selection over PARAMETERS beat")
    print("  a fixed config 13% of the time and cost 3pp. Same test, two scores.")

    off0 = offsets[0]
    fake = []
    for name in ("live", "flip_neg", "pullback"):
        r = streams[off0][name]
        m = calculate_performance_metrics(r)
        fake.append(ExperimentResult(
            name=name, config=cfg,
            result=type("R", (), {"returns": r, "metrics": m,
                                  "turnover": {"annual_turnover": np.nan}})(),
            note=""))
    try:
        wf = walk_forward(fake, train_years=args.train_years,
                          test_months=args.window_months)
        print_walk_forward(wf)
    except Exception as exc:
        print(f"  (walk_forward unavailable on these inputs: "
              f"{type(exc).__name__}: {exc})")
        print("  The head-to-head above is the part that matters; selection is "
              "a secondary question.")

    print(f"\n  Exported to {REPO_ROOT / OUT} and {REPO_ROOT / OUT_WINDOWS}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
