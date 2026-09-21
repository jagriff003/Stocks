"""
Why does the production backtest say +18% YTD when the account's realised result was far below it?

A divergence that size on the model actually being traded is larger than every
effect measured in the Track J work, and it decides whether any backtest delta
translates into the account at all.  Until it is explained, "the new model beats
the old one by 8pp in backtest" is a statement about a simulation.

TWO QUESTIONS, AND THEY NEED DIFFERENT EVIDENCE

  IS IT TRUE?   Reproduce the backtest figure, then decompose it.  A number that
                cannot be decomposed into named positions is not yet a fact.

  WHY?          Attribution.  The leading suspect is not execution -- it is that
                the backtest ranks TODAY's universe over the WHOLE year.  The
                universe was updated on 2026-09-17, replacing about a dozen
                names.  Every name added in September is in the January book of
                the backtest and was never in the account.  That is look-ahead
                at the universe level, and on a 4-name book it can be worth
                double-digit percentages in a year.

WHAT THIS MEASURES

  1. YTD for the CURRENT universe (what the backtest reports).
  2. YTD for the PRE-UPDATE universe (what was actually held until September),
     from `snapshots/universe/universe_2026-09-17_pre-update.csv`.
     The difference is the universe-change effect, isolated.
  3. Per-name contribution in both, so the specific wins and losses are named
     rather than inferred.
  4. Which held names were added in the September update -- those are the ones
     the backtest could not have known about.

WHAT IT CANNOT MEASURE

Execution.  This has no record of actual fills, so it cannot separate "traded
late" from "traded a different name".  What it CAN do is bound how much of the
gap is explained by the universe alone, which tells you whether to go looking at
execution at all.  That is TODO item 2 and it needs a trade log.

Run:  python scripts/analyze_ytd_gap.py
      python scripts/analyze_ytd_gap.py --from 2026-01-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.data import load_data
from momentum.experiments import production_config
from momentum.metrics import calculate_performance_metrics
from momentum.strategy import run_strategy
from momentum.universe import current_symbols

OUT = "rsi_ma_ytd_gap.csv"
PRE_UPDATE = "snapshots/universe/universe_2026-09-17_pre-update.csv"


def read_snapshot_symbols(path: Path):
    d = pd.read_csv(path)
    return sorted(d[d["Active"] == "Y"]["Symbol"].tolist())


def contributions(result, close, start):
    """
    Per-name contribution to the period return.

    Equal weight within the book on each day, so a name's contribution on a
    given day is its return divided by the number of names held.  Summed over
    the window this is an arithmetic approximation to its share of the
    compounded result -- close enough to rank the winners and losers, which is
    what it is for, and the total is printed against the true compounded figure
    so the size of the approximation is visible rather than assumed.
    """
    rets = close.pct_change()
    rows = {}
    for d, names in result.holdings.items():
        if d < pd.Timestamp(start) or not len(names):
            continue
        if d not in rets.index:
            continue
        w = 1.0 / len(names)
        for n in names:
            if n in rets.columns and pd.notna(rets.at[d, n]):
                rows[n] = rows.get(n, 0.0) + w * float(rets.at[d, n])
    s = pd.Series(rows, dtype=float).sort_values()
    return s


def days_held(result, start):
    out = {}
    for d, names in result.holdings.items():
        if d < pd.Timestamp(start):
            continue
        for n in names:
            out[n] = out.get(n, 0) + 1
    return pd.Series(out, dtype=int)


def main() -> int:
    p = argparse.ArgumentParser(description="Explain the YTD backtest gap")
    p.add_argument("--from", dest="start_ytd", default="2026-01-01")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    cfg = production_config()

    print("=" * 96)
    print("YTD GAP — why the backtest and the account disagree")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}   window from "
          f"{args.start_ytd}")
    print("=" * 96)

    now_syms = current_symbols()
    pre_path = REPO_ROOT / PRE_UPDATE
    if not pre_path.exists():
        print(f"  pre-update snapshot missing: {pre_path}")
        return 1
    pre_syms = read_snapshot_symbols(pre_path)

    added = sorted(set(now_syms) - set(pre_syms))
    removed = sorted(set(pre_syms) - set(now_syms))
    print(f"\n  Universe changed 2026-09-17:")
    print(f"    {len(pre_syms)} names before, {len(now_syms)} after")
    print(f"    added   ({len(added)}): {', '.join(added) if added else '-'}")
    print(f"    removed ({len(removed)}): {', '.join(removed) if removed else '-'}")

    arms = {}
    for label, syms in (("current universe (what the backtest uses)", now_syms),
                        ("pre-update universe (what was held until Sept)",
                         pre_syms)):
        prices = load_data(syms, start_date=args.start,
                           use_cache=not args.no_cache,
                           cache_max_age_hours=1e9, verbose=False)
        res = run_strategy(prices, cfg)
        arms[label] = (res, prices)

    print("\n" + "=" * 96)
    print("IS IT TRUE?  — the same model, two universes")
    print("=" * 96)
    hdr = f"    {'universe':<46}{'YTD':>10}{'since Jul':>11}{'full CAGR':>11}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    rows = []
    for label, (res, _) in arms.items():
        r = res.returns.dropna()
        ytd = (1 + r.loc[args.start_ytd:]).prod() - 1
        jul = (1 + r.loc["2026-07-01":]).prod() - 1
        cagr = calculate_performance_metrics(r)["cagr"]
        print(f"    {label:<46}{ytd:>10.2%}{jul:>11.2%}{cagr:>11.2%}")
        rows.append({"Universe": label, "YTD": ytd, "SinceJul": jul,
                     "CAGR": cagr})

    a = rows[0]["YTD"]
    b = rows[1]["YTD"]
    print(f"\n    Universe-change effect on YTD: {a - b:+.2%}")
    print("    That is the part of the backtest's YTD that comes from ranking")
    print("    names which were only added to the universe in September.")

    print("\n" + "=" * 96)
    print("WHY?  — named winners and losers, current universe")
    print("=" * 96)
    res, prices = arms["current universe (what the backtest uses)"]
    contrib = contributions(res, prices.close, args.start_ytd)
    held = days_held(res, args.start_ytd)
    total = float(contrib.sum())
    true_ytd = (1 + res.returns.dropna().loc[args.start_ytd:]).prod() - 1
    print(f"    contributions sum to {total:+.2%} against a compounded "
          f"{true_ytd:+.2%}")
    print(f"    (arithmetic approximation; gap {total - true_ytd:+.2%})")

    added_set = set(added)
    print(f"\n    {'name':<8}{'contrib':>10}{'days held':>11}   note")
    print("    " + "-" * 46)
    for n in list(contrib.tail(args.top).index)[::-1]:
        note = "ADDED IN SEPT — not in the account earlier" \
            if n in added_set else ""
        print(f"    {n:<8}{contrib[n]:>+10.2%}{held.get(n, 0):>11}   {note}")
    print("    ...")
    for n in list(contrib.head(args.top).index):
        note = "ADDED IN SEPT — not in the account earlier" \
            if n in added_set else ""
        print(f"    {n:<8}{contrib[n]:>+10.2%}{held.get(n, 0):>11}   {note}")

    from_added = float(contrib.reindex(sorted(added_set)).dropna().sum())
    print(f"\n    Total contribution from names added in September: "
          f"{from_added:+.2%}")
    print(f"    Share of the backtest's YTD: "
          f"{from_added / true_ytd:.0%}" if true_ytd else "")

    out = pd.DataFrame({"Contribution": contrib,
                        "DaysHeld": held.reindex(contrib.index).fillna(0),
                        "AddedSept": [n in added_set for n in contrib.index]})
    out.to_csv(REPO_ROOT / OUT)

    print("\n" + "=" * 96)
    print("WHAT IS LEFT UNEXPLAINED")
    print("=" * 96)
    print("  Whatever the universe does not account for is execution, timing,")
    print("  sizing or cash drag, and NONE of it is measurable from here --")
    print("  there is no trade log in this repo. That is TODO item 2, and it")
    print("  needs actual fills to close.")
    print(f"\n  Exported to {REPO_ROOT / OUT}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
