"""
How many names should the book hold?

Sweeps `top_n` while holding the volatility overlay fixed, which is the only
honest way to read the result: `vix.elevated_top_n` stays at its live value of 2
across every book size, so an elevated-VIX regime always keeps the same *number*
of momentum names and the defensive sleeve absorbs whatever slots remain.  That
means the defensive SHARE of the book is not constant across the sweep - it
cannot be, because only three defensive tickers exist (SHY/TLT/IAU) and the fill
is hard-capped at three names:

    top_n   normal book   elevated book             defensive share (elevated)
      1         1          1 mom + 0 def                       0%
      2         2          2 mom + 0 def                       0%
      3         3          2 mom + 1 def                      33%
      4         4          2 mom + 2 def                      50%   <- live
      5         5          2 mom + 3 def                      60%
      6         6          2 mom + 3 def (capped)             50%
      7         7          2 mom + 3 def (capped)             43%
      8         8          2 mom + 3 def (capped)             38%

Read that table before reading the results.  Book size and elevated-regime risk
posture move together here, by the choice to hold `elevated_top_n` fixed; the
REGIME EXPOSURE block below reports the realized defensive share per book size
so the confound is visible rather than implied.  Pass --scale-elevated to run
the ratio-preserving alternative instead.

CRISIS is invariant to `top_n` by construction - it holds `crisis_symbols`
(SHY, TLT) outright - so nothing in this sweep changes the crisis book.

Every threshold is a flag.  Nothing below is hard-coded to the live values.

Run:  python scripts/analyze_book_size.py
      python scripts/analyze_book_size.py --min-n 1 --max-n 8 --elevated-top-n 2
      python scripts/analyze_book_size.py --scale-elevated
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
from momentum.experiments import (Experiment, comparison_table,
                                  export_comparison, print_comparison,
                                  print_subperiods, production_config,
                                  run_experiments)
from momentum.universe import current_symbols, defensive_symbols

OUT_FILE = "rsi_ma_book_size{suffix}.csv"
OUT_EXPOSURE = "rsi_ma_book_size_exposure{suffix}.csv"


def build_sweep(min_n: int, max_n: int, elevated_top_n: int,
                scale_elevated: bool) -> list:
    """One config per book size, overlay held per the chosen policy."""
    experiments = []
    for n in range(min_n, max_n + 1):
        if scale_elevated:
            # Ratio-preserving alternative: keep the live 2-of-4 proportion.
            elev = max(1, int(round(n * elevated_top_n / 4)))
        else:
            # The default: the overlay keeps a FIXED number of momentum names,
            # clamped to the book so a 1-name book cannot grow to 2 in a regime
            # whose whole purpose is to cut exposure.
            elev = min(elevated_top_n, n)

        experiments.append(Experiment(
            name=f"top_n={n}",
            config=production_config(
                top_n=n,
                vix__elevated_top_n=elev,
                notes=f"book size sweep: top_n={n}, elevated_top_n={elev}",
            ),
            note=f"elev {elev} mom + up to {n - elev} def",
        ))
    return experiments


def exposure_table(results, defensive: list) -> pd.DataFrame:
    """
    What each book size actually held, by regime.

    `top_n` is a target, not an outcome.  The eligibility gate drops a date
    entirely when fewer than `top_n` names have enough history, the defensive
    fill is capped at three tickers, and CRISIS ignores `top_n` outright.  The
    realized mean book size is therefore the number to compare against, and the
    defensive share is the confound this sweep cannot design away.
    """
    defensive_set = set(defensive)
    rows = []
    for r in results:
        holdings = r.result.holdings.dropna()
        if holdings.empty:
            continue
        sizes = holdings.apply(len)
        def_count = holdings.apply(lambda h: sum(1 for s in h if s in defensive_set))
        def_share = def_count / sizes.replace(0, np.nan)

        rows.append({
            "Experiment": r.name,
            "Target N": r.config.top_n,
            "Elevated Top N": r.config.vix.elevated_top_n,
            "Mean Book Size": float(sizes.mean()),
            "Min Book Size": int(sizes.min()),
            "Max Book Size": int(sizes.max()),
            "Days Any Defensive": float((def_count > 0).mean()),
            "Mean Defensive Share": float(def_share.mean()),
            "Days All Defensive": float((def_share >= 1.0).mean()),
        })
    return pd.DataFrame(rows)


def print_exposure(table: pd.DataFrame) -> None:
    if table.empty:
        return
    print("\n" + "=" * 96)
    print("REGIME EXPOSURE  - what each book size actually held")
    print("=" * 96)
    header = (f"{'Experiment':<14}{'ElevN':>7}{'MeanSize':>10}{'Range':>9}"
              f"{'AnyDef%':>10}{'MeanDef%':>10}{'AllDef%':>10}")
    print(header)
    print("-" * len(header))
    for _, row in table.iterrows():
        rng = f"{row['Min Book Size']}-{row['Max Book Size']}"
        print(f"{row['Experiment']:<14}"
              f"{row['Elevated Top N']:>6.0f} "
              f"{row['Mean Book Size']:>9.2f} "
              f"{rng:>8} "
              f"{row['Days Any Defensive']:>9.1%} "
              f"{row['Mean Defensive Share']:>9.1%} "
              f"{row['Days All Defensive']:>9.1%}")
    print("-" * len(header))
    print("  AnyDef%  share of days holding at least one of SHY/TLT/IAU")
    print("  MeanDef% mean defensive weight of the book (equal-weighted)")
    print("  AllDef%  share of days fully defensive (crisis, or an all-fill book)")


def print_efficiency(table: pd.DataFrame, baseline: str) -> None:
    """
    The question actually asked: the SMALLEST book that gets the CAGR.

    Reported as CAGR per trade-per-year and as the CAGR give-up against the best
    book size, because 'maximize CAGR' and 'minimize trades' are different
    objectives and the trade-off between them is the decision.
    """
    if table.empty:
        return
    best = table.loc[table["CAGR"].idxmax()]
    cheapest = table.loc[table["Trades/Year"].idxmin()]

    print("\n" + "=" * 96)
    print("COST OF SIZE  - CAGR give-up against the best book, per unit of trading")
    print("=" * 96)
    header = (f"{'Experiment':<14}{'CAGR':>9}{'vs best':>10}{'Trd/Yr':>9}"
              f"{'Turnover':>10}{'CAGR/Trd':>10}{'Sharpe':>9}{'MaxDD':>10}")
    print(header)
    print("-" * len(header))
    for _, row in table.iterrows():
        per_trade = (row["CAGR"] / row["Trades/Year"]
                     if row["Trades/Year"] > 0 else np.nan)
        mark = ""
        if row["Experiment"] == best["Experiment"]:
            mark = "  <- best CAGR"
        elif row["Experiment"] == cheapest["Experiment"]:
            mark = "  <- fewest trades"
        elif row["Experiment"] == baseline:
            mark = "  <- live"
        print(f"{row['Experiment']:<14}"
              f"{row['CAGR']:>8.2%} "
              f"{row['CAGR'] - best['CAGR']:>+9.2%} "
              f"{row['Trades/Year']:>8.1f} "
              f"{row['Annual Turnover']:>9.1%} "
              f"{per_trade:>9.3%} "
              f"{row['Sharpe']:>8.2f} "
              f"{row['Max Drawdown']:>9.2%}"
              f"{mark}")
    print("=" * 96)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep book size (top_n) with the volatility overlay held fixed")
    parser.add_argument("--min-n", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=8)
    parser.add_argument("--elevated-top-n", type=int, default=2,
                        help="momentum names kept in an ELEVATED regime "
                             "(live value 2); clamped to top_n")
    parser.add_argument("--scale-elevated", action="store_true",
                        help="scale elevated_top_n with book size instead of "
                             "holding it fixed, preserving the live 2-of-4 ratio")
    parser.add_argument("--baseline", default="top_n=4",
                        help="the live configuration, for deltas")
    parser.add_argument("--subperiods", type=int, default=3)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--keep-unsettled", action="store_true",
                        help="keep an unclosed final session; excluded by "
                             "default so the sweep is reproducible")
    args = parser.parse_args()

    print("=" * 96)
    print("BOOK SIZE SWEEP")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"top_n {args.min_n}..{args.max_n}   "
          f"elevated_top_n {'scaled' if args.scale_elevated else 'fixed'} "
          f"at {args.elevated_top_n}")
    print("=" * 96)

    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache,
                       drop_unsettled=not args.keep_unsettled)

    experiments = build_sweep(args.min_n, args.max_n,
                              args.elevated_top_n, args.scale_elevated)
    results = run_experiments(experiments, prices)

    table = comparison_table(results)
    exposure = exposure_table(results, defensive_symbols())

    print_comparison(results, baseline=args.baseline)
    print_exposure(exposure)
    print_efficiency(table, args.baseline)
    print_subperiods(results, n_periods=args.subperiods)

    # The two overlay policies are different experiments and must not
    # overwrite each other's record.
    suffix = "_scaled" if args.scale_elevated else ""
    export_comparison(results, str(REPO_ROOT / OUT_FILE.format(suffix=suffix)))
    exposure.to_csv(REPO_ROOT / OUT_EXPOSURE.format(suffix=suffix), index=False)
    print(f"Exposure exported to: {REPO_ROOT / OUT_EXPOSURE.format(suffix=suffix)}")

    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
