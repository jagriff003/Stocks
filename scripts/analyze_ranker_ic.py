"""
Does the composite score carry cross-sectional information at all?

The mechanism half of the null-portfolio question.  `analyze_null_benchmark.py`
measures what ranking is *worth* in CAGR; this measures whether the thing doing
the ranking predicts anything, which is the more basic question and has never
been asked in this repo.

Information coefficient: on each date, the cross-sectional Spearman correlation
between the composite score and the forward return over a horizon.  Mean IC is
the headline; the decile spread is the same fact in units anyone can act on.

THREE THINGS THIS IS BUILT TO AVOID GETTING WRONG

1. Overlapping windows.  Daily dates with h-day forward returns share h-1 days
   of data, so a naive t-statistic over every date is inflated by roughly
   sqrt(h).  FINDINGS already records this trap for the health monitor's
   126-day windows.  The headline t-stat here is therefore computed on
   NON-OVERLAPPING dates (every h-th), and the naive figure is printed beside it
   so the size of the inflation stays visible rather than implied.

2. The wrong pool.  IC is measured only over names the model could actually
   have picked on that date - the history gate and the `min_level_threshold`
   floor both applied - because a correlation computed over ineligible names
   describes a portfolio nobody can hold.

3. The wrong prices.  Forward returns run open-to-open by default, matching the
   `next_open` fill convention, so the measured relationship is one the model
   could have traded.  `--price close` gives the conventional close-to-close
   version for comparison.

WHAT THE HORIZONS ARE FOR
   The live hold is 14 days.  The classic cross-sectional momentum literature
   runs on much longer formation and holding periods, and at the short end finds
   reversal rather than momentum.  Running 5 through 126 days separates "this
   signal does not work" from "this signal works, but not at 14 days" - which
   are very different findings with very different responses.

   The level-only score is scored alongside the velocity-blended one, because
   `velocity_window=5` was an in-sample selection worth a reported +6.22pp and
   has never been checked for whether it improves prediction or just fitted the
   sample.

Every threshold is a flag.

Run:  python scripts/analyze_ranker_ic.py
      python scripts/analyze_ranker_ic.py --horizons 5 14 63 --price close
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
from momentum.strategy import compute_scores
from momentum.universe import current_symbols, defensive_symbols

OUT_IC = "rsi_ma_ranker_ic.csv"
OUT_DECILE = "rsi_ma_ranker_decile.csv"
OUT_TOPK = "rsi_ma_ranker_topk.csv"


def eligible_mask(ranking_scores, base_scores, cfg, drop: set) -> pd.DataFrame:
    """
    Names the model could actually have picked, per date.

    Mirrors `_PortfolioBuilder.eligible`: enough history, and a level score above
    the freefall floor.  Defensive tickers are excluded - they enter the book by
    regime rule rather than on rank, so scoring them would measure the overlay,
    not the ranker.
    """
    counts = ranking_scores.notna().cumsum()
    mask = ranking_scores.notna() & (counts >= cfg.min_data_days)

    if cfg.velocity is not None:
        floor = cfg.velocity.min_level_threshold
        mask &= base_scores.reindex_like(ranking_scores).ge(floor)

    for sym in drop:
        if sym in mask.columns:
            mask[sym] = False
    return mask


def forward_returns(prices, horizon: int, price_field: str) -> pd.DataFrame:
    """
    Return earned over the next `horizon` sessions, on the fill convention.

    'open' measures open(T+1) -> open(T+1+h): the model signals on close(T) and
    fills at the next open, so this is the return it could actually capture.
    'close' is the conventional close(T) -> close(T+h) version.
    """
    panel = prices.open_ if price_field == "open" else prices.close
    if price_field == "open":
        panel = panel.shift(-1)
    return panel.shift(-horizon) / panel - 1


def row_spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Per-date cross-sectional Spearman, as Pearson on cross-sectional ranks."""
    ra = a.rank(axis=1)
    rb = b.rank(axis=1)
    return ra.corrwith(rb, axis=1)


def ic_stats(ic: pd.Series, horizon: int) -> dict:
    """
    Mean IC with an overlap-aware t-statistic.

    The headline t uses every h-th date, so no two observations share a forward
    window.  `t (naive)` uses all of them and is reported only to show how much
    the overlap would have flattered the result.
    """
    ic = ic.dropna()
    if ic.empty:
        return {}
    indep = ic.iloc[::horizon]
    n = len(indep)
    t_indep = (indep.mean() / (indep.std() / np.sqrt(n))) if n > 1 and indep.std() else np.nan
    t_naive = (ic.mean() / (ic.std() / np.sqrt(len(ic)))) if ic.std() else np.nan
    return {
        "Horizon": horizon,
        "Mean IC": float(ic.mean()),
        "Median IC": float(ic.median()),
        "IC sd": float(ic.std()),
        "Hit Rate": float((ic > 0).mean()),
        "N indep": n,
        "t": float(t_indep),
        "t (naive)": float(t_naive),
        "N dates": len(ic),
    }


def decile_table(scores, fwd, mask, horizon: int, n_bins: int) -> pd.DataFrame:
    """Mean forward return by cross-sectional score bin, best bin first."""
    s = scores.where(mask)
    f = fwd.where(mask & fwd.notna())

    # Rank to [0,1) per date, then bin; equal-count bins per date regardless of
    # how many names are eligible that day.
    pct = s.rank(axis=1, pct=True)
    bins = np.ceil(pct * n_bins).clip(1, n_bins)

    rows = []
    for b in range(1, n_bins + 1):
        sel = f.where(bins == b)
        vals = sel.stack().dropna()
        if vals.empty:
            continue
        rows.append({
            "Horizon": horizon,
            "Bin": b,
            "Label": f"D{b}" + (" (worst)" if b == 1 else
                                " (best)" if b == n_bins else ""),
            "Mean Fwd Return": float(vals.mean()),
            "Median Fwd Return": float(vals.median()),
            "Hit Rate": float((vals > 0).mean()),
            "Observations": int(vals.size),
        })
    return pd.DataFrame(rows).sort_values("Bin", ascending=False)


def print_ic(table: pd.DataFrame, label: str) -> None:
    print(f"\n  {label}")
    header = (f"    {'Horizon':>9}{'Mean IC':>11}{'IC sd':>9}{'Hit':>8}"
              f"{'t':>9}{'t (naive)':>12}{'N indep':>10}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for _, r in table.iterrows():
        flag = ""
        if abs(r["t"]) >= 2.0:
            flag = "  *" if r["t"] > 0 else "  * (negative)"
        print(f"    {r['Horizon']:>9.0f}{r['Mean IC']:>11.4f}{r['IC sd']:>9.3f}"
              f"{r['Hit Rate']:>8.1%}{r['t']:>9.2f}{r['t (naive)']:>12.2f}"
              f"{r['N indep']:>10.0f}{flag}")


def topk_table(scores, fwd, mask, horizon: int, ks) -> pd.DataFrame:
    """
    What the top K by score actually earned, against the eligible-pool average.

    This is the cut the quintile ladder cannot make.  The model holds 4 names
    out of ~41 eligible - the top 10% - so a flat quintile ladder (top 20%) is
    consistent with a signal that works only at the very top and is invisible
    once averaged over eight names.  `pick - pool` is the per-period edge the
    ranker delivers on the trade it actually makes.
    """
    s = scores.where(mask)
    f = fwd.where(mask & fwd.notna())
    rank = s.rank(axis=1, ascending=False, method="first")

    pool = f.stack().dropna()
    rows = []
    for k in ks:
        vals = f.where(rank <= k).stack().dropna()
        if vals.empty:
            continue
        rows.append({
            "Horizon": horizon,
            "K": k,
            "Mean Fwd Return": float(vals.mean()),
            "Pool Mean": float(pool.mean()),
            "Edge": float(vals.mean() - pool.mean()),
            "Hit Rate": float((vals > 0).mean()),
            "Pool Hit Rate": float((pool > 0).mean()),
            "Observations": int(vals.size),
        })
    return pd.DataFrame(rows)


def print_topk(table: pd.DataFrame, hold: int) -> None:
    print("\n" + "=" * 100)
    print("TOP-K CUT  - what the names the model actually buys went on to earn")
    print("=" * 100)
    print("  'Edge' is the top-K mean minus the mean over every eligible name "
          "that date.\n  This is the quantity the strategy monetizes; the "
          "quintile ladder above averages\n  it away by pooling the top pick "
          "with names ranked eighth.")
    for horizon in sorted(table["Horizon"].unique()):
        sub = table[table["Horizon"] == horizon]
        mark = "   <- the live hold" if horizon == hold else ""
        print(f"\n  Horizon {horizon:.0f} days{mark}")
        header = (f"    {'Top K':>7}{'Mean fwd':>11}{'Pool mean':>12}"
                  f"{'Edge':>10}{'Hit':>8}{'Pool hit':>10}{'Obs':>10}")
        print(header)
        print("    " + "-" * (len(header) - 4))
        for _, r in sub.iterrows():
            print(f"    {r['K']:>7.0f}{r['Mean Fwd Return']:>10.2%} "
                  f"{r['Pool Mean']:>11.2%} {r['Edge']:>+9.2%} "
                  f"{r['Hit Rate']:>7.1%} {r['Pool Hit Rate']:>9.1%} "
                  f"{r['Observations']:>9,}")


def print_deciles(table: pd.DataFrame, n_bins: int) -> None:
    for horizon in sorted(table["Horizon"].unique()):
        sub = table[table["Horizon"] == horizon]
        best = sub[sub["Bin"] == n_bins]["Mean Fwd Return"]
        worst = sub[sub["Bin"] == 1]["Mean Fwd Return"]
        spread = (float(best.iloc[0]) - float(worst.iloc[0])
                  if len(best) and len(worst) else np.nan)
        print(f"\n  Horizon {horizon:.0f} days   "
              f"top-minus-bottom spread {spread:+.2%}")
        header = f"    {'Bin':<14}{'Mean fwd':>11}{'Median':>11}{'Hit':>8}{'Obs':>10}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for _, r in sub.iterrows():
            print(f"    {r['Label']:<14}{r['Mean Fwd Return']:>10.2%} "
                  f"{r['Median Fwd Return']:>10.2%} {r['Hit Rate']:>7.1%} "
                  f"{r['Observations']:>9,}")


def print_subperiods(ic: pd.Series, horizon: int, n_periods: int) -> None:
    ic = ic.dropna()
    if ic.empty:
        return
    print(f"\n  Stability of the {horizon}-day IC across {n_periods} segments")
    splits = np.array_split(np.arange(len(ic)), n_periods)
    header = f"    {'Period':<8}{'Start':<13}{'End':<13}{'Mean IC':>11}{'t':>9}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for i, s in enumerate(splits, 1):
        seg = ic.iloc[s[0]:s[-1] + 1]
        st = ic_stats(seg, horizon)
        if not st:
            continue
        print(f"    P{i:<7}{seg.index[0]:%Y-%m-%d}  {seg.index[-1]:%Y-%m-%d}  "
              f"{st['Mean IC']:>10.4f}{st['t']:>9.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Information coefficient of the composite score")
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=[5, 10, 14, 21, 42, 63, 126])
    parser.add_argument("--bins", type=int, default=5,
                        help="cross-sectional score bins (default 5; the "
                             "eligible pool is ~50 names, so deciles would "
                             "hold about 5 names each)")
    parser.add_argument("--price", choices=["open", "close"], default="open",
                        help="forward-return convention; 'open' matches the "
                             "next_open fill")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8],
                        help="book sizes for the top-K cut; the live value is 4")
    parser.add_argument("--subperiods", type=int, default=3)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--keep-unsettled", action="store_true")
    args = parser.parse_args()

    print("=" * 100)
    print("RANKER INFORMATION COEFFICIENT")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"horizons {args.horizons}   bins {args.bins}   "
          f"forward returns {args.price}-to-{args.price}")
    print("=" * 100)

    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache,
                       drop_unsettled=not args.keep_unsettled)

    cfg = production_config()
    ranking_scores, base_scores, _ = compute_scores(prices, cfg)
    mask = eligible_mask(ranking_scores, base_scores, cfg,
                         set(defensive_symbols()))

    print(f"\nEligible pool: {mask.sum(axis=1).mean():.1f} names per date on "
          f"average, {mask.sum(axis=1).min():.0f} to {mask.sum(axis=1).max():.0f}.")
    print("Scored: the velocity-blended ranking score (live) and the "
          "level-only score.")

    panels = {"blended (live)": ranking_scores,
              "level only": base_scores.reindex_like(ranking_scores)}

    print("\n" + "=" * 100)
    print("INFORMATION COEFFICIENT")
    print("=" * 100)
    print("  Cross-sectional Spearman(score, forward return), averaged over "
          "dates.\n  '*' marks |t| >= 2 on non-overlapping dates. Compare t "
          "against t (naive) to\n  see what the overlap correction cost.")

    all_ic, all_dec, ic_series = [], [], {}
    for label, panel in panels.items():
        rows = []
        for h in args.horizons:
            fwd = forward_returns(prices, h, args.price)
            fwd = fwd.reindex_like(panel)
            ic = row_spearman(panel.where(mask), fwd.where(mask))
            ic_series[(label, h)] = ic
            st = ic_stats(ic, h)
            if st:
                st["Panel"] = label
                rows.append(st)
        table = pd.DataFrame(rows)
        all_ic.append(table)
        print_ic(table, label)

    print("\n" + "=" * 100)
    print(f"FORWARD RETURN BY SCORE BIN  ({args.bins} bins, live blended score)")
    print("=" * 100)
    print("  The same fact as the IC, in units you can act on. A monotone "
          "ladder from\n  D1 to D%d is what a working ranker looks like."
          % args.bins)
    for h in args.horizons:
        fwd = forward_returns(prices, h, args.price).reindex_like(ranking_scores)
        all_dec.append(decile_table(ranking_scores, fwd, mask, h, args.bins))
    dec = pd.concat(all_dec)
    print_deciles(dec, args.bins)

    topk = pd.concat([
        topk_table(ranking_scores,
                   forward_returns(prices, h, args.price).reindex_like(ranking_scores),
                   mask, h, args.topk)
        for h in args.horizons])
    print_topk(topk, cfg.hold_days)

    hold = cfg.hold_days
    if ("blended (live)", hold) in ic_series:
        print("\n" + "=" * 100)
        print(f"STABILITY AT THE LIVE HOLD ({hold} days)")
        print("=" * 100)
        print_subperiods(ic_series[("blended (live)", hold)], hold,
                         args.subperiods)

    ic_out = pd.concat(all_ic)
    ic_out.to_csv(REPO_ROOT / OUT_IC, index=False)
    dec.to_csv(REPO_ROOT / OUT_DECILE, index=False)
    topk.to_csv(REPO_ROOT / OUT_TOPK, index=False)
    print(f"\nIC exported to:      {REPO_ROOT / OUT_IC}")
    print(f"Deciles exported to: {REPO_ROOT / OUT_DECILE}")
    print(f"Top-K exported to:   {REPO_ROOT / OUT_TOPK}")

    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
