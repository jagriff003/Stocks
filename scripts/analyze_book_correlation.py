"""
Is the book a portfolio, or one bet held eight ways?

TODO 0h.4.  The correlation filter was configured for a book of 4 drawn from a
46-name universe curated across sectors.  This model picks 8 from ~610, and the
score has no diversification term at all — so nothing stops it loading the whole
book into one industry when that industry happens to be pulling back together.

The 2026 books are the motivating example: `AEHR, CIEN, FORM, VIAV, JBL` and then
`CSCO, SMTC, HLIT, LSCC, TTMI`. That is a semiconductor and optical-networking
bet wearing the clothes of an eight-name portfolio.

AND THE FILTER IS MOSTLY OFF

`CorrelationConfig.apply_above_vix = 25.0` means the filter only engages in
elevated volatility.  In a normal regime — most of the record — there is NO
diversification constraint operating.  That was a defensible setting for a
cross-sector universe of 46; it is a different proposition at 8-from-610.

WHAT THIS MEASURES

  1. Realized pairwise correlation of the names actually held, per rebalance,
     against random 8-name books drawn from the same eligible pool.  The random
     baseline matters: some correlation is the market, and only the excess is
     the model's doing.
  2. Sector concentration of the book — the largest single-sector share, and
     how often the book is majority one sector.
  3. Three filter settings run end to end: off, the inherited VIX>25 gate, and
     always on.  Because a concentration problem nobody pays for is a curiosity,
     not a defect.

Run:  python scripts/analyze_book_correlation.py
      python scripts/analyze_book_correlation.py --top-n 8 --hold 40
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

from momentum.config import CorrelationConfig
from momentum.data import PriceData
from momentum.experiments import production_config
from momentum.liquidity import (LiquidityConfig, apply_tradable,
                                slippage_panel, tradable_mask)
from momentum.reversal import ReversalConfig, build_terms, composite
from momentum.strategy import compute_scores
from momentum.universe import defensive_symbols

from scripts.analyze_reversal_backtest import load_volume, run_arm
from scripts.analyze_reversal_ic import NON_EQUITY, load_pool_panel, pool_symbols

OUT = "rsi_ma_book_correlation.csv"
POOL_FILE = "random_pool.csv"


def book_stats(holdings_history, returns_panel, sectors, window, rng,
               eligible_mask, n_random=200):
    """
    Per rebalance: mean pairwise correlation of the held names, and the same
    for random books of the same size drawn from that date's eligible pool.
    """
    rows = []
    for h in holdings_history:
        d = pd.Timestamp(h["Date"])
        names = [n for n in h["Holdings"] if n in returns_panel.columns]
        if len(names) < 2 or d not in returns_panel.index:
            continue
        win = returns_panel.loc[:d].iloc[-window:]
        if len(win) < window // 2:
            continue

        def mean_corr(cols):
            sub = win[cols].dropna(axis=1, how="all")
            if sub.shape[1] < 2:
                return np.nan
            c = sub.corr().to_numpy()
            iu = np.triu_indices_from(c, k=1)
            v = c[iu]
            return float(np.nanmean(v)) if v.size else np.nan

        held = mean_corr(names)

        pool = list(eligible_mask.columns[eligible_mask.loc[d]]) \
            if d in eligible_mask.index else []
        pool = [p for p in pool if p in returns_panel.columns]
        rnd = np.nan
        if n_random and len(pool) > len(names):
            draws = [mean_corr(list(rng.choice(pool, size=len(names),
                                               replace=False)))
                     for _ in range(n_random)]
            rnd = float(np.nanmean(draws))

        sec = pd.Series([sectors.get(n) for n in names]).dropna()
        top_share = float(sec.value_counts(normalize=True).iloc[0]) \
            if len(sec) else np.nan
        top_sector = sec.value_counts().index[0] if len(sec) else ""

        rows.append({"Date": d, "N": len(names), "HeldCorr": held,
                     "RandomCorr": rnd,
                     "Excess": held - rnd if pd.notna(rnd) else np.nan,
                     "TopSectorShare": top_share, "TopSector": top_sector,
                     "Holdings": " ".join(names)})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Book concentration and correlation")
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--hold", type=int, default=40)
    p.add_argument("--corr-window", type=int, default=50)
    p.add_argument("--random-draws", type=int, default=200)
    p.add_argument("--abs-thresholds", type=float, nargs="+",
                   default=[0.60, 0.65, 0.70, 0.75, 0.80],
                   help="absolute max-correlation values to sweep, always-on")
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--min-adv", type=float, default=10e6)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    cfg = replace(production_config(), top_n=args.top_n, hold_days=args.hold,
                  vix=None)
    rcfg = ReversalConfig()
    lcfg = LiquidityConfig()
    defensive = set(defensive_symbols())

    print("=" * 100)
    print("BOOK CORRELATION AND SECTOR CONCENTRATION")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"score pullback   top_n {args.top_n}   hold {args.hold}d")
    print("=" * 100)

    pool_meta = pd.read_csv(REPO_ROOT / POOL_FILE)
    sectors = pool_meta.set_index("symbol")["sector"].to_dict()

    raw = sorted(set(pool_meta["symbol"].dropna()) - NON_EQUITY)
    full = load_pool_panel(raw + sorted(defensive), start=args.start,
                           use_cache=not args.no_cache)
    allowed = set(pool_symbols(POOL_FILE)) | defensive
    keep = [c for c in full.close.columns if c in allowed]
    full = PriceData(close=full.close[keep], open_=full.open_[keep],
                     spy=full.spy, vix=full.vix)
    vol = load_volume(raw + sorted(defensive), start=args.start,
                      use_cache=not args.no_cache)
    vol = vol.reindex(index=full.close.index, columns=full.close.columns)

    ranking, base, _ = compute_scores(full, cfg)
    terms = build_terms(full.close, rcfg)
    pull = composite(terms, replace(rcfg, turn_weight=-1.0,
                                    strength_weight=0.0, room_weight=1.0),
                     "flip").reindex_like(ranking)
    tmask = tradable_mask(full.close, vol, args.min_adv, args.min_price, lcfg)
    pull = apply_tradable(pull, tmask, defensive)
    slip = slippage_panel(full.close, vol, lcfg, top_n=cfg.top_n)

    # The inherited setting uses a RELATIVE threshold (the 85th percentile of
    # that date's correlation distribution) and only engages above VIX 25.  The
    # absolute variants below are the rule as normally described: "no pair above
    # rho", applied every rebalance.  They are different rules and an earlier
    # version of this study conflated them.
    print("\n  Filter settings, end to end")
    variants = {
        "off": None,
        "relative 85th pct, VIX>25 (inherited)": replace(
            cfg.correlation, enabled=True, apply_above_vix=25.0),
        "relative 85th pct, always": replace(
            cfg.correlation, enabled=True, apply_above_vix=None),
    }
    for thresh in args.abs_thresholds:
        variants["absolute %.2f, always" % thresh] = replace(
            cfg.correlation, enabled=True, apply_above_vix=None,
            method="absolute", max_correlation=thresh)
    hdr = (f"    {'setting':<40}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
           f"{'Calmar':>8}{'Vol':>8}{'Turn':>8}{'held rho':>10}{'1-sector':>10}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))

    rets_all = full.close.pct_change()
    rng0 = np.random.default_rng(args.seed)
    results, rows = {}, []
    for label, corr in variants.items():
        c = replace(cfg, correlation=corr)
        m, res = run_arm(pull, None, full, c, slip, return_result=True)
        results[label] = res

        # What the filter actually achieved, not just what it cost: mean
        # pairwise correlation of the resulting book, and how often the book is
        # majority one sector. A filter that costs CAGR without moving these
        # has bought nothing.
        s = book_stats(res.holdings_history, rets_all, sectors,
                       args.corr_window, rng0, tmask, 0)
        held_rho = s["HeldCorr"].mean() if not s.empty else float("nan")
        one_sec = ((s["TopSectorShare"] >= 0.5).mean()
                   if not s.empty else float("nan"))

        print(f"    {label:<40}{m['CAGR']:>9.2%}{m['Sharpe']:>8.2f}"
              f"{m['MaxDD']:>9.2%}{m['Calmar']:>8.2f}{m['Volatility']:>8.2%}"
              f"{m['Annual Turnover']:>8.0%}{held_rho:>10.3f}"
              f"{one_sec:>10.0%}", flush=True)
        rows.append({"Setting": label, "HeldCorr": held_rho,
                     "MajorityOneSector": one_sec, **m})

    # --- what the book actually looks like, on the live-configured arm ---
    rets = rets_all
    rng = np.random.default_rng(args.seed)
    stats = book_stats(results["relative 85th pct, VIX>25 (inherited)"].holdings_history,
                       rets, sectors, args.corr_window, rng, tmask,
                       args.random_draws)

    print("\n" + "=" * 100)
    print("WHAT THE BOOK LOOKS LIKE  (inherited filter setting)")
    print("=" * 100)
    if stats.empty:
        print("  no books to analyse")
        return 1

    print(f"  {len(stats)} rebalances analysed, {args.corr_window}-day trailing "
          f"correlation, {args.random_draws} random books per date")
    print(f"\n    mean pairwise correlation of HELD names   "
          f"{stats['HeldCorr'].mean():.3f}")
    print(f"    same for RANDOM books from the same pool  "
          f"{stats['RandomCorr'].mean():.3f}")
    print(f"    excess attributable to the score          "
          f"{stats['Excess'].mean():+.3f}")
    hi = (stats["Excess"] > 0).mean()
    print(f"    rebalances where held > random            {hi:.0%}")

    print(f"\n    largest single-sector share of the book, mean "
          f"{stats['TopSectorShare'].mean():.0%}")
    for thresh in (0.5, 0.625, 0.75):
        share = (stats["TopSectorShare"] >= thresh).mean()
        print(f"    book at least {thresh:.0%} one sector: "
              f"{share:.0%} of rebalances")

    print("\n    most concentrated books:")
    worst = stats.nlargest(5, "TopSectorShare")
    for _, r in worst.iterrows():
        print(f"      {r['Date']:%Y-%m-%d}  {r['TopSectorShare']:.0%} "
              f"{r['TopSector']:<24} {r['Holdings']}")

    print("\n    most correlated books (excess over random):")
    for _, r in stats.nlargest(5, "Excess").iterrows():
        print(f"      {r['Date']:%Y-%m-%d}  held {r['HeldCorr']:.2f} vs "
              f"random {r['RandomCorr']:.2f}  ({r['Excess']:+.2f})  "
              f"{r['Holdings']}")

    stats.to_csv(REPO_ROOT / OUT, index=False)
    pd.DataFrame(rows).to_csv(REPO_ROOT / "rsi_ma_book_correlation_sweep.csv",
                              index=False)
    print(f"\n  Exported to {REPO_ROOT / OUT}")

    print("\n" + "=" * 100)
    print("READING THIS")
    print("=" * 100)
    print("  A concentration problem only matters if it costs something. Compare")
    print("  the three settings above: if 'always on' does not improve drawdown")
    print("  or Sharpe, the concentration is real but not worth constraining —")
    print("  and Track A's finding stands, that constraints on this book tend to")
    print("  cost more than they save.")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
