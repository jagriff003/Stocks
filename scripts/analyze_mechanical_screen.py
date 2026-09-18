"""
Does a MECHANICAL strength screen recover the live universe's edge?

Track H drew baskets from any liquid US equity and found the model gives up
8.80pp of CAGR to simply owning its own basket, winning on 2 of 100.  On the
live universe it gives up 1.44pp and cuts drawdown by 16pp.  That comparison
conflated two things: screening for strength, and the specific judgment about
which strong names to keep.  This separates them.

THE TEST
   Screen the pool mechanically as of a date, using only data available then.
   Draw baskets from whatever qualifies.  Run the unchanged model forward.  If
   mechanically screened baskets land near the live universe's -1.44pp, the
   screening RULE generalizes and the judgment was implementing something a
   script could do.  If they land near the unscreened -8.80pp, the edge was the
   judgment, which on a universe assembled with the answer visible is
   indistinguishable from hindsight.

   Three criteria, because a conclusion that holds across all three is worth far
   more than one that depends on the definition of "strength":
     mom12_1   trailing 12-month return skipping the last month - the academic
               cross-sectional momentum definition
     above_ma  price above its own 200-day moving average - a trend FILTER
               rather than a ranking, and the closest mechanical analogue to the
               Ivy/Faber rule this strategy descends from
     both      in the top slice on mom12_1 AND above the 200-day MA

   Two modes:
     frozen    screen once at the start, hold that basket forever.  The clean
               test of whether a screening rule works at all.
     rolling   re-screen every `--rescreen-days`, re-drawing the basket from
               whatever qualifies then.  Closer to real practice, and the gap
               between the two modes IS the value of re-screening - the
               universe-decay question nothing has answered yet.

HOW ROLLING MEMBERSHIP IS IMPLEMENTED
   The panel holds the union of every name the basket ever contains, and a
   name's ranking score is set to a large negative sentinel on dates it is not a
   member.  Masking to NaN would have been the obvious approach and is wrong:
   `build_target_portfolios` derives its history gate from
   `scores.notna().cumsum()`, so NaN-masking a name would stall its history
   counter and could exclude it permanently once it rejoined.  The sentinel
   keeps every name countable while ensuring it can never be selected.

WHAT THIS CANNOT DO
   It cannot bound survivorship.  A 2012 screen applied to a pool of names that
   are liquid in 2026 selects "strong in 2012 AND still around in 2026".  Levels
   stay inflated; only the within-basket differences are clean, and those are
   what is reported.

Every threshold is a flag.

Run:  python scripts/analyze_mechanical_screen.py
      python scripts/analyze_mechanical_screen.py --quantile 0.5 --baskets 40
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.backtest import build_target_portfolios, simulate_portfolio
from momentum.data import PriceData
from momentum.experiments import production_config
from momentum.metrics import calculate_performance_metrics
from momentum.strategy import compute_scores
from momentum.universe import current_symbols, defensive_symbols

OUT = "rsi_ma_mechanical_screen.csv"
POOL = "random_pool.csv"
SENTINEL = -1e9


# --------------------------------------------------------------------------
# Screens — each returns the symbols qualifying as of `asof`, using only
# information available on or before that date.
# --------------------------------------------------------------------------

def screen_mom12_1(close, asof, quantile):
    hist = close.loc[:asof]
    if len(hist) < 260:
        return []
    end = hist.iloc[-21]          # skip the most recent month
    start = hist.iloc[-252]
    ret = (end / start - 1).dropna()
    if ret.empty:
        return []
    return ret[ret >= ret.quantile(1 - quantile)].index.tolist()


def screen_above_ma(close, asof, quantile, window=200):
    hist = close.loc[:asof]
    if len(hist) < window + 5:
        return []
    ma = hist.iloc[-window:].mean()
    last = hist.iloc[-1]
    ratio = (last / ma).dropna()
    return ratio[ratio > 1.0].index.tolist()


def screen_both(close, asof, quantile):
    a = set(screen_mom12_1(close, asof, quantile))
    b = set(screen_above_ma(close, asof, quantile))
    return sorted(a & b)


SCREENS = {"mom12_1": screen_mom12_1,
           "above_ma": screen_above_ma,
           "both": screen_both}


# --------------------------------------------------------------------------

def rescreen_dates(index, start, every):
    dates = index[index >= pd.Timestamp(start)]
    return list(dates[::every])


def build_membership(close, pool_syms, screen, quantile, dates, size, rng):
    """
    date -> the basket in force from that date until the next.

    Each re-screen re-draws the basket from whatever qualifies, which is the
    honest model of an investor who re-runs their screen and rebuilds.
    """
    out = {}
    for d in dates:
        ok = [s for s in screen(close[pool_syms], d, quantile) if s in pool_syms]
        if len(ok) < size:
            ok = ok + [s for s in pool_syms if s not in ok]
        out[d] = list(rng.choice(ok, size=min(size, len(ok)), replace=False))
    return out


def masked_scores(ranking, membership, defensive):
    """
    Sentinel-mask every name outside the basket in force on each date.

    Built positionally with numpy rather than through `.loc[rows, cols] = True`.
    That pandas form SILENTLY NO-OPS on a large all-bool frame under pandas 3's
    copy-on-write - it sets nothing, raises nothing, and warns about nothing,
    which produced a membership matrix containing only the defensive sleeve and
    a set of entirely plausible-looking results.  Positional assignment cannot
    fail that way.
    """
    idx, cols = ranking.index, list(ranking.columns)
    col_pos = {c: i for i, c in enumerate(cols)}
    arr = np.zeros((len(idx), len(cols)), dtype=bool)

    keys = sorted(membership)
    starts = [idx.searchsorted(pd.Timestamp(k), side="left") for k in keys]
    for i, k in enumerate(keys):
        lo = starts[i]
        hi = starts[i + 1] if i + 1 < len(keys) else len(idx)
        take = [col_pos[str(c)] for c in membership[k] if str(c) in col_pos]
        if take and hi > lo:
            arr[lo:hi, take] = True
    for s in defensive:
        if s in col_pos:
            arr[:, col_pos[s]] = True

    members = pd.DataFrame(arr, index=idx, columns=cols)
    if not members.values.any():
        raise AssertionError(
            "Membership matrix is empty - no basket name is ever in force. "
            "Refusing to run on a universe that does not exist.")
    out = ranking.where(members, SENTINEL)
    return out.where(ranking.notna()), members


def ew_returns(close, members, cfg, defensive):
    """Equal-weight whatever the basket holds on each date, excluding the sleeve."""
    cols = [c for c in members.columns if c not in set(defensive)]
    m = members[cols]
    r = close[cols].pct_change().where(m)
    out = r.mean(axis=1).loc[close.index[cfg.min_data_days]:].dropna()
    return calculate_performance_metrics(
        out, risk_free_rate=cfg.execution.risk_free_rate)


def run_model(prices, cfg, ranking, base):
    targets, _ = build_target_portfolios(
        ranking, price_columns=list(prices.close.columns), top_n=cfg.top_n,
        min_data_days=cfg.min_data_days, hold_days=cfg.hold_days,
        vix_data=prices.vix, vix_config=cfg.vix, base_composite_scores=base,
        velocity_config=cfg.velocity, correlation_config=cfg.correlation,
        graduated_config=cfg.graduated_vix, exit_config=cfg.exits,
        close=prices.close, rank_offset=cfg.rank_offset,
        rank_offset_scope=cfg.rank_offset_scope)
    r = simulate_portfolio(targets, prices.close, prices.open_,
                           execution=cfg.execution)
    return r.metrics


def main() -> int:
    p = argparse.ArgumentParser(description="Mechanical strength screen test")
    p.add_argument("--baskets", type=int, default=25)
    p.add_argument("--basket-size", type=int, default=40)
    p.add_argument("--quantile", type=float, default=0.20,
                   help="top slice kept by the ranking screens")
    p.add_argument("--screen-date", default="2012-01-03")
    p.add_argument("--rescreen-days", type=int, default=126,
                   help="sessions between re-screens in rolling mode (~6 months)")
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--start", default="2010-01-01")
    args = p.parse_args()

    print("=" * 108)
    print("MECHANICAL STRENGTH SCREEN")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"screen date {args.screen_date}, top {args.quantile:.0%}, "
          f"rescreen every {args.rescreen_days} sessions")
    print("=" * 108)

    from scripts.analyze_random_baskets import download_panel, subpanel

    cfg = production_config()
    defensive = defensive_symbols()
    live = [s for s in current_symbols() if s not in set(defensive)]
    pool = pd.read_csv(REPO_ROOT / POOL)
    pool_syms = pool["symbol"].tolist()

    print(f"\nPool {len(pool_syms)} names.  Downloading (cached where possible).")
    close, open_, spy, vix = download_panel(
        set(pool_syms) | set(defensive) | set(live), args.start, verbose=False)
    pool_syms = [s for s in pool_syms if s in close.columns]
    print(f"{len(pool_syms)} usable after download.")

    rng = np.random.default_rng(args.seed)
    rows = []

    def record(family, i, arm, m):
        rows.append({"Family": family, "Basket": i, "Arm": arm,
                     "CAGR": m["cagr"], "Sharpe": m["sharpe_ratio"],
                     "MaxDD": m["max_drawdown"]})

    # Reference: the live universe, identical path.
    lp = subpanel(close, open_, spy, vix, live + defensive)
    lr, lb, _ = compute_scores(lp, cfg)
    m_live = run_model(lp, cfg, lr, lb)
    ew_live = calculate_performance_metrics(
        lp.close[live].pct_change().mean(axis=1)
          .loc[lp.close.index[cfg.min_data_days]:].dropna(),
        risk_free_rate=cfg.execution.risk_free_rate)
    print(f"\nLIVE: model {m_live['cagr']:.2%}  EW {ew_live['cagr']:.2%}  "
          f"edge {m_live['cagr'] - ew_live['cagr']:+.2%}")
    record("LIVE", 0, "model", m_live)
    record("LIVE", 0, "equal_weight", ew_live)

    all_dates = rescreen_dates(close.index, args.screen_date, args.rescreen_days)
    print(f"\nRe-screen dates in rolling mode: {len(all_dates)} "
          f"({all_dates[0]:%Y-%m-%d} to {all_dates[-1]:%Y-%m-%d})")

    for name, screen in SCREENS.items():
        qualifying = screen(close[pool_syms], pd.Timestamp(args.screen_date),
                            args.quantile)
        print(f"\n{name}: {len(qualifying)} names qualify at {args.screen_date}")

        for mode in ("frozen", "rolling"):
            fam = f"{name}/{mode}"
            dates = [pd.Timestamp(args.screen_date)] if mode == "frozen" else all_dates
            for i in range(args.baskets):
                mem = build_membership(close, pool_syms, screen, args.quantile,
                                       dates, args.basket_size, rng)
                names = sorted({s for v in mem.values() for s in v})
                pr = subpanel(close, open_, spy, vix, names + defensive)
                try:
                    rk, bs, _ = compute_scores(pr, cfg)
                    mk, members = masked_scores(rk, mem, defensive)
                    m = run_model(pr, cfg, mk, bs)
                    ew = ew_returns(pr.close, members.reindex(
                        columns=pr.close.columns, fill_value=False), cfg, defensive)
                except Exception as e:
                    print(f"    basket {i} failed: {type(e).__name__}: {e}")
                    continue
                record(fam, i, "model", m)
                record(fam, i, "equal_weight", ew)
            print(f"    {mode}: done")

    table = pd.DataFrame(rows)
    table.to_csv(REPO_ROOT / OUT, index=False)

    print("\n" + "=" * 108)
    print("MODEL MINUS EQUAL-WEIGHTING THE SAME BASKET")
    print("=" * 108)
    hdr = (f"{'Family':<24}{'n':>4}{'model':>9}{'equal wt':>10}{'edge':>9}"
           f"{'edge sd':>9}{'win':>7}{'model DD':>10}{'EW DD':>9}{'DD edge':>9}")
    print(hdr)
    print("-" * len(hdr))
    for fam in table["Family"].unique():
        s = table[table.Family == fam]
        mo = s[s.Arm == "model"].set_index("Basket")
        ew = s[s.Arm == "equal_weight"].set_index("Basket")
        e = (mo["CAGR"] - ew["CAGR"]).dropna()
        dd = (mo["MaxDD"] - ew["MaxDD"]).dropna()
        print(f"{fam:<24}{len(e):>4}{mo['CAGR'].mean():>8.2%} "
              f"{ew['CAGR'].mean():>9.2%} {e.mean():>+8.2%} {e.std():>8.2%} "
              f"{(e > 0).mean():>6.0%} {mo['MaxDD'].mean():>9.1%} "
              f"{ew['MaxDD'].mean():>8.1%} {dd.mean():>+8.1%}")
    print("-" * len(hdr))
    print("  Reference points: LIVE edge -1.44%, Track H unscreened random -8.80%.")
    print("  Near LIVE means the screening rule generalizes; near -8.80% means")
    print("  the edge was judgment, not the screen.")

    print(f"\nExported to: {REPO_ROOT / OUT}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
