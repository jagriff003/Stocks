"""
Does the model work on any basket, or only on the curated one?

Track F showed the universe produces the return and the ranker produces almost
none of it.  That was measured inside one universe.  This asks the question the
other way round: hand the unchanged model a basket of stocks it has never seen,
and see whether it still does what it does.

THE COMPARISON THAT MATTERS
   Not "what did random baskets return" - that number is survivorship-inflated
   and close to meaningless on its own.  The question is **how often does the
   model beat its own basket**, over many baskets.  Both sides of that comparison
   carry the identical survivorship premium, so the difference between them is
   clean even though neither level is.

   If the model beats equal-weighting its own basket on 45 of 50 draws, the
   mechanism generalizes and the curated universe is not load-bearing.  If it
   only wins on the live universe, the edge was curation all along.

FOUR FAMILIES
   Two choices crossed, because they answer different questions and a single
   design would confound them:

                     | cap-matched            | all-cap
       stratified    | sector AND cap profile | sector shape, any size
       pure random   | cap tier, no structure | no constraints at all

   Stratified baskets copy the live universe's sector proportions.  If those
   match the live universe but pure-random ones do not, the edge is
   diversification discipline rather than name selection - a different claim,
   and a more transferable one.

   Cap-matching exists because the live universe is large-cap tilted.  Without
   it a random basket would differ on cap tier AND curation simultaneously, and
   a gap could not be attributed to either.

WHAT THIS CANNOT DO
   It cannot bound survivorship.  Every basket is drawn from names that exist
   and are liquid in 2026 with sixteen years of history, which is a harder
   survivorship filter than the live universe passes.  Absolute CAGRs here are
   inflated, and the only defence is to read the WITHIN-basket comparisons and
   ignore the levels.  See TODO item 6.

POWER
   Phase noise alone is ~2-3pp of CAGR per run and basket-to-basket variation is
   larger.  At `--baskets` 25 per family this resolves differences of roughly
   1.5-2pp between families and no less.  Families landing within a point of
   each other means "no detectable difference", not a ranking.

Every threshold is a flag.

Run:  python scripts/build_random_pool.py     # once, builds random_pool.csv
      python scripts/analyze_random_baskets.py
      python scripts/analyze_random_baskets.py --baskets 50 --basket-size 40
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
from momentum.universe import current_symbols, defensive_symbols, sector_map

OUT = "rsi_ma_random_baskets.csv"
POOL = "random_pool.csv"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def download_panel(symbols, start, verbose=True):
    """One batched pull for every symbol any basket needs, plus SPY and VIX."""
    import yfinance as yf
    warnings.filterwarnings("ignore")
    syms = sorted(set(symbols))
    frames_c, frames_o = [], []
    chunk = 400
    for i in range(0, len(syms), chunk):
        part = syms[i:i + chunk]
        d = yf.download(part, start=start, interval="1d", auto_adjust=True,
                        progress=False, threads=True, group_by="column")
        frames_c.append(d["Close"])
        frames_o.append(d["Open"])
        if verbose:
            print(f"    {min(i + chunk, len(syms))}/{len(syms)} symbols")
    close = pd.concat(frames_c, axis=1)
    open_ = pd.concat(frames_o, axis=1)

    aux = yf.download(["SPY", "^VIX"], start=start, interval="1d",
                      auto_adjust=True, progress=False, group_by="column")["Close"]
    return close, open_, aux["SPY"], aux["^VIX"]


def subpanel(close, open_, spy, vix, names) -> PriceData:
    keep = [n for n in names if n in close.columns]
    return PriceData(close=close[keep].copy(), open_=open_[keep].copy(),
                     spy=spy, vix=vix)


# --------------------------------------------------------------------------
# Basket construction
# --------------------------------------------------------------------------

def cap_band(pool, live_caps, lo_q, hi_q):
    """The market-cap window the live universe occupies."""
    lo, hi = live_caps.quantile(lo_q), live_caps.quantile(hi_q)
    return pool[(pool["marketCap"] >= lo) & (pool["marketCap"] <= hi)], lo, hi


def draw_basket(pool, size, rng, target_sectors=None):
    """One basket: stratified to `target_sectors` if given, else uniform."""
    if target_sectors is None:
        n = min(size, len(pool))
        return list(rng.choice(pool["symbol"].to_numpy(), size=n, replace=False))

    picks = []
    for sector, want in target_sectors.items():
        avail = pool[pool["sector"] == sector]["symbol"].to_numpy()
        take = min(want, len(avail))
        if take:
            picks += list(rng.choice(avail, size=take, replace=False))
    # Top up from anywhere if a sector was too thin to fill its quota.
    if len(picks) < size:
        rest = pool[~pool["symbol"].isin(picks)]["symbol"].to_numpy()
        need = min(size - len(picks), len(rest))
        if need:
            picks += list(rng.choice(rest, size=need, replace=False))
    return picks[:size]


def sector_quota(live_sectors: pd.Series, size: int) -> dict:
    """Live universe sector proportions, scaled to `size` and rounded to fit."""
    share = live_sectors.value_counts(normalize=True)
    raw = (share * size)
    out = raw.apply(np.floor).astype(int)
    # Hand the rounding remainder to the largest fractional parts.
    short = size - int(out.sum())
    if short > 0:
        order = (raw - out).sort_values(ascending=False).index[:short]
        for s in order:
            out[s] += 1
    return {k: int(v) for k, v in out.items() if v > 0}


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

def run_model(prices, cfg, scores=None):
    ranking, base = scores if scores else compute_scores(prices, cfg)[:2]
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
    return r.metrics, (ranking, base)


def run_equal_weight(prices, cfg, names):
    """Own the whole basket, equal-weighted, on the simulator's convention."""
    held = [n for n in names if n in prices.close.columns]
    r = prices.close[held].loc[prices.close.index[cfg.min_data_days]:] \
             .pct_change().mean(axis=1).dropna()
    return calculate_performance_metrics(
        r, risk_free_rate=cfg.execution.risk_free_rate)


def main() -> int:
    p = argparse.ArgumentParser(description="Random-basket generalization study")
    p.add_argument("--baskets", type=int, default=25, help="baskets per family")
    p.add_argument("--basket-size", type=int, default=40)
    p.add_argument("--random4-draws", type=int, default=3,
                   help="random-4 control runs per basket")
    p.add_argument("--cap-lo", type=float, default=0.10,
                   help="lower quantile of the live cap distribution")
    p.add_argument("--cap-hi", type=float, default=1.00)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--start", default="2010-01-01",
                   help="panel start. MUST match the production load_data "
                        "start: moving it seven months earlier costs 6.6pp of "
                        "CAGR through the rotation-phase effect alone, and the "
                        "reconciliation check below exists to catch that.")
    p.add_argument("--tolerance", type=float, default=0.005,
                   help="max CAGR gap allowed between the live universe run "
                        "through this path and through production load_data")
    args = p.parse_args()

    print("=" * 104)
    print("RANDOM-BASKET GENERALIZATION STUDY")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 104)

    pool = pd.read_csv(REPO_ROOT / POOL)
    pool = pool[pool["sector"].notna() & pool["marketCap"].notna()]
    cfg = production_config()
    defensive = defensive_symbols()
    live = [s for s in current_symbols() if s not in set(defensive)]
    live_sectors = pd.Series(sector_map()).reindex(live).dropna()

    print(f"\nPool: {len(pool)} names.  Live universe: {len(live)} momentum names.")

    # Cap band from whatever live names the pool happens to know.
    live_caps = pool[pool["symbol"].isin(live)]["marketCap"]
    if len(live_caps) < 10:
        print("  (few live names in the pool; using the pool's own upper half)")
        live_caps = pool["marketCap"]
    capped, lo, hi = cap_band(pool, live_caps, args.cap_lo, args.cap_hi)
    print(f"Cap band from {len(live_caps)} live names: "
          f"${lo/1e9:.1f}B - ${hi/1e9:.0f}B  ->  {len(capped)} pool names")

    quota = sector_quota(live_sectors, args.basket_size)
    print(f"Sector quota for a {args.basket_size}-name basket: {quota}")

    rng = np.random.default_rng(args.seed)
    families = {
        "stratified/cap-matched": (capped, quota),
        "stratified/all-cap":     (pool, quota),
        "random/cap-matched":     (capped, None),
        "random/all-cap":         (pool, None),
    }

    baskets = {}
    for fam, (src, q) in families.items():
        baskets[fam] = [draw_basket(src, args.basket_size, rng, q)
                        for _ in range(args.baskets)]

    needed = {s for bs in baskets.values() for b in bs for s in b}
    needed |= set(defensive) | set(live)
    print(f"\nDownloading {len(needed)} symbols (batched, ~"
          f"{len(needed)//400 + 1} requests)")
    close, open_, spy, vix = download_panel(needed, args.start)

    rows = []

    def record(family, idx, arm, m, names=None):
        rows.append({"Family": family, "Basket": idx, "Arm": arm,
                     "CAGR": m["cagr"], "Sharpe": m["sharpe_ratio"],
                     "MaxDD": m["max_drawdown"], "Vol": m["volatility"],
                     "N": len(names) if names else np.nan})

    # The live universe, through the identical path, as the reference.
    lp = subpanel(close, open_, spy, vix, live + defensive)
    m_live, _ = run_model(lp, cfg)
    m_live_ew = run_equal_weight(lp, cfg, live)
    print(f"\nLive universe reference:  model {m_live['cagr']:.2%}  "
          f"equal-weight {m_live_ew['cagr']:.2%}  "
          f"edge {m_live['cagr'] - m_live_ew['cagr']:+.2%}")
    # Reconciliation: the live universe through THIS path must reproduce the
    # production path.  The panel start alone is worth 6.6pp here - it shifts
    # every rebalance date, and the rotation phase is worth ~3pp of CAGR (see
    # FINDINGS, "Rotation phase is worth ~3pp of the headline CAGR").  Without
    # this check a misaligned panel would quietly make every basket in the
    # study incomparable to the live model.
    from momentum.data import load_data as _load
    _ref = _load(current_symbols(), start_date=args.start, use_cache=True,
                 verbose=False, drop_unsettled=True)
    _m, _ = run_model(_ref, cfg)
    _gap = abs(_m["cagr"] - m_live["cagr"])
    print(f"Reconciliation vs production load_data: "
          f"{_m['cagr']:.2%} vs {m_live['cagr']:.2%}, gap {_gap:.3%}")
    if _gap > args.tolerance:
        raise SystemExit(
            f"Live universe does not reconcile (gap {_gap:.3%} > "
            f"{args.tolerance:.3%}). The panel is misaligned with production; "
            f"refusing to report baskets that cannot be compared to it.")

    record("LIVE", 0, "model", m_live, live)
    record("LIVE", 0, "equal_weight", m_live_ew, live)

    for fam, bs in baskets.items():
        print(f"\n{fam}: {len(bs)} baskets")
        for i, names in enumerate(bs):
            pr = subpanel(close, open_, spy, vix, list(names) + defensive)
            if pr.close.shape[1] < args.basket_size // 2:
                continue
            try:
                m, sc = run_model(pr, cfg)
            except Exception as e:
                print(f"    basket {i}: model failed ({type(e).__name__})")
                continue
            record(fam, i, "model", m, names)
            record(fam, i, "equal_weight",
                   run_equal_weight(pr, cfg, names), names)

            ranking, base = sc
            for _ in range(args.random4_draws):
                rnd = pd.DataFrame(rng.standard_normal(ranking.shape),
                                   index=ranking.index,
                                   columns=ranking.columns).where(ranking.notna())
                try:
                    mr, _ = run_model(pr, cfg, scores=(rnd, base))
                    record(fam, i, "random4", mr, names)
                except Exception:
                    pass
            if (i + 1) % 5 == 0:
                print(f"    {i + 1}/{len(bs)}")

    table = pd.DataFrame(rows)
    table.to_csv(REPO_ROOT / OUT, index=False)

    print("\n" + "=" * 104)
    print("DOES THE MODEL BEAT ITS OWN BASKET?")
    print("=" * 104)
    hdr = (f"{'Family':<26}{'n':>4}{'model':>9}{'equal wt':>10}{'edge':>9}"
           f"{'edge sd':>9}{'win rate':>10}{'vs rand4':>10}")
    print(hdr)
    print("-" * len(hdr))
    for fam in ["LIVE"] + list(families):
        sub = table[table.Family == fam]
        if sub.empty:
            continue
        mo = sub[sub.Arm == "model"].set_index("Basket")["CAGR"]
        ew = sub[sub.Arm == "equal_weight"].set_index("Basket")["CAGR"]
        edge = (mo - ew).dropna()
        r4 = sub[sub.Arm == "random4"].groupby("Basket")["CAGR"].mean()
        vr = (mo - r4).dropna()
        print(f"{fam:<26}{len(edge):>4}{mo.mean():>8.2%} {ew.mean():>9.2%} "
              f"{edge.mean():>+8.2%} {edge.std():>8.2%} "
              f"{(edge > 0).mean():>9.1%} {vr.mean() if len(vr) else np.nan:>+9.2%}")
    print("-" * len(hdr))
    print("  edge     = model CAGR minus equal-weighting the SAME basket")
    print("  win rate = share of baskets where the model beat its own basket")
    print("  vs rand4 = model minus a random-4 book from the same basket")
    print("\n  Read the win rate. Levels carry survivorship; the within-basket")
    print("  differences do not.")

    print(f"\nExported to: {REPO_ROOT / OUT}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
