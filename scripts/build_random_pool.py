"""
Build the candidate pool that random baskets are drawn from.

A one-time, cached artifact.  Separated from the study that uses it because
assembling it costs a few minutes of sector lookups and should not be repeated
on every run, and because the pool's composition is itself a result worth
inspecting before anything is drawn from it.

WHAT THIS POOL IS, AND WHAT IT IS NOT

   It is every US equity that is liquid TODAY and has a full price history back
   to the backtest's start.  That makes it the right control for testing whether
   the model's edge depends on the curated universe: baskets drawn from here and
   the live universe are both "things that exist and are liquid in 2026", so the
   DIFFERENCE between them isolates curation.

   It is not a fix for survivorship bias, and using it as one would be a
   mistake.  The history filter actively selects for companies that survived AND
   stayed liquid for sixteen years, which makes the pool *more* survivorship-
   contaminated than the screen it starts from, not less.  Absolute returns from
   any basket drawn here are inflated by an unknown amount, exactly as the live
   universe's are.  Fixing that needs point-in-time constituent data including
   delisted names, which no free source provides; see TODO item 6.

FILTERS, IN ORDER (all are flags)
   1. Yahoo's equity screener: US region, average 3-month volume and price above
      thresholds.  The volume floor is the one that matters for execution - a
      basket full of names that trade 50k shares a day would fail on slippage
      long before the model's merits came into it.
   2. First trade date before the panel start, which is cheap and discards most
      of what would fail step 4 without downloading it.
   3. Sector.  This is the only expensive step in REQUEST terms, and requests
      are the binding resource: yfinance's unofficial ceiling is roughly
      2,000-2,500 per hour per IP.  Batched price downloads cost one request per
      400-name chunk and are free by comparison; `.info` costs one request per
      NAME, so sectoring the whole pool would spend most of an hour's budget in
      one go.

      Three things keep that in bounds.  `yf.Sector.top_companies` is tried
      first and covers ~550 names for 11 requests.  Whatever is still missing is
      looked up individually only up to `--sector-budget`.  And every lookup
      ever made is cached to `sector_cache.csv`, so a second run spends nothing
      on names already seen and a budget-limited pool can be completed across
      several runs.
   4. A real price history: at least `--min-observations` daily closes.

Run:  python scripts/build_random_pool.py
      python scripts/build_random_pool.py --min-volume 2000000 --min-price 15
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_POOL = "random_pool.csv"


def screen_pool(min_volume: float, min_price: float, cap: int) -> pd.DataFrame:
    import yfinance as yf
    q = yf.EquityQuery("and", [
        yf.EquityQuery("eq", ["region", "us"]),
        yf.EquityQuery("gt", ["avgdailyvol3m", min_volume]),
        yf.EquityQuery("gt", ["intradayprice", min_price]),
    ])
    rows, off = [], 0
    while off < cap:
        r = yf.screen(q, size=250, offset=off,
                      sortField="avgdailyvol3m", sortAsc=False)
        qs = r.get("quotes", [])
        if not qs:
            break
        rows += qs
        off += 250
    df = pd.DataFrame(rows).drop_duplicates("symbol")
    if "quoteType" in df:
        df = df[df["quoteType"] == "EQUITY"]
    keep = [c for c in ("symbol", "shortName", "marketCap",
                        "averageDailyVolume3Month", "regularMarketPrice",
                        "firstTradeDateMilliseconds") if c in df.columns]
    return df[keep].reset_index(drop=True)


SECTOR_KEYS = ["technology", "financial-services", "healthcare",
               "consumer-cyclical", "consumer-defensive", "industrials",
               "energy", "basic-materials", "real-estate", "utilities",
               "communication-services"]
SECTOR_CACHE = "sector_cache.csv"


def load_sector_cache() -> dict:
    path = REPO_ROOT / SECTOR_CACHE
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["symbol"], df["sector"]))


def save_sector_cache(mapping: dict) -> None:
    frame = pd.DataFrame({"symbol": list(mapping),
                          "sector": list(mapping.values())})
    frame.to_csv(REPO_ROOT / SECTOR_CACHE, index=False)


def normalize_sector(value):
    """One spelling for both sources: slug or display form -> display form."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value).replace("-", " ").title().replace("Real Estate", "Real Estate")


def sectors_from_sector_api(verbose: bool = True) -> dict:
    """~550 names for 11 requests.  Free coverage before spending any budget."""
    import yfinance as yf
    out = {}
    for key in SECTOR_KEYS:
        try:
            tc = yf.Sector(key).top_companies
            if tc is not None and len(tc):
                # Normalize to the display form `.info` returns.  The two
                # sources disagree on spelling ("financial-services" vs
                # "Financial Services") and mixing them silently splits every
                # sector in two, which would quietly wreck stratified sampling.
                label = normalize_sector(key)
                for sym in tc.index:
                    out[str(sym)] = label
        except Exception:
            continue
    if verbose:
        print(f"    yf.Sector covered {len(out)} names for "
              f"{len(SECTOR_KEYS)} requests")
    return out


def add_sectors(df: pd.DataFrame, budget: int, sleep: float,
                verbose: bool = True) -> tuple:
    """
    Attach sectors, spending at most `budget` per-ticker requests.

    Returns (frame, requests_spent).  Names left without a sector are kept -
    they are still usable by the unstratified basket families, and dropping them
    would silently shrink the pool the two families share.
    """
    import yfinance as yf
    warnings.filterwarnings("ignore")

    cache = load_sector_cache()
    if verbose:
        print(f"    {len(cache)} sectors already cached")

    free = sectors_from_sector_api(verbose)
    for sym, sec in free.items():
        cache.setdefault(sym, sec)

    missing = [s for s in df["symbol"] if s not in cache or pd.isna(cache[s])]
    todo = missing[:max(0, budget)]
    if verbose:
        print(f"    {len(missing)} still missing; looking up {len(todo)} "
              f"(budget {budget})")

    t0, spent = time.time(), 0
    for i, sym in enumerate(todo, 1):
        try:
            cache[sym] = yf.Ticker(sym).info.get("sector")
        except Exception:
            cache[sym] = None
        spent += 1
        if sleep:
            time.sleep(sleep)
        if verbose and i % 100 == 0:
            print(f"    {i}/{len(todo)} ({time.time() - t0:.0f}s)")

    save_sector_cache({k: v for k, v in cache.items() if v})
    df = df.copy()
    df["sector"] = df["symbol"].map(cache).map(normalize_sector)
    return df, spent


def main() -> int:
    p = argparse.ArgumentParser(description="Build the random-basket candidate pool")
    p.add_argument("--min-volume", type=float, default=1_000_000,
                   help="minimum average 3-month daily volume (execution floor)")
    p.add_argument("--min-price", type=float, default=10.0)
    p.add_argument("--screen-cap", type=int, default=3000,
                   help="maximum rows to pull from the screener")
    p.add_argument("--start", default="2010-01-01",
                   help="panel start the pool must have history for")
    p.add_argument("--warmup-start", default="2009-06-01",
                   help="download start, earlier than --start for the 200d warmup")
    p.add_argument("--min-observations", type=int, default=4000,
                   help="daily closes required to count as a full history")
    p.add_argument("--sector-budget", type=int, default=700,
                   help="maximum per-ticker sector requests this run; yfinance "
                        "allows roughly 2000-2500/hour/IP and every lookup is "
                        "cached, so a large pool can be completed over runs")
    p.add_argument("--sector-sleep", type=float, default=0.0,
                   help="seconds to pause between sector lookups")
    p.add_argument("--no-sectors", action="store_true",
                   help="skip the sector lookup (disables stratified baskets)")
    args = p.parse_args()

    print("=" * 90)
    print("BUILDING THE RANDOM-BASKET POOL")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 90)

    print(f"\n1. Screening: US equities, volume > {args.min_volume:,.0f}, "
          f"price > ${args.min_price:.0f}")
    pool = screen_pool(args.min_volume, args.min_price, args.screen_cap)
    print(f"   {len(pool)} names")

    if "firstTradeDateMilliseconds" in pool.columns:
        cutoff = pd.Timestamp(args.warmup_start).timestamp() * 1000
        before = len(pool)
        pool = pool[pool["firstTradeDateMilliseconds"].fillna(9e15) <= cutoff]
        print(f"\n2. First trade before {args.warmup_start}: "
              f"{len(pool)} names ({before - len(pool)} dropped)")

    spent = 0
    if not args.no_sectors:
        print(f"\n3. Sectors for {len(pool)} names "
              f"(per-ticker request budget {args.sector_budget})")
        pool, spent = add_sectors(pool, args.sector_budget, args.sector_sleep)
        print(f"   {pool['sector'].notna().sum()}/{len(pool)} have a sector; "
              f"{spent} per-ticker requests spent")
    else:
        pool["sector"] = None

    print(f"\n4. Downloading prices to verify history "
          f"(>= {args.min_observations} closes)")
    import yfinance as yf
    syms = pool["symbol"].tolist()
    chunk, frames = 400, []
    for i in range(0, len(syms), chunk):
        part = syms[i:i + chunk]
        d = yf.download(part, start=args.warmup_start, interval="1d",
                        auto_adjust=True, progress=False, threads=True,
                        group_by="column")
        frames.append(d["Close"] if isinstance(d.columns, pd.MultiIndex) else d)
        print(f"    {min(i + chunk, len(syms))}/{len(syms)}")
    close = pd.concat(frames, axis=1)
    counts = close.notna().sum()
    good = counts[counts >= args.min_observations].index.tolist()
    pool = pool[pool["symbol"].isin(good)].reset_index(drop=True)
    print(f"   {len(pool)} names with a full history")

    pool.to_csv(REPO_ROOT / OUT_POOL, index=False)
    print(f"\nPool written to {REPO_ROOT / OUT_POOL}")

    print("\nSector composition:")
    vc = pool["sector"].value_counts(dropna=False)
    for sec, n in vc.items():
        print(f"    {str(sec):<26}{n:>5}  {n / len(pool):>6.1%}")
    if "marketCap" in pool.columns:
        mc = pool["marketCap"].dropna() / 1e9
        print(f"\nMarket cap ($B): median {mc.median():.1f}, "
              f"p10 {mc.quantile(.1):.1f}, p90 {mc.quantile(.9):.1f}, "
              f"max {mc.max():.0f}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
