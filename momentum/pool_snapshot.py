"""
Point-in-time snapshots of the eligible pool.

WHY THIS EXISTS

The Track J result rests on a pool built in 2026 from names that still exist.
The score buys pullbacks, so every dip in the sample was followed by a recovery —
because the dips that were terminal are not in the data.  The delisting bound
(TODO 0d) tried to quantify that and came out inconclusive: the synthetic names
distorted the pool in both directions by more than the effect being measured.

Only point-in-time data with delisted securities settles it.  Short of buying
that (CRSP, Norgate, Sharadar), the free option is to **start recording one**.
Every rebalance, write down which names were tradable.  Names that appear in
older snapshots and vanish from newer ones are exactly the delistings the
backtest could never see, and after a few years the snapshots ARE the
point-in-time dataset.

**This is worthless retroactively.** That is the entire argument for starting it
before the model goes live rather than after.

It mirrors `universe.snapshot_universe`, deliberately: same dated-CSV
convention, same overwrite-per-day rule, same "source, not output" status in
.gitignore so changes show up as reviewable diffs.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .config import REPO_ROOT

POOL_DIR = REPO_ROOT / "snapshots" / "pool"
POOL_COLUMNS = ["Symbol", "Tradable", "DollarVolume", "Price", "Score", "Rank"]


def snapshot_pool(tradable: pd.Series,
                  dollar_volume: Optional[pd.Series] = None,
                  price: Optional[pd.Series] = None,
                  scores: Optional[pd.Series] = None,
                  as_of: Optional[date] = None,
                  label: str = "") -> Path:
    """
    Write one dated snapshot of the eligible pool.

    `tradable` is the boolean screen result for the session, indexed by symbol.
    The other columns are recorded because they are what would let a future
    reader reconstruct WHY a name was in or out — a bare symbol list would say
    that a name disappeared but not whether it failed the volume floor, failed
    the price floor, or stopped existing.

    Overwrites an existing snapshot for the same date, matching
    `universe.snapshot_universe`: snapshotting twice in one day should not leave
    two competing records.
    """
    as_of = as_of or date.today()
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    name = f"pool_{as_of.isoformat()}{'_' + label if label else ''}.csv"
    path = POOL_DIR / name

    symbols = list(tradable.index)
    rank = None
    if scores is not None:
        rank = scores.reindex(symbols).rank(ascending=False, method="min")

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(POOL_COLUMNS + ["AsOf"])
        for sym in symbols:
            w.writerow([
                sym,
                "Y" if bool(tradable.get(sym, False)) else "N",
                _fmt(dollar_volume, sym),
                _fmt(price, sym),
                _fmt(scores, sym),
                _fmt(rank, sym, int_like=True),
                as_of.isoformat(),
            ])
    return path


def _fmt(series: Optional[pd.Series], sym: str, int_like: bool = False) -> str:
    if series is None or sym not in series.index:
        return ""
    v = series[sym]
    if pd.isna(v):
        return ""
    return str(int(v)) if int_like else f"{float(v):.6g}"


def list_snapshots() -> List[Path]:
    """Every pool snapshot on disk, oldest first."""
    if not POOL_DIR.exists():
        return []
    return sorted(POOL_DIR.glob("pool_*.csv"))


def read_snapshot(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def vanished(earlier: Path, later: Path) -> Dict[str, List[str]]:
    """
    Names present in `earlier` and absent from `later`, and vice versa.

    A name that vanishes is a *candidate* delisting — it may also have been
    dropped from the screener export, renamed, or merged. Which is why the
    snapshot records volume and price: a name that vanished after months of
    falling price and falling volume is a different story from one that vanished
    while trading normally, and only the first is the bias Track J cares about.
    """
    a = set(read_snapshot(earlier)["Symbol"])
    b = set(read_snapshot(later)["Symbol"])
    return {"gone": sorted(a - b), "new": sorted(b - a)}


def coverage() -> pd.DataFrame:
    """
    One row per snapshot: date, how many names, how many tradable.

    The point of printing this is to make the record's shortness visible.  Until
    it spans years it cannot answer the survivorship question, and a table that
    says "3 snapshots" is harder to over-read than a bare reassurance that
    snapshotting is happening.
    """
    rows = []
    for p in list_snapshots():
        d = read_snapshot(p)
        rows.append({
            "File": p.name,
            "AsOf": d["AsOf"].iloc[0] if len(d) else "",
            "Names": len(d),
            "Tradable": int((d["Tradable"] == "Y").sum()) if len(d) else 0,
        })
    return pd.DataFrame(rows)
