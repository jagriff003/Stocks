"""
Point-in-time ticker universe (Request #1).

EDITING THE UNIVERSE
--------------------
Open `universe.csv` in the repo root — in Excel, or any text editor.  One row
per ticker:

    Symbol,Sector,Role,Active,Note
    NVDA,Information Technology,momentum,Y,NVIDIA (Semiconductors)

  Symbol  ticker as Yahoo Finance spells it (Berkshire is BRK-B, not BRK.B)
  Sector  free text; used for concentration reporting, not for selection
  Role    'momentum'  ranked normally
          'defensive' held in elevated/crisis regimes (SHY, TLT, IAU)
          'monitor'   downloaded and scored but never held (e.g. SH)
  Active  Y or N.  Set N to retire a ticker instead of deleting the row, so the
          record of what was once in the universe survives.
  Note    free text

Add a row, save, re-run.  Nothing else to change — no Python edits.

The file is deliberately exempted from the repo's blanket `*.csv` gitignore, so
universe changes show up as reviewable diffs.

WHY SNAPSHOTS
-------------
The universe is revised quarterly off a screener, and until now it lived only as
a hard-coded list in the live script.  That made live-vs-backtest reconciliation
an archaeology exercise: to ask "was this ticker even in the universe when that
trade happened?" you had to read old commits.  Dated snapshots put the answer on
file.

Two things this buys:

1. Reconciliation.  `load_universe(as_of=date(2026, 4, 17))` returns exactly the
   symbols the model could have chosen from on that date.

2. Honest backtesting.  Running today's screened list back to 2010 is
   survivorship-biased — the screener picked these names partly *because* they
   already won.  A point-in-time backtest that rebuilds the universe as of each
   rebalance is the unbiased version.  Snapshots are the prerequisite; we can
   only do this from the first snapshot forward, so historical numbers stay
   biased and should be read as such.

# REPLACE: the screener that produces each quarterly revision is external to
# this repo.  If it can be scripted, point `SCREENER_SOURCE` at it and have the
# quarterly refresh write `universe.csv` automatically instead of by hand.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import REPO_ROOT, UNIVERSE_DIR


SCREENER_SOURCE = "manual quarterly screen"  # REPLACE: see module docstring

# The one file to edit when the universe changes.
UNIVERSE_FILE = REPO_ROOT / "universe.csv"

UNIVERSE_COLUMNS = ["Symbol", "Sector", "Role", "Active", "Note"]

ROLE_MOMENTUM = "momentum"
ROLE_DEFENSIVE = "defensive"
ROLE_MONITOR = "monitor"


# Seed used to create universe.csv the first time, and nothing after that.
# Once the CSV exists it is the single source of truth — this list is not
# consulted again, so don't maintain it in parallel.
_SEED_UNIVERSE: List[Tuple[str, str, str, str]] = [
    # (symbol, sector, role, note)
    ("GOOGL", "Communication Services", ROLE_MOMENTUM, "Alphabet (Google)"),
    ("META",  "Communication Services", ROLE_MOMENTUM, "Meta"),
    ("LYV",   "Communication Services", ROLE_MOMENTUM, "Live Nation (Entertainment)"),

    ("AMZN",  "Consumer Discretionary", ROLE_MOMENTUM, "Amazon (Broad Retail)"),
    ("HD",    "Consumer Discretionary", ROLE_MOMENTUM, "Home Depot (Home Improvement)"),
    ("BKNG",  "Consumer Discretionary", ROLE_MOMENTUM, "Booking Holdings (Leisure/Travel)"),
    ("F",     "Consumer Discretionary", ROLE_MOMENTUM, "Ford (Auto)"),
    ("TSLA",  "Consumer Discretionary", ROLE_MOMENTUM, "Tesla"),

    ("COST",  "Consumer Staples", ROLE_MOMENTUM, "Costco (Merch)"),
    ("PM",    "Consumer Staples", ROLE_MOMENTUM, "Philip Morris (Tobacco)"),
    ("KR",    "Consumer Staples", ROLE_MOMENTUM, "Kroger (Food)"),
    ("KO",    "Consumer Staples", ROLE_MOMENTUM, "Coca-Cola (Beverage)"),

    ("XOM",   "Energy", ROLE_MOMENTUM, "Exxon Mobil"),
    ("MPC",   "Energy", ROLE_MOMENTUM, "Marathon Petroleum (Refining)"),

    ("JPM",   "Financials", ROLE_MOMENTUM, "JP Morgan Chase"),
    ("V",     "Financials", ROLE_MOMENTUM, "Visa (Transaction)"),
    ("AXP",   "Financials", ROLE_MOMENTUM, "American Express (Consumer)"),
    ("STT",   "Financials", ROLE_MOMENTUM, "State Street (Asset Servicing)"),
    ("PGR",   "Financials", ROLE_MOMENTUM, "Progressive (Insurance)"),
    ("BRK-B", "Financials", ROLE_MOMENTUM, "Berkshire Hathaway"),

    ("LLY",   "Health Care", ROLE_MOMENTUM, "Eli Lilly (Pharma)"),
    ("JNJ",   "Health Care", ROLE_MOMENTUM, "Johnson & Johnson (Pharma)"),
    ("ABBV",  "Health Care", ROLE_MOMENTUM, "AbbVie (Biotech)"),
    ("GILD",  "Health Care", ROLE_MOMENTUM, "Gilead Sciences (Biotech)"),
    ("CAH",   "Health Care", ROLE_MOMENTUM, "Cardinal Health (Distribution)"),
    ("MCK",   "Health Care", ROLE_MOMENTUM, "McKesson (Provider)"),

    ("GE",    "Industrials", ROLE_MOMENTUM, "GE Aerospace (Aero & Defense)"),
    ("CAT",   "Industrials", ROLE_MOMENTUM, "Caterpillar (Machinery)"),
    ("JCI",   "Industrials", ROLE_MOMENTUM, "Johnson Controls (Building Products)"),
    ("PWR",   "Industrials", ROLE_MOMENTUM, "Quanta Services (Infrastructure)"),
    ("URI",   "Industrials", ROLE_MOMENTUM, "United Rentals (Trading & Distribution)"),
    ("BR",    "Industrials", ROLE_MOMENTUM, "Broadridge Financial (Data Processing)"),

    ("NVDA",  "Information Technology", ROLE_MOMENTUM, "NVIDIA (Semiconductors)"),
    ("TSM",   "Information Technology", ROLE_MOMENTUM, "TSMC (Semiconductors)"),
    ("AAPL",  "Information Technology", ROLE_MOMENTUM, "Apple (Hardware)"),
    ("MSFT",  "Information Technology", ROLE_MOMENTUM, "Microsoft (Software)"),
    ("IBM",   "Information Technology", ROLE_MOMENTUM, "IBM (Services)"),
    ("STX",   "Information Technology", ROLE_MOMENTUM, "Seagate (Data Storage)"),

    ("LIN",   "Materials", ROLE_MOMENTUM, "Linde (Chemical)"),
    ("CRH",   "Materials", ROLE_MOMENTUM, "CRH (Construction)"),
    ("AA",    "Materials", ROLE_MOMENTUM, "Alcoa (Metals)"),
    ("NEM",   "Materials", ROLE_MOMENTUM, "Newmont (Gold Miner)"),

    ("WELL",  "Real Estate", ROLE_MOMENTUM, "Welltower (Health Care REIT)"),
    ("SPG",   "Real Estate", ROLE_MOMENTUM, "Simon Property Group (Retail REIT)"),
    ("CBRE",  "Real Estate", ROLE_MOMENTUM, "CBRE (Services)"),

    ("CNP",   "Utilities", ROLE_MOMENTUM, "CenterPoint Energy (Electric & Gas)"),

    ("HYG",   "Credit",      ROLE_MOMENTUM, "High Yield Bond ETF"),
    ("IBIT",  "Alternative", ROLE_MOMENTUM, "Bitcoin Trust"),

    # Defensive sleeve — the fill for elevated and crisis regimes.  These stay
    # in the universe even when they never rank on momentum.
    ("IAU",   "Defensive", ROLE_DEFENSIVE, "Gold ETF"),
    ("TLT",   "Defensive", ROLE_DEFENSIVE, "Long-Term Treasury ETF"),
    ("SHY",   "Defensive", ROLE_DEFENSIVE, "Short-Term Treasury ETF"),

    ("SH",    "Hedge", ROLE_MOMENTUM, "Short SPY (monitor; signal to stay out when above -0.1)"),
]


def _write_universe_file(rows: List[Dict[str, str]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=UNIVERSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def ensure_universe_file(path: Optional[Path] = None) -> Path:
    """
    Create `universe.csv` from the seed list if it does not exist yet.

    Idempotent — never overwrites an existing file, because that file is the
    source of truth once created.
    """
    path = Path(path or UNIVERSE_FILE)
    if path.exists():
        return path

    rows = [
        {"Symbol": sym, "Sector": sector, "Role": role, "Active": "Y", "Note": note}
        for sym, sector, role, note in _SEED_UNIVERSE
    ]
    _write_universe_file(rows, path)
    print(f"Created {path.name} with {len(rows)} tickers — edit this file to "
          f"change the universe.")
    return path


def read_universe_file(path: Optional[Path] = None,
                       include_inactive: bool = False) -> List[Dict[str, str]]:
    """
    Read `universe.csv` as a list of row dicts, creating it from seed if absent.

    Tolerates the things a human editing a spreadsheet actually does: stray
    whitespace, lowercase 'y', blank rows, missing optional columns.
    """
    path = ensure_universe_file(path)

    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for raw in csv.DictReader(fh):
            symbol = (raw.get("Symbol") or "").strip().upper()
            if not symbol or symbol.startswith("#"):
                continue

            active = (raw.get("Active") or "Y").strip().upper()
            if not include_inactive and active not in ("Y", "YES", "TRUE", "1"):
                continue

            role = (raw.get("Role") or ROLE_MOMENTUM).strip().lower()
            if role not in (ROLE_MOMENTUM, ROLE_DEFENSIVE, ROLE_MONITOR):
                raise ValueError(
                    f"{path.name}: ticker {symbol} has unknown Role {role!r}; "
                    f"expected one of {ROLE_MOMENTUM}/{ROLE_DEFENSIVE}/{ROLE_MONITOR}"
                )

            rows.append({
                "Symbol": symbol,
                "Sector": (raw.get("Sector") or "").strip(),
                "Role": role,
                "Active": active,
                "Note": (raw.get("Note") or "").strip(),
            })

    duplicates = {s for s in (r["Symbol"] for r in rows)
                  if [r["Symbol"] for r in rows].count(s) > 1}
    if duplicates:
        raise ValueError(f"{path.name}: duplicate tickers {sorted(duplicates)}")

    if not rows:
        raise ValueError(f"{path.name}: no active tickers found")

    return rows


def current_symbols(path: Optional[Path] = None) -> List[str]:
    """Every active ticker to download and score, in file order."""
    return [r["Symbol"] for r in read_universe_file(path)]


def defensive_symbols(path: Optional[Path] = None) -> List[str]:
    """Tickers marked Role=defensive — the elevated/crisis fill."""
    return [r["Symbol"] for r in read_universe_file(path)
            if r["Role"] == ROLE_DEFENSIVE]


def momentum_symbols(path: Optional[Path] = None) -> List[str]:
    """Tickers that compete for a slot on momentum rank."""
    return [r["Symbol"] for r in read_universe_file(path)
            if r["Role"] == ROLE_MOMENTUM]


def sector_map(path: Optional[Path] = None) -> Dict[str, str]:
    """Symbol -> sector, for concentration reporting."""
    return {r["Symbol"]: r["Sector"] for r in read_universe_file(path)}


def snapshot_universe(rows: List[Dict[str, str]],
                      as_of: Optional[date] = None,
                      source: str = SCREENER_SOURCE) -> Path:
    """
    Write a dated universe snapshot to snapshots/universe/.

    Overwrites an existing snapshot for the same date — snapshotting twice in
    one day should not leave two competing records.  Returns the path.
    """
    as_of = as_of or date.today()
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    path = UNIVERSE_DIR / f"universe_{as_of.isoformat()}.csv"

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(UNIVERSE_COLUMNS + ["AsOf", "Source"])
        for row in rows:
            writer.writerow([row.get(c, "") for c in UNIVERSE_COLUMNS]
                            + [as_of.isoformat(), source])

    return path


def snapshot_current_universe(as_of: Optional[date] = None,
                              source: str = SCREENER_SOURCE) -> Path:
    """
    Snapshot whatever `universe.csv` currently holds.

    Called automatically on every live model run, so the record accumulates
    without you having to remember to do it after a quarterly revision.
    """
    return snapshot_universe(read_universe_file(include_inactive=True),
                             as_of=as_of, source=source)


def list_snapshots() -> List[Tuple[date, Path]]:
    """All universe snapshots on disk, oldest first."""
    if not UNIVERSE_DIR.exists():
        return []
    out = []
    for path in sorted(UNIVERSE_DIR.glob("universe_*.csv")):
        try:
            snap_date = date.fromisoformat(path.stem.removeprefix("universe_"))
        except ValueError:
            continue
        out.append((snap_date, path))
    return out


def load_universe(as_of: Optional[date] = None,
                  fallback_to_current: bool = True) -> List[str]:
    """
    Return the universe in effect on `as_of` — the most recent snapshot on or
    before that date.

    With `as_of=None`, returns the latest snapshot.  If no snapshot applies and
    `fallback_to_current` is set, falls back to today's `universe.csv`; that
    keeps callers working before any snapshot exists, at the cost of silently
    reintroducing survivorship bias, so pass `fallback_to_current=False`
    anywhere that distinction matters.
    """
    snapshots = list_snapshots()

    if as_of is not None:
        snapshots = [(d, p) for d, p in snapshots if d <= as_of]

    if not snapshots:
        if fallback_to_current:
            return current_symbols()
        raise FileNotFoundError(
            f"No universe snapshot on or before "
            f"{as_of.isoformat() if as_of else 'today'}"
        )

    _, path = snapshots[-1]
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return [row["Symbol"] for row in csv.DictReader(fh)
                if row.get("Symbol")
                and (row.get("Active") or "Y").strip().upper() in ("Y", "YES", "TRUE", "1")]


def universe_diff(earlier: date, later: date) -> Dict[str, List[str]]:
    """
    What changed between two snapshot dates.

    Answers "was this ticker in the universe at the time?" for a whole quarter
    at once — the question that made the last reconciliation painful.
    """
    before = set(load_universe(as_of=earlier, fallback_to_current=False))
    after = set(load_universe(as_of=later, fallback_to_current=False))
    return {
        "added": sorted(after - before),
        "removed": sorted(before - after),
        "retained": sorted(before & after),
    }
