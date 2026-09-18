"""
Hard trading restrictions.  Compliance, not strategy.

Some names must never be recommended, held, or even scored, regardless of what
any model says.  The operator works for Digital Realty (DLR), so DLR, its
competitors and the data-center sector generally are off limits — not because
trading them would necessarily be unlawful, but because the appearance of
insider trading is itself the thing being avoided, and an algorithmic
justification is no defence.

THIS IS NOT A PREFERENCE AND IT DOES NOT DEGRADE GRACEFULLY
   Every other exclusion in this package is a modelling judgment that trades off
   against return.  This one does not.  A restricted name appearing in a
   recommendation is a failure regardless of its effect on CAGR, so the
   functions here RAISE rather than filter-and-continue.  A silent filter would
   let a restricted name be quietly dropped and never noticed; a loud failure
   forces someone to look.

TWO KEYS, BECAUSE SYMBOLS ARE NOT STABLE
   Blocking by ticker alone is fragile.  COR was CyrusOne, a data-center REIT,
   until it was acquired — the ticker now belongs to Cencora, a pharmaceutical
   distributor with no connection to the sector.  A symbol blocklist would
   wrongly exclude Cencora while a newly listed data-center REIT under a fresh
   ticker would sail straight through.

   So restriction works on two keys:
     - explicit SYMBOLS, for named entities
     - SUB-INDUSTRY / INDUSTRY strings, which survive ticker churn and catch
       new entrants automatically

   The industry key needs screener metadata, which the live universe file does
   not carry; `check_screen` applies it where that metadata exists (the Schwab
   export), and `check_symbols` applies the symbol key everywhere.

MAINTENANCE
   The lists live in `restricted.csv` at the repo root so they can be edited
   without touching code, and so a change to them shows up in a diff.  When in
   doubt, add the name: the cost of an unnecessary exclusion is a fraction of a
   percent of return, and the cost of a necessary one missed is not measured in
   percent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESTRICTED_FILE = REPO_ROOT / "restricted.csv"


class RestrictedNameError(RuntimeError):
    """A restricted name reached somewhere it must never reach."""


def load_restrictions(path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Read the restriction lists.

    Returns {"symbols": [...], "industries": [...], "reasons": {sym: reason}}.
    A missing file is an error rather than an empty default: silently applying
    no restrictions because a file was renamed is exactly the failure this
    module exists to prevent.
    """
    path = Path(path) if path else RESTRICTED_FILE
    if not path.exists():
        raise RestrictedNameError(
            f"Restriction list not found at {path}. Refusing to run without "
            f"it — an empty default would silently permit restricted names.")
    df = pd.read_csv(path)
    for col in ("Kind", "Value"):
        if col not in df.columns:
            raise RestrictedNameError(
                f"{path} must have columns Kind,Value,Reason; got "
                f"{list(df.columns)}")
    syms = [str(v).strip().upper() for k, v in zip(df["Kind"], df["Value"])
            if str(k).strip().lower() == "symbol"]
    inds = [str(v).strip().lower() for k, v in zip(df["Kind"], df["Value"])
            if str(k).strip().lower() in ("industry", "sub-industry")]
    reasons = {str(v).strip().upper(): str(r)
               for k, v, r in zip(df["Kind"], df["Value"],
                                  df.get("Reason", [""] * len(df)))
               if str(k).strip().lower() == "symbol"}
    return {"symbols": syms, "industries": inds, "reasons": reasons}


def check_symbols(symbols: Iterable[str], where: str,
                  path: Optional[Path] = None) -> None:
    """Raise if any restricted symbol appears in `symbols`."""
    r = load_restrictions(path)
    blocked = sorted({s.strip().upper() for s in symbols} & set(r["symbols"]))
    if blocked:
        detail = "; ".join(f"{b} ({r['reasons'].get(b, 'restricted')})"
                           for b in blocked)
        raise RestrictedNameError(
            f"Restricted name(s) present in {where}: {detail}. "
            f"This is a compliance restriction, not a modelling preference — "
            f"remove them rather than overriding this check.")


def filter_screen(frame: pd.DataFrame, symbol_col: str = "symbol",
                  industry_cols: Sequence[str] = ("industry", "sub-industry"),
                  path: Optional[Path] = None) -> tuple:
    """
    Drop restricted rows from a screener frame.

    Returns (kept, dropped).  Filtering rather than raising is correct HERE and
    only here: a screener export is an external list nobody curated, so
    restricted names appearing in it is expected rather than a failure.  The
    dropped frame is returned so the caller can report exactly what was removed
    and why — a restriction that is applied invisibly cannot be audited.
    """
    r = load_restrictions(path)
    cols = {c.lower(): c for c in frame.columns}
    sym = cols.get(symbol_col.lower())
    if sym is None:
        raise RestrictedNameError(f"No {symbol_col!r} column in the screen")

    mask = frame[sym].astype(str).str.strip().str.upper().isin(r["symbols"])
    for ic in industry_cols:
        col = cols.get(ic.lower())
        if col is not None:
            vals = frame[col].astype(str).str.strip().str.lower()
            for bad in r["industries"]:
                mask = mask | vals.eq(bad)
    return frame[~mask].copy(), frame[mask].copy()


def describe(path: Optional[Path] = None) -> str:
    r = load_restrictions(path)
    lines = ["Trading restrictions in force (compliance):"]
    for s in sorted(r["symbols"]):
        lines.append(f"    {s:<8} {r['reasons'].get(s, '')}")
    for i in sorted(r["industries"]):
        lines.append(f"    [industry] {i}")
    return "\n".join(lines)
