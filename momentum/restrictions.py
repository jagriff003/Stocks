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

FOUR KEYS, BECAUSE SYMBOLS ARE NOT STABLE
   Blocking by ticker alone is fragile.  COR was CyrusOne, a data-center REIT,
   until it was acquired — the ticker now belongs to Cencora, a pharmaceutical
   distributor with no connection to the sector.  A symbol blocklist would
   wrongly exclude Cencora while a newly listed data-center REIT under a fresh
   ticker would sail straight through.

   So restriction works on four keys:
     - SYMBOL, a US-listed ticker, for named entities
     - FOREIGN-SYMBOL, an exchange-qualified foreign line such as "A17U SP".
       These never appear in our US price data, so they can never be matched
       by ticker.  They are carried because the list must stay auditable
       against the source DLR sent, and because the NAME key below is what
       actually catches them if they ever list here.
     - NAME, the issuer name, matched against the description text a screener
       carries.  This is the only key that works across listings: Keppel DC
       REIT is AJBU in Singapore and would be something else entirely as an
       ADR, but it is "Keppel DC" in both.
     - SUB-INDUSTRY / INDUSTRY strings, which survive ticker churn and catch
       new entrants automatically.

   The industry and name keys need screener metadata, which the live universe
   file does not carry; `check_screen` applies them where that metadata exists
   (the Schwab export, the liquid pool), and `check_symbols` applies the symbol
   key everywhere.

WHY THERE IS ONLY ONE INDUSTRY KEY
   An industry key is the strongest tool here, and for that reason the wrong
   one is expensive.  In the Schwab taxonomy only "Data Center REITs" maps
   cleanly onto this list.  "Internet Services & Infrastructure" was considered
   and rejected: in the current screen it contains Cloudflare and Verisign and
   no restricted name, so adding it would block two unrelated businesses and
   catch nothing.  The crypto/AI-hosting names have no shared sub-industry at
   all — they are scattered across the taxonomy — which is exactly why they are
   enumerated by symbol.

CATEGORY IS FOR PEOPLE, NOT FOR MATCHING
   The Category column groups the list into blocks a human can review — is the
   whole crypto/AI-hosting block still right? did a new data-center REIT get
   added? — and is deliberately NOT used for matching.  Matching on a label we
   assign ourselves would only restate the symbol list.

MAINTENANCE
   The lists live in `restricted.csv` at the repo root so they can be edited
   without touching code, and so a change to them shows up in a diff.  It has
   five columns — Kind, Value, Name, Category, Reason — and adding a name is
   one line; nothing here needs to change to add one.  When in doubt, add the
   name: the cost of an unnecessary exclusion is a fraction of a percent of
   return, and the cost of a necessary one missed is not measured in percent.

   `python -m momentum.restrictions` prints the list as it is actually parsed,
   which is the way to check that an edit did what was intended.

NOTES ON PARTICULAR ENTRIES
   KEEL is on DLR's list but does not resolve to an issuer in any data we hold.
   It is restricted anyway — blocking an unidentified ticker costs nothing and
   the alternative is dropping a name DLR named.  If it is ever identified,
   give it a Name so the name key can work.

   CDP and FRMI are restricted on the same basis: read as COPT Defense
   Properties and Fermi, but neither appears in our data to confirm it.  CDP in
   particular is a three-letter ticker of exactly the kind that gets reassigned
   — the COR/Cencora case above — so the Name column is what should be trusted
   if the two ever disagree.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESTRICTED_FILE = REPO_ROOT / "restricted.csv"


class RestrictedNameError(RuntimeError):
    """A restricted name reached somewhere it must never reach."""


def _normalize(text: str) -> str:
    """Lowercase, and reduce anything that is not alphanumeric to one space.

    Screener descriptions and issuer names disagree about punctuation far more
    often than they disagree about words: "Digital Realty Trust, Inc." and
    "Digital Realty Trust Inc" are the same company written two ways.
    Normalizing both sides removes that whole class of near-miss.
    """
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


# Corporate-form words carry no identifying information, and screeners append
# them inconsistently.  Stripping them shortens the needle, which makes a match
# MORE likely, not less — the risk runs toward over-matching, so the needle is
# still required to be reasonably long and to land on word boundaries.
_SUFFIXES = ("inc", "corp", "corporation", "co", "company", "ltd", "limited",
             "plc", "group", "holdings", "holding", "trust", "reit", "sa",
             "nv", "ag", "lp", "the")


def _needle(name: str) -> str:
    """The normalized, suffix-stripped form of an issuer name."""
    words = _normalize(name).split()
    while len(words) > 1 and words[-1] in _SUFFIXES:
        words.pop()
    return " ".join(words)


def load_restrictions(path: Optional[Path] = None) -> Dict[str, object]:
    """
    Read the restriction lists.

    Returns a dict with:
      symbols      US tickers that must never appear anywhere
      foreign      exchange-qualified foreign lines, for audit only
      names        {needle: display name} for description matching
      industries   lowercased industry / sub-industry strings
      reasons      {key: reason}
      categories   {key: category}
      entries      the parsed rows, for `describe`

    A missing file is an error rather than an empty default: silently applying
    no restrictions because a file was renamed is exactly the failure this
    module exists to prevent.
    """
    path = Path(path) if path else RESTRICTED_FILE
    if not path.exists():
        raise RestrictedNameError(
            f"Restriction list not found at {path}. Refusing to run without "
            f"it — an empty default would silently permit restricted names.")
    df = pd.read_csv(path).fillna("")
    for col in ("Kind", "Value"):
        if col not in df.columns:
            raise RestrictedNameError(
                f"{path} must have columns Kind,Value,Name,Category,Reason; "
                f"got {list(df.columns)}")
    for col in ("Name", "Category", "Reason"):
        if col not in df.columns:
            df[col] = ""

    symbols: List[str] = []
    foreign: List[str] = []
    industries: List[str] = []
    names: Dict[str, str] = {}
    reasons: Dict[str, str] = {}
    categories: Dict[str, str] = {}
    entries: List[dict] = []

    for _, row in df.iterrows():
        kind = str(row["Kind"]).strip().lower()
        value = str(row["Value"]).strip()
        name = str(row["Name"]).strip()
        entries.append({"kind": kind, "value": value, "name": name,
                        "category": str(row["Category"]).strip(),
                        "reason": str(row["Reason"]).strip()})
        if kind == "symbol":
            key = value.upper()
            symbols.append(key)
        elif kind == "foreign-symbol":
            key = value.upper()
            foreign.append(key)
        elif kind in ("industry", "sub-industry"):
            industries.append(value.lower())
            continue
        else:
            raise RestrictedNameError(
                f"{path}: unknown Kind {row['Kind']!r}. Expected symbol, "
                f"foreign-symbol, industry or sub-industry. Refusing to run "
                f"rather than ignore a row that was meant to restrict "
                f"something.")
        reasons[key] = str(row["Reason"]).strip()
        categories[key] = str(row["Category"]).strip()
        # A one- or two-character needle would match half the market, so short
        # names are carried for display but not used for matching.  KEEL, which
        # has no issuer name at all, lands here too.
        if name:
            n = _needle(name)
            if len(n) >= 4:
                names[n] = name

    if not symbols:
        raise RestrictedNameError(
            f"{path} parsed to zero restricted symbols. Refusing to run: a "
            f"malformed file must not read as 'nothing is restricted'.")
    return {"symbols": symbols, "foreign": foreign, "names": names,
            "industries": industries, "reasons": reasons,
            "categories": categories, "entries": entries}


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


def match_names(descriptions: Iterable[str],
                path: Optional[Path] = None) -> List[Optional[str]]:
    """
    For each description, the restricted issuer name it contains, or None.

    Matching is on word boundaries over normalized text, so "Equinix" hits
    "Equinix, Inc." but "IREN" does not hit "Environmental Services".
    """
    r = load_restrictions(path)
    needles = [(re.compile(rf"\b{re.escape(n)}\b"), disp)
               for n, disp in r["names"].items()]
    out: List[Optional[str]] = []
    for d in descriptions:
        norm = _normalize(d)
        hit = next((disp for pat, disp in needles if pat.search(norm)), None)
        out.append(hit)
    return out


def filter_screen(frame: pd.DataFrame, symbol_col: str = "symbol",
                  industry_cols: Sequence[str] = ("industry", "sub-industry"),
                  name_cols: Sequence[str] = ("description", "longname",
                                              "shortname", "name", "security"),
                  path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop restricted rows from a screener frame.

    Returns (kept, dropped).  `dropped` carries a `restricted_by` column saying
    which key fired — symbol, industry or name — because a restriction that is
    applied invisibly cannot be audited, and because a name match is the one
    that can plausibly be wrong and so is the one worth eyeballing.

    Filtering rather than raising is correct HERE and only here: a screener
    export is an external list nobody curated, so restricted names appearing in
    it is expected rather than a failure.
    """
    r = load_restrictions(path)
    cols = {c.lower(): c for c in frame.columns}
    sym = cols.get(symbol_col.lower())
    if sym is None:
        raise RestrictedNameError(f"No {symbol_col!r} column in the screen")

    why = pd.Series([None] * len(frame), index=frame.index, dtype=object)

    hit = frame[sym].astype(str).str.strip().str.upper().isin(r["symbols"])
    why[hit & why.isna()] = "symbol"

    for ic in industry_cols:
        col = cols.get(ic.lower())
        if col is None:
            continue
        vals = frame[col].astype(str).str.strip().str.lower()
        for bad in r["industries"]:
            why[vals.eq(bad) & why.isna()] = f"industry: {bad}"

    for nc in name_cols:
        col = cols.get(nc.lower())
        if col is None:
            continue
        found = match_names(frame[col].astype(str), path=path)
        for idx, f in zip(frame.index, found):
            if f and why.get(idx) is None:
                why[idx] = f"name: {f}"

    mask = why.notna()
    dropped = frame[mask].copy()
    dropped["restricted_by"] = why[mask]
    return frame[~mask].copy(), dropped


# The screen and the pool are the same shape of problem — an external list
# nobody curated — so this is the name to reach for when filtering either.
check_screen = filter_screen


def describe(path: Optional[Path] = None) -> str:
    """The list as it is actually parsed, grouped the way it is maintained."""
    r = load_restrictions(path)
    entries = r["entries"]
    cats: Dict[str, List[dict]] = {}
    for e in entries:
        cats.setdefault(e["category"] or "uncategorized", []).append(e)

    n_sym, n_for = len(r["symbols"]), len(r["foreign"])
    lines = [f"Trading restrictions in force (compliance): {n_sym} US symbols, "
             f"{n_for} foreign lines, {len(r['industries'])} industry, "
             f"{len(r['names'])} name."]
    for cat in sorted(cats):
        lines.append(f"  [{cat}]")
        for e in sorted(cats[cat], key=lambda x: (x["kind"], x["value"])):
            tag = {"symbol": "", "foreign-symbol": " (foreign)",
                   "sub-industry": " (industry)",
                   "industry": " (industry)"}.get(e["kind"], "")
            label = f"{e['value']}{tag}"
            lines.append(f"    {label:<28} {e['name']}")
    lines.append("  Foreign lines cannot be matched by ticker in US data; they "
                 "are caught, if at all, by the name key.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
