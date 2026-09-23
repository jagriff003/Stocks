"""
A persistent, append-only store of daily total returns for the instruments
the Track K hedge layer decides on.

WHY A STORE AND NOT A CACHE

`.price_cache` is throwaway: it is re-downloaded whole when stale and nothing
checks that today's download agrees with yesterday's.  The hedge layer decides
live from these series, so they need to be present, append cheaply every
session, and tell you when history changes underneath you.

WHY RETURNS, NOT PRICES

Yahoo's adjusted close is re-scaled back through all of history every time a
dividend goes ex.  Appending adjusted PRICES would splice two different scales
together at the seam, silently.  A daily RETURN computed within a single fetch
is invariant to that re-scaling, so returns are the unit that can be appended.
Price levels are recoverable as an index by compounding.

WHAT AN UPDATE CHECKS (and the run's exit status reports — RUNBOOK convention)

  overlap    each update re-fetches the last `OVERLAP_DAYS` sessions already
             stored and compares them.  A difference above `REVISION_TOL` is a
             REVISION: the symbol is not written, and the run exits 2, unless
             `accept_revisions=True`.  Late-posted dividends land here.
  settled    a bar for today is dropped before 16:15 ET, when RUNBOOK measured
             Yahoo's daily bar to have settled.
  stale      every symbol should end on the same session as SPY.
  gaps       no missing session inside a symbol's history where SPY has one.
  moves      a new daily return beyond `BIG_MOVE` is flagged for a look, not
             blocked — 2020 and 2008 had real ones (and SLV's -28.5% on
             2026-01-30 is real: SIVR and silver futures agree).
  holes      Yahoo sometimes has no bar for a session that SPY has — found on
             the first build, 2026-09-22, for nine ETFs at once.  Without a
             fill the next day's return cannot be computed and the move across
             the hole is lost for good.  Holes of up to `MAX_FILL` sessions are
             carried at the last price (0% that day, the whole move the next)
             and flagged FILLED; if Yahoo later posts the real bar, the overlap
             check sees it as a revision and asks.

`verify_full_history` re-fetches everything and compares the whole store
without writing: a monthly audit for revisions older than the overlap window.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = REPO_ROOT / "data" / "market"
RETURNS_FILE = "daily_returns.csv"
LOG_FILE = "update_log.csv"

REFERENCE = "SPY"            # the session calendar every symbol is checked against
OVERLAP_DAYS = 15
REVISION_TOL = 1e-5          # 0.1bp of daily return
BIG_MOVE = 0.20
MAX_FILL = 3                 # sessions of Yahoo hole carried at the last price
SETTLE_ET = dtime(16, 15)    # RUNBOOK: the daily bar has settled by 16:15 ET


@dataclass(frozen=True)
class Instrument:
    role: str          # the hedge-layer column it feeds
    backtest: str      # longest-history symbol, used for research
    live: str          # what would actually be traded
    note: str = ""


# The agreed menu (TODO 0l).  `backtest` and `live` differ only where the live
# choice is younger than the history the research needs.
INSTRUMENTS: List[Instrument] = [
    Instrument("GOLD", "IAU", "IAU"),
    Instrument("SILVER", "SLV", "SLV"),
    Instrument("CMDTY", "DBC", "PDBC", "same index family; PDBC avoids the K-1 (moot in a tax-advantaged account)"),
    Instrument("ENERGY", "XLE", "XLE", "Tier 1 may use the pool's own energy basket instead"),
    Instrument("REALEST", "VNQ", "VNQ", "not data-center focused, so within compliance"),
    Instrument("TIPS", "TIP", "SCHP", "VTIP (short TIPS) also stored, from 2012"),
    Instrument("UST_S", "SHY", "SHY"),
    Instrument("UST10", "IEF", "IEF"),
    Instrument("UST_L", "TLT", "TLT"),
    Instrument("DOLLAR", "UUP", "UUP"),
    Instrument("CASH", "BIL", "BIL"),
]
EXTRA = ["VTIP", REFERENCE]
DEFAULT_SYMBOLS = sorted({s for i in INSTRUMENTS for s in (i.backtest, i.live)} | set(EXTRA))

Fetcher = Callable[[Sequence[str], str], pd.DataFrame]


def _yahoo_adjusted_close(symbols: Sequence[str], start: str) -> pd.DataFrame:
    import warnings
    import yfinance as yf
    warnings.filterwarnings("ignore")
    px = yf.download(list(symbols), start=start, auto_adjust=True, progress=False,
                     threads=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(symbols[0])
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    return px


def _now_et() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


# --------------------------------------------------------------------------
# read / write
# --------------------------------------------------------------------------

def load_daily_returns(symbols: Optional[Sequence[str]] = None,
                       store_dir: Path = STORE_DIR) -> pd.DataFrame:
    path = Path(store_dir) / RETURNS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist - build it with `python scripts/update_market_data.py`")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if symbols is not None:
        missing = [s for s in symbols if s not in df.columns]
        if missing:
            raise KeyError(f"not in the store: {missing} - add with "
                           f"`python scripts/update_market_data.py --add {' '.join(missing)}`")
        df = df[list(symbols)]
    return df


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, float_format="%.10g", index_label="Date")
    os.replace(tmp, path)


def _append_log(rows: List[dict], store_dir: Path) -> None:
    path = Path(store_dir) / LOG_FILE
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)


def _returns_from_prices(px: pd.DataFrame) -> pd.DataFrame:
    """Daily return within one fetch; each column's first priced day has none."""
    return px.pct_change(fill_method=None).where(px.notna() & px.shift(1).notna())


def _returns_on_calendar(px: pd.Series, calendar: pd.DatetimeIndex) -> Tuple[pd.Series, list]:
    """
    Daily returns for one symbol on the reference calendar, with holes of up to
    MAX_FILL sessions carried at the last price.  Returns (returns, filled dates).
    The trailing edge is never filled: a missing LAST bar is unknown, not flat.
    """
    p = px.dropna()
    if p.empty:
        return p, []
    cal = calendar[(calendar >= p.index[0]) & (calendar <= p.index[-1])]
    full = p.reindex(cal.union(p.index))
    holes = full.index[full.isna()]
    full = full.ffill(limit=MAX_FILL)
    filled = [d for d in holes if pd.notna(full[d])]
    r = full.pct_change(fill_method=None).where(full.notna() & full.shift(1).notna())
    return r.dropna(), filled


def _drop_unsettled(px: pd.DataFrame, now_et: datetime) -> pd.DataFrame:
    today = pd.Timestamp(now_et.date())
    if now_et.time() < SETTLE_ET and today in px.index:
        return px.drop(index=today)
    return px


# --------------------------------------------------------------------------
# update
# --------------------------------------------------------------------------

def update_daily(symbols: Optional[Sequence[str]] = None, accept_revisions: bool = False,
                 store_dir: Path = STORE_DIR, fetch: Fetcher = _yahoo_adjusted_close,
                 now_et: Optional[datetime] = None) -> Tuple[pd.DataFrame, int]:
    """
    Bring every symbol up to the latest settled session.  Returns a per-symbol
    report and an exit status: 0 clean, 2 something needs a look (revision
    refused, stale, gap, big move).  Raises on outright failure.

    `symbols` defaults to everything already stored plus DEFAULT_SYMBOLS, so a
    symbol once added stays maintained.
    """
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    now_et = now_et or _now_et()
    path = store_dir / RETURNS_FILE
    stored = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()
    wanted = sorted(set(symbols or []) | set(DEFAULT_SYMBOLS if symbols is None else [])
                    | set(stored.columns) | {REFERENCE})

    known = [s for s in wanted if s in stored.columns and stored[s].notna().any()]
    new = [s for s in wanted if s not in known]

    prices = {}
    if known:
        last = min(stored[s].last_valid_index() for s in known)
        start = (last - pd.tseries.offsets.BDay(OVERLAP_DAYS + 5)).strftime("%Y-%m-%d")
        px = _drop_unsettled(fetch(known, start), now_et)
        prices.update({s: px[s] for s in known if s in px})
    if new:
        px = _drop_unsettled(fetch(new, "1990-01-01"), now_et)
        prices.update({s: px[s] for s in new if s in px})
    # the session calendar: every day SPY has, stored or just fetched
    calendar = pd.DatetimeIndex([])
    if REFERENCE in stored:
        calendar = calendar.union(stored[REFERENCE].dropna().index)
    if REFERENCE in prices:
        calendar = calendar.union(prices[REFERENCE].dropna().index)
    fetched, fills = {}, {}
    for sym, p in prices.items():
        fetched[sym], fills[sym] = _returns_on_calendar(p, calendar)

    out = stored.copy()
    report = []
    stamp = now_et.strftime("%Y-%m-%d %H:%M ET")
    for s in wanted:
        r = fetched.get(s)
        row = {"run": stamp, "symbol": s, "appended": 0, "overlap_days": 0,
               "max_overlap_diff": np.nan, "status": "ok", "detail": ""}

        def flag(status, detail):
            row["status"] = status if row["status"] == "ok" else f"{row['status']} | {status}"
            row["detail"] = detail if not row["detail"] else f"{row['detail']} | {detail}"

        if r is None or r.dropna().empty:
            row.update(status="FAILED", detail="no data returned")
            report.append(row)
            continue
        r = r.dropna()
        if s in out.columns and out[s].notna().any():
            old = out[s].dropna()
            common = old.index.intersection(r.index)
            # the first fetched return is only valid if its previous day was also fetched
            diff = (r.loc[common] - old.loc[common]).abs()
            row["overlap_days"] = int(len(common))
            row["max_overlap_diff"] = float(diff.max()) if len(diff) else np.nan
            if len(common) == 0:
                flag("GAP", "fetch does not overlap the stored history")
                report.append(row)
                continue
            revised = diff[diff > REVISION_TOL]
            if len(revised):
                detail = ", ".join(f"{d:%Y-%m-%d} {old[d]:+.5f}->{r[d]:+.5f}" for d in revised.index[:3])
                if not accept_revisions:
                    flag("REVISION REFUSED", detail)
                    report.append(row)
                    continue
                flag("revision accepted", detail)
                out.loc[revised.index, s] = r.loc[revised.index]
            fresh = r.loc[r.index > old.index[-1]]
        else:
            fresh = r
        if len(fresh):
            out = out.reindex(out.index.union(fresh.index))
            out.loc[fresh.index, s] = fresh
        row["appended"] = int(len(fresh))
        new_fills = [d for d in fills.get(s, []) if d in fresh.index]
        if new_fills:
            flag("FILLED", f"{len(new_fills)} Yahoo hole(s) carried flat, last "
                           f"{new_fills[-1]:%Y-%m-%d}")
        big = fresh[fresh.abs() > BIG_MOVE]
        if len(big):
            flag("BIG MOVE", ", ".join(f"{d:%Y-%m-%d} {v:+.1%}" for d, v in big.items()))
        report.append(row)

    out = out.sort_index()
    out = out[sorted(out.columns)]
    rep = pd.DataFrame(report).set_index("symbol")

    # calendar checks against the reference, after the write set is final
    if REFERENCE in out and out[REFERENCE].notna().any():
        ref_last = out[REFERENCE].last_valid_index()
        ref_days = out[REFERENCE].dropna().index
        for s in out.columns:
            col = out[s]
            if not col.notna().any():
                continue
            flags = []
            if col.last_valid_index() != ref_last:
                flags.append(("STALE", f"ends {col.last_valid_index():%Y-%m-%d}, "
                                       f"{REFERENCE} {ref_last:%Y-%m-%d}"))
            span = ref_days[(ref_days >= col.first_valid_index()) & (ref_days <= col.last_valid_index())]
            holes = col.reindex(span).isna()
            if holes.any():
                flags.append(("GAP", f"{int(holes.sum())} missing sessions, first "
                                     f"{holes[holes].index[0]:%Y-%m-%d}"))
            for st, de in flags:
                cur, det = rep.at[s, "status"], rep.at[s, "detail"]
                rep.at[s, "status"] = st if cur == "ok" else f"{cur} | {st}"
                rep.at[s, "detail"] = de if not det else f"{det} | {de}"

    if (rep["status"] == "FAILED").all():
        raise RuntimeError("every symbol failed to fetch - nothing written")
    _atomic_write(out, path)
    _append_log(rep.reset_index().to_dict("records"), store_dir)
    status = 0 if rep["status"].isin(["ok", "revision accepted"]).all() else 2
    return rep, status


def verify_full_history(store_dir: Path = STORE_DIR,
                        fetch: Fetcher = _yahoo_adjusted_close) -> pd.DataFrame:
    """Re-fetch everything and compare the whole store.  Writes nothing."""
    stored = load_daily_returns(store_dir=store_dir)
    px = fetch(list(stored.columns), "1990-01-01")
    fresh = _returns_from_prices(px)
    rows = []
    for s in stored.columns:
        a, b = stored[s].dropna(), fresh[s].dropna() if s in fresh else pd.Series(dtype=float)
        common = a.index.intersection(b.index)
        d = (a.loc[common] - b.loc[common]).abs()
        rows.append({"symbol": s, "stored": len(a), "compared": len(common),
                     "revised": int((d > REVISION_TOL).sum()),
                     "max_diff": float(d.max()) if len(d) else np.nan,
                     "first_revised": (d[d > REVISION_TOL].index[0].strftime("%Y-%m-%d")
                                       if (d > REVISION_TOL).any() else "")})
    return pd.DataFrame(rows).set_index("symbol")


def total_return_index(returns: pd.DataFrame) -> pd.DataFrame:
    """Compounded index, 1.0 before each symbol's first return."""
    return (1 + returns.fillna(0.0)).cumprod().where(returns.notna().cummax())


def monthly_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month total returns from the daily store; NaN for partial first months."""
    lr = np.log1p(returns)
    m = lr.resample("ME").sum(min_count=1)
    first = returns.apply(lambda c: c.first_valid_index())
    for s, d in first.items():
        if d is not None:
            m.loc[: d + pd.offsets.MonthEnd(0), s] = np.nan   # first month is partial
    return np.expm1(m)
