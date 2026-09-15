"""
Price data loading, cleaning and export.

Loads Open as well as Close.  The strategy ranks on closing prices but has to
fill somewhere, and `ExecutionConfig.execute_at='next_open'` needs the open —
without it the backtest can only model the unachievable same-close fill.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .config import REPO_ROOT, OUTPUT_DIR


CACHE_DIR = REPO_ROOT / ".price_cache"


@dataclass
class PriceData:
    """
    Aligned price panel for a backtest run.

    close / open_ : DataFrame (dates x tickers)
    spy, vix      : Series aligned to the same index
    """
    close: pd.DataFrame
    open_: pd.DataFrame
    spy: pd.Series
    vix: pd.Series

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> List[str]:
        return list(self.close.columns)

    def __repr__(self) -> str:
        return (f"PriceData({len(self.close.columns)} tickers, "
                f"{len(self.close)} days, "
                f"{self.index[0]:%Y-%m-%d} to {self.index[-1]:%Y-%m-%d})")


def _cache_path(key: str, start_date: str) -> Path:
    # Pickle rather than parquet: no pyarrow/fastparquet dependency, and this is
    # a throwaway cache, not an interchange format.  Use `export_price_data` for
    # anything meant to be read outside this repo.
    return CACHE_DIR / f"{key}_{start_date}.pkl"


def _universe_key(symbols: List[str]) -> str:
    """
    A cache key that is the same in every process.

    This used to be `abs(hash(tuple(sorted(symbols)))) % 10**10`.  Python salts
    string hashing per interpreter (PYTHONHASHSEED), so that key came out
    different on every single run: the cache never once hit, every run paid a
    full re-download, and `.price_cache` grew by one dead panel per invocation.
    md5 is not used for anything security-bearing here — it just has to be
    stable across processes, which `hash()` is not.
    """
    joined = ",".join(sorted(symbols)).encode()
    return f"panel_{len(symbols)}sym_{hashlib.md5(joined).hexdigest()[:12]}"


#: US equities close at 16:00 exchange time.
MARKET_CLOSE_ET = dt.time(16, 0)


def _et_from_utc(when: dt.datetime) -> dt.datetime:
    """
    UTC -> US Eastern, without a tz-database dependency.

    EDT is UTC-4 and EST is UTC-5.  The month test is a approximation of the
    DST boundary and is wrong for at most a few days in March and November; on
    those days it reports the market closing an hour later than it did, which
    is the safe direction for every caller here — a settled bar is briefly
    treated as unsettled, never the reverse.
    """
    offset = -4 if 3 <= when.month <= 10 else -5
    return when + dt.timedelta(hours=offset)


def _et_now() -> dt.datetime:
    return _et_from_utc(dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))


def _session_of(close: pd.DataFrame) -> Optional[dt.date]:
    """The date of the panel's newest row, or None if it has no rows."""
    if close is None or close.empty:
        return None
    last = close.index[-1]
    return last.date() if hasattr(last, "date") else last


def unsettled_tail(close: pd.DataFrame,
                   now: Optional[dt.datetime] = None) -> Tuple[bool, str]:
    """
    Is the newest row in the panel a session that has not closed yet?

    Yahoo serves the current day's daily bar live: from 09:30 ET onward the
    "Close" of today's row is the last trade, not the closing print, and it
    moves every few minutes.  Nothing downstream can tell the difference — the
    row has the right shape and no missing values — so a run started during
    market hours ranks on intraday prices and reports them as closes.

    That is the failure mode this guard exists for.  It is worse than a missing
    session, because a missing session is visible in the date range and a live
    partial bar is not.

    Returns (is_unsettled, message).  The caller decides whether to warn or
    refuse; this function does not have the context to make that call.
    """
    last_date = _session_of(close)
    if last_date is None:
        return False, ""

    et_now = _et_from_utc(now.replace(tzinfo=None)) if now else _et_now()

    # Only today's row can be unsettled.  Anything older already has its
    # closing print, and a future-dated row is a data error this guard is not
    # responsible for.
    if last_date != et_now.date() or et_now.time() >= MARKET_CLOSE_ET:
        return False, ""

    return True, (
        f"Newest row {last_date:%Y-%m-%d} is TODAY and the US market has not "
        f"closed yet ({et_now:%H:%M} ET). Those are live intraday prices, not "
        f"closes - the ranking will move if you re-run it. Wait until after "
        f"16:00 ET, or pass drop_unsettled=True to exclude the session."
    )


def expected_last_session(now_et: Optional[dt.datetime] = None) -> dt.date:
    """
    The most recent weekday whose closing print should already exist.

    Weekday-only, with no holiday calendar — adding one would mean a new
    dependency to remove a handful of false positives a year, and a false
    positive here costs a glance at a warning.  A false NEGATIVE costs a
    rebalance executed off stale prices, so the asymmetry is deliberate.
    """
    et = now_et or _et_now()
    day = et.date()
    if et.time() < MARKET_CLOSE_ET:
        day -= dt.timedelta(days=1)
    while day.weekday() >= 5:          # 5 = Saturday, 6 = Sunday
        day -= dt.timedelta(days=1)
    return day


def stale_tail(close: pd.DataFrame,
               now_et: Optional[dt.datetime] = None) -> Tuple[bool, str]:
    """
    Is the panel missing a session that should already be published?

    This is the other half of `unsettled_tail`, and the one that bites the
    night-before workflow.  On 2026-09-03 a run started at 22:04 ET — six hours
    after the close — produced a panel ending 2026-09-02.  Nothing in the output
    said so: the run printed its own date in the header, exported, and picked a
    book off the previous session.  The only way to notice was to open the CSV.

    A provider can be late, rate-limit a 52-symbol batch, or return the newest
    row empty for one request and populated for the next.  Rather than model
    those, this just asks the question the operator would ask, and asks it on
    every run.
    """
    last_date = _session_of(close)
    if last_date is None:
        return False, ""

    expected = expected_last_session(now_et)
    if last_date >= expected:
        return False, ""

    behind = (expected - last_date).days
    return True, (
        f"Panel ends {last_date:%Y-%m-%d} but {expected:%Y-%m-%d} should already "
        f"have closed ({behind} calendar day(s) behind). Either the provider has "
        f"not published it yet, the request came back short, or it was a market "
        f"holiday. Do NOT treat this as the rebalance-day book until the panel "
        f"reaches the session you mean to trade on."
    )


def load_data(symbols: List[str],
              start_date: str = "2010-01-01",
              use_cache: bool = True,
              cache_max_age_hours: float = 12.0,
              verbose: bool = True,
              drop_unsettled: bool = False) -> PriceData:
    """
    Download, clean and align price data.

    Cleaning keeps any ticker with at least as many non-null days *in total* as
    there are trading days in the last 12 months — i.e. roughly a year of
    history somewhere in the record.  This reproduces the pre-refactor filter
    (`data.dropna(axis=1, thresh=recent_data.shape[0])`) exactly.

    Note this is a weaker test than it first reads: it does not require the data
    to be recent.  A ticker delisted in 2015 with five years of history would
    pass.  That is the existing behaviour and parity depends on it; tightening
    it is a deliberate change to make separately, not a silent fix here.

    Parameters
    ----------
    symbols : list of str
    start_date : str
    use_cache : bool
        Reuse a local parquet cache younger than `cache_max_age_hours`.  Batch
        experiment runs re-load the same panel dozens of times; without this the
        network dominates the runtime.
    cache_max_age_hours : float
    verbose : bool
    drop_unsettled : bool
        Drop the newest row when it is a session that has not closed yet.  The
        default is to keep it and warn loudly, because silently changing the
        panel's end date under a caller that printed it is its own trap.  Set
        this for scheduled runs, where nobody is reading the warning.

    Returns
    -------
    PriceData

    Note on survivorship bias: `symbols` is normally today's screened universe
    applied backwards over the whole history, so absolute backtest returns are
    optimistic.  Use `momentum.universe.load_universe(as_of=...)` to build a
    point-in-time universe instead where the comparison demands it.
    """
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _universe_key(symbols)
    close_path = _cache_path(key + "_close", start_date)
    open_path = _cache_path(key + "_open", start_date)
    aux_path = _cache_path(key + "_aux", start_date)

    fresh = (
        use_cache
        and all(p.exists() for p in (close_path, open_path, aux_path))
        and (time.time() - close_path.stat().st_mtime) < cache_max_age_hours * 3600
    )

    close = None
    if fresh:
        cached_close = pd.read_pickle(close_path)
        # A cache written during market hours holds a live intraday bar in its
        # last row.  Age alone will not catch that: a panel written at 11:00 ET
        # is only five hours old at 16:05 ET and still inside the window, but
        # its newest "close" is now wrong.  Before the cache key was fixed this
        # could never happen, because the cache never hit — fixing one bug
        # exposed the other.
        et_now = _et_now()
        written_et = _et_from_utc(
            dt.datetime.utcfromtimestamp(close_path.stat().st_mtime))
        was_partial = (
            _session_of(cached_close) == et_now.date()
            and written_et.time() < MARKET_CLOSE_ET
            and et_now.time() >= MARKET_CLOSE_ET
        )
        if was_partial:
            if verbose:
                print("Cache holds a now-settled session written intraday — "
                      "re-downloading rather than trusting a partial bar")
            fresh = False
        else:
            if verbose:
                print(f"Loading price panel from cache ({close_path.name})")
            close = cached_close
            open_ = pd.read_pickle(open_path)
            aux = pd.read_pickle(aux_path)
            spy, vix = aux["SPY"], aux["VIX"]

    if not fresh:
        if verbose:
            print(f"Downloading data for {len(symbols)} symbols...")

        raw = yf.download(symbols, start=start_date, interval="1d",
                          progress=verbose, auto_adjust=True)
        close = raw["Close"]
        open_ = raw["Open"]

        spy_raw = yf.download("SPY", start=start_date, interval="1d",
                              progress=False, auto_adjust=True)["Close"]
        vix_raw = yf.download("^VIX", start=start_date, interval="1d",
                              progress=False, auto_adjust=True)["Close"]

        # Coerce to Series — yfinance returns single-column frames here, and
        # silently propagating that shape is what emptied the relative-strength
        # export in the pre-refactor model.
        from .signals import as_series
        spy = as_series(spy_raw, "SPY")
        vix = as_series(vix_raw, "VIX")

        # --- drop phantom rows ---
        # yfinance emits a placeholder row for a session it has no bars for yet
        # (typically today, before the close). Every ticker is NaN on that row,
        # which is harmless for ranking but poisons anything using
        # `dropna(axis=1, how='any')` — one phantom row drops the entire
        # universe.
        phantom = close.isna().all(axis=1)
        if phantom.any():
            if verbose:
                dates = ", ".join(f"{d:%Y-%m-%d}" for d in close.index[phantom][:3])
                print(f"Dropping {phantom.sum()} empty session row(s): {dates}")
            close = close.loc[~phantom]
            open_ = open_.reindex(close.index)

        # --- clean: drop tickers with less than ~a year of data in total ---
        one_year_ago = close.index[-1] - pd.DateOffset(months=12)
        recent_rows = close.loc[one_year_ago:].shape[0]
        before = list(close.columns)
        close = close.dropna(axis=1, thresh=recent_rows)
        dropped = [c for c in before if c not in close.columns]
        open_ = open_.reindex(columns=close.columns)

        if verbose and dropped:
            print(f"Dropped {len(dropped)} tickers for insufficient recent data: "
                  f"{', '.join(dropped)}")

        spy = spy.reindex(close.index).ffill()
        vix = vix.reindex(close.index).ffill()

        close.to_pickle(close_path)
        open_.to_pickle(open_path)
        pd.DataFrame({"SPY": spy, "VIX": vix}).to_pickle(aux_path)

    # --- unsettled session guard ---
    # Last, so it sees the panel every caller actually receives — cached or
    # freshly downloaded, before or after the phantom-row drop.
    unsettled, message = unsettled_tail(close)
    if unsettled:
        if drop_unsettled:
            close = close.iloc[:-1]
            open_ = open_.reindex(close.index)
            spy = spy.reindex(close.index)
            vix = vix.reindex(close.index)
            if verbose:
                print(f"Dropped unsettled session; panel now ends "
                      f"{close.index[-1]:%Y-%m-%d}")
        else:
            print(f"\n*** WARNING: {message}\n")

    stale, stale_message = stale_tail(close)
    if stale:
        print(f"\n*** WARNING: {stale_message}\n")

    if verbose:
        print(f"Data shape after cleaning: {close.shape}")
        print(f"Date range: {close.index[0]:%Y-%m-%d} to {close.index[-1]:%Y-%m-%d}")

    return PriceData(close=close, open_=open_, spy=spy, vix=vix)


def export_price_data(prices: PriceData,
                      out_dir: Optional[Path] = None,
                      prefix: str = "rsi_ma") -> dict:
    """
    Export the price panel as its own CSVs (Request #2).

    The rank-based analyses can tell you a stock dropped out of the top 4 on day
    3 of its hold, but not what that cost.  Answering the return question needs
    the price panel available as data, not trapped inside a run.

    Writes:
      {prefix}_prices_close.csv   dates x tickers, adjusted close
      {prefix}_prices_open.csv    dates x tickers, open
      {prefix}_prices_market.csv  SPY and VIX
    """
    out_dir = Path(out_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "close": out_dir / f"{prefix}_prices_close.csv",
        "open": out_dir / f"{prefix}_prices_open.csv",
        "market": out_dir / f"{prefix}_prices_market.csv",
    }

    prices.close.to_csv(paths["close"])
    prices.open_.to_csv(paths["open"])
    pd.DataFrame({"SPY": prices.spy, "VIX": prices.vix}).to_csv(paths["market"])

    for name, path in paths.items():
        print(f"- {name} prices: {path.name}")

    return paths
