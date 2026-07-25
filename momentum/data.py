"""
Price data loading, cleaning and export.

Loads Open as well as Close.  The strategy ranks on closing prices but has to
fill somewhere, and `ExecutionConfig.execute_at='next_open'` needs the open —
without it the backtest can only model the unachievable same-close fill.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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


def load_data(symbols: List[str],
              start_date: str = "2010-01-01",
              use_cache: bool = True,
              cache_max_age_hours: float = 12.0,
              verbose: bool = True) -> PriceData:
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
    key = f"panel_{len(symbols)}sym_{abs(hash(tuple(sorted(symbols)))) % 10**10}"
    close_path = _cache_path(key + "_close", start_date)
    open_path = _cache_path(key + "_open", start_date)
    aux_path = _cache_path(key + "_aux", start_date)

    fresh = (
        use_cache
        and all(p.exists() for p in (close_path, open_path, aux_path))
        and (time.time() - close_path.stat().st_mtime) < cache_max_age_hours * 3600
    )

    if fresh:
        if verbose:
            print(f"Loading price panel from cache ({close_path.name})")
        close = pd.read_pickle(close_path)
        open_ = pd.read_pickle(open_path)
        aux = pd.read_pickle(aux_path)
        spy, vix = aux["SPY"], aux["VIX"]
    else:
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
