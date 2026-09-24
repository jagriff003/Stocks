"""
"Smart money" data, made point-in-time: CFTC positioning and the DIX.

TODO 0m.  James's hypothesis: when informed money leaves stocks, the stock book
is getting "lonely", and Track K should know.  This module turns three free
sources into signals that only use what was knowable on each decision date.

SOURCES

  CFTC Traders in Financial Futures, E-mini S&P 500 (contract 13874A)
      asset managers (pensions, mutual funds, insurers) and leveraged funds
      (hedge funds, CTAs).  Weekly, 2006 onward.
  CFTC disaggregated, managed money in WTI crude (067651), gold (088691),
      silver (084691).  Speculator crowding in the assets Track K would buy.
  SqueezeMetrics DIX: dark-pool short volume as a proxy for off-exchange
      institutional BUYING.  Daily, 2011 onward.  Vendor-computed and not
      auditable; its history could be revised.  Vintaged on every fetch so a
      revision would show.

Contracts are joined on the CFTC contract code, not the name: every one of
these was renamed in February 2022 (crude became "WTI-PHYSICAL").

POINT-IN-TIME RULES

  CFTC positions are as of a Tuesday and published the following Friday.  A
  report is used only from 7 days after its position date, i.e. on a Tuesday
  decision date, last week's report — never the one that describes today.
  DIX is used from the next session.

Raw downloads are kept as dated vintages under data/smartmoney/raw/
(git-ignored), exactly like the long-history research data.
"""

from __future__ import annotations

import io
import json
import urllib.parse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .longhistory import _fetch

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "smartmoney" / "raw"
CFTC = "https://publicreporting.cftc.gov/resource/"
DIX_URL = "https://squeezemetrics.com/monitor/static/DIX.csv"

SP500 = "13874A"
COMMODITIES = {"CRUDE": "067651", "GOLD": "088691", "SILVER": "084691"}
CFTC_LAG_DAYS = 7          # position date -> first decision date allowed to use it


def _socrata(dataset: str, code: str, fields, max_age_days: float, verbose: bool) -> pd.DataFrame:
    q = {"$select": ",".join(["report_date_as_yyyy_mm_dd", "cftc_contract_market_code"] + list(fields)),
         "$where": f"cftc_contract_market_code='{code}'",
         "$order": "report_date_as_yyyy_mm_dd ASC", "$limit": "50000"}
    raw = _fetch(CFTC + dataset + ".json?" + urllib.parse.urlencode(q),
                 f"cftc_{dataset}_{code}.json", max_age_days, verbose, raw_dir=RAW_DIR)
    df = pd.DataFrame(json.loads(raw))
    if df.empty:
        raise RuntimeError(f"CFTC returned nothing for {code} in {dataset}")
    df.index = pd.to_datetime(df.pop("report_date_as_yyyy_mm_dd")).dt.normalize()
    df.index.name = "position_date"
    df = df.drop(columns="cftc_contract_market_code").apply(pd.to_numeric, errors="coerce")
    return df[~df.index.duplicated(keep="last")].sort_index()


def sp500_positioning(max_age_days: float = 7, verbose: bool = True) -> pd.DataFrame:
    """Net positions as a share of open interest, by trader class, weekly."""
    f = ["open_interest_all", "asset_mgr_positions_long", "asset_mgr_positions_short",
         "lev_money_positions_long", "lev_money_positions_short",
         "dealer_positions_long_all", "dealer_positions_short_all"]
    d = _socrata("gpe5-46if", SP500, f, max_age_days, verbose)
    oi = d["open_interest_all"]
    return pd.DataFrame({
        "asset_mgr_net": (d["asset_mgr_positions_long"] - d["asset_mgr_positions_short"]) / oi,
        "lev_money_net": (d["lev_money_positions_long"] - d["lev_money_positions_short"]) / oi,
        "dealer_net": (d["dealer_positions_long_all"] - d["dealer_positions_short_all"]) / oi,
    })


def commodity_positioning(max_age_days: float = 7, verbose: bool = True) -> pd.DataFrame:
    """Managed-money net position as a share of open interest, weekly."""
    out = {}
    for name, code in COMMODITIES.items():
        d = _socrata("72hh-3qpy", code, ["open_interest_all", "m_money_positions_long_all",
                                          "m_money_positions_short_all"], max_age_days, verbose)
        out[name] = (d["m_money_positions_long_all"] - d["m_money_positions_short_all"]) / d["open_interest_all"]
    return pd.DataFrame(out)


def dix(max_age_days: float = 1, verbose: bool = True) -> pd.DataFrame:
    raw = _fetch(DIX_URL, "dix.csv", max_age_days, verbose, raw_dir=RAW_DIR)
    d = pd.read_csv(io.BytesIO(raw), parse_dates=["date"]).set_index("date").sort_index()
    return d[["dix", "gex"]]


# --------------------------------------------------------------------------
# signals, as of decision dates
# --------------------------------------------------------------------------

def _z(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    m = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - m) / sd


def cftc_signals(pos: pd.DataFrame, flow_weeks: int = 13, crowd_weeks: int = 156) -> pd.DataFrame:
    """
    On the position-date calendar (weekly):
      <col>_flow   change in net %OI over `flow_weeks`  ("exiting" when negative)
      <col>_crowd  z-score of net %OI over `crowd_weeks` ("crowded" when high)
    """
    out = {}
    for c in pos.columns:
        out[f"{c}_flow"] = pos[c] - pos[c].shift(flow_weeks)
        out[f"{c}_crowd"] = _z(pos[c], crowd_weeks, crowd_weeks // 2)
    return pd.DataFrame(out)


def dix_signal(d: pd.DataFrame, smooth: int = 20, window: int = 252) -> pd.Series:
    """20-session mean DIX, z-scored against its trailing year (daily calendar)."""
    m = d["dix"].rolling(smooth, min_periods=smooth).mean()
    return _z(m, window, window // 2).rename("dix_z")


def as_of(signal: pd.DataFrame, dates: pd.DatetimeIndex, lag_days: int) -> pd.DataFrame:
    """
    For each decision date, the latest signal row dated at least `lag_days`
    calendar days before it.  Carries a `source_date` column so the lag can be
    asserted rather than trusted.
    """
    s = signal.copy()
    s["source_date"] = s.index
    s.index = s.index + pd.Timedelta(days=lag_days)          # first date it may be used
    s = s.sort_index()
    left = pd.DataFrame(index=pd.DatetimeIndex(dates).sort_values())
    left["_d"] = left.index
    m = pd.merge_asof(left, s, left_on="_d", right_index=True, direction="backward")
    return m.drop(columns="_d").set_index(left.index)
