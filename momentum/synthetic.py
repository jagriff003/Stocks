"""
Synthetic series: ratios and breadth, scored as monitors.

The universe is 48 US equities.  There are signals it structurally cannot see -
the dollar, credit appetite, market breadth - because no member of it moves on
those things alone.  A monitor makes such a signal visible in the same
cross-sectional ranking as everything else, which is the only place the model
looks.

WHY RATIOS RATHER THAN LEGS
   HYG on its own is a bond fund whose price mostly tracks duration.  HYG
   divided by LQD is credit appetite with duration cancelled out, and that
   ratio carries information neither leg does.  The same argument applies to
   RSP/SPY (equal weight against cap weight - leadership breadth) and IWM/SPY
   (small against large - risk appetite).

   A ratio is a price series like any other: RSI, moving averages and relative
   strength are all well defined on it.  It is not tradable, which is exactly
   why it must be a monitor and never a holding.

BREADTH WITHOUT A BREADTH FEED
   Yahoo serves none of the advance/decline indices - ^ADVN, ^DECN, $ADD, ^TRIN
   and ^NYAD all 404.  So the A/D line is computed here from a panel of names
   instead: each session, the share of names that advanced, accumulated into a
   cumulative line the way a classic A/D line is built.

   Computing it rather than buying it is the better option, not a fallback.  It
   has the same history as the price panel, costs no API calls, is reproducible,
   and is measured over the liquid universe the strategy actually draws from
   rather than over every listed issue including the ones nobody can trade.

OPEN PRICES
   Synthetics carry open == close.  They are never held, so no return is ever
   computed from them; setting the two equal makes that explicit rather than
   inventing an intraday path that means nothing.  If a synthetic ever became
   tradable this would have to change, and `augment` refuses to build one
   whose name is not registered as a monitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data import PriceData


@dataclass
class RatioSpec:
    """A synthetic series defined as numerator / denominator."""
    name: str
    numerator: str
    denominator: str
    note: str = ""


@dataclass
class BreadthSpec:
    """A cumulative advance/decline line computed over `members`."""
    name: str
    members: Sequence[str]
    note: str = ""


def ratio_series(close: pd.DataFrame, spec: RatioSpec) -> pd.Series:
    """num / den, rebased to 100 at its first valid point."""
    for leg in (spec.numerator, spec.denominator):
        if leg not in close.columns:
            raise KeyError(
                f"Cannot build {spec.name}: {leg} is not in the panel. "
                f"Add it to the download list even if it is never held.")
    num, den = close[spec.numerator], close[spec.denominator]
    r = (num / den.replace(0, np.nan)).dropna()
    if r.empty:
        raise ValueError(f"{spec.name} is empty after alignment")
    return (r / r.iloc[0] * 100.0).reindex(close.index)


def advance_decline_line(close: pd.DataFrame, members: Sequence[str],
                         min_names: int = 20) -> pd.Series:
    """
    Cumulative advance/decline line over `members`, rebased to 100.

    Each session contributes the net advancing share - (advancers - decliners)
    divided by the number of names with data that day - which is then
    accumulated.  Dividing by the daily count matters: the panel's membership
    grows over time as names come into history, and an unnormalized net count
    would drift purely because more names exist later.
    """
    cols = [c for c in members if c in close.columns]
    if len(cols) < min_names:
        raise ValueError(
            f"Advance/decline needs at least {min_names} names, got {len(cols)}")
    r = close[cols].pct_change()
    adv = (r > 0).sum(axis=1)
    dec = (r < 0).sum(axis=1)
    live = r.notna().sum(axis=1)
    net = ((adv - dec) / live.replace(0, np.nan)).fillna(0.0)
    keep = live >= min_names
    net = net.where(keep, 0.0)
    return 100.0 * np.exp(net.cumsum() * 0.01)


def augment(prices: PriceData,
            ratios: Sequence[RatioSpec] = (),
            breadth: Sequence[BreadthSpec] = (),
            monitor_symbols: Optional[Sequence[str]] = None) -> PriceData:
    """
    Return a PriceData with the synthetics appended as extra columns.

    Every synthetic must appear in `monitor_symbols`.  A synthetic that is
    selectable would be bought, and a ratio cannot be bought; the check is here
    because that mistake is silent at runtime and produces a plausible return
    stream.
    """
    monitors = set(monitor_symbols or ())
    close = prices.close.copy()
    open_ = prices.open_.copy()

    built: List[str] = []
    for spec in ratios:
        if spec.name not in monitors:
            raise ValueError(
                f"Synthetic {spec.name!r} is not in monitor_symbols. A ratio "
                f"cannot be traded; register it as a monitor or drop it.")
        s = ratio_series(close, spec)
        close[spec.name] = s
        open_[spec.name] = s
        built.append(spec.name)

    for spec in breadth:
        if spec.name not in monitors:
            raise ValueError(
                f"Synthetic {spec.name!r} is not in monitor_symbols.")
        s = advance_decline_line(close, spec.members)
        close[spec.name] = s
        open_[spec.name] = s
        built.append(spec.name)

    return PriceData(close=close, open_=open_, spy=prices.spy, vix=prices.vix)


def describe(prices: PriceData, names: Sequence[str]) -> pd.DataFrame:
    """Quick sanity table for synthetics: level, trend and correlation to SPY."""
    spy_r = prices.spy.reindex(prices.close.index).ffill().pct_change()
    rows = []
    for n in names:
        if n not in prices.close.columns:
            continue
        s = prices.close[n].dropna()
        r = s.pct_change()
        rows.append({
            "Series": n,
            "Obs": len(s),
            "Last": float(s.iloc[-1]),
            "1y change": float(s.iloc[-1] / s.iloc[-252] - 1) if len(s) > 252 else np.nan,
            "Ann. vol": float(r.std() * np.sqrt(252)),
            "Corr to SPY": float(r.corr(spy_r.reindex(r.index))),
            "Above 200d MA": bool(s.iloc[-1] > s.iloc[-200:].mean()),
        })
    return pd.DataFrame(rows)
