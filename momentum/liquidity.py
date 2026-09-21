"""
What a trade actually costs, per name and per date.

WHY A FLAT RATE STOPS WORKING

Every result in FINDINGS was computed at a flat 7.5 bps one way.  That is a
defensible number for a 49-name book of mega-caps: the spread is a basis point
or two and a retail order is an invisible fraction of a day's volume.  It is a
fiction the moment the book is allowed to hold a $2B name, and Track J's signal
holds a lot of them - median pick $9.1B against a $20.7B pool median.

A study that lets the book wander down the cap scale while still charging
mega-cap costs is not measuring a signal, it is measuring a subsidy.  So this
charges each name what it would plausibly cost.

THE MODEL

    one-way cost = half-spread + market impact

    half_spread_bps = floor + scale * (ref_adv / adv_notional) ** exponent
    impact_bps      = k * sigma_daily * sqrt(order_notional / adv_notional)

The impact term is the square-root law (Almgren and others): cost rises with
the square root of the fraction of daily volume you consume, scaled by the
name's own volatility.  It is the standard functional form and it has the
property that matters here - consuming 1% of a day's volume in a quiet name is
cheap, and the same 1% in a volatile one is not.

Shape at the default settings, for a $100k account holding four names
(so ~$62.5k per order):

    ADV $2B, sigma 1.5%/day   ->  ~1.5 + 0.2  =  1.7 bps
    ADV $100M, sigma 2.5%/day ->  ~4.5 + 2.5  =  7.0 bps
    ADV $20M, sigma 3.5%/day  ->  ~8.8 + 5.0  = 13.8 bps
    ADV $5M, sigma 4%/day     ->  ~16.6 + 11.3 = 27.9 bps

So mega-caps come in cheaper than the flat 7.5 bps assumption and small names
several times dearer, which is the whole point.

POINT-IN-TIME, DELIBERATELY

`adv_notional` is a trailing median of price x volume computed from the panel,
not a field scraped from a quote today.  Today's ADV applied backwards would
undercharge every name that has since grown - which is most of the survivors in
any pool built today, and precisely the names a momentum signal picks.  That
would put the thumb on the scale in the direction of the result we are trying
to test.

WHAT THIS STILL GETS WRONG

It is a model, not a fill log.  It has no borrow cost, no opening-auction
dynamics, no gap risk on the overnight it does not price, and it assumes the
order is worked over one day rather than dumped in one print.  The right way to
use it is alongside the flat-bps ladder and the breakeven figure, not as a
precise answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LiquidityConfig:
    """
    Every parameter of the cost model.  None of these are estimated from the
    data being tested; they are assumptions, and they are meant to be swept.
    """

    #: Total book value.  This is the parameter that matters most and the one
    #: with no right answer - impact scales with how much you are pushing
    #: through, so a $100k account and a $10M account face different signals.
    account_notional: float = 100_000.0

    #: half_spread_bps = floor + scale * (ref_adv / adv) ** exponent
    half_spread_floor_bps: float = 1.0
    half_spread_scale_bps: float = 35.0
    half_spread_ref_adv: float = 1_000_000.0
    half_spread_exponent: float = 0.5

    #: impact_bps = k * sigma_daily_bps * sqrt(participation)
    impact_coefficient: float = 0.8

    #: Windows for the two point-in-time inputs.
    adv_window: int = 60
    vol_window: int = 60

    #: Nothing is allowed to cost less than a tick or more than this, because
    #: the functional form is unreliable in both tails.
    min_bps: float = 0.5
    max_bps: float = 500.0

    def __post_init__(self):
        if self.account_notional <= 0:
            raise ValueError("account_notional must be positive")
        if not 0 < self.half_spread_exponent <= 2:
            raise ValueError("half_spread_exponent out of range")
        if self.min_bps > self.max_bps:
            raise ValueError("min_bps above max_bps")


def dollar_volume(close: pd.DataFrame, volume: pd.DataFrame,
                  window: int = 60) -> pd.DataFrame:
    """
    Trailing median daily dollar volume, in dollars.

    Median rather than mean: a single earnings-day volume spike should not make
    a name look permanently liquid, and the mean is badly exposed to exactly
    that.  `min_periods` is a third of the window so a name is not treated as
    untradable purely for being early in the panel.
    """
    dv = close.astype(float) * volume.reindex_like(close).astype(float)
    return dv.rolling(window, min_periods=max(5, window // 3)).median()


def realized_vol(close: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Trailing daily return standard deviation, as a fraction."""
    return close.astype(float).pct_change().rolling(
        window, min_periods=max(5, window // 3)).std()


def slippage_panel(close: pd.DataFrame, volume: pd.DataFrame,
                   config: Optional[LiquidityConfig] = None,
                   top_n: int = 4) -> pd.DataFrame:
    """
    One-way slippage as a FRACTION, per date and per symbol.

    `top_n` sets the order size: a book of `top_n` equal positions means each
    order is `account_notional / top_n`.  It is passed rather than inferred so
    that a book-size sweep also sweeps the cost of the bigger or smaller
    orders it implies - holding 16 names is cheaper per trade than holding 4,
    and a comparison that ignored that would flatter the concentrated book.

    Returns a frame aligned to `close`, suitable for
    `simulate_portfolio(slippage_by_symbol=...)`.
    """
    config = config or LiquidityConfig()

    adv = dollar_volume(close, volume, config.adv_window)
    sigma = realized_vol(close, config.vol_window)

    # A name with no usable volume estimate is charged the maximum rather than
    # silently costing nothing - the failure should be expensive, not free.
    adv = adv.where(adv > 0)

    half_spread = (config.half_spread_floor_bps
                   + config.half_spread_scale_bps
                   * (config.half_spread_ref_adv / adv) ** config.half_spread_exponent)

    order = config.account_notional / max(1, int(top_n))
    participation = (order / adv).clip(upper=1.0)
    impact = config.impact_coefficient * sigma * np.sqrt(participation) * 10_000.0

    bps = (half_spread + impact).clip(lower=config.min_bps, upper=config.max_bps)
    bps = bps.fillna(config.max_bps)
    return bps / 10_000.0


def tradable_mask(close: pd.DataFrame, volume: pd.DataFrame,
                  min_adv: float = 10e6, min_price: float = 5.0,
                  config: Optional[LiquidityConfig] = None) -> pd.DataFrame:
    """
    Per-date boolean: could this name be traded as of this date?

    POINT-IN-TIME, AND THAT IS THE ENTIRE POINT.  The obvious implementation —
    take each name's median dollar volume over the whole sample and keep the
    ones above a floor — is look-ahead of the worst kind: a name earns its place
    in the 2010 cross-section because of volume it had in 2020.  It also
    silently reintroduces survivorship, since the names that stayed liquid are
    exactly the ones that did well.

    Applied by masking the SCORE panel rather than by dropping columns, so a
    name that falls below the floor becomes unpickable on those dates and
    becomes pickable again if it recovers — which is what a screen re-run each
    rebalance actually does.

    The price floor is the blunt proxy for the main continued-listing rule, and
    it is the filter most likely to duck a compliance delisting before it
    happens.
    """
    config = config or LiquidityConfig()
    adv = dollar_volume(close, volume.reindex_like(close), config.adv_window)
    return (adv >= min_adv) & (close.astype(float) >= min_price)


def apply_tradable(scores: pd.DataFrame, mask: pd.DataFrame,
                   always: Optional[set] = None) -> pd.DataFrame:
    """
    Blank a score panel wherever the name is not tradable that day.

    `always` names (the defensive sleeve) bypass the screen — they enter the
    book by regime rule rather than on rank, and they are ETFs whose liquidity
    is not in question.
    """
    m = mask.reindex(index=scores.index, columns=scores.columns).fillna(False)
    if always:
        for sym in always:
            if sym in m.columns:
                m[sym] = True
    return scores.where(m)


def summarize(panel: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Per-symbol median cost in bps, for reporting what the model assumed."""
    med = (panel.median() * 10_000).rename("median_bps")
    out = med.to_frame()
    out["p90_bps"] = panel.quantile(0.90) * 10_000
    if label:
        out["panel"] = label
    return out.sort_values("median_bps")
