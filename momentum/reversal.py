"""
Trend-change signals: has a stock turned, how strong is the turn, and how much
range is left.

WHY THIS EXISTS, AND WHAT IT IS NOT

The production composite in `signals.py` scores a stock on where its 50/200 MA
gap sits and which way that gap is moving.  By the time that reads well the
move is weeks old.  Track F measured what it buys: a cross-sectional IC
indistinguishable from zero at 5-63 days, and negative at 42-63 days.  At the
live hold this universe reverses, it does not trend.

This module asks a different question, deliberately:

    1. TURN      has the trend changed sign, from falling to rising?
    2. STRENGTH  how strong is the new trend, relative to the universe?
    3. ROOM      how much of the 52-week range is left above the price?

Each term is built to be self-normalizing before it is standardized, because a
raw slope in log-price-per-day is not comparable between a $15 utility and a
$900 semiconductor, and a difference of two such slopes is worse.  Dividing by
the regression's own standard error turns both into t-statistics, which ARE
comparable, and which land naturally in roughly the -3..+3 range the rest of
this repo's scores occupy.  Cross-sectional standardization on top is what
makes "relative to the universe" mean something.

ON THE SIGN OF THE ROOM TERM

`range_pos` is exported as the POSITION in the 52-week range (0 = at the low,
1 = at the high), not as "headroom".  The direction is left to the weight.

This is not fence-sitting.  The best-documented version of this effect
(George & Hwang, 2004) is that PROXIMITY to the 52-week high predicts higher
forward returns - the opposite of a headroom story.  Both mechanisms are
plausible and they predict opposite signs, so the term is defined in its
neutral form and `room_weight` is allowed to be negative.  Read the sign off
the measurement, not off the construction.

NOTHING HERE APPLIES A THRESHOLD

No "cross above zero", no "RSI under 30".  Every term is continuous and is
combined by weight, for the same reason `apply_velocity_blend` weights rather
than AND-s: a hard condition throws away a candidate that misses by any
margin, and there is no evidence in this repo that any such threshold is
estimable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .signals import (calculate_cross_sectional_rank,
                      calculate_cross_sectional_zscore)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class ReversalConfig:
    """
    Every window and weight in the trend-change family.

    Kept here rather than in `config.py` because none of this is in
    production.  If a term graduates, its parameters move into `ScoringConfig`
    with the rest.
    """

    # --- Turn (b): slope-change t-statistic ---
    #
    # Two adjacent, non-overlapping OLS fits of log price on time.  60 sessions
    # is about a quarter: long enough that the slope is estimated rather than
    # guessed, short enough that a turn shows up inside one window rather than
    # being averaged against the year that preceded it.
    slope_window: int = 60

    # --- Turn (c): momentum sign flip ---
    #
    # The classic formation window is 12 months skipping the most recent one,
    # because the last month reverses.  Here the skipped month is not
    # discarded - it is the other half of the signal.
    long_window: int = 252      # ~12 months, the "was bad" leg
    long_skip: int = 21         # skip the most recent month from the long leg
    short_window: int = 63      # ~3 months, the "is now good" leg

    # --- Room: 52-week range position ---
    room_window: int = 252

    # --- Combination ---
    # Sign convention: a positive weight means a higher term value is better.
    # `room_weight` is deliberately allowed to be negative - see the module
    # docstring.
    turn_weight: float = 1.0
    strength_weight: float = 1.0
    room_weight: float = 0.0

    # 'cross_sectional' z-scores each date across names; 'rank' uses the
    # outlier-robust percentile form.  Both come out on unit variance, so the
    # weights above mean the same thing under either.
    normalization: str = "cross_sectional"

    def normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.normalization == "cross_sectional":
            return calculate_cross_sectional_zscore(frame)
        if self.normalization == "rank":
            return calculate_cross_sectional_rank(frame)
        raise ValueError(f"Unknown normalization: {self.normalization!r}")


# --------------------------------------------------------------------------
# Rolling OLS on log price
# --------------------------------------------------------------------------

def rolling_ols(prices: pd.DataFrame,
                window: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Slope and standard error of an OLS fit of log(price) on time, per column.

    Returns (slope, se).  The slope is in log-price per session - a
    continuously-compounded daily drift - and `se` is the classical OLS
    standard error of that estimate.

    Vectorized rather than looped.  The regressor is a fixed 0..n-1 ramp, so
    with `k` the global row index and a window ending at row `k`:

        Sxx = n(n^2 - 1)/12                        (constant)
        Sxy = SUM(k*y) - (k - (n-1)/2) * SUM(y)    (both rolling sums)
        Syy = SUM(y^2) - SUM(y)^2 / n
        SSE = Syy - slope^2 * Sxx
        se  = sqrt(SSE / (n-2) / Sxx)

    `analyze_reversal_ic.py` checks this against `np.polyfit` on sampled points
    before it reports anything, because a vectorization like this is exactly
    the kind of code that produces plausible numbers while being wrong.
    """
    y = np.log(prices.astype(float))
    n = int(window)
    k = pd.Series(np.arange(len(y), dtype=float), index=y.index)

    sum_y = y.rolling(n).sum()
    sum_yy = (y * y).rolling(n).sum()
    sum_ky = y.mul(k, axis=0).rolling(n).sum()

    sxx = n * (n * n - 1) / 12.0
    centre = k - (n - 1) / 2.0
    sxy = sum_ky.sub(sum_y.mul(centre, axis=0))

    slope = sxy / sxx

    syy = sum_yy - (sum_y ** 2) / n
    sse = (syy - slope ** 2 * sxx).clip(lower=0.0)
    se = np.sqrt(sse / (n - 2) / sxx)

    return slope, se.replace(0.0, np.nan)


# --------------------------------------------------------------------------
# The terms
# --------------------------------------------------------------------------

def slope_change_t(prices: pd.DataFrame,
                   config: ReversalConfig) -> pd.DataFrame:
    """
    Turn, definition (b): the t-statistic of the change in trend slope.

        t = (slope_recent - slope_prior) / sqrt(se_recent^2 + se_prior^2)

    Two adjacent windows of `slope_window` sessions, so they share no data and
    their errors add in quadrature.  A stock that fell for a quarter and has
    risen for a quarter scores high; one that rose steadily through both scores
    near zero, however strong that trend is.  That separation is the point -
    "has it changed" is a different question from "is it good", and the
    production composite conflates them.
    """
    slope, se = rolling_ols(prices, config.slope_window)
    prior_slope = slope.shift(config.slope_window)
    prior_se = se.shift(config.slope_window)

    diff = slope - prior_slope
    se_diff = np.sqrt(se ** 2 + prior_se ** 2)
    return diff / se_diff


def trend_strength_t(prices: pd.DataFrame,
                     config: ReversalConfig) -> pd.DataFrame:
    """
    Strength: the t-statistic of the CURRENT trend slope.

        t = slope_recent / se_recent

    Not a second copy of the turn term.  A stock can turn hard out of a steep
    decline and still only be drifting sideways now (high turn, low strength),
    or grind steadily upward with no change at all (low turn, high strength).
    Scoring both is how "it has turned AND the new trend is real" gets
    expressed without an AND.
    """
    slope, se = rolling_ols(prices, config.slope_window)
    return slope / se


def momentum_flip(prices: pd.DataFrame,
                  config: ReversalConfig) -> pd.DataFrame:
    """
    Turn, definition (c): recent strength minus long-run weakness, in ranks.

        flip = rank(short_return) - rank(long_return)

    The long leg runs t-252 -> t-21 and the short leg t-63 -> t, so the legs
    overlap by 42 sessions.  That overlap is intentional: the classic 12-1
    formation skips the last month because it reverses, and this signal is
    trying to catch precisely that reversal, so the skipped month belongs in
    the short leg rather than nowhere.

    Ranked before differencing rather than differenced then ranked.  Raw
    returns are heavily skewed, and one name up 300% would otherwise set the
    scale for the entire cross-section on that date.
    """
    px = prices.astype(float)
    long_ret = px.shift(config.long_skip) / px.shift(config.long_window) - 1.0
    short_ret = px / px.shift(config.short_window) - 1.0

    return (calculate_cross_sectional_rank(short_ret)
            - calculate_cross_sectional_rank(long_ret))


def range_position(prices: pd.DataFrame,
                   config: ReversalConfig) -> pd.DataFrame:
    """
    Room: where the price sits in its trailing 52-week range, 0 (low) to 1 (high).

        range_pos = (P - low) / (high - low)

    Computed on the adjusted-close panel, so the high and low are closing
    extremes rather than true intraday ones.  That is the panel the rest of the
    model ranks on, and mixing an intraday extreme into a close-based score
    would make the term's scale depend on how gappy a name trades.

    Exported in its neutral form.  `room_weight < 0` reads it as headroom (far
    from the high is good); `room_weight > 0` reads it as the documented
    52-week-high effect (near the high is good).
    """
    px = prices.astype(float)
    w = config.room_window
    high = px.rolling(w).max()
    low = px.rolling(w).min()
    span = (high - low).replace(0.0, np.nan)
    return (px - low) / span


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def build_terms(prices: pd.DataFrame,
                config: ReversalConfig) -> Dict[str, pd.DataFrame]:
    """Every raw term, before standardization."""
    return {
        "turn_t": slope_change_t(prices, config),
        "flip": momentum_flip(prices, config),
        "strength_t": trend_strength_t(prices, config),
        "range_pos": range_position(prices, config),
    }


def composite(terms: Dict[str, pd.DataFrame], config: ReversalConfig,
              turn_term: str = "turn_t") -> pd.DataFrame:
    """
    Weighted combination of the standardized terms.

    `turn_term` selects definition (b) `turn_t` or definition (c) `flip`.
    Each term is standardized first, so the weights describe actual influence -
    the defect `apply_velocity_blend` documents (a nominal 0.7/0.3 split that
    was not 70/30 in effect) came from skipping exactly this step.
    """
    if turn_term not in terms:
        raise KeyError(f"{turn_term!r} not in terms: {sorted(terms)}")

    score = (config.turn_weight * config.normalize(terms[turn_term])
             + config.strength_weight * config.normalize(terms["strength_t"]))

    if config.room_weight:
        score = score + config.room_weight * config.normalize(terms["range_pos"])

    return score
