"""
Signal construction: raw indicators, standardization, and the composite score.

Design philosophy (do not quietly erode this):

  * Standardization is the foundation.  An MA gap, a derivative and an RSI
    reading are not comparable quantities; z-scoring is what makes combining
    them meaningful at all.
  * Weighting gives the composite discriminating power but decays as markets
    evolve.  Weights are parameters to re-validate periodically, not constants
    to set once and trust.
  * Ranking is relative to the universe, not against an absolute bar, so the
    strategy always leans into the best currently-available opportunity rather
    than defaulting to cash.

Everything here extends that architecture with better inputs and better-scaled
weights.  Nothing here replaces it with absolute thresholds.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import ScoringConfig, VelocityConfig


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def as_series(obj, name: str = "") -> pd.Series:
    """
    Coerce a possibly-single-column DataFrame to a Series.

    yfinance returns MultiIndex columns, so `yf.download('SPY')["Close"]` is a
    one-column DataFrame, not a Series.  Dividing a Series by that DataFrame
    aligns the date index against the column axis and yields all-NaN — which is
    exactly how `Relative_Strength_vs_SPY_20d` came out empty in every
    underlying-measures export.  Coercing at the boundary kills that whole class
    of bug rather than patching one call site.
    """
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(
                f"Expected a single-column frame for {name or 'series'}, "
                f"got columns {list(obj.columns)}"
            )
        obj = obj.iloc[:, 0]
    return obj


# --------------------------------------------------------------------------
# Raw indicators
# --------------------------------------------------------------------------

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI using Wilder's smoothing.

    Computed and exported for diagnostics only — RSI is not in the composite.
    It has been tested twice and carries no usable signal for this strategy:
    entry RSI barely differs between stocks that later lost rank and those that
    held it, and under 1% of observations ever cross the classic overbought
    threshold.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, np.nan)
    loss = (-delta).where(delta < 0, np.nan)

    alpha = 1.0 / window
    gain_avg = gain.ewm(alpha=alpha, adjust=False).mean()
    loss_avg = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = gain_avg / loss_avg
    return 100 - (100 / (1 + rs))


def calculate_ma_difference(prices: pd.Series, short_window: int = 50,
                            long_window: int = 200) -> pd.Series:
    """Difference between the short and long simple moving averages."""
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    return short_ma - long_ma


def calculate_ma_derivative(ma_diff: pd.Series, derivative_window: int = 5) -> pd.Series:
    """First derivative (N-day change) of the MA difference."""
    return ma_diff.diff(derivative_window)


def calculate_relative_strength(stock_prices: pd.Series, spy_prices,
                                window: int = 20) -> pd.Series:
    """
    Relative strength vs SPY: ratio of trailing `window`-day returns.

    `spy_prices` is coerced to a Series — see `as_series` for why that matters.

    Caveat kept in view: this ratio is unstable when the SPY return is near
    zero, and flips sign when it is negative, so it is not a clean "outperforms
    the market" measure.  It is exported for diagnostics; treat any use of it in
    a composite as needing its own validation first.
    """
    spy_prices = as_series(spy_prices, "spy_prices")

    stock_returns = stock_prices.pct_change(window)
    spy_returns = spy_prices.pct_change(window)

    return stock_returns / (spy_returns + 1e-8)


# --------------------------------------------------------------------------
# Standardization
# --------------------------------------------------------------------------

def calculate_rolling_zscore(data, window: int = 252):
    """
    Rolling (time-series) z-score: each column standardized against its own
    trailing `window`-day history.
    """
    if isinstance(data, pd.DataFrame):
        mean = data.rolling(window).mean()
        std = data.rolling(window).std()
        return (data - mean) / std.replace(0, np.nan)
    mean = data.rolling(window).mean()
    std = data.rolling(window).std()
    return (data - mean) / (std if std.ne(0).all() else std.replace(0, np.nan))


def calculate_cross_sectional_zscore(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score: each row (date) standardized across the universe.
    """
    data = data.astype(float)
    mean = data.mean(axis=1)
    std = data.std(axis=1).replace(0, np.nan)
    return data.sub(mean, axis=0).div(std, axis=0)


def calculate_cross_sectional_rank(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional percentile rank, recentred to roughly the same scale as a
    z-score (mean 0, spread ~±1.7).

    Outlier-robust alternative to z-scoring: one stock with a wild reading can
    shift a cross-sectional mean and standard deviation enough to distort every
    other stock's score that day.  Ranks are immune to that.
    """
    pct = data.astype(float).rank(axis=1, pct=True)
    # Map (0, 1] to a symmetric range centred on zero.
    return (pct - 0.5) * 3.4641  # sqrt(12), so a uniform rank has unit variance


# --------------------------------------------------------------------------
# Composite score
# --------------------------------------------------------------------------

def calculate_composite_scores(data: pd.DataFrame, spy_data,
                               config: Optional[ScoringConfig] = None,
                               underlying_path: Optional[str] = None):
    """
    Build the composite momentum score for every stock and date.

        composite = (ma_diff_z + ma_deriv_z) / 2

    Parameters
    ----------
    data : DataFrame
        Adjusted close prices, dates x tickers.
    spy_data : Series or single-column DataFrame
        SPY closes, aligned to `data.index`.
    config : ScoringConfig
    underlying_path : str, optional
        If given, write the pre-standardization measures to this CSV.  Suppress
        in batch experiment runs — it is the slowest part of the calculation.

    Returns
    -------
    dict with keys: composite, rsi_z, ma_diff_z, ma_deriv_z, rel_strength_z,
    and the raw (pre-standardization) frames under raw_*.
    """
    config = config or ScoringConfig()
    spy_data = as_series(spy_data, "spy_data")

    idx, cols = data.index, data.columns
    rsi_scores = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float)
    ma_diff_scores = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float)
    ma_deriv_scores = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float)
    rel_strength_scores = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float)

    min_history = max(config.ma_long, config.zscore_window)

    for stock in cols:
        prices = data[stock]
        if prices.notna().sum() <= min_history:
            continue

        rsi_scores[stock] = calculate_rsi(prices, config.rsi_window)

        ma_diff = calculate_ma_difference(prices, config.ma_short, config.ma_long)
        ma_diff_scores[stock] = ma_diff
        ma_deriv_scores[stock] = calculate_ma_derivative(ma_diff, config.derivative_window)

        rel_strength_scores[stock] = calculate_relative_strength(
            prices, spy_data, config.rel_strength_window
        )

    if underlying_path:
        export_underlying_measures(
            {
                "RSI": rsi_scores,
                f"MA_Difference_{config.ma_short}_{config.ma_long}": ma_diff_scores,
                "MA_Derivative": ma_deriv_scores,
                f"Relative_Strength_vs_SPY_{config.rel_strength_window}d": rel_strength_scores,
            },
            underlying_path,
        )

    if config.zscore_method == "rolling":
        w = config.zscore_window
        rsi_z = -calculate_rolling_zscore(rsi_scores, window=w)
        ma_diff_z = calculate_rolling_zscore(ma_diff_scores, window=w)
        ma_deriv_z = calculate_rolling_zscore(ma_deriv_scores, window=w)
        rel_strength_z = calculate_rolling_zscore(rel_strength_scores, window=w)
    elif config.zscore_method == "cross_sectional":
        rsi_z = -calculate_cross_sectional_zscore(rsi_scores)
        ma_diff_z = calculate_cross_sectional_zscore(ma_diff_scores)
        ma_deriv_z = calculate_cross_sectional_zscore(ma_deriv_scores)
        rel_strength_z = calculate_cross_sectional_zscore(rel_strength_scores)
    else:
        raise ValueError(f"Unknown zscore_method: {config.zscore_method!r}")

    composite = (ma_diff_z + ma_deriv_z) / 2

    return {
        "composite": composite,
        "rsi_z": rsi_z,
        "ma_diff_z": ma_diff_z,
        "ma_deriv_z": ma_deriv_z,
        "rel_strength_z": rel_strength_z,
        "raw_rsi": rsi_scores,
        "raw_ma_diff": ma_diff_scores,
        "raw_ma_deriv": ma_deriv_scores,
        "raw_rel_strength": rel_strength_scores,
    }


def export_underlying_measures(measures: dict, path: str) -> None:
    """
    Write pre-standardization measures to a long-format CSV
    (Date, Symbol, Measure, Value), dropping NaNs.

    Vectorized via melt.  The original row-by-row loop over every (date, stock)
    pair took the bulk of a full model run.
    """
    frames = []
    for name, frame in measures.items():
        long = (frame.stack(future_stack=True)
                     .rename("Value")
                     .reset_index())
        long.columns = ["Date", "Symbol", "Value"]
        long = long.dropna(subset=["Value"])
        long["Measure"] = name
        frames.append(long[["Date", "Symbol", "Measure", "Value"]])

    if not frames:
        return

    out = pd.concat(frames, ignore_index=True).sort_values(["Date", "Symbol", "Measure"])
    out.to_csv(path, index=False)
    print(f"Underlying measures saved to {path} ({len(out):,} rows)")


# --------------------------------------------------------------------------
# Velocity blend
# --------------------------------------------------------------------------

def apply_velocity_blend(composite_scores: pd.DataFrame,
                         config: VelocityConfig) -> pd.DataFrame:
    """
    Blend the z-score level with its velocity (N-day rate of change).

        blended = level_weight * level + velocity_weight * velocity

    Both components are put on a common cross-sectional scale first, so the
    weights mean what they say.  This is the fix for a real defect in the
    pre-refactor code: it blended a *rolling* z-score level (each stock vs its
    own history) against a *cross-sectionally* normalized velocity.  Those have
    different cross-sectional dispersions, so a nominal 0.7/0.3 split was not
    70/30 in effect, and any weight-sweep run against it measured a confounded
    knob.

    `config.blend_normalization='legacy'` reproduces the old behaviour for
    parity checks.

    A high-level stock whose momentum is decelerating and a lower-level stock
    whose momentum is accelerating should not rank the same.  That is what the
    velocity term buys, and it is the mechanism Track C leans on.
    """
    composite_scores = composite_scores.astype(float)
    velocity_raw = composite_scores.diff(config.velocity_window)

    mode = config.blend_normalization
    if mode == "legacy":
        normalize = calculate_cross_sectional_zscore
        level = composite_scores
        velocity = normalize(velocity_raw)
    elif mode == "cross_sectional":
        normalize = calculate_cross_sectional_zscore
        level = normalize(composite_scores)
        velocity = normalize(velocity_raw)
    elif mode == "rank":
        normalize = calculate_cross_sectional_rank
        level = normalize(composite_scores)
        velocity = normalize(velocity_raw)
    else:
        raise ValueError(f"Unknown blend_normalization: {mode!r}")

    blended = config.level_weight * level + config.velocity_weight * velocity

    # --- Track C components, each on the same cross-sectional scale ---
    #
    # Added as weighted terms rather than as AND-ed threshold conditions. A
    # hard AND discards a candidate that misses one condition by any margin;
    # weighting lets a strong reading on one offset a marginal miss elsewhere,
    # which is the same reason the core composite is a blend and not a filter.

    if config.acceleration_weight:
        accel_raw = zscore_acceleration(composite_scores,
                                        config.acceleration_window)
        blended = blended + config.acceleration_weight * normalize(accel_raw)

    if config.best_rank_weight:
        # Best (lowest) rank in the trailing window, negated so higher is
        # better and the weight has the same sign convention as the others.
        ranks = composite_scores.rank(axis=1, ascending=False, method="min")
        best = ranks.rolling(config.best_rank_window, min_periods=1).min()
        blended = blended + config.best_rank_weight * normalize(-best)

    return blended


def zscore_acceleration(composite_scores: pd.DataFrame,
                        window: int = 5) -> pd.DataFrame:
    """
    Second derivative of the composite z-score — the change in velocity.

    Used by the Track C entry rule: a stock whose momentum is accelerating is a
    different proposition from one merely holding a high level, even when their
    current levels match.
    """
    return composite_scores.astype(float).diff(window).diff(window)
