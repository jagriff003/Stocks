"""
Volatility regime detection.

Phase 0 ports the legacy z-score regime unchanged so pre-refactor results stay
reproducible.  Track A (Phase 2) adds the graduated absolute-level ladder here
alongside it.

Why the legacy version needs replacing, stated once so it doesn't get lost:
a rolling z-score measures deviation from a *recent baseline*, not absolute
level.  A sustained moderate VIX becomes the new baseline and stops registering
— 88% of days in the VIX 15-20 band were classified 'normal'.  That band's
realized Sharpe was 0.88 against 3.30 below 15, so those are not days the
strategy should be treating as calm.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import GraduatedVixConfig, VixRegimeConfig
from .signals import as_series


NORMAL = "normal"
ELEVATED = "elevated"
CRISIS = "crisis"


def calculate_vix_regime(vix_data, config: VixRegimeConfig
                         ) -> Tuple[pd.Series, pd.Series]:
    """
    LEGACY z-score regime classifier.

    Returns
    -------
    regime : Series[str]   'normal' | 'elevated' | 'crisis'
    vix_z  : Series[float] rolling z-score of VIX, for diagnostics
    """
    vix_data = as_series(vix_data, "vix_data")

    rolling_mean = vix_data.rolling(config.zscore_window).mean()
    rolling_std = vix_data.rolling(config.zscore_window).std()
    vix_z = (vix_data - rolling_mean) / rolling_std.replace(0, np.nan)

    vix_roc = vix_data.pct_change(config.roc_window)

    regime = pd.Series(NORMAL, index=vix_data.index)
    regime[vix_z >= config.elevated_zscore] = ELEVATED
    regime[vix_z >= config.crisis_zscore] = CRISIS

    if config.use_roc_trigger:
        # A spike can promote 'normal' to 'elevated', but never demote 'crisis'.
        spike = (vix_roc >= config.roc_threshold) & (regime == NORMAL)
        regime[spike] = ELEVATED

    return regime, vix_z


def raw_vix_band(vix, config: GraduatedVixConfig) -> pd.Series:
    """
    Which band each day falls in, before hysteresis.

    Returns an integer Series: 0 = calmest band, rising with volatility.

    `config.band_basis` selects what the edges are measured against:

      'level'  absolute VIX.  Intuitive, but a threshold crossing is coincident
               with the drawdown rather than ahead of it, so acting on it sells
               into the loss.  Measured to be worse than no overlay at all —
               see GraduatedVixConfig.band_basis for the numbers.
      'zscore' VIX relative to its own trailing baseline.  Detects the
               transition into stress rather than its arrival.
    """
    vix = as_series(vix, "vix").astype(float)

    if config.smoothing_window > 1:
        vix = vix.rolling(config.smoothing_window, min_periods=1).mean()

    if config.band_basis == "zscore":
        rolling_mean = vix.rolling(config.zscore_window).mean()
        rolling_std = vix.rolling(config.zscore_window).std()
        measure = (vix - rolling_mean) / rolling_std.replace(0, np.nan)
        # Before the window warms up, treat the day as calm rather than
        # propagating NaN into the band arithmetic.
        measure = measure.fillna(0.0)
    elif config.band_basis == "level":
        measure = vix
    else:
        raise ValueError(f"Unknown band_basis: {config.band_basis!r}")

    band = pd.Series(0, index=vix.index, dtype=int)
    for edge in config.band_edges:
        band += (measure >= edge).astype(int)
    return band


def graduated_regime(vix, config: GraduatedVixConfig) -> pd.DataFrame:
    """
    Effective VIX band per day, with asymmetric hysteresis applied.

    De-risking is fast and re-risking is slow, by default.  That asymmetry is
    intentional: the entire premise of Track A is that the old design was too
    slow to reduce exposure, so a confirmation delay on the way down would
    reintroduce the problem being fixed.  On the way up, a delay is pure
    benefit — it stops VIX oscillating across a boundary from churning the book
    at 2x slippage per round trip.

    Returns
    -------
    DataFrame indexed by date with columns:
      raw_band        band from today's (optionally smoothed) VIX alone
      band            effective band after hysteresis
      momentum_slots  momentum positions permitted in the effective band
    """
    raw = raw_vix_band(vix, config)

    effective = np.empty(len(raw), dtype=int)
    current = int(raw.iloc[0])
    pending = current
    streak = 0

    raw_values = raw.to_numpy()
    for i, value in enumerate(raw_values):
        if value == current:
            pending, streak = current, 0
        else:
            if value == pending:
                streak += 1
            else:
                pending, streak = int(value), 1

            # Worse band (higher number) confirms faster than a better one.
            needed = (config.step_down_days if pending > current
                      else config.step_up_days)
            if streak >= max(1, needed):
                current = pending
                streak = 0

        effective[i] = current

    band = pd.Series(effective, index=raw.index, name="band")
    slots = band.map(lambda b: config.momentum_slots[b]).rename("momentum_slots")

    return pd.DataFrame({"raw_band": raw, "band": band, "momentum_slots": slots})


def describe_ladder(config: GraduatedVixConfig, top_n: int = 4) -> str:
    """Render the ladder as a table — worth printing before trusting a config."""
    lines = [f"{'VIX band':<12}{'Momentum':>10}{'Defensive':>11}{'Exposure':>10}"]
    lines.append("-" * 43)
    for label, slots in zip(config.band_labels, config.momentum_slots):
        lines.append(f"{label:<12}{slots:>10}{top_n - slots:>11}"
                     f"{slots / top_n:>9.0%}")
    lines.append(f"\nhysteresis: de-risk after {config.step_down_days}d, "
                 f"re-risk after {config.step_up_days}d"
                 + (f", VIX smoothed {config.smoothing_window}d"
                    if config.smoothing_window > 1 else ""))
    return "\n".join(lines)


def summarize_regime(regime: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """
    Cross-tabulate classified regime against absolute VIX band.

    This is the diagnostic that exposed the legacy classifier's blind spot; keep
    it runnable so the replacement can be checked the same way rather than
    trusted.
    """
    vix = as_series(vix, "vix")
    bands = pd.cut(vix.reindex(regime.index),
                   bins=[0, 15, 20, 25, np.inf],
                   labels=["<15", "15-20", "20-25", "25+"],
                   right=False)

    table = pd.crosstab(bands, regime, normalize="index")
    table.index.name = "VIX band"
    return table
