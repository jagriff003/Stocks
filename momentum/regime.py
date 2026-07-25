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

from .config import VixRegimeConfig
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
