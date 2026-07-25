"""
Momentum strategy package.

Layout
------
  config.py    every tunable parameter, plus dated config snapshotting
  universe.py  the editable ticker list (universe.csv) and its snapshots
  data.py      price loading, cleaning, caching, export
  signals.py   indicators, standardization, composite score, velocity blend
  regime.py    VIX regime classification
  backtest.py  target-portfolio construction and execution simulation
  metrics.py   performance, turnover and VIX-band segmentation
  strategy.py  end-to-end run_strategy()
  experiments.py  parameterized experiment runner and comparison tables

To change the ticker universe, edit `universe.csv` in the repo root.
To change model parameters, edit the ModelConfig built in scripts/run_live.py.
"""

from .config import (
    ExecutionConfig,
    ModelConfig,
    ScoringConfig,
    VelocityConfig,
    VixRegimeConfig,
    snapshot_config,
    load_config_snapshot,
)
from .data import PriceData, export_price_data, load_data
from .strategy import compute_scores, run_strategy
from .universe import current_symbols, defensive_symbols, load_universe

__all__ = [
    "ExecutionConfig",
    "ModelConfig",
    "ScoringConfig",
    "VelocityConfig",
    "VixRegimeConfig",
    "snapshot_config",
    "load_config_snapshot",
    "PriceData",
    "load_data",
    "export_price_data",
    "run_strategy",
    "compute_scores",
    "current_symbols",
    "defensive_symbols",
    "load_universe",
]
