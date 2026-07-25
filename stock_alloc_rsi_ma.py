"""
Entry point for the live momentum model.

The implementation moved into the `momentum` package.  Running this file still
works exactly as before:

    python stock_alloc_rsi_ma.py

...and is equivalent to `python scripts/run_live.py`.

WHERE THINGS LIVE NOW
---------------------
  universe.csv                  the ticker universe — edit this to add/remove
                                names.  One row per ticker, opens in Excel.
  scripts/run_live.py           this run's parameters, in build_config()
  momentum/config.py            every tunable, with docs on what it does
  momentum/signals.py           indicators, z-scores, composite, velocity blend
  momentum/regime.py            VIX regime classification
  momentum/backtest.py          selection and execution simulation
  momentum/experiments.py       parameter sweeps and comparison tables
  scripts/run_experiments.py    run the sweeps
  tests/test_parity.py          proves the package reproduces the old model

  stock_alloc_rsi_ma_legacy.py  frozen pre-refactor code, reference only

TWO DELIBERATE BEHAVIOUR CHANGES from the pre-refactor model — both are fixes,
both are documented where they live, and `--legacy` reverts both:

  1. The level/velocity blend used to mix a rolling z-score against a
     cross-sectionally normalized velocity, so `level_weight=0.7` did not mean
     70% influence.  Both components are now on the same scale.
     (momentum/config.py :: VelocityConfig.blend_normalization)

  2. The backtest used to fill at the same close that generated the signal,
     which is not achievable.  Fills now default to the next open, and slippage
     is charged.  (momentum/config.py :: ExecutionConfig)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.run_live import main

if __name__ == "__main__":
    sys.exit(main())
