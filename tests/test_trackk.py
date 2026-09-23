"""
Tests for the Track K live reading and the combined allocation.

The allocation is what gets traded, so it has to sum to one, shrink Track J by
exactly the hedge share, and merge a ticker both models hold.  The reading has
to be the same function the trigger study validated, on explicit rotation
dates, with "current" sharing the rotation history up to the last Tuesday.

Run:  python -m pytest tests/test_trackk.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.hedge import hedge_weights  # noqa: E402
from momentum.trackk import (LIVE_CONFIG, SIGNAL_SYMBOL, assets_from_store,  # noqa: E402
                             combine, recommendations)


def fake_store(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-09-22", periods=n)
    cols = sorted(set(SIGNAL_SYMBOL.values()) | {"SPY"})
    return pd.DataFrame(rng.normal(0.0003, 0.01, (n, len(cols))), index=idx, columns=cols)


def test_combine_scales_track_j_and_merges_shared_tickers():
    a = combine({"AAPL": 0.5, "IAU": 0.25, "XOM": 0.25}, 0.5, {"IAU": 0.25, "PDBC": 0.25})
    assert np.isclose(a["weight"].sum(), 1.0)
    assert np.isclose(a.at["AAPL", "weight"], 0.25)
    assert np.isclose(a.at["IAU", "weight"], 0.125 + 0.25)          # both models hold it
    assert np.isclose(a["Track J"].sum(), 0.5) and np.isclose(a["Track K"].sum(), 0.5)


def test_quiet_track_k_leaves_track_j_whole():
    a = combine({"AAPL": 0.75, "MSFT": 0.25}, 0.0, {})
    assert list(a.index) == ["AAPL", "MSFT"] and np.isclose(a["weight"].sum(), 1.0)


def test_recommendations_match_the_validated_function_on_rotation_dates():
    store = fake_store()
    rot = list(store.index[300::10])
    rec = recommendations(store, rot)
    market = store["SPY"]
    assets = assets_from_store(store)
    w = hedge_weights(market, assets, LIVE_CONFIG, decide_at=[d for d in rot if d <= market.index[-1]])
    last = max(d for d in rot if d <= market.index[-1])
    assert rec["last_rotation"]["date"] == last
    assert np.isclose(rec["last_rotation"]["hedge_share"], 1 - w.at[last, "STOCKS"])
    assert rec["current"]["date"] == market.index[-1]
    assert 0.0 <= rec["current"]["hedge_share"] <= LIVE_CONFIG.max_hedge + 1e-12


def test_record_decision_fills_the_latest_row_and_protects_it(tmp_path):
    import pytest
    from scripts.record_decision import record
    log = tmp_path / "log.csv"
    pd.DataFrame({"signal_session": ["2026-09-22", "2026-09-23"], "action": ["", ""],
                  "note": ["", ""]}).to_csv(log, index=False)
    row = record("held Track J", "commodities crowded", log_file=log)
    assert row["signal_session"] == "2026-09-23" and row["action"] == "held Track J"
    with pytest.raises(ValueError, match="overwrite"):
        record("changed my mind", "", log_file=log)
    record("took XLE", "", session="2026-09-22", log_file=log)
    back = pd.read_csv(log, dtype=str, keep_default_na=False)
    assert list(back["action"]) == ["took XLE", "held Track J"]


def test_firing_streak_counts_back_from_the_last_decision():
    from momentum.trackk import firing_streak
    store = fake_store()
    market = store["SPY"]
    assets = assets_from_store(store)
    dates = list(market.index[300::10])
    n = firing_streak(market, assets, dates)
    assert 0 <= n <= len(dates)
