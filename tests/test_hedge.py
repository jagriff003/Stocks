"""
Tests for the Track K hedge layer.

`test_no_lookahead_under_truncation` is the one that protects the study.  The
layer is a loop over rolling windows with carried state (hysteresis), and the
failure mode is a weight that quietly reads one period ahead and produces a
hedge that is always in the right asset.  Truncating the panel and requiring
the last weight to be unchanged is the cheap test that catches it.

`test_zero_cap_reproduces_stock_book` is the reconciliation: with no hedge
allowed the layer must return the stock book to the last bit, or the return
arithmetic is wrong somewhere in a way that would still print plausible numbers.

Run:  python -m pytest tests/test_hedge.py -v
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.hedge import HedgeConfig, apply_weights, hedge_weights, run_hedge  # noqa: E402


def panel(seed=0, n=240):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1970-01-31", periods=n, freq="ME")
    stock = pd.Series(rng.normal(0.008, 0.05, n), index=idx)
    assets = pd.DataFrame({
        "GOLD": rng.normal(0.005, 0.04, n),
        "CMDTY": rng.normal(0.004, 0.05, n),
        "UST10": rng.normal(0.004, 0.02, n) - 0.2 * stock.values,
        "CASH": np.full(n, 0.003),
    }, index=idx)
    return stock, assets


def test_no_lookahead_under_truncation():
    stock, assets = panel()
    cfg = HedgeConfig(exit_margin=0.01, fast_exit_lookback=1, corr_window=24)
    full = hedge_weights(stock, assets, cfg)
    for cut in (40, 77, 150, 239):
        part = hedge_weights(stock.iloc[:cut + 1], assets.iloc[:cut + 1], cfg)
        pd.testing.assert_series_equal(full.iloc[cut], part.iloc[-1], check_names=False)


def test_zero_cap_reproduces_stock_book():
    stock, assets = panel()
    _, res = run_hedge(stock, assets, HedgeConfig(max_hedge=0.0))
    assert (res["hedge_share"] == 0).all()
    np.testing.assert_array_equal(res["net"].values, stock.loc[res.index].values)


def test_weights_are_a_capped_long_only_book():
    stock, assets = panel(seed=3)
    w = hedge_weights(stock, assets, HedgeConfig())
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert (w >= 0).all().all()
    assert (w.drop(columns="STOCKS").sum(axis=1) <= 0.75 + 1e-12).all()


def test_corr_gate_blocks_a_bond_that_moves_with_stocks():
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(1)
    stock = pd.Series(rng.normal(-0.01, 0.04, 120), index=idx)
    # bond beats stocks on trend but is positively correlated with them
    assets = pd.DataFrame({"UST10": stock.values * 0.5 + 0.02, "CASH": 0.0}, index=idx)
    gated = hedge_weights(stock, assets, HedgeConfig(corr_window=24))
    open_ = hedge_weights(stock, assets, HedgeConfig(corr_window=24, corr_gate=None))
    assert gated["UST10"].sum() == 0
    assert open_["UST10"].sum() > 0


def test_hysteresis_stops_a_marginal_asset_flipping():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    stock = pd.Series(0.01, index=idx)
    # asset alternates just above / just below the stock book
    gold = pd.Series(np.where(np.arange(60) % 2 == 0, 0.013, 0.007), index=idx)
    assets = pd.DataFrame({"GOLD": gold, "CASH": 0.0})
    cfg = HedgeConfig(lookback=1, corr_gate=None)
    flips = hedge_weights(stock, assets, cfg)["GOLD"].diff().abs().sum()
    sticky = hedge_weights(stock, assets, replace(cfg, exit_margin=0.005))["GOLD"].diff().abs().sum()
    assert flips > 10 and sticky <= 0.25


def test_cash_takes_a_slot_when_stocks_trail_cash():
    idx = pd.date_range("2000-01-31", periods=24, freq="ME")
    stock = pd.Series(-0.02, index=idx)
    assets = pd.DataFrame({"GOLD": -0.03, "CASH": 0.002}, index=idx)
    w = hedge_weights(stock, assets, HedgeConfig(corr_gate=None))
    assert (w["CASH"].iloc[3:] == 0.25).all()
    assert (w["GOLD"] == 0).all()           # beats neither stocks nor cash


def test_weight_on_asset_without_return_fails_loudly():
    stock, assets = panel()
    w = hedge_weights(stock, assets, HedgeConfig())
    held = w["GOLD"].shift(1) > 0
    broken = assets.copy()
    broken.loc[held[held].index[0], "GOLD"] = np.nan
    with pytest.raises(AssertionError):
        apply_weights(stock, broken, w, 10.0)


def test_danger_gate_closes_slots_while_stocks_beat_cash():
    idx = pd.date_range("2000-01-31", periods=48, freq="ME")
    # stocks rise for two years then fall; gold beats them throughout
    stock = pd.Series(np.r_[np.full(24, 0.01), np.full(24, -0.02)], index=idx)
    assets = pd.DataFrame({"GOLD": 0.015, "CASH": 0.002}, index=idx)
    cfg = HedgeConfig(corr_gate=None, danger_lookback=6)
    w = hedge_weights(stock, assets, cfg)
    assert (w["GOLD"].iloc[:24] == 0).all()
    assert (w["GOLD"].iloc[30:] == 0.25).all()
    ungated = hedge_weights(stock, assets, HedgeConfig(corr_gate=None))
    assert (ungated["GOLD"].iloc[5:24] == 0.25).all()


def test_danger_gate_has_no_lookahead():
    stock, assets = panel(seed=7)
    cfg = HedgeConfig(danger_lookback=12, corr_window=24)
    full = hedge_weights(stock, assets, cfg)
    for cut in (30, 101, 200):
        part = hedge_weights(stock.iloc[:cut + 1], assets.iloc[:cut + 1], cfg)
        pd.testing.assert_series_equal(full.iloc[cut], part.iloc[-1], check_names=False)


def test_harvest_sells_the_run_and_returns_the_slot_to_stocks():
    idx = pd.date_range("2000-01-31", periods=40, freq="ME")
    stock = pd.Series(-0.02, index=idx)
    # oil runs 10%/month; a second asset also qualifies and must NOT take the slot
    assets = pd.DataFrame({"OIL": 0.10, "GOLD": 0.01, "CASH": 0.0}, index=idx)
    cfg = HedgeConfig(corr_gate=None, max_hedge=0.25, harvest_gain=0.30,
                      harvest_blackout=6)
    w = hedge_weights(stock, assets, cfg)
    first = w.index.get_loc(w["OIL"].gt(0).idxmax())
    # 1.1^3 = 1.331 >= 1.30: harvested at the third decision after entry
    assert (w["OIL"].iloc[first:first + 3] == 0.25).all()
    assert w["OIL"].iloc[first + 3] == 0
    # for the blackout the slot is the stock book's, not gold's
    assert (w["STOCKS"].iloc[first + 3:first + 3 + 6] == 1.0).all()
    assert w["OIL"].iloc[first + 3 + 6] == 0.25            # eligible again


def test_harvest_has_no_lookahead():
    stock, assets = panel(seed=11)
    cfg = HedgeConfig(harvest_gain=0.05, harvest_z=1.5, harvest_z_min_history=24,
                      harvest_blackout=3, corr_window=24)
    full = hedge_weights(stock, assets, cfg)
    assert (full.drop(columns="STOCKS").sum(axis=1) > 0).any()
    for cut in (60, 133, 239):
        part = hedge_weights(stock.iloc[:cut + 1], assets.iloc[:cut + 1], cfg)
        pd.testing.assert_series_equal(full.iloc[cut], part.iloc[-1], check_names=False)


def test_trailing_stop_lets_the_run_continue_then_sells_after_the_top():
    idx = pd.date_range("2000-01-31", periods=30, freq="ME")
    stock = pd.Series(-0.02, index=idx)
    # oil runs 10%/month for 12 months, then falls 8%/month
    oil = np.r_[np.full(12, 0.10), np.full(18, -0.08)]
    assets = pd.DataFrame({"OIL": oil, "CASH": 0.0}, index=idx)
    cfg = HedgeConfig(corr_gate=None, require_beats_cash=False, harvest_trail=0.15,
                      harvest_blackout=100)
    w = hedge_weights(stock, assets, cfg)
    held = w["OIL"] > 0
    assert held.iloc[3:12].all()                       # never sold into the run
    # peak at row 11; -8% then -15.4% from peak at row 13 -> exit decided at row 13
    assert held.iloc[12] and not held.iloc[13:].any()


def test_decisions_only_on_the_cadence_and_the_book_carries_between():
    stock, assets = panel(seed=5, n=200)
    cfg = HedgeConfig(decide_every=5, decide_phase=2)
    w = hedge_weights(stock, assets, cfg)
    for i in range(1, len(w)):
        if (i - 2) % 5 != 0:
            pd.testing.assert_series_equal(w.iloc[i], w.iloc[i - 1], check_names=False)
    assert w.drop(columns="STOCKS").sum(axis=1).gt(0).any()


def test_cadence_and_trail_have_no_lookahead():
    stock, assets = panel(seed=9)
    cfg = HedgeConfig(decide_every=5, decide_phase=3, harvest_trail=0.05,
                      harvest_blackout=4, corr_window=24)
    full = hedge_weights(stock, assets, cfg)
    for cut in (47, 118, 239):
        part = hedge_weights(stock.iloc[:cut + 1], assets.iloc[:cut + 1], cfg)
        pd.testing.assert_series_equal(full.iloc[cut], part.iloc[-1], check_names=False)


def test_exec_lag_shifts_the_earning_row():
    stock, assets = panel(seed=2)
    cfg = HedgeConfig()
    w = hedge_weights(stock, assets, cfg)
    one = apply_weights(stock, assets, w, 0.0, exec_lag=1)
    two = apply_weights(stock, assets, w, 0.0, exec_lag=2)
    t = two.index[10]
    decided = w.index[w.index.get_loc(t) - 2]
    rets = pd.concat([stock.rename("STOCKS"), assets], axis=1)
    assert np.isclose(two.at[t, "gross"], (w.loc[decided] * rets.loc[t]).sum())
    assert not one["gross"].equals(two["gross"])


def test_relative_reversal_hands_the_slot_back_when_stocks_recover():
    idx = pd.date_range("2000-01-31", periods=40, freq="ME")
    # stocks crash for 12 periods then rebound hard; gold is flat-positive
    stock = pd.Series(np.r_[np.full(12, -0.06), np.full(28, 0.05)], index=idx)
    assets = pd.DataFrame({"GOLD": 0.005, "CASH": 0.001}, index=idx)
    base = HedgeConfig(lookback=6, corr_gate=None, max_hedge=0.25)
    slow = hedge_weights(stock, assets, base)
    fast = hedge_weights(stock, assets, replace(base, harvest_rel_lookback=1, harvest_blackout=12))
    # the 6-period comparison keeps gold for periods into the rebound; the 1-period one does not
    assert slow["GOLD"].iloc[12:15].eq(0.25).all()          # 0.94^3 * 1.05^3 < 1.005^6
    assert fast["GOLD"].iloc[12:24].eq(0).all()
    assert fast["GOLD"].iloc[:12].gt(0).any()


def test_relative_reversal_has_no_lookahead():
    stock, assets = panel(seed=13)
    cfg = HedgeConfig(harvest_rel_lookback=2, harvest_blackout=3, decide_every=2, corr_window=24)
    full = hedge_weights(stock, assets, cfg)
    for cut in (51, 140, 239):
        part = hedge_weights(stock.iloc[:cut + 1], assets.iloc[:cut + 1], cfg)
        pd.testing.assert_series_equal(full.iloc[cut], part.iloc[-1], check_names=False)


def test_regime_trigger_needs_enough_assets_and_positive_stock_bond_corr():
    idx = pd.date_range("2000-01-31", periods=80, freq="ME")
    rng = np.random.default_rng(4)
    stock = pd.Series(rng.normal(-0.01, 0.03, 80), index=idx)
    bond_neg = -0.8 * stock + rng.normal(0, 0.005, 80)     # classic hedge
    bond_pos = 0.8 * stock + rng.normal(0, 0.005, 80)      # inflation signature
    base = dict(GOLD=0.03, OIL=0.03, SILVER=-0.02, CASH=0.001)
    cfg = HedgeConfig(corr_gate=None, regime_assets=("GOLD", "OIL", "SILVER"), regime_min=2,
                      regime_corr_asset="UST", regime_corr_above=0.0, corr_window=24)
    fire = hedge_weights(stock, pd.DataFrame({**base, "UST": bond_pos}, index=idx), cfg)
    quiet = hedge_weights(stock, pd.DataFrame({**base, "UST": bond_neg}, index=idx), cfg)
    one = hedge_weights(stock, pd.DataFrame({**base, "OIL": -0.02, "UST": bond_pos}, index=idx), cfg)
    assert fire.drop(columns="STOCKS").sum(axis=1).iloc[30:].gt(0).all()
    assert quiet.drop(columns="STOCKS").sum(axis=1).eq(0).all()    # correlation says no
    assert one.drop(columns="STOCKS").sum(axis=1).eq(0).all()      # only one asset qualifies


def test_regime_trigger_has_no_lookahead():
    stock, assets = panel(seed=17)
    cfg = HedgeConfig(regime_assets=("GOLD", "CMDTY"), regime_min=1, regime_corr_asset="UST10",
                      regime_corr_above=-0.5, corr_window=24, decide_every=2)
    full = hedge_weights(stock, assets, cfg)
    for cut in (61, 150, 239):
        part = hedge_weights(stock.iloc[:cut + 1], assets.iloc[:cut + 1], cfg)
        pd.testing.assert_series_equal(full.iloc[cut], part.iloc[-1], check_names=False)


def test_explicit_decision_dates_replace_the_positional_cadence():
    stock, assets = panel(seed=21, n=200)
    dates = stock.index[[30, 41, 55, 70, 90, 111, 150]]
    w = hedge_weights(stock, assets, HedgeConfig(), decide_at=dates)
    changes = w.index[w.diff().abs().sum(axis=1) > 0]
    assert set(changes) <= set(dates)
    part = hedge_weights(stock.iloc[:112], assets.iloc[:112], HedgeConfig(), decide_at=dates)
    pd.testing.assert_series_equal(w.iloc[111], part.iloc[-1], check_names=False)
