"""
Tests for position sizing, and for the claim that makes the comparison valid.

The whole sizing experiment rests on one thing: that routing the engine
through a weight vector did not change what equal weight means.  If it did,
every "inverse-vol beats equal weight by X" number is really measuring the
refactor rather than the scheme.  So the first block here is not a test of
sizing at all — it is a test that the baseline survived.

It survived with one deliberate exception, which is also tested: the old
symbol-based cost model charged nothing when the book GREW.  See
`test_old_cost_model_undercharged_book_expansion`.

Run:  python -m pytest tests/test_sizing.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.backtest import (  # noqa: E402
    _segment_return, _segment_return_weighted, _trade_cost, _trade_cost_weighted,
)
from momentum.config import ExecutionConfig  # noqa: E402
from momentum.sizing import (  # noqa: E402
    SizingConfig, _cap_and_renormalize, compute_weights,
)


def equal(symbols):
    if not symbols:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(symbols), index=symbols)


# --- the weighted engine reduces to the unweighted one --------------------

@pytest.mark.parametrize("symbols,p0,p1", [
    (["A", "B", "C"], [10.0, 20.0, 30.0], [11.0, 19.0, 33.0]),
    (["A", "B"], [100.0, 50.0], [90.0, 55.0]),
    (["A", "B", "C", "D", "E"], [1, 2, 3, 4, 5], [1.1, 2.2, 2.7, 4.4, 5.5]),
])
def test_equal_weights_reproduce_the_mean_return(symbols, p0, p1):
    start = pd.Series(dict(zip(symbols, map(float, p0))))
    end = pd.Series(dict(zip(symbols, map(float, p1))))
    assert _segment_return_weighted(equal(symbols), start, end) == \
        pytest.approx(_segment_return(symbols, start, end), abs=1e-15)


def test_unusable_prices_are_dropped_the_same_way():
    """A name with a missing price must not quietly become a zero return."""
    symbols = ["A", "B", "C"]
    start = pd.Series({"A": 10.0, "B": np.nan, "C": 30.0})
    end = pd.Series({"A": 11.0, "B": 25.0, "C": 33.0})
    w = _segment_return_weighted(equal(symbols), start, end)
    assert w == pytest.approx(_segment_return(symbols, start, end), abs=1e-15)
    assert w == pytest.approx(0.10, abs=1e-12)   # both survivors rose 10%


def test_cash_is_held_at_zero_when_the_book_is_derisked():
    """A book scaled to 60% invested earns 60% of the move, not all of it."""
    symbols = ["A", "B"]
    start = pd.Series({"A": 100.0, "B": 100.0})
    end = pd.Series({"A": 110.0, "B": 110.0})
    half = pd.Series({"A": 0.3, "B": 0.3})       # 60% invested
    assert _segment_return_weighted(half, start, end) == pytest.approx(0.06)


# --- the cost model, including the bug it fixes ---------------------------

@pytest.mark.parametrize("old,new", [
    (["A", "B", "C", "D", "E"], ["A", "B", "C", "X", "Y"]),   # partial turn
    ([], ["A", "B", "C"]),                                     # initial build
    (["A", "B", "C"], []),                                     # to cash
    (["A", "B", "C", "D", "E"], ["A", "B", "C"]),              # book shrinks
    (["A", "B", "C"], ["X", "Y", "Z"]),                        # full turn
    (["A", "B", "C"], ["A", "B", "C"]),                        # no change
    (["A", "B", "C", "D", "E", "F", "G"], ["A", "B"]),         # big shrink
])
def test_weighted_cost_matches_symbol_cost_under_equal_weight(old, new):
    ex = ExecutionConfig()
    assert _trade_cost_weighted(equal(old), equal(new), ex) == \
        pytest.approx(_trade_cost(old, new, ex), abs=1e-15)


def test_old_cost_model_undercharged_book_expansion():
    """
    The one case where the two disagree, and the weighted one is right.

    Going from 3 names to 5 while keeping all 3 means buying two new positions
    and trimming the three incumbents from 33% to 20% to pay for them.  That is
    80% of the book changing hands.  The old model computed turnover as
    `len(set(old) - set(new)) / len(old)`, which is zero here, so every
    expansion in the backtest's history was free.

    In the live config this fires 15 times over 3669 sessions and costs 2.7bp
    of CAGR — small, but it was a subsidy, and a weighted scheme that rebalances
    would have inherited it.
    """
    ex = ExecutionConfig()
    old, new = ["A", "B", "C"], ["A", "B", "C", "D", "E"]
    assert _trade_cost(old, new, ex) == 0.0
    assert _trade_cost_weighted(equal(old), equal(new), ex) == \
        pytest.approx(0.8 * ex.slippage_frac)


def test_reweighting_the_same_names_is_not_free():
    """
    The cost the symbol model structurally cannot see.

    Inverse-vol moves weights as volatility moves, so holding the same names
    still costs money.  If this were free, every weighted scheme would get a
    subsidy the baseline does not.
    """
    ex = ExecutionConfig()
    old = pd.Series({"A": 0.5, "B": 0.5})
    new = pd.Series({"A": 0.7, "B": 0.3})
    assert _trade_cost(["A", "B"], ["A", "B"], ex) == 0.0
    assert _trade_cost_weighted(old, new, ex) == pytest.approx(0.4 * ex.slippage_frac)


# --- the schemes themselves ----------------------------------------------

def _panel(n_days=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    # C is deliberately the low-vol name, A the high-vol one.
    vols = {"A": 0.030, "B": 0.015, "C": 0.005, "D": 0.020}
    data = {s: 100 * np.exp(np.cumsum(rng.normal(0, v, n_days)))
            for s, v in vols.items()}
    return pd.DataFrame(data, index=idx)


def test_equal_scheme_is_exactly_equal():
    close = _panel()
    w = compute_weights(["A", "B", "C"], close.index[-1], close,
                        SizingConfig(scheme="equal"))
    assert w.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_inverse_vol_puts_more_in_the_quiet_name():
    close = _panel()
    w = compute_weights(["A", "B", "C"], close.index[-1], close,
                        SizingConfig(scheme="inverse_vol", max_weight=None))
    assert w["C"] > w["B"] > w["A"]
    assert w.sum() == pytest.approx(1.0)


def test_weights_sum_to_one_for_every_non_targeting_scheme():
    close = _panel()
    scores = pd.DataFrame(np.random.default_rng(1).normal(size=(len(close), 4)),
                          index=close.index, columns=close.columns)
    for scheme in ("equal", "inverse_vol", "score_proportional"):
        w = compute_weights(["A", "B", "C", "D"], close.index[-1], close,
                            SizingConfig(scheme=scheme), scores)
        assert w.sum() == pytest.approx(1.0), scheme


def test_max_weight_binds_and_the_rest_renormalizes():
    close = _panel()
    w = compute_weights(["A", "B", "C"], close.index[-1], close,
                        SizingConfig(scheme="inverse_vol", max_weight=0.40))
    assert w.max() <= 0.40 + 1e-9
    assert w.sum() == pytest.approx(1.0)


def test_an_impossible_cap_degrades_to_equal_weight_not_to_nonsense():
    """cap 0.20 over a 3-name book cannot be met; equal weight is the answer."""
    w = _cap_and_renormalize(pd.Series({"A": 0.7, "B": 0.2, "C": 0.1}), 0.20)
    assert w.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_vol_target_never_levers_above_fully_invested():
    close = _panel()
    # A target far above anything the panel can produce must still cap at 1.0.
    w = compute_weights(["A", "B", "C"], close.index[-1], close,
                        SizingConfig(scheme="vol_target", target_vol=5.0))
    assert w.sum() == pytest.approx(1.0)


def test_vol_target_derisks_when_the_book_is_too_volatile():
    close = _panel()
    w = compute_weights(["A", "B"], close.index[-1], close,
                        SizingConfig(scheme="vol_target", target_vol=0.05))
    assert w.sum() < 1.0


def test_score_proportional_follows_the_score_order():
    close = _panel()
    d = close.index[-1]
    scores = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    scores.loc[d, ["A", "B", "C"]] = [3.0, 2.0, 1.0]
    w = compute_weights(["A", "B", "C"], d, close,
                        SizingConfig(scheme="score_proportional",
                                     max_weight=None), scores)
    assert w["A"] > w["B"] > w["C"]


def test_missing_inputs_fall_back_to_equal_rather_than_guessing():
    close = _panel()
    d = close.index[-1]
    # no scores at all
    w = compute_weights(["A", "B"], d, close,
                        SizingConfig(scheme="score_proportional"), None)
    assert w.tolist() == pytest.approx([0.5, 0.5])
    # a name with no price history
    w = compute_weights(["A", "ZZZ"], d, close,
                        SizingConfig(scheme="inverse_vol"))
    assert w.tolist() == pytest.approx([0.5, 0.5])


def test_weights_do_not_peek_past_the_signal_date():
    """
    Weights must use data up to and including the signal date, never beyond.

    Shifting the signal date back must change the weights; if it does not, the
    estimate is being taken from a fixed (probably final) window.
    """
    close = _panel()
    cfg = SizingConfig(scheme="inverse_vol", max_weight=None)
    early = compute_weights(["A", "B", "C"], close.index[100], close, cfg)
    late = compute_weights(["A", "B", "C"], close.index[-1], close, cfg)
    assert not np.allclose(early.values, late.values)

    # and truncating the future must not move the early answer at all
    truncated = close.iloc[:101]
    same = compute_weights(["A", "B", "C"], truncated.index[-1], truncated, cfg)
    assert np.allclose(early.values, same.values)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unknown sizing scheme"):
        SizingConfig(scheme="magic")


# --------------------------------------------------------------------------
# Per-symbol slippage
# --------------------------------------------------------------------------
#
# The per-name cost path exists so a book that wanders down the cap scale gets
# charged what it would actually cost.  The risk it introduces is that every
# historical number in FINDINGS was computed on the flat path, so the two must
# agree exactly whenever the rates are uniform.  That is asserted here rather
# than assumed, at the function AND through `simulate_portfolio`.

def test_uniform_rates_reproduce_the_flat_cost_exactly():
    ex = ExecutionConfig(slippage_bps=7.5)
    old = pd.Series({"A": 0.5, "B": 0.5})
    new = pd.Series({"B": 0.5, "C": 0.5})

    flat = _trade_cost_weighted(old, new, ex)
    uniform = pd.Series(ex.slippage_frac, index=["A", "B", "C", "D"])
    assert _trade_cost_weighted(old, new, ex, uniform) == pytest.approx(flat, rel=1e-15)

    # initial build, which takes the other branch
    flat0 = _trade_cost_weighted(None, new, ex)
    assert _trade_cost_weighted(None, new, ex, uniform) == pytest.approx(flat0, rel=1e-15)


def test_missing_symbols_fall_back_to_the_flat_rate():
    ex = ExecutionConfig(slippage_bps=10.0)
    old = pd.Series({"A": 1.0})
    new = pd.Series({"B": 1.0})
    # B priced at 50 bps, A absent so it pays the configured 10 bps.
    rates = pd.Series({"B": 0.0050})
    expected = 1.0 * ex.slippage_frac + 1.0 * 0.0050
    assert _trade_cost_weighted(old, new, ex, rates) == pytest.approx(expected)


def test_expensive_names_cost_more_and_scale_linearly():
    ex = ExecutionConfig(slippage_bps=7.5)
    old = pd.Series({"A": 1.0})
    new = pd.Series({"B": 1.0})
    cheap = pd.Series({"A": 0.00075, "B": 0.00075})
    dear = pd.Series({"A": 0.00075, "B": 0.00750})   # B ten times worse
    c = _trade_cost_weighted(old, new, ex, cheap)
    d = _trade_cost_weighted(old, new, ex, dear)
    assert d > c
    # only B's leg got more expensive, so the difference is exactly its delta
    assert d - c == pytest.approx(1.0 * (0.00750 - 0.00075))


def test_simulate_portfolio_uniform_rates_match_flat_end_to_end():
    """
    The check that actually protects FINDINGS: a full simulation with a uniform
    per-name vector must return the identical return stream and total cost as
    the flat path it replaces.
    """
    from momentum.backtest import simulate_portfolio

    close = _panel()
    open_ = close.shift(1).bfill()
    dates = close.index
    targets = pd.Series(
        [["A", "B"] if (i // 14) % 2 == 0 else ["B", "C"] for i in range(len(dates))],
        index=dates)
    ex = ExecutionConfig(execute_at="next_open", slippage_bps=7.5)

    flat = simulate_portfolio(targets, close, open_, execution=ex)
    uniform = pd.Series(ex.slippage_frac, index=close.columns)
    per_name = simulate_portfolio(targets, close, open_, execution=ex,
                                  slippage_by_symbol=uniform)

    assert per_name.total_cost == pytest.approx(flat.total_cost, rel=1e-12)
    assert np.allclose(per_name.returns.to_numpy(),
                       flat.returns.to_numpy(), equal_nan=True)
    assert per_name.metrics["cagr"] == pytest.approx(flat.metrics["cagr"], rel=1e-12)


def test_simulate_portfolio_charges_more_when_a_name_is_illiquid():
    from momentum.backtest import simulate_portfolio

    close = _panel()
    open_ = close.shift(1).bfill()
    dates = close.index
    targets = pd.Series(
        [["A", "B"] if (i // 14) % 2 == 0 else ["B", "C"] for i in range(len(dates))],
        index=dates)
    ex = ExecutionConfig(execute_at="next_open", slippage_bps=7.5)

    flat = simulate_portfolio(targets, close, open_, execution=ex)
    dear = pd.Series(ex.slippage_frac, index=close.columns)
    dear["C"] = 0.01   # 100 bps on the name that rotates in and out
    worse = simulate_portfolio(targets, close, open_, execution=ex,
                               slippage_by_symbol=dear)

    assert worse.total_cost > flat.total_cost
    assert worse.metrics["cagr"] < flat.metrics["cagr"]
