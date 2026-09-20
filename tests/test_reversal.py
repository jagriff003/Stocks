"""
Tests for the trend-change signal family.

Two of these matter more than the rest.

`test_no_lookahead_under_truncation` is the one that protects the whole study.
Every term here is built from rolling windows and cross-sectional operations,
and the failure mode is not a crash — it is a term that quietly reads one day
into the future and produces a beautiful, entirely fake information
coefficient.  Truncating the panel and requiring the last value to be unchanged
is the only cheap test that catches it.

`test_rolling_ols_matches_linregress` guards the vectorization.  The rolling
slope is computed from rolling sums and a closed form rather than by fitting,
which is fast and is exactly the kind of algebra that can be wrong in a way
that still returns plausible numbers.  scipy is an independent implementation.

Run:  python -m pytest tests/test_reversal.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.reversal import (  # noqa: E402
    ReversalConfig, build_terms, composite, momentum_flip, range_position,
    rolling_ols, slope_change_t, trend_strength_t,
)


def panel(seed=0, n=600, cols=6):
    """A price panel with enough history for a 252-day window."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    steps = rng.normal(0.0004, 0.015, size=(n, len(range(cols))))
    prices = 100 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(prices, index=idx,
                        columns=[f"S{i}" for i in range(cols)])


def v_shape(n=400, turn_at=200, slope=0.002, sign=1.0, seed=11):
    """
    A kinked trend: one rate until `turn_at`, the opposite rate after.

    `sign=+1` falls then rises (an upward turn), `sign=-1` rises then falls.

    A little noise is deliberate.  A noiseless piecewise-linear branch fits
    perfectly, so the regression's residual variance is zero and the
    t-statistic is either infinite or NaN depending on where the float lands —
    a degenerate case that says nothing about whether the term works.
    """
    idx = pd.bdate_range("2018-01-01", periods=n)
    t = np.arange(n, dtype=float)
    ramp = np.where(t <= turn_at, -slope * t,
                    -slope * turn_at + slope * (t - turn_at))
    noise = np.random.default_rng(seed).normal(0, 0.004, n)
    return pd.DataFrame({"V": 100 * np.exp(sign * ramp + noise)}, index=idx)


# --------------------------------------------------------------------------
# Rolling OLS
# --------------------------------------------------------------------------

def test_rolling_ols_matches_linregress():
    linregress = pytest.importorskip("scipy.stats").linregress
    px = panel()
    window = 60
    slope, se = rolling_ols(px, window)
    y = np.log(px)

    for col in px.columns:
        for i in (120, 300, 599):
            seg = y[col].iloc[i - window + 1:i + 1].to_numpy()
            fit = linregress(np.arange(window, dtype=float), seg)
            assert slope[col].iloc[i] == pytest.approx(fit.slope, abs=1e-12)
            assert se[col].iloc[i] == pytest.approx(fit.stderr, abs=1e-12)


def test_rolling_ols_recovers_a_known_slope_exactly():
    """A noiseless exponential has a known log-slope and zero residual."""
    n, slope = 300, 0.003
    idx = pd.bdate_range("2018-01-01", periods=n)
    px = pd.DataFrame({"E": 100 * np.exp(slope * np.arange(n))}, index=idx)

    s, se = rolling_ols(px, 60)
    assert s["E"].iloc[-1] == pytest.approx(slope, abs=1e-12)
    # Perfect fit: standard error is zero, which the module maps to NaN so that
    # dividing by it yields NaN rather than an infinite t-statistic.
    assert np.isnan(se["E"].iloc[-1])
    assert np.isnan(trend_strength_t(px, ReversalConfig(slope_window=60))["E"].iloc[-1])


def test_rolling_ols_needs_a_full_window():
    px = panel(n=100)
    slope, _ = rolling_ols(px, 60)
    assert slope.iloc[:59].isna().all().all()
    assert slope.iloc[59:].notna().any().any()


# --------------------------------------------------------------------------
# The terms behave the way their names claim
# --------------------------------------------------------------------------

def test_slope_change_is_positive_on_an_upward_turn():
    cfg = ReversalConfig(slope_window=60)
    turn = slope_change_t(v_shape(), cfg)["V"]
    # Just after the low, the recent window rises and the prior one fell.
    assert turn.iloc[265] > 3.0


def test_slope_change_is_negative_on_a_downward_turn():
    cfg = ReversalConfig(slope_window=60)
    assert slope_change_t(v_shape(sign=-1.0), cfg)["V"].iloc[265] < -3.0


def test_steady_trend_has_no_turn_but_has_strength():
    """The separation the two terms exist for: strong, but unchanged."""
    n = 400
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(3)
    y = 0.001 * np.arange(n) + rng.normal(0, 0.004, n)
    px = pd.DataFrame({"T": 100 * np.exp(y)}, index=idx)

    cfg = ReversalConfig(slope_window=60)
    assert abs(slope_change_t(px, cfg)["T"].iloc[-1]) < 2.0
    assert trend_strength_t(px, cfg)["T"].iloc[-1] > 3.0


def test_range_position_is_bounded_and_hits_its_endpoints():
    cfg = ReversalConfig(room_window=252)
    px = panel(n=600)
    rp = range_position(px, cfg)
    vals = rp.to_numpy()
    vals = vals[np.isfinite(vals)]
    assert vals.min() >= -1e-12
    assert vals.max() <= 1 + 1e-12

    # A series ending at its own 52-week high sits at 1.0, at its low at 0.0.
    idx = pd.bdate_range("2018-01-01", periods=300)
    up = pd.DataFrame({"U": np.linspace(10, 50, 300)}, index=idx)
    assert range_position(up, cfg)["U"].iloc[-1] == pytest.approx(1.0)
    down = pd.DataFrame({"D": np.linspace(50, 10, 300)}, index=idx)
    assert range_position(down, cfg)["D"].iloc[-1] == pytest.approx(0.0)


def test_momentum_flip_ranks_recent_strength_against_long_weakness():
    """
    Two names, opposite shapes.  The one that was weak for a year and strong
    lately must score above the one that was strong and has faded.
    """
    n = 400
    idx = pd.bdate_range("2018-01-01", periods=n)
    t = np.arange(n, dtype=float)
    # A: falls for the first 340 sessions, rises hard after.
    a = np.where(t <= 340, -0.002 * t, -0.68 + 0.006 * (t - 340))
    px = pd.DataFrame({"A": 100 * np.exp(a), "B": 100 * np.exp(-a)}, index=idx)

    flip = momentum_flip(px, ReversalConfig())
    assert flip["A"].iloc[-1] > flip["B"].iloc[-1]


# --------------------------------------------------------------------------
# The property the whole study rests on
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cut", [400, 500, 575])
def test_no_lookahead_under_truncation(cut):
    """
    Every term at date T must be computable from data up to T.

    If a window were centred, or a shift went the wrong way, or a
    normalization reached across the full history, the value at T would change
    when the future is removed.  Nothing else in the suite would notice.
    """
    px = panel(n=600, cols=8)
    cfg = ReversalConfig()
    full = build_terms(px, cfg)
    cropped = build_terms(px.iloc[:cut + 1], cfg)

    for name, frame in full.items():
        a, b = frame.iloc[cut], cropped[name].iloc[-1]
        both = a.notna() & b.notna()
        assert both.any(), f"{name} produced nothing to compare at {cut}"
        assert (a[both] - b[both]).abs().max() < 1e-12, name


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def test_room_weight_sign_flips_the_term_contribution():
    """
    The sign of the room term is a free parameter by design — the measurement
    decides it, not the construction.  This pins that the weight actually does
    what it says.
    """
    from dataclasses import replace
    px = panel(n=600, cols=8)
    cfg = ReversalConfig(turn_weight=1.0, strength_weight=1.0, room_weight=0.0)
    terms = build_terms(px, cfg)

    neutral = composite(terms, cfg, "turn_t")
    plus = composite(terms, replace(cfg, room_weight=1.0), "turn_t")
    minus = composite(terms, replace(cfg, room_weight=-1.0), "turn_t")

    row = slice(-1, None)
    assert not np.allclose(plus.iloc[row].to_numpy(),
                           minus.iloc[row].to_numpy(), equal_nan=True)
    # plus and minus straddle the neutral score by equal and opposite amounts.
    mid = (plus.iloc[row] + minus.iloc[row]) / 2
    assert np.allclose(mid.to_numpy(), neutral.iloc[row].to_numpy(),
                       equal_nan=True, atol=1e-12)


def test_composite_rejects_an_unknown_turn_term():
    px = panel(n=300)
    cfg = ReversalConfig()
    with pytest.raises(KeyError):
        composite(build_terms(px, cfg), cfg, "not_a_term")


def test_normalization_modes_are_unit_scale():
    """
    Both modes come out on roughly unit cross-sectional variance, which is what
    makes the weights in `composite` mean what they say.
    """
    px = panel(n=600, cols=30)
    terms = build_terms(px, ReversalConfig())
    for mode in ("cross_sectional", "rank"):
        cfg = ReversalConfig(normalization=mode)
        z = cfg.normalize(terms["strength_t"])
        sd = z.iloc[-1].std()
        assert 0.8 < sd < 1.2, (mode, sd)


def test_unknown_normalization_is_rejected():
    with pytest.raises(ValueError):
        ReversalConfig(normalization="nope").normalize(panel(n=300))
