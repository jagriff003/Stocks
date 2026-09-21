"""
Tests for the synthetic-delisting generator.

The generator's whole job is to produce names that behave like listings which
ended.  If it quietly produces names that keep trading past their own delisting
date, or that never live long enough to be picked, the survivorship bound built
on it would still print a smooth, plausible curve — and be measuring nothing.
So the properties asserted here are the ones whose violation is invisible
downstream.

`test_zero_rate_is_a_bit_identical_no_op` is the most important: the rate=0 row
of the bound curve is the un-augmented backtest, and every other row is read
relative to it.

Run:  python -m pytest tests/test_delisting.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.delisting import (  # noqa: E402
    DelistingBranch, DelistingConfig, audit, build_synthetic_panel, n_synthetic,
)


def panel(n_days=1200, n_cols=25, seed=0):
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, (n_days, n_cols)), axis=0)),
        index=idx, columns=[f"S{i}" for i in range(n_cols)])
    open_ = close * 1.001
    volume = pd.DataFrame(rng.lognormal(14, 1, (n_days, n_cols)),
                          index=idx, columns=close.columns)
    return close, open_, volume


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def test_branch_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to"):
        DelistingConfig(branches=[
            DelistingBranch("a", 0.5, -1.0, 0.0),
            DelistingBranch("b", 0.2, -1.0, 0.0),
        ])


def test_negative_rate_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        DelistingConfig(annual_rate=-0.01)


def test_count_scales_with_rate_and_history():
    assert n_synthetic(600, 2520, DelistingConfig(annual_rate=0.0)) == 0
    ten_years = n_synthetic(600, 2520, DelistingConfig(annual_rate=0.05))
    twenty = n_synthetic(600, 5040, DelistingConfig(annual_rate=0.05))
    assert twenty == pytest.approx(2 * ten_years, rel=0.02)


# --------------------------------------------------------------------------
# The properties the bound depends on
# --------------------------------------------------------------------------

def test_zero_rate_is_a_bit_identical_no_op():
    """
    The rate=0 row is the baseline every other row is read against.  If it is
    not exactly the un-augmented panel, the whole curve is measured from a
    moved origin.
    """
    close, open_, volume = panel()
    c, o, v, manifest = build_synthetic_panel(
        close, open_, volume, DelistingConfig(annual_rate=0.0), seed=3)
    assert manifest.empty
    assert c.equals(close)
    assert o.equals(open_)
    assert list(v.columns) == list(volume.columns)


def test_synthetic_names_exist_before_and_vanish_after_delisting():
    close, open_, volume = panel()
    cfg = DelistingConfig(annual_rate=0.10)
    c, o, v, manifest = build_synthetic_panel(close, open_, volume, cfg, seed=5)
    assert len(manifest) > 0

    for _, r in manifest.iterrows():
        col = c[r["Symbol"]]
        assert col.loc[:r["Delist"]].notna().sum() > 0, r["Symbol"]
        after = col.loc[r["Delist"]:].iloc[1:]
        assert after.notna().sum() == 0, f"{r['Symbol']} trades past its delisting"


def test_audit_agrees_with_the_panel():
    close, open_, volume = panel()
    cfg = DelistingConfig(annual_rate=0.10)
    c, _, _, manifest = build_synthetic_panel(close, open_, volume, cfg, seed=5)
    a = audit(manifest, c, cfg)
    assert a["count"] == len(manifest)
    assert a["alive_before"] == a["count"]
    assert a["dead_after"] == a["count"]


def test_terminal_return_is_booked_exactly():
    close, open_, volume = panel()
    cfg = DelistingConfig(annual_rate=0.10)
    c, _, _, manifest = build_synthetic_panel(close, open_, volume, cfg, seed=5)

    checked = 0
    for _, r in manifest.iterrows():
        if r["TerminalReturn"] <= -0.999:
            continue          # price is NaN'd, nothing to compare against
        i = c.index.get_loc(r["Delist"])
        col = c[r["Symbol"]]
        if not np.isfinite(col.iloc[i]) or not np.isfinite(col.iloc[i - 1]):
            continue
        realized = col.iloc[i] / col.iloc[i - 1] - 1
        assert realized == pytest.approx(r["TerminalReturn"], abs=1e-9)
        checked += 1
    assert checked > 5


def test_total_loss_branch_leaves_no_tradable_price():
    """A -100% terminal must not leave a zero price that something divides by."""
    close, open_, volume = panel()
    cfg = DelistingConfig(annual_rate=0.30, branches=[
        DelistingBranch("chapter_7", 1.0, terminal_return=-1.0,
                        decline=-0.5, decline_days=100)])
    c, o, _, manifest = build_synthetic_panel(close, open_, volume, cfg, seed=11)
    assert len(manifest) > 0
    for sym in manifest["Symbol"]:
        vals = c[sym].to_numpy()
        finite = vals[np.isfinite(vals)]
        assert (finite > 0).all(), sym


def test_branch_mixture_converges_to_its_weights():
    close, open_, volume = panel(n_cols=200)
    cfg = DelistingConfig(annual_rate=0.25)
    _, _, _, manifest = build_synthetic_panel(close, open_, volume, cfg, seed=7)
    assert len(manifest) > 100

    got = manifest["Branch"].value_counts(normalize=True)
    for b in cfg.branches:
        if b.weight >= 0.10:            # small branches need a bigger sample
            assert got.get(b.name, 0) == pytest.approx(b.weight, abs=0.08), b.name


def test_cause_branches_decline_before_they_die():
    """
    The pre-delisting decline is the reason this is a bound and not a caricature:
    the loss is mostly on the way down, where a screen can still act.
    """
    close, open_, volume = panel(n_cols=120)
    cfg = DelistingConfig(annual_rate=0.25)
    c, _, _, manifest = build_synthetic_panel(close, open_, volume, cfg, seed=7)

    ch11 = manifest[manifest["Branch"] == "chapter_11"]
    assert len(ch11) > 0
    branch = next(b for b in cfg.branches if b.name == "chapter_11")

    seen = 0
    for _, r in ch11.head(8).iterrows():
        i = c.index.get_loc(r["Delist"])
        window = c[r["Symbol"]].iloc[max(0, i - branch.decline_days):i].dropna()
        if len(window) < branch.decline_days // 2:
            continue
        drop = window.iloc[-1] / window.iloc[0] - 1
        assert drop < -0.3, f"{r['Symbol']} only fell {drop:.1%} before Ch11"
        seen += 1
    assert seen > 0


def test_mergers_do_not_decline_first():
    """The merger branch is a premium, not a collapse — and it is the branch
    that makes the bound honest rather than alarmist."""
    close, open_, volume = panel(n_cols=120)
    cfg = DelistingConfig(annual_rate=0.25)
    c, _, _, manifest = build_synthetic_panel(close, open_, volume, cfg, seed=7)

    merged = manifest[manifest["Branch"] == "merger"]
    assert len(merged) > 0
    for _, r in merged.head(5).iterrows():
        i = c.index.get_loc(r["Delist"])
        assert c[r["Symbol"]].iloc[i] / c[r["Symbol"]].iloc[i - 1] - 1 == \
            pytest.approx(0.20, abs=1e-9)


def test_synthetic_names_are_not_duplicates_of_their_donor():
    """
    Donor returns are circularly shifted so a synthetic name is not the same
    series sitting beside its donor in the same cross-section — which would
    make the pool artificially correlated and the ranking degenerate.
    """
    close, open_, volume = panel(n_cols=40)
    cfg = DelistingConfig(annual_rate=0.15)
    c, _, _, manifest = build_synthetic_panel(close, open_, volume, cfg, seed=9)

    for _, r in manifest.head(6).iterrows():
        a = c[r["Symbol"]].pct_change()
        b = c[r["Donor"]].pct_change()
        both = a.notna() & b.notna()
        if both.sum() < 100:
            continue
        assert abs(a[both].corr(b[both])) < 0.9, r["Symbol"]
