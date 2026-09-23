"""
Tests for the append-only market store.

The failure these guard against is silent: an append that splices two
different adjusted-price scales, a revised dividend that never surfaces, or an
intraday bar stored as a close.  Each would leave a plausible-looking series.
A fake fetcher stands in for Yahoo so every case is deterministic.

Run:  python -m pytest tests/test_marketstore.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum import marketstore as ms  # noqa: E402

ET_EVENING = datetime(2026, 3, 31, 18, 0)


class FakeYahoo:
    """Adjusted closes from a fixed return path, re-scaled like Yahoo re-scales."""

    def __init__(self, n=300, end="2026-03-31", seed=0, symbols=("SPY", "IAU")):
        rng = np.random.default_rng(seed)
        self.idx = pd.bdate_range(end=end, periods=n)
        self.rets = pd.DataFrame(rng.normal(0.0004, 0.01, (n, len(symbols))),
                                 index=self.idx, columns=list(symbols))
        self.scale = 1.0
        self.last = self.idx[-1]

    def __call__(self, symbols, start):
        r = self.rets.loc[start:self.last, [s for s in symbols if s in self.rets]]
        px = 100 * (1 + r).cumprod() * self.scale
        return px


def build(tmp_path, fake, **kw):
    return ms.update_daily(["SPY", "IAU"], store_dir=tmp_path, fetch=fake,
                           now_et=kw.pop("now_et", ET_EVENING), **kw)


def test_build_then_append_reproduces_one_uninterrupted_fetch(tmp_path):
    fake = FakeYahoo()
    fake.last = fake.idx[-40]
    build(tmp_path, fake)
    fake.last = fake.idx[-1]
    fake.scale = 0.97            # a dividend went ex: Yahoo re-scaled all history
    rep, status = build(tmp_path, fake)
    assert status == 0, rep
    stored = ms.load_daily_returns(store_dir=tmp_path)
    expect = fake.rets.iloc[1:]  # first day of the whole history has no return
    pd.testing.assert_frame_equal(stored[["IAU", "SPY"]], expect[["IAU", "SPY"]],
                                  check_freq=False, check_names=False, rtol=1e-9)
    assert (rep["appended"] == 39).all()


def test_revision_in_the_overlap_is_refused_then_accepted(tmp_path):
    fake = FakeYahoo()
    fake.last = fake.idx[-20]
    build(tmp_path, fake)
    fake.last = fake.idx[-1]
    d = fake.idx[-25]
    fake.rets.loc[d, "IAU"] += 0.004          # a late-posted dividend
    rep, status = build(tmp_path, fake)
    assert status == 2 and "REVISION REFUSED" in rep.at["IAU", "status"]
    stored = ms.load_daily_returns(store_dir=tmp_path)
    assert stored["IAU"].last_valid_index() == fake.idx[-20]    # IAU not written
    assert stored["SPY"].last_valid_index() == fake.idx[-1]     # SPY was
    rep, status = build(tmp_path, fake, accept_revisions=True)
    stored = ms.load_daily_returns(store_dir=tmp_path)
    assert np.isclose(stored.at[d, "IAU"], fake.rets.at[d, "IAU"])
    assert stored["IAU"].last_valid_index() == fake.idx[-1]


def test_unsettled_intraday_bar_is_not_stored(tmp_path):
    fake = FakeYahoo()
    afternoon = datetime(2026, 3, 31, 15, 30)
    build(tmp_path, fake, now_et=afternoon)
    stored = ms.load_daily_returns(store_dir=tmp_path)
    assert stored.index[-1] == fake.idx[-2]
    build(tmp_path, fake)                                        # after the settle
    assert ms.load_daily_returns(store_dir=tmp_path).index[-1] == fake.idx[-1]


def test_stale_symbol_is_flagged(tmp_path):
    fake = FakeYahoo()
    build(tmp_path, fake)
    # next session: SPY prints, IAU does not
    fake.rets.loc[pd.Timestamp("2026-04-01")] = [0.001, np.nan]
    fake.last = pd.Timestamp("2026-04-01")
    rep, status = ms.update_daily(["SPY", "IAU"], store_dir=tmp_path, fetch=fake,
                                  now_et=datetime(2026, 4, 1, 18, 0))
    assert status == 2 and "STALE" in rep.at["IAU", "status"]


def test_a_new_symbol_gets_its_whole_history(tmp_path):
    fake = FakeYahoo(symbols=("SPY", "IAU", "SLV"))
    ms.update_daily(["SPY", "IAU"], store_dir=tmp_path, fetch=fake, now_et=ET_EVENING)
    rep, status = ms.update_daily(["SLV"], store_dir=tmp_path, fetch=fake, now_et=ET_EVENING)
    stored = ms.load_daily_returns(store_dir=tmp_path)
    assert set(stored.columns) == {"SPY", "IAU", "SLV"}
    assert stored["SLV"].notna().sum() == len(fake.idx) - 1


def test_monthly_returns_compound_the_daily_store():
    idx = pd.bdate_range("2020-01-15", "2020-04-30")
    r = pd.DataFrame({"A": 0.001}, index=idx)
    m = ms.monthly_returns(r)
    assert np.isnan(m.loc["2020-01-31", "A"])                   # partial first month
    feb = r.loc["2020-02"]
    assert np.isclose(m.loc["2020-02-29", "A"], (1 + feb["A"]).prod() - 1)


def test_missing_symbol_error_says_how_to_fix_it(tmp_path):
    build(tmp_path, FakeYahoo())
    with pytest.raises(KeyError, match="--add"):
        ms.load_daily_returns(["GLD"], store_dir=tmp_path)


def test_a_yahoo_hole_is_carried_flat_and_the_move_is_kept(tmp_path):
    fake = FakeYahoo()
    hole = fake.idx[-2]

    def holed(symbols, start):
        px = fake(symbols, start)
        if "IAU" in px:
            px.loc[hole, "IAU"] = np.nan          # Yahoo has no IAU bar that day
        return px

    rep, status = build(tmp_path, holed)
    assert status == 2 and "FILLED" in rep.at["IAU", "status"]
    stored = ms.load_daily_returns(store_dir=tmp_path)
    assert stored.at[hole, "IAU"] == 0.0
    two_day = (1 + fake.rets.loc[fake.idx[-2:], "IAU"]).prod() - 1
    assert np.isclose(stored.at[fake.idx[-1], "IAU"], two_day)
    # Yahoo later posts the real bar: a revision, refused until accepted
    rep, status = build(tmp_path, fake)
    assert "REVISION REFUSED" in rep.at["IAU", "status"]
    build(tmp_path, fake, accept_revisions=True)
    stored = ms.load_daily_returns(store_dir=tmp_path)
    assert np.isclose(stored.at[hole, "IAU"], fake.rets.at[hole, "IAU"])


def test_a_missing_last_bar_is_stale_not_flat(tmp_path):
    fake = FakeYahoo()

    def late(symbols, start):
        px = fake(symbols, start)
        if "IAU" in px:
            px.loc[fake.idx[-1], "IAU"] = np.nan
        return px

    rep, status = build(tmp_path, late)
    stored = ms.load_daily_returns(store_dir=tmp_path)
    assert stored["IAU"].last_valid_index() == fake.idx[-2]
    assert "STALE" in rep.at["IAU", "status"]
