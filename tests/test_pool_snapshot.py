"""
Tests for the point-in-time pool record.

These snapshots are the only asset that can ever close the survivorship question
(TODO 0d), and they are worthless retroactively — a bug that silently writes an
incomplete or wrong snapshot destroys years of accumulation and would not be
noticed until someone tried to use it. So the properties asserted here are the
ones that would be expensive to discover late: that a non-tradable name is still
RECORDED (rather than dropped, which would make it indistinguishable from a
delisting), and that re-snapshotting a day replaces rather than duplicates.

Run:  python -m pytest tests/test_pool_snapshot.py -v
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum import pool_snapshot  # noqa: E402
from momentum.pool_snapshot import (  # noqa: E402
    coverage, read_snapshot, snapshot_pool, vanished,
)


@pytest.fixture
def tmp_pool_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pool_snapshot, "POOL_DIR", tmp_path)
    return tmp_path


def frame():
    tradable = pd.Series({"AAA": True, "BBB": False, "CCC": True})
    dv = pd.Series({"AAA": 5e7, "BBB": 3e6, "CCC": 2e8})
    px = pd.Series({"AAA": 42.0, "BBB": 3.1, "CCC": 190.0})
    sc = pd.Series({"AAA": 1.2, "BBB": -0.4, "CCC": 2.9})
    return tradable, dv, px, sc


def test_untradable_names_are_recorded_not_dropped(tmp_pool_dir):
    """
    The whole point is to tell 'failed the screen' apart from 'ceased to exist'.
    Dropping the failures would make those two indistinguishable in the record,
    which is precisely the distinction the survivorship question turns on.
    """
    tradable, dv, px, sc = frame()
    p = snapshot_pool(tradable, dv, px, sc, as_of=date(2026, 1, 15))
    d = read_snapshot(p)
    assert set(d["Symbol"]) == {"AAA", "BBB", "CCC"}
    assert d.set_index("Symbol").loc["BBB", "Tradable"] == "N"
    assert d.set_index("Symbol").loc["AAA", "Tradable"] == "Y"


def test_context_columns_survive_the_round_trip(tmp_pool_dir):
    tradable, dv, px, sc = frame()
    p = snapshot_pool(tradable, dv, px, sc, as_of=date(2026, 1, 15))
    d = read_snapshot(p).set_index("Symbol")
    assert float(d.loc["AAA", "Price"]) == pytest.approx(42.0)
    assert float(d.loc["CCC", "DollarVolume"]) == pytest.approx(2e8)
    # rank is on the score, best first
    assert int(d.loc["CCC", "Rank"]) == 1
    assert int(d.loc["BBB", "Rank"]) == 3


def test_resnapshotting_a_day_replaces_rather_than_duplicates(tmp_pool_dir):
    tradable, dv, px, sc = frame()
    snapshot_pool(tradable, dv, px, sc, as_of=date(2026, 1, 15))
    smaller = pd.Series({"AAA": True})
    p = snapshot_pool(smaller, as_of=date(2026, 1, 15))
    assert len(list(tmp_pool_dir.glob("pool_2026-01-15*.csv"))) == 1
    assert len(read_snapshot(p)) == 1


def test_missing_optional_columns_are_blank_not_an_error(tmp_pool_dir):
    p = snapshot_pool(pd.Series({"AAA": True, "BBB": False}),
                      as_of=date(2026, 2, 1))
    d = read_snapshot(p)
    assert len(d) == 2
    assert d["Price"].isna().all()


def test_vanished_reports_both_directions(tmp_pool_dir):
    a = snapshot_pool(pd.Series({"AAA": True, "BBB": True}),
                      as_of=date(2026, 1, 15))
    b = snapshot_pool(pd.Series({"AAA": True, "CCC": True}),
                      as_of=date(2026, 3, 15))
    v = vanished(a, b)
    assert v["gone"] == ["BBB"]
    assert v["new"] == ["CCC"]


def test_coverage_lists_snapshots_oldest_first(tmp_pool_dir):
    snapshot_pool(pd.Series({"AAA": True}), as_of=date(2026, 3, 15))
    snapshot_pool(pd.Series({"AAA": True, "BBB": False}),
                  as_of=date(2026, 1, 15))
    c = coverage()
    assert list(c["AsOf"]) == ["2026-01-15", "2026-03-15"]
    assert list(c["Names"]) == [2, 1]
    assert list(c["Tradable"]) == [1, 1]


def test_coverage_is_empty_without_error_when_nothing_recorded(tmp_pool_dir):
    assert coverage().empty
