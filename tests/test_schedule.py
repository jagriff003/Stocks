"""
Tests for the rotation calendar.

The whole point of this module is that the rotation weekday does NOT drift, so
that is what is asserted hardest — including across market holidays, which is
exactly where the session-count clock it replaces failed. A schedule that
quietly slips a weekday would be invisible until a book was traded two sessions
stale.

Run:  python -m pytest tests/test_schedule.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.schedule import (  # noqa: E402
    MONDAY, TUESDAY, current_sleeves, next_rotation, rotation_dates,
    sleeve_assignment,
)


def sessions(start="2026-01-01", periods=400, drop=()):
    """Business days, optionally with holidays removed."""
    idx = pd.bdate_range(start, periods=periods)
    if drop:
        idx = idx.drop([pd.Timestamp(d) for d in drop if pd.Timestamp(d) in idx])
    return idx


def test_every_rotation_lands_on_the_target_weekday():
    idx = sessions()
    for wd in (MONDAY, TUESDAY):
        dates = rotation_dates(idx, weekday=wd, every_weeks=2)
        assert len(dates) > 10
        assert all(d.weekday() == wd for d in dates), [
            (d.date(), d.day_name()) for d in dates if d.weekday() != wd]


def test_weekday_holds_across_holidays():
    """
    The failure mode this module exists for: a session-count clock drifts a
    weekday for every holiday inside the cycle. Dropping Tuesdays forces the
    schedule to fall back, and it must fall back FORWARD, not to the previous
    session.
    """
    holidays = ["2026-01-20", "2026-03-17", "2026-05-26"]
    idx = sessions(drop=holidays)
    dates = rotation_dates(idx, weekday=TUESDAY, every_weeks=2)

    for d in dates:
        # Either it is a Tuesday, or the Tuesday it replaces was not a session
        # and this is the next one available.
        if d.weekday() != TUESDAY:
            prior_tue = d - pd.Timedelta(days=(d.weekday() - TUESDAY) % 7)
            assert prior_tue not in idx, (d.date(), prior_tue.date())
            assert d > prior_tue, "fell back to an earlier session"


def test_spacing_is_the_requested_number_of_weeks():
    idx = sessions()
    dates = rotation_dates(idx, weekday=TUESDAY, every_weeks=2)
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    assert set(gaps) == {14}


def test_anchor_sets_the_phase():
    """
    Aligning to the production model's schedule is the reason `anchor` exists:
    it is what lets both models be run and traded on the same day.
    """
    idx = sessions()
    a = pd.Timestamp("2026-09-15")   # a Tuesday
    dates = rotation_dates(idx, weekday=TUESDAY, every_weeks=2, anchor=a)
    if a in idx:
        assert a in dates
    # every date is a whole number of fortnights from the anchor
    assert all(abs((d - a).days) % 14 == 0 for d in dates)


def test_a_different_anchor_gives_the_opposite_fortnight():
    idx = sessions()
    on = rotation_dates(idx, weekday=TUESDAY, every_weeks=2,
                        anchor=pd.Timestamp("2026-09-15"))
    off = rotation_dates(idx, weekday=TUESDAY, every_weeks=2,
                         anchor=pd.Timestamp("2026-09-22"))
    assert set(on).isdisjoint(set(off))


def test_sleeves_take_turns_and_each_holds_k_cycles():
    idx = sessions()
    dates = rotation_dates(idx, weekday=TUESDAY, every_weeks=2)
    assign = sleeve_assignment(dates, 4)
    seq = [assign[d] for d in dates]
    assert seq[:8] == [0, 1, 2, 3, 0, 1, 2, 3]

    # a sleeve's own rotations are 4 cycles = 8 weeks apart
    own = [d for d in dates if assign[d] == 2]
    assert all((b - a).days == 56 for a, b in zip(own, own[1:]))


def test_current_sleeves_reports_one_date_per_sleeve():
    idx = sessions()
    dates = rotation_dates(idx, weekday=TUESDAY, every_weeks=2)
    cur = current_sleeves(dates, 4, as_of=dates[9])
    assert set(cur) == {0, 1, 2, 3}
    # the most recently rotated sleeve is the one that rotated on dates[9]
    assert cur[sleeve_assignment(dates, 4)[dates[9]]] == dates[9]
    # and every sleeve's date is at or before the as-of
    assert all(v <= dates[9] for v in cur.values())


def test_next_rotation_is_projected_beyond_the_panel():
    idx = sessions()
    dates = rotation_dates(idx, weekday=TUESDAY, every_weeks=2)
    nxt, sleeve, projected = next_rotation(dates, 4, idx, weekday=TUESDAY,
                                           every_weeks=2, as_of=idx[-1])
    assert nxt is not None
    assert nxt.weekday() == TUESDAY
    assert nxt > idx[-1]
    assert projected is True
    assert sleeve in (0, 1, 2, 3)


def test_next_rotation_inside_the_panel_is_not_projected():
    idx = sessions()
    dates = rotation_dates(idx, weekday=TUESDAY, every_weeks=2)
    nxt, sleeve, projected = next_rotation(dates, 4, idx, as_of=dates[5])
    assert nxt == dates[6]
    assert projected is False
    assert sleeve == sleeve_assignment(dates, 4)[dates[6]]


def test_empty_calendar_is_handled():
    assert len(rotation_dates(pd.DatetimeIndex([]))) == 0
