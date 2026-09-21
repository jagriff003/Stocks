"""
The rotation calendar: a weekday anchor instead of a session count.

WHY A SESSION COUNT DOES NOT WORK

`hold_days=40` was chosen because 40 sessions is eight weeks, so the rotation
should land on the same weekday every cycle.  It does not.  Forty *sessions* is
eight *weeks* only when no market holiday falls inside the cycle; each holiday
stretches the cycle by a weekday, so the rotation walks forward and then
oscillates.  Measured on the live panel, Track J's last six rotations were Tue,
Mon, Mon, Tue, Mon, Mon — while the production model, on a 14-session clock,
landed on a Tuesday 28 times out of 30 by luck rather than design.

That matters operationally.  The pattern is: run after Tuesday's close, trade
Wednesday's open.  A rotation that drifts to Monday is traded two sessions late,
and the RUNBOOK's finding is that the cost is dispersion rather than drift — a
*one*-session-stale panel picks a different name 56% of the time.

WHAT THIS DOES INSTEAD

Rotation dates are defined on the calendar — every `every_weeks` weeks on a
named weekday — and then mapped onto actual trading sessions.  The schedule is
therefore publishable in advance and immune to holiday drift.  The hold length
becomes the variable instead, which is the right thing to let float: the
parameter sweep was flat across holds from 5 to 63 sessions at `top_n=8`, so a
hold that varies between 38 and 42 sessions costs nothing, while a rotation
weekday that varies costs a two-session-stale book.

WHEN THE TARGET DAY IS A HOLIDAY

The rotation moves to the next available session, not the previous one. Moving
earlier would mean signalling on a session that had not happened when the
schedule was published; moving later only delays.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd

#: Monday = 0, matching `Timestamp.weekday()`.
MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY = range(5)


def rotation_dates(sessions: pd.DatetimeIndex,
                   weekday: int = TUESDAY,
                   every_weeks: int = 2,
                   anchor: Optional[pd.Timestamp] = None,
                   start: Optional[pd.Timestamp] = None) -> pd.DatetimeIndex:
    """
    Trading sessions on which a rotation happens.

    Parameters
    ----------
    sessions : DatetimeIndex
        The trading calendar to map onto — normally the price panel's index.
    weekday : int
        0 = Monday ... 4 = Friday. Tuesday by default, so the book is built
        from Tuesday's close and traded at Wednesday's open.
    every_weeks : int
        Weeks between rotations. 2 puts Track J on the same biweekly cadence as
        the production model.
    anchor : Timestamp, optional
        A known rotation date to align the phase to. Passing the production
        model's most recent rotation makes the two schedules coincide, which is
        what allows both models to be run and traded on the same day.
    start : Timestamp, optional
        Ignore sessions before this.

    Returns
    -------
    DatetimeIndex of actual sessions, ascending.
    """
    if len(sessions) == 0:
        return pd.DatetimeIndex([])
    sessions = pd.DatetimeIndex(sessions).sort_values()

    lo = pd.Timestamp(start) if start is not None else sessions[0]
    hi = sessions[-1]

    ref = pd.Timestamp(anchor) if anchor is not None else lo
    # Snap to the target weekday, then walk the phase to the FIRST on-schedule
    # day at or after the panel start.
    #
    # Walking only backwards (the first version) left `ref` before the panel,
    # and a target day before the panel maps forward onto the panel's first
    # session — whatever weekday that happens to be. That silently produced
    # off-weekday rotations at the start of every record.
    ref = ref - pd.Timedelta(days=(ref.weekday() - weekday) % 7)
    stride = pd.Timedelta(weeks=every_weeks)
    while ref > lo:
        ref -= stride
    while ref < lo:
        ref += stride

    out: List[pd.Timestamp] = []
    d = ref
    while d <= hi:
        # The target calendar day, mapped to the next available session. Later,
        # never earlier: an earlier session had not happened when the schedule
        # was published. Capped at a week, so a target that falls in a gap
        # bigger than any real holiday run is skipped rather than displaced
        # somewhere arbitrary.
        nxt = sessions[sessions >= d]
        if len(nxt):
            s = nxt[0]
            if (s - d).days <= 6 and lo <= s <= hi and (not out or s > out[-1]):
                out.append(s)
        d += stride

    return pd.DatetimeIndex(out)


def sleeve_assignment(dates: Sequence[pd.Timestamp],
                      tranches: int) -> Dict[pd.Timestamp, int]:
    """
    Which sleeve rotates on each rotation date.

    Sleeves take turns, so with `tranches=4` each sleeve rotates every fourth
    rotation date and therefore holds for `4 * every_weeks` weeks. The
    assignment is positional rather than date-derived so it stays stable when
    the panel is extended by a day.
    """
    return {pd.Timestamp(d): i % max(1, tranches) for i, d in enumerate(dates)}


def current_sleeves(dates: Sequence[pd.Timestamp], tranches: int,
                    as_of: Optional[pd.Timestamp] = None
                    ) -> Dict[int, pd.Timestamp]:
    """
    For each sleeve, the rotation date whose book it is currently holding.

    Returns {sleeve index: date}. A sleeve missing from the mapping has not
    rotated yet — which happens only at the very start of a record.
    """
    dates = [pd.Timestamp(d) for d in dates]
    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        dates = [d for d in dates if d <= as_of]
    assign = sleeve_assignment(dates, tranches)

    latest: Dict[int, pd.Timestamp] = {}
    for d in dates:
        latest[assign[d]] = d
    return latest


def next_rotation(dates: Sequence[pd.Timestamp], tranches: int,
                  sessions: pd.DatetimeIndex,
                  weekday: int = TUESDAY, every_weeks: int = 2,
                  as_of: Optional[pd.Timestamp] = None):
    """
    The next rotation date and which sleeve it belongs to.

    Projected on the calendar when it lies beyond the panel, which is the normal
    case: the panel ends at the last close and the next rotation is in the
    future. Returned as (date, sleeve, is_projected) so a caller can say so
    rather than implying the date is known to be a trading session.
    """
    dates = [pd.Timestamp(d) for d in dates]
    as_of = pd.Timestamp(as_of) if as_of is not None else (
        pd.DatetimeIndex(sessions)[-1] if len(sessions) else None)
    past = [d for d in dates if d <= as_of] if as_of is not None else dates
    if not past:
        return (None, None, False)

    future = [d for d in dates if as_of is not None and d > as_of]
    if future:
        nxt = future[0]
        return (nxt, sleeve_assignment(dates, tranches)[nxt], False)

    # Beyond the panel: project the calendar forward from the last known one.
    last = past[-1]
    target = last + pd.Timedelta(weeks=every_weeks)
    target = target - pd.Timedelta(days=(target.weekday() - weekday) % 7)
    if target <= last:
        target += pd.Timedelta(weeks=every_weeks)
    return (target, (sleeve_assignment(dates, tranches)[last] + 1)
            % max(1, tranches), True)
