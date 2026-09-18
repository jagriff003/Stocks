"""
Tests for the parts of run_live that only matter when nobody is watching.

Interactively, a skipped subsystem is obvious: the line is on screen and the
missing section is where you expected to read it.  Under Windows Scheduler it
is not — the log goes to a file nobody opens, and the exit code is the only
signal that reaches anyone.  These tests cover that signal.

Run:  python -m pytest tests/test_run_live_automation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_live import Degradations  # noqa: E402


def test_a_clean_run_is_falsey():
    assert not Degradations()


def test_a_skipped_component_makes_it_truthy():
    d = Degradations()
    d.note("context panel", ValueError("no data for UUP"))
    assert d


def test_the_report_names_the_component_and_the_reason():
    """
    A scheduler log that says only "something failed" costs a debugging
    session.  Both the component and the exception text must survive.
    """
    d = Degradations()
    d.note("health monitor", KeyError("episodes"))
    d.note("charts", RuntimeError("no display"))
    assert len(d.items) == 2
    assert d.items[0][0] == "health monitor"
    assert "KeyError" in d.items[0][1]
    assert "episodes" in d.items[0][1]
    assert d.items[1][0] == "charts"
    assert "no display" in d.items[1][1]


def test_report_on_a_clean_run_prints_nothing(capsys):
    Degradations().report()
    assert capsys.readouterr().out == ""


def test_report_is_loud_when_something_skipped(capsys):
    d = Degradations()
    d.note("context panel", ValueError("boom"))
    d.report()
    out = capsys.readouterr().out
    assert "INCOMPLETE RUN" in out
    assert "context panel" in out
    # The book is still usable; the report must not imply otherwise.
    assert "still valid" in out


def test_exception_type_is_recorded_not_just_the_message():
    """
    A bare message loses the distinction between a missing file and a bad
    value, which is usually the first thing you want to know.
    """
    d = Degradations()
    d.note("context panel", FileNotFoundError("sector_map.csv"))
    assert d.items[0][1].startswith("FileNotFoundError:")


@pytest.mark.parametrize("flag", ["--no-plots", "--save-charts"])
def test_the_headless_flags_exist(flag):
    """
    Guards against the flag being renamed out from under a scheduled task.

    `--no-plots` predates this work; `--save-charts` was added because
    `plt.show()` blocks forever with no display and `--no-plots` throws the
    charts away, so neither alone lets a scheduled run produce pictures.
    """
    source = (REPO_ROOT / "scripts" / "run_live.py").read_text(encoding="utf-8")
    assert f'"{flag}"' in source
