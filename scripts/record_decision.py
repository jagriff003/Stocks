"""
Record what you actually did with a recommendation, without re-running the models.

`run_live_combined.py` appends a row to `data/decisions/decision_log.csv` on
every run, with `action` and `note` blank unless given.  On a rotation day the
natural sequence is: run the report, decide, trade, then say what you did —
which used to mean running the whole report a second time.  This fills in the
row instead.

By default it writes to the most recent row.  `--session` targets the latest
row for a given signal session.  A row that already has an action is not
overwritten without `--overwrite`, because the log is only worth anything if
it is a record of what was decided at the time.

Run:  python scripts/record_decision.py --action "held Track J; skipped Track K" \\
          --note "commodities already crowded; pundits on oil"
      python scripts/record_decision.py --show
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = REPO_ROOT / "data" / "decisions" / "decision_log.csv"


def record(action: str, note: str, session: str = None, overwrite: bool = False,
           log_file: Path = LOG_FILE) -> pd.Series:
    if not Path(log_file).exists():
        raise FileNotFoundError(f"{log_file} does not exist - run run_live_combined.py first")
    log = pd.read_csv(log_file, dtype=str, keep_default_na=False)
    if log.empty:
        raise ValueError("the decision log has no rows")
    rows = log.index if session is None else log.index[log["signal_session"] == session]
    if len(rows) == 0:
        raise ValueError(f"no row for signal session {session}")
    i = rows[-1]
    if log.at[i, "action"] and not overwrite:
        raise ValueError(f"row {i} ({log.at[i, 'signal_session']}) already records "
                         f"'{log.at[i, 'action']}' - use --overwrite to replace it")
    log.at[i, "action"] = action
    stamp = datetime.now().isoformat(timespec="minutes")
    log.at[i, "note"] = f"{note} [recorded {stamp}]" if note else f"[recorded {stamp}]"
    log.to_csv(log_file, index=False)
    return log.loc[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--session", default=None, help="signal session YYYY-MM-DD (default: latest row)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--show", action="store_true", help="print the last rows and exit")
    args = ap.parse_args()
    if args.show:
        log = pd.read_csv(LOG_FILE, dtype=str, keep_default_na=False)
        cols = ["signal_session", "on_cycle", "trackk_firing_at_rotation", "hedge_at_rotation",
                "action", "note"]
        print(log[[c for c in cols if c in log]].tail(10).to_string())
        return 0
    if not args.action:
        ap.error("--action is required (what did you do?)")
    row = record(args.action, args.note, args.session, args.overwrite)
    print(f"Recorded for {row['signal_session']}: {row['action']}  |  {row['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
