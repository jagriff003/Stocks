"""
Create the rebalance workbook (once).

The workbook turns the combined report's recommendation into share orders for
the two IRAs; `momentum/rebalance_book.py` documents how.  It holds account data,
so it is written to `private/` (git-ignored) and this script refuses to
overwrite an existing one without --force.

Run:  python scripts/build_rebalance_workbook.py
      python scripts/build_rebalance_workbook.py --force     # start over: loses what you typed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.rebalance_book import DEFAULT_PATH, build_workbook  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(DEFAULT_PATH))
    ap.add_argument("--force", action="store_true", help="overwrite an existing workbook")
    args = ap.parse_args()
    try:
        path = build_workbook(Path(args.path), overwrite=args.force)
    except FileExistsError as exc:
        print(exc)
        return 1
    print(f"Created {path}\nNext: run scripts/run_live_combined.py to fill its Target sheet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
