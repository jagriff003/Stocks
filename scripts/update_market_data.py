"""
Keep the data the hedge layer decides on up to date.

Two stores, updated differently on purpose (the reasons are in the module
docstrings of `momentum/marketstore.py` and `momentum/longhistory.py`):

  DAILY   data/market/daily_returns.csv — append-only daily total returns for
          every hedge instrument, SPY as the session calendar.  This is what a
          live decision reads.  Run every session after 16:15 ET.
  LONG    data/longhistory/ — dated vintages of every raw research download
          (Ken French, World Bank Pink Sheet, FRED, Nareit) and the built
          monthly panel.  Research only; monthly is plenty.

Exit status follows RUNBOOK: 0 clean; 2 written but something needs a look
(a refused revision, a stale or gapped symbol, a big move, a revised long
history); anything else is an outright failure.

Run:  python scripts/update_market_data.py                  # daily store
      python scripts/update_market_data.py --long           # + long-history vintages
      python scripts/update_market_data.py --add GLD SIVR   # start tracking symbols
      python scripts/update_market_data.py --accept-revisions
      python scripts/update_market_data.py --verify         # full-history audit, writes nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum import marketstore as ms  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=[], help="symbols to start tracking")
    ap.add_argument("--accept-revisions", action="store_true")
    ap.add_argument("--verify", action="store_true", help="compare the whole store to a fresh fetch")
    ap.add_argument("--long", action="store_true", help="also refresh long-history vintages and panel")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)

    if args.verify:
        v = ms.verify_full_history()
        print(v.to_string())
        return 2 if (v["revised"] > 0).any() else 0

    # stored symbols are always maintained; --add only widens the set
    symbols = sorted(set(ms.DEFAULT_SYMBOLS) | set(args.add)) if args.add else None
    rep, status = ms.update_daily(symbols, accept_revisions=args.accept_revisions)
    stored = ms.load_daily_returns()
    print(f"DAILY STORE  {ms.STORE_DIR / ms.RETURNS_FILE}")
    print(f"  {stored.shape[1]} symbols, last session {stored.index[-1]:%Y-%m-%d}")
    print(rep[["appended", "overlap_days", "max_overlap_diff", "status", "detail"]].to_string())

    if args.long:
        from momentum.longhistory import load_long_history, save_panel
        panel = load_long_history(max_age_days=0, verbose=True)
        changes = save_panel(panel)
        print(f"\nLONG HISTORY  panel {panel.index[0]:%Y-%m} .. {panel.index[-1]:%Y-%m}")
        print(changes.to_string())
        if (changes["revised > 1bp"] > 0).any():
            print("  revisions found — expected for the French series after its annual "
                  "CRSP update; look twice at anything else")
            status = max(status, 2)

    if status:
        print("\nNEEDS A LOOK: see the status column above.")
    return status


if __name__ == "__main__":
    sys.exit(main())
