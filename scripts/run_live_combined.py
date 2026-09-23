"""
The combined report: Track J's book, Track K's reading, and one allocation.

TODO 0l, option (a), agreed 2026-09-23.  Track J is the stock model; Track K
is the preparation model for an inflation or debasement regime that is not in
Track J's record.  They stay separate programs — this report reads Track J's
books from `live/trackj_book.json` (written by `run_live_trackj.py`) and
computes Track K from the daily store — and puts them side by side.

TWO RECOMMENDATIONS, ALWAYS (James, 2026-09-23)

  AT THE LAST ROTATION  what the book should be now: Track J's sleeves as they
                        were last rotated, Track K as it read on that Tuesday.
  CURRENT               what both models say on the latest close, as if today
                        were a rotation date.  Off-cycle this is information,
                        not a trade — trading it is a different, untested
                        strategy for either model.

On a rotation Tuesday the two coincide.

HOW THE ALLOCATION IS FORMED

While Track K is quiet: 100% Track J.  When its trigger fires, each real asset
that beats SPY by 10% over three months (and beats cash) takes a 25% slot, up
to 75%, and Track J's sleeves shrink pro-rata.  Track K evaluates on Track J's
rotation Tuesdays.

THE DECISION LOG

Every run appends a row to `data/decisions/decision_log.csv` (tracked in git):
what each model recommended, and — via --action / --note, or filled in later
by hand — what was actually done and why.  Discretion is the point of option
(a); the log is what lets it be measured instead of remembered.

Exit status follows RUNBOOK: 0 clean; 2 printed but something is stale or
flagged; anything else is a failure.

Run:  python scripts/run_live_trackj.py --no-plots      # first, after 16:15 ET
      python scripts/run_live_combined.py
      python scripts/run_live_combined.py --action "followed" --note "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum import marketstore as ms  # noqa: E402
from momentum.schedule import TUESDAY, rotation_dates  # noqa: E402
from momentum.trackk import LIVE_CONFIG, combine, recommendations  # noqa: E402

BOOK_FILE = REPO_ROOT / "live" / "trackj_book.json"
LOG_FILE = REPO_ROOT / "data" / "decisions" / "decision_log.csv"
LOG_COLUMNS = ["run_at", "signal_session", "last_rotation", "trackk_firing_at_rotation",
               "hedge_share_at_rotation", "hedge_at_rotation", "trackk_firing_now",
               "hedge_share_now", "hedge_now", "trackj_bought_at_rotation",
               "trackj_next_rotation", "trackj_would_buy_now", "action", "note"]


def show_trigger(label, rec):
    t = rec["trigger"]
    state = "FIRING" if t["firing"] else "quiet"
    print(f"  {label}: {t['date']:%Y-%m-%d}  Track K is {state}")
    print(f"    trigger assets beating SPY by 10% and cash: {t['assets_qualifying']} of 4 "
          f"(needs {t['needed']});  stock-bond correlation (1y) {t['stock_bond_corr']:+.2f} "
          f"(needs > 0);  SPY 3m {t['market_3m']:+.1%}")
    d = rec["diagnostics"]
    tbl = pd.DataFrame({"live": d["live"], "3m": d["3m return"].map("{:+.1%}".format),
                        "vs SPY": d["vs SPY"].map("{:+.1%}".format),
                        "qualifies": (d["beats SPY by margin"] & d["beats cash"]).map({True: "yes", False: ""}),
                        "trigger asset": d["trigger asset"].map({True: "*", False: ""})})
    print("    " + tbl.to_string().replace("\n", "\n    "))


def show_allocation(label, alloc, account):
    print(f"\n  {label}")
    print(f"    {'symbol':<8}{'Track J':>9}{'Track K':>9}{'weight':>9}{'$':>12}")
    print("    " + "-" * 47)
    for s, r in alloc.iterrows():
        print(f"    {s:<8}{r['Track J']:>9.1%}{r['Track K']:>9.1%}{r['weight']:>9.1%}"
              f"{account * r['weight']:>12,.0f}")
    print(f"    {'total':<8}{alloc['Track J'].sum():>9.1%}{alloc['Track K'].sum():>9.1%}"
          f"{alloc['weight'].sum():>9.1%}{account * alloc['weight'].sum():>12,.0f}")


def append_log(row: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])[LOG_COLUMNS]
    if LOG_FILE.exists():
        new = pd.concat([pd.read_csv(LOG_FILE, dtype=str), new.astype(str)], ignore_index=True)
    new.to_csv(LOG_FILE, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", type=float, default=100_000.0)
    ap.add_argument("--no-update", action="store_true", help="skip refreshing the daily store")
    ap.add_argument("--action", default="", help="what you did with this recommendation")
    ap.add_argument("--note", default="", help="why — the news, the pundits, the gut")
    ap.add_argument("--no-log", action="store_true", help="do not append to the decision log")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    status = 0

    print("=" * 100)
    print(f"COMBINED REPORT — Track J (stocks) + Track K (real-asset preparation)   "
          f"{datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 100)

    if not args.no_update:
        rep, st = ms.update_daily()
        flagged = rep[rep["status"] != "ok"]
        print(f"Daily store updated: last session {ms.load_daily_returns().index[-1]:%Y-%m-%d}"
              + (f"; {len(flagged)} symbol(s) flagged — see update_market_data.py" if len(flagged) else ""))
        if st:
            print(flagged[["status", "detail"]].to_string())
            status = 2
    store = ms.load_daily_returns()
    latest = store[ms.REFERENCE].dropna().index[-1]

    if not BOOK_FILE.exists():
        print(f"\n*** {BOOK_FILE.relative_to(REPO_ROOT)} missing — run "
              f"`python scripts/run_live_trackj.py --no-plots` first. ***")
        return 1
    book = json.loads(BOOK_FILE.read_text())
    tj_session = pd.Timestamp(book["signal_session"])
    print(f"Track J books: signal session {tj_session:%Y-%m-%d} (written {book['written']})")
    if tj_session != latest:
        print(f"*** Track J's books are from {tj_session:%Y-%m-%d} but the store ends "
              f"{latest:%Y-%m-%d}. Re-run run_live_trackj.py before acting. ***")
        status = 2

    rot = rotation_dates(store[ms.REFERENCE].dropna().index, weekday=TUESDAY,
                         every_weeks=book["every_weeks"], anchor=pd.Timestamp(book["anchor"]))
    k = recommendations(store, rot, LIVE_CONFIG)
    last_rot = pd.Timestamp(book["last_rotation"]["date"])
    if k["last_rotation"]["date"] != last_rot:
        print(f"*** rotation calendars disagree: Track J {last_rot:%Y-%m-%d}, "
              f"Track K {k['last_rotation']['date']:%Y-%m-%d} ***")
        status = 2
    on_cycle = latest == last_rot

    print("\n" + "=" * 100)
    print("TRACK K — the preparation model")
    print("=" * 100)
    show_trigger("AT THE LAST ROTATION", k["last_rotation"])
    if not on_cycle:
        print()
        show_trigger("CURRENT (off-cycle, information only)", k["current"])

    print("\n" + "=" * 100)
    print("TRACK J — the stock model")
    print("=" * 100)
    lr, cur = book["last_rotation"], book["current"]
    print(f"  AT THE LAST ROTATION {lr['date']}: sleeve {lr['sleeve']} bought {', '.join(lr['bought'])}")
    print(f"  CURRENT: next rotation {cur['next_rotation']} (sleeve {cur['next_sleeve']}"
          f"{', projected' if cur['projected'] else ''}); on today's close it would buy "
          f"{', '.join(cur['would_buy'])}")

    print("\n" + "=" * 100)
    print("RECOMMENDED ALLOCATION")
    print("=" * 100)
    a_last = combine(lr["weights"], k["last_rotation"]["hedge_share"], k["last_rotation"]["weights"])
    show_allocation(f"AT THE LAST ROTATION ({lr['date']}) — what to hold now", a_last, args.account)
    for a in (a_last,):
        assert abs(a["weight"].sum() - 1.0) < 1e-9, a["weight"].sum()
    if not on_cycle:
        a_now = combine(cur["weights"], k["current"]["hedge_share"], k["current"]["weights"])
        assert abs(a_now["weight"].sum() - 1.0) < 1e-9, a_now["weight"].sum()
        show_allocation(f"CURRENT ({latest:%Y-%m-%d}) — if today were a rotation "
                        f"(information, not a trade)", a_now, args.account)
        d = a_now["weight"].sub(a_last["weight"], fill_value=0.0)
        d = d[d.abs() > 1e-9].sort_values()
        if len(d):
            print("\n    difference, current minus last rotation: " +
                  ", ".join(f"{s} {v:+.1%}" for s, v in d.items()))
    else:
        print("\n  Today IS a rotation date: the two recommendations coincide.")

    if not args.no_log:
        kl, kc = k["last_rotation"], k["current"]
        append_log({
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "signal_session": f"{latest:%Y-%m-%d}", "last_rotation": lr["date"],
            "trackk_firing_at_rotation": kl["trigger"]["firing"],
            "hedge_share_at_rotation": round(kl["hedge_share"], 4),
            "hedge_at_rotation": " ".join(f"{s}:{w:.2f}" for s, w in kl["weights"].items()),
            "trackk_firing_now": kc["trigger"]["firing"],
            "hedge_share_now": round(kc["hedge_share"], 4),
            "hedge_now": " ".join(f"{s}:{w:.2f}" for s, w in kc["weights"].items()),
            "trackj_bought_at_rotation": " ".join(lr["bought"]),
            "trackj_next_rotation": cur["next_rotation"],
            "trackj_would_buy_now": " ".join(cur["would_buy"]),
            "action": args.action, "note": args.note})
        print(f"\n  Decision log: row appended to {LOG_FILE.relative_to(REPO_ROOT)}"
              + ("" if args.action else " (action/note blank — fill in what you did)"))

    print("\n  Track K is discretionary input. Obeyed mechanically on Track J 2012-2026 it")
    print("  cost 0.3-0.7pp a year and fired 3-4% of the time; its value is a regime that")
    print("  is not in that record. See FINDINGS, Track K.")
    return status


if __name__ == "__main__":
    sys.exit(main())
