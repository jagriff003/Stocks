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

Every run appends a row to `data/decisions/decision_log.csv` (personal, git-ignored):
what each model recommended, and — via --action / --note here, or afterwards
with `scripts/record_decision.py` without re-running the models — what was
actually done and why.  Discretion is the point of option
(a); the log is what lets it be measured instead of remembered.

CHARTS (default: shown AND saved to charts/)

  YYYY-MM-DD_combined_trigger.png     each trigger asset's 3-month excess over
                                      SPY against the +10% line, the stock-bond
                                      correlation against zero, firing shaded
  YYYY-MM-DD_combined_allocation.png  the recommended allocation(s), with Track
                                      J's energy & materials split out because
                                      they overlap Track K's bet

Track J's own charts (performance, held book, context) are opened alongside
when `run_live_trackj.py --save-charts charts` has saved them for the same
session — the saved images, not a recomputation, so what you see is exactly
what Track J produced.  They are not re-saved.

`--no-show` saves without opening windows (for a scheduled run, which would
otherwise block on plt.show()); `--no-plots` skips charts entirely.

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
from momentum.hedge import regime_open, trailing_return  # noqa: E402
from momentum.trackk import (LIVE_CONFIG, LIVE_SYMBOL, assets_from_store,  # noqa: E402
                             combine, recommendations)

BOOK_FILE = REPO_ROOT / "live" / "trackj_book.json"
LOG_FILE = REPO_ROOT / "data" / "decisions" / "decision_log.csv"
LOG_COLUMNS = ["run_at", "signal_session", "on_cycle", "last_rotation", "trackk_firing_at_rotation",
               "trackk_streak_at_rotation", "hedge_share_at_rotation", "hedge_at_rotation", "trackk_firing_now",
               "hedge_share_now", "hedge_now", "trackj_bought_at_rotation",
               "trackj_next_rotation", "trackj_would_buy_now", "action", "note"]


def show_trigger(label, rec):
    t = rec["trigger"]
    state = "FIRING" if t["firing"] else "quiet"
    streak = ""
    if t["firing"]:
        streak = (f"  — {t['streak']} consecutive firing decision(s)"
                  + ("; FIRST firing: requiring two (the flicker study) would wait for the next rotation"
                     if t["streak"] == 1 else ""))
    print(f"  {label}: {t['date']:%Y-%m-%d}  Track K is {state}{streak}")
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


# --- charts -------------------------------------------------------------------
#
# Colours are the reference palette's categorical slots, used in their
# validated order so adjacent stacked segments are pairs that were checked;
# every segment and every line end also carries a text label, so identity
# never rests on colour alone.  Colour follows the entity: gold is the same
# blue in both charts.  Text stays in ink, never in a series colour.
SURFACE, INK, INK2, GRID, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"
ENTITY = {
    "GOLD": ("gold", "#2a78d6"), "SILVER": ("silver", "#eb6834"),
    "CMDTY": ("commodities", "#1baf7a"), "ENERGY": ("energy", "#eda100"),
    "K_OTHER": ("Track K other", "#e87ba4"),
    "J_REAL": ("Track J energy & materials", "#008300"),
    "J_OTHER": ("Track J other stocks", "#4a3aa7"),
}
ROLE_OF_LIVE = {v: k for k, v in LIVE_SYMBOL.items()}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _shade(ax, fire):
    on = fire.astype(int).diff().fillna(fire.iloc[0]).ne(0).cumsum()[fire]
    for _, span in fire[fire].groupby(on):
        ax.axvspan(span.index[0], span.index[-1], color=SHADE, zorder=0, linewidth=0)


def chart_trigger(store, last_rot, path, days=252):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    cfg = LIVE_CONFIG
    market = store[ms.REFERENCE].dropna()
    assets = assets_from_store(store)
    tr = trailing_return(assets, cfg.lookback)
    tm = trailing_return(market.to_frame("m"), cfg.lookback)["m"]
    excess = tr[list(cfg.regime_assets)].sub(tm, axis=0).iloc[-days:]
    fire = pd.Series(regime_open(market, assets, cfg), index=market.index).iloc[-days:]
    corr = market.rolling(cfg.corr_window, min_periods=cfg.corr_window).corr(
        assets[cfg.regime_corr_asset]).iloc[-days:]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (a1, a2):
        _style(ax)
        _shade(ax, fire)
        ax.axvline(last_rot, color=INK2, linewidth=1, linestyle=":")
    ends = []
    for role in cfg.regime_assets:
        name, color = ENTITY[role]
        s = excess[role]
        a1.plot(s.index, s.values, color=color, linewidth=2, label=name)
        a1.plot(s.index[-1], s.iloc[-1], "o", color=color, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ends.append([s.iloc[-1], f"{name} {s.iloc[-1]:+.0%}", color])
    a1.axhline(cfg.enter_margin, color=INK2, linewidth=1.2, linestyle="--")
    a1.annotate(f"+{cfg.enter_margin:.0%} margin", (excess.index[0], cfg.enter_margin),
                xytext=(2, 5), textcoords="offset points", fontsize=8.5, color=INK2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    # labels at the right edge, spread so none overlap; a thin leader ties each
    # label to its line end
    lo, hi = a1.get_ylim()
    gap = (hi - lo) * 0.055
    ends.sort(key=lambda e: e[0])
    placed = []
    for v, _, _ in ends:
        placed.append(max(v, placed[-1] + gap) if placed else v)
    x_end = excess.index[-1]
    x_lab = x_end + pd.Timedelta(days=12)
    for (v, text, color), y in zip(ends, placed):
        a1.plot([x_end, x_lab], [v, y], color=color, linewidth=0.8)
        a1.annotate(text, (x_lab, y), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    a1.axhline(0, color=INK2, linewidth=0.8)
    a1.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=0))
    a1.set_title("Track K trigger: 3-month return over SPY, the four trigger assets\n"
                 "Fires when 2+ clear the dashed line (and beat cash) while the stock-bond "
                 "correlation is above 0.\nShaded = firing.  Dotted = last rotation.",
                 loc="left", fontsize=10, color=INK)
    a1.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=4, labelcolor=INK2)

    a2.plot(corr.index, corr.values, color=INK2, linewidth=2)
    a2.axhline(0, color=INK2, linewidth=1.2, linestyle="--")
    a2.annotate(f"{corr.iloc[-1]:+.2f}", (corr.index[-1], corr.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK)
    a2.set_title("Stock-bond correlation, 1 year (SPY vs 10-year Treasuries) - needs to be above 0",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.subplots_adjust(right=0.84)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    return fig


def _buckets(alloc, sectors):
    out = {k: 0.0 for k in ENTITY}
    for sym, r in alloc.iterrows():
        if r["Track K"] > 0:
            role = ROLE_OF_LIVE.get(sym)
            out[role if role in ("GOLD", "SILVER", "CMDTY", "ENERGY") else "K_OTHER"] += r["Track K"]
        if r["Track J"] > 0:
            real = sectors.get(sym) in ("Energy", "Basic Materials")
            out["J_REAL" if real else "J_OTHER"] += r["Track J"]
    return out


def chart_allocation(rows, sectors, path):
    """rows: list of (label, allocation frame), drawn top to bottom."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 1.6 + 1.1 * len(rows)))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    order = list(ENTITY)
    drawn = set()
    for i, (label, alloc) in enumerate(rows):
        b = _buckets(alloc, sectors)
        left = 0.0
        y = len(rows) - 1 - i
        for key in order:
            w = b[key]
            if w <= 1e-9:
                continue
            name, color = ENTITY[key]
            drawn.add(key)
            ax.barh(y, w, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=2)
            if w >= 0.06:
                dark = key in ("GOLD", "J_REAL", "J_OTHER", "SILVER")
                ax.text(left + w / 2, y, f"{name}\n{w:.0%}", ha="center", va="center",
                        fontsize=8.5, color="#ffffff" if dark else INK)
            left += w
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.5, color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["x"]).PercentFormatter(1.0))
    keys = [k for k in order if k in drawn]
    handles = [plt.Rectangle((0, 0), 1, 1, color=ENTITY[k][1]) for k in keys]
    ax.legend(handles, [ENTITY[k][0] for k in keys], loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=len(keys), frameon=False, fontsize=8.5,
              labelcolor=INK2)
    ax.set_title("Recommended allocation - Track J's energy & materials shown apart, because "
                 "they overlap Track K's bet", loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    return fig


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
    ap.add_argument("--charts-dir", default="charts", help="where charts are saved (default charts/)")
    ap.add_argument("--no-show", action="store_true", help="save charts without opening windows")
    ap.add_argument("--no-plots", action="store_true", help="no charts at all")
    args = ap.parse_args()
    if args.no_show and not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
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

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
            outdir = REPO_ROOT / args.charts_dir
            outdir.mkdir(parents=True, exist_ok=True)
            stamp = f"{latest:%Y-%m-%d}"
            sectors = pd.read_csv(REPO_ROOT / "random_pool.csv").set_index("symbol")["sector"]
            rows = [(f"At last rotation\n{lr['date']}", a_last)]
            if not on_cycle:
                rows.append((f"Current\n{stamp} (information)", a_now))
            chart_trigger(store, last_rot, outdir / f"{stamp}_combined_trigger.png")
            chart_allocation(rows, sectors, outdir / f"{stamp}_combined_allocation.png")
            print(f"\n  Charts saved to {args.charts_dir}/: {stamp}_combined_trigger.png, "
                  f"{stamp}_combined_allocation.png")
            if not args.no_show:
                jpngs = sorted(outdir.glob(f"{tj_session:%Y-%m-%d}_trackj_*.png"))
                for png in jpngs:
                    img = plt.imread(png)
                    h, w = img.shape[:2]
                    f = plt.figure(figsize=(w / 110, h / 110))
                    f.canvas.manager.set_window_title(png.name) if f.canvas.manager else None
                    ax = f.add_axes([0, 0, 1, 1])
                    ax.imshow(img)
                    ax.axis("off")
                print(f"  Track J charts shown alongside: {len(jpngs)}"
                      + ("" if jpngs else " (none saved for this session - run "
                                           "run_live_trackj.py --save-charts charts)"))
                plt.show()
            plt.close("all")
        except Exception as exc:
            print(f"\n  (charts skipped: {type(exc).__name__}: {exc})")
            status = max(status, 2)

    if not args.no_log:
        kl, kc = k["last_rotation"], k["current"]
        append_log({
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "signal_session": f"{latest:%Y-%m-%d}", "on_cycle": on_cycle, "last_rotation": lr["date"],
            "trackk_firing_at_rotation": kl["trigger"]["firing"],
            "trackk_streak_at_rotation": kl["trigger"]["streak"],
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
              + ("" if args.action else
                 " (action blank — record it later with\n    python scripts/record_decision.py "
                 "--action \"...\" --note \"...\")"))

    print("\n  Track K is discretionary input. Obeyed mechanically on Track J 2012-2026 it")
    print("  cost 0.3-0.7pp a year and fired 3-4% of the time; its value is a regime that")
    print("  is not in that record. See FINDINGS, Track K.")
    return status


if __name__ == "__main__":
    sys.exit(main())
