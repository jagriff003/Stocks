"""
Reconcile an actual trade history against the modelled book.

TODO item 2.  The production backtest reports +18.04% YTD, far above the account's realised result.  `analyze_ytd_gap.py` established that the universe change
does not explain it — it works the other way — and that the backtest's whole
year comes from five names.  This takes the other side: what was ACTUALLY held,
and where it diverged from what the model said to hold.

WHAT IT CAN AND CANNOT SETTLE

It can say, per rebalance date, which modelled names were held and which were
not, and what the missing ones were worth.  That is the difference between "the
model is wrong" and "the model was not followed", which is the single most
important fork in the road and has never been measured here.

It cannot compute the account's true return: a transaction export has no
starting balance and need not cover every account.  So it reports
the *traded* book's behaviour and the overlap, not a headline return.  Anything
claiming to be the account's P&L from this data alone would be made up.

NOT ALL DIVERGENCE IS EXECUTION

The model itself changed during the period — the scoring defects fixed in 2026
mean the book the code recommended in the spring is not the book today's code
reproduces for the spring.  Divergence before those fixes is a model-version
artifact, not a trading error, and the report segments by month so the two are
separable by eye rather than conflated.

PRIVACY

Reads the transaction file from wherever it is given and writes its output to
the scratchpad by default.  Account data does not belong in the repo, and the
repo's .gitignore would silently swallow a stray CSV rather than flag it.

Run:  python scripts/reconcile_trades.py --trades path/to/transactions.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.data import load_data
from momentum.experiments import production_config
from momentum.strategy import run_strategy
from momentum.universe import current_symbols


def load_trades(path: Path) -> pd.DataFrame:
    """Schwab-style transaction export -> tidy buys and sells."""
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    d = d[d["Action"].isin(["Buy", "Sell"])].copy()

    # "09/16/2026 as of 09/15/2026" -> the settlement-style prefix is the one
    # that matches a trading session.
    d["Date"] = pd.to_datetime(d["Date"].str.split(" as of ").str[0],
                               format="%m/%d/%Y", errors="coerce")
    for col in ("Quantity", "Price"):
        d[col] = pd.to_numeric(
            d[col].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce")
    d["Signed"] = np.where(d["Action"] == "Buy", d["Quantity"], -d["Quantity"])
    return d.dropna(subset=["Date", "Symbol", "Quantity"]).sort_values("Date")


def positions_over_time(trades: pd.DataFrame, index: pd.DatetimeIndex):
    """Daily share count per symbol, forward-filled between trades."""
    flows = trades.pivot_table(index="Date", columns="Symbol", values="Signed",
                               aggfunc="sum").fillna(0.0)
    flows = flows.reindex(index.union(flows.index)).fillna(0.0).sort_index()
    return flows.cumsum().reindex(index).ffill().fillna(0.0)


def main() -> int:
    p = argparse.ArgumentParser(description="Reconcile trades vs the model")
    p.add_argument("--trades", required=True)
    p.add_argument("--from", dest="start_ytd", default="2026-01-01")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--out", default=None,
                   help="where to write the detail CSV (default: scratchpad)")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    trades = load_trades(Path(args.trades))
    print("=" * 96)
    print("TRADE RECONCILIATION — what was held against what was modelled")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 96)
    print(f"\n  {len(trades)} trades, {trades['Symbol'].nunique()} symbols, "
          f"{trades['Date'].min():%Y-%m-%d} to {trades['Date'].max():%Y-%m-%d}")

    cfg = production_config()
    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache, cache_max_age_hours=1e9,
                       verbose=False)
    result = run_strategy(prices, cfg)

    ytd = pd.Timestamp(args.start_ytd)
    idx = prices.close.index[prices.close.index >= ytd]
    pos = positions_over_time(trades, idx)
    held_any = [c for c in pos.columns if (pos[c] != 0).any()]

    # --- which modelled names were actually held, per rebalance ---
    print("\n" + "=" * 96)
    print("OVERLAP — modelled book vs actual holdings, at each rotation")
    print("=" * 96)
    rows = []
    for rec in result.rebalance_history:
        d = pd.Timestamp(rec["Date"])
        if d < ytd or d not in pos.index:
            continue
        modelled = [s for s in rec["Selected_Stocks"]]
        actual = [c for c in pos.columns if pos.at[d, c] > 0]
        hit = [s for s in modelled if s in actual]
        rows.append({"Date": d, "Modelled": modelled, "Actual": actual,
                     "Overlap": len(hit), "N": len(modelled),
                     "Missing": [s for s in modelled if s not in actual]})

    if not rows:
        print("  no rebalances inside the window")
        return 1

    hdr = f"    {'date':<12}{'overlap':>9}   {'modelled':<34}{'missing'}"
    print(hdr)
    print("    " + "-" * 92)
    for r in rows:
        print(f"    {r['Date']:%Y-%m-%d}{r['Overlap']}/{r['N']:>7}   "
              f"{', '.join(r['Modelled']):<34}{', '.join(r['Missing'])}")

    mean_overlap = np.mean([r["Overlap"] / r["N"] for r in rows])
    print(f"\n    Mean overlap: {mean_overlap:.0%} of the modelled book was "
          f"actually held.")

    # --- what the never-held modelled names were worth ---
    print("\n" + "=" * 96)
    print("WHAT WAS MISSED — modelled names never held, and their contribution")
    print("=" * 96)
    rets = prices.close.pct_change()
    contrib = {}
    for d, names in result.holdings.items():
        if d < ytd or not len(names) or d not in rets.index:
            continue
        w = 1.0 / len(names)
        for n in names:
            if n in rets.columns and pd.notna(rets.at[d, n]):
                contrib[n] = contrib.get(n, 0.0) + w * float(rets.at[d, n])
    contrib = pd.Series(contrib, dtype=float).sort_values(ascending=False)

    never = [n for n in contrib.index if n not in held_any]
    print(f"    {'name':<8}{'model contrib':>15}   status")
    print("    " + "-" * 52)
    for n in contrib.index:
        status = "never held" if n in never else "held at some point"
        flag = "  <---" if n in never and abs(contrib[n]) > 0.01 else ""
        print(f"    {n:<8}{contrib[n]:>+15.2%}   {status}{flag}")

    missed = float(contrib.reindex(never).dropna().sum())
    print(f"\n    Modelled contribution from names NEVER held: {missed:+.2%}")
    print(f"    Total modelled contribution:                  "
          f"{float(contrib.sum()):+.2%}")

    # --- divergence by month, so model-version drift is separable ---
    print("\n" + "=" * 96)
    print("OVERLAP BY MONTH")
    print("=" * 96)
    print("  Divergence early in the year may be model-version drift rather")
    print("  than execution: the scoring defects fixed during 2026 mean today's")
    print("  code does not reproduce the book the old code recommended then.")
    bym = {}
    for r in rows:
        k = r["Date"].strftime("%Y-%m")
        bym.setdefault(k, []).append(r["Overlap"] / r["N"])
    for k in sorted(bym):
        v = np.mean(bym[k])
        print(f"    {k}   {v:>5.0%}   {'#' * int(round(v * 20))}")

    out = Path(args.out) if args.out else (
        Path(r"C:\Users\USER\AppData\Local\Temp\claude"
             r"\c--Users-USER-OneDrive-Analytics-Stocks"
             r"\07cca480-f9e9-48bc-b70b-944d5c94900b\scratchpad")
        / "trade_reconciliation.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**r, "Modelled": " ".join(r["Modelled"]),
                   "Actual": " ".join(r["Actual"]),
                   "Missing": " ".join(r["Missing"])} for r in rows]
                 ).to_csv(out, index=False)

    print("\n" + "=" * 96)
    print("WHAT THIS DOES NOT SAY")
    print("=" * 96)
    print("  It does not compute the account's return: a transaction export has")
    print("  no starting balance and may not cover every account. Overlap")
    print("  and missed contribution are the measurable parts; a headline P&L")
    print("  from this data alone would be invented.")
    print(f"\n  Detail written to {out}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
