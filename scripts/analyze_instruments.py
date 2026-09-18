"""
Testing limited tradables and synthetic monitors for inclusion.

Follows the SH result: enforcing the monitor role was worth +2.73pp, and the
decomposition showed that keeping a name for SCORING while forbidding it as a
HOLDING is better than either alternative.  That makes two questions separable
and worth asking separately of every candidate:

    should we TRADE it?   -> does it improve CAGR / drawdown when holdable
    should we WATCH it?   -> does it improve anything through the ranking alone

The second is nearly free and the bar is low.  The first is where the evidence
has to be strong, because every tradable name is a position that can be wrong.

THE CANDIDATES

  UUP       dollar index.  Correlation to SPY -0.18.  The universe is 48 US
            equities and cannot see the dollar; tested BOTH tradable and
            monitor-only, since the answer may differ.
  DBC       broad commodities.  Held previously and dropped - too volatile,
            never improved CAGR on a two-week hold.  Tested as a monitor only.
  IEF       7-10y treasuries, as a REPLACEMENT for TLT in the defensive sleeve.
            TLT is the hardest-working defensive name (-0.62pp and 1.7pp of
            drawdown to remove it) but carries 14.9% vol and a -48% drawdown.
            IEF is the same signal at roughly half the risk.
  HYG_LQD   credit appetite with duration cancelled out.  HYG stops being
            tradable on its own; the ratio becomes the monitor.
  RSP_SPY   equal weight against cap weight: leadership breadth.  Correlation
            to SPY -0.01, the most orthogonal series tested.
  IWM_SPY   small against large: risk appetite.
  AD_LINE   advance/decline over the liquid pool, computed rather than bought -
            Yahoo serves no breadth index (^ADVN, ^DECN, $ADD, ^TRIN, ^NYAD all
            404).

THE RISK OF GOING OVERBOARD, MEASURED
   Monitors are not quite free.  Scores are normalized cross-sectionally, so
   every added series shifts every other name's score slightly - the SH
   decomposition put that at -0.18pp for one name.  Six monitors could compound,
   so the all-monitors variant is run specifically to measure it rather than to
   be adopted.

Every threshold is a flag.

Run:  python scripts/analyze_instruments.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.data import PriceData
from momentum.experiments import production_config
from momentum.metrics import calculate_performance_metrics
from momentum.strategy import run_strategy
from momentum.synthetic import BreadthSpec, RatioSpec, augment, describe
from momentum.universe import current_symbols

OUT = "rsi_ma_instruments.csv"
LEGS = ["UUP", "DBC", "LQD", "IEF", "RSP", "IWM", "SPY", "HYG"]


def load(start, pool_path):
    import yfinance as yf
    warnings.filterwarnings("ignore")
    pool = pd.read_csv(pool_path)["symbol"].tolist()
    syms = sorted(set(current_symbols()) | set(LEGS) | set(pool))
    print(f"Downloading {len(syms)} symbols "
          f"(~{len(syms)//400 + 1} batched requests)")
    fc, fo = [], []
    for i in range(0, len(syms), 400):
        d = yf.download(syms[i:i + 400], start=start, interval="1d",
                        auto_adjust=True, progress=False, threads=True,
                        group_by="column")
        fc.append(d["Close"]); fo.append(d["Open"])
    close = pd.concat(fc, axis=1); open_ = pd.concat(fo, axis=1)
    aux = yf.download(["SPY", "^VIX"], start=start, interval="1d",
                      auto_adjust=True, progress=False,
                      group_by="column")["Close"]
    return close, open_, aux["SPY"], aux["^VIX"], [s for s in pool if s in close.columns]


def main() -> int:
    p = argparse.ArgumentParser(description="Instrument inclusion tests")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--pool", default=str(REPO_ROOT / "random_pool.csv"))
    args = p.parse_args()

    print("=" * 104)
    print("INSTRUMENT INCLUSION TESTS")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 104)

    close, open_, spy, vix, pool = load(args.start, args.pool)
    print(f"A/D line will be computed over {len(pool)} pool names")

    live = [s for s in current_symbols() if s in close.columns]
    RATIOS = [RatioSpec("HYG_LQD", "HYG", "LQD"),
              RatioSpec("RSP_SPY", "RSP", "SPY"),
              RatioSpec("IWM_SPY", "IWM", "SPY")]
    BREADTH = [BreadthSpec("AD_LINE", pool)]

    def panel(names, ratios=(), breadth=(), monitors=()):
        # Dedupe while preserving order.  Several call sites append a leg that
        # is already in the live universe (HYG), and duplicate columns make
        # close[sym] return a DataFrame instead of a Series, which fails deep
        # inside pandas alignment with an error that names neither the symbol
        # nor the caller.
        keep = list(dict.fromkeys(n for n in names if n in close.columns))
        base = PriceData(close=close[keep].copy(), open_=open_[keep].copy(),
                         spy=spy, vix=vix)
        if ratios or breadth:
            base = augment(base, ratios=ratios, breadth=breadth,
                           monitor_symbols=monitors)
        return base

    rows = []

    def go(label, names, monitors, ratios=(), breadth=(), **over):
        cfg = production_config(monitor_symbols=list(monitors), **over)
        try:
            pr = panel(names, ratios, breadth, monitors)
            r = run_strategy(pr, cfg)
        except Exception as e:
            print(f"  {label}: FAILED {type(e).__name__}: {e}")
            return
        m, t = r.metrics, r.turnover
        n = len(r.holdings)
        hold = {s: r.holdings.apply(lambda x: s in x).sum() / n
                for s in ("UUP", "IEF", "TLT", "HYG")}
        rows.append({"Variant": label, "CAGR": m["cagr"],
                     "Sharpe": m["sharpe_ratio"], "MaxDD": m["max_drawdown"],
                     "Calmar": m["calmar_ratio"], "Trd/Yr": t["trades_per_year"],
                     **{f"{k}%": v for k, v in hold.items()}})
        print(f"  {label:<38} CAGR {m['cagr']:>7.2%}  DD {m['max_drawdown']:>7.2%}")

    NO_TLT = dict(vix__defensive_symbols=["SHY", "IEF", "IAU"],
                  vix__crisis_symbols=["SHY", "IEF"])
    no_hyg = [s for s in live if s != "HYG"]

    print("\nRunning variants:")
    go("0  live today (SH monitor)", live, ["SH"])
    go("1  + UUP tradable", live + ["UUP"], ["SH"])
    go("2  + UUP monitor only", live + ["UUP"], ["SH", "UUP"])
    go("3  IEF replaces TLT in sleeve",
       [s for s in live if s != "TLT"] + ["IEF"], ["SH"], **NO_TLT)
    go("4  HYG -> monitor, + HYG/LQD", no_hyg + ["HYG", "LQD"],
       ["SH", "HYG", "HYG_LQD"], ratios=[RATIOS[0]])
    go("5  + DBC monitor", live + ["DBC"], ["SH", "DBC"])
    go("6  + RSP/SPY monitor", live + ["RSP", "SPY"],
       ["SH", "RSP", "SPY", "RSP_SPY"], ratios=[RATIOS[1]])
    go("7  + IWM/SPY monitor", live + ["IWM", "SPY"],
       ["SH", "IWM", "SPY", "IWM_SPY"], ratios=[RATIOS[2]])
    go("8  + A/D line monitor", live, ["SH", "AD_LINE"], breadth=BREADTH)
    allmon = ["SH", "DBC", "HYG", "LQD", "RSP", "SPY", "IWM",
              "HYG_LQD", "RSP_SPY", "IWM_SPY", "AD_LINE"]
    go("9  ALL monitors (drag test)",
       no_hyg + ["HYG", "LQD", "RSP", "SPY", "IWM", "DBC"],
       allmon, ratios=RATIOS, breadth=BREADTH)
    go("10 all monitors + UUP tradable",
       no_hyg + ["HYG", "LQD", "RSP", "SPY", "IWM", "DBC", "UUP"],
       allmon, ratios=RATIOS, breadth=BREADTH)
    go("11 all monitors + UUP + IEF for TLT",
       [s for s in no_hyg if s != "TLT"] +
       ["HYG", "LQD", "RSP", "SPY", "IWM", "DBC", "UUP", "IEF"],
       allmon, ratios=RATIOS, breadth=BREADTH, **NO_TLT)

    t = pd.DataFrame(rows)
    base = t.iloc[0]
    print("\n" + "=" * 104)
    print("RESULTS  (deltas against variant 0, the current live model)")
    print("=" * 104)
    hdr = (f"{'Variant':<38}{'CAGR':>8}{'dCAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
           f"{'Calmar':>8}{'Trd/Yr':>8}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in t.iterrows():
        print(f"{r['Variant']:<38}{r['CAGR']:>7.2%}"
              f"{r['CAGR'] - base['CAGR']:>+9.2%}{r['Sharpe']:>8.2f}"
              f"{r['MaxDD']:>9.2%}{r['Calmar']:>8.2f}{r['Trd/Yr']:>8.1f}")
    print("-" * len(hdr))
    print("\n  Share of book-days each tradable was actually held:")
    for _, r in t.iterrows():
        bits = [f"{k[:-1]} {r[k]:.1%}" for k in ("UUP%", "IEF%", "TLT%", "HYG%")
                if r[k] > 0]
        if bits:
            print(f"    {r['Variant']:<38}{'  '.join(bits)}")

    pr = panel(live + LEGS, RATIOS, BREADTH,
               ["SH", "HYG_LQD", "RSP_SPY", "IWM_SPY", "AD_LINE"])
    print("\n" + "=" * 104)
    print("SYNTHETIC SERIES CHARACTER")
    print("=" * 104)
    print(describe(pr, ["HYG_LQD", "RSP_SPY", "IWM_SPY", "AD_LINE",
                        "UUP", "DBC", "IEF"]).to_string(index=False))

    t.to_csv(REPO_ROOT / OUT, index=False)
    print(f"\nExported to: {REPO_ROOT / OUT}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
