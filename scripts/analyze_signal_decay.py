"""
How fast does the ranking's information decay, and is 14 days the right hold?

Track G established that the composite carries short-lived information about
RISK rather than return: ranked selection beats random by ~3pp of drawdown at a
14-day hold and loses to it by 7.5pp at 63 days, while random selection improves
with longer holds as turnover falls.  That was measured at two points.  This
measures the curve, and finds where it crosses.

The question matters because of what the ranking buys.  It is not forecasting
returns - the quintile table shows the WORST-ranked names beating the best at
six of seven horizons, which is short-horizon reversal and is what anyone
holding these names should expect.  What the ranking does is refresh a risk
posture.  A posture decays from the moment it is set, so the hold length is the
single parameter that governs how much of it survives, and every day a book
rides is a day of decay.

THE CONFOUND THIS EXISTS TO REMOVE
   Changing `hold_days` changes the entire schedule of rebalance dates, and the
   schedule matters enormously on its own.  Holding `hold_days` at 14 and
   shifting only the rotation PHASE moves CAGR across a 6.94pp range (12.84% to
   19.78%, sd 2.68%) - as large as any effect a hold-length sweep could detect.
   A naive sweep therefore measures schedule-alignment luck and reports it as
   signal decay.

   Every arm here is averaged over `--phases` rotation phases, set by walking
   `min_data_days` forward one session at a time.  That changes the eligibility
   gate by at most a few days out of 200, which is negligible, while shifting
   the rebalance schedule by exactly one session, which is the point.

WHAT IS MEASURED
   For each hold length: the live configuration, and the same configuration with
   the ranking scores replaced by iid draws carrying the same NaN pattern.  The
   difference between them is what ranking is worth at that hold length.  The
   crossover - where ranked stops beating random - is the practical half-life.

   Random arms are averaged over `--trials` draws, and the spread is reported,
   because a single draw says nothing.  Both arms pay the same slippage on the
   same schedule, so the comparison is not confounded by turnover.

READ THE DRAWDOWN COLUMN, NOT THE CAGR COLUMN
   The ranking's contribution is risk, so CAGR is the wrong place to look for
   it.  A hold length that improves CAGR while giving up drawdown has not
   improved anything this model is for.

Every threshold is a flag.

Run:  python scripts/analyze_signal_decay.py
      python scripts/analyze_signal_decay.py --holds 7 14 21 28 --trials 40
      python scripts/analyze_signal_decay.py --top-n 30   # the wide book
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

from momentum.backtest import build_target_portfolios, simulate_portfolio
from momentum.data import load_data
from momentum.experiments import production_config
from momentum.strategy import compute_scores
from momentum.universe import current_symbols

OUT = "rsi_ma_signal_decay.csv"


def run_one(scores, base, prices, cfg, min_data_days=None):
    mdd = cfg.min_data_days if min_data_days is None else min_data_days
    targets, _ = build_target_portfolios(
        scores, price_columns=list(prices.close.columns), top_n=cfg.top_n,
        min_data_days=mdd, hold_days=cfg.hold_days,
        vix_data=prices.vix, vix_config=cfg.vix, base_composite_scores=base,
        velocity_config=cfg.velocity, correlation_config=cfg.correlation,
        graduated_config=cfg.graduated_vix, exit_config=cfg.exits,
        close=prices.close, rank_offset=cfg.rank_offset,
        rank_offset_scope=cfg.rank_offset_scope)
    r = simulate_portfolio(targets, prices.close, prices.open_,
                           execution=cfg.execution)
    return {
        "CAGR": r.metrics["cagr"],
        "Sharpe": r.metrics["sharpe_ratio"],
        "MaxDD": r.metrics["max_drawdown"],
        "Calmar": r.metrics["calmar_ratio"],
        "Vol": r.metrics["volatility"],
        "Trades/Year": r.turnover["trades_per_year"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Ranking information decay by hold length")
    p.add_argument("--holds", type=int, nargs="+",
                   default=[5, 7, 10, 14, 18, 21, 28, 35, 42, 63])
    p.add_argument("--trials", type=int, default=8,
                   help="random draws per hold length, per phase")
    p.add_argument("--phases", type=int, default=10,
                   help="rotation phases to average over; the confound this "
                        "script exists to remove")
    p.add_argument("--top-n", type=int, default=None,
                   help="book size (default: the live value)")
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--keep-unsettled", action="store_true")
    args = p.parse_args()

    print("=" * 100)
    print("SIGNAL DECAY BY HOLD LENGTH")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 100)

    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache,
                       drop_unsettled=not args.keep_unsettled)
    over = {} if args.top_n is None else {"top_n": args.top_n}
    cfg0 = production_config(**over)
    ranking, base, _ = compute_scores(prices, cfg0)
    print(f"\ntop_n={cfg0.top_n}, {args.trials} random draws per hold length.")

    rng = np.random.default_rng(args.seed)
    base_mdd = cfg0.min_data_days
    rows = []
    for h in args.holds:
        cfg = production_config(hold_days=h, **over)

        # Ranked: one run per phase, then averaged.
        rk = pd.DataFrame([run_one(ranking, base, prices, cfg, base_mdd + k)
                           for k in range(args.phases)])
        rows.append({"Hold": h, "Arm": "ranked",
                     **{k: rk[k].mean() for k in rk.columns},
                     **{f"{k} sd": rk[k].std() for k in ("CAGR", "MaxDD", "Sharpe")}})

        # Random: each draw gets its own phase, so the two sources of
        # variation are averaged together rather than compounded.
        draws = []
        for k in range(args.phases):
            for _ in range(args.trials):
                rnd = pd.DataFrame(rng.standard_normal(ranking.shape),
                                   index=ranking.index,
                                   columns=ranking.columns).where(ranking.notna())
                draws.append(run_one(rnd, base, prices, cfg, base_mdd + k))
        d = pd.DataFrame(draws)
        rows.append({"Hold": h, "Arm": "random",
                     **{k: d[k].mean() for k in d.columns},
                     **{f"{k} sd": d[k].std() for k in ("CAGR", "MaxDD", "Sharpe")}})
        print(f"  hold {h:>3}d   ranked {rk['CAGR'].mean():>7.2%} "
              f"(sd {rk['CAGR'].std():>5.2%}) DD {rk['MaxDD'].mean():>8.2%}   |   "
              f"random {d['CAGR'].mean():>7.2%} (sd {d['CAGR'].std():>5.2%}) "
              f"DD {d['MaxDD'].mean():>8.2%}")

    table = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("RANKED MINUS RANDOM, BY HOLD LENGTH")
    print("=" * 100)
    print("  Positive dDD means the ranking gives a SHALLOWER drawdown than a")
    print("  random book held on the same schedule. That is what it is buying.")
    header = (f"{'Hold':>6}{'CAGR (rk)':>12}{'CAGR (rnd)':>12}{'dCAGR':>9}"
              f"{'DD (rk)':>10}{'DD (rnd)':>11}{'dDD':>9}{'dSharpe':>10}")
    print(header)
    print("-" * len(header))
    crossover = None
    for h in args.holds:
        rk = table[(table.Hold == h) & (table.Arm == "ranked")].iloc[0]
        rd = table[(table.Hold == h) & (table.Arm == "random")].iloc[0]
        d_dd = rk["MaxDD"] - rd["MaxDD"]
        mark = ""
        if crossover is None and d_dd < 0:
            crossover, mark = h, "   <- ranking stops helping"
        print(f"{h:>6}{rk['CAGR']:>11.2%} {rd['CAGR']:>11.2%} "
              f"{rk['CAGR'] - rd['CAGR']:>+8.2%} {rk['MaxDD']:>9.2%} "
              f"{rd['MaxDD']:>10.2%} {d_dd:>+8.2%} "
              f"{rk['Sharpe'] - rd['Sharpe']:>+9.2f}{mark}")
    print("-" * len(header))
    if crossover:
        print(f"\n  Crossover between {args.holds[args.holds.index(crossover)-1]} "
              f"and {crossover} days: beyond that the ranking is worse than "
              f"nothing.")
    else:
        print("\n  No crossover inside the tested range.")

    table.to_csv(REPO_ROOT / OUT, index=False)
    print(f"\nExported to: {REPO_ROOT / OUT}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
