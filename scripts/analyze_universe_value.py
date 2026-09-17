"""
Is the universe's return a large-cap growth tilt, and how small can it be?

Track F established that the universe, not the ranker, produces the return:
equal-weighting all 49 momentum names and never trading earns 19.99% against the
model's 19.58%.  That raises two questions this answers.

PART A - IS IT JUST A FACTOR TILT?
   The stated hypothesis is that the universe's advantage is one attribute:
   an overweight to large-cap domestic growth.  If so, the right benchmark is
   not SPY but a growth index, and the universe's edge over *that* is what is
   left to explain.  Reported as a head-to-head table and as a regression of
   universe returns on each benchmark, so the surviving alpha and its
   t-statistic are visible rather than inferred from a CAGR difference.

   A CAGR gap is not evidence of skill if it comes with a beta above one.  The
   regression is the part that separates "we out-earned growth" from "we held
   more of what growth holds".

PART B - HOW SMALL CAN THE UNIVERSE BE?
   Draw K of the momentum names at random, equal-weight, hold throughout, and
   report the distribution of CAGR and max drawdown across many draws.  This is
   the diversification curve for the universe itself, and it answers the
   maintenance question directly: if 20 names retain most of what 49 deliver,
   most of the screening work is optional.

   Note what this does NOT say.  It measures how much of the *historical* result
   survives a smaller universe.  The historical result carries the survivorship
   premium that TODO item 6 exists to bound, and a smaller draw from a
   contaminated set is still contaminated.  Read the spread, not the level: the
   useful output is how fast the distribution widens as K falls, because that
   width is the risk of getting the screen wrong, and it is not survivorship
   dependent in the same way.

A NOTE ON WHICH 'EQUAL-WEIGHT UNIVERSE' NUMBER TO QUOTE
   This script gates inclusion on PRICE availability and reports 21.04%.
   `analyze_null_benchmark.py` gates on SCORE availability and reports 20.02%.
   The gap is the first year: the composite needs 200 days of history, so the
   score-gated book holds ~20 names over the first 250 sessions where the
   price-gated one holds ~45.  Neither is wrong.  Quote the SCORE-gated figure
   when comparing against the model, because the model can only ever hold names
   it can score; quote this one when asking what the universe itself did.
   Part A and Part B below are internally consistent - both use the price-gated
   path - so the comparisons within each table are sound either way.

Every threshold is a flag.

Run:  python scripts/analyze_universe_value.py
      python scripts/analyze_universe_value.py --sizes 5 10 20 30 49 --trials 400
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

from momentum.data import load_data
from momentum.experiments import production_config
from momentum.metrics import calculate_performance_metrics
from momentum.universe import current_symbols, defensive_symbols

OUT_BENCH = "rsi_ma_universe_benchmarks.csv"
OUT_SIZES = "rsi_ma_universe_size_curve.csv"

DEFAULT_BENCHMARKS = ["SPY", "RSP", "QQQ", "IWF", "SPYG", "MTUM", "IWM", "VTV"]
BENCH_LABELS = {
    "SPY": "S&P 500",
    "RSP": "S&P 500 equal weight",
    "QQQ": "Nasdaq 100",
    "IWF": "Russell 1000 Growth",
    "SPYG": "S&P 500 Growth",
    "MTUM": "MSCI USA Momentum (from 2013)",
    "IWM": "Russell 2000 (small cap)",
    "VTV": "Vanguard Value",
}


def fetch_benchmarks(symbols, start: str) -> pd.DataFrame:
    """Adjusted closes for the benchmark ETFs, via yfinance."""
    import yfinance as yf
    warnings.filterwarnings("ignore")
    raw = yf.download(symbols, start=start, interval="1d", auto_adjust=True,
                      progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return close.dropna(how="all")


def equal_weight_returns(close: pd.DataFrame, names) -> pd.Series:
    """
    Daily return of an equal-weighted book, on the simulator's convention.

    `_segment_return` resets to equal weight every day, so the equal-weighted
    daily return is the cross-sectional mean of the constituents' daily returns.
    Computing it directly rather than through `simulate_portfolio` is ~2000x
    faster, which is what makes the size curve affordable; `verify_fast_path`
    checks the two agree before any of it is reported.
    """
    return close[list(names)].pct_change().mean(axis=1)


def verify_fast_path(prices, cfg, names, tol: float) -> float:
    """The fast path must reproduce the simulator, or the curve means nothing."""
    from momentum.backtest import simulate_portfolio
    dates = prices.close.index[cfg.min_data_days:]
    targets = pd.Series({d: sorted(names) for d in dates}, name="target")
    sim = simulate_portfolio(targets, prices.close, prices.open_,
                             execution=cfg.execution)
    fast = equal_weight_returns(prices.close, names).reindex(sim.returns.index)
    m_sim = calculate_performance_metrics(
        sim.returns, risk_free_rate=cfg.execution.risk_free_rate)
    m_fast = calculate_performance_metrics(
        fast.dropna(), risk_free_rate=cfg.execution.risk_free_rate)
    gap = abs(m_sim["cagr"] - m_fast["cagr"])
    print(f"  simulator {m_sim['cagr']:.2%}   fast path {m_fast['cagr']:.2%}   "
          f"gap {gap:.4%}")
    if gap > tol:
        raise SystemExit(
            f"Fast path disagrees with the simulator by {gap:.4%} "
            f"(tolerance {tol:.4%}). Refusing to report a curve built on it.")
    return gap


def regress(y: pd.Series, x: pd.Series, rf_annual: float, periods: int = 12):
    """
    OLS of excess y on excess x, at monthly frequency.

    Returns annualized alpha, beta, the alpha t-statistic and R-squared.  Monthly
    rather than daily because daily equity regressions are dominated by
    microstructure noise and the alpha standard error is what matters here.
    """
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 24:
        return None
    rf = rf_annual / periods
    yv = df.iloc[:, 0].to_numpy() - rf
    xv = df.iloc[:, 1].to_numpy() - rf

    X = np.column_stack([np.ones(len(xv)), xv])
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ coef
    dof = len(yv) - 2
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    return {
        "Alpha (ann.)": float(coef[0] * periods),
        "Alpha t": float(coef[0] / se[0]),
        "Beta": float(coef[1]),
        "R2": float(1 - (resid @ resid) / ss_tot) if ss_tot else np.nan,
        "Months": len(df),
    }


def part_a(prices, cfg, momentum_syms, start, benchmarks) -> pd.DataFrame:
    uni = equal_weight_returns(prices.close, momentum_syms)
    uni = uni.loc[prices.close.index[cfg.min_data_days]:].dropna()
    rf = cfg.execution.risk_free_rate

    bench_close = fetch_benchmarks(benchmarks, start)
    bench_close = bench_close.reindex(uni.index).ffill()

    m_uni = calculate_performance_metrics(uni, risk_free_rate=rf)
    rows = [{"Benchmark": "Equal-weight universe (49 names)",
             "CAGR": m_uni["cagr"], "Sharpe": m_uni["sharpe_ratio"],
             "MaxDD": m_uni["max_drawdown"], "Vol": m_uni["volatility"],
             "Alpha (ann.)": np.nan, "Alpha t": np.nan, "Beta": np.nan,
             "R2": np.nan, "Months": np.nan}]

    uni_m = (1 + uni).resample("ME").prod() - 1
    for sym in benchmarks:
        if sym not in bench_close.columns:
            continue
        b = bench_close[sym].pct_change().dropna()
        if b.empty:
            continue
        m = calculate_performance_metrics(b, risk_free_rate=rf)
        b_m = (1 + b).resample("ME").prod() - 1
        reg = regress(uni_m, b_m, rf) or {}
        rows.append({
            "Benchmark": f"{sym} - {BENCH_LABELS.get(sym, sym)}",
            "CAGR": m["cagr"], "Sharpe": m["sharpe_ratio"],
            "MaxDD": m["max_drawdown"], "Vol": m["volatility"], **reg,
        })
    return pd.DataFrame(rows)


def print_part_a(table: pd.DataFrame) -> None:
    print("\n" + "=" * 104)
    print("PART A - THE UNIVERSE AGAINST FACTOR BENCHMARKS")
    print("=" * 104)
    header = (f"{'Benchmark':<36}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Vol':>8}"
              f"{'Alpha':>9}{'t':>7}{'Beta':>7}{'R2':>7}")
    print(header)
    print("-" * len(header))
    for _, r in table.iterrows():
        a = f"{r['Alpha (ann.)']:>8.2%}" if pd.notna(r["Alpha (ann.)"]) else "       -"
        t = f"{r['Alpha t']:>6.2f}" if pd.notna(r["Alpha t"]) else "     -"
        b = f"{r['Beta']:>6.2f}" if pd.notna(r["Beta"]) else "     -"
        r2 = f"{r['R2']:>6.2f}" if pd.notna(r["R2"]) else "     -"
        print(f"{r['Benchmark']:<36}{r['CAGR']:>8.2%} {r['Sharpe']:>7.2f} "
              f"{r['MaxDD']:>8.2%} {r['Vol']:>7.2%} {a} {t} {b} {r2}")
    print("-" * len(header))
    print("  Alpha is annualized, from a monthly regression of the universe on")
    print("  the benchmark. Beta above 1 means the CAGR gap is partly leverage")
    print("  to that factor rather than selection.")


def part_b(prices, cfg, momentum_syms, sizes, trials, seed) -> pd.DataFrame:
    close = prices.close.loc[prices.close.index[cfg.min_data_days]:]
    rf = cfg.execution.risk_free_rate
    rng = np.random.default_rng(seed)
    pool = list(momentum_syms)

    rows = []
    for k in sizes:
        k = min(k, len(pool))
        n = 1 if k == len(pool) else trials
        for _ in range(n):
            pick = rng.choice(pool, size=k, replace=False) if k < len(pool) else pool
            r = equal_weight_returns(close, pick).dropna()
            if len(r) < 30:
                continue
            m = calculate_performance_metrics(r, risk_free_rate=rf)
            rows.append({"K": k, "CAGR": m["cagr"], "Sharpe": m["sharpe_ratio"],
                         "MaxDD": m["max_drawdown"], "Vol": m["volatility"]})
    return pd.DataFrame(rows)


def print_part_b(curve: pd.DataFrame, full_k: int) -> None:
    print("\n" + "=" * 104)
    print("PART B - HOW SMALL CAN THE UNIVERSE BE?")
    print("=" * 104)
    print("  Random K of the momentum names, equal-weighted, held throughout.")
    print("  Read the SPREAD, not the level: p5-to-p95 is the risk of picking")
    print("  the wrong K names, and it is what shrinking the universe costs you.")
    header = (f"{'K':>5}{'Median CAGR':>14}{'p5':>10}{'p95':>10}{'p95-p5':>10}"
              f"{'Median DD':>12}{'Worst DD':>11}{'Median Sh':>11}")
    print(header)
    print("-" * len(header))
    for k in sorted(curve["K"].unique()):
        sub = curve[curve["K"] == k]
        if len(sub) == 1:
            r = sub.iloc[0]
            print(f"{k:>5}{r['CAGR']:>13.2%} {'-':>9} {'-':>9} {'-':>9} "
                  f"{r['MaxDD']:>11.2%} {r['MaxDD']:>10.2%} {r['Sharpe']:>10.2f}"
                  f"   <- the whole universe")
            continue
        p5, p95 = sub["CAGR"].quantile(.05), sub["CAGR"].quantile(.95)
        print(f"{k:>5}{sub['CAGR'].median():>13.2%} {p5:>9.2%} {p95:>9.2%} "
              f"{p95 - p5:>9.2%} {sub['MaxDD'].median():>11.2%} "
              f"{sub['MaxDD'].min():>10.2%} {sub['Sharpe'].median():>10.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Universe factor attribution and size curve")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[3, 5, 8, 10, 15, 20, 25, 30, 40, 49])
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="max CAGR gap allowed between the fast path and "
                             "the simulator before the run aborts")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--keep-unsettled", action="store_true")
    args = parser.parse_args()

    print("=" * 104)
    print("UNIVERSE VALUE: FACTOR ATTRIBUTION AND SIZE CURVE")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 104)

    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache,
                       drop_unsettled=not args.keep_unsettled)
    cfg = production_config()
    defensive = set(defensive_symbols())
    momentum_syms = [s for s in current_symbols()
                     if s not in defensive and s in prices.close.columns]
    print(f"\n{len(momentum_syms)} momentum names in the universe.")

    print("\nVerifying the fast equal-weight path against the simulator:")
    verify_fast_path(prices, cfg, momentum_syms, args.tolerance)

    table = part_a(prices, cfg, momentum_syms, args.start, args.benchmarks)
    print_part_a(table)

    curve = part_b(prices, cfg, momentum_syms, args.sizes, args.trials, args.seed)
    print_part_b(curve, len(momentum_syms))

    table.to_csv(REPO_ROOT / OUT_BENCH, index=False)
    curve.to_csv(REPO_ROOT / OUT_SIZES, index=False)
    print(f"\nBenchmarks exported to: {REPO_ROOT / OUT_BENCH}")
    print(f"Size curve exported to: {REPO_ROOT / OUT_SIZES}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
