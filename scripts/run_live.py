"""
Live model run — produces the current signal and the standard exports.

    python scripts/run_live.py
    python scripts/run_live.py --legacy      reproduce pre-refactor behaviour
    python scripts/run_live.py --no-plots    skip charts (for scheduled runs)

To change the ticker universe, edit `universe.csv` in the repo root.
To change model parameters, edit `build_config()` below.

Every run snapshots the universe and the config to `snapshots/`, dated.  That is
what makes a future live-vs-backtest reconciliation answerable from the record
instead of from memory.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.config import (ExecutionConfig, ModelConfig, ScoringConfig,
                             VelocityConfig, VixRegimeConfig, snapshot_config)
from momentum.data import export_price_data, load_data
from momentum.experiments import legacy_config, production_config
from momentum.reports import (correlation_matrices, individual_stock_performance,
                              portfolio_concentration, portfolio_correlation,
                              summarize_correlations)
from momentum.strategy import compute_scores, run_strategy
from momentum.universe import (current_symbols, sector_map,
                               snapshot_current_universe)


def build_config(legacy: bool = False) -> ModelConfig:
    """
    The live model configuration.  Edit here.

    Current settings and where they came from:
      zscore_window=126     chosen by the z-score window comparison
      velocity 0.7/0.3      chosen by the velocity sweep — but see the note
                            below; that sweep ran against a scale mismatch and
                            is being re-run
      VIX z 1.5/2.5         chosen by the VIX regime comparison; superseded by
                            the graduated ladder once Track A lands
      execute_at next_open  realistic fill; see ExecutionConfig
    """
    if legacy:
        return legacy_config(notes="pre-refactor reproduction")

    return production_config(
        notes="post-refactor baseline: fixed blend scale, realistic execution"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the live momentum model")
    parser.add_argument("--legacy", action="store_true",
                        help="reproduce pre-refactor behaviour exactly")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip charts, for scheduled/headless runs")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true",
                        help="force a fresh download")
    args = parser.parse_args()

    config = build_config(legacy=args.legacy)
    today = date.today()

    print("=" * 78)
    print(f"MOMENTUM MODEL — LIVE RUN  {today.isoformat()}")
    if args.legacy:
        print("MODE: legacy (pre-refactor reproduction)")
    print("=" * 78)

    # --- point-in-time record (Request #1) ---
    uni_path = snapshot_current_universe(as_of=today)
    cfg_path = snapshot_config(config, as_of=today,
                               label="legacy" if args.legacy else "live")
    print(f"\nSnapshots written:\n  {uni_path.name}\n  {cfg_path.name}")

    symbols = current_symbols()
    prices = load_data(symbols, start_date=args.start,
                       use_cache=not args.no_cache)

    # --- price panel export (Request #2) ---
    print("\n=== EXPORTING PRICE DATA ===")
    export_price_data(prices)

    # --- scores ---
    print("\n=== CALCULATING COMPOSITE SCORES ===")
    ranking_scores, base_scores, detail = compute_scores(
        prices, config, underlying_path=str(REPO_ROOT / "rsi_ma_underlying_measures.csv")
    )

    # --- backtest / signal ---
    print("\n=== RUNNING STRATEGY ===")
    result = run_strategy(prices, config, verbose=True)

    m, t = result.metrics, result.turnover
    print("\n=== STRATEGY RESULTS ===")
    print(f"  Total Return   {m['total_return']:>10.2%}")
    print(f"  CAGR           {m['cagr']:>10.2%}")
    print(f"  Volatility     {m['volatility']:>10.2%}")
    print(f"  Sharpe Ratio   {m['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio  {m['sortino_ratio']:>10.2f}")
    print(f"  Max Drawdown   {m['max_drawdown']:>10.2%}")
    print(f"  Calmar Ratio   {m['calmar_ratio']:>10.2f}")
    print(f"  Trading Days   {m['num_periods']:>10,}")
    print("\n  --- turnover ---")
    print(f"  Trades / year  {t['trades_per_year']:>10.1f}")
    print(f"  Avg hold       {t['avg_hold_days']:>10.1f} days")
    print(f"  Median hold    {t['median_hold_days']:>10.1f} days")
    print(f"  Annual turnover{t['annual_turnover']:>10.1%}")
    print(f"  Slippage paid  {result.total_cost:>10.2%} cumulative")

    # --- current holdings ---
    if result.rebalance_history:
        latest = result.rebalance_history[-1]
        print(f"\n=== CURRENT HOLDINGS "
              f"(last rebalance {latest['Date']:%Y-%m-%d}, "
              f"regime {latest['Regime'].upper()}) ===")
        for i, (stock, score) in enumerate(latest["Scores"].items(), 1):
            print(f"  {i}. {stock:<6} score {score:>7.3f}")

        # Correlation first — it is the measure that reflects shared risk.
        # Sector labels are a poor proxy in both directions and are reported
        # only as context.
        pairs = portfolio_correlation(latest["Selected_Stocks"], prices.close,
                                      window=50)
        if not pairs.empty:
            worst = pairs.iloc[0]
            print(f"\n  Book correlation (50d): max pair "
                  f"{worst['Symbol A']}/{worst['Symbol B']} "
                  f"{worst['Correlation']:.2f}, mean "
                  f"{pairs['Correlation'].mean():.2f}")
            if worst["Correlation"] >= 0.70:
                print(f"    ^ {worst['Symbol A']} and {worst['Symbol B']} are "
                      f"effectively one position.")

        conc = portfolio_concentration(latest["Selected_Stocks"], sector_map())
        if not conc.empty and (conc["Positions"] > 1).any():
            top = conc.iloc[0]
            print(f"  Sector: {top['Weight']:.0%} {top['Sector']}"
                  + (" (label only — see correlation above)"
                     if not pairs.empty and pairs["Correlation"].max() < 0.5
                     else ""))

    # --- exports ---
    print("\n=== EXPORTING RESULTS ===")
    out = REPO_ROOT

    perf = pd.DataFrame({
        "Date": result.returns.index,
        "Portfolio_Return": result.returns.values,
        "Gross_Return": result.gross_returns.values,
        "Holdings": result.holdings.values,
    })
    perf.to_csv(out / "rsi_ma_portfolio_performance.csv", index=False)
    ranking_scores.to_csv(out / "rsi_ma_composite_scores.csv")
    base_scores.to_csv(out / "rsi_ma_composite_scores_level_only.csv")
    pd.DataFrame(result.rebalance_history).to_csv(
        out / "rsi_ma_rebalance_history.csv", index=False)

    perf_by_stock = individual_stock_performance(prices.close)
    perf_by_stock.to_csv(out / "rsi_ma_individual_stock_performance.csv", index=False)

    for name in ("rsi_ma_portfolio_performance.csv", "rsi_ma_composite_scores.csv",
                 "rsi_ma_composite_scores_level_only.csv",
                 "rsi_ma_rebalance_history.csv",
                 "rsi_ma_individual_stock_performance.csv"):
        print(f"- {name}")

    # --- correlations ---
    print("\n=== CORRELATION MATRICES ===")
    for period, matrix in correlation_matrices(prices.close).items():
        path = out / f"rsi_ma_correlation_{period}d.csv"
        matrix.to_csv(path)
        print(summarize_correlations(matrix, period))
        print(f"Exported to: {path.name}")

    # --- charts ---
    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt

            wealth = (1 + result.returns).cumprod()
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            axes[0].plot(wealth.index, wealth.values, linewidth=1.6)
            axes[0].set_title("Momentum strategy — cumulative return "
                              f"({config.execution.execute_at} execution, "
                              f"{config.execution.slippage_bps:g} bps slippage)")
            axes[0].set_ylabel("Growth of $1")
            axes[0].set_yscale("log")
            axes[0].grid(alpha=0.3)

            axes[1].plot(result.returns.index, result.returns.values * 100,
                         linewidth=0.6, alpha=0.8)
            axes[1].set_title("Daily returns (%)")
            axes[1].set_xlabel("Date")
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            plt.show()
        except Exception as exc:      # a headless box should not fail the run
            print(f"\n(Charts skipped: {exc})")

    print("\n=== RUN COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
