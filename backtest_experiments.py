"""
Backtesting Experiments Framework

Run parameterized backtests on the RSI/MA portfolio strategy to compare
different configurations (rebalancing frequency, parameters, etc.)

Usage:
    python backtest_experiments.py

This file imports from stock_alloc_rsi_ma.py and runs experiments without
modifying the production model.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

# Import functions from production model
from stock_alloc_rsi_ma import (
    calculate_composite_scores,
    select_top_stocks_biweekly,
    calculate_performance_metrics,
    etfs
)


@dataclass
class ExperimentConfig:
    """Configuration for a backtest experiment."""
    name: str
    hold_days: int = 14
    top_n: int = 4
    min_data_days: int = 200
    rsi_window: int = 14
    ma_short: int = 50
    ma_long: int = 200
    derivative_window: int = 5
    zscore_window: int = 252
    rel_strength_window: int = 20


@dataclass
class ExperimentResult:
    """Results from a backtest experiment."""
    config: ExperimentConfig
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    num_days: int
    num_rebalances: int
    portfolio_performance: pd.DataFrame = field(repr=False)
    rebalance_history: List[Dict] = field(repr=False)


def load_data(symbols: List[str], start_date: str = '2010-01-01', verbose: bool = True):
    """
    Load and clean historical price data.

    Parameters:
    symbols: List of stock symbols to download
    start_date: Start date for historical data
    verbose: Whether to print progress messages

    Returns:
    tuple: (data DataFrame, spy_data Series)
    """
    if verbose:
        print(f"Downloading data for {len(symbols)} symbols...")

    # Download stock data
    data = yf.download(symbols, start=start_date, interval='1d', progress=verbose)["Close"]

    # Download SPY for relative strength
    spy_data = yf.download('SPY', start=start_date, interval='1d', progress=False)["Close"]

    # Clean data - remove stocks with insufficient recent data
    one_year_ago = data.index[-1] - pd.DateOffset(months=12)
    recent_data = data.loc[one_year_ago:]
    data = data.dropna(axis=1, thresh=recent_data.shape[0])

    # Align SPY data
    spy_data = spy_data.reindex(data.index, method='ffill')

    if verbose:
        print(f"Data shape after cleaning: {data.shape}")
        print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")

    return data, spy_data


def run_experiment(config: ExperimentConfig, data: pd.DataFrame, spy_data: pd.Series,
                   verbose: bool = False) -> ExperimentResult:
    """
    Run a single backtest experiment with the given configuration.

    Parameters:
    config: ExperimentConfig with parameters
    data: Price data DataFrame
    spy_data: SPY price series for relative strength
    verbose: Whether to print rebalancing messages

    Returns:
    ExperimentResult with performance metrics
    """
    # Calculate composite scores
    composite_scores, _, _, _, _ = calculate_composite_scores(
        data,
        spy_data,
        rsi_window=config.rsi_window,
        ma_short=config.ma_short,
        ma_long=config.ma_long,
        derivative_window=config.derivative_window,
        zscore_window=config.zscore_window,
        rel_strength_window=config.rel_strength_window
    )

    # Temporarily suppress print statements from select_top_stocks_biweekly
    import sys
    from io import StringIO

    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = StringIO()

    try:
        portfolio_performance, rebalance_history = select_top_stocks_biweekly(
            composite_scores,
            data,
            top_n=config.top_n,
            min_data_days=config.min_data_days,
            hold_days=config.hold_days
        )
    finally:
        if not verbose:
            sys.stdout = old_stdout

    if portfolio_performance.empty:
        raise ValueError(f"No portfolio performance data generated for config: {config.name}")

    # Calculate metrics
    returns = portfolio_performance['Portfolio_Return']
    metrics = calculate_performance_metrics(returns, is_daily=True)

    # Parse metrics (they come back as formatted strings)
    def parse_pct(s): return float(s.replace('%', '')) / 100
    def parse_float(s): return float(s)

    return ExperimentResult(
        config=config,
        total_return=parse_pct(metrics['Total Return']),
        cagr=parse_pct(metrics['CAGR']),
        volatility=parse_pct(metrics['Volatility']),
        sharpe_ratio=parse_float(metrics['Sharpe Ratio']),
        max_drawdown=parse_pct(metrics['Max Drawdown']),
        num_days=len(returns),
        num_rebalances=len(rebalance_history),
        portfolio_performance=portfolio_performance,
        rebalance_history=rebalance_history
    )


def compare_experiments(results: List[ExperimentResult]) -> pd.DataFrame:
    """
    Create a comparison table of experiment results.

    Parameters:
    results: List of ExperimentResult objects

    Returns:
    DataFrame with comparison metrics
    """
    comparison_data = []

    for result in results:
        comparison_data.append({
            'Experiment': result.config.name,
            'Hold Days': result.config.hold_days,
            'Top N': result.config.top_n,
            'Total Return': f"{result.total_return:.2%}",
            'CAGR': f"{result.cagr:.2%}",
            'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
            'Volatility': f"{result.volatility:.2%}",
            'Max Drawdown': f"{result.max_drawdown:.2%}",
            'Trading Days': result.num_days,
            'Rebalances': result.num_rebalances,
        })

    return pd.DataFrame(comparison_data)


def print_comparison(results: List[ExperimentResult]):
    """Print a formatted comparison of experiment results."""
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)

    # Header
    headers = ['Metric'] + [r.config.name for r in results]
    col_widths = [max(20, len(h)+2) for h in headers]

    header_line = ""
    for i, h in enumerate(headers):
        header_line += h.ljust(col_widths[i])
    print(header_line)
    print("-" * sum(col_widths))

    # Metrics
    metrics = [
        ('Total Return', lambda r: f"{r.total_return:.2%}"),
        ('CAGR', lambda r: f"{r.cagr:.2%}"),
        ('Sharpe Ratio', lambda r: f"{r.sharpe_ratio:.2f}"),
        ('Volatility', lambda r: f"{r.volatility:.2%}"),
        ('Max Drawdown', lambda r: f"{r.max_drawdown:.2%}"),
        ('Trading Days', lambda r: f"{r.num_days:,}"),
        ('Rebalances', lambda r: f"{r.num_rebalances:,}"),
    ]

    for metric_name, metric_fn in metrics:
        line = metric_name.ljust(col_widths[0])
        for i, result in enumerate(results):
            line += metric_fn(result).ljust(col_widths[i+1])
        print(line)

    print("="*80)

    # Highlight best performers
    print("\nKey Findings:")

    # Best CAGR
    best_cagr = max(results, key=lambda r: r.cagr)
    print(f"  - Highest CAGR: {best_cagr.config.name} ({best_cagr.cagr:.2%})")

    # Best Sharpe
    best_sharpe = max(results, key=lambda r: r.sharpe_ratio)
    print(f"  - Best Sharpe Ratio: {best_sharpe.config.name} ({best_sharpe.sharpe_ratio:.2f})")

    # Lowest Drawdown (less negative is better)
    best_dd = max(results, key=lambda r: r.max_drawdown)
    print(f"  - Smallest Max Drawdown: {best_dd.config.name} ({best_dd.max_drawdown:.2%})")


def run_rebalance_frequency_comparison(data: pd.DataFrame, spy_data: pd.Series):
    """
    Run the experimental comparison.

    Parameters:
    data: Price data DataFrame
    spy_data: SPY price series

    Returns:
    List of ExperimentResult objects
    """
    configs = [
        ExperimentConfig(name="Experiment", top_n=4, hold_days=14, ma_short=50, ma_long=200, zscore_window=252, derivative_window=5),
        ExperimentConfig(name="Baseline", hold_days=14),
    ]

    results = []
    for config in configs:
        print(f"\nRunning experiment: {config.name}...")
        result = run_experiment(config, data, spy_data, verbose=False)
        results.append(result)
        print(f"  CAGR: {result.cagr:.2%}, Sharpe: {result.sharpe_ratio:.2f}")

    return results


def main():
    """Main entry point for running experiments."""
    print("="*80)
    print("PORTFOLIO BACKTEST EXPERIMENTS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data once (reused across experiments)
    print("\n--- Loading Data ---")
    data, spy_data = load_data(etfs)

    # Run weekly vs biweekly comparison
    print("\n--- Running Weekly vs Biweekly Comparison ---")
    results = run_rebalance_frequency_comparison(data, spy_data)

    # Print comparison
    print_comparison(results)

    # Export comparison to CSV
    comparison_df = compare_experiments(results)
    output_file = 'backtest_comparison.csv'
    comparison_df.to_csv(output_file, index=False)
    print(f"\nComparison exported to: {output_file}")

    # Export detailed results for each experiment
    for result in results:
        safe_name = result.config.name.replace(' ', '_').replace('(', '').replace(')', '')
        perf_file = f'backtest_{safe_name}_performance.csv'
        result.portfolio_performance.to_csv(perf_file, index=False)
        print(f"Performance details exported to: {perf_file}")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
