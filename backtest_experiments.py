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
    VixRegimeConfig,
    VelocityConfig,
    apply_velocity_blend,
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
    zscore_method: str = 'cross_sectional'  # 'cross_sectional' or 'rolling'
    vix_config: Optional[VixRegimeConfig] = None       # None = no VIX filter
    velocity_config: Optional[VelocityConfig] = None  # None = no velocity blend


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
    tuple: (data DataFrame, spy_data Series, vix_data Series)
    """
    if verbose:
        print(f"Downloading data for {len(symbols)} symbols...")

    # Download stock data
    data = yf.download(symbols, start=start_date, interval='1d', progress=verbose)["Close"]

    # Download SPY for relative strength, and VIX for regime detection
    spy_data = yf.download('SPY',  start=start_date, interval='1d', progress=False)["Close"]
    vix_data = yf.download('^VIX', start=start_date, interval='1d', progress=False)["Close"].squeeze()

    # Clean data - remove stocks with insufficient recent data
    one_year_ago = data.index[-1] - pd.DateOffset(months=12)
    recent_data = data.loc[one_year_ago:]
    data = data.dropna(axis=1, thresh=recent_data.shape[0])

    # Align SPY and VIX to our stock data index
    spy_data = spy_data.reindex(data.index, method='ffill')
    vix_data = vix_data.reindex(data.index, method='ffill')

    if verbose:
        print(f"Data shape after cleaning: {data.shape}")
        print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")

    return data, spy_data, vix_data


def run_experiment(config: ExperimentConfig, data: pd.DataFrame, spy_data: pd.Series,
                   vix_data: Optional[pd.Series] = None,
                   verbose: bool = False) -> ExperimentResult:
    """
    Run a single backtest experiment with the given configuration.

    Parameters:
    config: ExperimentConfig with parameters
    data: Price data DataFrame
    spy_data: SPY price series for relative strength
    vix_data: VIX price series for regime detection (required when config.vix_config is set)
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
        rel_strength_window=config.rel_strength_window,
        zscore_method=config.zscore_method,
        save_underlying=False,
    )

    # Apply velocity blend if configured; preserve originals for floor filter
    base_composite_scores = composite_scores.copy()
    if config.velocity_config is not None:
        composite_scores = apply_velocity_blend(composite_scores, config.velocity_config)

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
            hold_days=config.hold_days,
            vix_data=vix_data,
            vix_config=config.vix_config,
            base_composite_scores=base_composite_scores,
            velocity_config=config.velocity_config,
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
        vc = result.config.vix_config
        if vc is None:
            vix_summary = 'none'
        else:
            roc_note = f'+roc{vc.roc_threshold:.0%}' if vc.use_roc_trigger else ''
            vix_summary = (f'z>{vc.elevated_zscore}/{vc.crisis_zscore}'
                           f'(win={vc.zscore_window}){roc_note}')

        vel = result.config.velocity_config
        if vel is None:
            vel_summary = 'none'
        else:
            vel_summary = (f'win={vel.velocity_window} '
                           f'L{vel.level_weight:.0%}/V{vel.velocity_weight:.0%}')

        comparison_data.append({
            'Experiment': result.config.name,
            'VIX Config': vix_summary,
            'Velocity Config': vel_summary,
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


def run_rebalance_frequency_comparison(data: pd.DataFrame, spy_data: pd.Series,
                                        vix_data: Optional[pd.Series] = None):
    """
    Run the experimental comparison.

    Parameters:
    data: Price data DataFrame
    spy_data: SPY price series
    vix_data: VIX price series (passed through to run_experiment)

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
        result = run_experiment(config, data, spy_data, vix_data=vix_data, verbose=False)
        results.append(result)
        print(f"  CAGR: {result.cagr:.2%}, Sharpe: {result.sharpe_ratio:.2f}")

    return results


def run_zscore_method_comparison(data: pd.DataFrame, spy_data: pd.Series,
                                  vix_data: Optional[pd.Series] = None):
    """
    Compare cross-sectional vs rolling z-score normalization across multiple windows.

    Configs:
      - cross_sectional: baseline (current production behavior)
      - rolling 63/126/252/504 days: time-series normalization vs each stock's own history

    min_data_days is set to max(200, window) so the backtest only starts after both
    the 200-day MA and the rolling z-score window have fully warmed up.

    Parameters:
    data: Price data DataFrame
    spy_data: SPY price series
    vix_data: VIX price series (passed through to run_experiment)

    Returns:
    List of ExperimentResult objects
    """
    configs = [
        ExperimentConfig(name='cross_sectional', zscore_method='cross_sectional', zscore_window=252, min_data_days=200),
        ExperimentConfig(name='rolling_63d',     zscore_method='rolling',          zscore_window=63,  min_data_days=200),
        ExperimentConfig(name='rolling_126d',    zscore_method='rolling',          zscore_window=126, min_data_days=200),
        ExperimentConfig(name='rolling_252d',    zscore_method='rolling',          zscore_window=252, min_data_days=252),
        ExperimentConfig(name='rolling_504d',    zscore_method='rolling',          zscore_window=504, min_data_days=504),
    ]

    results = []
    for config in configs:
        print(f"\n  Running {config.name}  (zscore_window={config.zscore_window}, min_data_days={config.min_data_days})...")
        result = run_experiment(config, data, spy_data, vix_data=vix_data, verbose=False)
        results.append(result)
        print(f"    CAGR: {result.cagr:.2%}  Sharpe: {result.sharpe_ratio:.2f}  MaxDD: {result.max_drawdown:.2%}  Days: {result.num_days:,}")

    return results


def run_vix_regime_comparison(data: pd.DataFrame, spy_data: pd.Series, vix_data: pd.Series):
    """
    Compare VIX regime filter variants against the no-filter baseline.

    Each config uses the same core strategy (rolling z-score, 126-day window) and
    only varies the VIX regime thresholds so differences are attributable to the filter.

    Tuning levers in VixRegimeConfig:
      zscore_window   — rolling baseline for VIX z-score (longer = slower to react)
      elevated_zscore — z-score at which we shift to partial-defensive allocation
      crisis_zscore   — z-score at which we go fully defensive
      elevated_top_n  — momentum picks to keep in 'elevated' regime
      use_roc_trigger — also enter 'elevated' on a fast VIX spike
      roc_threshold   — spike threshold (% rise over roc_window days)

    Parameters:
    data: Price data DataFrame
    spy_data: SPY price series
    vix_data: VIX price series

    Returns:
    List of ExperimentResult objects
    """
    base_kwargs = dict(zscore_method='rolling', zscore_window=126, min_data_days=200)

    configs = [
        # Baseline — no VIX filter
        ExperimentConfig(name='no_vix_filter', **base_kwargs),

        # Moderate thresholds
        ExperimentConfig(name='vix_z1.0_2.0',
                         vix_config=VixRegimeConfig(elevated_zscore=1.0, crisis_zscore=2.0),
                         **base_kwargs),

        # Sensitive — react sooner, more time defensive
        ExperimentConfig(name='vix_z0.75_1.5',
                         vix_config=VixRegimeConfig(elevated_zscore=0.75, crisis_zscore=1.5),
                         **base_kwargs),

        # Conservative — only react to clear spikes
        ExperimentConfig(name='vix_z1.5_2.5',
                         vix_config=VixRegimeConfig(elevated_zscore=1.5, crisis_zscore=2.5),
                         **base_kwargs),

        # Shorter VIX baseline window (reacts faster to local spikes)
        ExperimentConfig(name='vix_z1.0_2.0_win30',
                         vix_config=VixRegimeConfig(elevated_zscore=1.0, crisis_zscore=2.0,
                                                    zscore_window=30),
                         **base_kwargs),

        # Add rate-of-change spike trigger on top of z-score thresholds
        ExperimentConfig(name='vix_z1.0_2.0_roc30',
                         vix_config=VixRegimeConfig(elevated_zscore=1.0, crisis_zscore=2.0,
                                                    use_roc_trigger=True, roc_threshold=0.30),
                         **base_kwargs),
    ]

    results = []
    for config in configs:
        vc = config.vix_config
        label = 'no filter' if vc is None else (
            f'elev={vc.elevated_zscore} crisis={vc.crisis_zscore} '
            f'win={vc.zscore_window}'
            + (f' +roc{vc.roc_threshold:.0%}' if vc.use_roc_trigger else '')
        )
        print(f"\n  Running {config.name}  ({label})...")
        result = run_experiment(config, data, spy_data, vix_data=vix_data, verbose=False)
        results.append(result)
        print(f"    CAGR: {result.cagr:.2%}  Sharpe: {result.sharpe_ratio:.2f}  "
              f"MaxDD: {result.max_drawdown:.2%}  Volatility: {result.volatility:.2%}")

    return results


def run_velocity_comparison(data: pd.DataFrame, spy_data: pd.Series, vix_data: pd.Series):
    """
    Compare velocity blend variants against the level-only baseline.

    All configs use the same core strategy (rolling z-score 126d, conservative VIX
    filter) so differences are attributable solely to the velocity blend.

    3 velocity windows × 3 weight splits = 9 configs + 1 baseline = 10 total.

    Tuning levers in VelocityConfig:
      velocity_window    — shorter reacts faster but noisier; longer smoother but laggier
      level_weight       — higher keeps momentum bias; lower leans toward mean reversion
      velocity_weight    — counterpart to level_weight
      min_level_threshold — how far a stock can fall before we ignore velocity bounces

    Parameters:
    data: Price data DataFrame
    spy_data: SPY price series
    vix_data: VIX price series

    Returns:
    List of ExperimentResult objects
    """
    # Use the conservative VIX filter as the common baseline so results are comparable
    # to the production config chosen from the VIX regime comparison.
    conservative_vix = VixRegimeConfig(elevated_zscore=1.5, crisis_zscore=2.5)
    base_kwargs = dict(
        zscore_method='rolling', zscore_window=126, min_data_days=200,
        vix_config=conservative_vix,
    )

    configs = [
        # Baseline — level only, no velocity
        ExperimentConfig(name='no_velocity', **base_kwargs),

        # --- velocity_window = 5 (fast, more reactive) ---
        ExperimentConfig(name='v5_L70V30',
                         velocity_config=VelocityConfig(velocity_window=5,  level_weight=0.7, velocity_weight=0.3),
                         **base_kwargs),
        ExperimentConfig(name='v5_L50V50',
                         velocity_config=VelocityConfig(velocity_window=5,  level_weight=0.5, velocity_weight=0.5),
                         **base_kwargs),
        ExperimentConfig(name='v5_L30V70',
                         velocity_config=VelocityConfig(velocity_window=5,  level_weight=0.3, velocity_weight=0.7),
                         **base_kwargs),

        # --- velocity_window = 10 (medium) ---
        ExperimentConfig(name='v10_L70V30',
                         velocity_config=VelocityConfig(velocity_window=10, level_weight=0.7, velocity_weight=0.3),
                         **base_kwargs),
        ExperimentConfig(name='v10_L50V50',
                         velocity_config=VelocityConfig(velocity_window=10, level_weight=0.5, velocity_weight=0.5),
                         **base_kwargs),
        ExperimentConfig(name='v10_L30V70',
                         velocity_config=VelocityConfig(velocity_window=10, level_weight=0.3, velocity_weight=0.7),
                         **base_kwargs),

        # --- velocity_window = 20 (slow, smoother signal) ---
        ExperimentConfig(name='v20_L70V30',
                         velocity_config=VelocityConfig(velocity_window=20, level_weight=0.7, velocity_weight=0.3),
                         **base_kwargs),
        ExperimentConfig(name='v20_L50V50',
                         velocity_config=VelocityConfig(velocity_window=20, level_weight=0.5, velocity_weight=0.5),
                         **base_kwargs),
        ExperimentConfig(name='v20_L30V70',
                         velocity_config=VelocityConfig(velocity_window=20, level_weight=0.3, velocity_weight=0.7),
                         **base_kwargs),
    ]

    results = []
    for config in configs:
        vel = config.velocity_config
        label = ('level only' if vel is None else
                 f'win={vel.velocity_window} L{vel.level_weight:.0%}/V{vel.velocity_weight:.0%}')
        print(f"\n  Running {config.name}  ({label})...")
        result = run_experiment(config, data, spy_data, vix_data=vix_data, verbose=False)
        results.append(result)
        print(f"    CAGR: {result.cagr:.2%}  Sharpe: {result.sharpe_ratio:.2f}  "
              f"MaxDD: {result.max_drawdown:.2%}  Volatility: {result.volatility:.2%}")

    return results


def main():
    """Main entry point for running experiments."""
    print("="*80)
    print("PORTFOLIO BACKTEST EXPERIMENTS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data once (reused across experiments)
    print("\n--- Loading Data ---")
    data, spy_data, vix_data = load_data(etfs)

    # --- VELOCITY BLEND COMPARISON ---
    print("\n--- Velocity Blend Comparison (10 configs — may take several minutes) ---")
    print("  Baseline = level only; remaining configs vary window and weight split")
    print("  Key question: does blending velocity improve CAGR / Sharpe vs level-only?\n")
    velocity_results = run_velocity_comparison(data, spy_data, vix_data)
    print_comparison(velocity_results)
    velocity_df = compare_experiments(velocity_results)
    velocity_df.to_csv('rsi_ma_velocity_comparison.csv', index=False)
    print(f"\nVelocity blend comparison exported to: rsi_ma_velocity_comparison.csv")

    # --- VIX REGIME COMPARISON ---
    print("\n--- VIX Regime Filter Comparison (6 configs — may take a few minutes) ---")
    print("  Baseline = no VIX filter; remaining configs vary regime thresholds")
    print("  Key question: does defensive switching improve Sharpe / reduce drawdown?\n")
    vix_results = run_vix_regime_comparison(data, spy_data, vix_data)
    print_comparison(vix_results)
    vix_df = compare_experiments(vix_results)
    vix_df.to_csv('rsi_ma_vix_regime_comparison.csv', index=False)
    print(f"\nVIX regime comparison exported to: rsi_ma_vix_regime_comparison.csv")

    # --- Z-SCORE METHOD COMPARISON ---
    print("\n--- Z-Score Method Comparison (5 configs — may take a few minutes) ---")
    print("  Note: CAGR and Sharpe are annualized; use these to compare, not Total Return")
    print("        (rolling configs have shorter effective test periods due to warmup)\n")
    zscore_results = run_zscore_method_comparison(data, spy_data, vix_data)
    print_comparison(zscore_results)
    zscore_df = compare_experiments(zscore_results)
    zscore_df.to_csv('rsi_ma_zscore_comparison.csv', index=False)
    print(f"\nZ-score comparison exported to: rsi_ma_zscore_comparison.csv")

    # Run weekly vs biweekly comparison
    print("\n--- Running Weekly vs Biweekly Comparison ---")
    results = run_rebalance_frequency_comparison(data, spy_data, vix_data)

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
