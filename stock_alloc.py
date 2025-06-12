import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def backtest_momentum_strategy(data, lookback_weeks_long=26, lookback_weeks_mid=12, lookback_weeks_short=6, top_n=3, exit_n=5, rebalance_weeks=4):
    """
    Backtest momentum strategy with weekly data and separate entry/exit signals.
    
    Parameters:
    data (DataFrame): Historical weekly price data of ETFs.
    lookback_weeks_long (int): Long-term momentum lookback (26 weeks).
    lookback_weeks_mid (int): Mid-term momentum lookback (13 weeks).
    lookback_weeks_short (int): Short-term momentum lookback (4 weeks).
    top_n (int): Number of top ETFs to select for entry.
    exit_n (int): ETFs must stay in top N to avoid exit (7).
    rebalance_weeks (int): Rebalancing frequency in weeks (4).
    
    Returns:
    DataFrame: Portfolio performance with weekly returns.
    """
    portfolio_returns = []
    selected_etfs_history = []
    current_holdings = {}  # Track current positions
    
    # Calculate 200-day (40-week) moving average for exit signals
    ma_200d = data.rolling(window=40).mean()
    
    # Start backtesting after we have enough data for momentum calculation
    start_idx = max(lookback_weeks_long, 40)  # Need 40 weeks for 200-day MA
    
    for i in range(start_idx, len(data) - 1):  # -1 because we need next week's data for returns
        current_date = data.index[i]
        next_date = data.index[i + 1]
        
        # Calculate momentum scores for ranking (using data up to current week)
        prev_data = data.iloc[:i+1]
        
        if len(prev_data) >= lookback_weeks_long:
            # Skip if we don't have enough valid ETFs with current price data
            current_prices = data.iloc[i]
            valid_current_etfs = current_prices.dropna()
            if len(valid_current_etfs) < 3:  # Need at least 3 ETFs with valid current data
                continue
                
            # Skip if we don't have next week's data for return calculation
            if i + 1 >= len(data):
                continue
            next_prices = data.iloc[i + 1]
            valid_next_etfs = next_prices.dropna()
            if len(valid_next_etfs) < 3:  # Need at least 3 ETFs with valid next week data
                continue
            # Calculate 4, 13 and 26 week momentum only for ETFs with current data
            momentum_4w = prev_data.pct_change(periods=lookback_weeks_short).iloc[-1]
            momentum_13w = prev_data.pct_change(periods=lookback_weeks_mid).iloc[-1]
            momentum_26w = prev_data.pct_change(periods=lookback_weeks_long).iloc[-1]
            
            # Only consider ETFs that have current price data
            momentum_4w = momentum_4w[valid_current_etfs.index]
            momentum_13w = momentum_13w[valid_current_etfs.index]
            momentum_26w = momentum_26w[valid_current_etfs.index]
            
            # Combine into blended score - use available data, require at least 2 valid periods
            momentum_count = pd.Series(index=momentum_4w.index, dtype=int)
            momentum_sum = pd.Series(index=momentum_4w.index, dtype=float)
            
            for mom in [momentum_4w, momentum_13w, momentum_26w]:
                valid = ~mom.isna()
                momentum_count[valid] = momentum_count[valid].fillna(0) + 1
                momentum_sum[valid] = momentum_sum[valid].fillna(0) + mom[valid]
            
            # Only calculate composite momentum for ETFs with at least 2 valid periods
            composite_momentum = pd.Series(index=momentum_4w.index, dtype=float)
            valid_etfs = momentum_count >= 2
            composite_momentum[valid_etfs] = momentum_sum[valid_etfs] / momentum_count[valid_etfs]
            
            # Rank all ETFs by composite momentum (only valid values)
            momentum_ranking = composite_momentum.rank(ascending=False, na_option='bottom')
            
            # Get current prices and 200-day MA for exit signals
            current_prices = data.iloc[i]
            current_ma200 = ma_200d.iloc[i]
            
            # Process exits first
            exits = []
            for etf in current_holdings.copy():
                exit_signal = False
                
                # Exit if ETF is not in current valid data
                if etf not in valid_current_etfs.index:
                    exit_signal = True
                    exits.append(f"{etf}(no_data)")
                
                # Exit if ETF is not in momentum calculations
                elif etf not in composite_momentum.index or pd.isna(composite_momentum[etf]):
                    exit_signal = True
                    exits.append(f"{etf}(no_momentum)")
                
                # Exit if falls out of top 7 momentum ranking
                elif momentum_ranking[etf] > exit_n:
                    exit_signal = True
                    exits.append(f"{etf}(rank>{exit_n})")
                
                # Exit if absolute momentum turns negative
                elif composite_momentum[etf] <= 0:
                    exit_signal = True
                    exits.append(f"{etf}(neg_mom)")
                
                # Exit if price falls below 200-day MA
                elif pd.isna(current_ma200[etf]) or current_prices[etf] < current_ma200[etf]:
                    exit_signal = True
                    exits.append(f"{etf}(below_MA200)")
                
                if exit_signal:
                    del current_holdings[etf]
            
            # Process entries (only on rebalancing weeks)
            if i % rebalance_weeks == 0 or len(current_holdings) < top_n:
                # Apply absolute momentum filter for entry candidates (exclude NaN values)
                entry_candidates = composite_momentum.dropna()
                entry_candidates = entry_candidates[entry_candidates > 0]
                
                if len(entry_candidates) > 0:
                    # Select top N ETFs by composite momentum for entry
                    top_etfs = entry_candidates.sort_values(ascending=False).head(top_n)
                    
                    # Add new positions (up to top_n total)
                    for etf in top_etfs.index:
                        if etf not in current_holdings and len(current_holdings) < top_n:
                            current_holdings[etf] = 1.0 / top_n  # Equal weight
                    
                    # Rebalance existing positions to equal weight
                    if len(current_holdings) > 0:
                        equal_weight = 1.0 / len(current_holdings)
                        for etf in current_holdings:
                            current_holdings[etf] = equal_weight
                
                # If no positive momentum ETFs, move to cash (SHY)
                if len(current_holdings) == 0 and 'SHY' in data.columns:
                    current_holdings['SHY'] = 1.0
            
            # Calculate portfolio return for NEXT week
            if current_holdings:
                weekly_returns = data.iloc[i + 1] / data.iloc[i] - 1
                portfolio_return = 0
                total_weight = 0
                
                for etf, weight in current_holdings.items():
                    if etf in weekly_returns.index and not pd.isna(weekly_returns[etf]):
                        portfolio_return += weight * weekly_returns[etf]
                        total_weight += weight
                
                # If we couldn't calculate returns for all holdings, normalize by actual weight used
                if total_weight > 0 and total_weight < 1.0:
                    portfolio_return = portfolio_return / total_weight
            else:
                portfolio_return = 0
                
            portfolio_returns.append({
                'Date': next_date,
                'Portfolio_Return': portfolio_return,
                'Holdings': current_holdings.copy(),
                'Returns': weekly_returns.copy(),
                'Exits': exits if exits else [],
                'Selection_Date': current_date
            })
            
            # Record selection history on rebalancing weeks
            if i % rebalance_weeks == 0:
                selected_etfs_history.append({
                    'Date': current_date,
                    'ETFs': list(current_holdings.keys()),
                    'Weights': current_holdings.copy(),
                    'Momentum_Scores': composite_momentum.to_dict(),
                    'Exits': exits if exits else []
                })
    
    return pd.DataFrame(portfolio_returns), selected_etfs_history

def calculate_performance_metrics(returns_series):
    """
    Calculate key performance metrics for the strategy.
    
    Parameters:
    returns_series (Series): Weekly returns series.
    
    Returns:
    dict: Performance metrics.
    """
    cumulative_return = (1 + returns_series).prod() - 1
    annualized_return = (1 + cumulative_return) ** (52 / len(returns_series)) - 1
    volatility = returns_series.std() * np.sqrt(52)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    max_drawdown = ((1 + returns_series).cumprod() / (1 + returns_series).cumprod().expanding().max() - 1).min()
    
    return {
        'Total Return': f"{cumulative_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Number of Weeks': len(returns_series)
    }

def plot_momentum_portolio(data, top_etfs):
    """
    Plot the performance of the selected ETFs in the portfolio.
    
    Parameters:
    data (DataFrame): Historical price data of ETFs.
    top_etfs (list): List of top ETFs to plot.
    """
    plt.figure(figsize=(14, 7))
    for etf in top_etfs:
        plt.plot(data[etf], label=etf)
    
    plt.title('Top ETFs Performance')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid()
    plt.show()

etfs = [
 	'XLV',  # Health Care Select Sector SPDR Fund
	'XLP',  # Consumer Staples Select Sector SPDR Fund
	'XLRE', # Real Estate Select Sector SPDR Fund
	'XLU',  # Utilities Select Sector SPDR Fund
	'XLC',  # Communication Services Select Sector SPDR Fund
	'XBI',  # Biotech SPDR ETF
	'ITA',  # iShares U.S. Aerospace & Defense ETF
	#'XOP',  # Oil & Gas Exploration & Production ETF
	'XME',  # Metals & Mining ETF
	'KBE',  # Bank ETF
	'XHB',  # Homebuilders ETF
	'VUG',  # Vanguard Growth ETF
	#'DBC',  # Commodities ETF
	'IAU',  # Gold ETF
	'TLT',  # Long-Term Treasury ETF
	'SHY',  # Short-Term Treasury ETF
	'USMV', # Minimum Volatility ETF
	'VWO',  # Emerging Markets ETF
    'SCZ',  # International Small Cap
	'IBIT', # Bitcoin Trust
	#'SVXY', # Volatility Short
	'HYG',  # High Yield Bond ETF
]

# Download historical data for the ETFs (weekly)
data = yf.download(etfs, start='2010-01-04', end=datetime.now().strftime('%Y-%m-%d'), interval='1wk')["Close"]

# Export data
data.to_csv('all_data.csv')

# Clean the data (require data for at least 52 weeks)
one_year_ago = data.index[-1] - pd.DateOffset(weeks=52)
recent_data = data.loc[one_year_ago:]
data = data.dropna(axis=1, thresh=recent_data.shape[0])

# Calculate 4, 13 and 26 week returns
momentum_4w = data.pct_change(periods=4).iloc[-1]
momentum_13w = data.pct_change(periods=13).iloc[-1]
momentum_26w = data.pct_change(periods=26).iloc[-1]

# Combine into a blended score (equally weighted)
composite_momentum = (momentum_4w + momentum_13w + momentum_26w) / 3

# Apply absolute momentum filter (only keep ETFS with positive composite momentum)
filtered = composite_momentum[composite_momentum > 0]

# export the total results to a CSV file
data.pct_change(periods=26).to_csv('momentum_etfs.csv')
data.to_csv('momentum_etfs_prices.csv')

# Select top 4 ETFs by composite score
topn_composite = filtered.sort_values(ascending=False).head(3)

# Format output
topn_composite.name = "Blended 4/13/26 week return"
topn_composite.index.name = "ETF"
topn_composite = topn_composite.to_frame()
topn_composite["Weight"] = 1/3 # Equal weight for each of the top 6 ETFs

# plotting
plot_momentum_portolio(data, topn_composite.index.tolist())

# Run backtest
backtest_results, etf_history = backtest_momentum_strategy(data)

if not backtest_results.empty:
    # Calculate performance metrics
    returns = backtest_results['Portfolio_Return']
    performance_metrics = calculate_performance_metrics(returns)
    
    # Create cumulative return series for plotting
    cumulative_returns = (1 + returns).cumprod()
    
    # Plot portfolio performance
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Cumulative returns
    plt.subplot(2, 1, 1)
    plt.plot(backtest_results['Date'], cumulative_returns, label='Momentum Portfolio', linewidth=2)
    plt.title('Weekly Momentum Strategy Backtest Results')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot 2: Weekly returns
    plt.subplot(2, 1, 2)
    plt.bar(backtest_results['Date'], returns * 100, alpha=0.7, width=5)
    plt.title('Weekly Returns (%)')
    plt.ylabel('Return (%)')
    plt.xlabel('Date')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print performance summary
    print("\n=== WEEKLY MOMENTUM STRATEGY BACKTEST RESULTS ===")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value}")
    
    # Show recent portfolio selections
    print("\n=== RECENT PORTFOLIO SELECTIONS ===")
    for i, selection in enumerate(etf_history[-13:]):  # Last 13 rebalancing periods (quarterly)
        date_str = selection['Date'].strftime('%Y-%m-%d')
        etfs = ', '.join(selection['ETFs'])
        exits = ', '.join(selection['Exits']) if selection['Exits'] else 'None'
        print(f"{date_str}: {etfs} | Exits: {exits}")
    
    # Export backtest results
    backtest_results.to_csv('momentum_backtest_results.csv', index=False)
    print(f"\nBacktest results exported to momentum_backtest_results.csv")
else:
    print("Insufficient data for backtesting")

# Current portfolio analysis (weekly momentum)
print("\n=== CURRENT PORTFOLIO (Weekly Momentum Analysis) ===")

# View results
print(topn_composite)