import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def backtest_momentum_strategy(data, lookback_months=12, min_momentum_months=6, top_n=4):
    """
    Backtest momentum strategy with monthly rebalancing.
    
    Parameters:
    data (DataFrame): Historical price data of ETFs.
    lookback_months (int): Months to look back for momentum calculation.
    min_momentum_months (int): Minimum months for momentum calculation.
    top_n (int): Number of top ETFs to select.
    
    Returns:
    DataFrame: Portfolio performance with monthly returns.
    """
    portfolio_returns = []
    selected_etfs_history = []
    
    # Start backtesting after we have enough data for momentum calculation
    start_idx = max(lookback_months, min_momentum_months)
    
    for i in range(start_idx, len(data) - 1):  # -1 because we need next month's data for returns
        current_date = data.index[i]
        next_date = data.index[i + 1]
        
        # Calculate momentum scores for portfolio selection (using data up to current month)
        prev_data = data.iloc[:i]
        
        if len(prev_data) >= lookback_months:
            # Calculate 6 and 12 month momentum
            momentum_6m = prev_data.pct_change(periods=min_momentum_months).iloc[-1]
            momentum_12m = prev_data.pct_change(periods=lookback_months).iloc[-1]
            
            # Combine into blended score
            composite_momentum = (momentum_6m + momentum_12m) / 2
            
            # Apply absolute momentum filter (only positive momentum)
            filtered = composite_momentum[composite_momentum > 0]
            
            if len(filtered) >= top_n:
                # Select top N ETFs
                top_etfs = filtered.sort_values(ascending=False).head(top_n)
                selected_etfs = top_etfs.index.tolist()
                weights = [1.0/top_n] * top_n  # Equal weight
            else:
                # If not enough positive momentum ETFs, use cash proxy (SHY)
                selected_etfs = ['SHY'] if 'SHY' in data.columns else []
                weights = [1.0] if selected_etfs else []
            
            # Calculate portfolio return for NEXT month (i+1 price / i price - 1)
            if selected_etfs:
                monthly_returns = data.iloc[i + 1] / data.iloc[i] - 1  # Next month return
                portfolio_return = sum(w * monthly_returns[etf] for w, etf in zip(weights, selected_etfs) if etf in monthly_returns.index)
            else:
                portfolio_return = 0
                
            portfolio_returns.append({
                'Date': next_date,  # Use next month's date for the return period
                'Portfolio_Return': portfolio_return,
                'Selected_ETFs': selected_etfs,
                'Weights': weights,
                'All_Returns': monthly_returns,
                'Selection_Date': current_date  # Track when selection was made
            })
            
            selected_etfs_history.append({
                'Date': current_date,
                'ETFs': selected_etfs,
                'Momentum_Scores': top_etfs.to_dict() if len(filtered) >= top_n else {},
                'Return_Period': f"{current_date.strftime('%Y-%m')} to {next_date.strftime('%Y-%m')}"
            })
    
    return pd.DataFrame(portfolio_returns), selected_etfs_history

def calculate_performance_metrics(returns_series):
    """
    Calculate key performance metrics for the strategy.
    
    Parameters:
    returns_series (Series): Monthly returns series.
    
    Returns:
    dict: Performance metrics.
    """
    cumulative_return = (1 + returns_series).prod() - 1
    annualized_return = (1 + cumulative_return) ** (12 / len(returns_series)) - 1
    volatility = returns_series.std() * np.sqrt(12)
    risk_free_rate = 0.02  # Assume 2% annual risk-free rate
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    max_drawdown = ((1 + returns_series).cumprod() / (1 + returns_series).cumprod().expanding().max() - 1).min()
    
    return {
        'Total Return': f"{cumulative_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Number of Months': len(returns_series)
    }

def calculate_individual_stock_performance(data):
    """
    Calculate performance metrics for each individual stock/ETF.
    
    Parameters:
    data (DataFrame): Historical price data of ETFs.
    
    Returns:
    DataFrame: Performance metrics for each stock.
    """
    performance_data = []
    
    for stock in data.columns:
        # Calculate monthly returns
        monthly_returns = data[stock].pct_change().dropna()
        
        if len(monthly_returns) > 0:
            # Calculate metrics
            cumulative_return = (1 + monthly_returns).prod() - 1
            cagr = (1 + cumulative_return) ** (12 / len(monthly_returns)) - 1
            volatility = monthly_returns.std() * np.sqrt(12)
            risk_free_rate = 0.02  # Assume 2% annual risk-free rate
            sharpe_ratio = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            cumulative_wealth = (1 + monthly_returns).cumprod()
            running_max = cumulative_wealth.expanding().max()
            drawdown = (cumulative_wealth / running_max - 1)
            max_drawdown = drawdown.min()
            
            performance_data.append({
                'Stock': stock,
                'Total Return': f"{cumulative_return:.2%}",
                'CAGR': f"{cagr:.2%}",
                'Volatility': f"{volatility:.2%}",
                'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                'Max Drawdown': f"{max_drawdown:.2%}",
                'Number of Months': len(monthly_returns)
            })
    
    return pd.DataFrame(performance_data)

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
	'XAR',  # Aerospace & Defense ETF
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
    'SCZ',  # International small cap
	'IBIT', # Bitcoin Trust
	#'SVXY', # Volatility Short
	'HYG',  # High Yield Bond ETF
]

# Download historical data for the ETFs
data = yf.download(etfs, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'), interval='1mo')["Close"]

# Clean the data
one_year_ago = data.index[-1] - pd.DateOffset(months=12)
recent_data = data.loc[one_year_ago:]
data = data.dropna(axis=1, thresh=recent_data.shape[0])

# Calculate 6 and 12 month returns
momentum_6m = data.pct_change(periods=6).iloc[-1]
momentum_12m = data.pct_change(periods=12).iloc[-1]

# Combine into a blended score (equally weighted)
composite_momentum = (momentum_6m + momentum_12m) / 2

# Apply absolute momentum filter (only keep ETFS with positive composite momentum)
filtered = composite_momentum[composite_momentum > 0]

# export the total results to a CSV file
data.pct_change(periods=12).to_csv('momentum_etfs.csv')
data.to_csv('momentum_etfs_prices.csv')

# Select top 4 ETFs by composite score
top4_composite = filtered.sort_values(ascending=False).head(4)

# Format output
top4_composite.name = "Blended 6/12 month return"
top4_composite.index.name = "ETF"
top4_composite = top4_composite.to_frame()
top4_composite["Weight"] = 0.25 # Equal weight for each of the top 4 ETFs

# plotting
plot_momentum_portolio(data, top4_composite.index.tolist())

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
    plt.title('Momentum Strategy Backtest Results')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot 2: Monthly returns
    plt.subplot(2, 1, 2)
    plt.bar(backtest_results['Date'], returns * 100, alpha=0.7, width=20)
    plt.title('Monthly Returns (%)')
    plt.ylabel('Return (%)')
    plt.xlabel('Date')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print performance summary
    print("\n=== MOMENTUM STRATEGY BACKTEST RESULTS ===")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value}")
    
    # Show recent portfolio selections
    print("\n=== RECENT PORTFOLIO SELECTIONS ===")
    for i, selection in enumerate(etf_history[-12:]):  # Last 12 months
        date_str = selection['Date'].strftime('%Y-%m')
        etfs = ', '.join(selection['ETFs'])
        print(f"{date_str}: {etfs}")
    
    # Export backtest results
    backtest_results.to_csv('momentum_backtest_results.csv', index=False)
    print(f"\nBacktest results exported to momentum_backtest_results.csv")
else:
    print("Insufficient data for backtesting")

# Calculate and export individual stock performance
print("\n=== CALCULATING INDIVIDUAL STOCK PERFORMANCE ===")
individual_performance = calculate_individual_stock_performance(data)
individual_performance = individual_performance.sort_values('CAGR', ascending=False)
individual_performance.to_csv('individual_stock_performance.csv', index=False)
print(f"Individual stock performance exported to individual_stock_performance.csv")

# Display top performers
print("\n=== TOP PERFORMING STOCKS (by CAGR) ===")
print(individual_performance.head(10))

# Original single-point analysis for comparison
print("\n=== CURRENT PORTFOLIO (Single Point Analysis) ===")

# View results
print(top4_composite)