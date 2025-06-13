import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_rsi(prices, window=14):
    """
    Calculate RSI (Relative Strength Index) using Wilder's smoothing method.
    
    Parameters:
    prices (Series): Price series
    window (int): RSI calculation window (default 14)
    
    Returns:
    Series: RSI values
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use Wilder's smoothing (exponential moving average with alpha = 1/window)
    alpha = 1.0 / window
    gain_avg = gain.ewm(alpha=alpha, adjust=False).mean()
    loss_avg = loss.ewm(alpha=alpha, adjust=False).mean()
    
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ma_difference(prices, short_window=50, long_window=200):
    """
    Calculate difference between short and long moving averages.
    
    Parameters:
    prices (Series): Price series
    short_window (int): Short MA window (default 50)
    long_window (int): Long MA window (default 200)
    
    Returns:
    Series: MA difference (short - long)
    """
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    return short_ma - long_ma

def calculate_ma_derivative(ma_diff, derivative_window=5):
    """
    Calculate first derivative of MA difference.
    
    Parameters:
    ma_diff (Series): Moving average difference series
    derivative_window (int): Window for derivative calculation (default 5)
    
    Returns:
    Series: First derivative of MA difference
    """
    return ma_diff.diff(derivative_window)

def calculate_rolling_zscore(data, window=252):
    """
    Calculate rolling z-score for a DataFrame or Series.
    
    Parameters:
    data (DataFrame/Series): Input data
    window (int): Rolling window for z-score calculation (default 252)
    
    Returns:
    DataFrame/Series: Rolling z-scores
    """
    if isinstance(data, pd.DataFrame):
        return data.apply(lambda x: (x - x.rolling(window).mean()) / x.rolling(window).std())
    else:
        return (data - data.rolling(window).mean()) / data.rolling(window).std()

def calculate_cross_sectional_zscore(data):
    """
    Calculate cross-sectional z-score (across stocks for each date).
    
    Parameters:
    data (DataFrame): Data with dates as index, stocks as columns
    
    Returns:
    DataFrame: Cross-sectional z-scores
    """
    return data.apply(lambda row: (row - row.mean()) / row.std(), axis=1)

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

def calculate_performance_metrics(returns_series, is_daily=True):
    """
    Calculate key performance metrics for the strategy.
    
    Parameters:
    returns_series (Series): Returns series (daily or monthly).
    is_daily (bool): Whether returns are daily (True) or monthly (False).
    
    Returns:
    dict: Performance metrics.
    """
    cumulative_return = (1 + returns_series).prod() - 1
    
    if is_daily:
        # For daily returns, annualize using actual number of trading days
        trading_days_per_year = 252
        years = len(returns_series) / trading_days_per_year
        annualized_return = (1 + cumulative_return) ** (1 / years) - 1
        volatility = returns_series.std() * np.sqrt(trading_days_per_year)
        period_label = f"{len(returns_series)} days"
    else:
        # For monthly returns (legacy code)
        annualized_return = (1 + cumulative_return) ** (12 / len(returns_series)) - 1
        volatility = returns_series.std() * np.sqrt(12)
        period_label = f"{len(returns_series)} months"
    
    risk_free_rate = 0.045  # choose a number corresponding roughly to the 10 year treasury yield
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    max_drawdown = ((1 + returns_series).cumprod() / (1 + returns_series).cumprod().expanding().max() - 1).min()
    
    return {
        'Total Return': f"{cumulative_return:.2%}",
        'CAGR': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Period': period_label
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
        # Calculate daily returns
        daily_returns = data[stock].pct_change().dropna()
        
        if len(daily_returns) > 0:
            # Calculate metrics (daily data)
            cumulative_return = (1 + daily_returns).prod() - 1
            trading_days_per_year = 252
            years = len(daily_returns) / trading_days_per_year
            cagr = (1 + cumulative_return) ** (1 / years) - 1
            volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
            risk_free_rate = 0.045  # choose a number corresponding roughly to the 10 year treasury yield
            sharpe_ratio = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            cumulative_wealth = (1 + daily_returns).cumprod()
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
                'Number of Days': len(daily_returns)
            })
    
    return pd.DataFrame(performance_data)

def calculate_composite_scores(data, rsi_window=14, ma_short=50, ma_long=200, derivative_window=5, zscore_window=252):
    """
    Calculate composite scores for RSI/MA strategy.
    
    Parameters:
    data (DataFrame): Historical price data
    rsi_window (int): RSI calculation window
    ma_short (int): Short MA window
    ma_long (int): Long MA window  
    derivative_window (int): Window for derivative calculation
    zscore_window (int): Window for z-score calculation
    
    Returns:
    DataFrame: Composite scores for each stock and date
    """
    # Initialize containers for each measure
    rsi_scores = pd.DataFrame(index=data.index, columns=data.columns)
    ma_diff_scores = pd.DataFrame(index=data.index, columns=data.columns)
    ma_deriv_scores = pd.DataFrame(index=data.index, columns=data.columns)
    
    # Calculate measures for each stock
    for stock in data.columns:
        if data[stock].notna().sum() > max(ma_long, zscore_window):  # Ensure sufficient data
            # RSI calculation
            rsi = calculate_rsi(data[stock], rsi_window)
            rsi_scores[stock] = -rsi   # Negative so that the lowest RSI/most oversold gives positive z-score
            
            # MA difference calculation
            ma_diff = calculate_ma_difference(data[stock], ma_short, ma_long)
            ma_diff_scores[stock] = ma_diff  
            
            # MA derivative calculation
            ma_deriv = calculate_ma_derivative(ma_diff, derivative_window)
            ma_deriv_scores[stock] = ma_deriv   
    
    # Save underlying measures before standardization
    underlying_measures = []
    
    for date in data.index:
        for stock in data.columns:
            # RSI measure
            if not pd.isna(rsi_scores.loc[date, stock]):
                underlying_measures.append({
                    'Date': date,
                    'Symbol': stock,
                    'Measure': 'RSI',
                    'Value': rsi_scores.loc[date, stock]
                })
            
            # MA Difference measure (before negation)
            if not pd.isna(ma_diff_scores.loc[date, stock]):
                underlying_measures.append({
                    'Date': date,
                    'Symbol': stock,
                    'Measure': 'MA_Difference_50_200',
                    'Value': -ma_diff_scores.loc[date, stock]  # Store original value (before negation)
                })
            
            # MA Derivative measure
            if not pd.isna(ma_deriv_scores.loc[date, stock]):
                underlying_measures.append({
                    'Date': date,
                    'Symbol': stock,
                    'Measure': 'MA_Derivative',
                    'Value': ma_deriv_scores.loc[date, stock]
                })
    
    # Convert to DataFrame and save
    underlying_df = pd.DataFrame(underlying_measures)
    if not underlying_df.empty:
        underlying_df.to_csv('rsi_ma_underlying_measures.csv', index=False)
        print(f"Underlying measures saved to rsi_ma_underlying_measures.csv")
    
    # Convert to cross-sectional z-scores (across stocks for each date)
    rsi_z = calculate_cross_sectional_zscore(rsi_scores)
    ma_diff_z = calculate_cross_sectional_zscore(ma_diff_scores)
    ma_deriv_z = calculate_cross_sectional_zscore(ma_deriv_scores)
    
    # Calculate composite scores (average of three z-scores)
    composite_scores = (rsi_z + ma_diff_z + ma_deriv_z) / 3
    
    return composite_scores, rsi_z, ma_diff_z, ma_deriv_z

def select_top_stocks_biweekly(composite_scores, data, top_n=4, min_data_days=200, hold_days=14):
    """
    Select top N stocks with biweekly rebalancing and minimum hold period.
    
    Parameters:
    composite_scores (DataFrame): Composite scores for each stock and date
    data (DataFrame): Price data for return calculations
    top_n (int): Number of top stocks to select
    min_data_days (int): Minimum number of data points required
    hold_days (int): Minimum hold period in days (default 14)
    
    Returns:
    DataFrame: Portfolio performance with biweekly rebalancing
    """
    portfolio_returns = []
    rebalance_history = []
    
    # Start after we have sufficient data
    start_idx = min_data_days
    dates = composite_scores.index[start_idx:]
    
    current_portfolio = []
    last_rebalance_date = None
    
    for i, date in enumerate(dates):
        # Check if it's time to rebalance (every 14 days or first selection)
        if (last_rebalance_date is None or 
            (date - last_rebalance_date).days >= hold_days):
            
            # Get valid scores for this date
            valid_scores = composite_scores.loc[date].dropna()
            
            # Filter for stocks with sufficient data history
            valid_stocks = []
            for stock in valid_scores.index:
                stock_data = composite_scores[stock].loc[:date]
                if stock_data.notna().sum() >= min_data_days:
                    valid_stocks.append(stock)
            
            if len(valid_stocks) >= top_n:
                # Select top N based on highest composite scores
                top_stocks = valid_scores[valid_stocks].sort_values(ascending=False).head(top_n)
                new_portfolio = top_stocks.index.tolist()
                
                # Record rebalancing
                rebalance_history.append({
                    'Date': date,
                    'Selected_Stocks': new_portfolio,
                    'Scores': top_stocks.to_dict()
                })
                
                current_portfolio = new_portfolio
                last_rebalance_date = date
                
                print(f"Rebalanced on {date.strftime('%Y-%m-%d')}: {', '.join(new_portfolio)}")
        
        # Calculate portfolio return for next day (if we have a portfolio and next day data)
        if current_portfolio and i < len(dates) - 1:
            current_date = date
            next_date = dates[i + 1]
            
            # Check if both dates exist in price data
            if current_date in data.index and next_date in data.index:
                # Calculate equal-weighted return for next day
                daily_returns = []
                for stock in current_portfolio:
                    if stock in data.columns:
                        stock_return = (data.loc[next_date, stock] / data.loc[current_date, stock]) - 1
                        daily_returns.append(stock_return)
                
                if daily_returns:
                    portfolio_return = sum(daily_returns) / len(daily_returns)  # Equal weight
                    
                    portfolio_returns.append({
                        'Date': next_date,  # Return is for next day
                        'Portfolio_Return': portfolio_return,
                        'Holdings': current_portfolio.copy(),
                        'Selection_Date': last_rebalance_date
                    })
    
    return pd.DataFrame(portfolio_returns), rebalance_history

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
	# Individual stocks with historically high Sharpe ratios
	'AAPL', # Apple Inc.
	'MSFT', # Microsoft Corporation
	'GOOGL',# Alphabet Inc.
	'NVDA', # NVIDIA Corporation
	'AMZN', # Amazon.com Inc.
	'TSLA', # Tesla Inc.
	'META', # Meta Platforms Inc.
	'BRK-B',# Berkshire Hathaway Inc.
    # Up and comers in the 11-20 market cap range with high CAGR
    'V',    # Visa (financial)
    'LLY',  # Eli Lilly (pharma)
]

# Download historical data for the ETFs (daily data)
data = yf.download(etfs, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'), interval='1d')["Close"]

# Clean the data - remove stocks with insufficient recent data
one_year_ago = data.index[-1] - pd.DateOffset(months=12)
recent_data = data.loc[one_year_ago:]
data = data.dropna(axis=1, thresh=recent_data.shape[0])

print(f"Data shape after cleaning: {data.shape}")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# Calculate composite scores using RSI/MA strategy
print("\n=== CALCULATING RSI/MA COMPOSITE SCORES ===")
composite_scores, rsi_z, ma_diff_z, ma_deriv_z = calculate_composite_scores(
    data, 
    rsi_window=14, 
    ma_short=50, 
    ma_long=200, 
    derivative_window=5,
    zscore_window=252
)

# Run biweekly rebalancing strategy
print("=== RUNNING BIWEEKLY REBALANCING STRATEGY ===")
portfolio_performance, rebalance_history = select_top_stocks_biweekly(
    composite_scores, data, top_n=4, min_data_days=200, hold_days=14
)

if not portfolio_performance.empty:
    # Calculate performance metrics
    returns = portfolio_performance['Portfolio_Return']
    performance_metrics = calculate_performance_metrics(returns, is_daily=True)
    
    # Create cumulative return series for plotting
    cumulative_returns = (1 + returns).cumprod()
    
    # Plot portfolio performance
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Cumulative returns
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_performance['Date'], cumulative_returns, label='RSI/MA Portfolio', linewidth=2)
    plt.title('RSI/MA Strategy Backtest Results (Biweekly Rebalancing)')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot 2: Daily returns
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_performance['Date'], returns * 100, alpha=0.7)
    plt.title('Daily Returns (%)')
    plt.ylabel('Return (%)')
    plt.xlabel('Date')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print performance summary
    print("\n=== RSI/MA STRATEGY BACKTEST RESULTS ===")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value}")
    
    # Show recent rebalancing history
    print(f"\n=== REBALANCING HISTORY (Last 10 rebalances) ===")
    for rebalance in rebalance_history[-10:]:
        date_str = rebalance['Date'].strftime('%Y-%m-%d')
        stocks = ', '.join(rebalance['Selected_Stocks'])
        print(f"{date_str}: {stocks}")
    
    # Export results
    portfolio_performance.to_csv('rsi_ma_portfolio_performance.csv', index=False)
    composite_scores.to_csv('rsi_ma_composite_scores.csv')
    
    # Export rebalancing history
    rebalance_df = pd.DataFrame(rebalance_history)
    rebalance_df.to_csv('rsi_ma_rebalance_history.csv', index=False)
    
    print(f"\nResults exported:")
    print(f"- Portfolio performance: rsi_ma_portfolio_performance.csv")
    print(f"- Composite scores: rsi_ma_composite_scores.csv")
    print(f"- Rebalancing history: rsi_ma_rebalance_history.csv")
    
    # Show current holdings
    if len(rebalance_history) > 0:
        latest_rebalance = rebalance_history[-1]
        print(f"\n=== CURRENT HOLDINGS (Last rebalance: {latest_rebalance['Date'].strftime('%Y-%m-%d')}) ===")
        for i, (stock, score) in enumerate(latest_rebalance['Scores'].items()):
            print(f"{i+1}. {stock}: {score:.3f}")
        
        # Plot recent performance of current holdings
        plot_momentum_portolio(data, latest_rebalance['Selected_Stocks'])
else:
    print("Insufficient data for backtesting")

# Calculate and export individual stock performance  
print("\n=== CALCULATING INDIVIDUAL STOCK PERFORMANCE ===")
individual_performance = calculate_individual_stock_performance(data)
individual_performance = individual_performance.sort_values('CAGR', ascending=False)
individual_performance.to_csv('rsi_ma_individual_stock_performance.csv', index=False)
print(f"Individual stock performance exported to rsi_ma_individual_stock_performance.csv")

# Display top performers
print("\n=== TOP PERFORMING STOCKS (by CAGR) ===")
print(individual_performance.head(10))

print("\n=== RSI/MA STRATEGY SETUP COMPLETE ===")
print("Files created:")
print("- rsi_ma_composite_scores.csv: Daily composite scores for all stocks")
print("- rsi_ma_stock_selections.csv: Top stock selections by date")  
print("- rsi_ma_individual_stock_performance.csv: Individual stock performance metrics")