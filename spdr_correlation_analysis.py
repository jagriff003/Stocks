import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def get_spdr_sector_etfs():
    """
    Get list of SPDR sector ETFs with their descriptions.
    
    Returns:
    dict: Dictionary mapping ETF symbols to descriptions.
    """
    spdr_sectors = {
        'XLV': 'Health Care Select Sector SPDR Fund',
        'XLP': 'Consumer Staples Select Sector SPDR Fund',
        'XLRE': 'Real Estate Select Sector SPDR Fund',
        'XLU': 'Utilities Select Sector SPDR Fund',
        'XLC': 'Communication Services Select Sector SPDR Fund',
        'XBI': 'Biotech SPDR ETF',
        'XAR': 'Aerospace & Defense ETF',
        'XOP': 'Oil & Gas Exploration & Production ETF',
        'XME': 'Metals & Mining ETF',
        'KBE': 'Bank ETF',
        'XHB': 'Homebuilders ETF',
        'VUG': 'Vanguard Growth ETF',
        'DBC': 'Commodities ETF',
        'IAU': 'Gold ETF',
        'TLT': 'Long-Term Treasury ETF',
        'SHY': 'Short-Term Treasury ETF',
        'USMV': 'Minimum Volatility ETF',
        'VWO': 'Emerging Markets ETF',
        'IBIT': 'Bitcoin Trust',
        'SVXY': 'Volatility Short',
        'HYG': 'High Yield Bond ETF'
    }
    return spdr_sectors

def download_weekly_data(etf_symbols, years_back=10):
    """
    Download weekly closing price data for given ETF symbols.
    
    Parameters:
    etf_symbols (list): List of ETF symbols to download.
    years_back (int): Number of years of historical data to fetch.
    
    Returns:
    DataFrame: Weekly closing prices with dates as index.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)
    
    print(f"Downloading weekly data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Download weekly data
    data = yf.download(
        etf_symbols, 
        start=start_date.strftime('%Y-%m-%d'), 
        end=end_date.strftime('%Y-%m-%d'), 
        interval='1wk'
    )["Close"]
    
    # Clean data - remove rows with any NaN values
    data = data.dropna()
    
    return data

def calculate_correlation_matrix(price_data):
    """
    Calculate correlation matrix from weekly price returns.
    
    Parameters:
    price_data (DataFrame): Weekly closing prices.
    
    Returns:
    DataFrame: Correlation matrix of weekly returns.
    """
    # Calculate weekly returns
    weekly_returns = price_data.pct_change().dropna()
    
    # Calculate correlation matrix
    correlation_matrix = weekly_returns.corr()
    
    return correlation_matrix, weekly_returns

def plot_correlation_heatmap(correlation_matrix, etf_descriptions):
    """
    Create a heatmap visualization of the correlation matrix.
    
    Parameters:
    correlation_matrix (DataFrame): Correlation matrix to visualize.
    etf_descriptions (dict): Dictionary mapping symbols to descriptions.
    """
    plt.figure(figsize=(12, 10))
    
    # Create labels with both symbol and sector name
    labels = [f"{symbol}\n({etf_descriptions.get(symbol, symbol)})" for symbol in correlation_matrix.columns]
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='RdYlBu_r', 
        center=0,
        fmt='.2f',
        square=True,
        mask=mask,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    
    plt.title('SPDR Sector ETFs - Weekly Returns Correlation Matrix\n(Last 10 Years)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('')
    plt.ylabel('')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()

def identify_high_correlations(correlation_matrix, threshold=0.7):
    """
    Identify ETF pairs with high correlation above threshold.
    
    Parameters:
    correlation_matrix (DataFrame): Correlation matrix.
    threshold (float): Correlation threshold for flagging pairs.
    
    Returns:
    list: List of tuples with highly correlated ETF pairs and their correlation.
    """
    high_correlations = []
    
    # Get upper triangle of correlation matrix (avoid duplicates)
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # Find correlations above threshold
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            corr_value = upper_triangle.loc[idx, col]
            if pd.notna(corr_value) and abs(corr_value) >= threshold:
                high_correlations.append((idx, col, corr_value))
    
    # Sort by correlation strength
    high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return high_correlations

def analyze_sector_correlations():
    """
    Main function to perform SPDR sector ETF correlation analysis.
    """
    # Get SPDR sector ETFs
    etf_descriptions = get_spdr_sector_etfs()
    etf_symbols = list(etf_descriptions.keys())
    
    print("Analyzing correlations for SPDR Sector ETFs:")
    for symbol, description in etf_descriptions.items():
        print(f"  {symbol}: {description}")
    
    # Download weekly data
    print("\nDownloading weekly price data...")
    weekly_prices = download_weekly_data(etf_symbols, years_back=10)
    
    # Calculate correlation matrix
    print("Calculating correlation matrix from weekly returns...")
    correlation_matrix, weekly_returns = calculate_correlation_matrix(weekly_prices)
    
    # Display basic statistics
    print(f"\nData Summary:")
    print(f"Date range: {weekly_prices.index[0].strftime('%Y-%m-%d')} to {weekly_prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"Number of weeks: {len(weekly_prices)}")
    print(f"ETFs analyzed: {len(weekly_prices.columns)}")
    
    # Display correlation matrix
    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(3))
    
    # Plot correlation heatmap
    plot_correlation_heatmap(correlation_matrix, etf_descriptions)
    
    # Identify high correlations
    high_corr_pairs = identify_high_correlations(correlation_matrix, threshold=0.7)
    
    print(f"\nHighly Correlated Pairs (correlation >= 0.70):")
    print("=" * 60)
    if high_corr_pairs:
        for etf1, etf2, corr in high_corr_pairs:
            desc1 = etf_descriptions.get(etf1, etf1)
            desc2 = etf_descriptions.get(etf2, etf2)
            print(f"{etf1} vs {etf2}: {corr:.3f}")
            print(f"  {desc1}")
            print(f"  {desc2}")
            print()
    else:
        print("No ETF pairs found with correlation >= 0.70")
    
    # Additional analysis - find lowest correlations
    low_corr_pairs = identify_high_correlations(correlation_matrix, threshold=-1.0)
    low_corr_pairs = [pair for pair in low_corr_pairs if pair[2] < 0.5]
    low_corr_pairs.sort(key=lambda x: x[2])
    
    print(f"\nLowest Correlated Pairs (correlation < 0.50):")
    print("=" * 60)
    if low_corr_pairs:
        for etf1, etf2, corr in low_corr_pairs[:5]:  # Show top 5 lowest
            desc1 = etf_descriptions.get(etf1, etf1)
            desc2 = etf_descriptions.get(etf2, etf2)
            print(f"{etf1} vs {etf2}: {corr:.3f}")
            print(f"  {desc1}")
            print(f"  {desc2}")
            print()
    
    # Export results
    correlation_matrix.to_csv('spdr_sector_correlation_matrix.csv')
    weekly_returns.to_csv('spdr_sector_weekly_returns.csv')
    
    print(f"Results exported to:")
    print(f"  - spdr_sector_correlation_matrix.csv")
    print(f"  - spdr_sector_weekly_returns.csv")
    
    return correlation_matrix, weekly_returns, high_corr_pairs

if __name__ == "__main__":
    correlation_matrix, weekly_returns, high_correlations = analyze_sector_correlations()