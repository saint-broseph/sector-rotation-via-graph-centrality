import yfinance as yf
import pandas as pd

def fetch_sector_returns(start_date="2015-01-01", end_date="2023-12-31"):
    """
    Downloads historical data for the 11 GICS Sector SPDR ETFs 
    and calculates daily returns.
    """
    tickers = ["XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]
    
    print(f"Downloading data for {len(tickers)} sector ETFs...")
    
    # Download the raw data
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Handle the yfinance version difference dynamically
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']
    
    # Calculate daily percentage change (returns) and drop the first row (NaN)
    returns = prices.pct_change().dropna()
    
    print("Data successfully downloaded and returns calculated!")
    return returns

if __name__ == "__main__":
    test_returns = fetch_sector_returns(start_date="2023-01-01", end_date="2023-12-31")
    print("\nFirst 5 rows of our daily returns data:")
    print(test_returns.head())