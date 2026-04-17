import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from src.data_loader import fetch_sector_returns
from src.graph_builder import calculate_rolling_centrality

def calculate_metrics(returns):
    ann_ret = (1 + returns).prod() ** (12 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    return ann_ret, ann_vol, sharpe, max_dd

def run_backtest(returns, centrality, top_n=3):
    print(f"Running backtest simulation (Holding Top {top_n} Sectors)...")
    monthly_centrality = centrality.resample('ME').last()
    
    target_holdings = monthly_centrality.apply(
        lambda x: x.nlargest(top_n).index.tolist(), axis=1
    ).shift(1).dropna()
    
    monthly_returns = (1 + returns).resample('ME').prod() - 1
    
    strategy_returns = []
    dates = []
    for date, tickers in target_holdings.items():
        if date in monthly_returns.index:
            strat_ret = monthly_returns.loc[date, tickers].mean()
            strategy_returns.append(strat_ret)
            dates.append(date)
            
    strategy_df = pd.DataFrame({'Strategy': strategy_returns}, index=dates)
    
    print("Fetching SPY benchmark data...")
    spy_data = yf.download("SPY", start=dates[0], end=dates[-1] + pd.Timedelta(days=31))
    spy_prices = spy_data['Adj Close'] if 'Adj Close' in spy_data.columns else spy_data['Close']
    spy_monthly_returns = spy_prices.resample('ME').last().pct_change().dropna()
    spy_monthly_returns.name = 'SPY'
    
    comparison = pd.concat([strategy_df['Strategy'], spy_monthly_returns], axis=1).dropna()
    
    strat_metrics = calculate_metrics(comparison['Strategy'])
    spy_metrics = calculate_metrics(comparison['SPY'])
    
    print("\n--- PERFORMANCE METRICS ---")
    print(f"{'Metric':<20} | {'Strategy':<10} | {'SPY':<10}")
    print("-" * 45)
    print(f"{'Ann. Return':<20} | {strat_metrics[0]:.2%}     | {spy_metrics[0]:.2%}")
    print(f"{'Ann. Volatility':<20} | {strat_metrics[1]:.2%}     | {spy_metrics[1]:.2%}")
    print(f"{'Sharpe Ratio':<20} | {strat_metrics[2]:.2f}       | {spy_metrics[2]:.2f}")
    print(f"{'Max Drawdown':<20} | {strat_metrics[3]:.2%}     | {spy_metrics[3]:.2%}")
    
    cumulative_returns = (1 + comparison).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns.index, cumulative_returns['Strategy'], label=f'Graph Strategy (Top {top_n})', color='blue')
    plt.plot(cumulative_returns.index, cumulative_returns['SPY'], label='SPY Benchmark', color='black', alpha=0.7)
    plt.title(f'Sector Rotation (Top {top_n} by Centrality) vs S&P 500')
    plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('equity_curve.png')
    print("\nSaved performance chart to 'equity_curve.png'")

if __name__ == "__main__":
    raw_returns = fetch_sector_returns(start_date="2015-01-01", end_date="2023-12-31")
    centrality_df = calculate_rolling_centrality(raw_returns, window=60)
    run_backtest(raw_returns, centrality_df, top_n=3)