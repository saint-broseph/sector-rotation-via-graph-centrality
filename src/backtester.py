import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def calculate_metrics(returns, benchmark_returns=None):
    ann_ret = (1 + returns).prod() ** (12 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    info_ratio = "N/A"
    if benchmark_returns is not None:
        active_return = returns - benchmark_returns
        tracking_error = active_return.std() * np.sqrt(12)
        info_ratio = (ann_ret - ((1 + benchmark_returns).prod() ** (12 / len(benchmark_returns)) - 1)) / tracking_error if tracking_error != 0 else 0
        info_ratio = f"{info_ratio:.2f}"
        
    return ann_ret, ann_vol, sharpe, max_dd, info_ratio

def run_momentum_baseline(returns, lookback=60, top_n=2):
    print(f"Running pure Price Momentum baseline (Holding Top {top_n})...")
    monthly_returns = (1 + returns).resample('ME').prod() - 1
    
    # Calculate rolling cumulative returns for the lookback window
    rolling_momentum = returns.rolling(lookback).sum()
    monthly_momentum = rolling_momentum.resample('ME').last()
    
    target_holdings = monthly_momentum.apply(
        lambda x: x.nlargest(top_n).index.tolist(), axis=1
    ).shift(1).dropna()
    
    strat_returns = []
    dates = []
    for date, tickers in target_holdings.items():
        if date in monthly_returns.index:
            strat_returns.append(monthly_returns.loc[date, tickers].mean())
            dates.append(date)
            
    return pd.DataFrame({'Momentum': strat_returns}, index=dates)

def run_backtest(returns, centrality, top_n=2):
    print(f"\nRunning Graph Centrality backtest (Holding Top {top_n} with Regime Switch)...")
    monthly_centrality = centrality.resample('ME').last()
    
    # .shift(1) prevents look-ahead bias
    target_holdings = monthly_centrality.apply(
        lambda x: x.nlargest(top_n).index.tolist(), axis=1
    ).shift(1).dropna()
    
    monthly_returns = (1 + returns).resample('ME').prod() - 1
    
    strategy_returns = []
    dates = []
    
    # --- THE TOPOLOGICAL REGIME SWITCH ---
    defensive_sectors = ['XLU', 'XLP'] 
    
    for date, tickers in target_holdings.items():
        if date in monthly_returns.index:
            # If the network identifies a flight-to-safety, move to Cash
            if any(sector in tickers for sector in defensive_sectors):
                strategy_returns.append(0.0) # 0% return for the month (Cash)
            else:
                strategy_returns.append(monthly_returns.loc[date, tickers].mean())
            dates.append(date)
            
    strategy_df = pd.DataFrame({'Graph Strategy': strategy_returns}, index=dates)
    
    # Get Momentum Baseline
    momentum_df = run_momentum_baseline(returns, top_n=top_n)
    
    print("Fetching SPY benchmark data...")
    spy_data = yf.download("SPY", start=dates[0], end=dates[-1] + pd.Timedelta(days=31), progress=False)
    spy_prices = spy_data['Adj Close'] if 'Adj Close' in spy_data.columns else spy_data['Close']
    
    # Convert SPY to log returns, then resample to monthly
    spy_daily_returns = np.log(spy_prices / spy_prices.shift(1)).dropna()
    spy_monthly_returns = (1 + spy_daily_returns).resample('ME').prod() - 1
    spy_monthly_returns.name = 'SPY'
    
    # Combine everything
    comparison = pd.concat([strategy_df['Graph Strategy'], momentum_df['Momentum'], spy_monthly_returns], axis=1).dropna()
    
    strat_metrics = calculate_metrics(comparison['Graph Strategy'], comparison['SPY'])
    mom_metrics = calculate_metrics(comparison['Momentum'], comparison['SPY'])
    spy_metrics = calculate_metrics(comparison['SPY'])
    
    print("\n--- PERFORMANCE METRICS (2015-2023) ---")
    print(f"{'Metric':<18} | {'Graph Strategy':<15} | {'Momentum':<15} | {'SPY':<10}")
    print("-" * 65)
    print(f"{'Ann. Return':<18} | {strat_metrics[0]:.2%}          | {mom_metrics[0]:.2%}          | {spy_metrics[0]:.2%}")
    print(f"{'Ann. Volatility':<18} | {strat_metrics[1]:.2%}          | {mom_metrics[1]:.2%}          | {spy_metrics[1]:.2%}")
    print(f"{'Sharpe Ratio':<18} | {strat_metrics[2]:.2f}             | {mom_metrics[2]:.2f}             | {spy_metrics[2]:.2f}")
    print(f"{'Max Drawdown':<18} | {strat_metrics[3]:.2%}         | {mom_metrics[3]:.2%}         | {spy_metrics[3]:.2%}")
    print(f"{'Info Ratio (vs SPY)':<18} | {strat_metrics[4]:<15} | {mom_metrics[4]:<15} | N/A")
    
    # Plotting
    cumulative_returns = (1 + comparison).cumprod()
    plt.figure(figsize=(10, 6))
    
    # Visualizing the Regime Switch periods (flat lines)
    plt.plot(cumulative_returns.index, cumulative_returns['Graph Strategy'], label=f'Graph Strategy (Top {top_n} + Regime Filter)', color='darkblue', linewidth=2)
    plt.plot(cumulative_returns.index, cumulative_returns['Momentum'], label=f'Price Momentum (Top {top_n})', color='darkred', linestyle='--', linewidth=1.5)
    plt.plot(cumulative_returns.index, cumulative_returns['SPY'], label='SPY Benchmark', color='black', alpha=0.5, linewidth=1.5)
    
    plt.title(f'Cumulative Returns: Topology Regime Switch vs S&P 500')
    plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('equity_curve.png', dpi=300)
    print("\nSaved final performance chart to 'equity_curve.png'")
