import pandas as pd
import numpy as np
import networkx as nx

def get_centrality_for_window(returns_window):
    corr_matrix = returns_window.corr()
    adj_matrix = (1 + corr_matrix) / 2
    
    # Bypass NumPy read-only issues by using Pandas directly
    for col in adj_matrix.columns:
        adj_matrix.loc[col, col] = 0
    
    G = nx.from_pandas_adjacency(adj_matrix)
    centrality = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
    
    return centrality

def calculate_rolling_centrality(returns, window=60):
    print(f"Calculating rolling eigenvector centrality (window={window} days)...")
    centrality_scores = []
    dates = []
    
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window:i]
        current_date = returns.index[i]
        scores = get_centrality_for_window(window_data)
        centrality_scores.append(scores)
        dates.append(current_date)
        
    centrality_df = pd.DataFrame(centrality_scores, index=dates)
    print("Centrality calculation complete!")
    return centrality_df

if __name__ == "__main__":
    from data_loader import fetch_sector_returns
    
    test_returns = fetch_sector_returns(start_date="2022-01-01", end_date="2023-12-31")
    centrality_df = calculate_rolling_centrality(test_returns, window=60)
    
    print("\nFirst 5 days of Centrality Scores (Higher is more central):")
    print(centrality_df.head())