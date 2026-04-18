import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx

def plot_centrality_heatmap(centrality_df):
    print("Generating Centrality Heatmap...")
    plt.figure(figsize=(14, 6))
    
    # Resample to monthly so the heatmap isn't too dense/noisy
    monthly_centrality = centrality_df.resample('ME').mean()
    
    # Transpose (.T) so sectors are on the Y-axis and time is on the X-axis
    sns.heatmap(
        monthly_centrality.T,
        cmap='YlOrRd',  # Yellow to Red colormap (Red = High Centrality)
        linewidths=0.3,
        cbar_kws={'label': 'Eigenvector Centrality'},
        xticklabels=12  # Only show every 12th month label to keep the axis clean
    )
    
    plt.title('Sector Eigenvector Centrality Over Time (2015–2023)')
    plt.xlabel('Date')
    plt.ylabel('Sector')
    plt.tight_layout()
    plt.savefig('centrality_heatmap.png', dpi=300) # High DPI for the research paper
    print("Saved 'centrality_heatmap.png'")


def plot_network_snapshot(returns, centrality_df, date_str, filename):
    print(f"Generating network snapshot for {date_str}...")
    date = pd.Timestamp(date_str)
    
    # Find the nearest available date in our data
    if date not in centrality_df.index:
        nearest_idx = centrality_df.index.get_indexer([date], method='nearest')[0]
        actual_date = centrality_df.index[nearest_idx]
    else:
        actual_date = date
        
    # Rebuild the correlation graph for that specific 60-day window
    idx = returns.index.get_loc(actual_date)
    window = returns.iloc[max(0, idx-60):idx]
    corr = window.corr()
    
    # Adjacency Transformation
    adj = (1 + corr) / 2
    for col in adj.columns:
        adj.loc[col, col] = 0
        
    G = nx.from_pandas_adjacency(adj)
    centrality_scores = centrality_df.loc[actual_date]
    
    # Visual scaling: Node size based on centrality, edge thickness based on weight
    node_sizes = [centrality_scores[n] * 10000 for n in G.nodes()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    plt.figure(figsize=(10, 10))
    # spring_layout tries to pull highly correlated nodes closer together
    pos = nx.spring_layout(G, weight='weight', seed=42) 
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                           node_color=list(centrality_scores.values), 
                           cmap='YlOrRd', alpha=0.9, edgecolors='black')
    
    nx.draw_networkx_edges(G, pos, width=[w * 3 for w in edge_weights], alpha=0.15)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title(f'S&P 500 Sector Topology: {actual_date.strftime("%Y-%m-%d")}', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved '{filename}'")


if __name__ == "__main__":
    from data_loader import fetch_sector_returns
    from graph_builder import calculate_rolling_centrality
    
    raw_returns = fetch_sector_returns(start_date="2015-01-01", end_date="2023-12-31")
    centrality_df = calculate_rolling_centrality(raw_returns, window=60)
    
    # 1. Generate Heatmap
    plot_centrality_heatmap(centrality_df)
    
    # 2. Generate the 3 Snapshots
    plot_network_snapshot(raw_returns, centrality_df, "2020-01-15", "network_pre_covid.png")
    plot_network_snapshot(raw_returns, centrality_df, "2020-03-20", "network_covid_crash.png")
    plot_network_snapshot(raw_returns, centrality_df, "2022-01-15", "network_2022_hikes.png")
