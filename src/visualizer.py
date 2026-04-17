import matplotlib.pyplot as plt
import seaborn as sns

def plot_centrality_heatmap(centrality_df):
    plt.figure(figsize=(14, 6))
    # Resample to monthly so the heatmap isn't too dense
    monthly = centrality_df.resample('ME').mean()
    sns.heatmap(
        monthly.T,
        cmap='YlOrRd',
        linewidths=0.3,
        cbar_kws={'label': 'Eigenvector Centrality'},
        xticklabels=12  # show every 12th month label
    )
    plt.title('Sector Eigenvector Centrality Over Time (2015–2023)')
    plt.xlabel('Date')
    plt.ylabel('Sector')
    plt.tight_layout()
    plt.savefig('centrality_heatmap.png', dpi=150)
    print("Saved centrality_heatmap.png")
