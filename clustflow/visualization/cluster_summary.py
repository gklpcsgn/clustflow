import seaborn as sns
import matplotlib.pyplot as plt

def plot_cluster_heatmap(summary_df, exclude_cols=['cluster', 'count'], title="Cluster Summary Heatmap"):
    """
    summary_df: output from summarize_clusters
    exclude_cols: columns to exclude from heatmap (non-numeric, IDs)
    """
    data = summary_df.drop(columns=exclude_cols, errors='ignore')
    data = data.set_index(summary_df['cluster'])

    plt.figure(figsize=(12, 6))
    sns.heatmap(data, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()
