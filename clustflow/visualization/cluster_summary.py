"""
Heatmap visualization of per-cluster summary stats.

Use with output from `summarize_clusters()`.

Example
-------
>>> from clustflow.visualization.cluster_summary import plot_cluster_heatmap
>>> plot_cluster_heatmap(summary_df)
"""

import seaborn as sns
import matplotlib.pyplot as plt

def plot_cluster_heatmap(summary_df, exclude_cols=['cluster', 'count'], title="Cluster Summary Heatmap"):
    """
    Plots heatmap from summary DataFrame (cluster x features).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from summarize_clusters().
    exclude_cols : list of str, optional
        Columns to exclude from heatmap (e.g., cluster id, counts).
    title : str, optional
        Title for the heatmap.

    Returns
    -------
    None
    """
    data = summary_df.drop(columns=exclude_cols, errors='ignore')
    data = data.set_index(summary_df['cluster'])

    plt.figure(figsize=(12, 6))
    sns.heatmap(data, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()
