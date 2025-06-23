"""
2D scatter plot for cluster visualization.

Supports UMAP, PCA, or any 2D-reduced data. Colors can show either cluster assignments or external labels.

Example
-------
>>> from clustflow.visualization.scatter import scatter_2d
>>> scatter_2d(X_pca, labels, color_by=df['target'], title="PCA Cluster View")
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def scatter_2d(X, labels, color_by=None, title="", save_path=None):
    """
    Plots 2D data points with coloring by cluster or label.

    Parameters
    ----------
    X : array-like or DataFrame
        2D-reduced feature matrix (n_samples x 2).
    labels : array-like
        Cluster assignments or identifiers.
    color_by : array-like, optional
        Optional external column (e.g., ground truth) to color by.
    title : str, optional
        Plot title.
    save_path : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    None
    """
    df = pd.DataFrame(X, columns=["dim1", "dim2"])
    df["cluster"] = labels
    if color_by is not None:
        df["color"] = color_by
        hue = "color"
    else:
        hue = "cluster"

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="dim1", y="dim2", hue=hue, palette="tab10", s=50, alpha=0.8, edgecolor="black")
    plt.title(title or "Cluster Visualization")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
