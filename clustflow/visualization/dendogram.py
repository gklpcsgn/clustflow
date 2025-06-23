"""
Plots hierarchical dendrogram using scipy linkage matrix.

Accepts either a precomputed linkage matrix (preferred) or raw feature matrix.

Example
-------
>>> from clustflow.visualization.dendrogram import plot_dendrogram
>>> plot_dendrogram(linkage_matrix=model.get_linkage_matrix())
"""

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def plot_dendrogram(X=None, linkage_matrix=None, method='ward', metric='euclidean', max_samples=200, title="Hierarchical Dendrogram"):
    """
    Draws a dendrogram plot from linkage matrix or raw data.

    Parameters
    ----------
    X : array-like, optional
        Input feature matrix (used if linkage_matrix is not provided).
    linkage_matrix : array-like, optional
        Precomputed linkage matrix.
    method : str, default='ward'
        Linkage method to use if X is provided.
    metric : str, default='euclidean'
        Distance metric (only used if X is provided).
    max_samples : int, default=200
        Max number of points to plot if X is large.
    title : str, optional
        Title of the plot.

    Returns
    -------
    None
    """
    if linkage_matrix is not None:
        Z = linkage_matrix
    elif X is not None:
        if len(X) > max_samples:
            X = X[:max_samples]
        Z = linkage(X, method=method, metric=metric)
    else:
        raise ValueError("You must provide either X or a linkage_matrix.")

    plt.figure(figsize=(10, 5))
    dendrogram(Z, orientation='top', distance_sort='descending', show_leaf_counts=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()
