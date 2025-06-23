import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

def plot_dendrogram(X=None, linkage_matrix=None, method='ward', metric='euclidean', max_samples=200, title="Hierarchical Dendrogram"):
    """
    Plots a dendrogram from either:
    - a linkage matrix (recommended), or
    - directly from data matrix X (fallback)
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
