"""
Clustering evaluation metrics for clustflow.

Supports both intrinsic (unsupervised) and external (label-aware) metrics.

Example
-------
>>> from clustflow.evaluation.metrics import compute_metrics
>>> result = compute_metrics(X_embedded, labels, true_labels=df['target'], extended=True)
>>> print(result)
"""

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def purity_score(y_true, y_pred):
    """
    Calculates weighted purity of clustering result.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Cluster assignments.

    Returns
    -------
    float
        Weighted purity score between 0 and 1.
    """
    contingency_matrix = pd.crosstab(y_pred, y_true)
    return np.sum(np.max(contingency_matrix.values, axis=1)) / np.sum(contingency_matrix.values)

def compute_metrics(X, cluster_labels, true_labels=None, extended=False):
    """
    Computes clustering metrics based on cluster labels and optionally true labels.

    Parameters
    ----------
    X : array-like
        Feature matrix (2D embedding for silhouette, DB, CH).
    cluster_labels : array-like
        Cluster assignments.
    true_labels : array-like, optional
        Ground truth labels. If provided, ARI, NMI, purity will be included.
    extended : bool, default=False
        Whether to compute optional metrics like FMI, V-measure.

    Returns
    -------
    pd.Series
        Dictionary-like object with clustering metric scores.
    """
    metrics = {}

    # Intrinsic (unsupervised) metrics
    metrics["Intrinsic: Silhouette"] = silhouette_score(X, cluster_labels)
    metrics["Intrinsic: Davies-Bouldin"] = davies_bouldin_score(X, cluster_labels)
    metrics["Intrinsic: Calinski-Harabasz"] = calinski_harabasz_score(X, cluster_labels)

    # External (if labels available)
    if true_labels is not None:
        metrics["External: ARI"] = adjusted_rand_score(true_labels, cluster_labels)
        metrics["External: NMI"] = normalized_mutual_info_score(true_labels, cluster_labels)
        metrics["External: Purity"] = purity_score(true_labels, cluster_labels)

        if extended:
            metrics["External: FMI"] = fowlkes_mallows_score(true_labels, cluster_labels)
            metrics["External: Homogeneity"] = homogeneity_score(true_labels, cluster_labels)
            metrics["External: Completeness"] = completeness_score(true_labels, cluster_labels)
            metrics["External: V-Measure"] = v_measure_score(true_labels, cluster_labels)

    return pd.Series(metrics).sort_index()

def plot_metrics(metrics_series):
    """
    Plots clustering metrics as a horizontal bar chart.
    Parameters
    ----------
    metrics_series : pd.Series
        Series containing metric names as index and scores as values.
    """
    plt.figure(figsize=(10, 6))
    metrics_series.sort_index().plot(kind="barh", figsize=(8,6), color="skyblue")
    plt.title("Clustering Metrics")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.show()
