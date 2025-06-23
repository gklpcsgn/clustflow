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
    """Weighted cluster purity"""
    contingency_matrix = pd.crosstab(y_pred, y_true)
    return np.sum(np.max(contingency_matrix.values, axis=1)) / np.sum(contingency_matrix.values)

def compute_metrics(X, cluster_labels, true_labels=None, extended=False):
    """
    X: array-like, used for intrinsic metrics (PCA/UMAP output)
    cluster_labels: result from clustering algorithm
    true_labels: ground truth labels (e.g., asthma control)
    extended: if True, show optional metrics like FMI and V-measure
    """
    metrics = {}

    # --- Intrinsic (unsupervised) ---
    metrics["Intrinsic: Silhouette"] = silhouette_score(X, cluster_labels)
    metrics["Intrinsic: Davies-Bouldin"] = davies_bouldin_score(X, cluster_labels)
    metrics["Intrinsic: Calinski-Harabasz"] = calinski_harabasz_score(X, cluster_labels)

    # --- External (requires ground truth) ---
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
    Takes pd.Series from compute_metrics
    """
    metrics_series.sort_index().plot(kind="barh", figsize=(8,6), color="skyblue")
    plt.title("Clustering Metrics")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.show()
