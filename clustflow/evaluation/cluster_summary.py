"""
Cluster summarization for clustflow.

Computes descriptive statistics per cluster:
- Mean for numeric columns
- Mode for categorical columns
- Label distribution (if `extra` label column is provided)

Example
-------
>>> from clustflow.evaluation.cluster_summary import summarize_clusters
>>> summary_df = summarize_clusters(X, labels, extra=df['target'], categorical_cols=['race', 'gender'])
>>> print(summary_df.head())
"""

import pandas as pd

def summarize_clusters(X, cluster_labels, extra=None, categorical_cols=None):
    """
    Produces summary statistics for each cluster.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    cluster_labels : array-like
        Cluster assignments.
    extra : pd.Series, optional
        Ground truth label or outcome column to include label breakdowns per cluster.
    categorical_cols : list of str, optional
        Columns to summarize by mode instead of mean.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by cluster, with per-cluster means/modes and optional label ratios.
    """
    X = X.copy()
    X['cluster'] = cluster_labels
    summary = []

    for cluster_id, group in X.groupby('cluster'):
        row = {'cluster': cluster_id, 'count': len(group)}

        for col in group.columns:
            if col == 'cluster':
                continue
            if categorical_cols and col in categorical_cols:
                try:
                    row[col] = group[col].mode().iloc[0]
                except:
                    row[col] = None
            else:
                row[col] = group[col].mean()

        if extra is not None:
            mask = X['cluster'] == cluster_id
            label_counts = extra[mask].value_counts(normalize=True).to_dict()
            for val, frac in label_counts.items():
                row[f'label_{val}_ratio'] = frac

        summary.append(row)

    return pd.DataFrame(summary).sort_values(by='cluster')
