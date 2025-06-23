import pandas as pd

def summarize_clusters(X, cluster_labels, extra=None, categorical_cols=None):
    """
    X: feature dataframe
    cluster_labels: array of cluster assignments
    extra: pd.Series (e.g., target/control) to summarize per cluster
    categorical_cols: list of categorical columns to summarize with mode
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
