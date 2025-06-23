"""
Feature importance analysis across clusters for clustflow.

Uses ANOVA F-test for numeric features and Chi² for categorical features to assess which features differentiate clusters.

Example
-------
>>> from clustflow.evaluation.cluster_feature_importance import compute_feature_importance
>>> result = compute_feature_importance(X, cluster_labels, categorical_cols=['gender', 'region'])
>>> print(result['numerical'].head())
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, chi2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def compute_feature_importance(X, cluster_labels, categorical_cols=None):
    """
    Computes statistical importance of features across clusters.

    For numeric features:
        - Uses ANOVA F-test (f_classif)
    For categorical features:
        - Uses Chi-squared test

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe.
    cluster_labels : array-like
        Cluster assignments (used as class labels).
    categorical_cols : list of str, optional
        Column names to treat as categorical.

    Returns
    -------
    dict
        Dictionary with:
        - 'numerical': DataFrame with F scores and p-values
        - 'categorical': DataFrame with Chi² scores and p-values
    """
    X = X.copy()
    cluster_labels = np.asarray(cluster_labels)

    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_cols = [col for col in X.columns if col not in categorical_cols]

    results = {}

    # Numerical: ANOVA F-test
    if num_cols:
        f_vals, p_vals = f_classif(X[num_cols], cluster_labels)
        results['numerical'] = pd.DataFrame({
            'feature': num_cols,
            'score': f_vals,
            'p_value': p_vals
        }).sort_values(by='score', ascending=False)

    # Categorical: Chi² test
    if categorical_cols:
        X_cat = X[categorical_cols].copy()
        for col in categorical_cols:
            X_cat[col] = LabelEncoder().fit_transform(X_cat[col].astype(str))

        chi_vals, chi_p = chi2(MinMaxScaler().fit_transform(X_cat), cluster_labels)
        results['categorical'] = pd.DataFrame({
            'feature': categorical_cols,
            'score': chi_vals,
            'p_value': chi_p
        }).sort_values(by='score', ascending=False)

    return results
