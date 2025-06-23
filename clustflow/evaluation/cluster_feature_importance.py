import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, chi2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def compute_feature_importance(X, cluster_labels, categorical_cols=None):
    """
    X: pd.DataFrame of features
    cluster_labels: array-like cluster assignments
    categorical_cols: list of column names to treat as categorical
    Returns: DataFrame with F or chi2 score and p-values
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

    # Categorical: Chi2 test (requires non-negative integers)
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
