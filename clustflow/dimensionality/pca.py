"""
PCA reducer for clustflow.

Wraps scikit-learn's PCA to reduce dimensionality for clustering or visualization.

Example
-------
>>> from clustflow.dimensionality.pca import PCAReducer
>>> reducer = PCAReducer(n_components=2)
>>> X_pca = reducer.fit_transform(X)
"""

from sklearn.decomposition import PCA
import pandas as pd

class PCAReducer:
    """
    Principal Component Analysis (PCA) wrapper.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to reduce to.
    random_state : int, default=42
        Random seed (for compatibility, though PCA is deterministic).

    Attributes
    ----------
    embedding_ : ndarray
        Transformed 2D array after reduction.
    """

    def __init__(self, n_components=2, random_state=42):
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.embedding_ = None

    def fit_transform(self, X):
        """
        Applies PCA to input features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        pd.DataFrame
            Transformed PCA components with labeled columns.
        """
        self.embedding_ = self.pca.fit_transform(X)
        return pd.DataFrame(self.embedding_, columns=[f"PCA{i+1}" for i in range(self.embedding_.shape[1])])
