"""
UMAP reducer for clustflow.

Applies non-linear dimensionality reduction using UMAP for visualization or clustering.

Example
-------
>>> from clustflow.dimensionality.umap import UMAPReducer
>>> reducer = UMAPReducer(n_components=2)
>>> X_umap = reducer.fit_transform(X)
"""

import umap
import pandas as pd

class UMAPReducer:
    """
    UMAP reducer for non-linear dimensionality reduction.

    Parameters
    ----------
    n_components : int, default=2
        Number of embedding dimensions.
    random_state : int, default=42
        Seed for reproducibility.
    **kwargs : additional keyword arguments passed to umap.UMAP

    Attributes
    ----------
    embedding_ : ndarray
        Resulting embedded coordinates.
    """

    def __init__(self, n_components=2, random_state=42, **kwargs):
        self.reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
        self.embedding_ = None

    def fit_transform(self, X):
        """
        Reduces data dimensionality using UMAP.

        Parameters
        ----------
        X : array-like
            Input feature matrix.

        Returns
        -------
        pd.DataFrame
            Transformed 2D data with column labels.
        """
        self.embedding_ = self.reducer.fit_transform(X)
        return pd.DataFrame(self.embedding_, columns=[f"UMAP{i+1}" for i in range(self.embedding_.shape[1])])
