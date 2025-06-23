
from sklearn.decomposition import PCA
import pandas as pd

class PCAReducer:
    def __init__(self, n_components=2, random_state=42):
        """
        PCA for dimensionality reduction.
        Note: PCA is deterministic by default, but we keep random_state for consistency.
        """
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.embedding_ = None

    def fit_transform(self, X):
        self.embedding_ = self.pca.fit_transform(X)
        return pd.DataFrame(self.embedding_, columns=[f"PCA{i+1}" for i in range(self.embedding_.shape[1])])
