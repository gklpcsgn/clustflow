import umap
import pandas as pd

class UMAPReducer:
    def __init__(self, n_components=2, random_state=42, **kwargs):
        """
        Wrapper for UMAP dimensionality reduction.
        """
        self.reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
        self.embedding_ = None

    def fit_transform(self, X):
        self.embedding_ = self.reducer.fit_transform(X)
        return pd.DataFrame(self.embedding_, columns=[f"UMAP{i+1}" for i in range(self.embedding_.shape[1])])
