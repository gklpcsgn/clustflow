from sklearn.cluster import KMeans

class KMeansCluster:
    def __init__(self,
                 n_clusters=10,
                 init='k-means++',
                 n_init='auto',
                 max_iter=300,
                 tol=1e-4,
                 algorithm='lloyd',
                 random_state=42):
        """
        Wrapper for sklearn KMeans.

        n_init='auto' (recommended for sklearn>=1.4)
        algorithm: 'lloyd', 'elkan'
        """
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            algorithm=algorithm,
            random_state=random_state
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.model.labels_

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X)

    def get_centers(self):
        return self.model.cluster_centers_

    def score(self, X):
        return self.model.inertia_
