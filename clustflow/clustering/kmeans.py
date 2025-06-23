"""
KMeans clustering wrapper for clustflow.

Provides a reproducible and configurable interface around scikit-learn's KMeans.

Example
-------
>>> from clustflow.clustering.kmeans import KMeansCluster
>>> model = KMeansCluster(n_clusters=8, max_iter=500)
>>> labels = model.fit_predict(X)
>>> centers = model.get_centers()
"""

from sklearn.cluster import KMeans

class KMeansCluster:
    """
    Enhanced KMeans clustering class with full configurability.

    Parameters
    ----------
    n_clusters : int, default=10
        Number of clusters to form.
    init : {'k-means++', 'random'}, default='k-means++'
        Initialization method.
    n_init : int or 'auto', default='auto'
        Number of initializations.
    max_iter : int, default=300
        Maximum number of iterations per run.
    tol : float, default=1e-4
        Convergence tolerance.
    algorithm : {'lloyd', 'elkan'}, default='lloyd'
        KMeans algorithm.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    model : sklearn.cluster.KMeans
        Fitted model instance.
    """

    def __init__(self,
                 n_clusters=10,
                 init='k-means++',
                 n_init='auto',
                 max_iter=300,
                 tol=1e-4,
                 algorithm='lloyd',
                 random_state=42):
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
        """
        Fits KMeans model on input data.

        Parameters
        ----------
        X : array-like
            Input feature matrix.

        Returns
        -------
        self
        """
        self.model.fit(X)
        return self

    def fit_predict(self, X):
        """
        Fits the model and returns cluster labels.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        labels : ndarray
            Cluster labels for each row.
        """
        self.fit(X)
        return self.model.labels_

    def predict(self, X):
        """
        Assigns new points to the nearest cluster.

        Parameters
        ----------
        X : array-like
            New data to classify.

        Returns
        -------
        labels : ndarray
            Predicted cluster assignments.
        """
        return self.model.predict(X)

    def get_centers(self):
        """
        Returns cluster centroids.

        Returns
        -------
        ndarray
            Cluster centers.
        """
        return self.model.cluster_centers_

    def score(self, X):
        """
        Returns KMeans inertia (sum of squared distances).

        Parameters
        ----------
        X : array-like

        Returns
        -------
        float
            Inertia value.
        """
        return self.model.inertia_
