"""
Hierarchical clustering wrapper for clustflow.

Supports multiple linkage and affinity options with optional linkage matrix output for dendrograms.

Example
-------
>>> from clustflow.clustering.hierarchical import HierarchicalCluster
>>> model = HierarchicalCluster(n_clusters=5, linkage_method='average', affinity='cosine')
>>> labels = model.fit_predict(X)
>>> linkage_matrix = model.get_linkage_matrix()
"""

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage

class HierarchicalCluster:
    """
    Hierarchical clustering using AgglomerativeClustering.

    Parameters
    ----------
    n_clusters : int, default=10
        Number of clusters to form. Ignored if distance_threshold is set.
    linkage_method : {'ward', 'complete', 'average', 'single'}
        Linkage algorithm.
    affinity : str
        Distance metric. Ignored if linkage_method='ward'.
    distance_threshold : float or None
        Cut threshold instead of n_clusters.
    compute_linkage_matrix : bool, default=False
        Whether to compute full linkage matrix for dendrograms.

    Attributes
    ----------
    model : fitted sklearn.cluster.AgglomerativeClustering
    linkage_matrix_ : ndarray or None
    """

    def __init__(self,
                 n_clusters=10,
                 linkage_method='ward',
                 affinity='euclidean',
                 distance_threshold=None,
                 compute_linkage_matrix=False):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.affinity = affinity
        self.distance_threshold = distance_threshold
        self.compute_linkage_matrix = compute_linkage_matrix

        self.model = None
        self.linkage_matrix_ = None

    def fit(self, X):
        """
        Fits hierarchical clustering on data.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        self
        """
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters if self.distance_threshold is None else None,
            linkage=self.linkage_method,
            affinity=self.affinity,
            distance_threshold=self.distance_threshold
        )
        self.model.fit(X)

        if self.compute_linkage_matrix:
            self.linkage_matrix_ = linkage(X, method=self.linkage_method, metric=self.affinity)

        return self

    def fit_predict(self, X):
        """
        Fits model and returns cluster labels.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        labels : ndarray
        """
        self.fit(X)
        return self.model.labels_

    def get_linkage_matrix(self):
        """
        Returns linkage matrix (if computed).

        Returns
        -------
        ndarray
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Linkage matrix not computed. Use compute_linkage_matrix=True.")
        return self.linkage_matrix_
