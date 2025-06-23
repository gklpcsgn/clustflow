from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import numpy as np

class HierarchicalCluster:
    def __init__(self,
                 n_clusters=10,
                 linkage_method='ward',
                 affinity='euclidean',
                 distance_threshold=None,
                 compute_linkage_matrix=False):
        """
        Hierarchical Clustering.
        
        linkage_method: 'ward', 'complete', 'average', 'single'
        affinity: 'euclidean', 'manhattan', 'cosine' (ignored if ward)
        distance_threshold: float or None. If set, n_clusters is ignored.
        compute_linkage_matrix: if True, compute linkage matrix separately
        """
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.affinity = affinity
        self.distance_threshold = distance_threshold
        self.compute_linkage_matrix = compute_linkage_matrix

        self.model = None
        self.linkage_matrix_ = None

    def fit(self, X):
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
        self.fit(X)
        return self.model.labels_

    def get_linkage_matrix(self):
        if self.linkage_matrix_ is None:
            raise ValueError("Linkage matrix not computed. Set compute_linkage_matrix=True.")
        return self.linkage_matrix_
