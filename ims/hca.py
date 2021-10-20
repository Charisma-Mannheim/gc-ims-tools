import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram


class HCA:
    """
    Hierarchical cluster analysis with scikit-learn AgglomerativeClustering
    and scipy dendrogram.

    Parameters
    ----------
    dataset : ims.Dataset, optional
        Dataset with GC-IMS data is needed for sample and label names
        in dendrogram. If not set uses leaves as labels instead,
        by default None.

    affinity : str, optional
        Metric used to compute the linkage.
        Can be "euclidean", "l1", "l2", "manhattan" or "cosine".
        If linkage is set to "ward" only "euclidean" is accepted,
        by default "euclidean".

    linkage : str, optional
        Linkage criterion which determines which distance to use.
        "ward", "complete", "average" or "single" are accepted,
        by default "ward".

    Attributes
    ----------
    clustering : sklearn.cluster.AgglomerativeClustering
        Scikit-learn algorithm used for the clustering.
        See the original documentation for details about attributes.

    linkage_matrix : numpy.ndarray
        Clustering results encoded as linkage matrix.

    R : dict
        scipy dendrogram output as dictionary.

    Example
    -------
    >>> import ims
    >>> ds = ims.Dataset.read_mea("IMS_data")
    >>> X, _ = ds.get_xy()
    >>> hca = ims.HCA(ds, linkage="ward", affinity="euclidean")
    >>> hca.fit(X)
    >>> hca.plot_dendrogram()
    """
    def __init__(self, dataset=None, affinity="euclidean", linkage="ward",):
        self.dataset = dataset
        self.linkage = linkage
        self.affinity = affinity
        self.clustering = AgglomerativeClustering(
            distance_threshold=0,
            n_clusters=None,
            affinity=affinity,
            linkage=linkage
        )

    def fit(self, X):
        """
        Fit the model from features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features to cluster.
        """
        self.clustering.fit(X)
        self.linkage_matrix = self._get_linkage_matrix()
        self.R = dendrogram(self.linkage_matrix, no_plot=True)
        self.labels = self.clustering.labels_

    def plot_dendrogram(self, width=9, height=10,
                        orientation="right", **kwargs):
        """
        Plots clustering results as dendrogram.

        Parameters
        ----------
        width : int, optional
            Width of the figure in inches, by default 9

        height : int, optional
            Width of the figure in inches, by default 10

        orientation : str, optional
            Root position of the clustering tree, by default "right"

        **kwargs
            See scipy.cluster.hierarchy.dendrogram documentation
            for information about valid keyword arguments.

        Returns
        -------
        matplotlib.pyplot.axes
        """        
        _, ax = plt.subplots(figsize=(width, height))

        if self.dataset is not None:
            labels = [f"{i[0]}; {i[1]}" for i in zip(self.dataset.labels,
                                                     self.dataset.samples)]
        else:
            labels = self.R["leaves"]

        dendrogram(
            self.linkage_matrix,
            ax=ax,
            orientation=orientation,
            labels=labels,
            **kwargs
            )
        plt.xlabel(f"Distances ({self.affinity} method)")
        return ax

    def _get_linkage_matrix(self):
        """Builds linkage matrix from AgglomerativeClustering output."""
        counts = np.zeros(self.clustering.children_.shape[0])
        n_samples = len(self.clustering.labels_)
        for i, merge in enumerate(self.clustering.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        return np.column_stack(
            [self.clustering.children_,
             self.clustering.distances_, counts]
            ).astype(float)
