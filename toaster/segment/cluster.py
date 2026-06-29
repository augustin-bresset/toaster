"""Extra clustering segmenters (scikit-learn + a small numpy k-medoids).

Heavy / quadratic methods (agglomerative, mean-shift, k-medoids) refuse very
large inputs with a clear message — run them on a selection instead.
"""

from __future__ import annotations

import numpy as np

from toaster.core import Grouping, PointCloud, Selection

from .base import Param, all_noise, resolve_points, scatter

__all__ = [
    "KMeansSegmenter",
    "KMedoidsSegmenter",
    "AgglomerativeSegmenter",
    "OPTICSSegmenter",
    "MeanShiftSegmenter",
]


def _assign_subsampled(xyz, labeler, max_points, seed=0):
    """Run a quadratic clusterer on a random subsample, then assign every point.

    Keeps O(n^2) methods usable on large clouds: cluster a subsample, take each
    cluster's centroid, and label every point by its nearest centroid.
    """
    n = len(xyz)
    if n <= max_points:
        return labeler(xyz)
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(seed)
    sub = rng.choice(n, max_points, replace=False)
    sub_labels = np.asarray(labeler(xyz[sub]))
    present = np.unique(sub_labels[sub_labels >= 0])
    if present.size == 0:
        return np.full(n, -1, dtype=np.int32)
    centers = np.array([xyz[sub][sub_labels == u].mean(axis=0) for u in present])
    nearest = cdist(xyz, centers).argmin(axis=1)
    return present[nearest].astype(np.int32)


class KMeansSegmenter:
    """Partition into exactly ``n_clusters`` blobs (``sklearn.cluster.KMeans``)."""

    name = "kmeans"
    PARAMS = [Param("n_clusters", "int", 8, 2, 200, 1)]

    def __init__(self, n_clusters: int = 8) -> None:
        self.n_clusters = int(n_clusters)

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        from sklearn.cluster import KMeans

        xyz, indices = resolve_points(cloud, selection)
        if len(xyz) < self.n_clusters:
            return all_noise(indices, cloud.n, source=self.name)
        labels = KMeans(n_clusters=self.n_clusters, n_init="auto").fit_predict(xyz)
        return scatter(labels, indices, cloud.n, source=self.name,
                       params={"n_clusters": self.n_clusters})  # fmt: skip


class KMedoidsSegmenter:
    """K-medoids (PAM-style): cluster centers are actual points, robust to outliers."""

    name = "kmedoids"
    PARAMS = [Param("n_clusters", "int", 8, 2, 50, 1)]
    MAX_POINTS = 6000

    def __init__(self, n_clusters: int = 8) -> None:
        self.n_clusters = int(n_clusters)

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        xyz, indices = resolve_points(cloud, selection)
        if len(xyz) < self.n_clusters:
            return all_noise(indices, cloud.n, source=self.name)
        labels = _assign_subsampled(
            np.asarray(xyz, dtype=np.float64),
            lambda x: _kmedoids(x, self.n_clusters),
            self.MAX_POINTS,
        )
        return scatter(labels, indices, cloud.n, source=self.name,
                       params={"n_clusters": self.n_clusters})  # fmt: skip


class AgglomerativeSegmenter:
    """Bottom-up hierarchical clustering (Ward linkage) into ``n_clusters``."""

    name = "agglomerative"
    PARAMS = [Param("n_clusters", "int", 8, 2, 200, 1)]
    MAX_POINTS = 12000

    def __init__(self, n_clusters: int = 8) -> None:
        self.n_clusters = int(n_clusters)

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        from sklearn.cluster import AgglomerativeClustering

        xyz, indices = resolve_points(cloud, selection)
        if len(xyz) < self.n_clusters:
            return all_noise(indices, cloud.n, source=self.name)
        labels = _assign_subsampled(
            np.asarray(xyz, dtype=np.float32),
            lambda x: AgglomerativeClustering(n_clusters=self.n_clusters).fit_predict(x),
            self.MAX_POINTS,
        )
        return scatter(labels, indices, cloud.n, source=self.name,
                       params={"n_clusters": self.n_clusters})  # fmt: skip


class OPTICSSegmenter:
    """Density clustering that handles varying density (``sklearn.cluster.OPTICS``)."""

    name = "optics"
    PARAMS = [Param("min_samples", "int", 10, 2, 1000, 1)]

    def __init__(self, min_samples: int = 10) -> None:
        self.min_samples = int(min_samples)

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        from sklearn.cluster import OPTICS

        xyz, indices = resolve_points(cloud, selection)
        if len(xyz) <= self.min_samples:
            return all_noise(indices, cloud.n, source=self.name)
        labels = OPTICS(min_samples=self.min_samples).fit_predict(xyz)
        return scatter(labels, indices, cloud.n, source=self.name,
                       params={"min_samples": self.min_samples})  # fmt: skip


class MeanShiftSegmenter:
    """Mode-seeking clustering; finds the number of clusters itself."""

    name = "meanshift"
    PARAMS = [Param("bandwidth", "float", 0.0, 0.0, 100.0, 0.1)]  # 0 = estimate
    MAX_POINTS = 8000

    def __init__(self, bandwidth: float = 0.0) -> None:
        self.bandwidth = float(bandwidth)

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        from sklearn.cluster import MeanShift

        xyz, indices = resolve_points(cloud, selection)
        if len(xyz) < 2:
            return all_noise(indices, cloud.n, source=self.name)
        bw = self.bandwidth if self.bandwidth > 0 else None
        labels = _assign_subsampled(
            np.asarray(xyz, dtype=np.float32),
            lambda x: MeanShift(bandwidth=bw, bin_seeding=True).fit_predict(x),
            self.MAX_POINTS,
        )
        return scatter(labels, indices, cloud.n, source=self.name,
                       params={"bandwidth": self.bandwidth})  # fmt: skip


def _kmedoids(x: np.ndarray, k: int, max_iter: int = 50, seed: int = 0) -> np.ndarray:
    """Voronoi-iteration k-medoids; returns a label per row of ``x``."""
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(seed)
    medoids = rng.choice(len(x), k, replace=False)
    labels = np.zeros(len(x), dtype=np.int32)
    for _ in range(max_iter):
        labels = cdist(x, x[medoids]).argmin(axis=1).astype(np.int32)
        new = medoids.copy()
        for j in range(k):
            members = np.flatnonzero(labels == j)
            if members.size:
                costs = cdist(x[members], x[members]).sum(axis=1)
                new[j] = members[costs.argmin()]
        if np.array_equal(new, medoids):
            break
        medoids = new
    return labels
