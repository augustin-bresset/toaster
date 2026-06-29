"""HDBSCAN clustering as a :class:`~toaster.segment.base.Segmenter`.

Uses scikit-learn's built-in ``HDBSCAN`` (>=1.3), so no extra C-extension build
is required. The standalone ``hdbscan`` package is only needed for features the
sklearn implementation lacks.
"""

from __future__ import annotations

from toaster.core import Grouping, PointCloud, Selection

from .base import all_noise, resolve_points, scatter

__all__ = ["HDBSCANSegmenter"]


class HDBSCANSegmenter:
    """Hierarchical density clustering; finds clusters without a fixed ``eps``.

    Args:
        min_cluster_size: Smallest grouping considered a cluster.
        min_samples: Conservativeness of the density estimate (defaults to
            ``min_cluster_size`` when ``None``).
    """

    name = "hdbscan"

    def __init__(self, min_cluster_size: int = 25, min_samples: int | None = None) -> None:
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = min_samples

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        from sklearn.cluster import HDBSCAN

        xyz, indices = resolve_points(cloud, selection)
        if len(xyz) < 2:  # sklearn aborts on a single sample; nothing to cluster
            return all_noise(indices, cloud.n, source=self.name)
        labels = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
        ).fit_predict(xyz)
        return scatter(
            labels,
            indices,
            cloud.n,
            source=self.name,
            params={
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
            },
        )
