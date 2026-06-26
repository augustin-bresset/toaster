"""DBSCAN clustering as a :class:`~toaster.segment.base.Segmenter`."""

from __future__ import annotations

from toaster.core import Grouping, PointCloud, Selection

from .base import resolve_points, scatter

__all__ = ["DBSCANSegmenter"]


class DBSCANSegmenter:
    """Euclidean density clustering (``sklearn.cluster.DBSCAN``) over xyz.

    Args:
        eps: Neighbourhood radius (in cloud units).
        min_samples: Minimum points to form a dense core.

    Produces anonymous clusters (no ``suggested_labels``): the human assigns a
    class to each cluster they click.
    """

    name = "dbscan"

    def __init__(self, eps: float = 0.5, min_samples: int = 10) -> None:
        self.eps = float(eps)
        self.min_samples = int(min_samples)

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        from sklearn.cluster import DBSCAN

        xyz, indices = resolve_points(cloud, selection)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(xyz)
        return scatter(
            labels,
            indices,
            cloud.n,
            source=self.name,
            params={"eps": self.eps, "min_samples": self.min_samples},
        )
