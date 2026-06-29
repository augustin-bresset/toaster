"""The :class:`Segmenter` extension point — *anything that groups points*.

A segmenter takes a cloud (optionally restricted to the current selection) and
returns a :class:`~toaster.core.Grouping`. Unsupervised clusterers (DBSCAN),
connected components, and supervised model inference are all the same shape: the
only difference is whether they attach ``suggested_labels`` to the grouping.

This is the seam a third party plugs into. The two helpers below handle the
boilerplate every segmenter shares: restricting to a selection, and scattering
per-subset cluster ids back into a full-length grouping with ``-1`` outside.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from toaster.core import Grouping, PointCloud, Selection
from toaster.core.types import NOISE

__all__ = ["Segmenter", "resolve_points", "gather_inputs", "scatter", "all_noise"]


@runtime_checkable
class Segmenter(Protocol):
    """Groups the points of a cloud into a :class:`~toaster.core.Grouping`."""

    #: Stable registry name (e.g. ``"dbscan"``).
    name: str

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        """Return a full-length grouping; ``-1`` for noise / outside ``selection``."""
        ...


def resolve_points(cloud: PointCloud, selection: Selection | None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz_subset, indices)`` to run on.

    With no selection (or an empty one) the whole cloud is used and ``indices``
    is ``arange(N)``. Otherwise only the selected points are returned, so a
    clusterer can carve a roughly-lassoed region into clean instances.
    """
    if selection is None or selection.is_empty():
        return cloud.xyz, np.arange(cloud.n, dtype=np.int64)
    indices = selection.indices
    return cloud.xyz[indices], indices


def gather_inputs(
    cloud: PointCloud, indices: np.ndarray, feature_keys: tuple[str, ...] = ()
) -> np.ndarray:
    """Assemble a model input array ``[xyz, *features]`` for the given points.

    Most labelling models want more than geometry (e.g. ``[x, y, z, intensity]``).
    ``feature_keys`` names the channels to append, in order; each is taken from
    ``cloud.features`` and column-stacked after xyz.

    Args:
        cloud: The cloud to read from.
        indices: ``(M,)`` indices to gather.
        feature_keys: Feature channel names to append after xyz.

    Returns:
        ``(M, 3 + sum(feature widths))`` float32 array.
    """
    cols: list[np.ndarray] = [cloud.xyz[indices]]
    for key in feature_keys:
        feat = np.asarray(cloud.features[key])[indices]
        cols.append(feat if feat.ndim == 2 else feat[:, None])
    out = cols[0] if len(cols) == 1 else np.hstack(cols)
    return np.ascontiguousarray(out, dtype=np.float32)


def all_noise(indices: np.ndarray, n: int, *, source: str = "unknown") -> Grouping:
    """A grouping with no clusters — every point is noise.

    Returned when there is nothing to cluster (e.g. a one-point selection), so a
    degenerate input yields an empty result instead of crashing the clusterer.
    """
    return scatter(np.full(len(indices), NOISE, dtype=np.int32), indices, n, source=source)


def scatter(
    local_ids: np.ndarray,
    indices: np.ndarray,
    n: int,
    *,
    source: str = "unknown",
    suggested_labels: dict[int, int] | None = None,
    params: dict | None = None,
) -> Grouping:
    """Scatter per-subset group ids into a full-length :class:`Grouping`.

    Args:
        local_ids: ``(M,)`` group id per point of the subset (``-1`` = noise).
        indices: ``(M,)`` indices of those points in the full cloud.
        n: Full cloud size.
        source: Segmenter name, stored on the grouping.
        suggested_labels: Optional ``group_id -> class_id`` mapping.
        params: Parameters used, stored on the grouping.
    """
    full = np.full(n, NOISE, dtype=np.int32)
    full[indices] = np.asarray(local_ids, dtype=np.int32)
    return Grouping(
        group_id=full,
        suggested_labels=suggested_labels,
        source=source,
        params=params or {},
    )
