"""The :class:`Grouping` — a transient over-segmentation of a cloud.

A grouping is the output of any :class:`~toaster.segment.base.Segmenter`. It is
*scaffolding*: it exists only to make selection fast (click a cluster, select
the whole group). It is cheap to recompute and is never the saved deliverable —
that is always the cloud's ``labels`` array. Keeping the two apart means a
re-run of a model can never clobber human annotation work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import NOISE

__all__ = ["Grouping"]


@dataclass
class Grouping:
    """A per-point group assignment, always full length ``N``.

    Args:
        group_id: ``(N,)`` int32, the group each point belongs to. ``-1``
            (:data:`~toaster.core.types.NOISE`) means noise, unassigned, or
            outside the subset that was segmented. Always full-length so indices
            stay aligned with ``cloud.xyz`` and ``cloud.labels``.
        suggested_labels: Optional ``group_id -> class_id`` mapping. A supervised
            model fills this so a click can accept its prediction directly; an
            unsupervised clusterer (DBSCAN) leaves it ``None``.
        source: Name of the segmenter that produced this grouping (provenance).
        params: The parameters the segmenter ran with (reproducibility / UI).

    Example:
        >>> import numpy as np
        >>> g = Grouping(np.array([0, 0, 1, -1], np.int32))
        >>> g.n_groups
        2
        >>> g.indices_of(0).tolist()
        [0, 1]
        >>> g.group_of(3)
        -1
    """

    group_id: np.ndarray
    suggested_labels: dict[int, int] | None = None
    source: str = "unknown"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.group_id = np.ascontiguousarray(self.group_id, dtype=np.int32)
        if self.group_id.ndim != 1:
            raise ValueError(f"group_id must be 1-D, got shape {self.group_id.shape}")
        # Precompute group -> indices once (sorted scan) so click-to-select and
        # from_group are O(1) lookups instead of an O(N) scan per click.
        order = np.argsort(self.group_id, kind="stable")
        sorted_ids = self.group_id[order]
        boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
        chunks = np.split(order, boundaries)
        self._index_of_group: dict[int, np.ndarray] = {}
        for chunk in chunks:
            gid = int(self.group_id[chunk[0]])
            if gid != NOISE:
                self._index_of_group[gid] = chunk.astype(np.int64, copy=False)

    @property
    def n(self) -> int:
        """Number of points the grouping spans."""
        return int(self.group_id.shape[0])

    @property
    def n_groups(self) -> int:
        """Number of real groups (excludes noise ``-1``)."""
        return len(self._index_of_group)

    def group_ids(self) -> np.ndarray:
        """Sorted array of the real group ids present (excludes ``-1``)."""
        return np.array(sorted(self._index_of_group), dtype=np.int32)

    def indices_of(self, group_id: int) -> np.ndarray:
        """Indices of every point in ``group_id`` (empty array if absent / noise)."""
        return self._index_of_group.get(int(group_id), _EMPTY)

    def group_of(self, point_index: int) -> int:
        """The group id of a single point (``-1`` if noise)."""
        return int(self.group_id[point_index])


_EMPTY = np.empty(0, dtype=np.int64)
