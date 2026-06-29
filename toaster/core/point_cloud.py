"""The :class:`PointCloud` — Toaster's in-memory representation of one frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["PointCloud"]


@dataclass
class PointCloud:
    """A single lidar frame: geometry, optional features, and working labels.

    The cloud is the shared mutable state that the viewer renders and that
    annotation operations write to. It is deliberately tiny and numpy-only so it
    can be used headless, in a script or a pipeline, without any GUI.

    Args:
        xyz: ``(N, 3)`` float32 point positions. Treated as immutable by
            convention (operations never reorder or mutate it in place).
        features: Optional per-point channels keyed by name. Conventional keys
            are ``"intensity"`` ``(N,)``, ``"rgb"`` ``(N, 3)`` uint8 and
            ``"normals"`` ``(N, 3)``. Using a dict keeps the model open to
            arbitrary sensor channels instead of hard-coding fields.
        labels: ``(N,)`` int32 semantic class id per point — the annotation
            being produced. Allocated lazily by :meth:`ensure_labels` if ``None``.
        source: Path the cloud was loaded from, used as the identity key when
            saving/loading the label sidecar. ``None`` for synthetic clouds.

    Example:
        >>> import numpy as np
        >>> cloud = PointCloud(xyz=np.zeros((3, 3), np.float32))
        >>> cloud.ensure_labels(unlabeled_id=0).tolist()
        [0, 0, 0]
        >>> cloud.n
        3
    """

    xyz: np.ndarray
    features: dict[str, np.ndarray] = field(default_factory=dict)
    labels: np.ndarray | None = None
    source: Path | None = None

    def __post_init__(self) -> None:
        self.xyz = np.ascontiguousarray(self.xyz, dtype=np.float32)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {self.xyz.shape}")
        if self.labels is not None:
            self.labels = self._coerce_labels(self.labels)
        if self.source is not None:
            self.source = Path(self.source)

    @property
    def n(self) -> int:
        """Number of points in the cloud."""
        return int(self.xyz.shape[0])

    def ensure_labels(self, unlabeled_id: int = 0) -> np.ndarray:
        """Return the label array, allocating it filled with ``unlabeled_id`` if absent.

        Returns the cloud's own array (not a copy) so callers can mutate labels
        in place through the annotation layer.
        """
        if self.labels is None:
            self.labels = np.full(self.n, unlabeled_id, dtype=np.int32)
        return self.labels

    def _coerce_labels(self, labels: np.ndarray) -> np.ndarray:
        labels = np.ascontiguousarray(labels, dtype=np.int32)
        if labels.shape != (self.n,):
            raise ValueError(f"labels must have shape ({self.n},), got {labels.shape}")
        return labels
