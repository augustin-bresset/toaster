"""The :class:`Selection` — the bridge between groupings and labels.

A selection is just a set of point indices over a cloud of size ``N``. The whole
annotation workflow is the pipeline ``Grouping -> Selection -> labels``: a model
produces groups, a click turns a group into a selection, and assigning a class
writes the selection into ``labels``.

It is stored as a boolean mask (``N`` bits of state, vectorized set algebra) and
is pure numpy — no VTK, no camera. The geometry-dependent constructors (picking
a point on screen, a frustum box) live in the *viewer*, which resolves them to
plain ``int`` indices and hands them here. That boundary is what keeps the whole
``core`` package headless and unit-testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .grouping import Grouping

__all__ = ["Selection"]


class Selection:
    """An index set over a cloud of ``N`` points, backed by a boolean mask.

    Supports boolean algebra so irregular regions compose naturally:

    - ``a | b`` — union (add to selection)
    - ``a & b`` — intersection
    - ``a - b`` — difference (subtract from selection)
    - ``a ^ b`` — symmetric difference (toggle)
    """

    __slots__ = ("_mask",)

    def __init__(self, mask: np.ndarray) -> None:
        """Wrap an existing ``(N,)`` boolean mask (not copied)."""
        mask = np.ascontiguousarray(mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError(f"mask must be 1-D, got shape {mask.shape}")
        self._mask = mask

    # -- constructors -----------------------------------------------------

    @classmethod
    def empty(cls, n: int) -> Selection:
        """An empty selection over ``n`` points."""
        return cls(np.zeros(n, dtype=bool))

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> Selection:
        """Build from a boolean mask."""
        return cls(mask)

    @classmethod
    def from_indices(cls, indices: np.ndarray, n: int) -> Selection:
        """Build from an array of point indices over a cloud of size ``n``."""
        mask = np.zeros(n, dtype=bool)
        mask[np.asarray(indices, dtype=np.int64)] = True
        return cls(mask)

    @classmethod
    def from_pick(cls, index: int, n: int) -> Selection:
        """Build from a single picked point index."""
        mask = np.zeros(n, dtype=bool)
        mask[index] = True
        return cls(mask)

    @classmethod
    def from_group(cls, grouping: Grouping, group_id: int) -> Selection:
        """Select every point of ``group_id`` — the click-a-cluster operation."""
        return cls.from_indices(grouping.indices_of(group_id), grouping.n)

    # -- queries ----------------------------------------------------------

    @property
    def mask(self) -> np.ndarray:
        """The underlying ``(N,)`` boolean mask (live, do not mutate)."""
        return self._mask

    @property
    def indices(self) -> np.ndarray:
        """The selected point indices as an ``(M,)`` int64 array."""
        return np.flatnonzero(self._mask)

    @property
    def count(self) -> int:
        """Number of selected points."""
        return int(self._mask.sum())

    @property
    def n(self) -> int:
        """Size of the cloud this selection spans."""
        return int(self._mask.shape[0])

    def is_empty(self) -> bool:
        """Whether nothing is selected."""
        return not self._mask.any()

    # -- boolean algebra --------------------------------------------------

    def _check(self, other: Selection) -> None:
        if other.n != self.n:
            raise ValueError(f"selection size mismatch: {self.n} vs {other.n}")

    def __or__(self, other: Selection) -> Selection:
        self._check(other)
        return Selection(self._mask | other._mask)

    def __and__(self, other: Selection) -> Selection:
        self._check(other)
        return Selection(self._mask & other._mask)

    def __sub__(self, other: Selection) -> Selection:
        self._check(other)
        return Selection(self._mask & ~other._mask)

    def __xor__(self, other: Selection) -> Selection:
        self._check(other)
        return Selection(self._mask ^ other._mask)

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return f"Selection(count={self.count}, n={self.n})"
