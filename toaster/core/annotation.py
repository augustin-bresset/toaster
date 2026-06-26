"""Label writes, with undo/redo.

Every change to a cloud's ``labels`` goes through :class:`AnnotationController`.
Routing all writes through one chokepoint buys two things at once: a cheap
undo/redo log (each edit stores only the touched indices and their previous
values), and the list of *touched* indices so the viewer can recolour just those
points instead of rebuilding the whole cloud.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .point_cloud import PointCloud
from .selection import Selection

__all__ = ["LabelEdit", "EditHistory", "AnnotationController"]


@dataclass(frozen=True)
class LabelEdit:
    """A reversible label change.

    Args:
        indices: ``(M,)`` int64 indices that were written.
        old_values: ``(M,)`` int32 values before the write (for undo).
        new_value: The class id that was written.
    """

    indices: np.ndarray
    old_values: np.ndarray
    new_value: int


class EditHistory:
    """A bounded undo/redo log of :class:`LabelEdit` operations."""

    def __init__(self, limit: int = 256) -> None:
        self._undo: list[LabelEdit] = []
        self._redo: list[LabelEdit] = []
        self._limit = limit

    def push(self, edit: LabelEdit) -> None:
        """Record a freshly applied edit, clearing the redo stack."""
        self._undo.append(edit)
        if len(self._undo) > self._limit:
            self._undo.pop(0)
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> LabelEdit | None:
        """Pop the last edit onto the redo stack and return it (``None`` if empty)."""
        if not self._undo:
            return None
        edit = self._undo.pop()
        self._redo.append(edit)
        return edit

    def redo(self) -> LabelEdit | None:
        """Pop the last undone edit back onto the undo stack and return it."""
        if not self._redo:
            return None
        edit = self._redo.pop()
        self._undo.append(edit)
        return edit

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()


class AnnotationController:
    """The single writer of ``cloud.labels``.

    Args:
        cloud: The cloud whose labels are edited. ``labels`` is allocated if
            missing (filled with ``unlabeled_id``).
        history: The undo/redo log to record edits into.
        unlabeled_id: Fill value used when allocating labels.
    """

    def __init__(
        self,
        cloud: PointCloud,
        history: EditHistory | None = None,
        unlabeled_id: int = 0,
    ) -> None:
        self.cloud = cloud
        self.history = history if history is not None else EditHistory()
        self._labels = cloud.ensure_labels(unlabeled_id)

    def assign(self, selection: Selection, class_id: int) -> np.ndarray:
        """Write ``class_id`` to every selected point; return the touched indices.

        A no-op (empty selection) records nothing and returns an empty array.
        """
        indices = selection.indices
        if indices.size == 0:
            return indices
        old_values = self._labels[indices].copy()
        self._labels[indices] = class_id
        self.history.push(LabelEdit(indices, old_values, int(class_id)))
        return indices

    def undo(self) -> np.ndarray | None:
        """Revert the last edit; return the reverted indices (``None`` if nothing)."""
        edit = self.history.undo()
        if edit is None:
            return None
        self._labels[edit.indices] = edit.old_values
        return edit.indices

    def redo(self) -> np.ndarray | None:
        """Re-apply the last undone edit; return the re-applied indices."""
        edit = self.history.redo()
        if edit is None:
            return None
        self._labels[edit.indices] = edit.new_value
        return edit.indices
