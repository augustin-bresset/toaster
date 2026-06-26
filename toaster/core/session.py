"""The :class:`Session` — the mutable application state of one annotation job.

Everything an annotation session needs lives here: the cloud, the class palette,
the current selection, the undo/redo history, the stack of groupings (with the
one that is *active*), and the active class. The UI and the interaction
controller read and drive this object; nothing else holds annotation state.

The "is a grouping active?" flag lives here because it decides the core
interaction: a pick selects a whole group when one is active, or a single point
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .annotation import AnnotationController, EditHistory
from .grouping import Grouping
from .label_schema import LabelSchema
from .point_cloud import PointCloud
from .selection import Selection

__all__ = ["Session"]


@dataclass
class Session:
    """Container for the live state of an annotation session.

    Args:
        cloud: The point cloud being annotated.
        schema: The class palette.
        selection: Current selection (defaults to empty over the cloud).
        history: Undo/redo log (defaults to a fresh one).
        groupings: Stored groupings, most-recent last. Flat — no hierarchy.
        active_grouping_index: Index into ``groupings`` of the active one, or
            ``None`` meaning a pick selects a single point.
        active_class: The class id that :meth:`assign` writes.
    """

    cloud: PointCloud
    schema: LabelSchema
    selection: Selection | None = None
    history: EditHistory = field(default_factory=EditHistory)
    groupings: list[Grouping] = field(default_factory=list)
    active_grouping_index: int | None = None
    active_class: int = 0

    def __post_init__(self) -> None:
        if self.selection is None:
            self.selection = Selection.empty(self.cloud.n)
        self._annotation = AnnotationController(self.cloud, self.history, self.schema.unlabeled_id)

    @property
    def annotation(self) -> AnnotationController:
        """The single label writer for this session."""
        return self._annotation

    @property
    def active_grouping(self) -> Grouping | None:
        """The active grouping, or ``None`` if pick-selects-single-point mode."""
        if self.active_grouping_index is None:
            return None
        return self.groupings[self.active_grouping_index]

    def add_grouping(self, grouping: Grouping, *, make_active: bool = True) -> int:
        """Store a grouping; optionally make it active. Returns its index."""
        if grouping.n != self.cloud.n:
            raise ValueError(f"grouping spans {grouping.n} points but cloud has {self.cloud.n}")
        self.groupings.append(grouping)
        index = len(self.groupings) - 1
        if make_active:
            self.active_grouping_index = index
        return index

    def set_active_grouping(self, index: int | None) -> None:
        """Set (or clear with ``None``) the active grouping by index."""
        if index is not None and not (0 <= index < len(self.groupings)):
            raise IndexError(f"no grouping at index {index}")
        self.active_grouping_index = index

    def clear_selection(self) -> None:
        """Reset the selection to empty."""
        self.selection = Selection.empty(self.cloud.n)
