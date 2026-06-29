"""Serializable read-model of the session — the front-end's *read* API.

Writes go through the controller's commands; this is the other half: a flat,
plain-data view of session state that any front-end can render without reaching
into the live domain objects. A Qt panel consumes it the same way a web client
would consume its JSON, so the read path stays presentation-agnostic.

Everything here is plain dataclasses of primitives (ints, strings, tuples) — no
numpy arrays, no Qt, no domain objects — so it is trivially serializable.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ClassInfo", "SegmentInfo", "GroupingInfo", "Snapshot"]

Color = tuple[int, int, int]


@dataclass(frozen=True)
class ClassInfo:
    """One annotation class, for the palette."""

    id: int
    name: str
    color: Color


@dataclass(frozen=True)
class SegmentInfo:
    """One segment of the active grouping, for the Groups panel."""

    id: int
    count: int
    color: Color
    suggested: int | None  # suggested class id, or None
    visible: bool = True


@dataclass(frozen=True)
class GroupingInfo:
    """One stored grouping, for the groupings selector."""

    index: int
    source: str
    n_groups: int


@dataclass(frozen=True)
class Snapshot:
    """A complete, flat read-model of the current session state."""

    classes: list[ClassInfo]
    active_class: int
    unlabeled_id: int
    display_mode: str
    selection_count: int
    groupings: list[GroupingInfo]
    active_grouping_index: int | None
    active_grouping: GroupingInfo | None
    segments: list[SegmentInfo]
    has_suggestions: bool

    def class_name(self, class_id: int) -> str:
        """Human name for ``class_id`` (falls back to the id as text)."""
        for c in self.classes:
            if c.id == class_id:
                return c.name
        return str(class_id)
