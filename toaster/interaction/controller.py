"""The interaction controller — the workflow glue, front-end agnostic.

It wires viewer callbacks (pick / box) and UI commands (assign / undo / run a
segmenter / change display) to the :class:`~toaster.core.Session`. Because it
talks to the viewer only through the :class:`~toaster.viewer.base.Viewer`
protocol (numpy in, indices out) and never imports Qt, it can be driven by any
front-end (Qt today, a web client tomorrow) and unit-tested with a fake viewer.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from toaster.core import Selection, Session
from toaster.core.types import NOISE
from toaster.segment.base import Segmenter
from toaster.viewer.base import Modifiers, Viewer
from toaster.viewer.colormap import (
    colors_from_grouping,
    colors_from_labels,
    colors_from_scalar,
    group_color,
)

from .snapshot import ClassInfo, GroupingInfo, SegmentInfo, Snapshot

__all__ = ["InteractionController", "DisplayMode"]

DisplayMode = str  # "labels" | "grouping" | "intensity" | "height"


class InteractionController:
    """Mediates between the viewer, the session and the UI.

    Args:
        session: The annotation state to drive.
        viewer: The renderer (any :class:`~toaster.viewer.base.Viewer`).
        on_state_changed: Optional callback invoked after any state change so the
            UI can refresh (selection count, undo availability, ...).
    """

    def __init__(
        self,
        session: Session,
        viewer: Viewer,
        on_state_changed: Callable[[], None] | None = None,
    ) -> None:
        self.session = session
        self.viewer = viewer
        self.display_mode: DisplayMode = "labels"
        self._hidden_groups: set[int] = set()
        self._on_state_changed = on_state_changed
        viewer.set_point_pick_callback(self.on_pick)
        viewer.set_box_pick_callback(self.on_box)

    # -- rendering --------------------------------------------------------

    def refresh_display(self) -> None:
        """Recolour the whole cloud according to the active display mode."""
        colors = self._colors_for_mode(self.display_mode)
        self.viewer.set_cloud(self.session.cloud.xyz, colors)
        self.viewer.highlight(self.session.selection.indices)
        # set_cloud rebuilds a fully-visible mesh; re-apply any hidden segments.
        self._apply_visibility()

    def set_display_mode(self, mode: DisplayMode) -> None:
        self.display_mode = mode
        self.refresh_display()
        self._changed()

    def _colors_for_mode(self, mode: DisplayMode) -> np.ndarray:
        cloud, schema = self.session.cloud, self.session.schema
        if mode == "grouping" and self.session.active_grouping is not None:
            return colors_from_grouping(self.session.active_grouping)
        if mode == "intensity" and "intensity" in cloud.features:
            return colors_from_scalar(cloud.features["intensity"])
        if mode == "height":
            return colors_from_scalar(cloud.xyz[:, 2])
        return colors_from_labels(cloud.ensure_labels(schema.unlabeled_id), schema)

    # -- selection --------------------------------------------------------

    def on_pick(self, index: int, modifiers: Modifiers = frozenset()) -> None:
        """Handle a single-point pick: select its group if one is active, else the point."""
        grouping = self.session.active_grouping
        if grouping is not None and grouping.group_of(index) != NOISE:
            new = Selection.from_group(grouping, grouping.group_of(index))
        else:
            new = Selection.from_pick(index, self.session.cloud.n)
        self._apply_selection(new, modifiers)

    def on_box(self, indices: np.ndarray, modifiers: Modifiers = frozenset()) -> None:
        """Handle a rubber-band box selection of many points."""
        new = Selection.from_indices(indices, self.session.cloud.n)
        self._apply_selection(new, modifiers)

    def _apply_selection(self, new: Selection, modifiers: Modifiers) -> None:
        current = self.session.selection
        if "shift" in modifiers:
            current = current | new
        elif "ctrl" in modifiers:
            current = current - new
        else:
            current = new
        self.session.selection = current
        self.viewer.highlight(current.indices)
        self._changed()

    def clear_selection(self) -> None:
        self.session.clear_selection()
        self.viewer.clear_highlight()
        self._changed()

    # -- annotation -------------------------------------------------------

    def set_active_class(self, class_id: int) -> None:
        self.session.active_class = class_id
        self._changed()

    def assign(self, class_id: int | None = None) -> None:
        """Write the active (or given) class to the current selection."""
        cid = self.session.active_class if class_id is None else class_id
        touched = self.session.annotation.assign(self.session.selection, cid)
        if touched.size:
            self._recolor_labels(touched)
        self.clear_selection()

    def undo(self) -> None:
        touched = self.session.annotation.undo()
        if touched is not None:
            self._recolor_labels(touched)
            self._changed()

    def redo(self) -> None:
        touched = self.session.annotation.redo()
        if touched is not None:
            self._recolor_labels(touched)
            self._changed()

    def _recolor_labels(self, indices: np.ndarray) -> None:
        # Only meaningful while showing labels; other modes recolour on switch.
        if self.display_mode == "labels":
            labels = self.session.cloud.labels[indices]
            self.viewer.update_colors(indices, self.session.schema.colors_for(labels))

    def set_pick_mode(self, mode: str) -> None:
        """Switch the viewer between ``"point"`` and ``"box"`` selection."""
        self.viewer.set_pick_mode(mode)

    # -- segmentation -----------------------------------------------------

    def run_segmenter(self, segmenter: Segmenter, *, scope_to_selection: bool = True) -> None:
        """Run a segmenter (optionally on the current selection) and make it active."""
        selection = (
            self.session.selection
            if scope_to_selection and not self.session.selection.is_empty()
            else None
        )
        grouping = segmenter.segment(self.session.cloud, selection)
        self.session.add_grouping(grouping)
        self._hidden_groups.clear()  # a fresh grouping starts fully visible
        self.set_display_mode("grouping")
        self._changed()

    # -- per-group operations (drive the Groups panel) --------------------

    def select_group(self, group_id: int, modifiers: Modifiers = frozenset()) -> None:
        """Select every point of one segment of the active grouping."""
        grouping = self.session.active_grouping
        if grouping is None:
            return
        self._apply_selection(Selection.from_group(grouping, group_id), modifiers)

    def assign_group(self, group_id: int, class_id: int | None = None) -> int:
        """Label a whole segment with the active (or given) class. Returns points labelled."""
        grouping = self.session.active_grouping
        if grouping is None:
            return 0
        cid = self.session.active_class if class_id is None else class_id
        touched = self.session.annotation.assign(Selection.from_group(grouping, group_id), cid)
        if touched.size:
            self._paint_labels(touched)
        self._changed()
        return int(touched.size)

    def apply_suggested(self, group_id: int | None = None) -> int:
        """Accept model predictions: label group(s) with their ``suggested_labels``.

        With ``group_id`` set, only that segment; otherwise every segment that
        carries a suggestion. Returns the number of points labelled.
        """
        grouping = self.session.active_grouping
        if grouping is None or not grouping.suggested_labels:
            return 0
        if group_id is not None:
            suggestion = grouping.suggested_labels.get(group_id)
            items = [(group_id, suggestion)] if suggestion is not None else []
        else:
            items = list(grouping.suggested_labels.items())
        total = 0
        for gid, cid in items:
            touched = self.session.annotation.assign(Selection.from_group(grouping, gid), cid)
            if touched.size:
                self._paint_labels(touched)
                total += int(touched.size)
        self._changed()
        return total

    def _paint_labels(self, indices: np.ndarray) -> None:
        # Recolour assigned points to their class colour in *any* view, so a
        # labelled segment is visibly "consumed" even while colouring by grouping.
        labels = self.session.cloud.labels[indices]
        self.viewer.update_colors(indices, self.session.schema.colors_for(labels))

    # -- per-group visibility ---------------------------------------------

    def set_group_visibility(self, group_id: int, visible: bool) -> None:
        """Show or hide one segment's points in the 3-D view."""
        if self.session.active_grouping is None:
            return
        if visible:
            self._hidden_groups.discard(group_id)
        else:
            self._hidden_groups.add(group_id)
        self._apply_visibility()
        self._changed()

    def solo_group(self, group_id: int) -> None:
        """Show only ``group_id``, hiding every other segment."""
        grouping = self.session.active_grouping
        if grouping is None:
            return
        self._hidden_groups = {int(g) for g in grouping.group_ids() if int(g) != group_id}
        self._apply_visibility()
        self._changed()

    def show_all_groups(self) -> None:
        """Make every segment visible again."""
        self._hidden_groups.clear()
        self._apply_visibility()
        self._changed()

    def reset_visibility(self) -> None:
        """Drop all hidden state (e.g. when the active grouping changes)."""
        self._hidden_groups.clear()
        self._apply_visibility()

    def _apply_visibility(self) -> None:
        grouping = self.session.active_grouping
        if grouping is None or not self._hidden_groups:
            self.viewer.set_visible_mask(None)
            return
        hidden = np.isin(grouping.group_id, list(self._hidden_groups))
        self.viewer.set_visible_mask(~hidden)

    # -- class (schema) editing -------------------------------------------

    def add_class(self, name: str, color=None) -> int:
        """Add a class to the palette and make it the active brush. Returns its id."""
        cls = self.session.schema.add_class(name, color)
        self.session.active_class = cls.id
        self._changed()
        return cls.id

    def rename_class(self, class_id: int, name: str) -> None:
        self.session.schema.rename(class_id, name)
        self._changed()

    def set_class_color(self, class_id: int, color) -> None:
        self.session.schema.set_color(class_id, color)
        self.refresh_display()  # recolour that class's points
        self._changed()

    def remove_class(self, class_id: int) -> None:
        """Remove a class; its points fall back to ``unlabeled``."""
        schema = self.session.schema
        if class_id == schema.unlabeled_id:
            return
        labels = self.session.cloud.ensure_labels(schema.unlabeled_id)
        labels[labels == class_id] = schema.unlabeled_id
        schema.remove(class_id)
        if self.session.active_class == class_id:
            self.session.active_class = schema.unlabeled_id
        self.refresh_display()
        self._changed()

    # -- read model -------------------------------------------------------

    def cloud_xyz(self) -> np.ndarray:
        """The point positions ``(N, 3)`` float32 (served once to a front-end)."""
        return self.session.cloud.xyz

    def label_array(self) -> np.ndarray:
        """The current per-point labels ``(N,)`` int32."""
        return self.session.cloud.ensure_labels(self.session.schema.unlabeled_id)

    def grouping_array(self) -> np.ndarray | None:
        """The active grouping ``group_id (N,)`` int32, or ``None`` if no grouping."""
        grouping = self.session.active_grouping
        return None if grouping is None else grouping.group_id

    def selection_indices(self) -> np.ndarray:
        """The currently selected point indices ``(M,)`` int64."""
        return self.session.selection.indices

    def snapshot(self) -> Snapshot:
        """A flat, serializable read-model of the session for any front-end."""
        session = self.session
        schema = session.schema
        classes = [ClassInfo(c.id, c.name, c.color) for c in schema.classes]
        groupings = [GroupingInfo(i, g.source, g.n_groups) for i, g in enumerate(session.groupings)]
        active_index = session.active_grouping_index
        active_info = groupings[active_index] if active_index is not None else None

        segments: list[SegmentInfo] = []
        has_suggestions = False
        active = session.active_grouping
        if active is not None:
            suggested = active.suggested_labels or {}
            has_suggestions = bool(suggested)
            for gid in active.group_ids():
                gid = int(gid)
                segments.append(
                    SegmentInfo(
                        id=gid,
                        count=int(active.indices_of(gid).size),
                        color=group_color(gid),
                        suggested=suggested.get(gid),
                        visible=gid not in self._hidden_groups,
                    )
                )

        return Snapshot(
            classes=classes,
            active_class=session.active_class,
            unlabeled_id=schema.unlabeled_id,
            display_mode=self.display_mode,
            selection_count=session.selection.count,
            groupings=groupings,
            active_grouping_index=active_index,
            active_grouping=active_info,
            segments=segments,
            has_suggestions=has_suggestions,
        )

    # -- helpers ----------------------------------------------------------

    def _changed(self) -> None:
        if self._on_state_changed is not None:
            self._on_state_changed()
