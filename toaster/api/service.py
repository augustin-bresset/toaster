"""The headless annotation service the HTTP layer drives.

Wraps a :class:`~toaster.interaction.InteractionController` (with a
:class:`~toaster.viewer.NullViewer`, since the web client renders the cloud
itself) and exposes high-level operations plus a serializable state. This is the
exact same engine the Qt app uses — only the front differs.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from toaster.builtin_schemas import builtin_schema
from toaster.core import LabelSchema, Session
from toaster.interaction import InteractionController
from toaster.io import load_cloud
from toaster.persistence import LabelStore
from toaster.segment import get_segmenter, segmenter_specs
from toaster.viewer import NullViewer

from .serialize import encode_array

__all__ = ["AnnotationService"]


class AnnotationService:
    """One annotation session, driven over HTTP."""

    def __init__(self, schema: LabelSchema | None = None) -> None:
        self.schema = schema or builtin_schema()
        self.label_store = LabelStore()
        self._session: Session | None = None
        self._controller: InteractionController | None = None

    # -- lifecycle --------------------------------------------------------

    def open_cloud(self, path: str | Path) -> dict[str, Any]:
        """Load a cloud (restoring saved labels) and start a session."""
        cloud = load_cloud(path)
        if cloud.source is not None:
            stored = self.label_store.load(cloud.source)
            if stored is not None and stored.shape == (cloud.n,):
                cloud.labels = stored
        cloud.ensure_labels(self.schema.unlabeled_id)
        self._session = Session(cloud, self.schema)
        self._controller = InteractionController(self._session, NullViewer())
        # Adopt the first real class as the active brush (skip 'unlabeled').
        for c in self.schema.classes:
            if c.id != self.schema.unlabeled_id:
                self._controller.set_active_class(c.id)
                break
        return self.meta()

    @property
    def has_cloud(self) -> bool:
        return self._controller is not None

    def _ctl(self) -> InteractionController:
        if self._controller is None:
            raise RuntimeError("no cloud loaded")
        return self._controller

    # -- reads ------------------------------------------------------------

    def meta(self) -> dict[str, Any]:
        """Small metadata: point count and available segmenters."""
        return {
            "n": self._session.cloud.n if self._session else 0,
            "segmenters": segmenter_specs(),  # [{name, params: [...]}]
            "source": str(self._session.cloud.source) if self._session else None,
        }

    def cloud(self) -> dict[str, Any]:
        """The geometry (and feature channels), encoded for the client."""
        ctl = self._ctl()
        features = {k: encode_array(v) for k, v in ctl.session.cloud.features.items()}
        return {"xyz": encode_array(ctl.cloud_xyz()), "features": features}

    def state(self) -> dict[str, Any]:
        """Everything the client needs to (re)colour: snapshot + bulk arrays."""
        ctl = self._ctl()
        grouping = ctl.grouping_array()
        return {
            "snapshot": asdict(ctl.snapshot()),
            "labels": encode_array(ctl.label_array()),
            "grouping": encode_array(grouping) if grouping is not None else None,
            # int32 so it maps to a JS Int32Array (int64 has no plain TypedArray).
            "selection": encode_array(ctl.selection_indices().astype(np.int32)),
        }

    # -- commands ---------------------------------------------------------

    def pick(self, index: int, modifiers: list[str] | None = None) -> dict[str, Any]:
        self._ctl().on_pick(index, _mods(modifiers))
        return self.state()

    def box(self, indices: list[int], modifiers: list[str] | None = None) -> dict[str, Any]:
        self._ctl().on_box(np.asarray(indices, dtype=np.int64), _mods(modifiers))
        return self.state()

    def assign(self, class_id: int | None = None) -> dict[str, Any]:
        self._ctl().assign(class_id)
        return self.state()

    def set_active_class(self, class_id: int) -> dict[str, Any]:
        self._ctl().set_active_class(class_id)
        return self.state()

    def set_display_mode(self, mode: str) -> dict[str, Any]:
        self._ctl().set_display_mode(mode)
        return self.state()

    def undo(self) -> dict[str, Any]:
        self._ctl().undo()
        return self.state()

    def redo(self) -> dict[str, Any]:
        self._ctl().redo()
        return self.state()

    def clear_selection(self) -> dict[str, Any]:
        self._ctl().clear_selection()
        return self.state()

    def run_segmenter(
        self, name: str, params: dict | None = None, scope_to_selection: bool = True
    ) -> dict[str, Any]:
        segmenter = get_segmenter(name, **(params or {}))
        self._ctl().run_segmenter(segmenter, scope_to_selection=scope_to_selection)
        return self.state()

    def select_group(self, group_id: int, modifiers: list[str] | None = None) -> dict[str, Any]:
        self._ctl().select_group(group_id, _mods(modifiers))
        return self.state()

    def assign_group(self, group_id: int, class_id: int | None = None) -> dict[str, Any]:
        self._ctl().assign_group(group_id, class_id)
        return self.state()

    def apply_suggested(self, group_id: int | None = None) -> dict[str, Any]:
        self._ctl().apply_suggested(group_id)
        return self.state()

    def assign_visible_groups(self, class_id: int | None = None) -> dict[str, Any]:
        self._ctl().assign_visible_groups(class_id)
        return self.state()

    def set_group_visibility(self, group_id: int, visible: bool) -> dict[str, Any]:
        self._ctl().set_group_visibility(group_id, visible)
        return self.state()

    def show_all_groups(self) -> dict[str, Any]:
        self._ctl().show_all_groups()
        return self.state()

    def hide_all_groups(self) -> dict[str, Any]:
        self._ctl().hide_all_groups()
        return self.state()

    def clear_grouping(self) -> dict[str, Any]:
        self._ctl().clear_grouping()
        return self.state()

    def add_class(self, name: str, color=None) -> dict[str, Any]:
        self._ctl().add_class(name, color)
        return self.state()

    def rename_class(self, class_id: int, name: str) -> dict[str, Any]:
        self._ctl().rename_class(class_id, name)
        return self.state()

    def set_class_color(self, class_id: int, color) -> dict[str, Any]:
        self._ctl().set_class_color(class_id, color)
        return self.state()

    def remove_class(self, class_id: int) -> dict[str, Any]:
        self._ctl().remove_class(class_id)
        return self.state()

    def save(self) -> dict[str, Any]:
        cloud = self._ctl().session.cloud
        if cloud.source is None:
            raise RuntimeError("cloud has no source path to save beside")
        out = self.label_store.save(cloud.source, cloud.labels)
        return {"saved": str(out)}


def _mods(modifiers: list[str] | None) -> frozenset:
    return frozenset(modifiers or ())
