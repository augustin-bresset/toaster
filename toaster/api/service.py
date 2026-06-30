"""The headless annotation service the HTTP layer drives.

Wraps a :class:`~toaster.interaction.InteractionController` (with a
:class:`~toaster.viewer.NullViewer`, since the web client renders the cloud
itself) and exposes high-level operations plus a serializable state. This is the
exact same engine the Qt app uses — only the front differs.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from toaster.builtin_schemas import builtin_schema
from toaster.core import LabelSchema, Session
from toaster.interaction import InteractionController
from toaster.io import load_cloud
from toaster.persistence import LabelStore, SchemaStore
from toaster.segment import get_segmenter, segmenter_specs
from toaster.viewer import NullViewer

from .serialize import encode_array

__all__ = ["AnnotationService"]


class AnnotationService:
    """One annotation session, driven over HTTP."""

    def __init__(self, schema: LabelSchema | None = None) -> None:
        self.schema = schema or builtin_schema()
        self.label_store = LabelStore()
        self.schema_store = SchemaStore()
        self._session: Session | None = None
        self._controller: InteractionController | None = None

    # -- lifecycle --------------------------------------------------------

    def open_cloud(self, path: str | Path) -> dict[str, Any]:
        """Load a cloud and start a session, restoring any saved labels and schema.

        A cloud that has been labelled before carries two sidecars beside it:
        ``<cloud>.toaster.npy`` (the labels) and ``<cloud>.toaster.schema.yaml``
        (the class names/colours). Both are restored so reopening the cloud
        continues exactly where it was left, with the right palette.
        """
        cloud = load_cloud(path)
        schema = self.schema
        if cloud.source is not None:
            stored_schema = self.schema_store.load(cloud.source)
            if stored_schema is not None:
                schema = stored_schema
            stored = self.label_store.load(cloud.source)
            if stored is not None and stored.shape == (cloud.n,):
                cloud.labels = stored
        cloud.ensure_labels(schema.unlabeled_id)
        self._session = Session(cloud, schema)
        self._controller = InteractionController(self._session, NullViewer())
        # Adopt the first real class as the active brush (skip 'unlabeled').
        for c in schema.classes:
            if c.id != schema.unlabeled_id:
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

    def browse(self, path: str | None = None) -> dict[str, Any]:
        """List a directory (folders first), flagging which files Toaster can open.

        ``path`` ``None`` starts from the open cloud's folder, else the working
        directory — so launching ``toaster-web`` with no file lands you somewhere
        useful. Dotfiles are hidden. Local use only (server binds to 127.0.0.1).
        """
        from toaster.io import supported_extensions

        exts = set(supported_extensions())
        base = (Path(path).expanduser() if path else self._start_dir()).resolve()
        if not base.is_dir():
            raise ValueError(f"not a directory: {base}")
        entries = []
        for child in sorted(base.iterdir(), key=lambda c: (not _is_dir(c), c.name.lower())):
            if child.name.startswith("."):
                continue
            is_dir = _is_dir(child)
            entries.append({
                "name": child.name, "path": str(child), "is_dir": is_dir,
                "openable": is_dir or child.suffix.lower() in exts,
            })  # fmt: skip
        parent = str(base.parent) if base.parent != base else None
        return {"path": str(base), "parent": parent, "entries": entries,
                "extensions": sorted(exts)}  # fmt: skip

    def _start_dir(self) -> Path:
        """Where the file browser opens by default."""
        if self._session is not None and self._session.cloud.source is not None:
            return Path(self._session.cloud.source).expanduser().resolve().parent
        return Path.cwd()

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

    def save(self, path: str | Path | None = None) -> dict[str, Any]:
        """Write the labels and the schema sidecars.

        With ``path`` given they are written beside that base path
        (``<path>.toaster.npy`` and ``<path>.toaster.schema.yaml``); otherwise
        they go beside the cloud's own source. Reopening a cloud only finds the
        sidecars that sit beside it, so the default keeps reopening seamless.
        """
        session = self._ctl().session
        cloud = session.cloud
        base = Path(path) if path is not None else cloud.source
        if base is None:
            raise RuntimeError("no save path given and the cloud has no source path")
        labels = cloud.ensure_labels(session.schema.unlabeled_id)
        labels_out = self.label_store.save(base, labels)
        schema_out = self.schema_store.save(base, session.schema, cloud_path=cloud.source)
        return {"saved": str(labels_out), "schema": str(schema_out)}

    def apairo_info(self) -> dict[str, Any]:
        """Tell the UI whether the open cloud can be written back as an apairo channel.

        Detection is filesystem-only (no ``apairo`` import), so the option can be
        *offered* even when the package is missing; ``apairo_installed`` flags
        whether the write itself is currently possible.
        """
        from toaster.io.apairo_dataset import detect_apairo_channel

        installed = importlib.util.find_spec("apairo") is not None
        target = detect_apairo_channel(self._session.cloud.source) if self._session else None
        if target is None:
            return {"is_apairo": False, "apairo_installed": installed}
        return {
            "is_apairo": True,
            "apairo_installed": installed,
            "seq_dir": target.seq_dir,
            "source_channel": target.source_channel,
            "stem": target.stem,
            "suggested_channel": "ground_truth",
        }

    def apairo_nav(self) -> dict[str, Any]:
        """Dataset → sequence → channel → frame position of the open cloud.

        ``is_dataset`` is False when the cloud isn't inside an apairo dataset, so
        the UI can hide the navigation menu. Otherwise it carries the sequences,
        the sequence's point channels and the current frame index/count.
        """
        from toaster.io.apairo_dataset import detect_apairo_nav

        nav = detect_apairo_nav(self._session.cloud.source) if self._session else None
        if nav is None:
            return {"is_dataset": False}
        stem = nav.frames[nav.frame_index] if 0 <= nav.frame_index < len(nav.frames) else None
        return {
            "is_dataset": True,
            "dataset_name": nav.dataset_name,
            "sequences": nav.sequences,
            "sequence": nav.sequence,
            "channels": nav.channels,
            "channel": nav.channel,
            "frame_index": nav.frame_index,
            "frame_count": len(nav.frames),
            "frame_stem": stem,
        }

    def apairo_open(self, sequence: str, channel: str, frame_index: int) -> dict[str, Any]:
        """Open a specific dataset frame (``frame_index`` clamped to the channel)."""
        from toaster.io.apairo_dataset import detect_apairo_nav, frame_path

        nav = detect_apairo_nav(self._session.cloud.source) if self._session else None
        if nav is None:
            raise RuntimeError("the open cloud is not inside an apairo dataset")
        path = frame_path(nav.dataset_root, sequence, channel, frame_index)
        if path is None or not path.is_file():
            raise RuntimeError(f"no frame {frame_index} in {sequence}/{channel}")
        return self.open_cloud(str(path))

    def save_apairo(self, channel: str = "ground_truth") -> dict[str, Any]:
        """Write the labels back into the apairo dataset as ``channel``.

        Labels are realigned to the source frame (dropped NaN points become
        ``unlabeled``) so the channel lines up index-for-index with the cloud it
        was derived from, the same convention apairo's own preprocess channels use.
        """
        from toaster.io.apairo_dataset import (
            detect_apairo_channel,
            frame_timestamp,
            write_labels_channel,
        )

        session = self._ctl().session
        cloud = session.cloud
        target = detect_apairo_channel(cloud.source)
        if target is None:
            raise RuntimeError("this cloud is not inside an apairo dataset")
        timestamp = frame_timestamp(target.seq_dir, target.source_channel, target.stem)
        if timestamp is None:
            raise RuntimeError(f"no timestamp found for frame {target.stem}")
        labels = cloud.ensure_labels(session.schema.unlabeled_id)
        full = cloud.to_source_frame(labels, fill=session.schema.unlabeled_id)
        out = write_labels_channel(target, full, timestamp, channel=channel)
        return {"written": str(out), "channel": channel, "points": int(full.size)}

    def make_dir(self, path: str | Path) -> dict[str, Any]:
        """Create a folder (the Save dialog's "New folder"). Local use only.

        Returns the resolved path so the dialog can navigate straight into it.
        """
        target = Path(path).expanduser()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"could not create folder {target}: {exc}") from exc
        return {"path": str(target.resolve())}


def _mods(modifiers: list[str] | None) -> frozenset:
    return frozenset(modifiers or ())


def _is_dir(p: Path) -> bool:
    try:
        return p.is_dir()
    except OSError:  # e.g. a broken symlink or permission error
        return False
