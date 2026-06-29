"""The Groups panel: every segment of the active grouping, one row each.

This is the payoff of running a segmenter: each cluster/segment the algorithm
produced is listed here. Click a row to select that whole segment in the 3-D
view; the buttons label it (with the active class, or — for a model that
predicted classes — with the segment's *suggested* class).

Like every panel, it only emits intent; the window routes it to the controller.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from toaster.interaction import Snapshot

__all__ = ["GroupsPanel"]


class GroupsPanel(QWidget):
    """Lists the segments of the active grouping and labels them."""

    group_selected = Signal(int)  # group id -> select that segment
    assign_active_requested = Signal(int)  # group id -> label with active class
    assign_suggested_requested = Signal(int)  # group id -> label with its suggested class
    assign_all_suggested_requested = Signal()  # label every suggested segment

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._header = QLabel("Segments")
        layout.addWidget(self._header)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_current)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list)

        row = QHBoxLayout()
        self._assign_active = QPushButton("Assign active class")
        self._assign_active.clicked.connect(self._emit_assign_active)
        self._assign_suggested = QPushButton("Assign suggested")
        self._assign_suggested.clicked.connect(self._emit_assign_suggested)
        row.addWidget(self._assign_active)
        row.addWidget(self._assign_suggested)
        layout.addLayout(row)

        self._assign_all = QPushButton("Assign all suggested")
        self._assign_all.clicked.connect(self.assign_all_suggested_requested)
        layout.addWidget(self._assign_all)

        # Track what is currently rendered so a plain selection/label change
        # (which calls refresh too) does not rebuild the list and drop the row
        # the user just clicked.
        self._signature: object = None
        self.refresh(None)

    def refresh(self, snap: Snapshot | None) -> None:
        """Rebuild the segment list, but only when the active grouping changed."""
        segments = snap.segments if snap is not None else []
        # A signature over what is shown: rebuild only when it actually changes.
        signature = (
            None
            if snap is None or snap.active_grouping is None
            else (snap.active_grouping_index, tuple((s.id, s.suggested) for s in segments))
        )
        if signature == self._signature:
            return
        self._signature = signature

        self._list.blockSignals(True)
        self._list.clear()

        if signature is None:
            self._header.setText("Segments — run a segmenter to populate")
            self._list.blockSignals(False)
            self._set_buttons_enabled(has_groups=False, has_suggestions=False)
            return

        for seg in segments:
            text = f"#{seg.id}  ·  {seg.count:,} pts"
            if seg.suggested is not None:
                text += f"  →  {snap.class_name(seg.suggested)}"
            item = QListWidgetItem(text)
            item.setIcon(_swatch(seg.color))
            item.setData(Qt.ItemDataRole.UserRole, seg.id)
            self._list.addItem(item)

        info = snap.active_grouping
        self._header.setText(f"{info.n_groups} segments  ·  {info.source}")
        self._list.blockSignals(False)
        self._set_buttons_enabled(has_groups=True, has_suggestions=snap.has_suggestions)

    # -- internals --------------------------------------------------------

    def _current_group(self) -> int | None:
        item = self._list.currentItem()
        return int(item.data(Qt.ItemDataRole.UserRole)) if item is not None else None

    def _set_buttons_enabled(self, *, has_groups: bool, has_suggestions: bool) -> None:
        self._assign_active.setEnabled(has_groups)
        self._assign_suggested.setEnabled(has_groups)
        self._assign_all.setEnabled(has_suggestions)

    def _on_current(self, item: QListWidgetItem | None) -> None:
        if item is not None:
            self.group_selected.emit(int(item.data(Qt.ItemDataRole.UserRole)))

    def _on_double_click(self, item: QListWidgetItem) -> None:
        self.assign_active_requested.emit(int(item.data(Qt.ItemDataRole.UserRole)))

    def _emit_assign_active(self) -> None:
        gid = self._current_group()
        if gid is not None:
            self.assign_active_requested.emit(gid)

    def _emit_assign_suggested(self) -> None:
        gid = self._current_group()
        if gid is not None:
            self.assign_suggested_requested.emit(gid)


def _swatch(color: tuple[int, int, int], size: int = 14) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(QColor(*color))
    return QIcon(pix)
