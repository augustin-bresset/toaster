"""Live session stats and the active-grouping selector."""

from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from toaster.core import Session

__all__ = ["LayersPanel"]


class LayersPanel(QWidget):
    """Shows selection/label stats and lets the user pick which grouping is active."""

    active_grouping_changed = Signal(object)  # int | None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Groupings"))
        self._combo = QComboBox()
        self._combo.currentIndexChanged.connect(self._emit_choice)
        layout.addWidget(self._combo)
        self._stats = QLabel()
        self._stats.setWordWrap(True)
        layout.addWidget(self._stats)
        layout.addStretch(1)
        self._suppress = False

    def refresh(self, session: Session) -> None:
        """Rebuild the grouping list and stats from the session state."""
        self._suppress = True
        self._combo.clear()
        self._combo.addItem("None (pick single points)", userData=None)
        for i, g in enumerate(session.groupings):
            self._combo.addItem(f"#{i} {g.source} ({g.n_groups} groups)", userData=i)
        active = session.active_grouping_index
        self._combo.setCurrentIndex(0 if active is None else active + 1)
        self._suppress = False

        self._stats.setText(
            f"Selected: {session.selection.count} pts\n"
            f"Active class: {session.active_class}\n"
            f"Groupings: {len(session.groupings)}"
        )

    def _emit_choice(self, _index: int) -> None:
        if self._suppress:
            return
        self.active_grouping_changed.emit(self._combo.currentData())
