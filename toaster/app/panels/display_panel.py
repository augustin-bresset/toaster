"""Point rendering controls: size and shape (flat squares vs spheres)."""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

__all__ = ["DisplayPanel"]


class DisplayPanel(QWidget):
    """Lets the user tune how points are drawn."""

    point_size_changed = Signal(int)
    spheres_toggled = Signal(bool)

    def __init__(
        self, point_size: int = 3, as_spheres: bool = False, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Point rendering"))

        row = QHBoxLayout()
        row.addWidget(QLabel("Size"))
        self._size = QSlider(Qt.Orientation.Horizontal)
        self._size.setRange(1, 15)
        self._size.setValue(point_size)
        self._size_value = QLabel(str(point_size))
        self._size.valueChanged.connect(self._on_size)
        row.addWidget(self._size)
        row.addWidget(self._size_value)
        layout.addLayout(row)

        self._spheres = QCheckBox("Round points (spheres)")
        self._spheres.setChecked(as_spheres)
        self._spheres.toggled.connect(self.spheres_toggled)
        layout.addWidget(self._spheres)
        layout.addStretch(1)

    def _on_size(self, value: int) -> None:
        self._size_value.setText(str(value))
        self.point_size_changed.emit(value)
