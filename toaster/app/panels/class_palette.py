"""The class palette: pick the active class, and edit class colours."""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QColorDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from toaster.core import LabelSchema

__all__ = ["ClassPalette"]


class ClassPalette(QWidget):
    """A list of classes with colour swatches.

    Selecting a row sets the active class; double-clicking opens a colour picker
    to recolour that class.
    """

    class_selected = Signal(int)
    color_changed = Signal(int, tuple)  # (class_id, (r, g, b))

    def __init__(self, schema: LabelSchema, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Classes  (double-click to recolour)"))
        self._list = QListWidget()
        layout.addWidget(self._list)

        for cls in schema.classes:
            item = QListWidgetItem(f"{cls.id}  {cls.name}")
            item.setIcon(_swatch(cls.color))
            item.setData(Qt.ItemDataRole.UserRole, cls.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, cls.color)
            self._list.addItem(item)

        self._list.currentItemChanged.connect(self._emit_current)
        self._list.itemDoubleClicked.connect(self._edit_color)
        if self._list.count():
            self._list.setCurrentRow(0)

    def set_active(self, class_id: int) -> None:
        """Select the row for ``class_id``."""
        item = self._item_for(class_id)
        if item is not None:
            self._list.setCurrentRow(self._list.row(item))

    def update_swatch(self, class_id: int, color: tuple[int, int, int]) -> None:
        """Refresh a class's colour swatch (e.g. after an external recolour)."""
        item = self._item_for(class_id)
        if item is not None:
            item.setIcon(_swatch(color))
            item.setData(Qt.ItemDataRole.UserRole + 1, color)

    def _edit_color(self, item: QListWidgetItem) -> None:
        class_id = int(item.data(Qt.ItemDataRole.UserRole))
        current = item.data(Qt.ItemDataRole.UserRole + 1) or (255, 255, 255)
        chosen = QColorDialog.getColor(QColor(*current), self, "Pick class colour")
        if chosen.isValid():
            rgb = (chosen.red(), chosen.green(), chosen.blue())
            self.update_swatch(class_id, rgb)
            self.color_changed.emit(class_id, rgb)

    def _emit_current(self, current: QListWidgetItem | None) -> None:
        if current is not None:
            self.class_selected.emit(int(current.data(Qt.ItemDataRole.UserRole)))

    def _item_for(self, class_id: int) -> QListWidgetItem | None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == class_id:
                return item
        return None


def _swatch(color: tuple[int, int, int], size: int = 14) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(QColor(*color))
    return QIcon(pix)
