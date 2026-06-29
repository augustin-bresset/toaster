"""The class palette: pick the active class, and edit the labelling classes.

This is where the user configures *what* they label the cloud into: add a class,
rename it, recolour it (double-click), or remove it. The panel only emits intent;
the :class:`~toaster.app.main_window.MainWindow` owns the schema and applies the
change, then calls :meth:`rebuild` to refresh the list.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from toaster.core import LabelSchema

__all__ = ["ClassPalette"]


class ClassPalette(QWidget):
    """A list of classes with colour swatches and Add / Rename / Remove controls.

    Selecting a row sets the active class; double-clicking opens a colour picker.
    The Add / Rename / Remove buttons emit intent signals — the window mutates the
    schema and calls :meth:`rebuild`.
    """

    class_selected = Signal(int)
    color_changed = Signal(int, tuple)  # (class_id, (r, g, b))
    class_added = Signal(str, object)  # (name, (r, g, b) | None for auto colour)
    class_renamed = Signal(int, str)  # (class_id, name)
    class_removed = Signal(int)  # (class_id)

    def __init__(self, schema: LabelSchema, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._schema = schema
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Classes  (double-click to recolour)"))

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._emit_current)
        self._list.itemDoubleClicked.connect(self._edit_color)
        layout.addWidget(self._list)

        buttons = QHBoxLayout()
        for text, slot in (
            ("Add", self._add_class),
            ("Rename", self._rename_class),
            ("Remove", self._remove_class),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        layout.addLayout(buttons)

        self.rebuild(schema)

    # -- population -------------------------------------------------------

    def rebuild(self, schema: LabelSchema) -> None:
        """Repopulate the list from ``schema``, preserving the active row if it survives."""
        self._schema = schema
        keep = self.active_class()
        self._list.blockSignals(True)
        self._list.clear()
        for cls in schema.classes:
            item = QListWidgetItem(f"{cls.id}  {cls.name}")
            item.setIcon(_swatch(cls.color))
            item.setData(Qt.ItemDataRole.UserRole, cls.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, cls.color)
            self._list.addItem(item)
        self._list.blockSignals(False)
        if keep is not None and self._item_for(keep) is not None:
            self.set_active(keep)
        else:
            self._select_default()

    def _select_default(self) -> None:
        """Select the first real class (skip ``unlabeled``) so the brush paints."""
        for row in range(self._list.count()):
            cid = int(self._list.item(row).data(Qt.ItemDataRole.UserRole))
            if cid != self._schema.unlabeled_id:
                self._list.setCurrentRow(row)
                return
        if self._list.count():
            self._list.setCurrentRow(0)

    def active_class(self) -> int | None:
        """The currently selected class id, or ``None`` when the list is empty."""
        item = self._list.currentItem()
        return int(item.data(Qt.ItemDataRole.UserRole)) if item is not None else None

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

    # -- editing actions --------------------------------------------------

    def _add_class(self) -> None:
        name, ok = QInputDialog.getText(self, "Add class", "Class name:")
        name = name.strip()
        if not (ok and name):
            return
        chosen = QColorDialog.getColor(QColor(200, 200, 200), self, f"Colour for '{name}'")
        rgb = (chosen.red(), chosen.green(), chosen.blue()) if chosen.isValid() else None
        self.class_added.emit(name, rgb)

    def _rename_class(self) -> None:
        class_id = self.active_class()
        if class_id is None:
            return
        current = self._schema.get(class_id).name
        name, ok = QInputDialog.getText(self, "Rename class", "Class name:", text=current)
        name = name.strip()
        if ok and name and name != current:
            self.class_renamed.emit(class_id, name)

    def _remove_class(self) -> None:
        class_id = self.active_class()
        if class_id is None:
            return
        if class_id == self._schema.unlabeled_id:
            QMessageBox.information(
                self, "Cannot remove", "The 'unlabeled' class cannot be removed."
            )
            return
        name = self._schema.get(class_id).name
        confirm = QMessageBox.question(
            self,
            "Remove class",
            f"Remove class '{name}'? Points already labelled with it become unlabeled.",
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.class_removed.emit(class_id)

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
