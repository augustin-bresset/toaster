"""The Toaster main window: 3-D viewport, side panels, menus and shortcuts."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QAction, QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QWidget,
)

from toaster.core import LabelSchema, Session
from toaster.io import load_cloud, supported_extensions
from toaster.persistence import LabelStore, SessionStore
from toaster.viewer import PyVistaViewer

from .controller import InteractionController
from .panels import ClassPalette, DisplayPanel, LayersPanel, SegmenterPanel
from .schema_loader import builtin_schema

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """Top-level window. Owns the viewer, the session and the panels."""

    def __init__(self, schema: LabelSchema | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Toaster")
        self.resize(1280, 800)

        self.schema = schema or builtin_schema()
        self.label_store = LabelStore()
        self.session_store = SessionStore()
        self._session: Session | None = None
        self._controller: InteractionController | None = None

        from pyvistaqt import QtInteractor

        self._interactor = QtInteractor(self)
        self.setCentralWidget(self._interactor)
        self.viewer = PyVistaViewer(plotter=self._interactor)

        self.palette = ClassPalette(self.schema)
        self.segmenter_panel = SegmenterPanel()
        self.display_panel = DisplayPanel(
            point_size=self.viewer.point_size, as_spheres=self.viewer.render_points_as_spheres
        )
        self.layers_panel = LayersPanel()
        self._add_dock("Classes", self.palette)
        self._add_dock("Display", self.display_panel)
        self._add_dock("Segmenter", self.segmenter_panel)
        self._add_dock("Session", self.layers_panel)

        self.palette.class_selected.connect(self._on_class_selected)
        self.palette.color_changed.connect(self._on_class_color_changed)
        self.display_panel.point_size_changed.connect(lambda n: self.viewer.set_point_style(size=n))
        self.display_panel.spheres_toggled.connect(
            lambda b: self.viewer.set_point_style(as_spheres=b)
        )
        self.segmenter_panel.run_requested.connect(self._on_run_segmenter)
        self.layers_panel.active_grouping_changed.connect(self._on_active_grouping_changed)

        self._build_menus()
        self._build_toolbar()
        self._build_shortcuts()

        self._status_perm = QLabel("No cloud")
        self.statusBar().addPermanentWidget(self._status_perm)
        self.statusBar().showMessage("Open a point cloud to begin (Ctrl+O).")

    # -- construction helpers --------------------------------------------

    def _add_dock(self, title: str, widget: QWidget) -> None:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(_action(self, "&Open…", "Ctrl+O", self._on_open))
        file_menu.addAction(_action(self, "&Save labels", "Ctrl+S", self._on_save))
        file_menu.addSeparator()
        file_menu.addAction(_action(self, "&Quit", "Ctrl+Q", self.close))

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(_action(self, "&Undo", "Ctrl+Z", self._on_undo))
        edit_menu.addAction(_action(self, "&Redo", "Ctrl+Shift+Z", self._on_redo))
        edit_menu.addAction(_action(self, "&Clear selection", "Escape", self._on_clear))

        select_menu = self.menuBar().addMenu("&Select")
        select_menu.addAction(
            _action(self, "&Point mode", "P", lambda: self._set_pick_mode("point"))
        )
        select_menu.addAction(_action(self, "&Box mode", "B", lambda: self._set_pick_mode("box")))

        view_menu = self.menuBar().addMenu("&View")
        for mode, label in [
            ("labels", "Colour by &labels"),
            ("grouping", "Colour by &grouping"),
            ("intensity", "Colour by &intensity"),
            ("height", "Colour by &height"),
        ]:
            view_menu.addAction(
                _action(self, label, None, lambda _=False, m=mode: self._set_mode(m))
            )
        view_menu.addSeparator()
        view_menu.addAction(_action(self, "&Reset camera", None, self._reset_camera))

    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(bar)
        bar.addAction(_action(self, "Open", "Ctrl+O", self._on_open))
        bar.addAction(_action(self, "Save", "Ctrl+S", self._on_save))
        bar.addSeparator()
        bar.addAction(_action(self, "Point", None, lambda: self._set_pick_mode("point")))
        bar.addAction(_action(self, "Box", None, lambda: self._set_pick_mode("box")))
        bar.addSeparator()
        for mode in ("labels", "grouping", "intensity", "height"):
            bar.addAction(
                _action(self, mode.capitalize(), None, lambda _=False, m=mode: self._set_mode(m))
            )
        bar.addSeparator()
        bar.addAction(_action(self, "Undo", "Ctrl+Z", self._on_undo))
        bar.addAction(_action(self, "Redo", "Ctrl+Shift+Z", self._on_redo))

    def _build_shortcuts(self) -> None:
        # Digit keys assign the i-th class to the current selection.
        for digit in range(1, 10):
            QShortcut(
                QKeySequence(str(digit)), self, activated=lambda d=digit: self._assign_digit(d)
            )

    # -- file actions -----------------------------------------------------

    def _on_open(self) -> None:
        exts = " ".join(f"*{e}" for e in supported_extensions())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open point cloud", "", f"Point clouds ({exts})"
        )
        if path:
            self.open_cloud(path)

    def open_cloud(self, path: str | Path) -> None:
        """Load a cloud, restore any saved labels, and start an annotation session."""
        try:
            cloud = load_cloud(path)
        except Exception as exc:  # surface load errors instead of crashing
            QMessageBox.critical(self, "Open failed", str(exc))
            return

        if cloud.source is not None:
            stored = self.label_store.load(cloud.source)
            if stored is not None and stored.shape == (cloud.n,):
                cloud.labels = stored
        cloud.ensure_labels(self.schema.unlabeled_id)

        self._session = Session(cloud, self.schema)
        self._controller = InteractionController(
            self._session, self.viewer, on_state_changed=self._refresh_panels
        )
        self._controller.refresh_display()
        self.viewer.reset_camera()
        self._refresh_panels()
        self.setWindowTitle(f"Toaster — {Path(path).name}")
        self.statusBar().showMessage(f"Loaded {cloud.n:,} points from {Path(path).name}")

    def _on_save(self) -> None:
        if self._session is None:
            return
        cloud = self._session.cloud
        if cloud.source is None:
            QMessageBox.information(self, "No source", "This cloud has no path to save beside.")
            return
        out = self.label_store.save(cloud.source, cloud.labels)
        self.statusBar().showMessage(f"Saved labels → {out.name}")

    # -- edit actions -----------------------------------------------------

    def _on_undo(self) -> None:
        if self._controller:
            self._controller.undo()

    def _on_redo(self) -> None:
        if self._controller:
            self._controller.redo()

    def _on_clear(self) -> None:
        if self._controller:
            self._controller.clear_selection()

    def _assign_digit(self, digit: int) -> None:
        if self._controller is None or digit - 1 >= len(self.schema.classes):
            return
        cls = self.schema.classes[digit - 1]
        self.palette.set_active(cls.id)
        self._controller.assign(cls.id)
        self.statusBar().showMessage(f"Assigned '{cls.name}'")

    # -- panel signals ----------------------------------------------------

    def _on_class_selected(self, class_id: int) -> None:
        if self._controller:
            self._controller.set_active_class(class_id)

    def _on_class_color_changed(self, class_id: int, color: tuple) -> None:
        self.schema.set_color(class_id, color)
        if self._controller:
            self._controller.refresh_display()
        self.statusBar().showMessage(f"Recoloured '{self.schema.get(class_id).name}'")

    def _on_run_segmenter(self, segmenter, scope_to_selection: bool) -> None:
        if self._controller is None:
            return
        self.statusBar().showMessage(f"Running {segmenter.name}…")
        self._controller.run_segmenter(segmenter, scope_to_selection=scope_to_selection)
        grouping = self._session.active_grouping
        self.statusBar().showMessage(f"{segmenter.name}: {grouping.n_groups} groups")

    def _on_active_grouping_changed(self, index) -> None:
        if self._session is None:
            return
        self._session.set_active_grouping(index)
        if self._controller.display_mode == "grouping":
            self._controller.refresh_display()

    # -- view actions -----------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        if self._controller:
            self._controller.set_display_mode(mode)

    def _set_pick_mode(self, mode: str) -> None:
        if self._controller:
            self._controller.set_pick_mode(mode)
            self.statusBar().showMessage(f"{mode.capitalize()} selection mode")

    def _reset_camera(self) -> None:
        self.viewer.reset_camera()
        self.viewer.render()

    # -- misc -------------------------------------------------------------

    def _refresh_panels(self) -> None:
        if self._session is None:
            return
        self.layers_panel.refresh(self._session)
        sel = self._session.selection
        try:
            class_name = self.schema.get(self._session.active_class).name
        except KeyError:
            class_name = str(self._session.active_class)
        mode = self._controller.display_mode if self._controller else "labels"
        self._status_perm.setText(f"Sel: {sel.count:,}  |  Class: {class_name}  |  View: {mode}")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        # Closing the interactor cleanly avoids a VTK segfault on exit.
        self._interactor.close()
        super().closeEvent(event)


def _action(parent, text, shortcut, slot) -> QAction:
    action = QAction(text, parent)
    if shortcut:
        action.setShortcut(QKeySequence(shortcut))
    action.triggered.connect(slot)
    return action
