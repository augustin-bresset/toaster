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

from toaster.core import LabelSchema, PointCloud, Session
from toaster.interaction import InteractionController
from toaster.io import load_cloud, supported_extensions
from toaster.persistence import LabelStore, SchemaStore, SessionStore
from toaster.viewer import PyVistaViewer

from .panels import ClassPalette, DisplayPanel, GroupsPanel, LayersPanel, SegmenterPanel
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
        self.schema_store = SchemaStore()
        self.session_store = SessionStore()
        self._cloud: PointCloud | None = None
        self._session: Session | None = None
        self._controller: InteractionController | None = None
        self._docks: dict[str, QDockWidget] = {}

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
        self.groups_panel = GroupsPanel()
        # Settings you set-and-leave go on the left; the segmentation workflow
        # goes on the right as a tab stack so one column never gets too dense.
        left = Qt.DockWidgetArea.LeftDockWidgetArea
        self._add_dock("Classes", self.palette, left)
        self._add_dock("Display", self.display_panel, left)
        seg = self._add_dock("Segmenter", self.segmenter_panel)
        grp = self._add_dock("Groups", self.groups_panel)
        ses = self._add_dock("Session", self.layers_panel)
        self.tabifyDockWidget(seg, grp)
        self.tabifyDockWidget(grp, ses)
        seg.raise_()

        self.palette.class_selected.connect(self._on_class_selected)
        self.palette.color_changed.connect(self._on_class_color_changed)
        self.palette.class_added.connect(self._on_class_added)
        self.palette.class_renamed.connect(self._on_class_renamed)
        self.palette.class_removed.connect(self._on_class_removed)
        self.display_panel.point_size_changed.connect(lambda n: self.viewer.set_point_style(size=n))
        self.display_panel.spheres_toggled.connect(
            lambda b: self.viewer.set_point_style(as_spheres=b)
        )
        self.segmenter_panel.run_requested.connect(self._on_run_segmenter)
        self.layers_panel.active_grouping_changed.connect(self._on_active_grouping_changed)
        self.groups_panel.group_selected.connect(self._on_group_selected)
        self.groups_panel.assign_active_requested.connect(self._on_assign_group_active)
        self.groups_panel.assign_suggested_requested.connect(self._on_assign_group_suggested)
        self.groups_panel.assign_all_suggested_requested.connect(self._on_assign_all_suggested)

        self._build_menus()
        self._build_toolbar()
        self._build_shortcuts()

        self._status_perm = QLabel("No cloud")
        self.statusBar().addPermanentWidget(self._status_perm)
        self.statusBar().showMessage("Open a point cloud to begin (Ctrl+O).")

    # -- construction helpers --------------------------------------------

    def _add_dock(self, title: str, widget: QWidget, area=None) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area or Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._docks[title] = dock
        return dock

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(_action(self, "&Open…", "Ctrl+O", self._on_open))
        file_menu.addAction(_action(self, "&Save labels", "Ctrl+S", self._on_save))
        file_menu.addSeparator()
        file_menu.addAction(_action(self, "&Load schema…", None, self._on_load_schema))
        file_menu.addAction(_action(self, "Save sc&hema…", None, self._on_save_schema))
        file_menu.addSeparator()
        file_menu.addAction(_action(self, "&Quit", "Ctrl+Q", self.close))

        edit_menu = self.menuBar().addMenu("&Edit")
        # Created once and shared with the toolbar so there is a clickable way to
        # label, not only the number-key shortcuts.
        self._assign_action = _action(self, "&Assign class to selection", "Return", self._on_assign)
        edit_menu.addAction(self._assign_action)
        edit_menu.addSeparator()
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

        # A closed dock is otherwise unreachable: give every panel a checkable
        # entry here so it can be hidden and brought back.
        panels_menu = self.menuBar().addMenu("&Panels")
        for title, dock in self._docks.items():
            action = dock.toggleViewAction()
            action.setText(title)
            panels_menu.addAction(action)

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
        bar.addAction(self._assign_action)  # label the selection with the active class
        bar.addSeparator()
        bar.addAction(_action(self, "Undo", "Ctrl+Z", self._on_undo))
        bar.addAction(_action(self, "Redo", "Ctrl+Shift+Z", self._on_redo))

    def _build_shortcuts(self) -> None:
        # A number key labels the selection with the class of that *id* — the
        # number shown beside each class in the Classes panel (0 = unlabeled).
        for digit in range(10):
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
        """Load a cloud, restore any saved labelling session, and start annotating."""
        try:
            cloud = load_cloud(path)
        except Exception as exc:  # surface load errors instead of crashing
            QMessageBox.critical(self, "Open failed", str(exc))
            return

        # Restore the labelling session saved beside this cloud: its schema first
        # (so the labels map to the right classes), then the labels themselves.
        if cloud.source is not None:
            saved_schema = self.schema_store.load(cloud.source)
            if saved_schema is not None:
                self._set_schema(saved_schema)
            stored = self.label_store.load(cloud.source)
            if stored is not None and stored.shape == (cloud.n,):
                cloud.labels = stored

        self._cloud = cloud
        self._start_session(cloud)
        self.setWindowTitle(f"Toaster — {Path(path).name}")
        self.statusBar().showMessage(f"Loaded {cloud.n:,} points from {Path(path).name}")

    def _start_session(self, cloud: PointCloud, *, reset_camera: bool = True) -> None:
        """(Re)build the session and controller for ``cloud`` under the current schema."""
        cloud.ensure_labels(self.schema.unlabeled_id)
        self._session = Session(cloud, self.schema)
        self._controller = InteractionController(
            self._session, self.viewer, on_state_changed=self._refresh_panels
        )
        # Adopt the class the palette already highlights as the active brush: its
        # initial selection signal fired during construction, before this session
        # (and the signal wiring) existed, so the session would otherwise default
        # to 'unlabeled' and the first Assign would paint nothing.
        active = self.palette.active_class()
        if active is not None:
            self._controller.set_active_class(active)
        self._controller.refresh_display()
        if reset_camera:
            self.viewer.reset_camera()
        self._refresh_panels()

    def _set_schema(self, schema: LabelSchema) -> None:
        """Swap in a new schema and refresh the palette (does not touch the session)."""
        self.schema = schema
        self.palette.rebuild(schema)

    def _on_save(self) -> None:
        if self._session is None:
            return
        cloud = self._session.cloud
        if cloud.source is None:
            QMessageBox.information(self, "No source", "This cloud has no path to save beside.")
            return
        out = self.label_store.save(cloud.source, cloud.labels)
        self.schema_store.save(cloud.source, self.schema)
        self.statusBar().showMessage(f"Saved labels + schema → {out.name}")

    def _on_load_schema(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load label schema", "", "Schema YAML (*.yaml *.yml)"
        )
        if not path:
            return
        try:
            schema = LabelSchema.from_yaml(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._set_schema(schema)
        if self._cloud is not None:  # re-bind the open cloud to the new palette
            self._start_session(self._cloud, reset_camera=False)
        self.statusBar().showMessage(f"Loaded schema with {len(schema)} classes")

    def _on_save_schema(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save label schema", "schema.yaml", "Schema YAML (*.yaml *.yml)"
        )
        if not path:
            return
        out = self.schema.to_yaml(path)
        self.statusBar().showMessage(f"Saved schema → {out.name}")

    # -- schema editing ---------------------------------------------------

    def _on_class_added(self, name: str, color) -> None:
        cls = self.schema.add_class(name, color)
        self.palette.rebuild(self.schema)
        self.palette.set_active(cls.id)  # make the new class the active brush
        self.statusBar().showMessage(f"Added class '{cls.name}' (id {cls.id})")

    def _on_class_renamed(self, class_id: int, name: str) -> None:
        self.schema.rename(class_id, name)
        self.palette.rebuild(self.schema)
        self.statusBar().showMessage(f"Renamed class {class_id} → '{name}'")

    def _on_class_removed(self, class_id: int) -> None:
        self.schema.remove(class_id)
        if self._session is not None and self._session.active_class == class_id:
            self._session.active_class = self.schema.unlabeled_id
        self.palette.rebuild(self.schema)
        if self._controller is not None:
            self._controller.refresh_display()  # orphaned points recolour to unlabeled
        self._refresh_panels()
        self.statusBar().showMessage(f"Removed class {class_id}")

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

    def _on_assign(self) -> None:
        """Label the current selection with the active class (toolbar / Edit / Enter)."""
        if self._session is not None:
            self._assign(self._session.active_class)

    def _assign_digit(self, digit: int) -> None:
        # The number is the class id, matching what the Classes panel shows.
        if digit not in self.schema:
            return
        self.palette.set_active(digit)  # also sets the active class via its signal
        self._assign(digit)

    def _assign(self, class_id: int) -> None:
        if self._controller is None or self._session is None:
            return
        if self._session.selection.is_empty():
            self.statusBar().showMessage(
                "Nothing selected — click a point (or box-select) first, then assign."
            )
            return
        try:
            name = self.schema.get(class_id).name
        except KeyError:
            name = str(class_id)
        count = self._session.selection.count
        self._controller.assign(class_id)
        self.statusBar().showMessage(f"Labelled {count:,} point(s) as '{name}'.")

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
        if self._controller is None or self._session is None:
            return
        # Clustering a tiny scoped selection is the common footgun (sklearn aborts
        # on one sample). Guide the user instead of failing.
        sel = self._session.selection
        if scope_to_selection and not sel.is_empty() and sel.count < 2:
            QMessageBox.information(
                self,
                "Selection too small",
                "Select at least a few points to cluster, or untick "
                "'Run on current selection only' to segment the whole cloud.",
            )
            return
        self.statusBar().showMessage(f"Running {segmenter.name}…")
        try:
            self._controller.run_segmenter(segmenter, scope_to_selection=scope_to_selection)
        except Exception as exc:  # never let a segmenter take down the window
            self.statusBar().showMessage("Segmentation failed.")
            QMessageBox.critical(self, "Segmentation failed", str(exc))
            return
        grouping = self._session.active_grouping
        self._docks["Groups"].raise_()  # surface the segments that were just made
        self.statusBar().showMessage(f"{segmenter.name}: {grouping.n_groups} groups")

    def _on_active_grouping_changed(self, index) -> None:
        if self._session is None:
            return
        self._session.set_active_grouping(index)
        self.groups_panel.refresh(self._session)
        if self._controller.display_mode == "grouping":
            self._controller.refresh_display()

    def _on_group_selected(self, group_id: int) -> None:
        if self._controller:
            self._controller.select_group(group_id)

    def _on_assign_group_active(self, group_id: int) -> None:
        if self._controller is None:
            return
        n = self._controller.assign_group(group_id)
        self.statusBar().showMessage(f"Labelled segment #{group_id} ({n:,} pts)")

    def _on_assign_group_suggested(self, group_id: int) -> None:
        if self._controller is None:
            return
        n = self._controller.apply_suggested(group_id)
        msg = (
            f"Applied suggestion to segment #{group_id} ({n:,} pts)"
            if n
            else f"Segment #{group_id} has no suggested class"
        )
        self.statusBar().showMessage(msg)

    def _on_assign_all_suggested(self) -> None:
        if self._controller is None:
            return
        n = self._controller.apply_suggested(None)
        self.statusBar().showMessage(f"Applied all suggestions ({n:,} pts)")

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
        self.groups_panel.refresh(self._session)
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
