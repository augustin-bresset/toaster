"""PyVista implementation of the :class:`~toaster.viewer.base.Viewer` protocol.

Owns a single points-only ``PolyData`` and its colour buffer, so recolouring a
subset is an in-place mutation plus a redraw — never a mesh rebuild. Picking
returns plain point indices (resolved via the cloud's point locator), keeping
the protocol boundary free of VTK types.

PyVista/VTK API specifics here were written against pyvista ~0.4x / VTK 9 and are
defensive (try/except around version-variable bits); confirm once the env is up.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BoxPickCallback, PointPickCallback

__all__ = ["PyVistaViewer"]


class PyVistaViewer:
    """A point-cloud viewport backed by a PyVista plotter.

    Args:
        plotter: An existing plotter to draw into. If ``None``, a
            :class:`pyvistaqt.QtInteractor` is created (needs a live
            ``QApplication``). Pass ``pyvista.Plotter(off_screen=True)`` for
            headless tests.
        point_size: Rendered point size in pixels.
        render_points_as_spheres: Larger hit area for picking at a small cost.
    """

    #: Highlight overlay colour for the current selection (warm yellow).
    HIGHLIGHT_COLOR = "#ffd400"
    #: Default viewport background (dark slate, like most annotation tools).
    BACKGROUND = "#1f2430"
    #: Pointer travel (px, Manhattan) above which a left-drag counts as an orbit
    #: rather than a click-to-select.
    DRAG_TOLERANCE_PX = 5

    def __init__(
        self,
        plotter: Any | None = None,
        *,
        point_size: int = 3,
        render_points_as_spheres: bool = False,
    ) -> None:
        if plotter is None:
            from pyvistaqt import QtInteractor

            plotter = QtInteractor()
        self.plotter = plotter
        self.point_size = point_size
        self.render_points_as_spheres = render_points_as_spheres

        self._cloud: Any | None = None
        self._actor: Any | None = None
        self._colors: np.ndarray | None = None
        self._highlight_actor: Any | None = None
        self._point_cb: PointPickCallback | None = None
        self._box_cb: BoxPickCallback | None = None
        self._pick_mode: str = "point"

        # Left-click vs. left-drag discrimination (see _enable_click_to_select).
        self._press_xy: tuple[int, int] | None = None
        self._dragged: bool = False
        self._gesture_observers: list[int] = []
        self._picker: Any | None = None

        try:
            self.plotter.set_background(self.BACKGROUND)
        except Exception:
            pass

    # -- display ----------------------------------------------------------

    def set_cloud(self, xyz: np.ndarray, colors: np.ndarray) -> None:
        import pyvista as pv

        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        self._colors = np.ascontiguousarray(colors, dtype=np.uint8)

        if self._actor is not None:
            self.plotter.remove_actor(self._actor)
        self.clear_highlight()

        cloud = pv.PolyData(xyz)
        cloud.point_data["colors"] = self._colors
        cloud.point_data["pid"] = np.arange(xyz.shape[0], dtype=np.int64)
        self._cloud = cloud
        self._actor = self.plotter.add_mesh(
            cloud,
            scalars="colors",
            rgb=True,
            point_size=self.point_size,
            render_points_as_spheres=self.render_points_as_spheres,
            lighting=False,
            reset_camera=False,
        )
        # NB: do not reset the camera here. ``set_cloud`` also runs on every
        # recolour / display-mode switch, and re-framing then would throw away
        # the view the user has orbited to. Initial framing is the caller's job
        # (MainWindow.open_cloud calls ``reset_camera`` once after the first load).

    def update_colors(self, indices: np.ndarray, colors: np.ndarray) -> None:
        if self._cloud is None or self._colors is None:
            return
        self._colors[np.asarray(indices, dtype=np.int64)] = colors
        # Re-bind so VTK picks up the mutation across pyvista versions (no copy
        # for a contiguous array).
        self._cloud.point_data["colors"] = self._colors
        self.render()

    def highlight(self, indices: np.ndarray) -> None:
        import pyvista as pv

        if self._cloud is None:
            return
        self.clear_highlight()
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            self.render()
            return
        pts = self._cloud.points[indices]
        overlay = pv.PolyData(pts)
        self._highlight_actor = self.plotter.add_mesh(
            overlay,
            color=self.HIGHLIGHT_COLOR,
            point_size=self.point_size + 4,
            render_points_as_spheres=True,
            lighting=False,
            # Adding an actor re-frames the camera by default unless the camera
            # is flagged "set" — which interactive orbiting never does — so every
            # selection would snap the view back. The overlay must never move the
            # camera, and must not intercept the next pick.
            reset_camera=False,
            pickable=False,
        )
        self.render()

    def clear_highlight(self) -> None:
        if self._highlight_actor is not None:
            self.plotter.remove_actor(self._highlight_actor)
            self._highlight_actor = None

    def set_point_style(self, size: int | None = None, as_spheres: bool | None = None) -> None:
        if size is not None:
            self.point_size = int(size)
        if as_spheres is not None:
            self.render_points_as_spheres = bool(as_spheres)
        if self._actor is not None:
            prop = self._actor.prop
            prop.point_size = self.point_size
            prop.render_points_as_spheres = self.render_points_as_spheres
            self.render()

    # -- interaction ------------------------------------------------------

    def set_point_pick_callback(self, callback: PointPickCallback) -> None:
        self._point_cb = callback
        self._refresh_picking()

    def set_box_pick_callback(self, callback: BoxPickCallback) -> None:
        self._box_cb = callback
        self._refresh_picking()

    def set_pick_mode(self, mode: str) -> None:
        """Switch between ``"point"`` (click) and ``"box"`` (rubber-band) selection.

        VTK's picker is single-use, so point and box picking cannot both be live
        at once — they are toggled here instead.
        """
        if mode not in ("point", "box"):
            raise ValueError(f"unknown pick mode {mode!r}")
        self._pick_mode = mode
        self._refresh_picking()

    def _refresh_picking(self) -> None:
        if not self._has_interactor():
            return
        try:
            self.plotter.disable_picking()
        except Exception:
            pass
        self._clear_gesture_observers()
        if self._pick_mode == "point" and self._point_cb is not None:
            self._enable_click_to_select()
        elif self._pick_mode == "box" and self._box_cb is not None:
            self.plotter.enable_cell_picking(
                callback=self._on_box, through=True, show=False, style="wireframe"
            )

    def _enable_click_to_select(self) -> None:
        """Select on a left *click*, while a left *drag* orbits the camera untouched.

        VTK's trackball style already orbits on left-drag, but PyVista's built-in
        point picking fires on the button *press* — so the instant an orbit began
        it would select a point (and, via the selection overlay, re-frame the
        camera). We watch the gesture ourselves instead: record where the button
        went down, mark it a drag once the pointer travels past
        :attr:`DRAG_TOLERANCE_PX`, and only pick on release when it stayed put.
        """
        from vtkmodules.vtkRenderingCore import vtkPointPicker

        if self._picker is None:
            self._picker = vtkPointPicker()
            self._picker.SetTolerance(0.025)

        iren = self.plotter.iren
        self._press_xy = None
        self._dragged = False
        self._gesture_observers = [
            iren.add_observer("LeftButtonPressEvent", self._on_left_press),
            iren.add_observer("MouseMoveEvent", self._on_mouse_move),
            iren.add_observer("LeftButtonReleaseEvent", self._on_left_release),
        ]

    def _clear_gesture_observers(self) -> None:
        iren = getattr(self.plotter, "iren", None)
        if iren is not None:
            for obs in self._gesture_observers:
                try:
                    iren.remove_observer(obs)
                except Exception:
                    pass
        self._gesture_observers = []
        self._press_xy = None
        self._dragged = False

    def _event_xy(self) -> tuple[int, int]:
        x, y = self.plotter.iren.interactor.GetEventPosition()
        return int(x), int(y)

    def _on_left_press(self, *args: Any) -> None:
        self._press_xy = self._event_xy()
        self._dragged = False

    def _on_mouse_move(self, *args: Any) -> None:
        if self._press_xy is None:
            return
        x, y = self._event_xy()
        if abs(x - self._press_xy[0]) + abs(y - self._press_xy[1]) > self.DRAG_TOLERANCE_PX:
            self._dragged = True

    def _on_left_release(self, *args: Any) -> None:
        press_xy, dragged = self._press_xy, self._dragged
        self._press_xy = None
        self._dragged = False
        if press_xy is None or dragged:
            return  # an orbit (or a press that began off-canvas) — not a selection
        self._select_at(*self._event_xy())

    def _select_at(self, x: int, y: int) -> None:
        if self._point_cb is None or self._cloud is None or self._picker is None:
            return
        renderer = self.plotter.iren.get_poked_renderer()
        self._picker.Pick(x, y, 0, renderer)
        if self._picker.GetDataSet() is None:
            return  # clicked empty space — leave the current selection alone
        point = np.asarray(self._picker.GetPickPosition())
        index = int(self._cloud.find_closest_point(point))
        self._point_cb(index, self._modifiers())

    def _on_box(self, picked: Any) -> None:
        if self._box_cb is None or picked is None:
            return
        data = getattr(picked, "point_data", {})
        if "pid" in data:
            indices = np.asarray(data["pid"], dtype=np.int64)
        else:
            indices = np.empty(0, dtype=np.int64)
        self._box_cb(indices, self._modifiers())

    def _modifiers(self) -> frozenset:
        mods: set[str] = set()
        try:
            iren = self.plotter.iren.interactor
            if iren.GetShiftKey():
                mods.add("shift")
            if iren.GetControlKey():
                mods.add("ctrl")
        except Exception:
            pass
        return frozenset(mods)

    def _has_interactor(self) -> bool:
        iren = getattr(self.plotter, "iren", None)
        return iren is not None

    # -- camera / redraw --------------------------------------------------

    def reset_camera(self) -> None:
        self.plotter.reset_camera()

    def render(self) -> None:
        self.plotter.render()
