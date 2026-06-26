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
        )
        self.reset_camera()

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
        if self._pick_mode == "point" and self._point_cb is not None:
            self.plotter.enable_point_picking(
                callback=self._on_pick, show_message=False, left_clicking=True
            )
        elif self._pick_mode == "box" and self._box_cb is not None:
            self.plotter.enable_cell_picking(
                callback=self._on_box, through=True, show=False, style="wireframe"
            )

    def _on_pick(self, *args: Any) -> None:
        if self._point_cb is None or self._cloud is None:
            return
        point = getattr(self.plotter, "picked_point", None)
        if point is None and args:
            point = args[0]
        if point is None:
            return
        index = int(self._cloud.find_closest_point(np.asarray(point)))
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
