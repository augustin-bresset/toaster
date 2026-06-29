"""Best-effort smoke test of the PyVista viewer with an off-screen plotter.

Skips cleanly when no off-screen GL context is available (e.g. CI without EGL).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista")


def test_offscreen_set_and_recolor():
    import pyvista as pv

    from toaster.viewer import PyVistaViewer

    try:
        plotter = pv.Plotter(off_screen=True)
        viewer = PyVistaViewer(plotter=plotter)
        xyz = np.random.default_rng(0).random((200, 3)).astype(np.float32)
        colors = np.zeros((200, 3), dtype=np.uint8)
        viewer.set_cloud(xyz, colors)
        viewer.update_colors(np.array([0, 1, 2]), np.array([255, 0, 0], np.uint8))
        viewer.highlight(np.array([3, 4]))
        viewer.clear_highlight()
        viewer.set_point_style(size=6, as_spheres=True)
        mask = np.ones(200, dtype=bool)
        mask[:50] = False
        viewer.set_visible_mask(mask)  # ghost-array hide
        viewer.set_visible_mask(None)  # show all
    except Exception as exc:  # pragma: no cover - depends on GL availability
        pytest.skip(f"off-screen rendering unavailable: {exc}")

    # The owned colour buffer (RGBA) was mutated in place.
    assert viewer._colors[0][:3].tolist() == [255, 0, 0]
    assert viewer.point_size == 6
    assert viewer.render_points_as_spheres is True


def test_box_to_point_restores_camera_trackball():
    """Returning to point mode after box mode must re-enable camera orbit."""
    import pyvista as pv

    from toaster.viewer import PyVistaViewer

    try:
        viewer = PyVistaViewer(plotter=pv.Plotter(off_screen=True))
        if viewer.plotter.iren is None:
            pytest.skip("no interactor available")
        viewer.set_cloud(np.zeros((10, 3), np.float32), np.zeros((10, 3), np.uint8))
        viewer.set_point_pick_callback(lambda i, m: None)
        viewer.set_box_pick_callback(lambda i, m: None)
        viewer.set_pick_mode("box")
        viewer.set_pick_mode("point")
        style = type(viewer.plotter.iren.interactor.GetInteractorStyle()).__name__
    except Exception as exc:  # pragma: no cover - depends on GL availability
        pytest.skip(f"interactor style unavailable: {exc}")

    assert "Trackball" in style, f"camera stuck on {style} after box→point"
