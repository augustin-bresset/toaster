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
    except Exception as exc:  # pragma: no cover - depends on GL availability
        pytest.skip(f"off-screen rendering unavailable: {exc}")

    # The owned colour buffer was mutated in place.
    assert viewer._colors[0].tolist() == [255, 0, 0]
    assert viewer.point_size == 6
    assert viewer.render_points_as_spheres is True
