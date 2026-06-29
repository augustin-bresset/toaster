"""A no-op :class:`~toaster.viewer.base.Viewer`.

For front-ends that render the cloud themselves (the web client computes colours
in JS from the labels/grouping it fetches), the controller still wants a viewer
to talk to — but none of its render calls need to do anything. This satisfies
the protocol while keeping all presentation on the client.
"""

from __future__ import annotations

import numpy as np

from .base import BoxPickCallback, PointPickCallback

__all__ = ["NullViewer"]


class NullViewer:
    """Implements the Viewer protocol with no rendering side effects."""

    def __init__(self) -> None:
        self.point_pick_callback: PointPickCallback | None = None
        self.box_pick_callback: BoxPickCallback | None = None

    def set_cloud(self, xyz: np.ndarray, colors: np.ndarray) -> None:
        pass

    def update_colors(self, indices: np.ndarray, colors: np.ndarray) -> None:
        pass

    def highlight(self, indices: np.ndarray) -> None:
        pass

    def clear_highlight(self) -> None:
        pass

    def set_point_pick_callback(self, callback: PointPickCallback) -> None:
        self.point_pick_callback = callback

    def set_box_pick_callback(self, callback: BoxPickCallback) -> None:
        self.box_pick_callback = callback

    def set_pick_mode(self, mode: str) -> None:
        pass

    def set_point_style(self, size: int | None = None, as_spheres: bool | None = None) -> None:
        pass

    def set_visible_mask(self, mask: np.ndarray | None) -> None:
        pass

    def reset_camera(self) -> None:
        pass

    def render(self) -> None:
        pass
