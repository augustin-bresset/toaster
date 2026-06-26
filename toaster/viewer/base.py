"""The :class:`Viewer` interface — the renderer boundary.

Only numpy arrays and plain indices cross this boundary; no VTK/Qt types leak
through it. That is what makes the rendering backend swappable: a VisPy viewer
could replace the PyVista one without the app noticing, as long as it speaks this
protocol.

Interaction results come back as primitives too — a picked point is an ``int``,
a box selection is an index array, and keyboard modifiers are a frozenset of
plain strings (subset of ``{"shift", "ctrl"}``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

__all__ = ["Viewer", "Modifiers", "PointPickCallback", "BoxPickCallback"]

#: Active keyboard modifiers during an interaction, e.g. ``frozenset({"shift"})``.
Modifiers = frozenset

#: Called with the picked point index and the active modifiers.
PointPickCallback = Callable[[int, Modifiers], None]

#: Called with the boxed point indices and the active modifiers.
BoxPickCallback = Callable[[np.ndarray, Modifiers], None]


class Viewer(Protocol):
    """A 3-D point-cloud renderer with picking, driven by the app."""

    def set_cloud(self, xyz: np.ndarray, colors: np.ndarray) -> None:
        """Display ``xyz`` ``(N, 3)`` float32 coloured by ``colors`` ``(N, 3)`` uint8."""
        ...

    def update_colors(self, indices: np.ndarray, colors: np.ndarray) -> None:
        """Recolour just ``indices``; ``colors`` is ``(3,)`` or ``(len(indices), 3)`` uint8."""
        ...

    def highlight(self, indices: np.ndarray) -> None:
        """Show a transient selection overlay on ``indices`` (does not alter base colours)."""
        ...

    def clear_highlight(self) -> None:
        """Remove the selection overlay."""
        ...

    def set_point_pick_callback(self, callback: PointPickCallback) -> None:
        """Register the single-point pick handler."""
        ...

    def set_box_pick_callback(self, callback: BoxPickCallback) -> None:
        """Register the rubber-band box-selection handler."""
        ...

    def set_pick_mode(self, mode: str) -> None:
        """Switch active selection mode: ``"point"`` (click) or ``"box"`` (rubber-band)."""
        ...

    def set_point_style(self, size: int | None = None, as_spheres: bool | None = None) -> None:
        """Adjust how points are drawn (pixel size and round-vs-flat shape)."""
        ...

    def reset_camera(self) -> None:
        """Frame the whole cloud."""
        ...

    def render(self) -> None:
        """Request a redraw."""
        ...
