"""Rendering: a backend-agnostic :class:`Viewer` protocol + a PyVista backend.

Importing this package pulls in PyVista, so it is *not* imported by
``toaster.core`` and is loaded lazily by the app. Colour helpers are numpy-only
and safe to import anywhere.
"""

from __future__ import annotations

from .base import BoxPickCallback, Modifiers, PointPickCallback, Viewer
from .colormap import colors_from_grouping, colors_from_labels, colors_from_scalar

__all__ = [
    "Viewer",
    "Modifiers",
    "PointPickCallback",
    "BoxPickCallback",
    "colors_from_labels",
    "colors_from_grouping",
    "colors_from_scalar",
    "PyVistaViewer",
]


def __getattr__(name: str):
    # Defer the actual PyVista import to first use of the backend.
    if name == "PyVistaViewer":
        from .pyvista_viewer import PyVistaViewer

        return PyVistaViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
