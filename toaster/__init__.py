"""Toaster — annotate lidar point clouds in 3D with pluggable segmentation models.

Importing :mod:`toaster` pulls in only the headless, numpy-only core (the domain
types re-exported below). The desktop entry point (:func:`run`) is imported
lazily, so ``import toaster`` never drags in the web/UI stack — handy for
scripts, tests and pipelines.
"""

from __future__ import annotations

from .core import (
    AnnotationController,
    EditHistory,
    Grouping,
    LabelClass,
    LabelEdit,
    LabelSchema,
    PointCloud,
    Selection,
    Session,
)

__version__ = "0.1.0"

__all__ = [
    "PointCloud",
    "LabelSchema",
    "LabelClass",
    "Selection",
    "Grouping",
    "LabelEdit",
    "EditHistory",
    "AnnotationController",
    "Session",
    "run",
    "__version__",
]


def __getattr__(name: str):
    # PEP 562: defer the web/desktop import until the app is actually launched.
    if name == "run":
        from .desktop import main as run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
