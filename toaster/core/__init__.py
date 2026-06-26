"""Toaster core domain — the headless, numpy-only annotation library.

Nothing in this package imports Qt, VTK or PyVista, so it can be used in a
script, a notebook or a data pipeline without a display. The public API is
exactly the names re-exported here.
"""

from __future__ import annotations

from .annotation import AnnotationController, EditHistory, LabelEdit
from .grouping import Grouping
from .label_schema import LabelClass, LabelSchema
from .point_cloud import PointCloud
from .selection import Selection
from .session import Session
from .types import NOISE, ClassId, GroupId, IndexArray

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
    "IndexArray",
    "ClassId",
    "GroupId",
    "NOISE",
]
