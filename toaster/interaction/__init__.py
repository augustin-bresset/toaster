"""The interaction layer — presentation-agnostic workflow glue.

Sits between the headless :mod:`toaster.core` state and *any* front-end. It
never imports Qt/VTK, talking to the renderer only through the
:class:`~toaster.viewer.base.Viewer` protocol, so a Qt window, a VisPy widget or
a future web client can all drive the same :class:`InteractionController`.
"""

from __future__ import annotations

from .controller import DisplayMode, InteractionController
from .snapshot import ClassInfo, GroupingInfo, SegmentInfo, Snapshot

__all__ = [
    "InteractionController",
    "DisplayMode",
    "Snapshot",
    "ClassInfo",
    "SegmentInfo",
    "GroupingInfo",
]
