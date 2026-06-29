"""Qt side panels for the Toaster main window."""

from __future__ import annotations

from .class_palette import ClassPalette
from .display_panel import DisplayPanel
from .groups_panel import GroupsPanel
from .layers_panel import LayersPanel
from .segmenter_panel import SegmenterPanel

__all__ = ["ClassPalette", "SegmenterPanel", "LayersPanel", "DisplayPanel", "GroupsPanel"]
