"""Guard the core's headlessness: importing it must not drag in Qt/VTK."""

from __future__ import annotations

import subprocess
import sys

_GUI_MODULES = ("pyvista", "vtk", "vtkmodules", "PySide6", "PyQt6", "qtpy")


def _imported_gui_modules(import_target: str) -> str:
    code = (
        "import sys\n"
        f"import {import_target}\n"
        f"bad = [m for m in {_GUI_MODULES!r} if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_core_import_is_headless():
    assert _imported_gui_modules("toaster.core") == ""


def test_top_level_import_is_headless():
    assert _imported_gui_modules("toaster") == ""


def test_segment_import_is_headless():
    # Segmenters use sklearn, but must not pull the rendering stack.
    assert _imported_gui_modules("toaster.segment") == ""


def test_interaction_import_is_headless():
    # The interaction controller drives any front-end; importing it must not
    # drag in Qt/VTK, so a non-Qt client can reuse it.
    assert _imported_gui_modules("toaster.interaction") == ""
