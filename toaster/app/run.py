"""Launch the Qt application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from toaster.core import LabelSchema

__all__ = ["run"]


def run(path: str | Path | None = None, schema: LabelSchema | None = None) -> int:
    """Start the Toaster GUI, optionally opening ``path``. Returns the exit code."""
    os.environ.setdefault("QT_API", "pyside6")

    from qtpy.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(schema=schema)
    window.show()
    if path is not None:
        window.open_cloud(path)
    return app.exec()
