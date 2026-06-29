"""Launch the Qt application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from toaster.core import LabelSchema

__all__ = ["run"]


def _force_xcb_on_wayland() -> None:
    """Run Qt through XWayland when the session is native Wayland.

    VTK (and thus the embedded ``QtInteractor``) has no Wayland render backend:
    it issues raw X11 calls against the widget's window handle. On a native
    Wayland surface that handle is not an X11 window, so the X server aborts the
    process with ``BadWindow`` on ``X_ConfigureWindow`` the moment the interactor
    is built. Forcing the ``xcb`` platform makes Qt use XWayland too, giving VTK
    the real X11 window it expects.

    Only applied when the session is Wayland *and* an X display is reachable
    (XWayland is up); an explicit ``QT_QPA_PLATFORM`` is always respected.
    """
    if os.environ.get("QT_QPA_PLATFORM"):
        return
    on_wayland = os.environ.get("WAYLAND_DISPLAY") or (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    )
    if on_wayland and os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def run(path: str | Path | None = None, schema: LabelSchema | None = None) -> int:
    """Start the Toaster GUI, optionally opening ``path``. Returns the exit code."""
    os.environ.setdefault("QT_API", "pyside6")
    _force_xcb_on_wayland()

    from qtpy.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(schema=schema)
    window.show()
    if path is not None:
        window.open_cloud(path)
    return app.exec()
