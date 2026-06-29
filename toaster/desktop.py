"""Desktop shell: a native window rendering the Toaster web UI (no browser).

This is the Spotify / VS Code model — one web front, shown in a native window —
done in pure Python with pywebview. A local FastAPI server runs in a background
thread and the OS webview (or QtWebEngine, depending on what's installed) renders
the page. ``toaster`` launches this; ``toaster-web`` serves the same UI for a
plain browser instead.
"""

from __future__ import annotations

import argparse
import importlib.resources
import importlib.util
import socket
import sys
import threading
import time
from contextlib import closing

__all__ = ["main"]


def _free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_until_ready(port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with closing(socket.socket()) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.05)
    return False


def _serve_in_background(app, port: int):
    """Run uvicorn in a daemon thread; return the server handle."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="toaster", description="Toaster — point-cloud labeler.")
    parser.add_argument("path", nargs="?", help="point cloud to open on startup")
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE",
        help="import a module before starting so its custom segmenters/loaders register "
        "(repeatable)",
    )
    args = parser.parse_args(argv)

    import webview

    from .api.app import create_app

    for module in args.plugin:
        importlib.import_module(module)
    app = create_app()
    if args.path:
        app.state.service.open_cloud(args.path)

    port = _free_port()
    server = _serve_in_background(app, port)
    if not _wait_until_ready(port):
        print("toaster: the local server did not start in time", file=sys.stderr)
        return 1

    webview.create_window("TOASTER", f"http://127.0.0.1:{port}/", width=1320, height=860)
    # Pick the backend explicitly so pywebview does not noisily try (and fail)
    # GTK before falling back to Qt when WebKitGTK is not installed.
    gui = "gtk" if importlib.util.find_spec("gi") is not None else "qt"
    start_kwargs: dict = {"gui": gui}
    icon = importlib.resources.files("toaster") / "web" / "icon.png"
    try:
        if icon.is_file():
            start_kwargs["icon"] = str(icon)  # taskbar / window icon
    except Exception:
        pass
    webview.start(**start_kwargs)  # blocks until the window is closed
    server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
