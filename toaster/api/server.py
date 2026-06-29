"""``toaster-web`` — run the REST API + web UI with uvicorn."""

from __future__ import annotations

import argparse
import sys

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="toaster-web", description="Serve the Toaster web app (REST API + UI)."
    )
    parser.add_argument("path", nargs="?", help="point cloud to open on startup")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import uvicorn

    from .app import create_app

    app = create_app()
    if args.path:
        app.state.service.open_cloud(args.path)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
