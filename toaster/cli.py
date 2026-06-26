"""Command-line entry point: ``toaster [path] [--schema schema.yaml]``."""

from __future__ import annotations

import argparse
import importlib
import os
import sys

from toaster.core import LabelSchema

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and launch the GUI."""
    parser = argparse.ArgumentParser(
        prog="toaster",
        description="Annotate lidar point clouds in 3D with pluggable segmentation models.",
    )
    parser.add_argument("path", nargs="?", help="point cloud file to open (.ply/.bin/.las/.pcd)")
    parser.add_argument("--schema", help="path to a label-schema YAML (apairo format)")
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE",
        help="import a module that registers custom segmenters/loaders "
        "(repeatable, e.g. --plugin my_segmenters)",
    )
    args = parser.parse_args(argv)

    # Pin the Qt binding before qtpy is imported anywhere downstream.
    os.environ.setdefault("QT_API", "pyside6")

    # Import plugin modules so their @register_segmenter / register_loader run
    # before the UI (which lists what is registered) is built.
    for module in args.plugin:
        importlib.import_module(module)

    schema = LabelSchema.from_yaml(args.schema) if args.schema else None

    from .app.run import run

    return run(args.path, schema=schema)


if __name__ == "__main__":
    sys.exit(main())
