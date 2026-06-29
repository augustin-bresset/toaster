"""Access to the label schemas bundled with the package (front-agnostic).

Lives at the package root so any front-end — the desktop app, the web API — can
load a default schema without dragging in heavy deps.
"""

from __future__ import annotations

import importlib.resources as resources

from toaster.core import LabelSchema

__all__ = ["builtin_schema", "builtin_schema_names"]


def builtin_schema(name: str = "default") -> LabelSchema:
    """Load a bundled schema by name (without the ``.yaml`` suffix)."""
    path = resources.files("toaster") / "schemas" / f"{name}.yaml"
    with resources.as_file(path) as p:
        return LabelSchema.from_yaml(p)


def builtin_schema_names() -> list[str]:
    """Names of the bundled schema YAMLs."""
    root = resources.files("toaster") / "schemas"
    return sorted(p.name[:-5] for p in root.iterdir() if p.name.endswith(".yaml"))
