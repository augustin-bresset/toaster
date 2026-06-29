"""Persisting the *labeling schema* as a sidecar next to the cloud.

The classes you label into (names, colours, ignore index) are part of a cloud's
annotation session, so they are saved beside the cloud just like the labels are
(``<cloud>.toaster.schema.yaml``). Reopening the cloud restores the exact palette
it was annotated with, without depending on a global config or a ``--schema``
flag being passed again.
"""

from __future__ import annotations

from pathlib import Path

from toaster.core import LabelSchema

__all__ = ["SchemaStore", "SCHEMA_SIDECAR_SUFFIX"]

#: Appended to the cloud path to form the schema sidecar path.
SCHEMA_SIDECAR_SUFFIX = ".toaster.schema.yaml"


class SchemaStore:
    """Reads and writes the label-schema sidecar for a cloud."""

    suffix = SCHEMA_SIDECAR_SUFFIX

    def path_for(self, source: str | Path) -> Path:
        """The schema sidecar path for a given cloud source path."""
        return Path(str(source) + self.suffix)

    def save(self, source: str | Path, schema: LabelSchema) -> Path:
        """Write ``schema`` to the sidecar for ``source`` and return its path."""
        return schema.to_yaml(self.path_for(source))

    def load(self, source: str | Path) -> LabelSchema | None:
        """Load the schema for ``source``, or ``None`` if no sidecar exists."""
        path = self.path_for(source)
        if not path.exists():
            return None
        return LabelSchema.from_yaml(path)
