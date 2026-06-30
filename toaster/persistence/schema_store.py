"""Persisting the *labeling schema* as a sidecar next to the cloud.

The classes you label into (names, colours, ignore index) are part of a cloud's
annotation session, so they are saved beside the cloud just like the labels are
(``<cloud-without-ext>_toaster_schema.yaml``). Reopening the cloud restores the
exact palette it was annotated with, without depending on a global config or a
``--schema`` flag being passed again. The YAML also records the original cloud
path it was annotated from.
"""

from __future__ import annotations

import os
from pathlib import Path

from toaster.core import LabelSchema

__all__ = ["SchemaStore", "SCHEMA_SIDECAR_SUFFIX"]

#: Appended to the cloud stem (extension dropped) to form the schema sidecar path.
SCHEMA_SIDECAR_SUFFIX = "_toaster_schema.yaml"


class SchemaStore:
    """Reads and writes the label-schema sidecar for a cloud."""

    suffix = SCHEMA_SIDECAR_SUFFIX

    def path_for(self, source: str | Path) -> Path:
        """The schema sidecar path for a given cloud source path (its extension dropped)."""
        p = Path(source)
        return p.with_name(p.stem + self.suffix)

    def save(
        self, source: str | Path, schema: LabelSchema, cloud_path: str | Path | None = None
    ) -> Path:
        """Write ``schema`` to the sidecar for ``source`` and return its path.

        ``cloud_path`` (the original cloud this annotation came from) is recorded
        in the YAML under ``cloud`` *relative to the sidecar's folder*, so moving
        the whole folder keeps the reference valid (beside the cloud it is just
        the filename).
        """
        out = self.path_for(source)
        extra = None
        if cloud_path is not None:
            rel = os.path.relpath(os.path.abspath(str(cloud_path)), os.path.abspath(out.parent))
            extra = {"cloud": rel}
        return schema.to_yaml(out, extra=extra)

    def load(self, source: str | Path) -> LabelSchema | None:
        """Load the schema for ``source``, or ``None`` if no sidecar exists."""
        path = self.path_for(source)
        if not path.exists():
            return None
        return LabelSchema.from_yaml(path)
