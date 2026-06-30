"""Persisting the deliverable: per-point ``labels`` saved next to the cloud.

The deliverable is the cloud's ``labels`` array and nothing else. It is written
to a sidecar keyed by the cloud's stem (``<cloud-without-ext>_toaster.npy``), so
reopening a cloud restores its annotation. The original extension is dropped so
the name reads cleanly (``001813_toaster.npy``, not ``001813.npy.toaster.npy``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["LabelStore", "SIDECAR_SUFFIX"]

#: Appended to the cloud stem (extension dropped) to form the label sidecar path.
SIDECAR_SUFFIX = "_toaster.npy"


class LabelStore:
    """Reads and writes the ``labels`` sidecar for a cloud."""

    suffix = SIDECAR_SUFFIX

    def path_for(self, source: str | Path) -> Path:
        """The sidecar path for a given cloud source path (its extension dropped)."""
        p = Path(source)
        return p.with_name(p.stem + self.suffix)

    def save(self, source: str | Path, labels: np.ndarray) -> Path:
        """Write ``labels`` to the sidecar for ``source`` and return its path."""
        out = self.path_for(source)
        np.save(out, np.ascontiguousarray(labels, dtype=np.int32))
        return out

    def load(self, source: str | Path) -> np.ndarray | None:
        """Load labels for ``source``, or ``None`` if no sidecar exists."""
        path = self.path_for(source)
        if not path.exists():
            return None
        return np.load(path).astype(np.int32)
