"""The :class:`Loader` extension point.

A loader is anything that turns a file path into a :class:`~toaster.core.PointCloud`.
Add support for a new format by writing a class with an ``extensions`` tuple and
a ``load`` method, then calling :func:`~toaster.io.register_loader`. Heavy format
libraries should be imported *inside* ``load`` so registering a loader never pays
their import cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from toaster.core import PointCloud

__all__ = ["Loader"]


@runtime_checkable
class Loader(Protocol):
    """Reads one point-cloud file into a :class:`~toaster.core.PointCloud`."""

    #: File extensions this loader handles, lowercase, including the dot.
    extensions: tuple[str, ...]

    def load(self, path: str | Path) -> PointCloud:
        """Load ``path`` and return a populated cloud."""
        ...
