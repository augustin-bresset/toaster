"""The Qt application layer — the only layer that touches every other.

Importing names here pulls in Qt/PyVista, so ``toaster.app`` is imported lazily
(via :func:`toaster.run`) and never by the headless core.
"""

from __future__ import annotations

from .run import run
from .schema_loader import builtin_schema, builtin_schema_names

__all__ = ["run", "builtin_schema", "builtin_schema_names"]
