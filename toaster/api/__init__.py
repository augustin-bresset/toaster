"""The web API layer — FastAPI over the same headless engine the Qt app uses.

Importing :func:`create_app` requires the ``api`` extra (FastAPI/uvicorn). The
:class:`~toaster.api.service.AnnotationService` itself is plain Python and can be
used without a web server.
"""

from __future__ import annotations

from .serialize import decode_array, encode_array
from .service import AnnotationService

__all__ = ["AnnotationService", "encode_array", "decode_array", "create_app"]


def __getattr__(name: str):
    # Defer the FastAPI import (the `api` extra) until create_app is used.
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
