"""The wire protocol for numpy arrays: ``{dtype, shape, data(base64)}``.

A front-end decodes ``data`` straight into a typed array (e.g. a JS
``Float32Array``) using ``dtype``/``shape`` — no per-element JSON, so large point
clouds transfer compactly.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

__all__ = ["encode_array", "decode_array"]


def encode_array(arr: np.ndarray) -> dict[str, Any]:
    """Encode a numpy array as ``{dtype, shape, data}`` (data is base64 bytes)."""
    arr = np.ascontiguousarray(arr)
    return {
        "dtype": arr.dtype.str,  # e.g. '<f4', '<i4' — maps to a JS TypedArray
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`encode_array`."""
    raw = base64.b64decode(payload["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(payload["dtype"]))
    return arr.reshape(payload["shape"])
