"""Loader for ``.pcd`` files (PCL).

Handles the two common encodings — ``ascii`` and uncompressed ``binary`` — with
single-element fields, which covers the vast majority of PCD files in the wild.
``binary_compressed`` is not decoded here; install the ``open3d`` extra and the
:mod:`~toaster.io` registry will prefer Open3D for ``.pcd`` instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from toaster.core import PointCloud

__all__ = ["PcdLoader"]

_NP_TYPE = {
    ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8",
    ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8",
    ("F", 4): "f4", ("F", 8): "f8",
}  # fmt: skip


class PcdLoader:
    """Reads ascii / uncompressed-binary PCD point clouds."""

    extensions = (".pcd",)

    def load(self, path: str | Path) -> PointCloud:
        path = Path(path)
        with open(path, "rb") as fh:
            header, fields, sizes, types, counts, n_points, data_kind = _read_header(fh)
            if any(c != 1 for c in counts):
                raise ValueError(
                    f"{path.name}: multi-count PCD fields are not supported by the "
                    "built-in reader; install the 'open3d' extra."
                )
            dtype = np.dtype(
                [(f, _NP_TYPE[(t, s)]) for f, s, t in zip(fields, sizes, types, strict=True)]
            )
            if data_kind == "ascii":
                rows = np.loadtxt(fh, dtype=np.float64).reshape(n_points, len(fields))
                record = {f: rows[:, i] for i, f in enumerate(fields)}
            elif data_kind == "binary":
                buf = fh.read(n_points * dtype.itemsize)
                arr = np.frombuffer(buf, dtype=dtype, count=n_points)
                record = {f: arr[f] for f in fields}
            else:  # binary_compressed
                raise ValueError(
                    f"{path.name}: binary_compressed PCD is not supported by the "
                    "built-in reader; install the 'open3d' extra."
                )

        xyz = np.stack([record["x"], record["y"], record["z"]], axis=1).astype(np.float32)
        features: dict[str, np.ndarray] = {}
        if "intensity" in record:
            features["intensity"] = np.asarray(record["intensity"], dtype=np.float32)
        if "rgb" in record or "rgba" in record:
            features["rgb"] = _unpack_rgb(record.get("rgb", record.get("rgba")))

        return PointCloud(xyz=xyz, features=features, source=path)


def _read_header(fh):
    fields = sizes = types = counts = None
    n_points = 0
    data_kind = "ascii"
    raw_header = []
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("unexpected end of PCD header")
        text = line.decode("ascii", "replace").strip()
        raw_header.append(text)
        if not text or text.startswith("#"):
            continue
        key, _, rest = text.partition(" ")
        key = key.upper()
        if key == "FIELDS":
            fields = rest.split()
        elif key == "SIZE":
            sizes = [int(x) for x in rest.split()]
        elif key == "TYPE":
            types = rest.split()
        elif key == "COUNT":
            counts = [int(x) for x in rest.split()]
        elif key == "POINTS":
            n_points = int(rest)
        elif key == "DATA":
            data_kind = rest.strip().lower() or "ascii"
            break
    if fields is None or sizes is None or types is None:
        raise ValueError("PCD header missing FIELDS/SIZE/TYPE")
    if counts is None:
        counts = [1] * len(fields)
    return raw_header, fields, sizes, types, counts, n_points, data_kind


def _unpack_rgb(packed: np.ndarray) -> np.ndarray:
    """Decode PCD's packed RGB (a 32-bit value carried as float/uint) to ``(N, 3)`` uint8."""
    as_u32 = np.asarray(packed).astype(np.float32).view(np.uint32)
    r = (as_u32 >> 16) & 0xFF
    g = (as_u32 >> 8) & 0xFF
    b = as_u32 & 0xFF
    return np.stack([r, g, b], axis=1).astype(np.uint8)
