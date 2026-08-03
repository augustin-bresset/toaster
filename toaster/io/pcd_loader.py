"""Loader for ``.pcd`` files (PCL).

Handles the three encodings PCL writes — ``ascii``, ``binary`` and
``binary_compressed`` (LZF, decoded by :mod:`toaster.io._lzf`) — with any field
layout: multi-element fields (``COUNT`` > 1), the ``_`` padding fields PCL emits
for its aligned point types, and organized clouds whose empty pixels are NaN.

``x``/``y``/``z`` become the geometry; ``intensity``, ``rgb``/``rgba`` and
normals become features, and a ``label`` field is read as the starting
annotation. Everything else in the file is sensor metadata the viewer does not
use and is dropped.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toaster.core import PointCloud

from . import _lzf

__all__ = ["PcdLoader"]

_NP_TYPE = {
    ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8",
    ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8",
    ("F", 4): "f4", ("F", 8): "f8",
}  # fmt: skip


class PcdLoader:
    """Reads ascii / binary / binary_compressed PCD point clouds."""

    extensions = (".pcd",)

    def load(self, path: str | Path) -> PointCloud:
        path = Path(path)
        with open(path, "rb") as fh:
            header = _read_header(fh, path.name)
            record = _read_data(fh, header, path.name)
        return _to_cloud(record, path)


# -- header ---------------------------------------------------------------


@dataclass
class _Header:
    fields: list[str]
    sizes: list[int]
    types: list[str]
    counts: list[int]
    n_points: int
    data_kind: str


def _read_header(fh, name: str) -> _Header:
    """Consume the text header, leaving ``fh`` positioned on the data block."""
    fields: list[str] | None = None
    sizes: list[int] | None = None
    types: list[str] | None = None
    counts: list[int] | None = None
    n_points: int | None = None
    width = height = None
    data_kind = "ascii"
    while True:
        line = fh.readline()
        if not line:
            raise ValueError(f"{name}: unexpected end of PCD header")
        text = line.decode("ascii", "replace").strip()  # .strip() also eats CRLF
        if not text or text.startswith("#"):
            continue
        key, _, rest = text.partition(" ")
        key = key.upper()
        if key == "FIELDS":
            fields = rest.split()
        elif key == "SIZE":
            sizes = [int(x) for x in rest.split()]
        elif key == "TYPE":
            types = [x.upper() for x in rest.split()]
        elif key == "COUNT":
            counts = [int(x) for x in rest.split()]
        elif key == "POINTS":
            n_points = int(rest)
        elif key == "WIDTH":
            width = int(rest)
        elif key == "HEIGHT":
            height = int(rest)
        elif key == "DATA":
            data_kind = rest.strip().lower() or "ascii"
            break
    if fields is None or sizes is None or types is None:
        raise ValueError(f"{name}: PCD header missing FIELDS/SIZE/TYPE")
    if counts is None:
        counts = [1] * len(fields)
    if not len(fields) == len(sizes) == len(types) == len(counts):
        raise ValueError(f"{name}: PCD header FIELDS/SIZE/TYPE/COUNT lengths disagree")
    if n_points is None:
        # POINTS is optional in practice; WIDTH * HEIGHT is the same number
        # (HEIGHT > 1 only for organized clouds).
        n_points = width * height if width is not None and height is not None else 0
    if data_kind not in ("ascii", "binary", "binary_compressed"):
        raise ValueError(f"{name}: unknown PCD data encoding '{data_kind}'")
    return _Header(fields, sizes, types, counts, n_points, data_kind)


def _dtype_for(header: _Header, name: str) -> tuple[np.dtype, dict[str, str]]:
    """Build the packed record dtype, plus a ``field name -> dtype slot`` map.

    PCL pads its aligned point types with fields all literally named ``_``, so
    slot names are made unique; padding (and any repeated field name) is kept in
    the dtype — it occupies bytes — but left out of the returned map.
    """
    spec: list[tuple] = []
    slots: dict[str, str] = {}
    for i, (field, size, typ, count) in enumerate(
        zip(header.fields, header.sizes, header.types, header.counts, strict=True)
    ):
        np_type = _NP_TYPE.get((typ, size))
        if np_type is None:
            raise ValueError(f"{name}: unsupported PCD field type '{typ}{size}' for '{field}'")
        if count < 1:
            raise ValueError(f"{name}: field '{field}' has COUNT {count}")
        slot = f"_pad{i}" if field == "_" or field in slots else field
        if field != "_" and field not in slots:
            slots[field] = slot
        spec.append((slot, np_type) if count == 1 else (slot, np_type, (count,)))
    return np.dtype(spec), slots


# -- data blocks ----------------------------------------------------------


def _read_data(fh, header: _Header, name: str) -> dict[str, np.ndarray]:
    """Read the data block into a ``field name -> (N,) or (N, count)`` mapping."""
    dtype, slots = _dtype_for(header, name)
    if header.data_kind == "ascii":
        arr = _read_ascii(fh, header, dtype, name)
    elif header.data_kind == "binary_compressed":
        arr = _read_binary_compressed(fh.read(), header, dtype, name)
    else:
        blob = fh.read()
        n = header.n_points or len(blob) // dtype.itemsize
        if len(blob) < n * dtype.itemsize:
            raise ValueError(
                f"{name}: truncated PCD data block — {len(blob)} bytes for "
                f"{n} points of {dtype.itemsize} bytes"
            )
        arr = np.frombuffer(blob, dtype=dtype, count=n)
    return {field: arr[slot] for field, slot in slots.items()}


def _read_ascii(fh, header: _Header, dtype: np.dtype, name: str) -> np.ndarray:
    """Parse ascii rows into a record array of ``dtype``.

    Every column is read as float64 first (it is the only type that swallows the
    ``nan`` tokens PCL writes for missing points), then cast back to each field's
    declared type — which matters for the integer-packed ``rgba``.
    """
    n_cols = sum(header.counts)
    rows = np.loadtxt(fh, dtype=np.float64, ndmin=2)
    if rows.size == 0:
        rows = rows.reshape(0, n_cols)
    if rows.shape[1] != n_cols:
        raise ValueError(f"{name}: expected {n_cols} ascii columns per point, got {rows.shape[1]}")

    arr = np.zeros(len(rows), dtype=dtype)
    col = 0
    for slot, count in zip(dtype.names, header.counts, strict=True):
        block = rows[:, col : col + count]
        arr[slot] = block[:, 0] if count == 1 else block
        col += count
    return arr


def _read_binary_compressed(blob: bytes, header: _Header, dtype: np.dtype, name: str) -> np.ndarray:
    """Decode a ``binary_compressed`` block into a record array.

    The block starts with two uint32 (compressed and uncompressed size); the
    payload decompresses to a *field-major* buffer — every point's ``x``, then
    every point's ``y``, and so on — which is transposed back here.
    """
    if len(blob) < 8:
        raise ValueError(f"{name}: truncated binary_compressed PCD data header")
    comp_size, raw_size = struct.unpack("<II", blob[:8])
    payload = blob[8 : 8 + comp_size]
    if len(payload) < comp_size:
        raise ValueError(f"{name}: truncated binary_compressed PCD block")
    raw = _lzf.decompress(payload, raw_size)

    n = header.n_points or raw_size // dtype.itemsize
    if raw_size != n * dtype.itemsize:
        raise ValueError(
            f"{name}: binary_compressed PCD declares {raw_size} bytes, "
            f"expected {n * dtype.itemsize} for {n} points"
        )
    out = np.empty(n, dtype=dtype)
    offset = 0
    for slot in dtype.names:
        sub = dtype[slot]
        per_point = sub.itemsize // sub.base.itemsize
        block = np.frombuffer(raw, dtype=sub.base, count=n * per_point, offset=offset)
        out[slot] = block.reshape((n, *sub.shape)) if sub.shape else block
        offset += block.nbytes
    return out


# -- cloud assembly -------------------------------------------------------


def _to_cloud(record: dict[str, np.ndarray], path: Path) -> PointCloud:
    missing = [axis for axis in ("x", "y", "z") if axis not in record]
    if missing:
        raise ValueError(f"{path.name}: PCD has no {'/'.join(missing)} field")
    xyz = np.stack([_column(record[axis]) for axis in ("x", "y", "z")], axis=1).astype(np.float32)

    features: dict[str, np.ndarray] = {}
    if "intensity" in record:
        features["intensity"] = _column(record["intensity"]).astype(np.float32)
    packed = record.get("rgb", record.get("rgba"))
    if packed is not None:
        features["rgb"] = _unpack_rgb(packed)
    normals = _normals(record)
    if normals is not None:
        features["normals"] = normals

    labels = None
    if "label" in record:
        # PCL's `label` is uint32 and is used both for semantic classes and for
        # raw cluster ids; keep it as the starting annotation only when it fits
        # int32, so an id soup can't wrap into negative class ids.
        values = _column(record["label"]).astype(np.int64)
        if values.size == 0 or (values.min() >= -(2**31) and values.max() < 2**31):
            labels = values.astype(np.int32)

    # Drop invalid returns before anything downstream sees them: organized PCDs
    # store empty pixels as NaN xyz, which have no geometry to show, pick or
    # label and would poison the camera-framing bounds. The survivor mask lets
    # derived arrays be realigned to the on-disk frame (PointCloud.to_source_frame).
    finite = np.isfinite(xyz).all(axis=1)
    source_index = source_count = None
    if not finite.all():
        source_count = int(len(xyz))
        source_index = np.flatnonzero(finite)
        xyz = xyz[finite]
        features = {key: value[finite] for key, value in features.items()}
        if labels is not None:
            labels = labels[finite]

    return PointCloud(
        xyz=xyz, features=features, labels=labels, source=path,
        source_index=source_index, source_count=source_count,
    )  # fmt: skip


def _column(values: np.ndarray) -> np.ndarray:
    """First element of a field: a no-op unless the field has ``COUNT`` > 1."""
    values = np.asarray(values)
    return values if values.ndim == 1 else values[:, 0]


def _normals(record: dict[str, np.ndarray]) -> np.ndarray | None:
    """Normals from PCL's ``normal_x/normal_y/normal_z``, or a ``COUNT 3`` field."""
    axes = ("normal_x", "normal_y", "normal_z")
    if all(axis in record for axis in axes):
        return np.stack([_column(record[axis]) for axis in axes], axis=1).astype(np.float32)
    packed = record.get("normal", record.get("normals"))
    if packed is not None and np.asarray(packed).ndim == 2 and np.asarray(packed).shape[1] >= 3:
        return np.asarray(packed)[:, :3].astype(np.float32)
    return None


def _unpack_rgb(packed: np.ndarray) -> np.ndarray:
    """Decode PCD's packed RGB (a 32-bit value carried as float or uint) to ``(N, 3)`` uint8."""
    packed = _column(packed)
    if packed.dtype.kind == "f":
        # PCL stores the colour bits inside a float32 — reinterpret, never convert.
        as_u32 = packed.astype(np.float32).view(np.uint32)
    else:
        as_u32 = packed.astype(np.uint32)
    r = (as_u32 >> 16) & 0xFF
    g = (as_u32 >> 8) & 0xFF
    b = as_u32 & 0xFF
    return np.stack([r, g, b], axis=1).astype(np.uint8)
