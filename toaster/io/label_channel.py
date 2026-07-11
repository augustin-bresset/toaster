"""Load a per-point *label channel* — one integer per point — for an open cloud.

A label channel is an external labelisation of the *same* points as the open
cloud: a semantic class id, an instance id, a height bucket — any integer tag,
one per point. Loaded as a :class:`~toaster.core.Grouping` it behaves exactly
like a segmenter's output — each distinct value is a group you can browse,
select and assign — only the groups come from a file instead of a clusterer.

Two on-disk shapes are supported:

* ``.npy`` — a 1-D (or ``(N, 1)``) integer array saved with :func:`numpy.save`.
  Float arrays are accepted only when every value is whole (e.g. ``3.0``).
* anything else (``.bin``, ``.label``, ``.height_label``, a free extension) —
  a flat binary of packed integers. The element width is inferred from the file
  size and the point count (1/2/4/8 bytes → ``int8/16/32/64``); the values are
  assumed to be **integers** — store float labels as ``.npy`` instead.

The channel must cover the same points as the cloud: its length must equal the
cloud's point count, or the count of the *full on-disk frame* when the loader
dropped invalid returns (in which case it is realigned onto the kept points).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from toaster.core import PointCloud

__all__ = ["load_label_channel"]

#: Raw-binary element widths (bytes) → signed integer dtype. Signed so a ``-1``
#: "noise / unlabeled" convention survives the read.
_INT_BY_ITEMSIZE = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}


def load_label_channel(path: str | Path, cloud: PointCloud) -> np.ndarray:
    """Read ``path`` as a per-point int label channel aligned to ``cloud``.

    Args:
        path: The channel file (``.npy`` or a raw integer binary).
        cloud: The open cloud the channel labels; its point count (and NaN
            survivor mask, if any) decides the accepted length and alignment.

    Returns:
        ``(cloud.n,)`` int32, ready to wrap in a :class:`~toaster.core.Grouping`.

    Raises:
        ValueError: The file is not a 1-D integer array, its size doesn't map to
            a whole number of points, or its length matches neither the cloud
            nor its full on-disk frame.
    """
    path = Path(path)
    # Candidate point counts, most specific first: the cloud as loaded, then the
    # full on-disk frame when invalid returns were dropped (a channel written
    # against the raw frame spans that instead).
    candidates = [cloud.n]
    if cloud.source_count is not None and cloud.source_count != cloud.n:
        candidates.append(cloud.source_count)

    if path.suffix.lower() == ".npy":
        values = _load_npy(path)
    else:
        values = _load_raw(path, candidates)

    if values.size == cloud.n:
        aligned = values
    elif cloud.source_count is not None and values.size == cloud.source_count:
        # A whole-frame channel: gather onto the points the cloud actually kept.
        aligned = cloud.from_source_frame(values)
    else:
        expected = str(cloud.n)
        if cloud.source_count is not None and cloud.source_count != cloud.n:
            expected += f" (or {cloud.source_count} for the full frame)"
        raise ValueError(
            f"{path.name}: channel has {values.size} points but the cloud has {expected}"
        )
    return np.ascontiguousarray(aligned, dtype=np.int32)


def _load_npy(path: Path) -> np.ndarray:
    """Read an ``.npy`` label array as a flat int array."""
    # allow_pickle stays False (the default): never execute code from a data file.
    arr = np.load(str(path))
    if arr.dtype.names is not None:
        raise ValueError(f"{path.name}: structured/record arrays are not supported")
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(
            f"{path.name}: expected a 1-D label array (one int per point), got shape {arr.shape}"
        )
    if arr.dtype.kind in "iu":
        return arr.astype(np.int32)
    if arr.dtype.kind in "fc" and np.all(arr == np.round(arr.real)):
        # Whole-valued floats (e.g. labels stored as 3.0) read as their integers.
        return arr.real.astype(np.int32)
    raise ValueError(f"{path.name}: label channel must be integer-valued, got dtype {arr.dtype}")


def _load_raw(path: Path, candidates: list[int]) -> np.ndarray:
    """Read a headerless binary of packed integers, inferring the element width."""
    size = path.stat().st_size
    for length in candidates:
        if length <= 0 or size % length:
            continue
        itemsize = size // length
        dtype = _INT_BY_ITEMSIZE.get(itemsize)
        if dtype is not None:
            return np.fromfile(str(path), dtype=dtype).astype(np.int32)
    counts = " or ".join(str(c) for c in candidates)
    raise ValueError(
        f"{path.name}: {size} bytes don't divide into {counts} integer labels "
        "of 1/2/4/8 bytes — is this the right file, or a float array? (save floats as .npy)"
    )
