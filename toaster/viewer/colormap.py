"""Colour helpers: turn labels / groupings / scalars into ``(N, 3)`` uint8 buffers.

All vectorized (LUT indexing, no Python per-point loops) so they stay cheap at
hundreds of thousands of points.
"""

from __future__ import annotations

import numpy as np

from toaster.core import Grouping, LabelSchema

__all__ = [
    "colors_from_labels",
    "colors_from_grouping",
    "colors_from_scalar",
    "group_color",
    "GROUP_NOISE_COLOR",
]

#: Colour drawn for noise / unassigned points when colouring by grouping.
GROUP_NOISE_COLOR = np.array([90, 90, 90], dtype=np.uint8)

# A fixed, perceptually-spread palette for anonymous groups (Glasbey-like).
_GROUP_PALETTE = np.array(
    [
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
        [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
        [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
    ],
    dtype=np.uint8,
)  # fmt: skip


def group_color(group_id: int) -> tuple[int, int, int]:
    """The display colour of one group — matches :func:`colors_from_grouping`."""
    if group_id < 0:
        return tuple(int(c) for c in GROUP_NOISE_COLOR)
    return tuple(int(c) for c in _GROUP_PALETTE[group_id % len(_GROUP_PALETTE)])


def colors_from_labels(labels: np.ndarray, schema: LabelSchema) -> np.ndarray:
    """``(N, 3)`` uint8 colours for a label array, via the schema LUT."""
    return schema.colors_for(labels)


def colors_from_grouping(grouping: Grouping) -> np.ndarray:
    """``(N, 3)`` uint8 colours for a grouping; noise is grey, groups cycle a palette."""
    gid = grouping.group_id
    colors = np.empty((gid.shape[0], 3), dtype=np.uint8)
    noise = gid < 0
    colors[noise] = GROUP_NOISE_COLOR
    real = ~noise
    colors[real] = _GROUP_PALETTE[gid[real] % len(_GROUP_PALETTE)]
    return colors


def colors_from_scalar(values: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """``(N, 3)`` uint8 colours from a scalar channel (e.g. intensity/height).

    Normalises to the 2nd–98th percentile to resist outliers. Falls back to a
    plain grey ramp if matplotlib is unavailable.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = np.percentile(values, [2, 98]) if values.size else (0.0, 1.0)
    norm = np.clip((values - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    try:
        from matplotlib import colormaps

        rgba = colormaps[cmap](norm)
        return (rgba[:, :3] * 255).astype(np.uint8)
    except Exception:
        grey = (norm * 255).astype(np.uint8)
        return np.stack([grey, grey, grey], axis=1)
