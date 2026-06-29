"""Loader for ``.las`` / ``.laz`` lidar files via ``laspy``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from toaster.core import PointCloud

__all__ = ["LasLoader"]


class LasLoader:
    """Reads LAS/LAZ geometry plus intensity and (if present) RGB.

    The ``classification`` dimension is read into ``labels`` so an existing
    classification can be edited and (in a later milestone) written back.
    """

    extensions = (".las", ".laz")

    def load(self, path: str | Path) -> PointCloud:
        import laspy

        path = Path(path)
        las = laspy.read(str(path))
        # `las.x/y/z` apply scale + offset and return float64 world coordinates.
        xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)

        features: dict[str, np.ndarray] = {}
        if "intensity" in las.point_format.dimension_names:
            features["intensity"] = np.asarray(las.intensity, dtype=np.float32)
        dims = set(las.point_format.dimension_names)
        if {"red", "green", "blue"} <= dims:
            # LAS stores 16-bit colour; scale down to 8-bit for display.
            rgb16 = np.stack([las.red, las.green, las.blue], axis=1).astype(np.uint32)
            features["rgb"] = (rgb16 >> 8).astype(np.uint8)

        labels = None
        if "classification" in dims:
            labels = np.asarray(las.classification, dtype=np.int32)

        return PointCloud(xyz=xyz, features=features, labels=labels, source=path)
