"""Loader for ``.ply`` point clouds (ascii or binary) via ``plyfile``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from toaster.core import PointCloud

__all__ = ["PlyLoader"]


class PlyLoader:
    """Reads the ``vertex`` element of a PLY file.

    Picks up ``x/y/z`` plus, when present, ``intensity``, ``red/green/blue``
    (as ``rgb``) and ``nx/ny/nz`` (as ``normals``).
    """

    extensions = (".ply",)

    def load(self, path: str | Path) -> PointCloud:
        from plyfile import PlyData

        path = Path(path)
        ply = PlyData.read(str(path))
        vertex = ply["vertex"].data
        names = vertex.dtype.names or ()

        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        features: dict[str, np.ndarray] = {}

        if "intensity" in names:
            features["intensity"] = np.asarray(vertex["intensity"], dtype=np.float32)
        if {"red", "green", "blue"} <= set(names):
            features["rgb"] = np.stack(
                [vertex["red"], vertex["green"], vertex["blue"]], axis=1
            ).astype(np.uint8)
        if {"nx", "ny", "nz"} <= set(names):
            features["normals"] = np.stack(
                [vertex["nx"], vertex["ny"], vertex["nz"]], axis=1
            ).astype(np.float32)

        return PointCloud(xyz=xyz, features=features, source=path)
