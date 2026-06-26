"""Loader for KITTI-style ``.bin`` velodyne scans (``(N, 4)`` float32)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from toaster.core import PointCloud

__all__ = ["BinLoader"]


class BinLoader:
    """Reads raw ``float32`` ``[x, y, z, intensity]`` records, as KITTI stores them."""

    extensions = (".bin",)

    def load(self, path: str | Path) -> PointCloud:
        path = Path(path)
        raw = np.fromfile(str(path), dtype=np.float32)
        if raw.size % 4 != 0:
            raise ValueError(
                f"{path.name}: expected a multiple of 4 float32 values "
                f"([x, y, z, intensity]), got {raw.size}"
            )
        pts = raw.reshape(-1, 4)
        return PointCloud(
            xyz=pts[:, :3].copy(),
            features={"intensity": pts[:, 3].copy()},
            source=path,
        )
