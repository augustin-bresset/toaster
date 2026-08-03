"""Optional Open3D-backed loader (``open3d`` extra).

Not registered by default: the built-in ``.pcd`` / ``.ply`` readers cover the
same files and keep the intensity and label channels Open3D's ``PointCloud``
has no room for. Kept as an escape hatch for a file the built-ins choke on —
:func:`~toaster.io.register_loader` it yourself to take over those extensions.
Open3D is heavy, so it is imported lazily inside :meth:`load`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from toaster.core import PointCloud

__all__ = ["Open3DLoader"]


class Open3DLoader:
    """Reads ``.pcd`` / ``.ply`` via ``open3d.io.read_point_cloud``."""

    extensions = (".pcd", ".ply")

    def load(self, path: str | Path) -> PointCloud:
        import open3d as o3d

        path = Path(path)
        pc = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(pc.points, dtype=np.float32)
        features: dict[str, np.ndarray] = {}
        if pc.has_colors():
            features["rgb"] = (np.asarray(pc.colors) * 255).astype(np.uint8)
        if pc.has_normals():
            features["normals"] = np.asarray(pc.normals, dtype=np.float32)
        return PointCloud(xyz=xyz, features=features, source=path)
