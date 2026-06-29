"""Loader for NumPy ``.npy`` arrays holding a single point cloud.

The file is one 2-D array ``(N, C)``; the column count decides the layout:

==========  =====================================================
``C``       Interpretation
==========  =====================================================
``3``       ``x, y, z``
``4``       ``x, y, z, intensity`` (as KITTI ``.bin`` stores it)
``6``       ``x, y, z`` + three channels guessed as ``rgb`` or
            ``normals`` (see :meth:`NpyLoader._guess_trailing`)
==========  =====================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from toaster.core import PointCloud

__all__ = ["NpyLoader"]


class NpyLoader:
    """Reads a single ``(N, 3|4|6)`` numeric array saved with :func:`numpy.save`."""

    extensions = (".npy",)

    def load(self, path: str | Path) -> PointCloud:
        path = Path(path)
        # allow_pickle stays False (the default): never execute code from a data file.
        arr = np.load(str(path))
        if arr.dtype.names is not None:
            raise ValueError(
                f"{path.name}: structured/record arrays are not supported; "
                "save a plain numeric (N, 3|4|6) array"
            )
        arr = np.atleast_2d(arr)
        if arr.ndim != 2 or arr.shape[1] not in (3, 4, 6):
            raise ValueError(
                f"{path.name}: expected a 2-D array with 3, 4 or 6 columns "
                f"([x,y,z] / [x,y,z,intensity] / [x,y,z,rgb|normals]), got shape {arr.shape}"
            )

        xyz = arr[:, :3]
        features: dict[str, np.ndarray] = {}
        if arr.shape[1] == 4:
            features["intensity"] = np.ascontiguousarray(arr[:, 3], dtype=np.float32)
        elif arr.shape[1] == 6:
            key, values = self._guess_trailing(arr[:, 3:])
            features[key] = values

        return PointCloud(xyz=xyz, features=features, source=path)

    @staticmethod
    def _guess_trailing(trailing: np.ndarray) -> tuple[str, np.ndarray]:
        """Classify the three trailing columns as ``rgb`` (uint8) or ``normals``.

        Colour channels are never negative, so any negative value means ``normals``.
        Otherwise the columns are read as ``rgb``: a max above 1 is taken as already
        0–255, and a max within [0, 1] is scaled up to 0–255. (All-positive normals
        are rare and degenerate; they read as colour — supply a custom loader naming
        the channel if you have them.)
        """
        if trailing.size and trailing.min() < 0:
            return "normals", np.ascontiguousarray(trailing, dtype=np.float32)
        scale = 1.0 if (trailing.size and trailing.max() > 1.0) else 255.0
        rgb = np.clip(np.rint(trailing * scale), 0, 255).astype(np.uint8)
        return "rgb", np.ascontiguousarray(rgb)
