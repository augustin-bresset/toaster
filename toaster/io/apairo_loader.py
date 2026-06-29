"""Optional bridge to apairo datasets (``apairo`` extra).

apairo addresses *datasets* (directories of frames) rather than single files, so
it sits beside the file-extension registry rather than inside it. This wraps one
frame of an ``apairo.RawDataset`` as a :class:`~toaster.core.PointCloud`, which
is what the app's frame navigation uses. ``apairo`` is imported lazily.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from toaster.core import PointCloud

__all__ = ["ApairoFrames"]


class ApairoFrames:
    """A navigable view over an apairo dataset, yielding Toaster point clouds.

    Args:
        root: Dataset root (containing ``.apairo/channels.yaml``).
        point_key: Channel holding the ``(N, >=3)`` point array.
        label_key: Optional channel holding ``(N,)`` per-point labels.
    """

    def __init__(self, root: str | Path, point_key: str = "lidar", label_key: str | None = None):
        import apairo

        self.root = Path(root)
        self.point_key = point_key
        self.label_key = label_key
        keys = [point_key] + ([label_key] if label_key else [])
        self._ds = apairo.RawDataset(str(root), keys=keys)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> PointCloud:
        sample = self._ds[index]
        pts = np.asarray(sample.data[self.point_key], dtype=np.float32)
        xyz = pts[:, :3].copy()
        features: dict[str, np.ndarray] = {}
        if pts.shape[1] > 3:
            features["intensity"] = pts[:, 3].copy()
        labels = None
        if self.label_key is not None and self.label_key in sample.data:
            labels = np.asarray(sample.data[self.label_key], dtype=np.int32)
        # Synthesise a per-frame identity so the label sidecar is per-frame.
        source = self.root / f"{self.point_key}_{index:06d}"
        return PointCloud(xyz=xyz, features=features, labels=labels, source=source)
