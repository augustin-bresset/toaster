"""Ground-detection segmenters: split a cloud into ground vs. non-ground.

Each returns exactly two groups — ``0`` = ground, ``1`` = non-ground — and
attaches ``suggested_labels`` mapping them to class ids (defaults 1 and 2, i.e.
the bundled *traversable* / *obstacle* schema). So on the default schema,
"Assign all suggested" turns ground into traversable and the rest into obstacle
in one click.
"""

from __future__ import annotations

import numpy as np

from toaster.core import Grouping, PointCloud, Selection

from .base import Param, all_noise, resolve_points, scatter

__all__ = ["RANSACGroundSegmenter", "GroundGridSegmenter", "CSFGroundSegmenter"]


def _ground_grouping(ground_mask, indices, n, source, ground_class, obstacle_class, params):
    ids = np.where(ground_mask, 0, 1).astype(np.int32)
    return scatter(
        ids, indices, n, source=source,
        suggested_labels={0: ground_class, 1: obstacle_class}, params=params,
    )  # fmt: skip


class RANSACGroundSegmenter:
    """Fit the dominant plane with RANSAC; points near it are ground.

    Best for roughly-flat scenes. For uneven/offroad terrain prefer
    :class:`GroundGridSegmenter`.
    """

    name = "ransac_ground"
    PARAMS = [
        Param("threshold", "float", 0.2, 0.01, 5.0, 0.01),
        Param("iterations", "int", 200, 10, 2000, 10),
    ]

    def __init__(
        self, threshold: float = 0.2, iterations: int = 200,
        ground_class: int = 1, obstacle_class: int = 2,
    ) -> None:  # fmt: skip
        self.threshold = float(threshold)
        self.iterations = int(iterations)
        self.ground_class = int(ground_class)
        self.obstacle_class = int(obstacle_class)

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        xyz, indices = resolve_points(cloud, selection)
        if len(xyz) < 3:
            return all_noise(indices, cloud.n, source=self.name)
        ground = _ransac_plane(np.asarray(xyz, np.float64), self.threshold, self.iterations)
        return _ground_grouping(
            ground, indices, cloud.n, self.name, self.ground_class, self.obstacle_class,
            {"threshold": self.threshold, "iterations": self.iterations},
        )  # fmt: skip


class GroundGridSegmenter:
    """Per-cell lowest-point ground: handles uneven terrain (no plane assumption).

    The XY plane is binned into ``cell``-sized squares; points within
    ``threshold`` above their cell's lowest point are ground.
    """

    name = "ground_grid"
    PARAMS = [
        Param("cell", "float", 1.0, 0.1, 20.0, 0.1),
        Param("threshold", "float", 0.3, 0.01, 3.0, 0.01),
    ]

    def __init__(
        self, cell: float = 1.0, threshold: float = 0.3,
        ground_class: int = 1, obstacle_class: int = 2,
    ) -> None:  # fmt: skip
        self.cell = float(cell)
        self.threshold = float(threshold)
        self.ground_class = int(ground_class)
        self.obstacle_class = int(obstacle_class)

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        xyz, indices = resolve_points(cloud, selection)
        if len(xyz) < 1:
            return all_noise(indices, cloud.n, source=self.name)
        ij = np.floor(xyz[:, :2] / self.cell).astype(np.int64)
        _, inv = np.unique(ij, axis=0, return_inverse=True)
        min_z = np.full(inv.max() + 1, np.inf)
        np.minimum.at(min_z, inv, xyz[:, 2])
        ground = xyz[:, 2] - min_z[inv] < self.threshold
        return _ground_grouping(
            ground, indices, cloud.n, self.name, self.ground_class, self.obstacle_class,
            {"cell": self.cell, "threshold": self.threshold},
        )  # fmt: skip


class CSFGroundSegmenter:
    """Cloth Simulation Filter ground extraction (needs the ``csf`` extra)."""

    name = "csf"
    PARAMS = [
        Param("cloth_resolution", "float", 0.5, 0.05, 5.0, 0.05),
        Param("threshold", "float", 0.3, 0.01, 3.0, 0.01),
    ]

    def __init__(
        self, cloth_resolution: float = 0.5, threshold: float = 0.3,
        ground_class: int = 1, obstacle_class: int = 2,
    ) -> None:  # fmt: skip
        self.cloth_resolution = float(cloth_resolution)
        self.threshold = float(threshold)
        self.ground_class = int(ground_class)
        self.obstacle_class = int(obstacle_class)

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        try:
            import CSF
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "CSF ground detection needs the 'csf' extra: pip install cloth-simulation-filter"
            ) from exc

        xyz, indices = resolve_points(cloud, selection)
        if len(xyz) < 3:
            return all_noise(indices, cloud.n, source=self.name)
        csf = CSF.CSF()
        csf.params.cloth_resolution = self.cloth_resolution
        csf.params.class_threshold = self.threshold
        csf.setPointCloud(np.asarray(xyz, np.float64))
        ground_idx, non_ground_idx = CSF.VecInt(), CSF.VecInt()
        # exportCloth defaults to True and dumps a 'cloth_nodes.txt' in the cwd.
        csf.do_filtering(ground_idx, non_ground_idx, False)
        ground = np.zeros(len(xyz), dtype=bool)
        ground[np.asarray(ground_idx, dtype=np.int64)] = True
        return _ground_grouping(
            ground, indices, cloud.n, self.name, self.ground_class, self.obstacle_class,
            {"cloth_resolution": self.cloth_resolution, "threshold": self.threshold},
        )  # fmt: skip


def _ransac_plane(xyz: np.ndarray, threshold: float, iterations: int, seed: int = 0) -> np.ndarray:
    """Boolean inlier mask for the best-fit plane found by RANSAC."""
    rng = np.random.default_rng(seed)
    n = len(xyz)
    best = np.zeros(n, dtype=bool)
    best_count = -1
    for _ in range(iterations):
        p = xyz[rng.choice(n, 3, replace=False)]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        dist = np.abs((xyz - p[0]) @ normal)
        inliers = dist < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best = count, inliers
    return best
