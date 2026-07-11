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


# A ground filter only works if it knows which way is up. By default that is the
# cloud's +Z, but a scan need not be gravity-aligned — so the caller may pass an
# ``up`` vector (e.g. the camera's screen-up, once the view has been turned the
# right way round) and the filter reasons relative to that instead of the cloud.
_GROUND_MAX_TILT = float(np.cos(np.radians(35.0)))  # ground-normal-vs-up tolerance


def _unit_up(up) -> np.ndarray | None:
    """Normalise an ``up`` vector to a unit 3-vector; ``None`` means the cloud's +Z."""
    if up is None:
        return None
    u = np.asarray(up, dtype=np.float64).reshape(-1)
    if u.shape != (3,):
        raise ValueError(f"up must be a 3-vector, got shape {np.shape(up)}")
    norm = float(np.linalg.norm(u))
    if norm < 1e-9:
        raise ValueError("up vector must be non-zero")
    return u / norm


def _align_up_to_z(xyz: np.ndarray, up: np.ndarray | None) -> np.ndarray:
    """Rotate points so ``up`` becomes +Z (a no-op when ``up`` is None or already +Z).

    Filters that key off Z — lowest-point binning, a cloth dropped along -Z —
    then operate in this gravity-aligned frame. Only a boolean ground mask is
    handed back to the caller, so the rotation never has to be undone.
    """
    if up is None:
        return xyz
    z = np.array([0.0, 0.0, 1.0])
    c = float(up @ z)
    if c > 1 - 1e-9:
        return xyz  # already pointing up
    if c < -1 + 1e-9:
        # Exactly upside-down: flip about X (z -> -z) and keep a right-handed frame.
        return xyz @ np.diag([1.0, -1.0, -1.0]).T
    v = np.cross(up, z)
    s = float(np.linalg.norm(v))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))  # Rodrigues: rot @ up == z
    return xyz @ rot.T


class RANSACGroundSegmenter:
    """Fit the dominant plane with RANSAC; points near it are ground.

    Best for roughly-flat scenes. For uneven/offroad terrain prefer
    :class:`GroundGridSegmenter`.
    """

    name = "ransac_ground"
    USES_GRAVITY = True
    PARAMS = [
        Param("threshold", "float", 0.2, 0.01, 5.0, 0.01),
        Param("iterations", "int", 200, 10, 2000, 10),
    ]

    def __init__(
        self, threshold: float = 0.2, iterations: int = 200,
        ground_class: int = 1, obstacle_class: int = 2, up=None,
    ) -> None:  # fmt: skip
        self.threshold = float(threshold)
        self.iterations = int(iterations)
        self.ground_class = int(ground_class)
        self.obstacle_class = int(obstacle_class)
        # With an up direction, RANSAC keeps only near-horizontal planes — so it
        # locks onto the ground rather than the largest plane (often a wall).
        self.up = _unit_up(up)

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        xyz, indices = resolve_points(cloud, selection)
        if len(xyz) < 3:
            return all_noise(indices, cloud.n, source=self.name)
        ground = _ransac_plane(
            np.asarray(xyz, np.float64), self.threshold, self.iterations, up=self.up
        )
        params = {"threshold": self.threshold, "iterations": self.iterations}
        if self.up is not None:
            params["up"] = self.up.tolist()
        return _ground_grouping(
            ground, indices, cloud.n, self.name, self.ground_class, self.obstacle_class, params,
        )  # fmt: skip


class GroundGridSegmenter:
    """Per-cell lowest-point ground: handles uneven terrain (no plane assumption).

    The XY plane is binned into ``cell``-sized squares; points within
    ``threshold`` above their cell's lowest point are ground.
    """

    name = "ground_grid"
    USES_GRAVITY = True
    PARAMS = [
        Param("cell", "float", 1.0, 0.1, 20.0, 0.1),
        Param("threshold", "float", 0.3, 0.01, 3.0, 0.01),
    ]

    def __init__(
        self, cell: float = 1.0, threshold: float = 0.3,
        ground_class: int = 1, obstacle_class: int = 2, up=None,
    ) -> None:  # fmt: skip
        self.cell = float(cell)
        self.threshold = float(threshold)
        self.ground_class = int(ground_class)
        self.obstacle_class = int(obstacle_class)
        self.up = _unit_up(up)

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        xyz, indices = resolve_points(cloud, selection)
        if len(xyz) < 1:
            return all_noise(indices, cloud.n, source=self.name)
        # Bin and measure height along the up axis (the cloud's +Z, or the given up).
        xyz = _align_up_to_z(np.asarray(xyz, np.float64), self.up)
        ij = np.floor(xyz[:, :2] / self.cell).astype(np.int64)
        _, inv = np.unique(ij, axis=0, return_inverse=True)
        min_z = np.full(inv.max() + 1, np.inf)
        np.minimum.at(min_z, inv, xyz[:, 2])
        ground = xyz[:, 2] - min_z[inv] < self.threshold
        params = {"cell": self.cell, "threshold": self.threshold}
        if self.up is not None:
            params["up"] = self.up.tolist()
        return _ground_grouping(
            ground, indices, cloud.n, self.name, self.ground_class, self.obstacle_class, params,
        )  # fmt: skip


class CSFGroundSegmenter:
    """Cloth Simulation Filter ground extraction (needs the ``csf`` extra)."""

    name = "csf"
    USES_GRAVITY = True
    PARAMS = [
        Param("cloth_resolution", "float", 0.5, 0.05, 5.0, 0.05),
        Param("threshold", "float", 0.3, 0.01, 3.0, 0.01),
    ]

    def __init__(
        self, cloth_resolution: float = 0.5, threshold: float = 0.3,
        ground_class: int = 1, obstacle_class: int = 2, up=None,
    ) -> None:  # fmt: skip
        self.cloth_resolution = float(cloth_resolution)
        self.threshold = float(threshold)
        self.ground_class = int(ground_class)
        self.obstacle_class = int(obstacle_class)
        self.up = _unit_up(up)

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
        # CSF drops the cloth along -Z, so feed it the gravity-aligned points.
        csf.setPointCloud(_align_up_to_z(np.asarray(xyz, np.float64), self.up))
        ground_idx, non_ground_idx = CSF.VecInt(), CSF.VecInt()
        # exportCloth defaults to True and dumps a 'cloth_nodes.txt' in the cwd.
        csf.do_filtering(ground_idx, non_ground_idx, False)
        ground = np.zeros(len(xyz), dtype=bool)
        ground[np.asarray(ground_idx, dtype=np.int64)] = True
        params = {"cloth_resolution": self.cloth_resolution, "threshold": self.threshold}
        if self.up is not None:
            params["up"] = self.up.tolist()
        return _ground_grouping(
            ground, indices, cloud.n, self.name, self.ground_class, self.obstacle_class, params,
        )  # fmt: skip


def _ransac_plane(
    xyz: np.ndarray, threshold: float, iterations: int, seed: int = 0, up=None
) -> np.ndarray:
    """Boolean inlier mask for the best-fit plane found by RANSAC.

    With ``up`` set, only planes whose normal is within :data:`_GROUND_MAX_TILT`
    of that axis are considered, so the search finds the dominant *horizontal*
    plane (the ground) instead of the largest plane overall (often a wall).
    """
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
        if up is not None and abs(float(normal @ up)) < _GROUND_MAX_TILT:
            continue  # too tilted to be the ground under this gravity direction
        dist = np.abs((xyz - p[0]) @ normal)
        inliers = dist < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best = count, inliers
    return best
