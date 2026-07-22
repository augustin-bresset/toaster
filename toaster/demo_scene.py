"""Procedural natural terrain — the scene behind `toaster demo` / `examples/make_sample.py`.

One consistent 3D world (Perlin-noise hills, scattered trees, patchy grass) is built
once; a single lidar sweep is then *derived* from it by a real ray scan, instead of
hand-painted gaussian blobs and boxes.

The world engine itself lives in `projector.terrain` (shared across tools); this
module keeps only the toaster-specific hooks that drive it for the demo cloud.
"""

from __future__ import annotations

import secrets
from pathlib import Path

import numpy as np
from projector.terrain import (
    CANOPY_COLOR,
    DIRT_COLOR,
    GRASS_COLOR,
    SKY_HORIZON,
    SKY_ZENITH,
    SUN_COLOR,
    TRUNK_COLOR,
    Heightfield,
    Scene,
    Trees,
    build_scene,
    render_camera,
    scan_lidar,
)

__all__ = [
    "CANOPY_COLOR",
    "DIRT_COLOR",
    "GRASS_COLOR",
    "SKY_HORIZON",
    "SKY_ZENITH",
    "SUN_COLOR",
    "TRUNK_COLOR",
    "Heightfield",
    "Scene",
    "Trees",
    "build_scene",
    "render_camera",
    "scan_lidar",
    "build_demo_points",
    "demo_cloud_path",
    "write_demo_cloud",
]


# ------------------------------------------------------------------ CLI/API demo hook


def build_demo_points(seed: int = 7) -> np.ndarray:
    """One synthetic outdoor lidar sweep — Perlin-noise terrain, a few trees, patchy
    grass — as `(N, 4)` float32 `[x, y, z, intensity]`. Same scene/scan parameters as
    the bundled `examples/sample.bin`; only the `seed` changes the layout."""
    scene = build_scene(
        seed=seed, x_range=(-22.0, 22.0), y_range=(-22.0, 22.0), n_trees=6, hill_amplitude=1.7
    )
    sensor_z = float(scene.field.height(np.array([0.0]), np.array([0.0]))[0]) + 1.7
    sensor_pos = np.array([0.0, 0.0, sensor_z])
    rng = np.random.default_rng(seed)
    points, _materials = scan_lidar(
        scene, sensor_pos, rng,
        n_rings=48, n_az=520, elev_range=(-25.0, 15.0), max_range=32.0, march_steps=46,
    )  # fmt: skip
    return points.astype(np.float32)


def demo_cloud_path() -> Path:
    """Where `toaster demo` writes the generated scene — a fixed path (not a
    scattered tempfile) so the usual recent-files / reopen-last-folder logic still
    finds it next launch."""
    path = Path.home() / ".config" / "toaster" / "demo.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_demo_cloud(seed: int | None = None) -> Path:
    """Generate a fresh terrain scene (a random layout each call, unless `seed` is
    given) and write it to :func:`demo_cloud_path` in KITTI `.bin` format, ready for
    `load_cloud`. Returns the path."""
    if seed is None:
        seed = secrets.randbits(31)
    path = demo_cloud_path()
    build_demo_points(seed).tofile(path)
    return path
