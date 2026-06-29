"""Generate a small synthetic outdoor scene as a KITTI-style .bin sample.

Run: ``python examples/make_sample.py`` -> writes ``examples/sample.bin``
(ground + a few obstacles), enough to try every segmenter, ground detection and
the voxel mode. Open it with ``toaster examples/sample.bin``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_scene(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []

    # Gently undulating ground over ~30 x 30 m.
    g = rng.uniform([-15, -15, 0], [15, 15, 0], (16000, 3))
    g[:, 2] = 0.15 * np.sin(g[:, 0] * 0.3) + 0.1 * np.cos(g[:, 1] * 0.25)
    g[:, 2] += rng.normal(0, 0.02, len(g))
    parts.append(g)

    # "Trees": tall gaussian blobs.
    for cx, cy in [(-9, 6), (0, 9), (10, 7), (-3, -9), (7, -5)]:
        parts.append(rng.normal([cx, cy, 2.2], [0.7, 0.7, 1.2], (700, 3)))

    # "Rocks / obstacles": low filled boxes.
    for cx, cy in [(-6, -3), (4, 2), (8, -8)]:
        parts.append(rng.uniform([cx - 1, cy - 1, 0], [cx + 1, cy + 1, 0.8], (500, 3)))

    xyz = np.vstack(parts).astype(np.float32)
    intensity = (xyz[:, 2] - xyz[:, 2].min())[:, None].astype(np.float32)
    return np.hstack([xyz, intensity]).astype(np.float32)


if __name__ == "__main__":
    out = Path(__file__).with_name("sample.bin")
    pts = make_scene()
    pts.tofile(out)
    print(f"wrote {out} ({len(pts):,} points)")
