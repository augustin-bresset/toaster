"""Generate a small synthetic outdoor scene as a KITTI-style .bin sample.

Run: ``python examples/make_sample.py`` -> writes ``examples/sample.bin`` (a real
lidar sweep of Perlin-noise terrain with trees and patchy grass — ground, natural
obstacles and vegetation, enough to try every segmenter, ground detection and the
voxel mode). Open it with ``toaster examples/sample.bin``.

The same generator (``toaster.demo_scene``) backs ``toaster demo`` / typing "demo"
into the Open dialog, which builds a fresh scene on the fly with a random seed
instead of reading this fixed file.
"""

from __future__ import annotations

from pathlib import Path

from toaster.demo_scene import build_demo_points

if __name__ == "__main__":
    out = Path(__file__).with_name("sample.bin")
    pts = build_demo_points()
    pts.tofile(out)
    print(f"wrote {out} ({len(pts):,} points)")
