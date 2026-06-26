from __future__ import annotations

import numpy as np
import pytest

from toaster.io import load_cloud, supported_extensions


def test_supported_extensions_include_builtins():
    exts = supported_extensions()
    assert {".ply", ".bin", ".las", ".laz", ".pcd"} <= set(exts)


def test_bin_roundtrip(tmp_path):
    pts = np.array([[0, 0, 0, 0.1], [1, 2, 3, 0.5]], dtype=np.float32)
    path = tmp_path / "scan.bin"
    pts.tofile(path)
    cloud = load_cloud(path)
    assert cloud.n == 2
    assert np.allclose(cloud.xyz, pts[:, :3])
    assert np.allclose(cloud.features["intensity"], pts[:, 3])
    assert cloud.source == path


def test_ply_roundtrip(tmp_path):
    plyfile = pytest.importorskip("plyfile")
    verts = np.array(
        [(0.0, 0.0, 0.0, 0.2), (1.0, 1.0, 1.0, 0.8)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")],
    )
    el = plyfile.PlyElement.describe(verts, "vertex")
    path = tmp_path / "scan.ply"
    plyfile.PlyData([el], text=True).write(str(path))

    cloud = load_cloud(path)
    assert cloud.n == 2
    assert np.allclose(cloud.xyz[1], [1, 1, 1])
    assert np.allclose(cloud.features["intensity"], [0.2, 0.8])


def test_pcd_ascii_roundtrip(tmp_path):
    path = tmp_path / "scan.pcd"
    path.write_text(
        "# .PCD v0.7\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        "WIDTH 3\nHEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS 3\nDATA ascii\n"
        "0 0 0 0.1\n1 1 1 0.2\n2 2 2 0.3\n"
    )
    cloud = load_cloud(path)
    assert cloud.n == 3
    assert np.allclose(cloud.xyz[2], [2, 2, 2])
    assert np.allclose(cloud.features["intensity"], [0.1, 0.2, 0.3])


def test_las_roundtrip(tmp_path):
    laspy = pytest.importorskip("laspy")
    las = laspy.create(point_format=3)
    las.x = np.array([0.0, 1.0, 2.0])
    las.y = np.array([0.0, 1.0, 2.0])
    las.z = np.array([0.0, 0.5, 1.0])
    las.intensity = np.array([10, 20, 30])
    path = tmp_path / "scan.las"
    las.write(str(path))

    cloud = load_cloud(path)
    assert cloud.n == 3
    assert np.allclose(cloud.xyz[1], [1.0, 1.0, 0.5], atol=1e-3)


def test_unknown_extension_raises(tmp_path):
    path = tmp_path / "x.foo"
    path.write_text("nope")
    with pytest.raises(ValueError):
        load_cloud(path)
