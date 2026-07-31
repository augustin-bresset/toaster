"""TF resolution for apairo frames: detection (filesystem) and the resolved 4x4.

``detect_tf`` is pure filesystem/YAML, so its tests build a tiny sequence on disk
and need no apairo. ``resolve_tf_matrix`` needs apairo + apairo_transform; that
test builds a minimal loadable dataset and is skipped where they are absent.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from toaster.io.apairo_dataset import detect_tf, list_tf_targets, resolve_tf_matrix

CLOUD = "ouster_points"
POSE = "tf__odom__base_link"
LIDAR_FRAME = "os_sensor"
BASE_FRAME = "base_link"

# Mount edge ``os_sensor_to_base_link`` = T_os_sensor_from_base_link (a 90° tilt +
# offset). get_tf(os_sensor, base_link) yields its inverse, so a resolve with an
# identity dynamic pose must equal inv(_MOUNT_EDGE).
_MOUNT_EDGE = np.array([[0, 0, 1, 1.0], [0, 1, 0, 0.0], [-1, 0, 0, 0.5], [0, 0, 0, 1]], dtype=float)


def _write_seq(
    tmp_path,
    *,
    cloud_frame: str | None = LIDAR_FRAME,
    calib_edges=("os_sensor_to_base_link",),
    with_pose_channel: bool = True,
    with_map: bool = False,
    with_calibration: bool = True,
    n_points: int = 5,
    n_poses: int = 3,
):
    """Write a minimal apairo sequence and return the open cloud frame's path."""
    seq = tmp_path / "seq"
    ap = seq / ".apairo"
    ap.mkdir(parents=True)

    def pose_ch(parent, child):
        return {
            "kind": "raw",
            "loader": "npy",
            "transform": {"parent": parent, "child": child, "format": "t_xyz_q_xyzw"},
        }

    def write_pose(name):
        pdir = seq / name
        pdir.mkdir()
        # stacked (M,7) [tx ty tz qx qy qz qw]; identity rotation, moving translation
        poses = np.zeros((n_poses, 7), np.float64)
        poses[:, 0] = np.arange(n_poses) * 10.0  # tx walks away
        poses[:, 6] = 1.0  # qw = 1 (identity rotation)
        np.save(pdir / f"{name}.npy", poses)
        (pdir / "timestamps.txt").write_text(
            "\n".join(str(1000.0 + i) for i in range(n_poses)) + "\n"
        )

    channels: dict = {CLOUD: {"kind": "raw", "loader": "npys"}}
    if cloud_frame is not None:
        channels[CLOUD]["frame"] = cloud_frame
    if with_pose_channel:
        channels[POSE] = pose_ch("odom", BASE_FRAME)
    if with_map:
        channels["tf__map__odom"] = pose_ch("map", "odom")
    (ap / "channels.yaml").write_text(yaml.safe_dump({"channels": channels}))

    if with_calibration:
        transforms = {}
        for edge in calib_edges:
            parent, _, child = edge.partition("_to_")
            transforms[edge] = {"parent": parent, "child": child, "matrix": _MOUNT_EDGE.tolist()}
        (ap / "calibration.yaml").write_text(yaml.safe_dump({"transforms": transforms}))

    # cloud frames (npys loader: one .npy per frame) + timestamps
    cdir = seq / CLOUD
    cdir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        arr = rng.standard_normal((n_points, 4)).astype(np.float32)
        np.save(cdir / f"{i:06d}.npy", arr)
    (cdir / "timestamps.txt").write_text("\n".join(str(1000.0 + i) for i in range(2)) + "\n")

    if with_pose_channel:
        write_pose(POSE)
    if with_map:
        write_pose("tf__map__odom")

    return cdir / "000000.npy"


# ── detect_tf (filesystem only) ──────────────────────────────────────────────


def test_detect_tf_resolvable_chain(tmp_path):
    tf = detect_tf(_write_seq(tmp_path))
    assert tf is not None
    assert (tf.target, tf.base_frame, tf.lidar_frame, tf.pose_channels) == (
        "odom",
        BASE_FRAME,
        LIDAR_FRAME,
        (POSE,),
    )


def test_detect_tf_none_for_non_apairo_path(tmp_path):
    (tmp_path / "loose.npy").write_bytes(b"")
    assert detect_tf(tmp_path / "loose.npy") is None
    assert detect_tf(None) is None


def test_detect_tf_none_without_calibration(tmp_path):
    assert detect_tf(_write_seq(tmp_path, with_calibration=False)) is None


def test_detect_tf_none_when_no_base_frame_in_tree(tmp_path):
    # calibration connects the lidar to some other frame, but no base_link.
    p = _write_seq(tmp_path, calib_edges=("os_sensor_to_os_lidar",))
    assert detect_tf(p) is None


def test_detect_tf_none_when_lidar_frame_absent_from_tree(tmp_path):
    # base_link exists but the cloud's own frame is not in the calibration tree.
    p = _write_seq(tmp_path, calib_edges=("imu_link_to_base_link",))
    assert detect_tf(p) is None


def test_detect_tf_none_without_pose_channel(tmp_path):
    assert detect_tf(_write_seq(tmp_path, with_pose_channel=False)) is None


def test_detect_tf_none_when_cloud_channel_has_no_frame(tmp_path):
    assert detect_tf(_write_seq(tmp_path, cloud_frame=None)) is None


def test_detect_tf_none_for_other_target(tmp_path):
    # Only tf__odom__base_link exists, so resolving to "map" is unavailable.
    assert detect_tf(_write_seq(tmp_path), target="map") is None


def test_detect_tf_map_needs_the_full_chain(tmp_path):
    assert detect_tf(_write_seq(tmp_path / "a"), target="map") is None  # odom hop only
    tf = detect_tf(_write_seq(tmp_path / "b", with_map=True), target="map")
    assert tf is not None and tf.pose_channels == (POSE, "tf__map__odom")


def test_detect_tf_base_frame_is_static_only(tmp_path):
    tf = detect_tf(_write_seq(tmp_path), target=BASE_FRAME)
    assert tf is not None and tf.pose_channels == ()  # no dynamic hop


# ── list_tf_targets ──────────────────────────────────────────────────────────


def test_list_tf_targets_base_outward(tmp_path):
    assert list_tf_targets(_write_seq(tmp_path, with_map=True)) == [BASE_FRAME, "odom", "map"]


def test_list_tf_targets_without_map(tmp_path):
    assert list_tf_targets(_write_seq(tmp_path)) == [BASE_FRAME, "odom"]


def test_list_tf_targets_empty_without_calibration(tmp_path):
    assert list_tf_targets(_write_seq(tmp_path, with_calibration=False)) == []


# ── resolve_tf_matrix guards (no apairo needed) ──────────────────────────────


def test_resolve_returns_none_when_unresolvable(tmp_path):
    assert resolve_tf_matrix(_write_seq(tmp_path, with_pose_channel=False), 0) is None


def test_resolve_returns_none_for_negative_index(tmp_path):
    assert resolve_tf_matrix(_write_seq(tmp_path), -1) is None


# ── resolve_tf_matrix integration (needs apairo + apairo_transform) ──────────


def test_resolve_matrix_composes_mount_and_pose(tmp_path):
    pytest.importorskip("apairo")
    pytest.importorskip("apairo_transform")
    import apairo  # noqa: F401 -- ensure the synthetic layout is loadable

    path = _write_seq(tmp_path)
    tf = resolve_tf_matrix(path, 0, "odom")
    assert tf is not None and tf.shape == (4, 4)
    # Frame 0's timestamp (1000.0) coincides with pose[0] (identity), so the
    # resolved transform must equal the static mount alone: get_tf(os_sensor,
    # base_link) = inv(mount edge).
    np.testing.assert_allclose(tf, np.linalg.inv(_MOUNT_EDGE), atol=1e-6)


def test_resolve_base_frame_is_static_mount_only(tmp_path):
    pytest.importorskip("apairo")
    pytest.importorskip("apairo_transform")
    # Resolving into the base frame is the static mount alone: get_tf(sensor, base).
    tf = resolve_tf_matrix(_write_seq(tmp_path), 0, BASE_FRAME)
    np.testing.assert_allclose(tf, np.linalg.inv(_MOUNT_EDGE), atol=1e-6)


def test_resolve_map_composes_the_two_hops(tmp_path):
    pytest.importorskip("apairo")
    pytest.importorskip("apairo_transform")
    # Both dynamic hops are identity at t0, so map == odom == the static mount.
    tf = resolve_tf_matrix(_write_seq(tmp_path, with_map=True), 0, "map")
    assert tf is not None
    np.testing.assert_allclose(tf, np.linalg.inv(_MOUNT_EDGE), atol=1e-6)


def test_upright_zeroes_translation_keeps_rotation():
    """A diverged odometry can place a frame millions of metres out; the service
    keeps only the rotation so float32 point coordinates stay well-conditioned."""
    from toaster.api.service import AnnotationService

    tf = np.array([[0, -1, 0, 9.4e6], [1, 0, 0, 4.8e6], [0, 0, 1, -2.3e7], [0, 0, 0, 1]], float)
    up = AnnotationService._upright(tf)
    np.testing.assert_array_equal(up[:3, :3], tf[:3, :3])  # orientation preserved
    np.testing.assert_array_equal(up[:3, 3], [0.0, 0.0, 0.0])  # world translation dropped
    assert tf[0, 3] == 9.4e6  # input not mutated
