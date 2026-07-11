"""Loading an external per-point label channel as a grouping.

The channel is any file with one integer per point (``.npy`` or a raw binary);
loaded, it behaves like a segmenter's output — each distinct value a group. The
loader validates the length and, for a NaN-filtered cloud, realigns a whole-frame
channel onto the kept points.
"""

from __future__ import annotations

import numpy as np
import pytest

from toaster.api.service import AnnotationService
from toaster.core import PointCloud
from toaster.io import load_label_channel


def _cloud(n: int) -> PointCloud:
    return PointCloud(xyz=np.zeros((n, 3), np.float32))


# -- npy channels -----------------------------------------------------------


def test_npy_int_channel_maps_to_groups(tmp_path):
    path = tmp_path / "labels.npy"
    np.save(path, np.array([0, 0, 1, 2, 2], np.int64))
    out = load_label_channel(path, _cloud(5))
    assert out.dtype == np.int32
    assert out.tolist() == [0, 0, 1, 2, 2]


def test_npy_column_vector_is_flattened(tmp_path):
    path = tmp_path / "labels.npy"
    np.save(path, np.array([[3], [3], [7]], np.int32))  # (N, 1)
    assert load_label_channel(path, _cloud(3)).tolist() == [3, 3, 7]


def test_npy_whole_valued_floats_are_accepted(tmp_path):
    path = tmp_path / "labels.npy"
    np.save(path, np.array([1.0, 2.0, 2.0], np.float32))
    assert load_label_channel(path, _cloud(3)).tolist() == [1, 2, 2]


def test_npy_fractional_floats_are_rejected(tmp_path):
    path = tmp_path / "labels.npy"
    np.save(path, np.array([1.0, 2.5, 3.0], np.float32))
    with pytest.raises(ValueError, match="integer-valued"):
        load_label_channel(path, _cloud(3))


def test_wrong_length_is_rejected(tmp_path):
    path = tmp_path / "labels.npy"
    np.save(path, np.array([0, 1, 2, 3], np.int32))
    with pytest.raises(ValueError, match="4 points but the cloud has 3"):
        load_label_channel(path, _cloud(3))


# -- raw binary channels (dtype inferred from the file size) ----------------


def test_raw_int32_channel(tmp_path):
    path = tmp_path / "labels.height_label"  # free extension -> raw path
    np.array([5, 5, 9, 9], np.int32).tofile(path)
    assert load_label_channel(path, _cloud(4)).tolist() == [5, 5, 9, 9]


def test_raw_int8_channel_infers_narrow_width(tmp_path):
    path = tmp_path / "labels.bin"
    np.array([1, 2, 3, 4], np.int8).tofile(path)  # 4 bytes / 4 points -> int8
    assert load_label_channel(path, _cloud(4)).tolist() == [1, 2, 3, 4]


def test_raw_preserves_negative_noise_label(tmp_path):
    path = tmp_path / "labels.bin"
    np.array([-1, 0, 1], np.int32).tofile(path)
    assert load_label_channel(path, _cloud(3)).tolist() == [-1, 0, 1]


def test_raw_indivisible_size_is_rejected(tmp_path):
    path = tmp_path / "labels.bin"
    path.write_bytes(b"\x00\x01\x02")  # 3 bytes, 4 points -> no whole itemsize
    with pytest.raises(ValueError, match="don't divide"):
        load_label_channel(path, _cloud(4))


# -- alignment to a NaN-filtered cloud --------------------------------------


def test_whole_frame_channel_realigns_onto_kept_points(tmp_path):
    # A cloud whose middle point was a NaN return: n=2, full frame=3.
    frame = np.array([[0, 0, 0], [np.nan, np.nan, np.nan], [1, 1, 1]], np.float32)
    cloud_path = tmp_path / "scan.npy"
    np.save(cloud_path, frame)
    from toaster.io import load_cloud

    cloud = load_cloud(cloud_path)
    assert cloud.n == 2 and cloud.source_count == 3

    # A channel written against the full on-disk frame (length 3).
    ch_path = tmp_path / "gt.npy"
    np.save(ch_path, np.array([7, 99, 8], np.int32))
    # Realigned to the kept rows (0 and 2); the NaN row's label is dropped.
    assert load_label_channel(ch_path, cloud).tolist() == [7, 8]


def test_cloud_length_wins_over_frame_length(tmp_path):
    # When nothing was dropped, source_count == n; a length-n channel is direct.
    ch_path = tmp_path / "gt.npy"
    np.save(ch_path, np.array([1, 2, 3], np.int32))
    assert load_label_channel(ch_path, _cloud(3)).tolist() == [1, 2, 3]


# -- through the service (the web path) -------------------------------------


def test_open_channel_makes_a_grouping_active(tmp_path):
    cloud_path = tmp_path / "cloud.npy"
    np.save(cloud_path, np.zeros((5, 3), np.float32))
    ch_path = tmp_path / "labels.npy"
    np.save(ch_path, np.array([0, 0, 1, 2, -1], np.int32))

    svc = AnnotationService()
    svc.open_cloud(str(cloud_path))
    state = svc.open_channel(str(ch_path))

    snap = state["snapshot"]
    assert snap["display_mode"] == "grouping"
    assert snap["active_grouping"]["source"] == "labels"
    assert snap["active_grouping"]["n_groups"] == 3  # 0, 1, 2 (-1 is noise)
    assert state["grouping"] is not None


def test_open_channel_length_mismatch_raises(tmp_path):
    cloud_path = tmp_path / "cloud.npy"
    np.save(cloud_path, np.zeros((5, 3), np.float32))
    ch_path = tmp_path / "labels.npy"
    np.save(ch_path, np.array([0, 1, 2], np.int32))

    svc = AnnotationService()
    svc.open_cloud(str(cloud_path))
    with pytest.raises(ValueError, match="3 points but the cloud has 5"):
        svc.open_channel(str(ch_path))
