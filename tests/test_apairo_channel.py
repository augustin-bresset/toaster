"""Writing Toaster labels back into an apairo dataset as a channel.

The real ``apairo.ChannelWriter`` is an optional extra, so the write path is
exercised against a fake ``apairo`` module injected into ``sys.modules`` — this
checks the *wiring* (writer arguments, frame-aligned labels) without the package.
Detection, timestamps and the survivor-mask scatter need no apairo at all.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from toaster.core import PointCloud
from toaster.io import load_cloud
from toaster.io.apairo_dataset import (
    detect_apairo_channel,
    detect_apairo_nav,
    frame_path,
    frame_timestamp,
)

# -- survivor mask / realignment (pure core) --------------------------------


def test_to_source_frame_scatters_with_fill():
    pc = PointCloud(
        xyz=np.zeros((3, 3), np.float32), source_index=np.array([0, 2, 4]), source_count=5
    )
    assert pc.to_source_frame(np.array([7, 8, 9]), fill=-1).tolist() == [7, -1, 8, -1, 9]


def test_to_source_frame_is_noop_for_a_full_cloud():
    pc = PointCloud(xyz=np.zeros((3, 3), np.float32))  # nothing dropped
    assert pc.to_source_frame(np.array([1, 2, 3]), fill=0).tolist() == [1, 2, 3]


def test_npy_loader_records_nan_survivor_mask(tmp_path):
    pts = np.array([[0, 0, 0, 0.1], [np.nan, np.nan, np.nan, 0.2], [1, 2, 3, 0.3]], np.float32)
    path = tmp_path / "scan.npy"
    np.save(path, pts)
    cloud = load_cloud(path)
    assert cloud.n == 2 and cloud.source_count == 3
    assert cloud.source_index.tolist() == [0, 2]
    assert cloud.to_source_frame(np.array([5, 6]), fill=0).tolist() == [5, 0, 6]


# -- detection / timestamps -------------------------------------------------


def _apairo_seq(tmp_path, loader="npys"):
    seq = tmp_path / "seq"
    (seq / ".apairo").mkdir(parents=True)
    (seq / ".apairo" / "channels.yaml").write_text(
        f"channels:\n  ouster_points:\n    loader: {loader}\n    kind: raw\nversion: 1\n"
    )
    (seq / "ouster_points").mkdir()
    return seq


def test_detect_apairo_channel(tmp_path):
    seq = _apairo_seq(tmp_path)
    frame = seq / "ouster_points" / "001813.npy"
    np.save(frame, np.zeros((4, 4), np.float32))
    target = detect_apairo_channel(frame)
    assert target is not None
    assert target.source_channel == "ouster_points"
    assert target.stem == "001813"
    assert target.seq_dir == str(seq.resolve())


def test_detect_returns_none_outside_apairo(tmp_path):
    frame = tmp_path / "loose" / "x.npy"
    frame.parent.mkdir()
    np.save(frame, np.zeros((4, 4), np.float32))
    assert detect_apairo_channel(frame) is None


def test_detect_rejects_stacked_loader(tmp_path):
    # A 'npy' (stacked, whole-sequence) channel can't take per-frame appends.
    seq = _apairo_seq(tmp_path, loader="npy")
    frame = seq / "ouster_points" / "0.npy"
    np.save(frame, np.zeros((4, 4), np.float32))
    assert detect_apairo_channel(frame) is None


def test_frame_timestamp_uses_sorted_position(tmp_path):
    seq = _apairo_seq(tmp_path)
    ch = seq / "ouster_points"
    for i in range(3):
        np.save(ch / f"00000{i}.npy", np.zeros((2, 4), np.float32))
    (ch / "timestamps.txt").write_text("1.0\n2.0\n3.0\n")
    assert frame_timestamp(seq, "ouster_points", "000001") == 2.0
    assert frame_timestamp(seq, "ouster_points", "absent") is None


# -- save_apairo wiring (fake apairo module) --------------------------------


class _FakeWriter:
    last: _FakeWriter | None = None

    def __init__(self, seq_dir, channel, **kw):
        self.seq_dir, self.channel, self.kw, self.added = seq_dir, channel, kw, []
        _FakeWriter.last = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def add(self, arr, stem, timestamp):
        self.added.append((np.asarray(arr).copy(), stem, timestamp))


@pytest.fixture
def fake_apairo(monkeypatch):
    mod = types.ModuleType("apairo")
    mod.ChannelWriter = _FakeWriter
    monkeypatch.setitem(sys.modules, "apairo", mod)
    _FakeWriter.last = None
    return _FakeWriter


def test_save_apairo_writes_full_frame_aligned_labels(tmp_path, fake_apairo):
    from toaster.api.service import AnnotationService

    seq = _apairo_seq(tmp_path)
    ch = seq / "ouster_points"
    np.save(ch / "000000.npy", np.zeros((3, 4), np.float32))
    # Frame 000001 has 6 points, one NaN at original index 2 (-> dropped on load).
    frame = np.array(
        [[0, 0, 0, 0.1], [1, 0, 0, 0.1], [np.nan, np.nan, np.nan, 0.1],
         [2, 0, 0, 0.1], [3, 0, 0, 0.1], [4, 0, 0, 0.1]], np.float32,
    )  # fmt: skip
    np.save(ch / "000001.npy", frame)
    (ch / "timestamps.txt").write_text("10.0\n20.0\n")

    svc = AnnotationService()
    svc.open_cloud(str(ch / "000001.npy"))
    unlabeled = svc._session.schema.unlabeled_id
    svc.box([0, 1])  # the first two kept points (original rows 0 and 1)
    svc.assign(class_id=7)

    out = svc.save_apairo("ground_truth")

    w = fake_apairo.last
    assert w.channel == "ground_truth"
    assert w.seq_dir == str(seq.resolve())
    assert w.kw == {
        "loader": "npys", "timestamps_from": "ouster_points", "sources": ["ouster_points"]
    }  # fmt: skip
    arr, stem, ts = w.added[0]
    assert stem == "000001" and ts == 20.0  # timestamp from ouster_points line 2
    assert arr.shape == (6,)  # realigned to the full on-disk frame
    assert arr[0] == 7 and arr[1] == 7  # labelled points
    assert arr[2] == unlabeled  # the NaN-dropped point fills with 'unlabeled'
    assert out["points"] == 6 and out["channel"] == "ground_truth"


def test_apairo_info_reports_detection(tmp_path):
    from toaster.api.service import AnnotationService

    seq = _apairo_seq(tmp_path)
    np.save(seq / "ouster_points" / "000000.npy", np.zeros((3, 4), np.float32))
    svc = AnnotationService()
    svc.open_cloud(str(seq / "ouster_points" / "000000.npy"))
    info = svc.apairo_info()
    assert info["is_apairo"] is True
    assert info["source_channel"] == "ouster_points"
    assert info["suggested_channel"] == "ground_truth"


def test_save_apairo_real_roundtrip_incremental(tmp_path):
    # Against the real apairo.ChannelWriter (skipped when the extra is absent):
    # two frames written to the same channel across separate opens/saves.
    pytest.importorskip("apairo")
    import yaml

    from toaster.api.service import AnnotationService

    seq = _apairo_seq(tmp_path)
    ch = seq / "ouster_points"
    np.save(ch / "000000.npy", np.zeros((3, 4), np.float32))
    np.save(ch / "000001.npy", np.zeros((4, 4), np.float32))
    (ch / "timestamps.txt").write_text("10.0\n20.0\n")

    svc = AnnotationService()
    svc.open_cloud(str(ch / "000000.npy"))
    svc.box([0])
    svc.assign(class_id=5)
    svc.save_apairo("ground_truth")

    svc.open_cloud(str(ch / "000001.npy"))  # resumes the existing channel
    svc.box([1, 2])
    svc.assign(class_id=6)
    svc.save_apairo("ground_truth")

    gt = seq / "ground_truth"
    assert np.load(gt / "000000.npy").tolist() == [5, 0, 0]
    assert np.load(gt / "000001.npy").tolist() == [0, 6, 6, 0]
    chans = yaml.safe_load((seq / ".apairo" / "channels.yaml").read_text())["channels"]
    assert chans["ground_truth"]["kind"] == "preprocess"
    assert chans["ground_truth"]["timestamps_from"] == "ouster_points"


# -- dataset navigation (sequences / channels / frames) ---------------------


def _dataset(tmp_path):
    root = tmp_path / "ds"
    (root / ".apairo").mkdir(parents=True)
    (root / ".apairo" / "dataset.yaml").write_text(
        "name: testset\nsequences:\n- seqA\n- seqB\nversion: 1\n"
    )
    for seq, nframes in [("seqA", 3), ("seqB", 2)]:
        (root / seq / ".apairo").mkdir(parents=True)
        (root / seq / ".apairo" / "channels.yaml").write_text(
            "channels:\n  ouster_points:\n    loader: npys\n    alias: lidar\n"
            "  zed_img:\n    loader: npys\nversion: 1\n"
        )
        op = root / seq / "ouster_points"
        op.mkdir()
        (op / "metadata.yaml").write_text("msgtype: sensor_msgs/msg/PointCloud2\n")
        for i in range(nframes):
            np.save(op / f"00000{i}.npy", np.zeros((4, 4), np.float32))
        (op / "timestamps.txt").write_text("".join(f"{10.0 + i}\n" for i in range(nframes)))
        zed = root / seq / "zed_img"  # same loader (npys) but an image -> excluded
        zed.mkdir()
        (zed / "metadata.yaml").write_text("msgtype: sensor_msgs/msg/Image\n")
        np.save(zed / "000000.npy", np.zeros((2, 2, 3), np.float32))
    return root


def test_detect_apairo_nav(tmp_path):
    root = _dataset(tmp_path)
    nav = detect_apairo_nav(root / "seqA" / "ouster_points" / "000001.npy")
    assert nav is not None
    assert nav.dataset_name == "testset"
    assert nav.sequences == ["seqA", "seqB"]
    assert nav.sequence == "seqA"
    assert nav.channels == ["ouster_points"]  # zed_img (Image msgtype) excluded
    assert nav.channel == "ouster_points"
    assert nav.frame_index == 1 and len(nav.frames) == 3


def test_frame_path_clamps_to_range(tmp_path):
    root = _dataset(tmp_path)
    assert frame_path(root, "seqB", "ouster_points", 0).name == "000000.npy"
    assert frame_path(root, "seqB", "ouster_points", 99).name == "000001.npy"  # clamped


def test_apairo_open_navigates_across_sequences(tmp_path):
    from toaster.api.service import AnnotationService

    root = _dataset(tmp_path)
    svc = AnnotationService()
    svc.open_cloud(str(root / "seqA" / "ouster_points" / "000000.npy"))
    nav = svc.apairo_nav()
    assert nav["is_dataset"] and nav["sequence"] == "seqA" and nav["frame_count"] == 3

    svc.apairo_open("seqB", "ouster_points", 1)  # jump to another sequence
    nav2 = svc.apairo_nav()
    assert nav2["sequence"] == "seqB" and nav2["frame_index"] == 1 and nav2["frame_count"] == 2


def test_apairo_nav_false_outside_dataset(tmp_path):
    from toaster.api.service import AnnotationService

    path = tmp_path / "loose.npy"
    np.save(path, np.zeros((4, 4), np.float32))
    svc = AnnotationService()
    svc.open_cloud(str(path))
    assert svc.apairo_nav()["is_dataset"] is False


def test_apairo_info_false_for_loose_cloud(tmp_path):
    from toaster.api.service import AnnotationService

    path = tmp_path / "loose.npy"
    np.save(path, np.zeros((4, 4), np.float32))
    svc = AnnotationService()
    svc.open_cloud(str(path))
    assert svc.apairo_info()["is_apairo"] is False
