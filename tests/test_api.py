"""Integration tests for the REST API, via FastAPI's TestClient (httpx)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from toaster.api import create_app, decode_array  # noqa: E402


@pytest.fixture
def client(tmp_path):
    # A two-cluster scan written as KITTI-style .bin.
    rng = np.random.default_rng(0)
    a = rng.normal((0, 0, 0), 0.08, (50, 3))
    b = rng.normal((10, 0, 0), 0.08, (50, 3))
    pts = np.hstack([np.vstack([a, b]), rng.random((100, 1))]).astype(np.float32)
    path = tmp_path / "scan.bin"
    pts.tofile(path)

    app = create_app()
    c = TestClient(app)
    c.post("/api/open", json={"path": str(path)})
    return c


def test_state_before_open_is_conflict():
    client = TestClient(create_app())
    assert client.get("/api/state").status_code == 409


def test_cloud_roundtrips_geometry(client):
    payload = client.get("/api/cloud").json()
    xyz = decode_array(payload["xyz"])
    assert xyz.shape == (100, 3)
    assert xyz.dtype == np.float32
    assert "intensity" in payload["features"]


def test_state_shape(client):
    state = client.get("/api/state").json()
    assert decode_array(state["labels"]).shape == (100,)
    assert state["grouping"] is None  # no segmenter run yet
    assert state["snapshot"]["classes"][0]["name"] == "unlabeled"


def test_segment_then_assign_group(client):
    state = client.post(
        "/api/segment",
        json={"name": "dbscan", "params": {"eps": 0.5, "min_samples": 5}},
    ).json()
    snap = state["snapshot"]
    assert len(snap["segments"]) == 2
    assert state["grouping"] is not None

    gid = snap["segments"][0]["id"]
    # Select the segment -> its points are selected.
    sel = client.post("/api/group/select", json={"group_id": gid}).json()
    assert decode_array(sel["selection"]).size == snap["segments"][0]["count"]

    # Label that segment with class 4.
    after = client.post("/api/group/assign", json={"group_id": gid, "class_id": 4}).json()
    labels = decode_array(after["labels"])
    assert int((labels == 4).sum()) == snap["segments"][0]["count"]


def test_group_visibility_in_state(client):
    state = client.post("/api/segment", json={"name": "dbscan", "params": {"eps": 0.5}}).json()
    gid = state["snapshot"]["segments"][0]["id"]
    hidden = client.post("/api/group/visibility", json={"group_id": gid, "visible": False}).json()
    seg = next(s for s in hidden["snapshot"]["segments"] if s["id"] == gid)
    assert seg["visible"] is False
    shown = client.post("/api/groups/show_all").json()
    assert all(s["visible"] for s in shown["snapshot"]["segments"])


def test_pick_assign_undo(client):
    client.post("/api/active_class", json={"class_id": 2})
    client.post("/api/pick", json={"index": 5})
    after = client.post("/api/assign", json={}).json()
    assert decode_array(after["labels"])[5] == 2
    back = client.post("/api/undo").json()
    assert decode_array(back["labels"])[5] == 0
