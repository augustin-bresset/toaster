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
    allhidden = client.post("/api/groups/hide_all").json()
    assert all(s["visible"] is False for s in allhidden["snapshot"]["segments"])
    shown = client.post("/api/groups/show_all").json()
    assert all(s["visible"] for s in shown["snapshot"]["segments"])


def test_assign_visible_groups_endpoint(client):
    state = client.post("/api/segment", json={"name": "dbscan", "params": {"eps": 0.5}}).json()
    segs = state["snapshot"]["segments"]
    assert len(segs) == 2
    hide = segs[0]
    # Uncheck one segment, then assign class 3 to the visible (checked) ones.
    client.post("/api/group/visibility", json={"group_id": hide["id"], "visible": False})
    after = client.post("/api/groups/assign_visible", json={"class_id": 3}).json()
    labels = decode_array(after["labels"])
    # Exactly the visible segment's points became class 3; the hidden one did not.
    assert int((labels == 3).sum()) == sum(s["count"] for s in segs if s["id"] != hide["id"])


def test_clear_grouping_endpoint(client):
    state = client.post("/api/segment", json={"name": "dbscan", "params": {"eps": 0.5}}).json()
    assert state["grouping"] is not None
    cleared = client.post("/api/grouping/clear").json()
    assert cleared["grouping"] is None
    assert cleared["snapshot"]["active_grouping"] is None
    assert cleared["snapshot"]["display_mode"] == "labels"


def test_class_editing_endpoints(client):
    s = client.post("/api/class/add", json={"name": "tree", "color": "#0a0b0c"}).json()
    new = max(c["id"] for c in s["snapshot"]["classes"])
    assert s["snapshot"]["active_class"] == new

    s = client.post("/api/class/rename", json={"class_id": new, "name": "shrub"}).json()
    assert next(c for c in s["snapshot"]["classes"] if c["id"] == new)["name"] == "shrub"

    s = client.post("/api/class/color", json={"class_id": new, "color": [1, 2, 3]}).json()
    assert next(c for c in s["snapshot"]["classes"] if c["id"] == new)["color"] == [1, 2, 3]

    s = client.post("/api/class/remove", json={"class_id": new}).json()
    assert new not in [c["id"] for c in s["snapshot"]["classes"]]


def test_bad_segment_params_return_400(client):
    r = client.post("/api/segment", json={"name": "dbscan", "params": {"min_samples": -22}})
    assert r.status_code == 400  # invalid params -> clean client error, not a 500


def test_pick_assign_undo(client):
    client.post("/api/active_class", json={"class_id": 2})
    client.post("/api/pick", json={"index": 5})
    after = client.post("/api/assign", json={}).json()
    assert decode_array(after["labels"])[5] == 2
    back = client.post("/api/undo").json()
    assert decode_array(back["labels"])[5] == 0


def test_browse_lists_dir_and_flags_openable(client, tmp_path):
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / "sub").mkdir()
    data = client.get("/api/browse", params={"path": str(tmp_path)}).json()
    by = {e["name"]: e for e in data["entries"]}
    assert by["scan.bin"]["openable"] is True  # the cloud .bin from the fixture
    assert by["notes.txt"]["openable"] is False  # unsupported format
    assert by["sub"]["is_dir"] and by["sub"]["openable"]
    assert ".bin" in data["extensions"]


def test_browse_defaults_to_open_cloud_folder(client):
    data = client.get("/api/browse").json()  # no path -> the open cloud's folder
    assert any(e["name"] == "scan.bin" for e in data["entries"])


def test_browse_bad_dir_is_400(client):
    assert client.get("/api/browse", params={"path": "/no/such/dir/zzz"}).status_code == 400
