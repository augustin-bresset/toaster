from __future__ import annotations

import numpy as np

from toaster.core import LabelClass, LabelSchema
from toaster.persistence import LabelStore, SchemaStore, SessionState, SessionStore


def test_schema_store_roundtrip(tmp_path):
    store = SchemaStore()
    source = tmp_path / "scan.ply"
    schema = LabelSchema(
        classes=[LabelClass(0, "unlabeled", (0, 0, 0)), LabelClass(1, "tree", (1, 2, 3))],
        unlabeled_id=0,
    )
    out = store.save(source, schema)
    assert out == store.path_for(source)
    assert out.name == "scan.ply.toaster.schema.yaml"
    loaded = store.load(source)
    assert loaded is not None
    assert loaded.get(1).name == "tree"
    assert loaded.get(1).color == (1, 2, 3)


def test_schema_store_missing_returns_none(tmp_path):
    assert SchemaStore().load(tmp_path / "absent.ply") is None


def test_label_store_roundtrip(tmp_path):
    store = LabelStore()
    source = tmp_path / "scan.ply"
    labels = np.array([0, 1, 2, 1], dtype=np.int32)
    out = store.save(source, labels)
    assert out == store.path_for(source)
    assert out.name == "scan.ply.toaster.npy"
    loaded = store.load(source)
    assert np.array_equal(loaded, labels)


def test_label_store_missing_returns_none(tmp_path):
    assert LabelStore().load(tmp_path / "absent.ply") is None


def test_session_store_roundtrip(tmp_path):
    store = SessionStore(tmp_path / "session.json")
    state = SessionState(cloud_path="a.ply", active_class=3)
    state.remember("a.ply")
    state.remember("b.ply")
    store.save(state)
    loaded = store.load()
    assert loaded.active_class == 3
    assert loaded.recent_files[:2] == ["b.ply", "a.ply"]


def test_session_store_missing_returns_defaults(tmp_path):
    state = SessionStore(tmp_path / "none.json").load()
    assert state.active_class == 0
    assert state.recent_files == []
