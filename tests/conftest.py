"""Shared fixtures: synthetic clouds and a small schema. No GUI involved."""

from __future__ import annotations

import numpy as np
import pytest

from toaster.core import LabelClass, LabelSchema, PointCloud


@pytest.fixture(autouse=True)
def isolate_toaster_config(tmp_path_factory, monkeypatch):
    """Point ``$HOME`` at a throwaway dir so no test reads or writes the user's
    real ``~/.config/toaster/session.json``. ``AnnotationService`` persists the
    session (recent files, session channel) there, so without this a stale local
    session would leak into tests (and tests would clobber the developer's own).
    """
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: home)


@pytest.fixture
def schema() -> LabelSchema:
    return LabelSchema(
        classes=[
            LabelClass(0, "unlabeled", (0, 0, 0)),
            LabelClass(1, "a", (255, 0, 0)),
            LabelClass(2, "b", (0, 255, 0)),
        ],
        unlabeled_id=0,
    )


@pytest.fixture
def two_clusters() -> PointCloud:
    """Two tight, well-separated blobs of 50 points each (no noise)."""
    rng = np.random.default_rng(0)
    a = rng.normal((0.0, 0.0, 0.0), 0.08, (50, 3))
    b = rng.normal((10.0, 0.0, 0.0), 0.08, (50, 3))
    xyz = np.vstack([a, b]).astype(np.float32)
    return PointCloud(xyz=xyz, features={"intensity": rng.random(100).astype(np.float32)})
