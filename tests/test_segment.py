from __future__ import annotations

import numpy as np
import pytest

from toaster.core import PointCloud, Selection
from toaster.segment import (
    FunctionSegmenter,
    ModelSegmenter,
    available_segmenters,
    get_segmenter,
    segmenter_specs,
)


def test_dbscan_finds_two_clusters(two_clusters):
    grouping = get_segmenter("dbscan", eps=0.5, min_samples=5).segment(two_clusters)
    assert grouping.n_groups == 2
    assert grouping.n == two_clusters.n
    assert grouping.source == "dbscan"
    assert grouping.params["eps"] == 0.5


def test_segmenter_scoped_to_selection_marks_rest_noise(two_clusters):
    # Restrict to the first blob only; the rest must be noise (-1).
    sel = Selection.from_indices(np.arange(50), two_clusters.n)
    grouping = get_segmenter("dbscan", eps=0.5, min_samples=5).segment(two_clusters, sel)
    assert np.all(grouping.group_id[50:] == -1)
    assert grouping.n_groups == 1


def test_clusterers_on_one_point_return_all_noise(two_clusters):
    # A one-point selection must not crash the clusterer (sklearn raises on
    # n_samples=1); it yields an empty grouping instead.
    one = Selection.from_indices(np.array([0]), two_clusters.n)
    for name in ("dbscan", "hdbscan"):
        grouping = get_segmenter(name).segment(two_clusters, one)
        assert grouping.n_groups == 0
        assert np.all(grouping.group_id == -1)


def test_registry_lists_builtins():
    names = available_segmenters()
    assert "dbscan" in names and "hdbscan" in names


def test_function_segmenter(two_clusters):
    seg = FunctionSegmenter(lambda xyz: (xyz[:, 0] > 5).astype(int), name="split_x")
    grouping = seg.segment(two_clusters)
    assert grouping.n_groups == 2
    assert grouping.source == "split_x"


def test_registry_has_the_new_algorithms():
    names = available_segmenters()
    for n in ["kmeans", "kmedoids", "agglomerative", "optics", "meanshift",
              "ransac_ground", "ground_grid"]:  # fmt: skip
        assert n in names


def test_segmenter_specs_carry_params():
    specs = {s["name"]: s["params"] for s in segmenter_specs()}
    assert {p["name"] for p in specs["dbscan"]} == {"eps", "min_samples"}
    assert specs["kmeans"][0]["name"] == "n_clusters"
    assert specs["ground_grid"][0]["type"] == "float"


@pytest.mark.parametrize("name", ["kmeans", "kmedoids", "agglomerative"])
def test_partitioning_clusterers_make_k_groups(name, two_clusters):
    grouping = get_segmenter(name, n_clusters=2).segment(two_clusters)
    assert grouping.n_groups == 2


@pytest.fixture
def ground_scene():
    rng = np.random.default_rng(1)
    ground = rng.uniform([-5, -5, -0.02], [5, 5, 0.02], (300, 3))
    obstacle = rng.uniform([0, 0, 1.0], [1, 1, 2.0], (60, 3))
    return PointCloud(np.vstack([ground, obstacle]).astype(np.float32))


@pytest.mark.parametrize("name", ["ground_grid", "ransac_ground"])
def test_ground_detection_splits_and_suggests(name, ground_scene):
    grouping = get_segmenter(name).segment(ground_scene)
    # Group 0 = ground, group 1 = non-ground, suggested -> traversable / obstacle.
    assert grouping.suggested_labels == {0: 1, 1: 2}
    assert (grouping.group_id[:300] == 0).all()  # the flat plane is ground
    assert (grouping.group_id[300:] == 1).all()  # the raised box is non-ground


def test_model_segmenter_attaches_suggested_labels(two_clusters):
    seg = ModelSegmenter(lambda xyz: np.where(xyz[:, 0] > 5, 2, 1), name="fake_nn")
    grouping = seg.segment(two_clusters)
    assert grouping.suggested_labels == {1: 1, 2: 2}


def test_model_segmenter_passes_features(two_clusters):
    seen = {}

    def predict(points):
        seen["shape"] = points.shape
        return np.where(points[:, 0] > 5, 2, 1)

    # With intensity requested, the model receives [x, y, z, intensity].
    ModelSegmenter(predict, name="nn", feature_keys=["intensity"]).segment(two_clusters)
    assert seen["shape"] == (two_clusters.n, 4)


def test_register_model_appears_in_app_and_runs(two_clusters):
    from toaster.segment import register_model

    register_model("toy_net", lambda p: np.where(p[:, 0] > 5, 2, 1), feature_keys=["intensity"])
    assert "toy_net" in available_segmenters()
    # Constructible with no params (how the app's panel instantiates it).
    grouping = get_segmenter("toy_net").segment(two_clusters)
    assert grouping.n_groups == 2
    assert grouping.suggested_labels == {1: 1, 2: 2}
