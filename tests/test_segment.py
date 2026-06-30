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


def _tilt(xyz, deg):
    """Rotate a scene about X so gravity no longer points along +Z; return (xyz, up)."""
    a = np.radians(deg)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    up = rot @ np.array([0.0, 0.0, 1.0])
    return (xyz @ rot.T).astype(np.float32), up.tolist()


@pytest.mark.parametrize("name", ["ground_grid", "csf"])
def test_z_based_ground_filters_honour_up_on_tilted_scene(name, ground_scene):
    # ground_grid and CSF both key off Z; on a tipped scene the given up vector
    # lets them recover the ground a naive +Z assumption would miss.
    tilted, up = _tilt(ground_scene.xyz, 40.0)
    cloud = PointCloud(tilted)
    aware = get_segmenter(name, up=up).segment(cloud)
    assert (aware.group_id[:300] == 0).mean() > 0.9  # ground recovered
    assert (aware.group_id[300:] == 1).mean() > 0.7  # obstacle kept separate


def test_ground_grid_without_up_misreads_tilted_scene(ground_scene):
    tilted, _ = _tilt(ground_scene.xyz, 40.0)
    naive = get_segmenter("ground_grid").segment(PointCloud(tilted))  # assumes cloud +Z
    assert (naive.group_id[:300] == 0).mean() < 0.9  # the slope confuses Z-binning


def test_ransac_with_up_locks_onto_ground_not_largest_plane():
    # A small horizontal ground and a *bigger* vertical wall. Plain RANSAC takes
    # the wall (more inliers); with an up hint it must keep the horizontal ground.
    rng = np.random.default_rng(0)
    ground = np.c_[rng.uniform(-5, 5, 300), rng.uniform(-5, 5, 300), np.zeros(300)]
    wall = np.c_[np.full(700, 4.0), rng.uniform(-5, 5, 700), rng.uniform(0, 5, 700)]
    cloud = PointCloud(np.vstack([ground, wall]).astype(np.float32))
    is_ground = np.r_[np.ones(300, bool), np.zeros(700, bool)]

    res = get_segmenter("ransac_ground", threshold=0.1, iterations=400, up=[0, 0, 1]).segment(cloud)
    pred = res.group_id == 0
    assert pred[is_ground].mean() > 0.9  # the horizontal ground is found
    assert pred[~is_ground].mean() < 0.1  # the bigger vertical wall is not "ground"


def test_segmenter_specs_flag_gravity_for_ground_filters():
    gravity = {s["name"]: s["gravity"] for s in segmenter_specs()}
    assert gravity["ransac_ground"] and gravity["ground_grid"] and gravity["csf"]
    assert not gravity["dbscan"] and not gravity["kmeans"]


def test_bad_up_vector_is_rejected():
    with pytest.raises(ValueError):
        get_segmenter("ground_grid", up=[0, 0, 0])  # zero vector has no direction
    with pytest.raises(ValueError):
        get_segmenter("ransac_ground", up=[1, 2])  # not a 3-vector


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
