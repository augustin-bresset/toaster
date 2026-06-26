from __future__ import annotations

import numpy as np

from toaster.core import Selection
from toaster.segment import FunctionSegmenter, ModelSegmenter, available_segmenters, get_segmenter


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


def test_registry_lists_builtins():
    names = available_segmenters()
    assert "dbscan" in names and "hdbscan" in names


def test_function_segmenter(two_clusters):
    seg = FunctionSegmenter(lambda xyz: (xyz[:, 0] > 5).astype(int), name="split_x")
    grouping = seg.segment(two_clusters)
    assert grouping.n_groups == 2
    assert grouping.source == "split_x"


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
