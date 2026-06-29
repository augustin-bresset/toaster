from __future__ import annotations

import numpy as np

from toaster.core import Grouping


def test_groups_exclude_noise():
    g = Grouping(np.array([0, 0, 1, 1, 1, -1], np.int32))
    assert g.n_groups == 2
    assert g.group_ids().tolist() == [0, 1]
    assert g.n == 6


def test_indices_of_and_group_of():
    g = Grouping(np.array([2, 0, 2, -1, 0], np.int32))
    assert g.indices_of(2).tolist() == [0, 2]
    assert g.indices_of(0).tolist() == [1, 4]
    assert g.group_of(3) == -1
    # Unknown group id -> empty.
    assert g.indices_of(99).tolist() == []


def test_metadata_defaults():
    g = Grouping(np.array([0, 1], np.int32), source="dbscan", params={"eps": 0.5})
    assert g.source == "dbscan"
    assert g.params["eps"] == 0.5
    assert g.suggested_labels is None
