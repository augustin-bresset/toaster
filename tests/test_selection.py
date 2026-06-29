from __future__ import annotations

import numpy as np
import pytest

from toaster.core import Grouping, Selection


def test_constructors_and_indices():
    sel = Selection.from_indices(np.array([1, 3]), n=5)
    assert sel.indices.tolist() == [1, 3]
    assert sel.count == 2
    assert Selection.empty(5).is_empty()
    assert Selection.from_pick(2, 5).indices.tolist() == [2]


def test_boolean_algebra():
    a = Selection.from_indices([0, 1, 2], 5)
    b = Selection.from_indices([1, 2, 3], 5)
    assert (a | b).indices.tolist() == [0, 1, 2, 3]
    assert (a & b).indices.tolist() == [1, 2]
    assert (a - b).indices.tolist() == [0]
    assert (a ^ b).indices.tolist() == [0, 3]


def test_from_group():
    grouping = Grouping(np.array([0, 0, 1, -1, 1], np.int32))
    sel = Selection.from_group(grouping, 1)
    assert sel.indices.tolist() == [2, 4]


def test_size_mismatch_raises():
    with pytest.raises(ValueError):
        Selection.empty(4) | Selection.empty(5)
