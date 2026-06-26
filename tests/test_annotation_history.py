from __future__ import annotations

import numpy as np

from toaster.core import AnnotationController, PointCloud, Selection


def _cloud(n=6):
    return PointCloud(xyz=np.zeros((n, 3), np.float32))


def test_assign_writes_and_returns_touched():
    cloud = _cloud()
    ann = AnnotationController(cloud)
    touched = ann.assign(Selection.from_indices([1, 2, 4], 6), class_id=3)
    assert touched.tolist() == [1, 2, 4]
    assert cloud.labels.tolist() == [0, 3, 3, 0, 3, 0]


def test_undo_is_inverse_of_assign():
    cloud = _cloud()
    before = cloud.ensure_labels().copy()
    ann = AnnotationController(cloud)
    ann.assign(Selection.from_indices([0, 5], 6), class_id=2)
    assert not np.array_equal(cloud.labels, before)
    ann.undo()
    assert np.array_equal(cloud.labels, before)


def test_redo_reapplies():
    cloud = _cloud()
    ann = AnnotationController(cloud)
    ann.assign(Selection.from_indices([3], 6), class_id=9)
    ann.undo()
    ann.redo()
    assert cloud.labels[3] == 9


def test_empty_selection_is_noop():
    cloud = _cloud()
    ann = AnnotationController(cloud)
    assert ann.assign(Selection.empty(6), class_id=1).size == 0
    assert ann.undo() is None  # nothing recorded


def test_new_edit_clears_redo():
    cloud = _cloud()
    ann = AnnotationController(cloud)
    ann.assign(Selection.from_indices([0], 6), 1)
    ann.undo()
    ann.assign(Selection.from_indices([1], 6), 2)
    assert ann.redo() is None  # redo stack was cleared by the new edit
