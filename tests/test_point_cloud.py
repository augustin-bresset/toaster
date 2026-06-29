from __future__ import annotations

import numpy as np
import pytest

from toaster.core import PointCloud


def test_n_and_dtype():
    cloud = PointCloud(xyz=np.zeros((5, 3), dtype=np.float64))
    assert cloud.n == 5
    assert cloud.xyz.dtype == np.float32


def test_ensure_labels_allocates_with_fill():
    cloud = PointCloud(xyz=np.zeros((3, 3), np.float32))
    labels = cloud.ensure_labels(unlabeled_id=7)
    assert labels.tolist() == [7, 7, 7]
    # Returns the same array on subsequent calls (mutable in place).
    assert cloud.ensure_labels() is labels


def test_bad_xyz_shape_raises():
    with pytest.raises(ValueError):
        PointCloud(xyz=np.zeros((3, 2), np.float32))


def test_label_shape_validation():
    with pytest.raises(ValueError):
        PointCloud(xyz=np.zeros((3, 3), np.float32), labels=np.zeros(2, np.int32))
