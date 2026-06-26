"""Shared type aliases for the Toaster core domain.

These aliases document *intent* at call sites (what an array of indices means,
what a class id is) without imposing any runtime machinery. They are plain
aliases, so they cost nothing and stay friction-free for third-party code.
"""

from __future__ import annotations

import numpy as np

#: A 1-D array of point indices into a cloud, ``shape (M,)`` dtype ``int64``.
IndexArray = np.ndarray

#: A semantic class identifier (a key into a :class:`~toaster.core.LabelSchema`).
ClassId = int

#: A group identifier produced by a segmenter. ``-1`` always means
#: "noise / unassigned / outside the segmented subset".
GroupId = int

#: Sentinel group id for points that belong to no group.
NOISE: GroupId = -1

__all__ = ["IndexArray", "ClassId", "GroupId", "NOISE"]
