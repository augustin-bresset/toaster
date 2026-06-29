"""Wrap arbitrary callables / models as segmenters.

Two adapters cover the "plug in anything" promise:

- :class:`FunctionSegmenter` — any ``points -> group_ids`` callable becomes a
  segmenter (custom clustering, region growing, a quick numpy heuristic).
- :class:`ModelSegmenter` — a supervised ``points -> class_ids`` predictor (ONNX,
  torch, anything). Because the model knows what each group *means*, the
  resulting grouping carries ``suggested_labels`` so one click accepts the
  predicted class.

Both can feed the model more than geometry via ``feature_keys`` (e.g.
``["intensity"]`` to pass ``[x, y, z, intensity]``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from toaster.core import Grouping, PointCloud, Selection

from .base import gather_inputs, resolve_points, scatter

__all__ = ["FunctionSegmenter", "ModelSegmenter"]


class FunctionSegmenter:
    """Adapt a bare ``callable(points) -> (M,) group ids`` into a segmenter.

    Args:
        fn: Maps the input array to ``(M,)`` integer group ids (``-1`` = noise).
        name: Registry / provenance name.
        feature_keys: Feature channels appended after xyz to form the input
            (default: xyz only).
    """

    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        name: str = "function",
        feature_keys: Sequence[str] = (),
    ) -> None:
        self._fn = fn
        self.name = name
        self.feature_keys = tuple(feature_keys)

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        _, indices = resolve_points(cloud, selection)
        inputs = gather_inputs(cloud, indices, self.feature_keys)
        group_ids = np.asarray(self._fn(inputs), dtype=np.int32)
        return scatter(group_ids, indices, cloud.n, source=self.name)


class ModelSegmenter:
    """Adapt a supervised ``callable(points) -> (M,) class ids`` predictor.

    The predicted class becomes the group id, and ``suggested_labels`` maps each
    group to that same class — so clicking a predicted region can accept the
    model's class directly.

    Args:
        predict: Maps the input array to ``(M,)`` class ids.
        name: Registry / provenance name.
        feature_keys: Feature channels appended after xyz to form the model input
            (e.g. ``["intensity"]`` for an ``[x, y, z, intensity]`` model).
        ignore_id: Class id treated as "no prediction" (mapped to ``-1`` noise).

    Example:
        >>> seg = ModelSegmenter(my_net.predict, name="my_net",
        ...                      feature_keys=["intensity"])   # doctest: +SKIP
        >>> grouping = seg.segment(cloud)                      # doctest: +SKIP
    """

    def __init__(
        self,
        predict: Callable[[np.ndarray], np.ndarray],
        name: str = "model",
        feature_keys: Sequence[str] = (),
        ignore_id: int | None = None,
    ) -> None:
        self._predict = predict
        self.name = name
        self.feature_keys = tuple(feature_keys)
        self.ignore_id = ignore_id

    def segment(self, cloud: PointCloud, selection: Selection | None = None) -> Grouping:
        _, indices = resolve_points(cloud, selection)
        inputs = gather_inputs(cloud, indices, self.feature_keys)
        classes = np.asarray(self._predict(inputs), dtype=np.int32)
        if self.ignore_id is not None:
            classes = np.where(classes == self.ignore_id, -1, classes)
        present = np.unique(classes[classes >= 0])
        suggested = {int(c): int(c) for c in present}
        return scatter(classes, indices, cloud.n, source=self.name, suggested_labels=suggested)
