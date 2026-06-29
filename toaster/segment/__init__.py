"""Segmenters — pluggable "group the points" models, behind a tiny registry.

Built-ins (``dbscan``, ``hdbscan``) are registered on import. Register your own
with :func:`register_segmenter` (see :class:`~toaster.segment.base.Segmenter`)::

    @register_segmenter
    class MySegmenter:
        name = "mine"
        def segment(self, cloud, selection=None):
            ...

Then construct it by name: ``get_segmenter("mine", **params)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict

import numpy as np

from .base import Param, Segmenter, gather_inputs, params_of, resolve_points, scatter
from .cluster import (
    AgglomerativeSegmenter,
    KMeansSegmenter,
    KMedoidsSegmenter,
    MeanShiftSegmenter,
    OPTICSSegmenter,
)
from .dbscan import DBSCANSegmenter
from .ground import CSFGroundSegmenter, GroundGridSegmenter, RANSACGroundSegmenter
from .hdbscan import HDBSCANSegmenter
from .model import FunctionSegmenter, ModelSegmenter

__all__ = [
    "Segmenter",
    "Param",
    "resolve_points",
    "gather_inputs",
    "scatter",
    "DBSCANSegmenter",
    "HDBSCANSegmenter",
    "KMeansSegmenter",
    "KMedoidsSegmenter",
    "AgglomerativeSegmenter",
    "OPTICSSegmenter",
    "MeanShiftSegmenter",
    "RANSACGroundSegmenter",
    "GroundGridSegmenter",
    "CSFGroundSegmenter",
    "FunctionSegmenter",
    "ModelSegmenter",
    "SEGMENTERS",
    "register_segmenter",
    "register_model",
    "segmenter_specs",
    "get_segmenter",
    "available_segmenters",
]

#: Registry name -> segmenter class (constructed by :func:`get_segmenter`).
SEGMENTERS: dict[str, type] = {}


def register_segmenter(cls: type) -> type:
    """Class decorator that registers a segmenter under its ``name`` attribute."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name` to register")
    SEGMENTERS[name] = cls
    return cls


def register_model(
    name: str,
    predict: Callable[[np.ndarray], np.ndarray],
    *,
    feature_keys: Sequence[str] = (),
    ignore_id: int | None = None,
) -> type:
    """Register a labelling model as a named segmenter — the one-call way in.

    Given a model whose ``predict`` maps a point array to per-point class ids,
    this exposes it under ``name`` so it shows up in the app's segmenter panel
    and via :func:`get_segmenter`. Predicted classes become groups *and*
    ``suggested_labels`` (one click accepts the prediction).

    Args:
        name: Stable name shown in the UI / used by ``get_segmenter``.
        predict: ``(M, 3+F) -> (M,)`` class ids.
        feature_keys: Feature channels appended after xyz (e.g. ``["intensity"]``).
        ignore_id: Class id meaning "no prediction" (mapped to noise ``-1``).

    Example:
        >>> register_model("my_net", my_net.predict,
        ...                 feature_keys=["intensity"])   # doctest: +SKIP
    """
    seg = ModelSegmenter(predict, name=name, feature_keys=feature_keys, ignore_id=ignore_id)
    plugin = type(
        f"Model_{name}",
        (),
        {
            "name": name,
            "segment": lambda self, cloud, selection=None: seg.segment(cloud, selection),
        },
    )
    return register_segmenter(plugin)


def get_segmenter(name: str, **params) -> Segmenter:
    """Instantiate a registered segmenter by name with the given parameters."""
    if name not in SEGMENTERS:
        raise KeyError(f"unknown segmenter {name!r} (have: {', '.join(available_segmenters())})")
    return SEGMENTERS[name](**params)


def available_segmenters() -> list[str]:
    """Sorted names of registered segmenters."""
    return sorted(SEGMENTERS)


def segmenter_specs() -> list[dict]:
    """Each registered segmenter as ``{name, params: [...]}`` for a front-end."""
    return [
        {"name": name, "params": [asdict(p) for p in params_of(SEGMENTERS[name])]}
        for name in available_segmenters()
    ]


for _seg in (
    DBSCANSegmenter,
    HDBSCANSegmenter,
    KMeansSegmenter,
    KMedoidsSegmenter,
    AgglomerativeSegmenter,
    OPTICSSegmenter,
    MeanShiftSegmenter,
    RANSACGroundSegmenter,
    GroundGridSegmenter,
    CSFGroundSegmenter,
):
    register_segmenter(_seg)
