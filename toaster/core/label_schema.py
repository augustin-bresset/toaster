"""The :class:`LabelSchema` — the palette of annotation classes.

The on-disk format is intentionally the same one apairo uses for its label
configs (``color_map`` + ``semantic_map`` + ``ignore_index``), so existing
GOOSE / RELLIS / SemanticKITTI configs load into Toaster unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

__all__ = ["LabelClass", "LabelSchema"]

Color = tuple[int, int, int]


@dataclass(frozen=True)
class LabelClass:
    """One annotation class.

    Args:
        id: Integer class id stored in ``labels``.
        name: Human-readable name shown in the UI.
        color: ``(r, g, b)`` in ``0..255``.
    """

    id: int
    name: str
    color: Color


@dataclass
class LabelSchema:
    """An ordered palette of :class:`LabelClass` plus the "unlabeled" id.

    Args:
        classes: The classes, in display order.
        unlabeled_id: The id meaning "not yet annotated" (apairo's
            ``ignore_index``). Used as the default fill for new label arrays.

    Example:
        >>> schema = LabelSchema(
        ...     classes=[LabelClass(0, "unlabeled", (0, 0, 0)),
        ...              LabelClass(1, "car", (255, 0, 0))],
        ... )
        >>> schema.colors_for(np.array([1, 0])).tolist()
        [[255, 0, 0], [0, 0, 0]]
    """

    classes: list[LabelClass]
    unlabeled_id: int = 0

    def __post_init__(self) -> None:
        self._by_id: dict[int, LabelClass] = {c.id: c for c in self.classes}
        self._lut: np.ndarray | None = None

    # -- lookups ----------------------------------------------------------

    def get(self, class_id: int) -> LabelClass:
        """Return the :class:`LabelClass` for ``class_id`` (raises if unknown)."""
        return self._by_id[class_id]

    def __contains__(self, class_id: int) -> bool:
        return class_id in self._by_id

    def set_color(self, class_id: int, color: Color) -> None:
        """Recolour a class in place (and invalidate the cached colour LUT)."""
        old = self._by_id[class_id]
        new = LabelClass(
            id=old.id, name=old.name, color=(int(color[0]), int(color[1]), int(color[2]))
        )
        self._by_id[class_id] = new
        self.classes[self.classes.index(old)] = new
        self._lut = None

    def __iter__(self):
        return iter(self.classes)

    def __len__(self) -> int:
        return len(self.classes)

    @property
    def max_id(self) -> int:
        """Largest class id (used to size the colour LUT)."""
        return max(self._by_id) if self._by_id else 0

    # -- colour helpers (vectorized) -------------------------------------

    def color_lut(self) -> np.ndarray:
        """A ``(max_id + 1, 3)`` uint8 lookup table; unknown ids map to black.

        Cached; invalidate by constructing a new schema.
        """
        if self._lut is None:
            lut = np.zeros((self.max_id + 1, 3), dtype=np.uint8)
            for c in self.classes:
                lut[c.id] = c.color
            self._lut = lut
        return self._lut

    def colors_for(self, labels: np.ndarray) -> np.ndarray:
        """Map a ``(N,)`` label array to ``(N, 3)`` uint8 colours via the LUT.

        Ids beyond the LUT (or negative) clamp to ``unlabeled``'s slot rather
        than raising, so a stray label never crashes the renderer.
        """
        labels = np.asarray(labels)
        lut = self.color_lut()
        safe = np.where((labels >= 0) & (labels < lut.shape[0]), labels, self.unlabeled_id)
        return lut[safe]

    # -- io ---------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> LabelSchema:
        """Load a schema from an apairo-style label config YAML.

        Recognised keys: ``color_map`` (``id -> '#rrggbb'`` or ``[r, g, b]``),
        ``semantic_map`` (``id -> name``), and ``ignore_index``.
        """
        with open(path) as fh:
            data = yaml.safe_load(fh)
        return cls.from_config(data)

    @classmethod
    def from_config(cls, data: dict) -> LabelSchema:
        """Build a schema from an already-parsed apairo-style config dict."""
        color_map: dict = data.get("color_map", {})
        semantic_map: dict = data.get("semantic_map", {})
        unlabeled_id = int(data.get("ignore_index", 0))
        ids = sorted({int(k) for k in color_map} | {int(k) for k in semantic_map})
        classes = [
            LabelClass(
                id=i,
                name=str(semantic_map.get(i, f"class_{i}")),
                color=_parse_color(color_map.get(i, (128, 128, 128))),
            )
            for i in ids
        ]
        return cls(classes=classes, unlabeled_id=unlabeled_id)


def _parse_color(value: str | list | tuple) -> Color:
    """Parse a colour given as ``'#rrggbb'`` or an ``[r, g, b]`` sequence."""
    if isinstance(value, str):
        h = value.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    r, g, b = value[:3]
    return (int(r), int(g), int(b))
