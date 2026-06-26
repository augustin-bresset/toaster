"""Point-cloud loading — a small registry keyed by file extension.

``load_cloud(path)`` dispatches on the file's suffix. Register your own format
with :func:`register_loader`; see :class:`~toaster.io.base.Loader`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from toaster.core import PointCloud

from .apairo_loader import ApairoFrames
from .base import Loader
from .bin_loader import BinLoader
from .las_loader import LasLoader
from .pcd_loader import PcdLoader
from .ply_loader import PlyLoader

__all__ = [
    "Loader",
    "LOADERS",
    "register_loader",
    "load_cloud",
    "supported_extensions",
    "ApairoFrames",
]

#: Extension (lowercase, with dot) -> loader instance.
LOADERS: dict[str, Loader] = {}


def register_loader(loader: Loader, *, override: bool = True) -> None:
    """Register ``loader`` for each of its extensions.

    Args:
        loader: A :class:`~toaster.io.base.Loader` instance.
        override: If ``False``, keep any loader already registered for an
            extension instead of replacing it.
    """
    for ext in loader.extensions:
        ext = ext.lower()
        if override or ext not in LOADERS:
            LOADERS[ext] = loader


def supported_extensions() -> list[str]:
    """Sorted list of currently loadable extensions."""
    return sorted(LOADERS)


def load_cloud(path: str | Path) -> PointCloud:
    """Load a point cloud from ``path``, dispatching on its extension."""
    path = Path(path)
    ext = path.suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"no loader for '{ext}' (supported: {', '.join(supported_extensions())})")
    return loader.load(path)


# -- built-in registrations ----------------------------------------------

register_loader(PlyLoader())
register_loader(BinLoader())
register_loader(LasLoader())
register_loader(PcdLoader())

# Prefer Open3D for PCD when available (it also decodes binary_compressed);
# leave .ply to the lighter built-in loader. Detected without importing the
# heavy module; the loader imports it lazily.
if importlib.util.find_spec("open3d") is not None:
    from .open3d_loader import Open3DLoader

    LOADERS[".pcd"] = Open3DLoader()
