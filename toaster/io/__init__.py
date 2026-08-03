"""Point-cloud loading — a small registry keyed by file extension.

``load_cloud(path)`` dispatches on the file's suffix. Register your own format
with :func:`register_loader`; see :class:`~toaster.io.base.Loader`.
"""

from __future__ import annotations

from pathlib import Path

from toaster.core import PointCloud

from .apairo_loader import ApairoFrames
from .base import Loader
from .bin_loader import BinLoader
from .label_channel import load_label_channel
from .las_loader import LasLoader
from .npy_loader import NpyLoader
from .pcd_loader import PcdLoader
from .ply_loader import PlyLoader

__all__ = [
    "Loader",
    "LOADERS",
    "register_loader",
    "load_cloud",
    "supported_extensions",
    "load_label_channel",
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
register_loader(NpyLoader())

# Open3D is *not* registered even when installed: the built-in readers cover the
# same files (PcdLoader decodes binary_compressed too) and keep the intensity and
# label channels Open3D's point cloud drops. Register it yourself if you want it:
#     from toaster.io.open3d_loader import Open3DLoader
#     register_loader(Open3DLoader())
