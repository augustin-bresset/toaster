"""Detect when a cloud belongs to an apairo dataset and write labels back as a
proper apairo channel.

Detection and timestamp lookup are pure filesystem + YAML — they need no
``apairo`` import, so the Save dialog can *offer* "write to the dataset" even
where the package isn't installed. Only :func:`write_labels_channel` actually
imports ``apairo`` (lazily), so the dependency stays optional — Toaster is never
coupled to apairo just by importing this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

__all__ = ["ApairoTarget", "detect_apairo_channel", "frame_timestamp", "write_labels_channel"]

#: apairo loaders that store one file per frame (a frame a writer can append to).
#: A "stacked" loader (``npy`` — one array for the whole sequence) is rejected.
_PER_FRAME_LOADERS = {"npys", "bin"}


@dataclass(frozen=True)
class ApairoTarget:
    """Where and how to write a label channel for one cloud frame."""

    seq_dir: str  #: dataset root (holds ``.apairo/channels.yaml``)
    source_channel: str  #: channel the cloud came from, e.g. ``"ouster_points"``
    stem: str  #: frame stem, e.g. ``"001813"``


def detect_apairo_channel(cloud_source: str | Path | None) -> ApairoTarget | None:
    """Return where to write labels for ``cloud_source``, or ``None``.

    A cloud counts as apairo-managed when it sits at
    ``<seq_dir>/<channel>/<stem>.<ext>`` and ``<seq_dir>/.apairo/channels.yaml``
    lists ``<channel>`` with a per-frame loader (``npys``/``bin``). Anything else
    — no ``.apairo``, an unknown channel, or a stacked loader — yields ``None``.
    """
    if cloud_source is None:
        return None
    p = Path(cloud_source).resolve()
    source_channel = p.parent.name
    seq_dir = p.parent.parent
    channels_yaml = seq_dir / ".apairo" / "channels.yaml"
    if not channels_yaml.is_file():
        return None
    try:
        data = yaml.safe_load(channels_yaml.read_text()) or {}
    except yaml.YAMLError:
        return None
    spec = (data.get("channels") or {}).get(source_channel)
    if not isinstance(spec, dict) or spec.get("loader") not in _PER_FRAME_LOADERS:
        return None
    return ApairoTarget(seq_dir=str(seq_dir), source_channel=source_channel, stem=p.stem)


def frame_timestamp(seq_dir: str | Path, source_channel: str, stem: str) -> float | None:
    """Timestamp of frame ``stem`` from ``<source_channel>/timestamps.txt``.

    The line is found by the frame's position among the channel's sorted ``.npy``
    files (robust to gaps), matching how the loader pairs frames to timestamps.
    Returns ``None`` if there is no timestamps file or the frame isn't found.
    """
    channel_dir = Path(seq_dir) / source_channel
    ts_file = channel_dir / "timestamps.txt"
    if not ts_file.is_file():
        return None
    frames = sorted(f.stem for f in channel_dir.glob("*.npy"))
    try:
        idx = frames.index(stem)
    except ValueError:
        return None
    lines = ts_file.read_text().split()
    if idx >= len(lines):
        return None
    return float(lines[idx])


def write_labels_channel(
    target: ApairoTarget, labels: np.ndarray, timestamp: float, channel: str = "ground_truth"
) -> Path:
    """Write ``labels`` for one frame as an apairo ``preprocess`` channel.

    Uses ``apairo.ChannelWriter`` (imported lazily — needs the ``apairo`` extra),
    which owns the on-disk format: per-frame file, ``timestamps.txt`` upkeep, and
    registration in ``.apairo/channels.yaml``. Resuming an existing channel makes
    annotation incremental across frames. Returns the written frame's path.
    """
    import apairo

    with apairo.ChannelWriter(
        target.seq_dir,
        channel,
        loader="npys",
        timestamps_from=target.source_channel,
        sources=[target.source_channel],
    ) as writer:
        writer.add(labels, stem=target.stem, timestamp=timestamp)
    return Path(target.seq_dir) / channel / f"{target.stem}.npy"
