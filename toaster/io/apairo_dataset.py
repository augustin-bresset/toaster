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

__all__ = [
    "ApairoTarget",
    "ApairoNav",
    "TfTarget",
    "detect_apairo_channel",
    "detect_apairo_nav",
    "detect_tf",
    "resolve_tf_matrix",
    "frame_path",
    "frame_timestamp",
    "write_labels_channel",
]

#: Preferred body frame to resolve into (gravity-fixed on rough terrain).
DEFAULT_TF_TARGET = "odom"
#: Frame names, in preference order, used to pick the rigid body ("base") frame.
_BASE_FRAME_HINTS = ("base_link", "base_footprint")

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


@dataclass(frozen=True)
class ApairoNav:
    """The dataset → sequence → channel → frame position of the open cloud."""

    dataset_root: str  #: dir holding ``.apairo/dataset.yaml``
    dataset_name: str
    sequences: list[str]  #: sequences listed in ``dataset.yaml``
    sequence: str  #: current sequence
    channels: list[str]  #: annotatable point-cloud channels in this sequence
    channel: str  #: current channel
    frames: list[str]  #: frame stems in the current channel, sorted
    frame_index: int  #: position of the open frame in ``frames`` (-1 if unknown)


def _point_channels(seq_dir: Path) -> list[str]:
    """Channels of a sequence whose frames are point clouds (``msgtype`` PointCloud2).

    This distinguishes lidar channels from same-loader image channels (a ZED
    ``npys`` channel is ``msgtype`` Image), so only annotatable channels surface.
    """
    channels_yaml = seq_dir / ".apairo" / "channels.yaml"
    try:
        data = yaml.safe_load(channels_yaml.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return []
    out = []
    for name in data.get("channels") or {}:
        meta = seq_dir / name / "metadata.yaml"
        if not meta.is_file():
            continue
        try:
            msg = (yaml.safe_load(meta.read_text()) or {}).get("msgtype", "")
        except (OSError, yaml.YAMLError):
            continue
        if "PointCloud2" in str(msg):
            out.append(name)
    return sorted(out)


def _frame_stems(channel_dir: Path) -> list[str]:
    """Sorted frame stems of a channel (excludes Toaster ``_toaster`` sidecars)."""
    return sorted(f.stem for f in channel_dir.glob("*.npy") if "_toaster" not in f.stem)


def detect_apairo_nav(cloud_source: str | Path | None) -> ApairoNav | None:
    """Resolve the dataset navigation context for ``cloud_source``, or ``None``.

    A cloud navigates a dataset when it sits at
    ``<dataset>/<sequence>/<channel>/<stem>.<ext>`` where ``<dataset>/.apairo/
    dataset.yaml`` lists ``<sequence>``. Returns the sequences, the sequence's
    point channels, and the current channel's frames so a UI can move between
    sequences, channels and frames.
    """
    if cloud_source is None:
        return None
    p = Path(cloud_source).resolve()
    channel = p.parent.name
    seq_dir = p.parent.parent
    dataset_dir = seq_dir.parent
    dataset_yaml = dataset_dir / ".apairo" / "dataset.yaml"
    if not dataset_yaml.is_file():
        return None
    try:
        data = yaml.safe_load(dataset_yaml.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    sequences = [str(s) for s in (data.get("sequences") or [])]
    if seq_dir.name not in sequences:
        return None
    frames = _frame_stems(seq_dir / channel)
    try:
        idx = frames.index(p.stem)
    except ValueError:
        idx = -1
    return ApairoNav(
        dataset_root=str(dataset_dir),
        dataset_name=str(data.get("name", dataset_dir.name)),
        sequences=sequences,
        sequence=seq_dir.name,
        channels=_point_channels(seq_dir),
        channel=channel,
        frames=frames,
        frame_index=idx,
    )


def frame_path(
    dataset_root: str | Path, sequence: str, channel: str, frame_index: int
) -> Path | None:
    """Path of frame ``frame_index`` in ``<dataset_root>/<sequence>/<channel>``.

    ``frame_index`` is clamped to the available range. Returns ``None`` if the
    channel has no frames.
    """
    channel_dir = Path(dataset_root) / sequence / channel
    frames = _frame_stems(channel_dir)
    if not frames:
        return None
    i = max(0, min(int(frame_index), len(frames) - 1))
    return channel_dir / f"{frames[i]}.npy"


def frame_timestamp(seq_dir: str | Path, source_channel: str, stem: str) -> float | None:
    """Timestamp of frame ``stem`` from ``<source_channel>/timestamps.txt``.

    The line is found by the frame's position among the channel's sorted ``.npy``
    files (robust to gaps), matching how the loader pairs frames to timestamps.
    Must use :func:`_frame_stems` (not a raw glob): a bare ``*.npy`` also matches
    Toaster's own local sidecar labels saved beside the source channel
    (``<stem>_toaster.npy``, from the plain "Save" button, unrelated to this
    apairo channel), which would silently shift every later frame's position —
    and therefore the timestamp this function writes into the label channel —
    by however many sidecars happen to sit alphabetically before ``stem``.
    Returns ``None`` if there is no timestamps file or the frame isn't found.
    """
    channel_dir = Path(seq_dir) / source_channel
    ts_file = channel_dir / "timestamps.txt"
    if not ts_file.is_file():
        return None
    frames = _frame_stems(channel_dir)
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


# ── TF resolution ────────────────────────────────────────────────────────────
# Optional: place a frame's cloud into a gravity-fixed body frame (odom) so a
# tilted sensor scan stands upright. The heavy lifting is apairo's — the static
# mount comes from `Calibration.get_tf`, the dynamic base->target pose from a
# `tf__<target>__<base>` channel resynced onto the lidar clock with Se3Interp.
# Detection below is filesystem-only (no apairo import); resolution imports
# apairo + apairo_transform lazily, so the dependency stays optional.


@dataclass(frozen=True)
class TfTarget:
    """A resolvable TF chain for a cloud: sensor -> base (static) -> target (dynamic)."""

    target: str  #: body frame to resolve into, e.g. ``"odom"``
    base_frame: str  #: rigid base frame, e.g. ``"base_link"``
    lidar_frame: str  #: the cloud's own frame, e.g. ``"os_sensor"``
    pose_channel: str  #: dynamic pose of base in target, e.g. ``"tf__odom__base_link"``


def _yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _calibration_frames(seq_dir: Path) -> set[str]:
    """Frame nodes of the static transform tree, parsed from ``A_to_B`` edge keys."""
    edges = (_yaml(seq_dir / ".apairo" / "calibration.yaml").get("transforms") or {}).keys()
    frames: set[str] = set()
    for edge in edges:
        head, sep, tail = str(edge).partition("_to_")
        if sep:
            frames.update((head, tail))
    return frames


def detect_tf(cloud_source: str | Path | None, target: str = DEFAULT_TF_TARGET) -> TfTarget | None:
    """Whether ``cloud_source``'s frame can be resolved into ``target``, else ``None``.

    Pure filesystem/YAML (no ``apairo`` import): needs a calibration tree, the
    cloud channel's declared ``frame`` reachable in it, a base frame, and a
    ``tf__<target>__<base>`` dynamic pose channel present in the sequence.
    """
    if cloud_source is None:
        return None
    p = Path(cloud_source).resolve()
    channel, seq_dir = p.parent.name, p.parent.parent
    channels = _yaml(seq_dir / ".apairo" / "channels.yaml").get("channels") or {}
    lidar_frame = (channels.get(channel) or {}).get("frame")
    if not lidar_frame:
        return None
    calib = _calibration_frames(seq_dir)
    if lidar_frame not in calib:
        return None
    base_frame = next((b for b in _BASE_FRAME_HINTS if b in calib), None)
    if base_frame is None or base_frame == lidar_frame:
        return None
    pose_channel = f"tf__{target}__{base_frame}"
    if pose_channel not in channels or not (seq_dir / pose_channel).is_dir():
        return None
    return TfTarget(
        target=target, base_frame=base_frame, lidar_frame=lidar_frame, pose_channel=pose_channel
    )


#: (seq_dir, cloud_channel, target) -> (synced-dataset, static-mount 4x4). The
#: synchronized view is lazy and immutable, so it is safe to reuse across frames
#: of the same sequence instead of rebuilding it on every navigation.
_TF_CACHE: dict[tuple[str, str, str], tuple[object, np.ndarray]] = {}


def resolve_tf_matrix(
    cloud_source: str | Path, frame_index: int, target: str = DEFAULT_TF_TARGET
) -> np.ndarray | None:
    """The 4x4 ``T_target_from_sensor`` for this frame, or ``None`` if unresolvable.

    Composes the static sensor->base mount (``Calibration.get_tf``) with the
    dynamic base->target pose interpolated onto the cloud's timestamp
    (``Se3Interp``). Apply it to the cloud's ``(N, 3)`` xyz to stand it upright.
    Requires ``apairo`` and ``apairo_transform``; returns ``None`` if either the
    chain is absent or the packages are missing.
    """
    tf = detect_tf(cloud_source, target)
    if tf is None or frame_index < 0:
        return None
    try:
        import apairo
        from apairo_transform import PoseTo4x4
        from apairo_transform.interp import Se3Interp
    except ImportError:
        return None
    p = Path(cloud_source).resolve()
    channel, seq_dir = p.parent.name, p.parent.parent
    key = (str(seq_dir), channel, target)
    entry = _TF_CACHE.get(key)
    if entry is None:
        ds = apairo.RawDataset(str(seq_dir), keys=[channel, tf.pose_channel])
        mount = np.asarray(ds.calibration.get_tf(tf.lidar_frame, tf.base_frame), dtype=np.float64)
        synced = ds.synchronize(reference=channel, method={tf.pose_channel: Se3Interp()}).transform(
            tf.pose_channel, PoseTo4x4()
        )
        entry = (synced, mount)
        _TF_CACHE[key] = entry
    synced, mount = entry
    t_target_base = np.asarray(synced[frame_index].data[tf.pose_channel], dtype=np.float64).reshape(
        4, 4
    )
    return t_target_base @ mount
