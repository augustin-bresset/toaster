# Changelog

All notable changes to Toaster are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Domain core** (`toaster.core`): `PointCloud`, `LabelSchema`, `Selection`,
  `Grouping`, `EditHistory` / `AnnotationController`, `Session` — headless,
  numpy-only, fully typed and unit-tested.
- **Pluggable IO** loaders behind a registry: `.ply`, `.bin` (KITTI-style),
  `.las` / `.laz`, `.pcd`, plus an optional `apairo` loader.
- **Pluggable segmenters** behind a registry (`Segmenter` protocol): clustering
  (`dbscan`, `hdbscan`, `kmeans`, `kmedoids`, `agglomerative`, `optics`,
  `meanshift`) and ground detection (`ransac_ground`, `ground_grid`, `csf`).
  Heavy / quadratic methods stay usable on large clouds by clustering a bounded
  subsample and assigning the rest to the nearest cluster. `register_model`
  wraps any `predict` callable as a named segmenter in one call.
- **Web app** — a FastAPI service (`toaster-web`) and a vanilla Three.js
  front-end (no build step). The browser receives only numpy arrays and a flat
  snapshot; all colouring happens client-side.
- **Native desktop app** (`toaster`) — the same web UI in a pywebview window.
- **Point / Box / Voxel** selection modes; **double-click (left or right) to
  label** a cluster, point, voxel, or a drawn box in one gesture; Shift/Ctrl to
  add/subtract; undo/redo; labels saved beside the cloud and restored on reopen.
- **Segments** panel: per-group visibility (hidden groups grey out while already
  labelled points keep their class colour), **Assign checked** to label every
  visible group at once, and discard-on-close.
- **File browser**: launch with no path and browse the filesystem in-app — or
  type a path with **Tab**-completion.
- **Classes** manager (add / rename / recolour / remove) and display modes
  (Labels / Grouping / Intensity / Height).
- **Themes** (Toaster, Café Toaster, Arcade Quest), each with its own animated
  logo, plus a neon-flicker on label and a glitch on a finished segmentation.
- `--plugin MODULE` on `toaster` / `toaster-web` to import custom
  segmenters/loaders at launch.
- Packaging: `py.typed`, MIT `LICENSE`, GitHub Actions CI (lint + format + tests
  on Python 3.11 / 3.12), `CONTRIBUTING.md`, a runnable example
  (`examples/make_sample.py`) and a Docker image + deploy guide.

[Unreleased]: https://github.com/augustin-bresset/toaster/commits/main
