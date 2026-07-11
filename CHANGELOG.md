# Changelog

All notable changes to Toaster are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Rerun-style point shading + finer, capped outline** — each point sprite
  now gets a soft centre-to-rim highlight (the same falloff `re_renderer`,
  rerun's renderer, uses for its points/spheres) plus anti-aliased edges,
  giving a subtle 3D "bead" look instead of a flat disc. The outline is
  thinner, blends into that shading rather than being a flat painted border,
  and is now capped to a fixed pixel width so it stays a fine rim instead of
  ballooning into a thick ring at large point sizes.
- **XY-plane reference grid** — a 1 m / 10 m grid at world Z=0, scaled to the
  open cloud, for spatial/scale reference. It's a shader-lit ground plane
  rather than a fixed line list: 10 m lines read thicker/brighter, the 1 m
  lines self-fade before they'd alias as the camera pulls back, and the whole
  grid fades smoothly — toward the camera and its own edge — instead of
  stopping abruptly.
- **`toaster demo`** — pass `demo` as the path (CLI or the web `/open` route)
  to generate a fresh procedural terrain scene (Perlin-noise hills, trees,
  patchy grass — the same generator behind `examples/sample.bin`, now in
  `toaster.demo_scene`) and open it on the spot, no sample file required. A
  new random layout every time unless a seed is passed programmatically.
- **Select all** (`Ctrl+A`) — selects every point of the cloud by index,
  regardless of the camera's current framing or the visibility mask. A
  screen-space box drag can silently miss points that are hidden (mask) or
  clipped by the near/far planes; this closes that gap for "label the whole
  cloud" workflows.
- **Load channel…** moved from the floating Session window to the top
  toolbar — rare enough an action that it didn't warrant its own window.
- **Point visibility mask** — one client-side mask composed of: hide
  already-labelled points (`H`), per-class 👁 toggles in the Classes panel, and
  isolate-the-selection (`I`). `X` (or the bottom-right "hidden" pill) reveals
  everything; selected points always stay visible; hidden points cannot be
  picked or labelled.
- **Motion-time LOD** — clouds above one million points render a pre-shuffled
  uniform subset while the camera moves (drag, fly, zoom, arrow rotations) and
  the full cloud the moment it settles. Only the draw list changes: picking,
  selection and labelling always operate on every point.

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
