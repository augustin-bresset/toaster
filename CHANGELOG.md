# Changelog

All notable changes to Toaster are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Core domain library (`toaster.core`): `PointCloud`, `LabelSchema`, `Selection`,
  `Grouping`, `EditHistory`/`AnnotationController`, `Session`. Headless,
  numpy-only, fully typed.
- Pluggable IO loaders (`.ply`, `.bin`, `.las/.laz`, `.pcd`) with a registry and
  an optional `apairo` loader.
- Pluggable segmenters (`Segmenter` protocol + registry): DBSCAN, HDBSCAN, and
  generic function/model wrappers.
- PyVista-based interactive viewer behind a backend-agnostic `Viewer` protocol.
- Qt desktop application with point/box selection, class palette, undo/redo and
  label persistence.

## [0.1.0] - unreleased
- Initial scaffolding.
