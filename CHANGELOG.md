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
- Editable label schema from the UI: add, rename, recolour and remove classes in
  the *Classes* panel (`LabelSchema.add_class`/`rename`/`remove`), plus
  *File ▸ Load/Save schema…* (`LabelSchema.to_yaml`).
- The labelling schema is now saved beside the cloud
  (`<cloud>.toaster.schema.yaml`) and restored on reopen, so a cloud's classes
  come back exactly as left.
- *Panels* menu with a checkable entry per dock, so a closed panel can be
  reopened (previously it was gone until restart).

### Fixed
- Labelling is now discoverable and works: an **Assign** toolbar button / *Edit ▸
  Assign* / **Enter** labels the selection with the active class. Number keys map
  to the class *id* shown in the panel (so `1` labels class 1, not "unlabeled"),
  and a new session adopts the panel's highlighted class as the brush.
- Running a clusterer on a one-point selection no longer crashes (sklearn aborts
  on a single sample): segmenters return an empty grouping, and the window warns
  on a too-small selection instead of failing.
- Rotating the camera no longer snaps the view back to the cloud: the selection
  overlay and recolours keep the current camera instead of re-framing.
- Left-drag now orbits cleanly without selecting; a point is picked only on a
  stationary left click.
- On a Wayland session, Qt is routed through XWayland (`xcb`) so VTK gets a real
  X11 window instead of aborting with `BadWindow`.

## [0.1.0] - unreleased
- Initial scaffolding.
