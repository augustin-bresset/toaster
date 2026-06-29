# Toaster

Annotate lidar **point clouds** in 3D — walk through them, select points one by
one or by zone, assign semantic classes — and, its headline feature, **plug in
any model that groups points together** (clustering like DBSCAN, or neural-net
inference) so that **clicking one cluster labels the whole group at once**.

## The idea in one picture

A clustering/segmentation model and a manual zone selection are the *same thing*:
both produce **groups of points**. So Toaster keeps two layers strictly apart:

| Layer | Object | Nature |
|---|---|---|
| Grouping | `Grouping` (`group_id`, `-1` = noise) | **transient**, produced by a model, disposable |
| Annotation | `labels` (one class per point) | **persistent** — the only thing saved |

`Selection` is the bridge: `Grouping → Selection → labels`. Run a segmenter to
get a grouping, click a cluster to select its whole group, assign a class.

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```

Optional extras: `apairo` (load apairo datasets), `open3d` (robust `.pcd`),
`hdbscan`, `models` (ONNX), `torch`.

## Run the app

`toaster` opens a native desktop window (the web UI in a pywebview shell);
`toaster-web` serves the same UI for a plain browser.

```bash
python examples/make_sample.py     # writes examples/sample.bin
toaster examples/sample.bin        # native window — or .ply / .las / .laz / .pcd
toaster-web examples/sample.bin    # then open http://127.0.0.1:8000
```

- Left-click a point to select it (whole cluster, if a grouping is active).
  Hold **Shift** to add, **Ctrl** to subtract. Press **R** then drag for a box select.
  Left-drag orbits the camera (it no longer selects); scroll to zoom.
- **To label: select points, then assign a class** — press the number shown beside
  the class in the *Classes* panel, click the **Assign** toolbar button, or press
  **Enter** (uses the highlighted class; `0` clears back to unlabeled).
  **Ctrl+Z / Ctrl+Shift+Z** to undo/redo. **Ctrl+S** saves labels *and* the schema
  beside the cloud (`<cloud>.toaster.npy` / `.toaster.schema.yaml`), both restored
  on reopen.
- The *Classes* panel configures what you label into: **Add / Rename / Remove** a
  class, double-click to recolour. *File ▸ Load/Save schema…* imports/exports the
  palette as apairo-style YAML.
- The *Panels* menu hides/shows each dock — handy if you closed one by accident.
- The *Segmenter* panel runs DBSCAN/HDBSCAN (optionally on the current selection);
  the result becomes the active grouping, then a click labels a whole cluster.

## Use it as a library (headless)

`toaster.core` is numpy-only and never imports Qt/VTK, so it works in a script
or a pipeline:

```python
import numpy as np
from toaster.io import load_cloud
from toaster.core import LabelSchema, Selection
from toaster.segment import get_segmenter

cloud = load_cloud("scan.ply")
cloud.ensure_labels()

# Cluster, then label whole clusters programmatically.
grouping = get_segmenter("dbscan", eps=0.4, min_samples=12).segment(cloud)
from toaster.core import AnnotationController
ann = AnnotationController(cloud)
for gid in grouping.group_ids():
    ann.assign(Selection.from_group(grouping, gid), class_id=4)  # e.g. "vehicle"

np.save("scan.labels.npy", cloud.labels)
```

## Extend it — the two seams

**A custom segmenter** (anything that groups points):

```python
from toaster.segment import register_segmenter, scatter
from toaster.segment.base import resolve_points

@register_segmenter
class SliceByHeight:
    name = "height_slices"
    def __init__(self, step: float = 1.0):
        self.step = step
    def segment(self, cloud, selection=None):
        xyz, indices = resolve_points(cloud, selection)
        group_ids = (xyz[:, 2] / self.step).astype(int)
        return scatter(group_ids, indices, cloud.n, source=self.name)
```

It now appears in the app's segmenter panel and via `get_segmenter("height_slices")`.

**I already have a Python model that labels points.** One call registers it as a
named segmenter — its predicted classes become groups *and* `suggested_labels`,
so a click accepts the prediction:

```python
# my_segmenters.py
from toaster.segment import register_model
import my_net   # your model

def predict(points):           # points is (M, 3+F); returns (M,) class ids
    return my_net.run(points)  # torch / ONNX / sklearn — anything

register_model("my_net", predict, feature_keys=["intensity"], ignore_id=0)
# feature_keys=["intensity"] => the model receives [x, y, z, intensity]
```

Then make the app load it:

```bash
toaster scan.ply --plugin my_segmenters     # --plugin imports the module first
```

`my_net` now appears in the segmenter panel; run it (optionally scoped to the
current selection), then click a predicted region to label it. In a script you
don't need `--plugin` — just import the module, then `get_segmenter("my_net")`.
For full control (a custom param UI, no `suggested_labels`, custom grouping) write
the `Segmenter` class directly as above, or use `ModelSegmenter` / `FunctionSegmenter`.

**A custom loader** (a new file format):

```python
from toaster.io import register_loader
from toaster.core import PointCloud

class XyzLoader:
    extensions = (".xyz",)
    def load(self, path):
        import numpy as np
        return PointCloud(xyz=np.loadtxt(path, dtype="float32")[:, :3], source=path)

register_loader(XyzLoader())
```

## Architecture

```
toaster/
  core/         domain, numpy-only, headless, 100% unit-tested
  io/           pluggable loaders (registry) — .ply/.bin/.las/.pcd (+apairo)
  segment/      pluggable segmenters (registry) — dbscan/hdbscan/model
  viewer/       Viewer protocol + PyVista backend (swappable; no VTK leaks out)
  interaction/  front-end-agnostic controller (workflow glue, no Qt/VTK)
  app/          Qt UI — wires widgets to the interaction controller
  persistence/  label sidecar + session JSON
```

Dependency rule: `core` depends on nothing; `io/segment/persistence` depend only
on `core`; `viewer` adds PyVista; `interaction` glues core + viewer protocol but
stays headless; `app` is the only layer that touches everything. The `Viewer`
protocol passes only numpy arrays and indices, so the renderer is replaceable,
and the headless `interaction` controller can be driven by a non-Qt front-end.

## Development

```bash
make check   # ruff + pytest
```
