---
name: verify
description: Verify Toaster changes by driving the web UI headlessly with Playwright against toaster-web.
---

# Verify Toaster

The desktop app (`toaster`) is pywebview/QtWebEngine over a local FastAPI
server; `toaster-web` serves the exact same UI + REST API for a plain
browser. Verify UI/JS/API changes against `toaster-web` with Playwright
(both are already in `.venv`).

## Launch

```bash
.venv/bin/toaster-web examples/sample.bin --port 8765   # background it
```

`examples/sample.bin` is a 21k-point KITTI-style float32 xyzi cloud that
loads on startup. Wait ~2.5 s after `page.goto` for the cloud to arrive.

## Drive

```python
from playwright.sync_api import sync_playwright
# chromium headless works; viewport 1280x800 shows the full layout
```

- The 3D view is the `canvas` element; drag = mouse.down/move/up on it,
  wheel zoom = `page.mouse.wheel`, fly = hold `w/a/s/d` (physical codes).
- Click a point → INSPECTOR panel (bottom left) shows `Point #<i>`, pos,
  label; the picked point renders as a yellow highlight.
- Rendering is on demand: the rAF loop is parked while idle and restarts
  on interaction. To assert render activity, wrap rAF in an init script
  (`page.add_init_script`) that counts calls — expect ~0 over 3 s of
  idle and ~60/s during a drag or held fly key.
- Collect `pageerror` + console errors; the app should produce none.
- `window.__toaster.viewer` exposes the viewer (camera, controls, geom) for
  assertions — e.g. `camera.position.distanceTo(controls.target)` to check
  orbit-vs-zoom behaviour, since module scope is unreachable from the page.

## Gotchas

- The old Qt/VTK GUI notes (QT_QPA_PLATFORM=xcb DISPLAY=:1) are for the
  legacy PyVista viewer, not this web UI.
- The desktop shell picks GTK if `gi` is importable, else Qt/QtWebEngine.
  QtWebEngine on this box falls back to Vulkan ("GBM is not supported"),
  which leaks per composited frame — hence idle must produce zero frames.
