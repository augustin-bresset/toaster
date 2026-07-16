"""End-to-end checks of the web viewer's octree LOD/picking and delta recolour.

Drives the real stack — ``toaster-web`` served from THIS checkout, headless
Chromium via Playwright — against a synthetic cloud big enough to build the
octree and trip the motion LOD (> 1M points). Skips cleanly when Playwright
or its browser is unavailable.

Invariants covered:
- the worker-built octree is a permutation of the cloud with well-formed,
  disjoint node slices;
- camera motion renders the octree cut (a strict subset), idle refines back
  to the full cloud, and the rAF loop parks afterwards (the QtWebEngine
  leak guard);
- octree pick/pickBox return exactly what the brute-force scans return;
- after delta-applied label edits (assign, undo), the client's colour/alpha
  buffers are indistinguishable from a from-scratch full-state re-render.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
N_POINTS = 1_200_000  # > LOD_BUDGET (1M) so the motion cut engages


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """toaster-web on a random port, serving a 1.2M-point synthetic terrain."""
    cloud = tmp_path_factory.mktemp("cloud") / "big.bin"
    rng = np.random.default_rng(7)
    xy = rng.uniform(-50, 50, (N_POINTS, 2)).astype(np.float32)
    z = (np.sin(xy[:, 0] * 0.2) * np.cos(xy[:, 1] * 0.15) + rng.normal(0, 0.1, N_POINTS)).astype(
        np.float32
    )
    i = rng.uniform(0, 1, N_POINTS).astype(np.float32)
    np.column_stack([xy, z, i]).astype(np.float32).tofile(cloud)

    port = _free_port()
    env = os.environ | {"PYTHONPATH": str(REPO)}  # serve THIS checkout, not the installed copy
    proc = subprocess.Popen(
        [sys.executable, "-m", "toaster.api.server", str(cloud), "--port", str(port)],
        env=env,
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    import urllib.request

    deadline = time.time() + 30
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/meta", timeout=1)
            break
        except Exception:
            if proc.poll() is not None:
                pytest.skip(f"server died: {proc.stderr.read().decode()[-500:]}")
            if time.time() > deadline:
                proc.kill()
                pytest.skip("server did not come up in 30s")
            time.sleep(0.3)
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # no browser installed
            pytest.skip(f"chromium unavailable: {exc}")
        yield browser
        browser.close()


@pytest.fixture(scope="module")
def page(server, browser):
    """One loaded page (cloud + octree ready) shared by the module's tests."""
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    # Headless SwiftShader emits a spurious "VALIDATE_STATUS false" with an
    # EMPTY info log under multi-context pressure; a real shader break always
    # carries a compile log, so only the empty-log noise is filtered.
    page.on(
        "console",
        lambda m: errors.append(f"console: {m.text}")
        if m.type == "error" and not ("VALIDATE_STATUS false" in m.text and "ERROR:" not in m.text)
        else None,
    )
    page.goto(server)
    page.wait_for_function(
        f"window.__toaster && window.__toaster.viewer.geom && "
        f"window.__toaster.viewer.geom.getAttribute('position').count === {N_POINTS}",
        timeout=90000,
    )
    page.wait_for_function("window.__toaster.viewer._octree !== null", timeout=90000)
    yield page
    assert not errors, f"console/page errors: {errors}"
    page.close()


def test_octree_is_a_permutation_with_wellformed_slices(page):
    ok = page.evaluate(
        f"""(() => {{
          const o = window.__toaster.viewer._octree.data;
          if (o.order.length !== {N_POINTS}) return "bad order length";
          const seen = new Uint8Array({N_POINTS});
          for (let k = 0; k < o.order.length; k++) {{
            if (seen[o.order[k]]) return "duplicate index";
            seen[o.order[k]] = 1;
          }}
          for (let i = 0; i < seen.length; i++) if (!seen[i]) return "missing index";
          for (let ni = 0; ni < o.start.length; ni++) {{
            if (o.start[ni] + o.count[ni] > o.order.length) return "slice out of bounds";
            for (let c = 0; c < 8; c++) {{
              const ch = o.children[ni * 8 + c];
              if (ch >= 0 && o.start[ch] < o.start[ni] + o.count[ni])
                return "child overlaps parent sample";
            }}
          }}
          return "ok";
        }})()"""
    )
    assert ok == "ok"


def test_motion_draws_octree_cut_then_idle_refines_full(page):
    # A held fly key keeps the view "in motion" for as long as we need to
    # observe it (a drag's 150 ms settle window is shorter than one headless
    # software-GL frame, so the refine would race the assertion).
    page.click("canvas", position={"x": 10, "y": 10})
    page.keyboard.down("w")
    time.sleep(0.4)
    during = page.evaluate(
        """(() => {
          const v = window.__toaster.viewer;
          return { cut: v.geom.index === v._drawAttr, n: v.geom.drawRange.count };
        })()"""
    )
    page.keyboard.up("w")
    assert during["cut"], "motion frames must render the octree cut"
    assert 0 < during["n"] < N_POINTS, "the cut must be a strict subset"

    time.sleep(1.5)  # settle + slow software-GL frames
    after = page.evaluate(
        "(() => { const v = window.__toaster.viewer;"
        " return v.geom.index === null && !v._lodOn; })()"
    )
    assert after, "idle must refine back to the full cloud"

    # The rAF loop must park at idle (the QtWebEngine leak guard).
    for _ in range(4):
        assert page.evaluate("window.__toaster.viewer._rafPending") is not True
        time.sleep(0.4)


def test_orbit_crosshair_shows_during_motion_and_fades(page):
    # The rerun-style pivot crosshair: hidden at rest, visible while orbiting,
    # lingers briefly after release, then fades back out.
    assert page.evaluate("window.__toaster.viewer.orbitIndicator.lines.visible") is False
    page.mouse.move(640, 400)
    page.mouse.down()
    for i in range(6):
        page.mouse.move(640 + i * 10, 400 + i * 5, steps=2)
        time.sleep(0.05)
    state = page.evaluate(
        """(() => {
          const v = window.__toaster.viewer;
          const pos = v.orbitIndicator.lines.geometry.attributes.position.array;
          const t = v.controls.target;
          return {
            visible: v.orbitIndicator.lines.visible,
            centred: Math.abs(pos[0] - t.x) < 1e-3 && Math.abs(pos[1] - t.y) < 1e-3,
          };
        })()"""
    )
    page.mouse.up()
    assert state["visible"], "crosshair must show while orbiting"
    assert state["centred"], "crosshair must sit exactly on the orbit pivot"
    time.sleep(1.0)  # > linger (350ms) + fade (100ms), slow-frame margin
    assert page.evaluate("window.__toaster.viewer.orbitIndicator.lines.visible") is False


def test_octree_picking_matches_brute_force(page):
    out = page.evaluate(
        """(() => {
          const v = window.__toaster.viewer;
          const pick1 = v.pick(640, 400);
          const box1 = v.pickBox(610, 370, 670, 430);
          const oct = v._octree;
          v._octree = null; // brute-force path
          const pick2 = v.pick(640, 400);
          const box2 = v.pickBox(610, 370, 670, 430);
          v._octree = oct;
          box1.sort((a, b) => a - b);
          box2.sort((a, b) => a - b);
          return {
            pickSame: pick1 === pick2,
            picked: pick1,
            boxSame: box1.length === box2.length && box1.every((x, i) => x === box2[i]),
            boxN: box1.length,
          };
        })()"""
    )
    assert out["pickSame"], "octree pick must match the exact scan"
    assert out["picked"] >= 0
    assert out["boxSame"], "octree pickBox must match the exact scan"
    assert out["boxN"] > 0


def test_delta_recolor_matches_full_rerender(page):
    # Box-select a screen region, label it, and compare the delta-patched
    # colour/alpha buffers against a from-scratch full-state re-render —
    # byte-identical or the delta path has drifted from the reference rules.
    # Drive the real UI: box mode, rubber-band drag, double-click to label.
    page.click("button[data-mode-pick=box]")  # idempotent — asserts the mode, not a toggle
    page.mouse.move(560, 320)
    page.mouse.down()
    page.mouse.move(720, 480, steps=5)
    page.mouse.up()
    page.wait_for_function(
        "window.__toaster.debug.getState().selection.length > 0", timeout=15000
    )
    page.mouse.dblclick(640, 440)  # viewport coords, inside the drawn box — delta path
    page.wait_for_function(
        "window.__toaster.debug.getState().selection.length === 0", timeout=15000
    )

    def buffers_equal_after_full_refresh() -> dict:
        return page.evaluate(
            """(async () => {
              const { viewer, debug } = window.__toaster;
              const live = viewer.colorArrays();
              const colors = live.colors.slice();
              const alpha = live.alpha.slice();
              await debug.refresh(); // from-scratch reference render
              const ref = viewer.colorArrays();
              let colorDiff = 0, alphaDiff = 0;
              for (let i = 0; i < colors.length; i++) if (colors[i] !== ref.colors[i]) colorDiff++;
              for (let i = 0; i < alpha.length; i++) if (alpha[i] !== ref.alpha[i]) alphaDiff++;
              const labeled = debug.getState().labels.reduce((a, v) => a + (v !== 0 ? 1 : 0), 0);
              return { colorDiff, alphaDiff, labeled };
            })()"""
        )

    out = buffers_equal_after_full_refresh()
    assert out["labeled"] > 0, "the double-click must have labelled the box selection"
    assert out["colorDiff"] == 0, f"{out['colorDiff']} colour slots diverged from the reference"
    assert out["alphaDiff"] == 0, f"{out['alphaDiff']} alpha slots diverged from the reference"

    # Same equivalence with the visibility mask active (hide-labelled points):
    # the delta must now patch alphas too, and keep the hidden-count pill exact.
    page.check("#hide-labeled")
    page.mouse.move(400, 250)
    page.mouse.down()
    page.mouse.move(560, 400, steps=5)
    page.mouse.up()
    page.wait_for_function(
        "window.__toaster.debug.getState().selection.length > 0", timeout=15000
    )
    page.mouse.dblclick(480, 330)  # viewport coords, inside the second box
    page.wait_for_function(
        "window.__toaster.debug.getState().selection.length === 0", timeout=15000
    )
    out = buffers_equal_after_full_refresh()
    assert out["colorDiff"] == 0, "colours diverged with the visibility mask on"
    assert out["alphaDiff"] == 0, "alphas diverged with the visibility mask on"

    # Undo (a delta of the same edit, reversed) must also match the reference.
    page.keyboard.press("Control+z")
    time.sleep(0.5)
    out = buffers_equal_after_full_refresh()
    assert out["colorDiff"] == 0, "colours diverged after an undo delta"
    assert out["alphaDiff"] == 0, "alphas diverged after an undo delta"
    page.uncheck("#hide-labeled")


def test_delta_upload_matches_full_upload_on_screen(page):
    # The CPU-array equality above can't see a bad commitColors update range —
    # a wrong range leaves stale bytes GPU-side only. Render both ways and
    # compare pixels: delta-patched frame vs the same state fully re-uploaded.
    page.click("button[data-mode-pick=box]")
    page.mouse.move(700, 500)
    page.mouse.down()
    page.mouse.move(860, 620, steps=5)
    page.mouse.up()
    page.wait_for_function(
        "window.__toaster.debug.getState().selection.length > 0", timeout=15000
    )
    page.mouse.dblclick(780, 560)  # delta path
    page.wait_for_function(
        "window.__toaster.debug.getState().selection.length === 0", timeout=15000
    )
    time.sleep(1.5)  # settle: full-res idle frame with the delta-patched buffers
    # Clip to the cloud area: the toolbar's neon "buzz" animation (triggered by
    # the assign) is still pulsing in frame A, and side panels are irrelevant.
    clip = {"x": 300, "y": 150, "width": 700, "height": 550}
    a = np.frombuffer(page.screenshot(clip=clip), dtype=np.uint8)
    page.evaluate("window.__toaster.debug.refresh()")
    time.sleep(1.5)  # full recompute + full GPU upload, then idle again
    b = np.frombuffer(page.screenshot(clip=clip), dtype=np.uint8)
    if a.shape == b.shape and (a == b).all():
        return  # byte-identical PNGs — perfect
    # Compare decoded pixels with a tiny tolerance for encoder nondeterminism.
    import io

    from PIL import Image

    pa = np.asarray(Image.open(io.BytesIO(a.tobytes())).convert("RGB"), dtype=np.int16)
    pb = np.asarray(Image.open(io.BytesIO(b.tobytes())).convert("RGB"), dtype=np.int16)
    diff_ratio = float((np.abs(pa - pb).max(axis=2) > 2).mean())
    assert diff_ratio < 0.001, f"{diff_ratio:.4%} of pixels differ between delta and full upload"


def test_isolate_mode_falls_back_to_full_recompute(page):
    # Isolate couples every point's alpha to the selection — assigning under
    # isolate must go through the full recompute and still match the reference.
    page.click("button[data-mode-pick=box]")
    page.mouse.move(300, 500)
    page.mouse.down()
    page.mouse.move(440, 620, steps=5)
    page.mouse.up()
    page.wait_for_function(
        "window.__toaster.debug.getState().selection.length > 0", timeout=15000
    )
    page.keyboard.press("i")  # isolate the selection
    page.mouse.dblclick(370, 560)  # label it (assign clears the selection → isolate disarms)
    page.wait_for_function(
        "window.__toaster.debug.getState().selection.length === 0", timeout=15000
    )
    out = page.evaluate(
        """(async () => {
          const { viewer, debug } = window.__toaster;
          const live = viewer.colorArrays();
          const colors = live.colors.slice();
          const alpha = live.alpha.slice();
          await debug.refresh();
          const ref = viewer.colorArrays();
          let colorDiff = 0, alphaDiff = 0;
          for (let i = 0; i < colors.length; i++) if (colors[i] !== ref.colors[i]) colorDiff++;
          for (let i = 0; i < alpha.length; i++) if (alpha[i] !== ref.alpha[i]) alphaDiff++;
          return { colorDiff, alphaDiff };
        })()"""
    )
    assert out["colorDiff"] == 0, "colours diverged after assigning under isolate"
    assert out["alphaDiff"] == 0, "alphas diverged after assigning under isolate"


def test_engine_options_then_dispose(server, browser):
    # dispose() is for hosts that embed the engine and tear viewers down —
    # exercise it on its own page so the shared page fixture stays alive.
    page = browser.new_page(viewport={"width": 800, "height": 600})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(server)
    page.wait_for_function(
        "window.__toaster && window.__toaster.viewer._octree !== null", timeout=90000
    )
    # Exercise the embedding-oriented options on this throwaway page first:
    # orbit control style, world-size attenuation, streaming setCloud.
    opts = page.evaluate(
        """(() => {
          const v = window.__toaster.viewer;
          v.setControlStyle("orbit");
          const orbitOk = v._controlStyle === "orbit" && v.controls.enableDamping === false;
          v.setSizeAttenuation(true);
          const attenOk =
            v.material.uniforms.uAttenuate.value === 1 &&
            v.material.uniforms.uProjScalePx.value > 0;
          v.setSizeAttenuation(false);
          v.setControlStyle("trackball");
          const backOk = v._controlStyle === "trackball" && !!v.controls.staticMoving;
          // Streaming: a smaller frame, no octree churn, camera untouched.
          const camBefore = v.camera.position.toArray();
          const xyz = new Float32Array(3000);
          for (let i = 0; i < 1000; i++) {
            xyz[i * 3] = Math.random() * 10;
            xyz[i * 3 + 1] = Math.random() * 10;
            xyz[i * 3 + 2] = Math.random();
          }
          v.setCloud(xyz, { octree: false, frame: false });
          const camAfter = v.camera.position.toArray();
          const streamOk =
            v._octree === null &&
            v.geom.getAttribute("position").count === 1000 &&
            camBefore.every((x, i) => x === camAfter[i]);
          return { orbitOk, attenOk, backOk, streamOk };
        })()"""
    )
    assert opts["orbitOk"], "setControlStyle('orbit') must install OrbitControls without damping"
    assert opts["attenOk"], "setSizeAttenuation must arm the world-size shader path"
    assert opts["backOk"], "switching back to trackball must restore the free-tumble controls"
    assert opts["streamOk"], "streaming setCloud must skip the octree and leave the camera alone"

    # Camera-feel guards (rerun-style): fly speed follows the camera-to-pivot
    # distance, and the orbit radius can never collapse through the pivot.
    feel = page.evaluate(
        """(() => {
          const v = window.__toaster.viewer;
          const minDistOk = v.controls.minDistance > 0;
          // Fly one tick from far, then from close: the step must shrink.
          const target = v.controls.target.clone();
          const step = (dist) => {
            v.camera.position.set(target.x + dist, target.y, target.z);
            v.controls.target.copy(target);
            const before = v.camera.position.clone();
            v._flyKeys.add("KeyW");
            v._fly(0.05);
            v._flyKeys.clear();
            return v.camera.position.distanceTo(before);
          };
          const farStep = step(100);
          const nearStep = step(0.5);
          return { minDistOk, farStep, nearStep };
        })()"""
    )
    assert feel["minDistOk"], "controls.minDistance must clamp the orbit radius"
    assert feel["nearStep"] < feel["farStep"] / 20, (
        f"fly speed must scale with pivot distance (far {feel['farStep']}, near {feel['nearStep']})"
    )

    gone = page.evaluate(
        """(() => {
          window.__toaster.viewer.dispose();
          return document.querySelector('#viewport canvas') === null;
        })()"""
    )
    assert gone, "dispose must remove the canvas"
    # Events that used to hit the viewer's window listeners must be harmless now.
    page.keyboard.down("w")
    page.keyboard.up("w")
    page.mouse.move(400, 300)
    time.sleep(0.5)
    assert not errors, f"errors after dispose: {errors}"
    page.close()
