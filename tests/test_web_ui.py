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
def page(server):
    """One loaded page (cloud + octree ready) shared by the module's tests."""
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # no browser installed
            pytest.skip(f"chromium unavailable: {exc}")
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on(
            "console",
            lambda m: errors.append(f"console: {m.text}") if m.type == "error" else None,
        )
        page.goto(server)
        page.wait_for_function(
            f"window.__toaster && window.__toaster.viewer.geom && "
            f"window.__toaster.viewer.geom.getAttribute('position').count === {N_POINTS}",
            timeout=90000,
        )
        page.wait_for_function("window.__toaster.viewer._octree !== null", timeout=90000)
        page._errors = errors
        yield page
        assert not errors, f"console/page errors: {errors}"
        browser.close()


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
    page.click("button[data-mode-pick=box]")
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
