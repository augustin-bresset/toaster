// Orchestrates the web front: load the cloud, render panels from the snapshot,
// wire interactions to the API, and re-render from each returned state.

import { api, decodeArray } from "./api.js";
import { computeColors } from "./colors.js";
import { Viewer } from "./viewer.js";

const el = (id) => document.getElementById(id);
const viewer = new Viewer(el("viewport"));

let cloud = { xyz: null, features: {} };
let state = null; // decoded { snapshot, labels, grouping, selection }
let pickMode = "point";
let currentGroup = null;
let focusGroup = null; // a group to scroll/flash in the Segments window after a re-render
let topZ = 10;

const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
const hex = (c) => "#" + c.map((x) => x.toString(16).padStart(2, "0")).join("");
const modifiers = (e) => [e.shiftKey && "shift", e.ctrlKey && "ctrl"].filter(Boolean);

boot();

async function boot() {
  const meta = await api.meta();
  el("seg-name").innerHTML = meta.segmenters.map((s) => `<option>${s}</option>`).join("");
  buildModes();
  wire();
  if (meta.n > 0) await loadCloud();
  else el("status").textContent = "no cloud — start: toaster-web <file>";
}

async function loadCloud() {
  const c = await api.cloud();
  cloud.xyz = decodeArray(c.xyz);
  cloud.features = {};
  for (const [k, v] of Object.entries(c.features)) cloud.features[k] = decodeArray(v);
  viewer.setCloud(cloud.xyz);
  applyState(await api.state());
}

function applyState(raw) {
  state = {
    snapshot: raw.snapshot,
    labels: decodeArray(raw.labels),
    grouping: decodeArray(raw.grouping),
    selection: decodeArray(raw.selection),
  };
  const { colors, alpha } = computeColors(state, cloud);
  viewer.setColors(colors, alpha);
  viewer.setHighlight(state.selection, cloud.xyz);
  renderClasses();
  renderModes();
  renderGroups();
  const s = state.snapshot;
  el("status").textContent =
    `Sel: ${state.selection.length.toLocaleString()} · Class: ${className(s.active_class)} · View: ${s.display_mode}`;
}

const className = (id) => (state.snapshot.classes.find((c) => c.id === id) || {}).name || String(id);

// -- panels -----------------------------------------------------------------

function renderClasses() {
  const box = el("classes");
  box.innerHTML = "";
  for (const c of state.snapshot.classes) {
    const row = document.createElement("div");
    row.className = "item" + (c.id === state.snapshot.active_class ? " active" : "");
    const color = document.createElement("input");
    color.type = "color";
    color.className = "swatch-input";
    color.value = hex(c.color);
    color.title = "Change colour";
    color.onclick = (e) => e.stopPropagation();
    color.oninput = () => api.classColor(c.id, color.value).then(applyState);
    const label = document.createElement("span");
    label.textContent = `${c.id}  ${c.name}`;
    row.append(color, label);
    row.onclick = () => api.activeClass(c.id).then(applyState);
    box.appendChild(row);
  }
}

function buildModes() {
  el("modes").innerHTML = ["labels", "grouping", "intensity", "height"]
    .map((m) => `<button data-mode="${m}" class="toggle">${m}</button>`)
    .join("");
  el("modes").querySelectorAll("[data-mode]").forEach((b) => {
    b.onclick = () => api.displayMode(b.dataset.mode).then(applyState);
  });
}

function renderModes() {
  const m = state ? state.snapshot.display_mode : "labels";
  el("modes").querySelectorAll("[data-mode]").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === m);
  });
}

function renderGroups() {
  const box = el("groups");
  box.innerHTML = "";
  const segs = state.snapshot.segments;
  el("groups-header").textContent = state.snapshot.active_grouping
    ? `${state.snapshot.active_grouping.n_groups} segments · ${state.snapshot.active_grouping.source}`
    : "Segments — run a segmenter";
  for (const seg of segs) {
    const row = document.createElement("div");
    row.className = "item" + (seg.id === currentGroup ? " active" : "");
    row.dataset.gid = seg.id;
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = seg.visible;
    cb.onclick = (e) => e.stopPropagation();
    cb.onchange = () => api.groupVisibility(seg.id, cb.checked).then(applyState);
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = rgb(seg.color);
    const label = document.createElement("span");
    label.textContent =
      `#${seg.id}` + (seg.suggested != null ? ` → ${className(seg.suggested)}` : "");
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = seg.count.toLocaleString();
    row.append(cb, sw, label, count);
    row.onclick = () => {
      currentGroup = seg.id;
      api.groupSelect(seg.id).then(applyState);
    };
    box.appendChild(row);
  }

  // A click in the 3-D cloud asked us to focus its segment here: scroll + flash.
  if (focusGroup != null) {
    showGroupsWindow();
    const row = box.querySelector(`[data-gid="${focusGroup}"]`);
    if (row) {
      row.scrollIntoView({ block: "nearest" });
      row.classList.add("flash");
      setTimeout(() => row.classList.remove("flash"), 800);
    }
    focusGroup = null;
  }
}

const withGroup = (fn) => {
  if (currentGroup != null) fn(currentGroup);
};

// -- events -----------------------------------------------------------------

function wire() {
  el("psize").oninput = (e) => viewer.setPointSize(+e.target.value);
  el("round").onchange = (e) => viewer.setRound(e.target.checked);
  el("seg-run").onclick = runSegmenter;
  el("grp-solo").onclick = () => withGroup((g) => api.groupSolo(g).then(applyState));
  el("grp-showall").onclick = () => api.groupsShowAll().then(applyState);
  el("grp-assign").onclick = () => withGroup((g) => api.groupAssign(g).then(applyState));
  el("grp-suggested").onclick = () => withGroup((g) => api.groupSuggested(g).then(applyState));

  el("cls-add").onclick = () => {
    const n = prompt("New class name:");
    if (n && n.trim()) api.classAdd(n.trim()).then(applyState);
  };
  el("cls-rename").onclick = () => {
    const id = state.snapshot.active_class;
    const n = prompt("Rename class:", className(id));
    if (n && n.trim()) api.classRename(id, n.trim()).then(applyState);
  };
  el("cls-remove").onclick = () => {
    const id = state.snapshot.active_class;
    if (id === state.snapshot.unlabeled_id) {
      alert("The 'unlabeled' class cannot be removed.");
      return;
    }
    if (confirm(`Remove class '${className(id)}'? Its points become unlabeled.`))
      api.classRemove(id).then(applyState);
  };

  document.querySelectorAll("[data-act]").forEach((b) => (b.onclick = () => act(b.dataset.act)));
  document.querySelectorAll("[data-mode-pick]").forEach((b) => {
    b.onclick = () => setPickMode(b.dataset.modePick);
  });
  document.querySelectorAll("[data-close]").forEach((b) => {
    b.onclick = () => (el(b.dataset.close).style.display = "none");
  });
  setupWindows();
  setupPointer();
  window.addEventListener("keydown", onKey);
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// Make each floating window draggable by its title bar and raise it on focus.
// Windows are position:fixed, so client coordinates map straight to left/top —
// no offset-parent mismatch. The drag is clamped to the viewport (and below the
// toolbar) so a window can never be lost behind the menu bar.
function setupWindows() {
  const minTop = () => el("loading").getBoundingClientRect().bottom + 4;
  document.querySelectorAll(".window").forEach((win) => {
    win.addEventListener("pointerdown", () => (win.style.zIndex = ++topZ));
    const bar = win.querySelector(".titlebar");
    let dx = 0, dy = 0, dragging = false;
    bar.addEventListener("pointerdown", (e) => {
      if (e.target.closest("button")) return;
      dragging = true;
      const r = win.getBoundingClientRect();
      dx = e.clientX - r.left;
      dy = e.clientY - r.top;
      win.style.right = "auto"; // pin to left/top from now on
      bar.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    bar.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      win.style.left = clamp(e.clientX - dx, 0, window.innerWidth - win.offsetWidth) + "px";
      win.style.top = clamp(e.clientY - dy, minTop(), window.innerHeight - 32) + "px";
    });
    const end = (e) => {
      dragging = false;
      if (bar.hasPointerCapture(e.pointerId)) bar.releasePointerCapture(e.pointerId);
    };
    bar.addEventListener("pointerup", end);
    bar.addEventListener("pointercancel", end);
  });
}

function setLoading(on) {
  el("loading").classList.toggle("on", on);
}

async function act(name) {
  if (name === "assign") applyState(await api.assign());
  else if (name === "undo") applyState(await api.undo());
  else if (name === "redo") applyState(await api.redo());
  else if (name === "save") {
    const r = await api.save();
    el("status").textContent = "saved " + r.saved.split("/").pop();
  }
}

function onKey(e) {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
  if (e.key === "Enter") act("assign");
  else if (e.ctrlKey && e.key === "z") act("undo");
  else if (e.ctrlKey && (e.key === "y" || (e.shiftKey && e.key === "Z"))) act("redo");
  else if (/^[0-9]$/.test(e.key)) {
    const c = state?.snapshot.classes[+e.key];
    if (c) api.activeClass(c.id).then(() => act("assign"));
  }
}

async function runSegmenter() {
  const n = el("seg-name").value;
  const eps = Math.max(0.001, +el("seg-eps").value || 0.5);
  const ms = Math.max(1, Math.round(+el("seg-ms").value || 10));
  const params = n === "dbscan" ? { eps, min_samples: ms } : { min_cluster_size: ms };
  el("status").textContent = `running ${n}…`;
  setLoading(true);
  try {
    applyState(await api.segment(n, params, el("seg-scope").checked));
    showGroupsWindow();
  } catch (e) {
    el("status").textContent = "segmentation failed: " + e.message;
  } finally {
    setLoading(false);
  }
}

// The Segments window appears (and comes to front) when a grouping is produced.
function showGroupsWindow() {
  const win = el("win-groups");
  win.style.display = "flex";
  win.style.zIndex = ++topZ;
}

function setPickMode(mode) {
  pickMode = mode;
  document.querySelectorAll("[data-mode-pick]").forEach((b) => {
    b.classList.toggle("active", b.dataset.modePick === mode);
  });
  viewer.setBoxMode(mode === "box"); // box mode keeps camera on right-drag / wheel
}

function setupPointer() {
  const dom = viewer.renderer.domElement;
  const rubber = el("rubber");
  let down = null;

  dom.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    down = { x: e.clientX, y: e.clientY, moved: false };
    if (pickMode === "box") {
      rubber.style.display = "block";
      updateRubber(rubber, down, down);
    }
  });

  dom.addEventListener("pointermove", (e) => {
    if (!down) return;
    if (Math.abs(e.clientX - down.x) + Math.abs(e.clientY - down.y) > 4) down.moved = true;
    if (pickMode === "box") updateRubber(rubber, down, { x: e.clientX, y: e.clientY });
  });

  window.addEventListener("pointerup", (e) => {
    if (!down || !state) return;
    const start = down;
    down = null;
    rubber.style.display = "none";
    if (pickMode === "box") {
      const r = dom.getBoundingClientRect();
      const idx = viewer.pickBox(start.x - r.left, start.y - r.top, e.clientX - r.left, e.clientY - r.top);
      if (idx.length) api.box(idx, modifiers(e)).then(applyState);
    } else if (!start.moved) {
      const i = viewer.pick(e.clientX, e.clientY);
      if (i >= 0) {
        // If a grouping is active, remember which segment was clicked so the
        // Segments window scrolls to and flashes it after the re-render.
        if (state.grouping) {
          currentGroup = state.grouping[i];
          focusGroup = currentGroup;
        }
        api.pick(i, modifiers(e)).then(applyState);
      }
    }
  });
}

function updateRubber(rubber, a, b) {
  rubber.style.left = Math.min(a.x, b.x) + "px";
  rubber.style.top = Math.min(a.y, b.y) + "px";
  rubber.style.width = Math.abs(a.x - b.x) + "px";
  rubber.style.height = Math.abs(a.y - b.y) + "px";
}
