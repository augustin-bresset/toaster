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
let boxRect = null; // the drawn box (client coords) kept on screen so a double-click inside it labels
const inRect = (e, r) => e.clientX >= r.l && e.clientX <= r.r && e.clientY >= r.t && e.clientY <= r.b;
function clearBox() {
  boxRect = null;
  el("rubber").style.display = "none";
}
let topZ = 10;
let segSpecs = {}; // segmenter name -> [{name, type, default, min, max, step}]
let segGravity = {}; // segmenter name -> bool (accepts an "up" gravity vector)
let voxel = { size: 0.5, showEmpty: false, map: null, centers: new Float32Array(0) };
const VOXEL_GRID_CAP = 30000; // max cubes to draw / cells to enumerate

// -- themes ------------------------------------------------------------------

const THEME_BG = { toaster: 0x1f2430, cafe: 0x181310, arcade: 0x050507 };

function applyTheme(name) {
  if (!THEME_BG[name]) name = "toaster";
  document.body.dataset.theme = name;
  el("theme").value = name;
  try {
    localStorage.setItem("toaster-theme", name);
  } catch {
    /* private mode */
  }
  viewer.setBackground(THEME_BG[name]);
}

function initTheme() {
  let saved = "toaster";
  try {
    saved = localStorage.getItem("toaster-theme") || "toaster";
  } catch {
    /* ignore */
  }
  applyTheme(saved);
  el("theme").onchange = () => applyTheme(el("theme").value);
}

const theme = () => document.body.dataset.theme;

// A short-lived element that animates then removes itself.
function flash(className, text, x, y) {
  const t = document.createElement("div");
  t.className = className;
  t.textContent = text;
  if (x != null) {
    t.style.left = x + "px";
    t.style.top = y + "px";
  }
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 1200);
}

// Café Toaster: a toast jumps out when a grouping is produced.
function toastPop() {
  if (theme() === "cafe") flash("toast-pop", "🍞");
}

// Café Toaster: the toaster logo "presses" its lever on Run.
function logoPress() {
  if (theme() !== "cafe") return;
  const logo = document.querySelector(".logo");
  logo.classList.remove("press");
  void logo.offsetWidth; // restart the animation
  logo.classList.add("press");
}

// Café Toaster: a little "Ding!" on save (+ a soft beep).
function dingPop() {
  if (theme() !== "cafe") return;
  flash("ding-pop", "🔔 Ding!");
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.frequency.value = 880;
    o.connect(g);
    g.connect(ctx.destination);
    g.gain.setValueAtTime(0.12, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
    o.start();
    o.stop(ctx.currentTime + 0.3);
  } catch {
    /* audio not allowed */
  }
}

// Arcade: score pop on a label, with a combo when you label quickly.
let _combo = 0;
let _comboAt = 0;
function scorePop(points, x, y) {
  if (theme() !== "arcade") return;
  const now = performance.now();
  _combo = now - _comboAt < 2500 ? _combo + 1 : 1;
  _comboAt = now;
  const label = `+${points} PTS!` + (_combo > 1 ? `  COMBO x${_combo}` : "");
  flash("score-pop", label, (x ?? window.innerWidth / 2) + 12, y ?? window.innerHeight / 2);
}

// Arcade: "LEVEL UP!" when a segmenter produces clusters.
function levelUp() {
  if (theme() === "arcade") flash("levelup", "LEVEL UP!");
}

const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
const hex = (c) => "#" + c.map((x) => x.toString(16).padStart(2, "0")).join("");
const modifiers = (e) => [e.shiftKey && "shift", e.ctrlKey && "ctrl"].filter(Boolean);

boot();

async function boot() {
  const meta = await api.meta();
  segSpecs = {};
  segGravity = {};
  for (const s of meta.segmenters) {
    segSpecs[s.name] = s.params;
    segGravity[s.name] = s.gravity;
  }
  el("seg-name").innerHTML = meta.segmenters.map((s) => `<option>${s.name}</option>`).join("");
  renderSegParams(el("seg-name").value);
  buildModes();
  wire();
  initTheme();
  if (meta.n > 0) await loadCloud();
  else {
    el("status").textContent = "no cloud — pick one";
    openBrowser(); // launched without a file → let the user find one in a folder
  }
}

// Render the parameter fields for the selected segmenter from its spec.
function renderSegParams(name) {
  const box = el("seg-params");
  box.innerHTML = "";
  for (const p of segSpecs[name] || []) {
    const row = document.createElement("label");
    row.className = "row";
    const input = document.createElement("input");
    input.type = "number";
    input.dataset.param = p.name;
    input.dataset.ptype = p.type;
    input.value = p.default;
    if (p.min != null) input.min = p.min;
    if (p.max != null) input.max = p.max;
    input.step = p.step != null ? p.step : p.type === "int" ? 1 : "any";
    row.append(document.createTextNode(p.name + " "), input);
    box.appendChild(row);
  }
  // Ground filters: let the user say "the camera is looking the right way up, so
  // use its up as gravity" — sent as the `up` param instead of the cloud's +Z.
  if (segGravity[name]) {
    const row = document.createElement("label");
    row.className = "row";
    row.title = "Tumble the view until the scene looks upright, then tick this so the filter uses your view's up as gravity";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.id = "seg-gravity";
    row.append(cb, document.createTextNode(" Use camera view as up (gravity)"));
    box.appendChild(row);
  }
}

async function loadCloud() {
  const c = await api.cloud();
  cloud.xyz = decodeArray(c.xyz);
  cloud.features = {};
  for (const [k, v] of Object.entries(c.features)) cloud.features[k] = decodeArray(v);
  viewer.setCloud(cloud.xyz);
  applyState(await api.state());
}

// -- file browser (Open without a path, then find the cloud in a folder) ----

let openDir = null; // the folder currently shown in the browser
let openExts = []; // extensions a loader can open (from /api/browse), e.g. [".ply", ".bin"]

async function openBrowser(path) {
  let data;
  try {
    data = await api.browse(path);
  } catch (e) {
    el("status").textContent = "✗ " + e.message;
    return;
  }
  openDir = data.path;
  openExts = data.extensions || openExts;
  el("open-input").value = data.path;
  renderEntries(data, "");
  const w = el("win-open");
  w.style.display = "flex";
  w.style.zIndex = ++topZ;
  el("open-input").focus();
}

// Render a folder's entries, optionally keeping only those starting with `filter`.
function renderEntries(data, filter) {
  const list = el("open-list");
  list.innerHTML = "";
  if (data.parent && !filter) list.appendChild(browseRow("..", () => openBrowser(data.parent)));
  let shown = 0;
  for (const e of data.entries) {
    if (filter && !e.name.startsWith(filter)) continue;
    shown++;
    if (e.is_dir) list.appendChild(browseRow(e.name + "/", () => openBrowser(e.path)));
    else if (e.openable) list.appendChild(browseRow(e.name, () => openFile(e.path)));
    else list.appendChild(browseRow(e.name, null)); // unsupported format
  }
  if (shown === 0) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.style.padding = "6px";
    empty.textContent = filter ? "no match" : "empty folder";
    list.appendChild(empty);
  }
}

const joinPath = (base, name) => (base.endsWith("/") ? base + name : base + "/" + name);

// Tab-completion of the typed path, shell-style: complete a lone match fully
// (drilling into a folder), or to the longest common prefix when several match.
async function completePath() {
  const input = el("open-input");
  const val = input.value;
  const slash = val.lastIndexOf("/");
  const dirPart = slash < 0 ? openDir : slash === 0 ? "/" : val.slice(0, slash);
  const partial = val.slice(slash + 1);
  let data;
  try {
    data = await api.browse(dirPart);
  } catch {
    return;
  }
  const matches = data.entries.filter((e) => e.name.startsWith(partial));
  if (matches.length === 0) return renderEntries(data, partial); // shows "no match"
  if (matches.length === 1) {
    const m = matches[0];
    if (m.is_dir) {
      const sub = await api.browse(m.path);
      openDir = sub.path;
      input.value = sub.path.endsWith("/") ? sub.path : sub.path + "/";
      renderEntries(sub, "");
    } else {
      input.value = joinPath(data.path, m.name);
      renderEntries(data, m.name);
    }
    return;
  }
  let lcp = matches[0].name;
  for (const m of matches) while (!m.name.startsWith(lcp)) lcp = lcp.slice(0, -1);
  input.value = joinPath(data.path, lcp);
  renderEntries(data, lcp);
}

// Enter on the path field: open it as a file if it ends with a loadable
// extension, otherwise browse into it as a folder.
function submitPath() {
  const val = el("open-input").value.trim();
  if (!val) return;
  const lower = val.toLowerCase();
  if (openExts.some((x) => lower.endsWith(x))) openFile(val);
  else openBrowser(val);
}

function browseRow(label, onClick) {
  const row = document.createElement("div");
  row.className = "item";
  row.textContent = label;
  if (onClick) row.onclick = onClick;
  else row.style.opacity = "0.4";
  return row;
}

async function openFile(path) {
  el("status").textContent = "opening…";
  try {
    await api.open(path); // starts a fresh server-side session
    el("win-open").style.display = "none";
    el("win-groups").style.display = "none";
    currentGroup = null;
    clearBox();
    await loadCloud();
    if (pickMode === "voxel") rebuildVoxels();
    el("status").textContent = "opened " + (path.split("/").pop() || path);
  } catch (e) {
    el("status").textContent = "✗ " + e.message;
  }
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
  renderInspector();
  const s = state.snapshot;
  el("status").textContent =
    `Sel: ${state.selection.length.toLocaleString()} · Class: ${className(s.active_class)} · View: ${s.display_mode}`;
}

const className = (id) => (state.snapshot.classes.find((c) => c.id === id) || {}).name || String(id);
const classColor = (id) =>
  (state.snapshot.classes.find((c) => c.id === id) || { color: [120, 120, 120] }).color;

// Show details of the current selection: a single point's label + position, or
// the count and per-label distribution of a multi-point (box) selection.
function renderInspector() {
  const box = el("inspector");
  const sel = state.selection;
  const labels = state.labels;
  if (sel.length === 0) {
    box.innerHTML = '<div class="muted">No selection</div>';
    return;
  }
  if (sel.length === 1) {
    const i = sel[0];
    const x = cloud.xyz;
    box.innerHTML =
      `<div>Point <b>#${i}</b></div>` +
      `<div class="kv">pos <span class="count">${x[i * 3].toFixed(2)}, ${x[i * 3 + 1].toFixed(2)}, ${x[i * 3 + 2].toFixed(2)}</span></div>` +
      `<div class="kv"><span class="swatch" style="background:${rgb(classColor(labels[i]))}"></span>${className(labels[i])}</div>`;
    return;
  }
  const counts = {};
  for (let k = 0; k < sel.length; k++) counts[labels[sel[k]]] = (counts[labels[sel[k]]] || 0) + 1;
  const rows = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(
      ([id, c]) =>
        `<div class="kv"><span class="swatch" style="background:${rgb(classColor(+id))}"></span>${className(+id)}<span class="count">${c.toLocaleString()}</span></div>`,
    )
    .join("");
  box.innerHTML = `<div><b>${sel.length.toLocaleString()}</b> points selected</div>${rows}`;
}

// -- panels -----------------------------------------------------------------

function renderClasses() {
  const box = el("classes");
  box.innerHTML = "";
  for (const c of state.snapshot.classes) {
    const row = document.createElement("div");
    row.className = "item" + (c.id === state.snapshot.active_class ? " active" : "");
    row.innerHTML = `<span class="swatch" style="background:${rgb(c.color)}"></span>${c.id}  ${c.name}`;
    row.onclick = () => api.activeClass(c.id).then(applyState);
    box.appendChild(row);
  }
  renderClassManager();
}

// The "Manage classes" window: an editable colour per row (+ add/rename/remove).
function renderClassManager() {
  const box = el("class-manager");
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
  const all = state.snapshot.segments;
  const CAP = 400; // a voxel/k-means grouping can have thousands; keep the DOM light
  const segs = all.slice(0, CAP);
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
    label.textContent = `#${seg.id}`;
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
  if (all.length > segs.length) {
    const more = document.createElement("div");
    more.className = "muted";
    more.style.padding = "4px 6px";
    more.textContent = `… ${(all.length - segs.length).toLocaleString()} more (click points in the cloud)`;
    box.appendChild(more);
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

// -- voxel selection mode ---------------------------------------------------

function buildVoxels() {
  const s = voxel.size;
  const xyz = cloud.xyz;
  const n = xyz.length / 3;
  const map = new Map(); // "ix,iy,iz" -> [point indices]
  for (let i = 0; i < n; i++) {
    const key =
      Math.floor(xyz[i * 3] / s) + "," + Math.floor(xyz[i * 3 + 1] / s) + "," + Math.floor(xyz[i * 3 + 2] / s);
    let arr = map.get(key);
    if (!arr) map.set(key, (arr = []));
    arr.push(i);
  }
  voxel.map = map;

  let cells;
  if (voxel.showEmpty) {
    cells = allCellsInBBox(s);
  } else {
    cells = [];
    for (const key of map.keys()) {
      const [ix, iy, iz] = key.split(",").map(Number);
      cells.push((ix + 0.5) * s, (iy + 0.5) * s, (iz + 0.5) * s);
    }
  }
  voxel.centers = new Float32Array(cells);
}

function allCellsInBBox(s) {
  const xyz = cloud.xyz;
  const n = xyz.length / 3;
  let lo = [Infinity, Infinity, Infinity];
  let hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++) {
      const v = xyz[i * 3 + a];
      if (v < lo[a]) lo[a] = v;
      if (v > hi[a]) hi[a] = v;
    }
  const i0 = lo.map((v) => Math.floor(v / s));
  const i1 = hi.map((v) => Math.floor(v / s));
  const total = (i1[0] - i0[0] + 1) * (i1[1] - i0[1] + 1) * (i1[2] - i0[2] + 1);
  if (total > VOXEL_GRID_CAP) return []; // too many — caller hides the grid
  const cells = [];
  for (let i = i0[0]; i <= i1[0]; i++)
    for (let j = i0[1]; j <= i1[1]; j++)
      for (let k = i0[2]; k <= i1[2]; k++) cells.push((i + 0.5) * s, (j + 0.5) * s, (k + 0.5) * s);
  return cells;
}

function voxelIndicesOf(i) {
  const s = voxel.size;
  const xyz = cloud.xyz;
  const key =
    Math.floor(xyz[i * 3] / s) + "," + Math.floor(xyz[i * 3 + 1] / s) + "," + Math.floor(xyz[i * 3 + 2] / s);
  return (voxel.map && voxel.map.get(key)) || [i];
}

function rebuildVoxels() {
  if (!cloud.xyz) return;
  buildVoxels();
  const count = voxel.centers.length / 3;
  if (el("vox-grid").checked && count > 0) viewer.setVoxelGrid(voxel.centers, voxel.size);
  else viewer.clearVoxelGrid();
  const occupied = voxel.map ? voxel.map.size : 0;
  let info = `${occupied.toLocaleString()} occupied voxels`;
  if (count === 0 && el("vox-grid").checked) info += " · grid too large to show";
  el("vox-info").textContent = info;
}

function enterVoxelMode() {
  const w = el("win-voxel");
  w.style.display = "flex";
  w.style.zIndex = ++topZ;
  rebuildVoxels();
}

function exitVoxelMode() {
  el("win-voxel").style.display = "none";
  viewer.clearVoxelGrid();
}

// -- events -----------------------------------------------------------------

function wire() {
  el("psize").oninput = (e) => viewer.setPointSize(+e.target.value);
  el("round").onchange = (e) => viewer.setRound(e.target.checked);
  el("seg-name").onchange = () => renderSegParams(el("seg-name").value);
  el("seg-run").onclick = runSegmenter;
  el("open-file").onclick = () => openBrowser(openDir);
  el("open-input").addEventListener("keydown", (e) => {
    if (e.key === "Tab") {
      e.preventDefault();
      completePath();
    } else if (e.key === "Enter") {
      e.preventDefault();
      submitPath();
    } else if (e.key === "Escape") {
      el("win-open").style.display = "none";
    }
  });
  el("grp-showall").onclick = () => api.groupsShowAll().then(applyState);
  el("grp-hideall").onclick = () => api.groupsHideAll().then(applyState);
  // Assign the active class to every checked (visible) segment at once.
  el("grp-assign").onclick = () =>
    api.groupsAssignVisible().then((s) => {
      applyState(s);
      buzz("flicker");
    });
  // Closing the Segments window discards the (transient) segmentation entirely.
  el("grp-close").onclick = () => {
    currentGroup = null;
    el("win-groups").style.display = "none";
    api.clearGrouping().then(applyState);
  };

  el("vox-size").onchange = (e) => {
    voxel.size = Math.max(0.05, +e.target.value || 0.5);
    rebuildVoxels();
  };
  el("vox-grid").onchange = rebuildVoxels;
  el("vox-empty").onchange = (e) => {
    voxel.showEmpty = e.target.checked;
    rebuildVoxels();
  };

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
  document.querySelectorAll("[data-open]").forEach((b) => {
    b.onclick = () => {
      const w = el(b.dataset.open);
      w.style.display = "flex";
      w.style.zIndex = ++topZ;
    };
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

// Make the top neon "bug": a quick flicker on a label, a bigger glitch when a
// segmenter finishes. kind = "flicker" | "glitch".
function buzz(kind) {
  const tb = el("toolbar");
  tb.classList.remove("flicker", "glitch");
  void tb.offsetWidth; // restart the animation if it's mid-run
  tb.classList.add(kind);
}

async function act(name) {
  if (!state) return; // no cloud yet — nothing to assign / undo / save
  if (name === "assign") {
    const n = state ? state.selection.length : 0;
    applyState(await api.assign());
    if (n > 0) {
      buzz("flicker"); // the neon stutters each time you stamp a label
      scorePop(n * 10); // arcade score (no-op in other themes)
    }
  } else if (name === "undo") applyState(await api.undo());
  else if (name === "redo") applyState(await api.redo());
  else if (name === "save") {
    const r = await api.save();
    el("status").textContent = "saved " + r.saved.split("/").pop();
    dingPop(); // café "Ding!"
  }
}

// Clear the current selection (no-op / no round-trip if nothing is selected).
function clearSelectionIfAny() {
  clearBox(); // a drawn box is part of the selection's visual — drop it too
  if (state && state.selection.length > 0) api.clearSelection().then(applyState);
}

function onKey(e) {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;

  // Arrow keys step-rotate the view: ←/→ roll, ↑/↓ pitch, Shift+←/→ yaw — so a
  // tilted scan can be turned to any angle without dragging. Hold to repeat.
  if (e.key.startsWith("Arrow")) {
    const STEP = Math.PI / 36; // 5° per press
    if (e.key === "ArrowLeft") viewer.rotateView(e.shiftKey ? "yaw" : "roll", STEP);
    else if (e.key === "ArrowRight") viewer.rotateView(e.shiftKey ? "yaw" : "roll", -STEP);
    else if (e.key === "ArrowUp") viewer.rotateView("pitch", STEP);
    else if (e.key === "ArrowDown") viewer.rotateView("pitch", -STEP);
    e.preventDefault();
    return;
  }

  if (!state) return; // shortcuts do nothing until a cloud is loaded
  if (e.key === "Enter") act("assign");
  else if (e.key === "Escape") clearSelectionIfAny();
  else if (e.ctrlKey && e.key === "z") act("undo");
  else if (e.ctrlKey && (e.key === "y" || (e.shiftKey && e.key === "Z"))) act("redo");
  else if (/^[0-9]$/.test(e.key)) {
    const c = state?.snapshot.classes[+e.key];
    if (c) api.activeClass(c.id).then(() => act("assign"));
  }
}

async function runSegmenter() {
  if (!state) {
    el("status").textContent = "open a cloud first";
    return;
  }
  const n = el("seg-name").value;
  const params = {};
  for (const inp of el("seg-params").querySelectorAll("[data-param]")) {
    const v = +inp.value;
    params[inp.dataset.param] = inp.dataset.ptype === "int" ? Math.round(v) : v;
  }
  const grav = el("seg-params").querySelector("#seg-gravity");
  if (grav && grav.checked) params.up = viewer.worldUp(); // current view's up = gravity
  const btn = el("seg-run");
  btn.disabled = true;
  btn.classList.add("busy");
  btn.textContent = "Running…";
  el("status").textContent = `running ${n}…`;
  setLoading(true);
  logoPress();
  const t0 = performance.now();
  try {
    applyState(await api.segment(n, params, el("seg-scope").checked));
    showGroupsWindow();
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    const g = state.snapshot.active_grouping;
    const detail = g ? `${g.source}: ${g.n_groups} segments · ${dt}s` : "done";
    el("status").textContent =
      theme() === "cafe"
        ? `🍞 Perfectly toasted — ${detail}`
        : theme() === "arcade"
          ? `LEVEL UP! ${detail}`
          : `✓ ${detail}`;
    buzz("glitch"); // the neon glitches hard when the run lands
    toastPop();
    levelUp();
  } catch (e) {
    el("status").textContent =
      theme() === "cafe" ? `🔥 Burnt! ${e.message}` : "✗ segmentation failed: " + e.message;
  } finally {
    setLoading(false);
    btn.disabled = false;
    btn.classList.remove("busy");
    btn.textContent = "Run";
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
  clearBox(); // a box from a previous session in box mode shouldn't linger
  if (mode === "voxel") enterVoxelMode();
  else exitVoxelMode();
}

function setupPointer() {
  const dom = viewer.renderer.domElement;
  const rubber = el("rubber");
  let down = null;

  dom.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    down = { x: e.clientX, y: e.clientY, moved: false };
  });

  dom.addEventListener("pointermove", (e) => {
    if (!down) return;
    if (Math.abs(e.clientX - down.x) + Math.abs(e.clientY - down.y) > 4) down.moved = true;
    // Draw the rubber band only once it's an actual drag, so a plain click in
    // box mode leaves the previously-drawn box on screen.
    if (pickMode === "box" && down.moved) {
      rubber.style.display = "block";
      updateRubber(rubber, down, { x: e.clientX, y: e.clientY });
    }
  });

  window.addEventListener("pointerup", (e) => {
    if (!down || !state) return;
    const start = down;
    down = null;
    const mods = modifiers(e);

    if (pickMode === "box") {
      if (start.moved) {
        // Box drag -> frustum select, and KEEP the box drawn so you can
        // double-click inside it to label the whole selection.
        const r = dom.getBoundingClientRect();
        const idx = viewer.pickBox(start.x - r.left, start.y - r.top, e.clientX - r.left, e.clientY - r.top);
        if (idx.length) {
          boxRect = { l: Math.min(start.x, e.clientX), t: Math.min(start.y, e.clientY),
                      r: Math.max(start.x, e.clientX), b: Math.max(start.y, e.clientY) };  // fmt: skip
          api.box(idx, mods).then(applyState);
        } else {
          clearBox(); // an empty drag draws no box
        }
        return;
      }
      // A plain click inside the box is the first half of a double-click to
      // label — keep everything. A click elsewhere dismisses the box.
      if (!(boxRect && inRect(e, boxRect)) && mods.length === 0) clearSelectionIfAny();
      return;
    }

    if (start.moved) return; // a drag = orbit, never a selection

    // A click: hit a point, or — on empty space with no modifier — deselect.
    const i = viewer.pick(e.clientX, e.clientY);
    if (i < 0) {
      if (mods.length === 0) clearSelectionIfAny();
      return;
    }
    if (pickMode === "voxel") {
      api.box(voxelIndicesOf(i), mods).then(applyState); // whole voxel
    } else {
      // If a grouping is active, remember which segment was clicked so the
      // Segments window scrolls to and flashes it after the re-render.
      if (state.grouping) {
        currentGroup = state.grouping[i];
        focusGroup = currentGroup;
      }
      api.pick(i, mods).then(applyState);
    }
  });

  // Double-click = label in one gesture: select what a click would select
  // (the whole group if a grouping is active, the voxel in voxel mode, else the
  // single point) and stamp it with the active class. Works the same in point,
  // box and voxel modes. Bound to BOTH buttons — a plain left double-click, and
  // a right double-click (browser menu suppressed) for users who reach for the
  // right button; neither moves the camera or disturbs the left-click selection.
  dom.addEventListener("dblclick", labelUnderCursor);
  let lastRight = 0;
  dom.addEventListener("contextmenu", (e) => {
    e.preventDefault(); // no browser context menu over the cloud
    const now = performance.now();
    if (now - lastRight < 400) {
      lastRight = 0;
      labelUnderCursor(e);
    } else {
      lastRight = now;
    }
  });
}

async function labelUnderCursor(e) {
  if (!state) return;
  // Box mode: a double-click inside the drawn box labels the whole box selection.
  if (pickMode === "box") {
    if (boxRect && inRect(e, boxRect) && state.selection.length) {
      await act("assign");
      clearBox();
    }
    return;
  }
  const i = viewer.pick(e.clientX, e.clientY);
  if (i < 0) return;
  applyState(pickMode === "voxel" ? await api.box(voxelIndicesOf(i), []) : await api.pick(i, []));
  await act("assign");
}

function updateRubber(rubber, a, b) {
  rubber.style.left = Math.min(a.x, b.x) + "px";
  rubber.style.top = Math.min(a.y, b.y) + "px";
  rubber.style.width = Math.abs(a.x - b.x) + "px";
  rubber.style.height = Math.abs(a.y - b.y) + "px";
}
