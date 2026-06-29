// Compute per-point colours from the semantic state the API returns — the same
// job toaster.viewer.colormap does in Python, here on the client so the wire only
// carries labels/grouping, never colour buffers. Alpha is kept at 1 for every
// point (nothing is hidden); de-emphasised segments are greyed, not removed.

// Kept in sync with toaster/viewer/colormap.py (_GROUP_PALETTE).
const GROUP_PALETTE = [
  [51, 34, 136], [136, 204, 238], [68, 170, 153], [17, 119, 51],
  [153, 153, 51], [221, 204, 119], [204, 102, 119], [136, 34, 85],
  [170, 68, 153], [119, 170, 221], [102, 204, 170], [187, 204, 51],
  [238, 221, 136], [255, 170, 187], [153, 221, 255], [170, 170, 170],
];
const NOISE = [74, 80, 92];
const UNKNOWN = [60, 64, 74];
const DIM = [92, 96, 106]; // greyed-out (non-focused) segment — visible, not hidden

function setRGB(out, i, c) {
  out[i * 3] = c[0] / 255;
  out[i * 3 + 1] = c[1] / 255;
  out[i * 3 + 2] = c[2] / 255;
}

function rampInto(out, vals) {
  let lo = Infinity, hi = -Infinity;
  for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const d = hi - lo || 1;
  for (let i = 0; i < vals.length; i++) setRGB(out, i, turbo((vals[i] - lo) / d));
}

function turbo(t) {
  t = Math.min(1, Math.max(0, t));
  const stops = [[40, 60, 180], [40, 180, 120], [230, 200, 40], [220, 60, 40]];
  const x = t * (stops.length - 1), i = Math.floor(x), f = x - i;
  const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// decoded: { snapshot, labels:Int32Array, grouping:Int32Array|null }
// cloud:   { xyz:Float32Array, features:{ intensity?:Float32Array } }
export function computeColors(decoded, cloud) {
  const snap = decoded.snapshot;
  const labels = decoded.labels;
  const grouping = decoded.grouping;
  const n = labels.length;
  const colors = new Float32Array(n * 3);
  const alpha = new Float32Array(n).fill(1); // every point stays visible & pickable

  const mode = snap.display_mode;
  if (mode === "grouping" && grouping) {
    // Segments toggled off (Hide all / Solo / a row's checkbox) are not removed —
    // they go grey, so the focused segment(s) pop while the rest stay as context.
    const hidden = new Set(snap.segments.filter((s) => !s.visible).map((s) => s.id));
    for (let i = 0; i < n; i++) {
      const g = grouping[i];
      if (hidden.has(g)) setRGB(colors, i, DIM);
      else setRGB(colors, i, g < 0 ? NOISE : GROUP_PALETTE[g % GROUP_PALETTE.length]);
    }
  } else if (mode === "intensity" && cloud.features.intensity) {
    rampInto(colors, cloud.features.intensity);
  } else if (mode === "height") {
    const z = new Float32Array(n);
    for (let i = 0; i < n; i++) z[i] = cloud.xyz[i * 3 + 2];
    rampInto(colors, z);
  } else {
    const lut = {};
    for (const c of snap.classes) lut[c.id] = c.color;
    for (let i = 0; i < n; i++) setRGB(colors, i, lut[labels[i]] || UNKNOWN);
  }
  return { colors, alpha };
}

export function groupColor(id) {
  return id < 0 ? NOISE : GROUP_PALETTE[id % GROUP_PALETTE.length];
}
