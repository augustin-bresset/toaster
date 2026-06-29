// Compute per-point colours (and visibility alpha) from the semantic state the
// API returns — the same job toaster.viewer.colormap does in Python, here on the
// client so the wire only carries labels/grouping, never colour buffers.

const GROUP_PALETTE = [
  [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
  [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
  [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
  [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
  [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
];
const NOISE = [90, 90, 90];
const UNKNOWN = [40, 40, 40];

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
  const alpha = new Float32Array(n).fill(1);

  const hidden = new Set(snap.segments.filter((s) => !s.visible).map((s) => s.id));
  if (grouping && hidden.size) {
    for (let i = 0; i < n; i++) if (hidden.has(grouping[i])) alpha[i] = 0;
  }

  const mode = snap.display_mode;
  if (mode === "grouping" && grouping) {
    for (let i = 0; i < n; i++) {
      const g = grouping[i];
      setRGB(colors, i, g < 0 ? NOISE : GROUP_PALETTE[g % GROUP_PALETTE.length]);
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
