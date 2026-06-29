// REST client for the Toaster API, plus the {dtype, shape, data} array decoder.

const CTORS = {
  "<f4": Float32Array,
  "<f8": Float64Array,
  "<i4": Int32Array,
  "<i2": Int16Array,
  "<u1": Uint8Array,
  "|u1": Uint8Array,
};

export function decodeArray(payload) {
  if (!payload) return null;
  const bin = atob(payload.data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const Ctor = CTORS[payload.dtype];
  if (!Ctor) throw new Error("unsupported dtype " + payload.dtype);
  return new Ctor(bytes.buffer);
}

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

export const api = {
  meta: () => jget("/api/meta"),
  cloud: () => jget("/api/cloud"),
  state: () => jget("/api/state"),
  browse: (path = null) => jget("/api/browse" + (path ? "?path=" + encodeURIComponent(path) : "")),
  open: (path) => jpost("/api/open", { path }),
  pick: (index, modifiers = []) => jpost("/api/pick", { index, modifiers }),
  box: (indices, modifiers = []) => jpost("/api/box", { indices, modifiers }),
  assign: (class_id = null) => jpost("/api/assign", { class_id }),
  activeClass: (class_id) => jpost("/api/active_class", { class_id }),
  displayMode: (mode) => jpost("/api/display_mode", { mode }),
  undo: () => jpost("/api/undo"),
  redo: () => jpost("/api/redo"),
  clearSelection: () => jpost("/api/clear_selection"),
  save: () => jpost("/api/save"),
  segment: (name, params, scope_to_selection) =>
    jpost("/api/segment", { name, params, scope_to_selection }),
  groupSelect: (group_id, modifiers = []) =>
    jpost("/api/group/select", { group_id, modifiers }),
  groupAssign: (group_id, class_id = null) =>
    jpost("/api/group/assign", { group_id, class_id }),
  groupSuggested: (group_id = null) => jpost("/api/group/suggested", { group_id }),
  groupVisibility: (group_id, visible) =>
    jpost("/api/group/visibility", { group_id, visible }),
  groupsAssignVisible: (class_id = null) => jpost("/api/groups/assign_visible", { class_id }),
  groupsShowAll: () => jpost("/api/groups/show_all"),
  groupsHideAll: () => jpost("/api/groups/hide_all"),
  clearGrouping: () => jpost("/api/grouping/clear"),
  classAdd: (name, color = null) => jpost("/api/class/add", { name, color }),
  classRename: (class_id, name) => jpost("/api/class/rename", { class_id, name }),
  classColor: (class_id, color) => jpost("/api/class/color", { class_id, color }),
  classRemove: (class_id) => jpost("/api/class/remove", { class_id }),
};
