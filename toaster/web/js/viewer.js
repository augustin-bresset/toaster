// Three.js point-cloud viewport: shader points (per-point colour + visibility
// alpha), free-tumble (trackball) camera, screen-space picking and rubber-band
// box select.

import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";

// Above this many points, camera motion is rendered decimated; the full cloud
// is redrawn the moment the camera settles. Only the draw list changes —
// attributes (and CPU picking) always see every point, so nothing is ever
// decimated while actually labelling. The decimation itself is an octree cut
// (frustum-culled, dense near the camera, thinning with distance) built in a
// worker; a pre-shuffled random subset covers the seconds until it's ready.
const LOD_BUDGET = 1_000_000;
// How long after the last camera move the view still counts as "in motion".
const LOD_SETTLE_MS = 150;

// Build the octree for any cloud bigger than this: even below the motion-LOD
// threshold it makes picking O(nodes-near-cursor) instead of O(n).
const OCTREE_MIN_POINTS = 200_000;
// Keep descending into a node while its projected diameter exceeds this many
// pixels — below it, the node's own sample is already ~1 point per pixel.
const OCTREE_SPLIT_PX = 110;
// Bounding-sphere radius of a node = half-edge * sqrt(3).
const SQRT3 = Math.sqrt(3);

// Orbit-pivot crosshair (rerun-style "what am I centred on" cue): stays fully
// visible for this long after the last camera move, then fades out — makes the
// TrackballControls pivot legible instead of an invisible point somewhere in
// the cloud, which is what made fast tumbling feel disorienting.
const ORBIT_INDICATOR_LINGER_MS = 350;
const ORBIT_INDICATOR_FADE_MS = 100;
// Crosshair half-length as a fraction of the current camera-to-pivot distance,
// so it reads the same size on screen whether you're framing the whole cloud
// or zoomed into one corner.
const ORBIT_INDICATOR_SIZE = 0.03;

function smoothstep(edge0, edge1, x) {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

const VERT = `
  attribute vec3 acolor;
  attribute float aalpha;
  uniform float uSize;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vColor = acolor;
    vAlpha = aalpha;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uSize;
  }`;

const FRAG = `
  varying vec3 vColor;
  varying float vAlpha;
  uniform float uRound;
  uniform float uSize;
  // Outline width as a fraction of the point's half-width/radius — relative,
  // not pixel-based, so it stays a thin sliver at any point size — but
  // clamped below to a fixed pixel budget so it doesn't grow into a fat,
  // ugly ring once the point is scaled way up.
  uniform float uOutline;
  // Hard cap on the rim's on-screen width, in pixels.
  uniform float uOutlineMaxPx;
  void main() {
    if (vAlpha < 0.5) discard;
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d);   // 0 at centre, 0.5 at a round point's silhouette
    float rNorm = r * 2.0; // 0..1 across the radius

    // Centre-to-rim highlight — the same falloff re_renderer (rerun's point
    // shader) uses for its points/spheres — gives each point a soft 3D "bead"
    // look instead of a flat disc, which does most of the work of making
    // overlapping points readable.
    float shade = max(0.4, sqrt(max(0.0, 1.2 - rNorm)));
    vec3 shaded = vColor * shade;

    // uOutline is a fraction of the half-width (uSize / 2 px); re-derive the
    // fraction that a uOutlineMaxPx-wide rim would need at the current point
    // size, and never exceed it.
    float outline = uSize > 0.0 ? min(uOutline, uOutlineMaxPx * 2.0 / uSize) : uOutline;

    if (uRound > 0.5) {
      float aa = fwidth(r);
      float coverage = 1.0 - smoothstep(0.5 - aa, 0.5 + aa, r);
      if (coverage <= 0.0) discard;
      float rim = smoothstep(0.5 - outline - aa, 0.5 - outline + aa, r);
      gl_FragColor = vec4(mix(shaded, vec3(0.0), rim), coverage);
    } else {
      vec2 e = min(gl_PointCoord, 1.0 - gl_PointCoord);
      float rim = outline > 0.0 ? step(min(e.x, e.y), outline) : 0.0;
      gl_FragColor = vec4(mix(shaded, vec3(0.0), rim), 1.0);
    }
  }`;

// Ground-plane reference grid: a single shader-lit quad rather than
// GridHelper's fixed line list, so it can do three things GridHelper can't —
// anti-aliased 1 m minor lines, brighter/thicker 10 m major lines, and a soft
// fade instead of a hard edge.
const GRID_VERT = `
  varying vec3 vWorldPos;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }`;

const GRID_FRAG = `
  varying vec3 vWorldPos;
  uniform vec3 uMinorColor;
  uniform vec3 uMajorColor;
  uniform float uCellSize;     // minor line spacing (1 m)
  uniform float uSectionSize;  // major line spacing (10 m)
  uniform float uOpacity;
  uniform float uFadeDistance; // camera distance the grid is lost by
  uniform float uPlaneHalf;    // plane half-extent, for a matching edge fade

  // Anti-aliased grid-line coverage at the given coord spacing and line
  // width (in line-widths). Fades itself to 0 once a cell covers less than
  // ~a screen pixel — past that point the lines would alias into flat grey,
  // so this spacing simply drops out and only a coarser one (called
  // separately, with a bigger width) keeps reading as lines.
  float gridLine(vec2 coord, float width) {
    vec2 deriv = fwidth(coord);
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / max(deriv * width, 1e-6);
    float line = 1.0 - clamp(min(grid.x, grid.y), 0.0, 1.0);
    float fade = 1.0 - clamp(max(deriv.x, deriv.y) - 1.0, 0.0, 1.0);
    return line * fade;
  }

  void main() {
    float minor = gridLine(vWorldPos.xy / uCellSize, 1.0);
    float major = gridLine(vWorldPos.xy / uSectionSize, 1.8);
    float coverage = max(minor, major);
    if (coverage <= 0.0) discard;
    vec3 color = mix(uMinorColor, uMajorColor, major);

    // Lost-in-the-distance fade: from the camera (so it reads whichever way
    // you look, not just radially from the grid's centre) and a matching
    // fade toward the plane's own edge so that boundary never shows as a hard
    // cutoff either.
    float camFade = 1.0 - smoothstep(uFadeDistance * 0.4, uFadeDistance, distance(vWorldPos, cameraPosition));
    float edgeFade = 1.0 - smoothstep(uPlaneHalf * 0.7, uPlaneHalf, length(vWorldPos.xy));

    float alpha = coverage * uOpacity * mix(1.0, 1.5, major) * camFade * edgeFade;
    if (alpha <= 0.002) discard;
    gl_FragColor = vec4(color, alpha);
  }`;

export class Viewer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1f2430);
    this.camera = new THREE.PerspectiveCamera(55, this._aspect(), 0.01, 100000);
    this.camera.up.set(0, 0, 1); // Z-up, the natural convention for lidar
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);
    // TrackballControls (not OrbitControls): the cloud tumbles freely in any
    // direction with no locked up-axis, so a scan that isn't gravity-aligned
    // (e.g. a forest-path lidar frame) can be turned to any angle — OrbitControls
    // would wall off at the poles ("blocked at a plane").
    this.controls = new TrackballControls(this.camera, this.renderer.domElement);
    this.controls.staticMoving = true; // no inertia drift — crisp, predictable
    // TrackballControls ships hidden modifier keys (A/S/D force a drag to
    // rotate/zoom/pan). Those are our fly keys: flying with S or D while
    // orbiting turned the drag into abrupt zooms/pans. Neutralize them.
    this.controls.keys = ["", "", ""];
    // Render on demand: the rAF loop only runs while something can still move
    // the view (a drag in progress, a fly key held, or a pending dirty frame)
    // and stops entirely otherwise. Merely returning early from a 60 fps rAF
    // callback is not enough: each scheduled frame keeps Chromium's BeginFrame
    // pipeline alive, and under QtWebEngine's Vulkan fallback that leaks Oilpan
    // memory until the renderer OOMs after ~15 minutes of idling.
    this._dirty = true;
    this._rafPending = false;
    this._interacting = false; // between TrackballControls "start" and "end"
    this.controls.addEventListener("start", () => {
      this._interacting = true;
      this._schedule();
    });
    this.controls.addEventListener("end", () => {
      this._interacting = false;
      this._requestRender();
    });
    // A drag can be interrupted without ever delivering a "pointerup" to the
    // canvas — losing window focus mid-drag (alt-tab, a native dialog) is the
    // common case in the desktop shell's embedded webview. Without this, the
    // "end" event above never fires, _interacting stays stuck true, and the
    // rAF loop below spins forever instead of parking — which is exactly the
    // Vulkan-fallback Oilpan leak this render-on-demand scheme exists to avoid.
    window.addEventListener("blur", () => {
      this._interacting = false;
      this._requestRender();
    });
    // "change" fires on every actual camera move (drag, wheel zoom, fly — the
    // controls detect external position changes in update() — and the arrow-key
    // rotations). That is the motion signal for the LOD: a plain click fires
    // "start" but no "change", so selecting never blinks the subset in.
    this._motionUntil = 0;
    this._motionLod = false; // motion frames render decimated (n > LOD_BUDGET)
    this._lodIndex = null; // random-subset fallback until the octree is built
    this._lodOn = false; // the frame currently on screen was drawn decimated
    this._octree = null; // worker-built node table + point permutation
    this._octreeWorker = null;
    this._drawAttr = null; // reusable index attribute the octree cut writes into
    this._orbitIndicatorUntil = 0; // linger deadline; fade-out runs for ORBIT_INDICATOR_FADE_MS past it
    this._orbitFadeIn = false;
    this._orbitFadeChangeTime = 0;
    this.controls.addEventListener("change", () => {
      this._motionUntil = performance.now() + LOD_SETTLE_MS;
      this._orbitIndicatorUntil = performance.now() + ORBIT_INDICATOR_LINGER_MS;
      this._requestRender();
    });
    this.controls.rotateSpeed = 3.0;
    this.controls.zoomSpeed = 1.2;
    this.controls.panSpeed = 0.8;
    this._boxMode = false;
    // Box mode uses LEFT for the rubber band, so the camera moves on RIGHT
    // (orbit) / MIDDLE (pan). Holding Shift turns a RIGHT-drag into a pan too —
    // TrackballControls' button map has no modifier support, so we follow Shift.
    window.addEventListener("keydown", (e) => e.key === "Shift" && this._applyMouseButtons(true));
    window.addEventListener("keyup", (e) => e.key === "Shift" && this._applyMouseButtons(false));

    // Fly navigation on physical WASD + QE (`e.code`, so layout-independent):
    // forward / back along the view axis, strafe left / right, and Q/E down /
    // up rerun-style along the screen's vertical. Keydown/keyup only track
    // which keys are held; the motion itself runs per-frame in _tick so it is
    // smooth and diagonals (e.g. forward+left) combine.
    this._flyKeys = new Set();
    this._shiftDown = false;
    this._radius = 10; // scene scale — refreshed by frame()
    this._clock = new THREE.Clock();
    window.addEventListener("keydown", (e) => this._flyKey(e, true));
    window.addEventListener("keyup", (e) => this._flyKey(e, false));
    window.addEventListener("blur", () => this._flyKeys.clear());

    this.geom = null;
    this.points = null;
    this.highlight = null;
    this.grid = null;
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uSize: { value: 2.0 },
        uRound: { value: 0.0 },
        uOutline: { value: 0.06 },
        uOutlineMaxPx: { value: 1.25 },
      },
      vertexShader: VERT,
      fragmentShader: FRAG,
    });

    this._buildOrbitIndicator();

    window.addEventListener("resize", () => this._resize());
    this._requestRender();
  }

  // Mark the view dirty and make sure a frame is scheduled to draw it.
  _requestRender() {
    this._dirty = true;
    this._schedule();
  }

  _schedule() {
    if (this._rafPending) return;
    this._rafPending = true;
    requestAnimationFrame(() => this._tick());
  }

  setCloud(xyz) {
    if (this.points) {
      this.scene.remove(this.points);
      this.geom.dispose();
    }
    const n = xyz.length / 3;
    this.geom = new THREE.BufferGeometry();
    this.geom.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
    this.geom.setAttribute("acolor", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    this.geom.setAttribute("aalpha", new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
    this._lodOn = false;
    this._lodIndex = null;
    this._octree = null;
    this._drawAttr = null;
    this._motionLod = n > LOD_BUDGET;
    if (this._octreeWorker) {
      this._octreeWorker.terminate();
      this._octreeWorker = null;
    }
    if (this._motionLod) {
      // Partial Fisher-Yates: the first LOD_BUDGET slots become a uniform random
      // sample, so drawing them alone still shows the whole scene, just sparser.
      // This is only the stopgap for the second or two the octree takes to build.
      const idx = new Uint32Array(n);
      for (let i = 0; i < n; i++) idx[i] = i;
      for (let i = 0; i < LOD_BUDGET; i++) {
        const j = i + Math.floor(Math.random() * (n - i));
        const t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
      }
      this._lodIndex = new THREE.BufferAttribute(idx.slice(0, LOD_BUDGET), 1);
    }
    if (n > OCTREE_MIN_POINTS) this._buildOctree(xyz, n);
    this.points = new THREE.Points(this.geom, this.material);
    this.scene.add(this.points);
    this.frame();
    this._buildGrid(this._radius);
  }

  // A faint 1 m / 10 m reference grid on the world XY plane (Z = 0), scaled to
  // the cloud so it reads as ground scale rather than an arbitrary fixed
  // extent. It fades out — toward the camera and toward its own edge —
  // instead of stopping abruptly, and its 1 m lines self-fade once the camera
  // pulls back far enough that they'd alias, leaving only the thicker 10 m
  // lines standing.
  _buildGrid(radius) {
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.geometry.dispose();
      this.grid.material.dispose();
      this.grid = null;
    }
    const fadeDistance = Math.max(radius * 4, 20);
    const half = fadeDistance * 1.15;
    const geom = new THREE.PlaneGeometry(half * 2, half * 2);
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uMinorColor: { value: new THREE.Color(0x333c4d) },
        uMajorColor: { value: new THREE.Color(0x5a6478) },
        uCellSize: { value: 1.0 },
        uSectionSize: { value: 10.0 },
        uOpacity: { value: 0.35 },
        uFadeDistance: { value: fadeDistance },
        uPlaneHalf: { value: half },
      },
      vertexShader: GRID_VERT,
      fragmentShader: GRID_FRAG,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      extensions: { derivatives: true },
    });
    // PlaneGeometry already lies flat in the local XY plane (normal +Z) —
    // no rotation needed for our Z-up world, unlike GridHelper's default XZ.
    const grid = new THREE.Mesh(geom, material);
    this.scene.add(grid);
    this.grid = grid;
  }

  // The rerun-style orbit-pivot crosshair: three short segments centred on
  // controls.target, fixed to world axes (not the camera's, which can roll
  // freely under TrackballControls) so it also doubles as a "how tilted am I"
  // reference. Built once; _tickOrbitIndicator() rewrites its 6 vertices and
  // fades it in/out every frame it's active.
  _buildOrbitIndicator() {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6 * 3), 3));
    geom.setDrawRange(0, 6);
    const material = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.85,
      // Depth-tested on purpose (like rerun): the cloud occludes the crosshair,
      // so how much of it shows tells you how deep the pivot sits in the scene.
      depthWrite: false,
    });
    this._orbitIndicator = new THREE.LineSegments(geom, material);
    this._orbitIndicator.visible = false;
    this._orbitIndicator.frustumCulled = false; // its extent doesn't track the geometry's bounding sphere
    this.scene.add(this._orbitIndicator);
  }

  // Updates and fades the orbit-pivot crosshair for the current frame. Returns
  // true while it still needs the render loop kept alive (mid-fade or lingering
  // after the last camera move), false once it's fully settled/hidden.
  _tickOrbitIndicator(now) {
    const active = now < this._orbitIndicatorUntil + ORBIT_INDICATOR_FADE_MS;
    if (!active) {
      if (this._orbitIndicator.visible) {
        this._orbitIndicator.visible = false;
        this._dirty = true;
      }
      return false;
    }
    const showing = now < this._orbitIndicatorUntil;
    if (showing !== this._orbitFadeIn) {
      this._orbitFadeChangeTime = now;
      this._orbitFadeIn = showing;
    }
    const elapsed = now - this._orbitFadeChangeTime;
    const fade = this._orbitFadeIn
      ? smoothstep(0, ORBIT_INDICATOR_FADE_MS, elapsed)
      : smoothstep(ORBIT_INDICATOR_FADE_MS, 0, elapsed);

    this._orbitIndicator.visible = fade > 0.001;
    if (this._orbitIndicator.visible) {
      const target = this.controls.target;
      const half = this.camera.position.distanceTo(target) * ORBIT_INDICATOR_SIZE * fade;
      const pos = this._orbitIndicator.geometry.attributes.position.array;
      // Up: half-length, drawn upward only (mirrors rerun — reads as "ground"
      // without a stray line poking below the pivot). Right/forward: full
      // length, both ways.
      pos[0] = target.x; pos[1] = target.y; pos[2] = target.z;
      pos[3] = target.x; pos[4] = target.y; pos[5] = target.z + half * 0.5;
      pos[6] = target.x - half; pos[7] = target.y; pos[8] = target.z;
      pos[9] = target.x + half; pos[10] = target.y; pos[11] = target.z;
      pos[12] = target.x; pos[13] = target.y - half; pos[14] = target.z;
      pos[15] = target.x; pos[16] = target.y + half; pos[17] = target.z;
      this._orbitIndicator.geometry.attributes.position.needsUpdate = true;
    }
    this._dirty = true;
    return true;
  }

  setColors(colors, alpha) {
    const ca = this.geom.getAttribute("acolor");
    ca.array.set(colors);
    ca.needsUpdate = true;
    const aa = this.geom.getAttribute("aalpha");
    aa.array.set(alpha);
    aa.needsUpdate = true;
    this._requestRender();
  }

  setHighlight(indices, xyz) {
    if (this.highlight) {
      this.scene.remove(this.highlight);
      this.highlight.geometry.dispose();
      this.highlight.material.dispose(); // each call makes a new material — leaks its GL program otherwise
      this.highlight = null;
    }
    if (!indices || indices.length === 0) return;
    const pos = new Float32Array(indices.length * 3);
    for (let i = 0; i < indices.length; i++) {
      const p = indices[i] * 3;
      pos[i * 3] = xyz[p];
      pos[i * 3 + 1] = xyz[p + 1];
      pos[i * 3 + 2] = xyz[p + 2];
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const m = new THREE.PointsMaterial({
      color: 0xf4d35e,
      size: this.material.uniforms.uSize.value + 4,
      sizeAttenuation: false,
      depthTest: false,
    });
    this.highlight = new THREE.Points(g, m);
    this.scene.add(this.highlight);
    this._requestRender();
  }

  setPointSize(s) {
    this.material.uniforms.uSize.value = s;
    this._requestRender();
  }
  setRound(on) {
    this.material.uniforms.uRound.value = on ? 1 : 0;
    this._requestRender();
  }
  setBackground(hex) {
    this.scene.background = new THREE.Color(hex);
    this._requestRender();
  }
  setControlsEnabled(on) {
    this.controls.enabled = on;
  }

  // Box mode frees the LEFT button for the rubber band, so the camera moves on
  // RIGHT (orbit), MIDDLE (pan) and the wheel (zoom). Holding Shift turns a
  // RIGHT-drag into a pan as well — the same lateral move a plain right-drag
  // gives in point/voxel mode.
  setBoxMode(on) {
    this._boxMode = on;
    this.controls.enabled = true;
    this._applyMouseButtons(false);
  }

  _flyKey(e, down) {
    this._shiftDown = e.shiftKey;
    if (!["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"].includes(e.code)) return;
    if (down) {
      // Chorded shortcuts are not fly input: on AZERTY, Ctrl+Z (undo) is the
      // physical KeyW — without this guard every undo lurched the camera forward.
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      if (!this._rafPending) this._clock.getDelta(); // discard idle time — no jump on the first step
      this._flyKeys.add(e.code);
      this._schedule(); // the loop is parked while idle — restart it for the fly motion
    } else {
      // Always release, even if focus moved to an input mid-hold — otherwise
      // the key would stick and the camera would drift forever.
      this._flyKeys.delete(e.code);
    }
  }

  // Move the camera by the fly keys currently held. The orbit target shifts by
  // the same vector, so the pivot stays in front of the camera and the next
  // mouse-drag orbits around where the user is now looking. Speed scales with
  // the scene radius (like near/far in frame()); Shift boosts it.
  _fly(dt) {
    if (this._flyKeys.size === 0) return;
    const has = (c) => this._flyKeys.has(c);
    const fwd = this.camera.getWorldDirection(new THREE.Vector3());
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion); // screen up
    const dir = new THREE.Vector3();
    if (has("KeyW")) dir.add(fwd);
    if (has("KeyS")) dir.sub(fwd);
    if (has("KeyD")) dir.add(right);
    if (has("KeyA")) dir.sub(right);
    if (has("KeyE")) dir.add(up); // rerun-style: E up, Q down
    if (has("KeyQ")) dir.sub(up);
    if (dir.lengthSq() === 0) return; // opposite keys cancel out
    const speed = this._radius * (this._shiftDown ? 1.5 : 0.5);
    // Cap dt: after the tab was backgrounded the first delta can be huge, and
    // one giant step would teleport the camera out of the scene.
    dir.normalize().multiplyScalar(speed * Math.min(dt, 0.1));
    this.camera.position.add(dir);
    this.controls.target.add(dir);
    this._dirty = true;
  }

  // TrackballControls quirk: `mouseButtons` maps an ACTION (LEFT→rotate,
  // MIDDLE→zoom, RIGHT→pan) to the BUTTON INDEX that triggers it (0=left,
  // 1=middle, 2=right); a value no button has (-1) disables that action.
  _applyMouseButtons(shift) {
    if (!this._boxMode) {
      this.controls.mouseButtons = { LEFT: 0, MIDDLE: 1, RIGHT: 2 }; // rotate / zoom / pan
    } else if (shift) {
      this.controls.mouseButtons = { LEFT: -1, MIDDLE: -1, RIGHT: 2 }; // shift+right → pan
    } else {
      this.controls.mouseButtons = { LEFT: 2, MIDDLE: -1, RIGHT: 1 }; // right → orbit, middle → pan
    }
  }

  // Draw a translucent wireframe cube around each voxel centre (one merged
  // LineSegments — cheap even for tens of thousands of voxels).
  setVoxelGrid(centers, size) {
    this.clearVoxelGrid();
    const n = centers.length / 3;
    if (n === 0) return;
    const h = size / 2;
    const corner = [
      [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
      [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ];
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
      [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    const pos = new Float32Array(n * edges.length * 2 * 3);
    let k = 0;
    for (let i = 0; i < n; i++) {
      const cx = centers[i * 3], cy = centers[i * 3 + 1], cz = centers[i * 3 + 2];
      for (const [a, b] of edges) {
        pos[k++] = cx + corner[a][0]; pos[k++] = cy + corner[a][1]; pos[k++] = cz + corner[a][2];
        pos[k++] = cx + corner[b][0]; pos[k++] = cy + corner[b][1]; pos[k++] = cz + corner[b][2];
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const m = new THREE.LineBasicMaterial({ color: 0xe10600, transparent: true, opacity: 0.25 });
    this.voxelGrid = new THREE.LineSegments(g, m);
    this.scene.add(this.voxelGrid);
    this._requestRender();
  }

  clearVoxelGrid() {
    if (this.voxelGrid) {
      this.scene.remove(this.voxelGrid);
      this.voxelGrid.geometry.dispose();
      this.voxelGrid.material.dispose();
      this.voxelGrid = null;
      this._requestRender();
    }
  }

  // Conservative screen-space bound of a node's bounding sphere: null when the
  // whole sphere is behind the near plane (safe to prune), the string
  // "straddle" when the camera is inside / the sphere crosses the near plane
  // (projection untrustworthy — descend without pruning), else the projected
  // centre in canvas px plus a radius that can only over-estimate.
  _nodeCircle(ni, w, h, fovPx) {
    const oct = this._octree;
    const r = oct.half[ni] * SQRT3;
    const v = this._pickV.set(oct.cx[ni], oct.cy[ni], oct.cz[ni]);
    const zView = -this._pickV2.copy(v).applyMatrix4(this.camera.matrixWorldInverse).z;
    if (zView + r < this.camera.near) return null;
    if (zView - r < this.camera.near) return "straddle";
    v.project(this.camera);
    return {
      sx: (v.x * 0.5 + 0.5) * w,
      sy: (-v.y * 0.5 + 0.5) * h,
      rpx: (r / (zView - r)) * fovPx, // nearest possible depth → max apparent size
    };
  }

  // Test one node slice's points exactly like the brute-force loops below;
  // `perPoint(i, sx, sy)` receives each visible on-screen point.
  _scanSlice(ni, w, h, perPoint) {
    const { order, start, count } = this._octree;
    const pos = this.geom.getAttribute("position").array;
    const alpha = this.geom.getAttribute("aalpha").array;
    const v = this._pickV;
    const end = start[ni] + count[ni];
    for (let k = start[ni]; k < end; k++) {
      const i = order[k];
      if (alpha[i] < 0.5) continue;
      v.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).project(this.camera);
      if (v.z < -1 || v.z > 1) continue;
      perPoint(i, (v.x * 0.5 + 0.5) * w, (-v.y * 0.5 + 0.5) * h);
    }
  }

  // Depth-first octree walk that prunes subtrees whose screen bound fails
  // `hit(sx, sy, rpx)`; every surviving node's slice goes through perPoint.
  _pickWalk(w, h, hit, perPoint) {
    const { children } = this._octree;
    const fovPx = h / 2 / Math.tan((this.camera.fov * Math.PI) / 360);
    const stack = [0];
    while (stack.length > 0) {
      const ni = stack.pop();
      const c = this._nodeCircle(ni, w, h, fovPx);
      if (c === null) continue;
      if (c !== "straddle" && !hit(c.sx, c.sy, c.rpx)) continue;
      this._scanSlice(ni, w, h, perPoint);
      const cb = ni * 8;
      for (let k = 0; k < 8; k++) if (children[cb + k] >= 0) stack.push(children[cb + k]);
    }
  }

  // Nearest *visible* point to a screen position, or -1. Robust (no raycaster
  // threshold tuning). With the octree: only nodes whose screen bound overlaps
  // the cursor are visited; without it: projects every point once per click.
  pick(clientX, clientY) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const mx = clientX - rect.left, my = clientY - rect.top;
    const w = rect.width, h = rect.height;
    let best = -1, bestD = 14 * 14;
    if (this._octree) {
      if (!this._pickV) { this._pickV = new THREE.Vector3(); this._pickV2 = new THREE.Vector3(); }
      this._pickWalk(
        w, h,
        (sx, sy, rpx) => {
          const dx = sx - mx, dy = sy - my;
          const reach = rpx + 14;
          return dx * dx + dy * dy <= reach * reach;
        },
        (i, sx, sy) => {
          const dx = sx - mx, dy = sy - my;
          const d = dx * dx + dy * dy;
          if (d < bestD) { bestD = d; best = i; }
        },
      );
      return best;
    }
    const pos = this.geom.getAttribute("position").array;
    const alpha = this.geom.getAttribute("aalpha").array;
    const v = new THREE.Vector3();
    for (let i = 0; i < alpha.length; i++) {
      if (alpha[i] < 0.5) continue;
      v.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).project(this.camera);
      if (v.z < -1 || v.z > 1) continue;
      const dx = (v.x * 0.5 + 0.5) * w - mx;
      const dy = (-v.y * 0.5 + 0.5) * h - my;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  // Indices of visible points inside a screen rectangle (CSS px relative to viewport).
  pickBox(x0, y0, x1, y1) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const lo = [Math.min(x0, x1), Math.min(y0, y1)];
    const hi = [Math.max(x0, x1), Math.max(y0, y1)];
    const w = rect.width, h = rect.height;
    const out = [];
    if (this._octree) {
      if (!this._pickV) { this._pickV = new THREE.Vector3(); this._pickV2 = new THREE.Vector3(); }
      this._pickWalk(
        w, h,
        (sx, sy, rpx) => {
          // Distance from the node's screen circle to the rect: prune only
          // when even the circle's closest approach misses the rect.
          const dx = Math.max(lo[0] - sx, 0, sx - hi[0]);
          const dy = Math.max(lo[1] - sy, 0, sy - hi[1]);
          return dx * dx + dy * dy <= rpx * rpx;
        },
        (i, sx, sy) => {
          if (sx >= lo[0] && sx <= hi[0] && sy >= lo[1] && sy <= hi[1]) out.push(i);
        },
      );
      return out;
    }
    const pos = this.geom.getAttribute("position").array;
    const alpha = this.geom.getAttribute("aalpha").array;
    const v = new THREE.Vector3();
    for (let i = 0; i < alpha.length; i++) {
      if (alpha[i] < 0.5) continue;
      v.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).project(this.camera);
      if (v.z < -1 || v.z > 1) continue;
      const sx = (v.x * 0.5 + 0.5) * w, sy = (-v.y * 0.5 + 0.5) * h;
      if (sx >= lo[0] && sx <= hi[0] && sy >= lo[1] && sy <= hi[1]) out.push(i);
    }
    return out;
  }

  frame() {
    this.geom.computeBoundingSphere();
    const s = this.geom.boundingSphere;
    this._radius = s.radius;
    this.controls.target.copy(s.center);
    // 3/4 aerial view for a Z-up scene (above, and to the side). Tumbling and
    // the arrow-key rotations change camera.up — restore it too, or a "reset"
    // after rolling the scene would keep the roll.
    this.camera.up.set(0, 0, 1);
    const off = new THREE.Vector3(1.3, -1.3, 0.9).multiplyScalar(s.radius);
    this.camera.position.copy(s.center.clone().add(off));
    this.camera.near = Math.max(s.radius / 1000, 0.001);
    this.camera.far = s.radius * 50;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this._requestRender();
  }

  // The world-space "up" the current view shows (the camera's local +Y) — i.e.
  // which way is up on screen right now. Lets a ground filter know where gravity
  // is once a not-level scan has been turned the right way round.
  worldUp() {
    const v = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    return [v.x, v.y, v.z];
  }

  // Step-rotate the view around the target by `angle` (radians): "roll" spins
  // about the view axis (camera stays put), "pitch" tips over the screen's
  // horizontal axis, "yaw" turns about the screen's vertical axis.
  rotateView(kind, angle) {
    const cam = this.camera;
    const target = this.controls.target;
    let axis;
    if (kind === "roll") axis = new THREE.Vector3().subVectors(target, cam.position).normalize();
    else if (kind === "pitch") axis = new THREE.Vector3(1, 0, 0).applyQuaternion(cam.quaternion);
    else axis = new THREE.Vector3(0, 1, 0).applyQuaternion(cam.quaternion); // yaw
    const q = new THREE.Quaternion().setFromAxisAngle(axis, angle);
    const offset = cam.position.clone().sub(target).applyQuaternion(q);
    cam.position.copy(target).add(offset);
    cam.up.applyQuaternion(q);
    cam.lookAt(target);
    this.controls.update();
    this._requestRender();
  }

  _aspect() {
    return this.container.clientWidth / Math.max(1, this.container.clientHeight);
  }
  _resize() {
    this.camera.aspect = this._aspect();
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.controls.handleResize(); // TrackballControls caches the canvas rect for its math
    this._requestRender();
  }
  // Hand the cloud to the octree worker; adopt the result when it lands.
  // The xyz copy's buffer is transferred, so the only real cost here is one
  // memcpy — the build itself happens off-thread.
  _buildOctree(xyz, n) {
    const worker = new Worker(new URL("./octree-worker.js", import.meta.url), { type: "module" });
    this._octreeWorker = worker;
    worker.onmessage = (e) => {
      worker.terminate();
      if (worker !== this._octreeWorker) return; // a newer cloud superseded this build
      this._octreeWorker = null;
      this._octree = e.data;
      // The octree replaces the random fallback outright; free its 4 MB/M.
      this._lodIndex = null;
      // The cut writes into one preallocated index attribute. Capacity: the
      // budget plus one worst-case node (traversal stops *after* the node that
      // crosses the budget), clamped to the cloud itself.
      const capacity = Math.min(n, LOD_BUDGET + 16384);
      this._drawAttr = new THREE.BufferAttribute(new Uint32Array(capacity), 1);
      this._drawAttr.setUsage(THREE.DynamicDrawUsage);
      console.log(`octree: ${e.data.start.length} nodes over ${n} points in ${e.data.buildMs} ms`);
    };
    const copy = xyz.slice();
    worker.postMessage({ xyz: copy, n }, [copy.buffer]);
  }

  // Breadth-first cut through the octree for the current camera: skip nodes
  // outside the frustum, draw every visited node's own slice, and descend only
  // while a node still covers more than OCTREE_SPLIT_PX on screen. BFS order
  // means coarse coverage lands before fine detail, so running out of budget
  // degrades resolution, never coverage.
  _octreeCut() {
    const oct = this._octree;
    const { cx, cy, cz, half, start, count, children, order } = oct;
    if (!this._frustum) {
      this._frustum = new THREE.Frustum();
      this._projMat = new THREE.Matrix4();
      this._sphere = new THREE.Sphere();
    }
    // The cut runs before render(), so the camera's world matrix may not have
    // caught up with this frame's controls.update() yet.
    this.camera.updateMatrixWorld();
    this._projMat.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    this._frustum.setFromProjectionMatrix(this._projMat);
    const camPos = this.camera.position;
    const fovPx =
      this.container.clientHeight / 2 / Math.tan((this.camera.fov * Math.PI) / 360);

    const dst = this._drawAttr.array;
    const capacity = dst.length;
    const queue = [0];
    let used = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const ni = queue[qi];
      const r = half[ni] * SQRT3;
      this._sphere.center.set(cx[ni], cy[ni], cz[ni]);
      this._sphere.radius = r;
      if (!this._frustum.intersectsSphere(this._sphere)) continue;
      const cnt = count[ni];
      if (used + cnt > capacity) break;
      dst.set(order.subarray(start[ni], start[ni] + cnt), used);
      used += cnt;
      const dist = camPos.distanceTo(this._sphere.center) - r;
      if (dist <= 0 || (2 * r * fovPx) / dist > OCTREE_SPLIT_PX) {
        const cb = ni * 8;
        for (let c = 0; c < 8; c++) if (children[cb + c] >= 0) queue.push(children[cb + c]);
      }
    }
    return used;
  }

  // Swap the draw list between the full cloud and a decimated one — the octree
  // cut when it's ready, the pre-shuffled random subset until then.
  _applyLod(on) {
    if (!this.geom) return;
    if (on && this._octree && this._drawAttr) {
      const used = this._octreeCut();
      if (this._drawAttr.addUpdateRange) {
        this._drawAttr.clearUpdateRanges();
        this._drawAttr.addUpdateRange(0, used);
      }
      this._drawAttr.needsUpdate = true;
      if (this.geom.index !== this._drawAttr) this.geom.setIndex(this._drawAttr);
      this.geom.setDrawRange(0, used);
      this._lodOn = true;
      return;
    }
    const want = on ? this._lodIndex : null;
    if (this.geom.index === want && this._lodOn === (want !== null)) return;
    this.geom.setIndex(want);
    this.geom.setDrawRange(0, Infinity);
    this._lodOn = want !== null;
  }

  _tick() {
    this._rafPending = false;
    this._fly(this._clock.getDelta());
    this.controls.update(); // fires "change" (→ dirty + motion window) when the camera moved
    const now = performance.now();
    const moving = this._motionLod && (this._flyKeys.size > 0 || now < this._motionUntil);
    const orbitIndicatorActive = this._tickOrbitIndicator(now);
    if (this._dirty) {
      this._dirty = false;
      this._applyLod(moving);
      this.renderer.render(this.scene, this.camera);
    } else if (this._lodOn && !moving) {
      // The camera settled with a decimated frame on screen — redraw it full.
      this._applyLod(false);
      this.renderer.render(this.scene, this.camera);
    }
    // Keep looping while something can still move the camera, while a
    // decimated frame is showing (its full-res refine is still owed), or while
    // the orbit-pivot crosshair is still lingering/fading; otherwise the loop
    // dies here and _requestRender()/_schedule() restarts it.
    if (this._interacting || this._flyKeys.size > 0 || this._dirty || this._lodOn || orbitIndicatorActive)
      this._schedule();
  }
}
