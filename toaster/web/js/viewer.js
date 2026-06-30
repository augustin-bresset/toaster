// Three.js point-cloud viewport: shader points (per-point colour + visibility
// alpha), free-tumble (trackball) camera, screen-space picking and rubber-band
// box select.

import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";

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
  void main() {
    if (vAlpha < 0.5) discard;
    if (uRound > 0.5) {
      vec2 d = gl_PointCoord - vec2(0.5);
      if (dot(d, d) > 0.25) discard;
    }
    gl_FragColor = vec4(vColor, 1.0);
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
    this.controls.rotateSpeed = 3.0;
    this.controls.zoomSpeed = 1.2;
    this.controls.panSpeed = 0.8;
    this._boxMode = false;
    // Box mode uses LEFT for the rubber band, so the camera moves on RIGHT
    // (orbit) / MIDDLE (pan). Holding Shift turns a RIGHT-drag into a pan too —
    // TrackballControls' button map has no modifier support, so we follow Shift.
    window.addEventListener("keydown", (e) => e.key === "Shift" && this._applyMouseButtons(true));
    window.addEventListener("keyup", (e) => e.key === "Shift" && this._applyMouseButtons(false));

    this.geom = null;
    this.points = null;
    this.highlight = null;
    this.material = new THREE.ShaderMaterial({
      uniforms: { uSize: { value: 3.0 }, uRound: { value: 0.0 } },
      vertexShader: VERT,
      fragmentShader: FRAG,
    });

    window.addEventListener("resize", () => this._resize());
    this._tick();
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
    this.points = new THREE.Points(this.geom, this.material);
    this.scene.add(this.points);
    this.frame();
  }

  setColors(colors, alpha) {
    const ca = this.geom.getAttribute("acolor");
    ca.array.set(colors);
    ca.needsUpdate = true;
    const aa = this.geom.getAttribute("aalpha");
    aa.array.set(alpha);
    aa.needsUpdate = true;
  }

  setHighlight(indices, xyz) {
    if (this.highlight) {
      this.scene.remove(this.highlight);
      this.highlight.geometry.dispose();
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
  }

  setPointSize(s) {
    this.material.uniforms.uSize.value = s;
  }
  setRound(on) {
    this.material.uniforms.uRound.value = on ? 1 : 0;
  }
  setBackground(hex) {
    this.scene.background = new THREE.Color(hex);
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
  }

  clearVoxelGrid() {
    if (this.voxelGrid) {
      this.scene.remove(this.voxelGrid);
      this.voxelGrid.geometry.dispose();
      this.voxelGrid = null;
    }
  }

  // Nearest *visible* point to a screen position, or -1. Robust (no raycaster
  // threshold tuning): projects every point once per click.
  pick(clientX, clientY) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const mx = clientX - rect.left, my = clientY - rect.top;
    const pos = this.geom.getAttribute("position").array;
    const alpha = this.geom.getAttribute("aalpha").array;
    const v = new THREE.Vector3();
    const w = rect.width, h = rect.height;
    let best = -1, bestD = 14 * 14;
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
    const pos = this.geom.getAttribute("position").array;
    const alpha = this.geom.getAttribute("aalpha").array;
    const v = new THREE.Vector3();
    const w = rect.width, h = rect.height;
    const out = [];
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
    this.controls.target.copy(s.center);
    // 3/4 aerial view for a Z-up scene (above, and to the side).
    const off = new THREE.Vector3(1.3, -1.3, 0.9).multiplyScalar(s.radius);
    this.camera.position.copy(s.center.clone().add(off));
    this.camera.near = Math.max(s.radius / 1000, 0.001);
    this.camera.far = s.radius * 50;
    this.camera.updateProjectionMatrix();
    this.controls.update();
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
  }

  _aspect() {
    return this.container.clientWidth / Math.max(1, this.container.clientHeight);
  }
  _resize() {
    this.camera.aspect = this._aspect();
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.controls.handleResize(); // TrackballControls caches the canvas rect for its math
  }
  _tick() {
    requestAnimationFrame(() => this._tick());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
