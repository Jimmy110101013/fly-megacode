/**
 * The mushroom body as traced morphology, on the GPU.
 *
 * The 2D version rasterised 642,640 line segments on the CPU every time anything
 * changed, which is why it could only afford a cached still image and could not be
 * rotated. Here the whole atlas is one indexed LineSegments buffer: the geometry is
 * uploaded once, the camera is free, and activity is computed in the vertex shader
 * from a small per-cell texture. Per-frame CPU work is a uniform and an 80 kB
 * texture upload, regardless of how many neurites are on screen.
 *
 * Coordinates are FlyWire FAFB units (0.4 um): x medial-lateral, y
 * anterior-posterior, z dorsal-ventral, with z growing toward the calyx.
 */

const CLASS_COLOR = [
  [0.24, 0.86, 0.72],   // KC   teal
  [0.43, 0.59, 1.00],   // MBON periwinkle
  [1.00, 0.30, 0.55],   // DAN  magenta
  [0.69, 0.42, 1.00],   // APL  violet
  [0.45, 0.52, 0.68],   // CX   slate -- the reservoir, quiet until it holds something
];
// Additive blending sums every segment along the view ray, so these are much
// lower than the flat 2D composite needed -- overdraw does the accumulating.
const CLASS_ALPHA = [0.012, 0.013, 0.017, 0.015, 0.009];
// Two activities, two colours: yellow is the mushroom body deciding, cyan-white is
// the central complex holding what has already been done.
const FIRE_COLOR = [1.0, 0.85, 0.36];
const HOLD_COLOR = [0.55, 0.95, 1.0];

const NEUROPIL = {
  optic:     [0.24, 0.41, 0.61], olfactory: [0.20, 0.56, 0.49],
  mushroom:  [0.69, 0.52, 0.25], central:   [0.49, 0.38, 0.66],
  lateral:   [0.31, 0.48, 0.59], ventrolat: [0.27, 0.42, 0.53],
  superior:  [0.41, 0.41, 0.57], inferior:  [0.34, 0.46, 0.52],
  ventromed: [0.30, 0.39, 0.49], periesoph: [0.38, 0.36, 0.45],
  other:     [0.35, 0.38, 0.44],
};

const SWEEP = 0.45;      // seconds for the wavefront to cross a cell
const DECAY = 1.15;      // 1/s fade once the wave has passed

const VERT = `
attribute float aFlow;
attribute float aCell;
attribute float aClass;
uniform sampler2D uCells;
uniform float uCellW;
uniform float uMode;          // 0 = firing, 1 = engram
uniform float uGain;
uniform float uNear;          // view-space depth of the nearest structure
uniform float uRange;
varying vec3 vColor;
varying float vAlpha;

const vec3 KC   = vec3(${CLASS_COLOR[0].join(',')});
const vec3 MBON = vec3(${CLASS_COLOR[1].join(',')});
const vec3 DAN  = vec3(${CLASS_COLOR[2].join(',')});
const vec3 APL  = vec3(${CLASS_COLOR[3].join(',')});
const vec3 CXC  = vec3(${CLASS_COLOR[4].join(',')});
const vec4 BASE = vec4(${CLASS_ALPHA.slice(0, 4).join(',')});
const float BASE_CX = float(${CLASS_ALPHA[4]});
const vec3 HOT  = vec3(${FIRE_COLOR.join(',')});
const vec3 HOLD = vec3(${HOLD_COLOR.join(',')});

void main() {
  int ci = int(aClass + 0.5);
  bool isCX = ci == 4;
  vec3 base = isCX ? CXC : ci == 0 ? KC : ci == 1 ? MBON : ci == 2 ? DAN : APL;
  float a = isCX ? BASE_CX : ci == 0 ? BASE.x : ci == 1 ? BASE.y : ci == 2 ? BASE.z : BASE.w;

  float lit = 0.0;
  if (aCell >= 0.0) {
    vec4 s = texture2D(uCells, vec2((aCell + 0.5) / uCellW, 0.5));
    if (isCX) {
      lit = s.a;                                  // how much this cell is holding
    } else if (uMode > 0.5) {
      lit = s.b;                                  // engram strength
    } else if (s.g > 0.0) {
      // s.r is the wavefront position along the cell, s.g the envelope
      float lead = 1.0 - clamp(abs(s.r - aFlow) * 3.2, 0.0, 1.0);
      lit = aFlow <= s.r + 0.03 ? s.g * (0.34 + 0.66 * lead) : 0.0;
    }
  }
  vColor = mix(base, isCX ? HOLD : HOT, clamp(lit * 1.4, 0.0, 1.0));
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  // Near structures brighter than far ones: without this the projection is a
  // flat mat of lines and rotating it tells you nothing about depth.
  float fade = clamp(1.0 - (-mv.z - uNear) / uRange, 0.18, 1.0);
  vAlpha = (a + lit * 0.42) * uGain * (0.35 + 0.65 * fade);
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = `
precision mediump float;
varying vec3 vColor;
varying float vAlpha;
void main() { gl_FragColor = vec4(vColor * vAlpha, 1.0); }`;

export class Atlas3D {
  /**
   * @param canvas   target
   * @param data     parsed atlas.json
   * @param indexOf  { kc: Map(root_id -> model index), mbon, dan }
   * @param neuropils parsed neuropils.json, optional
   */
  constructor(canvas, data, indexOf, neuropils = null, cxAtlas = null) {
    const THREE = window.THREE;
    this.THREE = THREE;
    this.canvas = canvas;
    this.data = data;
    this.np = neuropils;
    this.cxAtlas = cxAtlas;
    this.showContext = false;
    this.view = 'frontal';
    this.overlay = false;
    this.lastDraw = 0;
    this.gain = 1;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setClearColor(0x05070a, 1);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 1, 20000);

    this._decode();
    this._buildCells(indexOf);
    this._buildLines();
    if (cxAtlas) this._buildCX();
    if (neuropils) this._buildNeuropils();
    this._frame();
    this.setView('frontal');
  }

  /* ------------------------------------------------------------- data ---- */

  _decode() {
    const d = this.data;
    const xyz = Int32Array.from(d.xyz);
    if (d.encoding === 'delta-per-polyline') {
      let at = 0;
      for (const len of d.plens) {
        for (let k = 1; k < len; k++) {
          xyz[at + k * 3] += xyz[at + (k - 1) * 3];
          xyz[at + k * 3 + 1] += xyz[at + (k - 1) * 3 + 1];
          xyz[at + k * 3 + 2] += xyz[at + (k - 1) * 3 + 2];
        }
        at += len * 3;
      }
    }
    this.coords = xyz;
    const b = d.bbox;
    this.center = [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2];
    this.radius = Math.max(b[3] - b[0], b[4] - b[1], b[5] - b[2]) / 2;
    this.flowLo = b[2]; this.flowHi = b[5];
  }

  _buildCells(indexOf) {
    const n = this.data.neurons;
    this.cellOf = new Int32Array(n.length).fill(-1);   // atlas neuron -> model KC
    this.mbonOf = new Int32Array(n.length).fill(-1);
    let maxCell = 0;
    n.forEach((neuron, i) => {
      const k = indexOf.kc.get(neuron.id);
      if (k !== undefined) { this.cellOf[i] = k; maxCell = Math.max(maxCell, k + 1); }
      const m = indexOf.mbon.get(neuron.id);
      if (m !== undefined) this.mbonOf[i] = m;
    });
    this.nKCCells = Math.max(1, maxCell);
    // The texture covers the mushroom body's cells and then the reservoir's, so
    // one lookup serves both populations.
    this.nCX = this.cxAtlas ? this.cxAtlas.neurons.length : 0;
    this.nCells = this.nKCCells + this.nCX;
    this.cellData = new Float32Array(this.nCells * 4);
    this.firedAt = new Float64Array(this.nKCCells);
    this.counts = n.reduce((a, x) => (a[x.s] = (a[x.s] || 0) + 1, a), {});

    const THREE = this.THREE;
    this.cellTex = new THREE.DataTexture(this.cellData, this.nCells, 1,
                                         THREE.RGBAFormat, THREE.FloatType);
    this.cellTex.magFilter = this.cellTex.minFilter = THREE.NearestFilter;
    this.cellTex.needsUpdate = true;
  }

  /** One indexed LineSegments buffer for every neurite in the atlas. */
  _buildLines() {
    const THREE = this.THREE, d = this.data;
    const nV = this.coords.length / 3;
    let nSeg = 0;
    for (const l of d.plens) nSeg += l - 1;

    const pos = new Float32Array(nV * 3);
    const flow = new Float32Array(nV);
    const cell = new Float32Array(nV);
    const cls = new Float32Array(nV);
    const idx = new Uint32Array(nSeg * 2);

    const span = (this.flowHi - this.flowLo) || 1;
    const c = this.center;
    for (let v = 0; v < nV; v++) {
      pos[v * 3] = this.coords[v * 3] - c[0];
      pos[v * 3 + 1] = this.coords[v * 3 + 1] - c[1];
      pos[v * 3 + 2] = this.coords[v * 3 + 2] - c[2];
      flow[v] = (this.flowHi - this.coords[v * 3 + 2]) / span;   // 0 at the calyx
    }

    let vAt = 0, iAt = 0;
    for (let n = 0; n < d.neurons.length; n++) {
      const cellId = this.cellOf[n];
      const klass = d.neurons[n].c;
      for (let li = d.offsets[n]; li < d.offsets[n + 1]; li++) {
        const len = d.plens[li];
        for (let k = 0; k < len; k++) {
          cell[vAt + k] = cellId;
          cls[vAt + k] = klass;
          if (k) { idx[iAt++] = vAt + k - 1; idx[iAt++] = vAt + k; }
        }
        vAt += len;
      }
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aFlow', new THREE.BufferAttribute(flow, 1));
    g.setAttribute('aCell', new THREE.BufferAttribute(cell, 1));
    g.setAttribute('aClass', new THREE.BufferAttribute(cls, 1));
    g.setIndex(new THREE.BufferAttribute(idx, 1));

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uCells: { value: this.cellTex },
        uCellW: { value: this.nCells },
        uMode: { value: 0 },
        uGain: { value: 1 },
        uNear: { value: 0 },
        uRange: { value: 1 },
      },
      vertexShader: VERT,
      fragmentShader: FRAG,
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    this.lines = new THREE.LineSegments(g, this.material);
    this.lines.frustumCulled = false;
    this.scene.add(this.lines);
    this.nSeg = nSeg;
  }

  /** The central complex, drawn as a second line buffer keyed to reservoir state. */
  _buildCX() {
    const THREE = this.THREE, d = this.cxAtlas;
    const xyz = Int32Array.from(d.xyz);
    if (d.encoding === 'delta-per-polyline') {
      let at = 0;
      for (const len of d.plens) {
        for (let k = 1; k < len; k++) {
          xyz[at + k * 3] += xyz[at + (k - 1) * 3];
          xyz[at + k * 3 + 1] += xyz[at + (k - 1) * 3 + 1];
          xyz[at + k * 3 + 2] += xyz[at + (k - 1) * 3 + 2];
        }
        at += len * 3;
      }
    }
    const nV = xyz.length / 3;
    let nSeg = 0;
    for (const l of d.plens) nSeg += l - 1;
    const pos = new Float32Array(nV * 3);
    const flow = new Float32Array(nV);
    const cell = new Float32Array(nV);
    const cls = new Float32Array(nV).fill(4);
    const idx = new Uint32Array(nSeg * 2);
    const c = this.center;
    for (let v = 0; v < nV; v++) {
      pos[v * 3] = xyz[v * 3] - c[0];
      pos[v * 3 + 1] = xyz[v * 3 + 1] - c[1];
      pos[v * 3 + 2] = xyz[v * 3 + 2] - c[2];
    }
    let vAt = 0, iAt = 0;
    for (let n = 0; n < d.neurons.length; n++) {
      const slot = this.nKCCells + d.neurons[n].cx;
      for (let li = d.offsets[n]; li < d.offsets[n + 1]; li++) {
        const len = d.plens[li];
        for (let k = 0; k < len; k++) {
          cell[vAt + k] = slot;
          if (k) { idx[iAt++] = vAt + k - 1; idx[iAt++] = vAt + k; }
        }
        vAt += len;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aFlow', new THREE.BufferAttribute(flow, 1));
    g.setAttribute('aCell', new THREE.BufferAttribute(cell, 1));
    g.setAttribute('aClass', new THREE.BufferAttribute(cls, 1));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    this.cxLines = new THREE.LineSegments(g, this.material);
    this.cxLines.frustumCulled = false;
    this.cxLines.visible = true;
    this.scene.add(this.cxLines);
    this.nCXSeg = nSeg;
  }

  /** Neuropil volumes, one translucent mesh per anatomical family. */
  _buildNeuropils() {
    const THREE = this.THREE, np = this.np, c = this.center;
    const byFam = {};
    for (const r of np.regions) (byFam[r.family] ??= []).push(r);

    this.npGroup = new THREE.Group();
    this.npGroup.visible = false;
    for (const [fam, regions] of Object.entries(byFam)) {
      let nv = 0, nf = 0;
      for (const r of regions) { nv += r.nv; nf += r.nf; }
      const pos = new Float32Array(nv * 3);
      const idx = new Uint32Array(nf * 3);
      let vAt = 0, iAt = 0;
      for (const r of regions) {
        for (let i = 0; i < r.nv; i++) {
          pos[(vAt + i) * 3] = np.verts[(r.v0 + i) * 3] - c[0];
          pos[(vAt + i) * 3 + 1] = np.verts[(r.v0 + i) * 3 + 1] - c[1];
          pos[(vAt + i) * 3 + 2] = np.verts[(r.v0 + i) * 3 + 2] - c[2];
        }
        for (let f = 0; f < r.nf * 3; f++) {
          idx[iAt + f] = np.tris[r.f0 * 3 + f] - r.v0 + vAt;
        }
        vAt += r.nv; iAt += r.nf * 3;
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setIndex(new THREE.BufferAttribute(idx, 1));
      g.computeVertexNormals();
      const col = NEUROPIL[fam] ?? NEUROPIL.other;
      const m = new THREE.MeshBasicMaterial({
        color: new THREE.Color(col[0], col[1], col[2]),
        transparent: true, opacity: 0.045,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide, depthWrite: false,
      });
      this.npGroup.add(new THREE.Mesh(g, m));
    }
    this.scene.add(this.npGroup);
    // Re-centre so the whole brain is framed when the volumes are shown.
    const b = np.bbox;
    this.npRadius = Math.max(b[3] - b[0], b[4] - b[1], b[5] - b[2]) / 2;
  }

  /* --------------------------------------------------------- controls ---- */

  /** Orbit by dragging. Enabled by the page only where a pointer makes sense. */
  enableRotate() {
    if (this._rotateOn) return;
    this._rotateOn = true;
    const el = this.canvas;
    let down = false, px = 0, py = 0;
    const onDown = (e) => { down = true; px = e.clientX; py = e.clientY; el.setPointerCapture?.(e.pointerId); };
    const onMove = (e) => {
      if (!down) return;
      this.theta -= (e.clientX - px) * 0.008;
      this.phi = Math.max(0.08, Math.min(Math.PI - 0.08, this.phi - (e.clientY - py) * 0.008));
      px = e.clientX; py = e.clientY;
      this._place();
    };
    const onUp = (e) => { down = false; el.releasePointerCapture?.(e.pointerId); };
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerup', onUp);
    el.addEventListener('pointercancel', onUp);
    el.style.cursor = 'grab';
    el.style.touchAction = 'none';
  }

  setView(v) {
    this.view = v;
    // z is dorsal-ventral and grows toward the calyx, so +z is "up".
    if (v === 'frontal') { this.theta = 0; this.phi = Math.PI / 2; }
    else { this.theta = 0; this.phi = 0.12; }
    this._place();
  }

  toggleContext() {
    this.showContext = !this.showContext;
    if (this.npGroup) this.npGroup.visible = this.showContext;
    this._place();
    return this.showContext;
  }

  toggleOverlay() {
    this.overlay = !this.overlay;
    this.material.uniforms.uMode.value = this.overlay ? 1 : 0;
    return this.overlay;
  }

  _place() {
    const r = (this.showContext && this.npRadius ? this.npRadius : this.radius);
    const dist = r / Math.tan((this.camera.fov * Math.PI) / 360) * 0.92;
    const sp = Math.sin(this.phi), cp = Math.cos(this.phi);
    this.camera.position.set(
      dist * sp * Math.sin(this.theta),
      -dist * sp * Math.cos(this.theta),
      dist * cp,
    );
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(0, 0, 0);
  }

  _frame() { this.theta = 0; this.phi = Math.PI / 2; }

  /* ----------------------------------------------------------- driving --- */

  fire(kcIndices, t = performance.now() / 1000) {
    for (const k of kcIndices) if (k < this.nCells) this.firedAt[k] = t;
  }

  /** How much each central-complex neuron is currently holding. */
  setReservoir(h) {
    if (!this.nCX) return;
    let mx = 1e-6;
    for (let i = 0; i < h.length; i++) mx = Math.max(mx, Math.abs(h[i]));
    for (let i = 0; i < this.nCX; i++) {
      this.cellData[(this.nKCCells + i) * 4 + 3] = Math.abs(h[i] ?? 0) / mx;
    }
  }

  setEngram(engram) {
    let mx = 1e-9;
    for (let i = 0; i < engram.length && i < this.nCells; i++) mx = Math.max(mx, engram[i]);
    for (let i = 0; i < this.nCells; i++) this.cellData[i * 4 + 2] = (engram[i] ?? 0) / mx;
  }

  render() {
    const t0 = performance.now();
    const now = t0 / 1000;

    // Per-cell wavefront and envelope: 5,177 cells, once a frame, then one upload.
    const d = this.cellData;
    for (let i = 0; i < this.nKCCells; i++) {
      const t = this.firedAt[i];
      if (!t) { d[i * 4] = 0; d[i * 4 + 1] = 0; continue; }
      const age = now - t;
      if (age > 2.6) { d[i * 4] = 0; d[i * 4 + 1] = 0; this.firedAt[i] = 0; continue; }
      d[i * 4] = Math.min(1, age / SWEEP);
      d[i * 4 + 1] = Math.exp(-DECAY * Math.max(0, age - SWEEP));
    }
    this.cellTex.needsUpdate = true;

    const r = this.canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, r.width < 760 ? 1.5 : 2);
    const w = Math.max(1, r.width), h = Math.max(1, r.height);
    if (this._w !== w || this._h !== h || this._dpr !== dpr) {
      this.renderer.setPixelRatio(dpr);
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this._w = w; this._h = h; this._dpr = dpr;
    }
    this.material.uniforms.uGain.value = this.gain;
    const rad = this.showContext && this.npRadius ? this.npRadius : this.radius;
    const dist = this.camera.position.length();
    this.material.uniforms.uNear.value = Math.max(1, dist - rad);
    this.material.uniforms.uRange.value = 2 * rad;
    this.renderer.render(this.scene, this.camera);
    this.lastDraw = performance.now() - t0;
  }
}
