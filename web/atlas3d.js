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
// The verdict colours the rest of the page uses for the two dopamine populations.
const DOPAMINE = { pam: [0.23, 0.69, 0.94], ppl1: [1.0, 0.30, 0.43] };
// Texture rows are wrapped at this width: a phone GPU may refuse anything wider.
const TEX_W = 4096;
// Memory layer: remaining strength r is drawn as alpha * MEM_GAIN * r^3. The cube is
// there because additive overdraw compresses contrast -- a synapse at half strength
// has to look clearly dimmer than one at baseline, not 50% of a faint glow.
// Set once from the measured spread, not per screenshot: after 900 megacodes the
// median lobe sits at r = 0.72-0.87 (r^3 = 0.37-0.66) and the tenth percentile at
// 0.34-0.58, so at this gain a naive lobe is bright without saturating and a trained
// one is visibly dimmer with nothing firing at all.
const MEM_GAIN = 1.2;

const NEUROPIL = {
  optic:     [0.24, 0.41, 0.61], olfactory: [0.20, 0.56, 0.49],
  mushroom:  [0.69, 0.52, 0.25], central:   [0.49, 0.38, 0.66],
  lateral:   [0.31, 0.48, 0.59], ventrolat: [0.27, 0.42, 0.53],
  superior:  [0.41, 0.41, 0.57], inferior:  [0.34, 0.46, 0.52],
  ventromed: [0.30, 0.39, 0.49], periesoph: [0.38, 0.36, 0.45],
  other:     [0.35, 0.38, 0.44],
};

// Regions that contain what is drawn when the whole brain is off.
const MODEL_FAMILIES = new Set(['mushroom', 'central']);
// Past ~4x the camera sits inside the lobes and the view stops meaning much.
const ZOOM_MIN = 0.6, ZOOM_MAX = 4;

/** q-th quantile of the first n values (|v| if abs; skipping zeros if nonzero). */
function percentile(values, n, q, abs = false, nonzero = false) {
  const tmp = [];
  for (let i = 0; i < n; i++) {
    const v = abs ? Math.abs(values[i] ?? 0) : (values[i] ?? 0);
    if (!nonzero || v > 0) tmp.push(v);
  }
  if (!tmp.length) return 0;
  tmp.sort((a, b) => a - b);
  return tmp[Math.min(tmp.length - 1, Math.floor(q * tmp.length))];
}

const SWEEP = 0.45;      // seconds for the wavefront to cross a cell
const DECAY = 1.15;      // 1/s fade once the wave has passed

const VERT = `
attribute float aFlow;
attribute float aCell;
attribute float aClass;
attribute float aLobe;        // Kenyon-cell vertices: which lobe they sit in, else -1
uniform sampler2D uCells;
uniform vec2 uCellSize;
uniform sampler2D uMem;       // remaining KC->MBON strength per Kenyon cell per lobe
uniform vec2 uMemSize;
uniform float uNLobes;
uniform float uMemGain;
uniform vec3 uDop;            // colour of the dopamine population that last fired
uniform float uMode;          // 0 = activity, 1 = memory, 2 = working memory
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
const vec3 MEMC = vec3(0.62, 1.0, 0.9);       // a path cell in the memory layer: bright teal
const vec3 WM_POS = vec3(0.55, 0.95, 1.0);    // working memory: a reservoir cell held above rest
const vec3 WM_NEG = vec3(1.0, 0.62, 0.36);    // ... and held below it
const vec3 WRITE  = vec3(1.0, 0.85, 0.36);    // the mushroom body writing in, in its firing yellow
const vec3 READ   = vec3(1.0, 1.0, 1.0);      // the eight cells read back out

vec4 texel(sampler2D t, vec2 size, float i) {
  return texture2D(t, vec2((mod(i, size.x) + 0.5) / size.x, (floor(i / size.x) + 0.5) / size.y));
}

void main() {
  int ci = int(aClass + 0.5);
  bool isCX = ci == 4;
  vec3 base = isCX ? CXC : ci == 0 ? KC : ci == 1 ? MBON : ci == 2 ? DAN : APL;
  float a = isCX ? BASE_CX : ci == 0 ? BASE.x : ci == 1 ? BASE.y : ci == 2 ? BASE.z : BASE.w;
  vec3 hot = isCX ? HOLD : HOT;

  float lit = 0.0;
  if (aCell >= 0.0) {
    vec4 s = texel(uCells, uCellSize, aCell);
    if (uMode > 1.5) {
      // Working memory: the reservoir speaks, everything else is context (below).
      // s.b is the signed state scaled to the largest, s.g the write pulse from the
      // MBON->CX pathway, s.r marks the cells the mushroom body reads.
      if (isCX) {
        float v = s.b;
        hot = v >= 0.0 ? WM_POS : WM_NEG;
        lit = abs(v);
        if (s.g > 0.02) { hot = mix(hot, WRITE, clamp(s.g * 1.5, 0.0, 1.0)); lit = max(lit, s.g); }
        if (s.r > 0.5) { hot = READ; lit = max(lit, 0.9); a *= 4.0; }
      }
    } else if (ci == 2) {
      lit = s.a;                                  // dopamine release, in activity and memory
      hot = uDop;
    } else if (isCX) {
      lit = uMode > 0.5 ? 0.0 : s.a;              // how much this cell is holding
    } else if (uMode > 0.5) {
      // Memory: brightness is what is left of this cell's output where the vertex
      // is. Brightness, never width -- a line here is traced neurite. A lobe the
      // cell makes no synapses in has nothing to lose and stays at baseline.
      // The cells on the decision just taken light up for as long as their firing
      // envelope lasts, as bright as what is left of them in each lobe; every other
      // cell stays as faint context. Averaged over all 5,177 cells, training only
      // dims each lobe as a whole, which says nothing about which path was punished.
      // Added like firing rather than multiplied into the base: 78 cells at a
      // multiple of a 0.012 base alpha do not show at all.
      hot = MEMC;
      float m = aLobe >= 0.0 ? texel(uMem, uMemSize, aCell * uNLobes + aLobe).r : -1.0;
      // Context, not memory: tracts, lobes this cell makes no synapses in, and the
      // calyx (lobe 3), which holds 3% of KC->MBON connections but so much dendrite
      // that at full brightness it outshone every lobe that actually learns.
      if (m < 0.0 || aLobe > 2.5) {
        a *= 0.3;
        lit = s.g * 0.2;
      } else {
        float r = clamp(m, 0.0, 1.0);
        float r3 = r * r * r;
        a *= uMemGain * r3;
        lit = s.g * r3;
      }
    } else if (s.g > 0.0) {
      // s.r is the wavefront position along the cell, s.g the envelope
      float lead = 1.0 - clamp(abs(s.r - aFlow) * 3.2, 0.0, 1.0);
      lit = aFlow <= s.r + 0.03 ? s.g * (0.34 + 0.66 * lead) : 0.0;
    }
  }
  // Dimmer than the memory layer's context: 5,177 Kenyon cells at 0.3 still out-glow
  // the reservoir they sit beside.
  if (uMode > 1.5 && !isCX) { a *= 0.12; lit = 0.0; }
  vColor = mix(base, hot, clamp(lit * 1.4, 0.0, 1.0));
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
  constructor(canvas, data, indexOf, neuropils = null, cxAtlas = null, lobes = null) {
    const THREE = window.THREE;
    this.THREE = THREE;
    this.canvas = canvas;
    this.data = data;
    this.np = neuropils;
    this.cxAtlas = cxAtlas;
    this.lobes = lobes;           // parsed mb_lobes.json, optional: the memory layer needs it
    this.dopAt = 0;
    this.dopPAM = 1;
    this.showContext = false;
    this.view = 'frontal';
    this.overlay = false;
    this.lastDraw = 0;
    this.gain = 1;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setClearColor(0x05070a, 1);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 1, 20000);
    this.zoom = 1;
    this.target = new THREE.Vector3();    // orbit centre; moves when zooming toward the pointer
    this.raycaster = new THREE.Raycaster();

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
    // Drawn in the 400 nm units the neuropil and central-complex files use. The phone
    // atlas is quantised to 500 nm, and drawn unscaled it came out 20% smaller than
    // the lobe surfaces and the reservoir around it.
    const s = (d.unit_nm ?? 400) / 400;
    this.coords = s === 1 ? xyz : Float32Array.from(xyz, (v) => v * s);
    const b = d.bbox.map((v) => v * s);
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
    // The texture covers the Kenyon cells, then the reservoir, then the dopamine
    // neurons, so one lookup serves all three populations.
    this.nCX = this.cxAtlas ? this.cxAtlas.neurons.length : 0;
    this.danBase = this.nKCCells + this.nCX;
    const dans = [];
    n.forEach((neuron, i) => { if (neuron.c === 2) dans.push(i); });
    // Same split the model makes: PAM is the reward arm, every other DAN the
    // punishment arm (mushroom-body.js, _buildCompartments).
    this.danPAM = Uint8Array.from(dans, (i) => (String(n[i].t).startsWith('PAM') ? 1 : 0));
    dans.forEach((i, j) => { this.cellOf[i] = this.danBase + j; });
    this.nCells = this.danBase + dans.length;
    this.firedAt = new Float64Array(this.nKCCells);
    this.counts = n.reduce((a, x) => (a[x.s] = (a[x.s] || 0) + 1, a), {});

    const THREE = this.THREE;
    const texture = (count) => {
      const w = Math.min(count, TEX_W), h = Math.ceil(count / w);
      const data = new Float32Array(w * h * 4);
      const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.FloatType);
      tex.magFilter = tex.minFilter = THREE.NearestFilter;
      tex.needsUpdate = true;
      return { data, tex, size: new THREE.Vector2(w, h) };
    };
    ({ data: this.cellData, tex: this.cellTex, size: this.cellSize } = texture(this.nCells));
    this.nLobes = this.lobes?.lobes.length ?? 1;
    ({ data: this.memData, tex: this.memTex, size: this.memSize } = texture(this.nKCCells * this.nLobes));
    for (let i = 0; i < this.memData.length; i += 4) this.memData[i] = 1;   // naive: all at baseline
  }

  /** The lobe surfaces as a voxel grid, from pipeline/extract_lobes.py, or null. */
  _lobeGrid() {
    const g = this.lobes?.grid;
    if (!g) return null;
    const [nx, ny, nz] = g.dims;
    const lab = new Uint8Array(nx * ny * nz);
    for (let i = 0, at = 0; i < g.rle.length; i += 2) {
      lab.fill(g.rle[i], at, at + g.rle[i + 1]);
      at += g.rle[i + 1];
    }
    const step = g.unit_nm / 400, o = g.origin_nm.map((v) => v / 400);
    return (x, y, z) => {
      const i = Math.floor((x - o[0]) / step), j = Math.floor((y - o[1]) / step);
      const k = Math.floor((z - o[2]) / step);
      if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return -1;
      const L = lab[(i * ny + j) * nz + k];      // x-major: numpy's ravel of [x][y][z]
      return L ? L - 1 : -1;
    };
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
    const lobe = new Float32Array(nV).fill(-1);
    const idx = new Uint32Array(nSeg * 2);
    const lobeAt = this._lobeGrid();

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
          const v = vAt + k;
          cell[v] = cellId;
          cls[v] = klass;
          if (lobeAt && klass === 0) {
            lobe[v] = lobeAt(this.coords[v * 3], this.coords[v * 3 + 1], this.coords[v * 3 + 2]);
          }
          if (k) { idx[iAt++] = v - 1; idx[iAt++] = v; }
        }
        vAt += len;
      }
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aFlow', new THREE.BufferAttribute(flow, 1));
    g.setAttribute('aCell', new THREE.BufferAttribute(cell, 1));
    g.setAttribute('aClass', new THREE.BufferAttribute(cls, 1));
    g.setAttribute('aLobe', new THREE.BufferAttribute(lobe, 1));
    g.setIndex(new THREE.BufferAttribute(idx, 1));

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uCells: { value: this.cellTex },
        uCellSize: { value: this.cellSize },
        uMem: { value: this.memTex },
        uMemSize: { value: this.memSize },
        uNLobes: { value: this.nLobes },
        uMemGain: { value: MEM_GAIN },
        uDop: { value: new THREE.Vector3(...DOPAMINE.pam) },
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
    g.setAttribute('aLobe', new THREE.BufferAttribute(new Float32Array(nV).fill(-1), 1));
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
    // Picking keeps its own list rather than raycasting the group: the volumes stay
    // hidden until the whole brain is switched on, and the mushroom body's own
    // regions still need names while they are.
    this.npPick = [];
    for (const [fam, regions] of Object.entries(byFam)) {
      let nv = 0, nf = 0;
      for (const r of regions) { nv += r.nv; nf += r.nf; }
      const pos = new Float32Array(nv * 3);
      const idx = new Uint32Array(nf * 3);
      const spans = [];
      let vAt = 0, iAt = 0;
      for (const r of regions) {
        // Families are merged into one mesh, so a hit's face index is mapped back
        // to its region through these triangle ranges.
        spans.push({ name: r.name.replace(/_[LR]$/, ''), side: r.side ?? null,
                     family: fam, f0: iAt / 3, nf: r.nf });
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
      const mesh = new THREE.Mesh(g, m);
      mesh.userData = { family: fam, spans, color: col };
      this.npGroup.add(mesh);
      this.npPick.push(mesh);
    }
    this.scene.add(this.npGroup);
    // One highlight, re-pointed at whichever region is under the pointer. It shares
    // the family's buffers and draws only that region's triangle range. Kept faint:
    // with no depth test every fold of the surface adds, and zoomed in a brighter
    // value turns the lobe into a solid wall that hides the neurons it names.
    this.npHi = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({
      transparent: true, opacity: 0.05, blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide, depthWrite: false, depthTest: false,
    }));
    this.npHi.visible = false;
    this.npHi.frustumCulled = false;
    this.scene.add(this.npHi);
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
      this.dragging = true;
      this.theta -= (e.clientX - px) * 0.008;
      this.phi = Math.max(0.08, Math.min(Math.PI - 0.08, this.phi - (e.clientY - py) * 0.008));
      px = e.clientX; py = e.clientY;
      this._place();
    };
    const onUp = (e) => { down = false; this.dragging = false; el.releasePointerCapture?.(e.pointerId); };
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerup', onUp);
    el.addEventListener('pointercancel', onUp);
    el.style.cursor = 'grab';
    el.style.touchAction = 'none';
  }

  /**
   * Zoom. The page's scroll wheel is only taken with Ctrl/⌘ held (a trackpad pinch
   * arrives as exactly that): the panel is most of a screen tall, and a plain wheel
   * would trap anyone scrolling past it.
   */
  enableZoom(onBlockedWheel = null) {
    if (this._zoomOn) return;
    this._zoomOn = true;
    this.canvas.addEventListener('wheel', (e) => {
      if (!e.ctrlKey && !e.metaKey) { onBlockedWheel?.(); return; }
      e.preventDefault();
      // A mouse notch is ~100 px, a pinch a few px per event: clamping keeps one
      // notch from jumping 3x while leaving a pinch smooth.
      const px = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      const dy = Math.max(-50, Math.min(50, px));
      this.zoomBy(Math.exp(-dy * 0.008), e.clientX, e.clientY);
    }, { passive: false });
    // With no preset buttons, this is the only way home after orbiting: reset
    // rotation as well as zoom, back to the frontal view the page opens on.
    this.canvas.addEventListener('dblclick', () => this.setView('frontal'));
  }

  /**
   * Zooming in anchors on the point under the pointer, so the structure being looked
   * at stays put. Zooming out walks the centre back home, so reaching 1x always
   * frames the whole atlas again.
   */
  zoomBy(factor, clientX = null, clientY = null) {
    const next = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, this.zoom * factor));
    if (next === this.zoom) return next;
    if (next > this.zoom && clientX !== null) {
      const p = this._focalPoint(clientX, clientY);
      if (p) this.target.lerp(p, 1 - this.zoom / next);
    } else if (next < this.zoom) {
      this.target.multiplyScalar(this.zoom > 1 && next > 1 ? (next - 1) / (this.zoom - 1) : 0);
    }
    this.zoom = next;
    this._place();
    this.onZoom?.(next);
    return next;
  }

  resetZoom() {
    this.zoom = 1;
    this.target.set(0, 0, 0);
    this._place();
    this.onZoom?.(1);
  }

  /** Where the pointer's ray crosses the plane through the orbit centre. */
  _focalPoint(clientX, clientY) {
    const THREE = this.THREE;
    if (!this._setRay(clientX, clientY)) return null;
    const n = new THREE.Vector3().subVectors(this.camera.position, this.target).normalize();
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(n, this.target);
    return this.raycaster.ray.intersectPlane(plane, new THREE.Vector3());
  }

  _setRay(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    if (!r.width || !r.height) return false;
    this._ndc ??= new this.THREE.Vector2();
    this._ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.camera.aspect = r.width / r.height;
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld();
    this.raycaster.setFromCamera(this._ndc, this.camera);
    return true;
  }

  /**
   * The neuropil under a screen point, or null. With only the mushroom body and the
   * central complex drawn, only their regions answer -- otherwise pointing at a lobe
   * would name whatever larger shell happens to sit in front of it.
   */
  pick(clientX, clientY) {
    if (!this.npPick || !this._setRay(clientX, clientY)) return null;
    const hits = [];
    for (const mesh of this.npPick) {
      if (!this.showContext && !MODEL_FAMILIES.has(mesh.userData.family)) continue;
      mesh.raycast(this.raycaster, hits);
    }
    if (!hits.length) return null;
    hits.sort((a, b) => a.distance - b.distance);
    const { object, faceIndex } = hits[0];
    const span = object.userData.spans.find((s) => faceIndex >= s.f0 && faceIndex < s.f0 + s.nf);
    return span ? { ...span, mesh: object } : null;
  }

  /** Light one region's volume, or clear it with null. */
  highlight(hit) {
    if (!this.npHi) return;
    if (!hit) { this.npHi.visible = false; return; }
    const src = hit.mesh.geometry;
    let g = hit.mesh.userData.hiGeom;
    if (!g) {
      g = new this.THREE.BufferGeometry();
      g.setAttribute('position', src.getAttribute('position'));
      g.setIndex(src.getIndex());
      hit.mesh.userData.hiGeom = g;
    }
    g.setDrawRange(hit.f0 * 3, hit.nf * 3);
    this.npHi.geometry = g;
    this.npHi.material.color.setRGB(...hit.mesh.userData.color);
    this.npHi.visible = true;
  }

  /**
   * Name the region under the pointer. A mouse gets it on hover; touch has no hover,
   * so a tap that did not move names it instead.
   */
  enablePick(onPick) {
    if (!this.npPick) return;
    const el = this.canvas;
    let queued = null, inside = false, sx = 0, sy = 0;
    const run = () => {
      const q = queued;
      queued = null;
      if (!inside || !q) return;
      const hit = this.dragging ? null : this.pick(q.x, q.y);
      this.highlight(hit);
      onPick(hit, q.x, q.y);
    };
    el.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return;
      inside = true;
      if (!queued) requestAnimationFrame(run);   // at most one raycast a frame
      queued = { x: e.clientX, y: e.clientY };
    });
    el.addEventListener('pointerleave', () => {
      inside = false;
      this.highlight(null);
      onPick(null);
    });
    el.addEventListener('pointerdown', (e) => { sx = e.clientX; sy = e.clientY; });
    el.addEventListener('pointerup', (e) => {
      if (e.pointerType === 'mouse' || Math.hypot(e.clientX - sx, e.clientY - sy) > 8) return;
      const hit = this.pick(e.clientX, e.clientY);
      this.highlight(hit);
      onPick(hit, e.clientX, e.clientY);
    });
  }

  setView(v) {
    this.view = v;
    // z is dorsal-ventral and grows toward the calyx, so +z is "up".
    if (v === 'frontal') { this.theta = 0; this.phi = Math.PI / 2; }
    else { this.theta = 0; this.phi = 0.12; }
    // A preset view is a way home: it frames the whole atlas again.
    this.zoom = 1;
    this.target.set(0, 0, 0);
    this.onZoom?.(1);
    this._place();
  }

  toggleContext() {
    this.showContext = !this.showContext;
    if (this.npGroup) this.npGroup.visible = this.showContext;
    this._place();
    return this.showContext;
  }

  /** 0 = activity, 1 = memory, 2 = working memory. One at a time: they share the lines. */
  setMode(m) {
    this.mode = m;
    this.overlay = m === 1;
    this.material.uniforms.uMode.value = m;
    return m;
  }

  _place() {
    const r = (this.showContext && this.npRadius ? this.npRadius : this.radius);
    const dist = r / Math.tan((this.camera.fov * Math.PI) / 360) * 0.92 / this.zoom;
    const sp = Math.sin(this.phi), cp = Math.cos(this.phi), t = this.target;
    this.camera.position.set(
      t.x + dist * sp * Math.sin(this.theta),
      t.y - dist * sp * Math.cos(this.theta),
      t.z + dist * cp,
    );
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(t);
  }

  _frame() { this.theta = 0; this.phi = Math.PI / 2; }

  /* ----------------------------------------------------------- driving --- */

  fire(kcIndices, t = performance.now() / 1000) {
    for (const k of kcIndices) if (k < this.nKCCells) this.firedAt[k] = t;
  }

  /** How much each central-complex neuron is currently holding. */
  setReservoir(h) {
    if (!this.nCX) return;
    let mx = 1e-6;
    for (let i = 0; i < h.length; i++) mx = Math.max(mx, Math.abs(h[i]));
    // The working-memory layer scales to the 95th percentile, not the maximum: in the
    // model only ~8 of 2,875 cells sit above half the largest |h|, so a max scale
    // shows a handful of cells and leaves the state being held invisible.
    const p95 = percentile(h, this.nCX, 0.95, true) || mx;
    for (let i = 0; i < this.nCX; i++) {
      const v = h[i] ?? 0;
      this.cellData[(this.nKCCells + i) * 4 + 3] = Math.abs(v) / mx;
      this.cellData[(this.nKCCells + i) * 4 + 2] = Math.max(-1, Math.min(1, v / p95));
    }
  }

  /** Mark the cells the mushroom body reads back. Which ones is modelled (hub cells). */
  setReadouts(indices) {
    if (!this.nCX) return;
    for (const i of indices) if (i < this.nCX) this.cellData[(this.nKCCells + i) * 4] = 1;
  }

  /** How hard the MBON->CX pathway wrote into each reservoir cell on the last step. */
  cxWrite(drive, t = performance.now() / 1000) {
    if (!this.nCX || !drive) return;
    const lv = (this.writeLevel ??= new Float32Array(this.nCX));
    // Scaled among the cells actually driven (~295 a step), at their 90th percentile:
    // against the maximum only ~7 of them reach 30%, and the flash is invisible.
    const p90 = percentile(drive, this.nCX, 0.9, false, true) || 1e-9;
    for (let i = 0; i < this.nCX; i++) lv[i] = Math.min(1, (drive[i] ?? 0) / p90);
    this.writeAt = t;
  }

  /**
   * Remaining strength per Kenyon cell per lobe, laid out as the model's
   * `memoryByLobe` returns it: cell * nLobes + lobe. Absolute, not rescaled to the
   * current maximum -- a rescale would make the first depressed synapse look as
   * dark as a fully trained one.
   */
  setMemory(mem) {
    const d = this.memData;
    for (let i = 0; i < mem.length && i * 4 < d.length; i++) d[i * 4] = mem[i];
    this.memTex.needsUpdate = true;
  }

  get hasMemory() { return !!this.lobes; }

  /** A verdict: the reward (PAM) or the punishment population releases dopamine. */
  dopamine(correct, t = performance.now() / 1000) {
    this.dopAt = t;
    this.dopPAM = correct ? 1 : 0;
    this.material.uniforms.uDop.value.set(...(correct ? DOPAMINE.pam : DOPAMINE.ppl1));
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
    // The whole population releases together, as it does in the model: the teaching
    // signal is one bit, not a per-compartment choice.
    if (this.dopAt) {
      const e = Math.exp(-1.4 * (now - this.dopAt));
      const on = e > 0.01 ? e : 0;
      for (let j = 0; j < this.danPAM.length; j++) {
        d[(this.danBase + j) * 4 + 3] = this.danPAM[j] === this.dopPAM ? on : 0;
      }
      if (!on) this.dopAt = 0;
    }
    if (this.writeAt) {
      const e = Math.exp(-2.2 * (now - this.writeAt));
      const on = e > 0.01 ? e : 0;
      for (let i = 0; i < this.nCX; i++) d[(this.nKCCells + i) * 4 + 1] = this.writeLevel[i] * on;
      if (!on) this.writeAt = 0;
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
    const dist = this.camera.position.distanceTo(this.target);
    this.material.uniforms.uNear.value = Math.max(1, dist - rad);
    this.material.uniforms.uRange.value = 2 * rad;
    this.renderer.render(this.scene, this.camera);
    this.lastDraw = performance.now() - t0;
  }
}
