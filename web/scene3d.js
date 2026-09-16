/**
 * The resuscitation bay, built procedurally -- no model files, no asset pipeline.
 * A fly is mostly convex blobs, which is exactly what primitives are good at.
 *
 * Everything the scene shows is driven by the engine and the fly's own decisions;
 * this module only renders. It never decides anything.
 */

const C = {
  ecg:   0x43e08a,
  amber: 0xf0a93b,
  pam:   0x3bb0f0,
  ppl1:  0xff4d6d,
  wing:  0x9aa7ff,
  skin:  0xc9a68a,
  linen: 0x2a3340,
  steel: 0x4a5663,
};

/** Drug syringe colours, keyed to the action name. */
const DRUG_COLOR = {
  epinephrine: 0xffffff,
  amiodarone:  0xf5c451,
  lidocaine:   0x6fa8ff,
};

/* ---------------------------------------------------------------- ECG ---- */

/** Voltage at time t for a rhythm, in millivolt-ish units. */
export function ecgAt(rhythm, t) {
  const noise = (Math.sin(t * 131.1) + Math.sin(t * 57.7)) * 0.012;
  switch (rhythm) {
    case 'VF':
      return noise + 0.34 * Math.sin(t * 19) * Math.sin(t * 7.3 + 1.1)
                   + 0.22 * Math.sin(t * 31.7 + 0.4);
    case 'pVT': {
      const p = (t * 3.1) % 1;                       // ~185 bpm, wide and regular
      return noise + 0.62 * Math.sin(p * Math.PI * 2) * (1 - 0.3 * Math.cos(p * Math.PI * 4));
    }
    case 'asystole':
      return noise * 0.5;
    case 'PEA':
    case 'ROSC': {
      const rate = rhythm === 'ROSC' ? 1.3 : 0.9;    // organised complexes
      const p = (t * rate) % 1;
      let v = 0;
      if (p < 0.10) v = 0.10 * Math.sin((p / 0.10) * Math.PI);            // P
      else if (p < 0.16) v = -0.09 * Math.sin(((p - 0.10) / 0.06) * Math.PI); // Q
      else if (p < 0.22) v = 1.00 * Math.sin(((p - 0.16) / 0.06) * Math.PI);  // R
      else if (p < 0.30) v = -0.22 * Math.sin(((p - 0.22) / 0.08) * Math.PI); // S
      else if (p < 0.52) v = 0.20 * Math.sin(((p - 0.30) / 0.22) * Math.PI);  // T
      return noise + v;
    }
    default:
      return noise;
  }
}

/* ------------------------------------------------------------- scene ---- */

// Everything the bay draws except the floor sits inside this box. The camera is pushed
// back until all eight corners are in frame, so no panel shape can cut the defibrillator
// or the monitor off the edge.
const BAY_BOX = { min: [-3.8, 0.15, -1.6], max: [3.2, 3.0, 2.5] };
const BASE_DIST = 7.44;     // the framing the page's old wide strip used, before any push

export class Bay {
  constructor(canvas) {
    const THREE = window.THREE;
    this.THREE = THREE;
    this.canvas = canvas;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    // Phones report devicePixelRatio 3; rendering the bay at 3x costs nine times
    // the fill rate and looks the same at arm's length.
    // A page measured before it has a size reports 0; treat that as desktop
    // rather than permanently downgrading a real machine.
    const vw = Math.max(document.documentElement?.clientWidth || 0,
                        window.innerWidth || 0) || 1024;
    const small = vw < 760;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, small ? 1.25 : 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x07090c);
    this.scene.fog = new THREE.Fog(0x07090c, 11, 26);

    this.camera = new THREE.PerspectiveCamera(37, 1, 0.1, 100);
    // The camera keeps this direction and this target; only its distance changes, set
    // by _frame() from the shape of the panel it is drawn in.
    this.target = new THREE.Vector3(-0.25, 1.35, 0);
    this.viewDir = new THREE.Vector3(4.6, 3.0, 5.4).sub(this.target).normalize();
    this.camera.lookAt(this.target);

    this.scene.add(new THREE.AmbientLight(0x2c3644, 1.0));
    const key = new THREE.DirectionalLight(0xdbe8ff, 1.15);
    key.position.set(4, 8, 5);
    this.scene.add(key);
    const rim = new THREE.PointLight(C.ecg, 0.9, 14);
    rim.position.set(-2.6, 2.4, -1.2);
    this.scene.add(rim);
    this.roscLight = new THREE.PointLight(C.ecg, 0, 10);
    this.roscLight.position.set(0, 2.4, 0);
    this.scene.add(this.roscLight);

    this._floor();
    this._gurney();
    this._patient();
    this._monitor();
    this._defib();
    this._cart();
    this._fly();

    this.t = 0;
    this.rhythm = 'VF';
    this.action = 'cpr';
    this.flashV = 0;
    this.joltV = 0;
    this.cprPhase = 0;
    this.carrying = null;
    this.verdictV = 0;
    this.verdictColor = C.pam;
    this.resize();
  }

  /* -- set dressing -- */

  _mat(color, opts = {}) {
    return new this.THREE.MeshLambertMaterial({ color, ...opts });
  }

  _floor() {
    const THREE = this.THREE;
    const g = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), this._mat(0x0b0f14));
    g.rotation.x = -Math.PI / 2;
    this.scene.add(g);
    const grid = new THREE.GridHelper(40, 40, 0x18222e, 0x121a24);
    grid.position.y = 0.002;
    this.scene.add(grid);
  }

  _gurney() {
    const THREE = this.THREE;
    const top = new THREE.Mesh(new THREE.BoxGeometry(4.6, 0.16, 1.9), this._mat(C.linen));
    top.position.set(0, 0.92, 0);
    this.scene.add(top);
    const rail = this._mat(C.steel);
    for (const x of [-2.1, 2.1]) {
      for (const z of [-0.8, 0.8]) {
        const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.84, 10), rail);
        leg.position.set(x, 0.42, z);
        this.scene.add(leg);
      }
    }
  }

  _patient() {
    const THREE = this.THREE;
    this.patient = new THREE.Group();
    const skin = this._mat(C.skin);
    const gown = this._mat(0x5d6f86);

    this.chest = new THREE.Mesh(new THREE.SphereGeometry(0.62, 20, 14), gown);
    this.chest.scale.set(1.5, 0.62, 1.0);
    this.chest.position.set(-0.35, 1.18, 0);
    this.patient.add(this.chest);

    const belly = new THREE.Mesh(new THREE.SphereGeometry(0.52, 18, 12), gown);
    belly.scale.set(1.5, 0.55, 0.95);
    belly.position.set(0.75, 1.12, 0);
    this.patient.add(belly);

    const head = new THREE.Mesh(new THREE.SphereGeometry(0.33, 20, 16), skin);
    head.position.set(-1.55, 1.24, 0);
    this.patient.add(head);

    for (const z of [-0.62, 0.62]) {                       // arms
      const arm = new THREE.Mesh(new THREE.CapsuleGeometry(0.11, 1.25, 6, 10), skin);
      arm.rotation.z = Math.PI / 2;
      arm.position.set(-0.1, 1.06, z);
      this.patient.add(arm);
    }
    for (const z of [-0.3, 0.3]) {                         // legs
      const leg = new THREE.Mesh(new THREE.CapsuleGeometry(0.15, 1.5, 6, 10), gown);
      leg.rotation.z = Math.PI / 2;
      leg.position.set(1.85, 1.07, z);
      this.patient.add(leg);
    }
    this.scene.add(this.patient);

    // Electrode pads on the chest -- where the shock lands.
    this.pads = [];
    for (const [x, z] of [[-0.85, -0.42], [0.05, 0.42]]) {
      const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.03, 16),
        new THREE.MeshBasicMaterial({ color: 0x1d2836 }));
      pad.position.set(x, 1.52, z);
      this.scene.add(pad);
      this.pads.push(pad);
    }
  }

  _monitor() {
    const THREE = this.THREE;
    const g = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(2.3, 1.6, 0.18), this._mat(0x151c26));
    g.add(body);

    this.ecgCanvas = document.createElement('canvas');
    this.ecgCanvas.width = 512; this.ecgCanvas.height = 352;
    this.ecgCtx = this.ecgCanvas.getContext('2d');
    this.ecgTex = new THREE.CanvasTexture(this.ecgCanvas);
    const screen = new THREE.Mesh(new THREE.PlaneGeometry(2.05, 1.38),
      new THREE.MeshBasicMaterial({ map: this.ecgTex }));
    screen.position.z = 0.1;
    g.add(screen);

    g.position.set(-2.55, 2.15, -1.25);
    g.rotation.y = 0.55;
    this.scene.add(g);
    this.monitorPos = new THREE.Vector3(-2.1, 2.05, -0.7);
  }

  _defib() {
    const THREE = this.THREE;
    const g = new THREE.Group();
    const shell = this._mat(0x243140), trim = this._mat(0x2f3f52), darkFace = this._mat(0x11171f);

    // A case with a lip and a recessed face, not one flat box: the silhouette is what
    // makes it read as a machine at this size. Kept inside BAY_BOX so _frame() still
    // fits it on screen.
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.92, 0.58, 0.68), shell);
    g.add(body);
    const lip = new THREE.Mesh(new THREE.BoxGeometry(0.96, 0.07, 0.72), trim);
    lip.position.y = 0.3;
    g.add(lip);
    const foot = new THREE.Mesh(new THREE.BoxGeometry(0.86, 0.06, 0.62), darkFace);
    foot.position.y = -0.31;
    g.add(foot);

    const face = new THREE.Mesh(new THREE.BoxGeometry(0.66, 0.4, 0.04), darkFace);
    face.position.set(0, 0.03, 0.35);
    g.add(face);
    const trace = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.03, 0.01),
      new THREE.MeshBasicMaterial({ color: C.ecg }));
    trace.position.set(0, 0.08, 0.38);
    g.add(trace);
    for (let i = 0; i < 3; i++) {                        // control knobs on the face
      const k = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.03, 12), trim);
      k.rotation.x = Math.PI / 2;
      k.position.set(-0.2 + i * 0.2, -0.11, 0.37);
      g.add(k);
    }
    const handle = new THREE.Mesh(new THREE.TorusGeometry(0.13, 0.022, 8, 20, Math.PI), trim);
    handle.position.set(0, 0.33, -0.12);
    g.add(handle);

    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.055, 12, 10),
      new THREE.MeshBasicMaterial({ color: C.amber }));
    lamp.position.set(0.3, 0.16, 0.36);
    g.add(lamp);
    this.defibLamp = lamp;

    g.position.set(2.7, 1.25, -1.1);
    this.scene.add(g);

    // Paddles: a domed plate, a collar, an angled grip with a discharge button -- two
    // bare cylinders read as coins. The group's position is animated, not its rotation,
    // so everything is built plates-down and looks right docked or over the chest.
    this.paddles = new THREE.Group();
    // A little emissive so the plate's underside is not solid black when the paddles are
    // held over the chest, where the key light above them never reaches.
    const metal = this._mat(0x8b96a4, { emissive: 0x2b333d });
    const grip = this._mat(0x1b2430), button = this._mat(0xd8342a);
    for (const z of [-0.22, 0.22]) {
      const p = new THREE.Group();
      const plate = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.135, 0.05, 20), metal);
      p.add(plate);
      const dome = new THREE.Mesh(new THREE.SphereGeometry(0.135, 16, 10), metal);
      dome.scale.set(1, 0.32, 1);
      dome.position.y = 0.04;
      p.add(dome);
      const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.075, 0.07, 12), grip);
      collar.position.y = 0.09;
      p.add(collar);
      const stock = new THREE.Mesh(new THREE.CapsuleGeometry(0.048, 0.16, 6, 12), grip);
      stock.position.set(-0.05, 0.19, 0);
      stock.rotation.z = 0.42;                           // held at an angle, as they are
      p.add(stock);
      const b = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.02, 10), button);
      b.rotation.z = Math.PI / 2;
      b.position.set(0.02, 0.25, 0);
      p.add(b);
      p.position.z = z;
      this.paddles.add(p);
    }
    this.paddles.position.copy(new THREE.Vector3(2.7, 1.66, -1.1));   // resting on the case
    this.scene.add(this.paddles);
  }

  _cart() {
    const THREE = this.THREE;
    const top = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.1, 0.9), this._mat(0x2b3644));
    top.position.set(-1.0, 1.05, 2.0);
    this.scene.add(top);

    this.syringes = {};
    let i = 0;
    for (const [name, color] of Object.entries(DRUG_COLOR)) {
      const s = new THREE.Group();
      const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.36, 10),
        new THREE.MeshLambertMaterial({ color, transparent: true, opacity: 0.92 }));
      barrel.rotation.z = Math.PI / 2;
      s.add(barrel);
      const needle = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.16, 6), this._mat(0xb9c4d0));
      needle.rotation.z = Math.PI / 2;
      needle.position.x = 0.26;
      s.add(needle);
      s.position.set(-1.45 + i * 0.45, 1.16, 2.0);
      this.scene.add(s);
      this.syringes[name] = s;
      this.syringeHome = this.syringeHome || {};
      this.syringeHome[name] = s.position.clone();
      i++;
    }
  }

  /** The fly: thorax, abdomen, head, compound eyes, six legs, two wings. */
  _fly() {
    const THREE = this.THREE;
    const f = new THREE.Group();

    const cuticle = new THREE.MeshLambertMaterial({ color: 0xc98a3a });
    const dark = new THREE.MeshLambertMaterial({ color: 0x5c4220 });

    // Rounded volumes rather than one ellipsoid per part: a fly at this scale reads as
    // a toy, and the toy should still have a back, a waist and a tapering abdomen.
    const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.2, 20, 16), cuticle);
    thorax.scale.set(1.05, 0.9, 0.95);
    f.add(thorax);
    const notum = new THREE.Mesh(new THREE.SphereGeometry(0.15, 18, 14), cuticle);
    notum.scale.set(1.15, 0.72, 1.0);
    notum.position.set(-0.03, 0.09, 0);                 // the humped back
    f.add(notum);
    const waist = new THREE.Mesh(new THREE.SphereGeometry(0.1, 14, 12), dark);
    waist.scale.set(0.8, 0.95, 0.95);
    waist.position.x = -0.17;
    f.add(waist);

    // Four tapering segments instead of a single ellipsoid with rings painted on it.
    // Both tones stay in the dark half: alternating dark with the bright thorax colour
    // gave it wasp stripes, and a fruit fly's abdomen is banded much more quietly.
    const band = new THREE.MeshLambertMaterial({ color: 0x7c5626 });
    const SEG = [[0.145, -0.26], [0.128, -0.39], [0.102, -0.5], [0.07, -0.58]];
    SEG.forEach(([r, x], i) => {
      const s = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 14), i % 2 ? band : dark);
      s.scale.set(1.05, 0.86, 0.88);
      s.position.x = x;
      f.add(s);
    });

    const head = new THREE.Mesh(new THREE.SphereGeometry(0.135, 18, 14), cuticle);
    head.scale.set(0.92, 1.0, 1.0);
    head.position.x = 0.2;
    f.add(head);
    // Mouthparts, tucked under the head. Any further forward and it reads as a muzzle,
    // which makes the whole animal look like a small mammal rather than a fly.
    const face = new THREE.Mesh(new THREE.SphereGeometry(0.058, 14, 12), dark);
    face.scale.set(0.45, 0.7, 0.7);
    face.position.set(0.235, -0.085, 0);
    f.add(face);

    this.eyes = [];
    for (const z of [-0.075, 0.075]) {
      // A fly is mostly eye, and wrapping them around the front is the cue that still
      // reads at the size the page draws this at. It is also the cute reading.
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.1, 18, 14),
        new THREE.MeshBasicMaterial({ color: 0xd8342a }));
      eye.scale.set(0.88, 1.0, 0.85);
      eye.position.set(0.215, 0.035, z * 1.2);
      f.add(eye);
      this.eyes.push(eye);
      // A highlight sphere is what makes a round eye read as round.
      const glint = new THREE.Mesh(new THREE.SphereGeometry(0.025, 8, 8),
        new THREE.MeshBasicMaterial({ color: 0xffd7cf }));
      glint.position.set(0.275, 0.095, z * 1.3);
      f.add(glint);
      // Aristae rise from the top of the head between the eyes. Pointed forward instead,
      // they sit either side of the mouthparts and the whole face reads as a snout.
      const ant = new THREE.Mesh(new THREE.CapsuleGeometry(0.008, 0.09, 4, 8), dark);
      ant.position.set(0.235, 0.16, z * 0.55);
      ant.rotation.z = 0.32;
      ant.rotation.x = z > 0 ? -0.28 : 0.28;
      f.add(ant);
    }

    // Each wing hangs off a hinge at its root, so flapping sweeps the tip rather than
    // rotating the whole wing about its own middle.
    this.wings = [];
    for (const z of [-0.07, 0.07]) {
      const hinge = new THREE.Group();
      hinge.position.set(-0.08, 0.12, z);
      const w = new THREE.Mesh(new THREE.SphereGeometry(0.3, 14, 10),
        new THREE.MeshBasicMaterial({ color: C.wing, transparent: true, opacity: 0.22 }));
      // Long and narrow, angled back along the body. Broad and square-on, a wing reads
      // as a grey plate stuck to the fly rather than as a wing.
      w.scale.set(1.0, 0.035, 0.25);
      w.position.set(-0.22, 0.01, Math.sign(z) * 0.2);
      w.rotation.y = -Math.sign(z) * 0.32;
      hinge.add(w);
      f.add(hinge);
      this.wings.push(hinge);
    }

    // Two-segment legs with a knee. update() still swings each leg by rotation.z, so
    // the group is hinged at the hip and the segments hang inside it.
    this.legs = [];
    for (let i = 0; i < 6; i++) {
      const side = i % 2 ? 1 : -1;
      const row = Math.floor(i / 2);
      const hip = new THREE.Group();
      hip.position.set(0.08 - row * 0.16, -0.12, side * 0.13);
      hip.rotation.x = side * 0.7;
      const femur = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.012, 0.18, 6), dark);
      femur.position.y = -0.09;
      hip.add(femur);
      const knee = new THREE.Group();
      knee.position.y = -0.18;
      knee.rotation.z = -0.8;
      const tibia = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.007, 0.17, 6), dark);
      tibia.position.y = -0.085;
      knee.add(tibia);
      const foot = new THREE.Mesh(new THREE.SphereGeometry(0.014, 8, 6), dark);
      foot.position.y = -0.17;
      knee.add(foot);
      hip.add(knee);
      f.add(hip);
      this.legs.push({ mesh: hip, side, row });
    }

    // Dopamine halo -- flares cyan when the fly is right, red when it is wrong.
    this.halo = new THREE.Mesh(new THREE.TorusGeometry(0.46, 0.02, 8, 36),
      new THREE.MeshBasicMaterial({ color: C.pam, transparent: true, opacity: 0 }));
    f.add(this.halo);

    // The fly carries its own warm light: at true scale it would be a speck,
    // and the whole point is to watch it work.
    const lamp = new THREE.PointLight(C.amber, 1.5, 3.2);
    lamp.position.set(0, 0.1, 0);
    f.add(lamp);
    this.flyLamp = lamp;

    f.position.set(-0.4, 2.15, 0.9);
    f.scale.setScalar(2.3);
    this.scene.add(f);
    this.fly = f;
    this.flyTarget = f.position.clone();
    this.flyFacing = 0;
  }

  /* -- driving the scene -- */

  setRhythm(r) {
    this.rhythm = r;
    this.roscLight.intensity = r === 'ROSC' ? 0.8 : 0;
  }

  /** Point the fly at whatever the action needs, and fire off any one-shot effect. */
  setAction(action, correct) {
    const THREE = this.THREE;
    this.action = action;
    const at = (x, y, z) => new THREE.Vector3(x, y, z);

    const spots = {
      rhythm_check:     at(-1.85, 2.15, -0.45),
      shock:            at(-0.4, 1.95, 0.0),
      cpr:              at(-0.3, 1.72, 0.6),
      epinephrine:      at(-0.15, 1.5, 0.75),
      amiodarone:       at(-0.15, 1.5, 0.75),
      lidocaine:        at(-0.15, 1.5, 0.75),
      access:           at(-0.1, 1.42, 0.78),
      airway:           at(-1.6, 1.72, 0.05),
      treat_cause:      at(1.1, 1.72, 0.7),
      post_arrest_care: at(-1.2, 2.0, 0.9),
    };
    this.flyTarget = (spots[action] ?? at(0, 2.1, 1.1)).clone();

    if (action === 'shock') { this.flashV = 1; this.joltV = 1; }
    if (DRUG_COLOR[action]) this.carrying = action;
    else if (action !== 'access') this.carrying = null;

    this.verdictV = 1;
    this.verdictColor = correct ? C.pam : C.ppl1;
    this.halo.material.color.setHex(this.verdictColor);
  }

  _drawECG() {
    const ctx = this.ecgCtx, W = this.ecgCanvas.width, H = this.ecgCanvas.height;
    ctx.fillStyle = '#05080b';
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(67,224,138,0.10)';
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 32) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 32) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    ctx.strokeStyle = '#43e08a';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    const mid = H * 0.52, span = 3.4;
    for (let px = 0; px < W; px++) {
      const tt = this.t - span * (1 - px / W);
      const y = mid - ecgAt(this.rhythm, tt) * H * 0.3;
      px ? ctx.lineTo(px, y) : ctx.moveTo(px, y);
    }
    ctx.stroke();

    ctx.fillStyle = '#43e08a';
    ctx.font = '600 30px ui-monospace, monospace';
    ctx.fillText(this.rhythm, 16, 40);
    this.ecgTex.needsUpdate = true;
  }

  update(dt) {
    const THREE = this.THREE;
    this.t += dt;
    this._drawECG();

    // Wings blur at a speed no one can follow, as they should.
    const flap = Math.sin(this.t * 62) * 0.85;
    this.wings[0].rotation.x = flap;
    this.wings[1].rotation.x = -flap;

    // Approach the target with a little hover wobble.
    const f = this.fly;
    const wob = new THREE.Vector3(
      Math.sin(this.t * 2.3) * 0.035,
      Math.sin(this.t * 3.1) * 0.045,
      Math.cos(this.t * 1.9) * 0.035,
    );
    const goal = this.flyTarget.clone().add(wob);
    f.position.lerp(goal, Math.min(1, dt * 3.4));
    const dir = goal.clone().sub(f.position);
    if (dir.lengthSq() > 1e-5) {
      this.flyFacing += (Math.atan2(dir.x, dir.z) - this.flyFacing) * Math.min(1, dt * 3);
    }
    f.rotation.y = this.flyFacing - Math.PI / 2;
    f.rotation.z = Math.sin(this.t * 2.6) * 0.06;

    // Legs paddle while hovering; they reach when the fly is working.
    this.legs.forEach((l, i) => {
      l.mesh.rotation.z = 0.35 + Math.sin(this.t * 7 + i) * 0.12;
    });

    // Chest compressions.
    if (this.action === 'cpr') {
      this.cprPhase += dt * 1.9;                 // ~110 per minute
      const d = Math.max(0, Math.sin(this.cprPhase * Math.PI * 2)) * 0.12;
      this.chest.position.y = 1.18 - d;
      this.chest.scale.y = 0.62 - d * 0.35;
    } else {
      this.chest.position.y += (1.18 - this.chest.position.y) * Math.min(1, dt * 5);
      this.chest.scale.y += (0.62 - this.chest.scale.y) * Math.min(1, dt * 5);
    }

    // Carried syringe rides just under the fly; otherwise it sits on the cart.
    for (const [name, s] of Object.entries(this.syringes)) {
      const home = this.syringeHome[name];
      const goalPos = this.carrying === name
        ? f.position.clone().add(new THREE.Vector3(0.1, -0.18, 0))
        : home;
      s.position.lerp(goalPos, Math.min(1, dt * 6));
      s.rotation.z = this.carrying === name ? -0.5 : 0;
    }

    // Paddles ride with the fly while it is shocking.
    const padGoal = this.action === 'shock'
      ? f.position.clone().add(new THREE.Vector3(0, -0.3, 0))
      : new THREE.Vector3(2.7, 1.66, -1.1);
    this.paddles.position.lerp(padGoal, Math.min(1, dt * 5));

    // Defibrillator charge lamp.
    this.defibLamp.material.color.setHex(this.action === 'shock' ? 0xffffff : C.amber);

    // Shock flash and the jolt it puts through the patient.
    if (this.flashV > 0) {
      this.flashV = Math.max(0, this.flashV - dt * 2.6);
      const v = this.flashV;
      this.scene.background.setRGB(0.027 + v * 0.75, 0.035 + v * 0.8, 0.047 + v * 0.7);
      for (const p of this.pads) p.material.color.setRGB(v, v, v);
    }
    if (this.joltV > 0) {
      this.joltV = Math.max(0, this.joltV - dt * 3.2);
      this.patient.position.y = Math.sin(this.joltV * 18) * this.joltV * 0.06;
      this.patient.rotation.z = Math.sin(this.joltV * 14) * this.joltV * 0.02;
    }

    // Dopamine halo.
    if (this.verdictV > 0) {
      this.verdictV = Math.max(0, this.verdictV - dt * 1.5);
      this.halo.material.opacity = this.verdictV * 0.85;
      this.halo.scale.setScalar(1 + (1 - this.verdictV) * 1.1);
      this.halo.rotation.x = Math.PI / 2;
      this.halo.rotation.z = this.t * 1.4;
    }

    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Set the camera's distance so the whole bay is in frame. A hand-set distance suited
   * the wide strip the page used to be; in a half-width column it cut the defibrillator
   * off the right edge. Widening the lens instead would distort the near end of the bed.
   */
  _frame() {
    const { Vector3 } = this.THREE;
    const cam = this.camera;
    cam.position.copy(this.viewDir).multiplyScalar(BASE_DIST).add(this.target);
    cam.lookAt(this.target);
    cam.updateMatrixWorld(true);

    const tanV = Math.tan((cam.fov * Math.PI) / 360), tanH = tanV * cam.aspect;
    let push = 0;
    for (let i = 0; i < 8; i++) {
      // Camera space looks down -z, so a corner in front has negative z and needs the
      // camera moved back by |x| / tan(half fov) + z before it is inside the frustum.
      const v = cam.worldToLocal(new Vector3(
        BAY_BOX[i & 1 ? 'max' : 'min'][0],
        BAY_BOX[i & 2 ? 'max' : 'min'][1],
        BAY_BOX[i & 4 ? 'max' : 'min'][2]));
      push = Math.max(push, Math.abs(v.x) / tanH + v.z, Math.abs(v.y) / tanV + v.z);
    }
    if (push > 0) cam.position.addScaledVector(this.viewDir, push * 1.04);
    cam.lookAt(this.target);

    // The fog was tuned as a depth behind the patient, not as an absolute distance.
    const d = cam.position.distanceTo(this.target);
    this.scene.fog.near = d + 3.6;
    this.scene.fog.far = d + 18.6;
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    const w = Math.max(1, r.width), h = Math.max(1, r.height);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this._frame();
  }
}
