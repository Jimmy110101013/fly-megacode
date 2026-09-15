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
    this.camera.position.set(4.6, 3.0, 5.4);
    this.camera.lookAt(-0.25, 1.35, 0);

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
    g.add(new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.62, 0.7), this._mat(0x243140)));
    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.07, 10, 8),
      new THREE.MeshBasicMaterial({ color: C.amber }));
    lamp.position.set(0, 0.38, 0.2);
    g.add(lamp);
    this.defibLamp = lamp;
    g.position.set(2.7, 1.25, -1.1);
    this.scene.add(g);

    this.paddles = new THREE.Group();
    for (const z of [-0.22, 0.22]) {
      const p = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.07, 14), this._mat(0x8b96a4));
      p.position.set(0, 0, z);
      this.paddles.add(p);
    }
    this.paddles.position.copy(new THREE.Vector3(2.7, 1.72, -1.1));
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

    const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.2, 18, 14), cuticle);
    thorax.scale.set(1.0, 0.86, 0.9);
    f.add(thorax);

    const abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.19, 18, 14), dark);
    abdomen.scale.set(1.55, 0.8, 0.82);
    abdomen.position.x = -0.32;
    f.add(abdomen);
    for (let i = 0; i < 3; i++) {                       // abdominal banding
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.15 - i * 0.022, 0.012, 6, 16), cuticle);
      ring.rotation.y = Math.PI / 2;
      ring.position.x = -0.24 - i * 0.11;
      f.add(ring);
    }

    const head = new THREE.Mesh(new THREE.SphereGeometry(0.135, 16, 12), cuticle);
    head.position.x = 0.2;
    f.add(head);
    this.eyes = [];
    for (const z of [-0.085, 0.085]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.082, 14, 12),
        new THREE.MeshBasicMaterial({ color: 0xd8342a }));
      eye.scale.set(0.8, 1.05, 0.85);
      eye.position.set(0.22, 0.02, z);
      f.add(eye);
      this.eyes.push(eye);
    }

    this.wings = [];
    for (const z of [-0.1, 0.1]) {
      const w = new THREE.Mesh(new THREE.SphereGeometry(0.29, 12, 8),
        new THREE.MeshBasicMaterial({ color: C.wing, transparent: true, opacity: 0.22 }));
      w.scale.set(1.0, 0.045, 0.34);
      w.position.set(-0.13, 0.13, z);
      f.add(w);
      this.wings.push(w);
    }

    this.legs = [];
    for (let i = 0; i < 6; i++) {
      const side = i % 2 ? 1 : -1;
      const seg = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.009, 0.3, 6), dark);
      seg.position.set(0.08 - Math.floor(i / 2) * 0.16, -0.13, side * 0.14);
      seg.rotation.x = side * 0.7;
      seg.rotation.z = 0.35;
      f.add(seg);
      this.legs.push({ mesh: seg, side, row: Math.floor(i / 2) });
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
      : new THREE.Vector3(2.7, 1.72, -1.1);
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

  resize() {
    const r = this.canvas.getBoundingClientRect();
    const w = Math.max(1, r.width), h = Math.max(1, r.height);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}
