/**
 * The central complex as a fixed recurrent reservoir.
 *
 * The mushroom body stores associations in its synapses but carries nothing
 * between decisions. A fly that had to run a megacode would need to hold "that
 * was the third shock" somewhere, and the central complex is where persistent
 * activity lives: the ellipsoid body and protocerebral bridge sustain a heading
 * in complete darkness, which is a state held by recurrent wiring rather than by
 * changing synapses.
 *
 * So nothing here is trained. The recurrent matrix is the animal's own CX→CX
 * connectivity (311,710 connections among 2,875 neurons), signed by transmitter,
 * and the drive into it is the real MBON→CX pathway (1,368 connections) carrying
 * the mushroom body's own output. The plastic part of the loop stays where the
 * fly actually has plasticity: the KC→MBON synapses reading this state back.
 *
 * Which cells are read out is modelled, not anatomical -- the hub neurons, by
 * synaptic in-degree, stand in for the projections that leave the CX.
 */

const SIGN_SCALE = 1 / 40;      // synapse counts -> weights before normalisation

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

export class CentralComplex {
  /**
   * @param circuit  parsed data/out/cx_circuit.json
   * @param opts     { readouts, rho, leak, drive, minSyn, seed }
   */
  constructor(circuit, opts = {}) {
    const {
      readouts = 12,   // how many hub cells the mushroom body gets to see
      rho = 0.97,      // spectral radius; below 1 so the state fades rather than blows up
      leak = 0.18,     // low: counting shocks means integrating over ~20 decisions
      drive = 1.6,     // gain on the MBON -> CX pathway
      minSyn = 2,      // ignore the weakest connections
      seed = 3,
    } = opts;
    this.p = { readouts, rho, leak, drive, minSyn, seed };
    this.n = circuit.neurons.length;
    this.rand = rng(seed);

    const sign = Float32Array.from(circuit.neurons, (x) => x.sign);
    const edges = circuit.edges.filter((e) => e[2] >= minSyn);
    this.pre = Int32Array.from(edges, (e) => e[0]);
    this.post = Int32Array.from(edges, (e) => e[1]);
    this.w = Float32Array.from(edges, (e) => sign[e[0]] * e[2] * SIGN_SCALE);
    this.nEdges = edges.length;

    this._normalise();
    this._buildDrive(circuit);
    this._pickReadouts(circuit);

    this.h = new Float32Array(this.n);
    this.buf = new Float32Array(this.n);
    this.out = new Float32Array(readouts);
    /*
     * Per-channel gain control. The reservoir's readouts sit in a narrow band
     * around zero, so a downstream layer slicing [-1, 1] into levels puts 92% of
     * everything in the middle bucket and learns nothing. Adapting each channel to
     * its own spread is what sensory pathways do anyway.
     */
    this.mu = new Float32Array(readouts);
    this.mad = new Float32Array(readouts).fill(0.05);
  }

  /** Scale the recurrent matrix to a chosen spectral radius, by power iteration. */
  _normalise() {
    let v = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) v[i] = this.rand() - 0.5;
    let lambda = 1;
    const next = new Float32Array(this.n);
    for (let it = 0; it < 30; it++) {
      next.fill(0);
      for (let e = 0; e < this.nEdges; e++) next[this.post[e]] += this.w[e] * v[this.pre[e]];
      let norm = 0;
      for (let i = 0; i < this.n; i++) norm += next[i] * next[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-12) break;
      lambda = norm;
      for (let i = 0; i < this.n; i++) v[i] = next[i] / norm;
    }
    this.spectral = lambda;
    const k = this.p.rho / (lambda || 1);
    for (let e = 0; e < this.nEdges; e++) this.w[e] *= k;
  }

  /** MBON -> CX, the mushroom body's own output pathway. */
  _buildDrive(circuit) {
    const d = circuit.mbon_cx ?? [];
    this.dPre = Int32Array.from(d, (e) => e[0]);
    this.dPost = Int32Array.from(d, (e) => e[1]);
    let mx = 1;
    for (const e of d) mx = Math.max(mx, e[2]);
    this.dW = Float32Array.from(d, (e) => e[2] / mx);
    this.nDrive = d.length;
    this.nMbon = circuit.n_mbon ?? 0;
  }

  /** Hub cells by in-degree: the ones with most to integrate. */
  _pickReadouts(circuit) {
    const deg = new Float64Array(this.n);
    for (const [, j, w] of circuit.edges) deg[j] += w;
    const order = Array.from(deg.keys()).sort((a, b) => deg[b] - deg[a]);
    this.readout = Int32Array.from(order.slice(0, this.p.readouts));
  }

  /** New patient, clear head. Gain control persists: it is a property of the
   *  pathway, not of the episode. */
  reset() { this.h.fill(0); this.out.fill(0); }

  /**
   * One decision's worth of integration.
   * @param mbon  MBON activity from the mushroom body's readout, or null
   */
  step(mbon) {
    const { leak, drive } = this.p;
    const buf = this.buf;
    buf.fill(0);
    for (let e = 0; e < this.nEdges; e++) buf[this.post[e]] += this.w[e] * this.h[this.pre[e]];

    if (mbon && this.nDrive) {
      // The drive pathway was extracted for both hemispheres. A model running on
      // one hemisphere has half as many MBONs, and indexing past the end poisons
      // the whole reservoir with NaN, silently -- so the range is checked.
      let mx = 1e-9;
      for (let i = 0; i < mbon.length; i++) mx = Math.max(mx, Math.abs(mbon[i]));
      for (let e = 0; e < this.nDrive; e++) {
        const src = this.dPre[e];
        if (src >= mbon.length) continue;
        buf[this.dPost[e]] += drive * this.dW[e] * (mbon[src] / mx);
      }
    }
    for (let i = 0; i < this.n; i++) {
      this.h[i] = (1 - leak) * this.h[i] + leak * Math.tanh(buf[i]);
    }
    for (let i = 0; i < this.readout.length; i++) {
      const raw = this.h[this.readout[i]];
      this.mu[i] += (raw - this.mu[i]) * 0.002;
      const dev = Math.abs(raw - this.mu[i]);
      this.mad[i] += (dev - this.mad[i]) * 0.002;
      this.out[i] = Math.max(-1, Math.min(1, (raw - this.mu[i]) / (2.5 * this.mad[i] + 1e-4)));
    }
    return this.out;
  }

  /** True if the state has gone non-finite -- cheap guard against silent poisoning. */
  broken() {
    for (let i = 0; i < this.n; i++) if (!Number.isFinite(this.h[i])) return true;
    return false;
  }

  /** Rough measure of how much state is actually being held. */
  energy() {
    let s = 0;
    for (let i = 0; i < this.n; i++) s += this.h[i] * this.h[i];
    return Math.sqrt(s / this.n);
  }
}
