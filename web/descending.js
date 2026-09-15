/**
 * The fly's own output channel, in place of a menu of actions.
 *
 * Until now the action was an *input*: one Kenyon-cell code was built per candidate
 * action and the model scored all ten. A fly has no such menu. Mushroom-body output
 * reaches descending neurons, and the descending neurons are what the brain sends to
 * the body -- so here the action is simply whichever descending channel wins.
 *
 * Everything in this file is anatomy and none of it is trained:
 *
 *   MBON --(direct, 350 edges)--------------------------------> DN
 *   MBON --(7,916)--> 1,951 central interneurons --(27,253)---> DN
 *
 * The relay matters. Only 31 of 96 MBONs reach a descending neuron directly, and
 * pipeline/dn-separable.mjs shows that under depression-only plasticity the direct
 * pathway alone can only ever make 5 of 10 channels win. Through the relay all 96
 * MBONs are on a path and all 10 channels are reachable. That is not a convenience:
 * it is how the pathway is wired in the animal.
 *
 * What this does NOT remove is the naming. Which descending type is called `shock`
 * and which `epinephrine` is assigned by hand, fixed from the anatomical drive
 * ranking before any training and never tuned. The outputs are named, not discovered.
 */

const RELAY_SCALE = 0.02;      // keeps rectified relay output on the same scale as MBON drive
const ADAPT = 0.002;           // how fast each channel adapts to its own drive statistics

export class DescendingReadout {
  /**
   * @param circuit   data/out/dn_circuit.json
   * @param nActions  how many output channels to carve out
   */
  constructor(circuit, nActions) {
    this.nMbon = circuit.n_mbon;
    this.nActions = nActions;
    this.nRelay = circuit.n_relay;

    // One channel per descending *type*, taken off the top of the anatomical drive
    // ranking. Left and right cells of a type are the same command neuron on two
    // sides of the animal, so they are pooled rather than handed separate actions.
    const names = [], seen = new Set();
    for (const j of circuit.by_mb_drive) {
      const t = circuit.neurons[j].t;
      if (t === 'unknown' || seen.has(t)) continue;
      seen.add(t); names.push(t);
      if (names.length === nActions) break;
    }
    if (names.length < nActions) {
      throw new Error(`only ${names.length} descending types carry mushroom-body drive, need ${nActions}`);
    }
    this.channels = names;
    const member = names.map((t) => new Set(circuit.by_mb_drive.filter((j) => circuit.neurons[j].t === t)));
    this.cellCount = member.map((s) => s.size);

    // Direct MBON -> DN, one dense column per channel (96 numbers each -- tiny).
    this.direct = member.map((set) => {
      const v = new Float32Array(this.nMbon);
      for (const [i, j, w] of circuit.mbon_dn) if (set.has(j)) v[i] += w;
      for (let i = 0; i < v.length; i++) v[i] /= set.size;
      return v;
    });

    // MBON -> relay, sparse, because 96 x 1,951 dense would be mostly zero.
    this.rPre = Int32Array.from(circuit.mbon_relay, (e) => e[0]);
    this.rPost = Int32Array.from(circuit.mbon_relay, (e) => e[1]);
    this.rW = Float32Array.from(circuit.mbon_relay, (e) => e[2]);

    // relay -> DN, dense per channel for the same reason as `direct`.
    this.relayed = member.map((set) => {
      const v = new Float32Array(this.nRelay);
      for (const [c, j, w] of circuit.relay_dn) if (set.has(j)) v[c] += w;
      for (let i = 0; i < v.length; i++) v[i] /= set.size;
      return v;
    });

    this.relay = new Float32Array(this.nRelay);
    this.out = new Float32Array(nActions);
    this.raw = new Float32Array(nActions);

    // Per-channel gain. A descending neuron with more input is not thereby more
    // likely to fire: each cell sets its own threshold against its own drive
    // statistics. Without it one loud channel (DNa03 integrates four times the
    // synapses of the next) simply wins every decision on anatomy alone.
    //
    // It is calibrated on the NAIVE animal and then frozen. Left adapting -- the
    // way the central complex runs it, where the signal is within-episode -- it
    // has a time constant of ~500 decisions and quietly absorbs everything the
    // learning does: measured, drives collapse to a mean of 0.00 on every channel
    // and behaviour goes back to uniform. The set point is a property of the
    // pathway, established before learning; the learning then moves within it.
    this.mu = new Float32Array(nActions);
    this.mad = new Float32Array(nActions).fill(1);
    this.scale = new Float32Array(nActions).fill(1);
    this.adapting = true;
    this.calibrated = 0;
  }

  /**
   * Channel drive for one MBON population response. `mbon` is absolute synaptic
   * drive, not the learned deviation: what arrives at a descending neuron is the
   * output the cell actually produces, and depression-only plasticity moves it
   * down from an anatomical baseline rather than around zero.
   */
  drive(mbon) {
    const r = this.relay;
    r.fill(0);
    for (let e = 0; e < this.rPre.length; e++) r[this.rPost[e]] += this.rW[e] * mbon[this.rPre[e]];
    for (let c = 0; c < r.length; c++) r[c] = r[c] > 0 ? r[c] * RELAY_SCALE : 0;   // rectify

    for (let a = 0; a < this.nActions; a++) {
      let s = 0;
      const d = this.direct[a], q = this.relayed[a];
      for (let i = 0; i < this.nMbon; i++) s += d[i] * mbon[i];
      for (let c = 0; c < this.nRelay; c++) if (r[c] > 0) s += q[c] * r[c];
      this.raw[a] = s;
    }
    if (this.adapting) {
      this.calibrated++;
      const rate = Math.max(ADAPT, 1 / this.calibrated);    // converge fast, then settle
      for (let a = 0; a < this.nActions; a++) {
        this.mu[a] += (this.raw[a] - this.mu[a]) * rate;
        this.mad[a] += (Math.abs(this.raw[a] - this.mu[a]) - this.mad[a]) * rate;
        this.scale[a] = 2.5 * this.mad[a] + 1e-3;
      }
    }
    for (let a = 0; a < this.nActions; a++) {
      this.out[a] = (this.raw[a] - this.mu[a]) / this.scale[a];
    }
    return this.out;
  }

  /**
   * How much each MBON contributed to one channel's drive, given the relay state
   * left by the last `drive()`. This is the exact derivative through the rectified
   * relay, and it is what makes dopamine action-specific once the action is no
   * longer part of the Kenyon-cell code: it says which compartments were
   * responsible for the movement that was just made.
   *
   * The anatomical warrant for a signal like this is the MBON -> DAN feedback the
   * mushroom body already carries (3,376 connections in mb_circuit.json). The exact
   * form here is a modelling assumption, not a measurement.
   */
  contribution(channel) {
    const g = Float32Array.from(this.direct[channel]);
    const q = this.relayed[channel], r = this.relay;
    for (let e = 0; e < this.rPre.length; e++) {
      const c = this.rPost[e];
      if (r[c] > 0) g[this.rPre[e]] += this.rW[e] * q[c] * RELAY_SCALE;
    }
    // Through the same gain the channel is read with, so contributions from a
    // quiet channel and a loud one are on one scale.
    const k = 1 / this.scale[channel];
    for (let i = 0; i < g.length; i++) g[i] *= k;
    return g;
  }

  /** Stop tracking, and hold the set point the naive animal arrived at. */
  freeze() { this.adapting = false; }

  /** New patient: the gain persists, it is a property of the pathway. */
  reset() { this.relay.fill(0); }
}
