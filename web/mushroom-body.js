/**
 * The fly -- a mushroom body built on real FlyWire FAFB v783 wiring.
 *
 * What is real, taken from the connectome:
 *   - 2,597 Kenyon cells (right hemisphere), 48 MBONs, 165 DANs
 *   - 33,799 KC->MBON synapses with their measured synapse counts (the initial weights)
 *   - which DAN innervates which MBON compartment (831 DAN->MBON edges)
 *   - the PAM (reward, n=153) / PPL1 (punishment, n=12) split, which is also where
 *     each MBON's behavioural polarity comes from -- not from its transmitter
 *
 * What is modelled, because a clinical state is not an odour:
 *   - the projection-neuron layer. Each cell samples 5 clinical channels and 2
 *     action channels at random and fires on their coincidence, which is what
 *     Kenyon cells do with olfactory input (Caron et al. 2013).
 *   - APL-style global inhibition, as k-winners-take-all at 1.5% -- 39 of 2,597
 *     cells per pattern. This number matters more than anything else here: at 5%
 *     the codes for two clinically opposite states overlap 80%, the fly learns
 *     their average, and accuracy sits at 37% no matter how long it trains.
 *
 * Learning is the fly's own rule: dopamine released in a compartment depresses
 * the KC->MBON synapses of exactly those Kenyon cells that were active when the
 * dopamine arrived, with slow recovery back toward the anatomical baseline.
 * The decision reads how far those weights have been pushed from that baseline,
 * not their absolute strength -- a naive fly has no innate preference.
 */

const DEFAULTS = {
  stateClaws: 5,     // clinical-context channels sampled per Kenyon cell
  actionClaws: 2,    // candidate-action channels, so the code is conjunctive
  sparsity: 0.015,   // fraction of KCs that survive APL inhibition
  lr: 0.06,          // depression rate
  recovery: 0.010,   // drift back toward the anatomical weight
  // 'channel': a claw listens to one state of one variable and is silent in the
  // others. 'glomerular': a claw follows the whole variable with a weight per
  // state -- anatomically tidier, but it makes every cell respond to everything
  // and accuracy collapses to ~10%. Kept because that negative result is useful.
  coding: 'channel',
  tempFloor: 0.06,   // softmax never anneals below this, so a little exploring never stops
};
const TEMP0 = 1.4;          // softmax temperature, annealed
// Total dopamine delivered per update once the two arms are equalised. Chosen so
// one event moves a channel by a fraction of the drive spread, not all of it.
const NORMALISED = 6.0;

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5;  s >>>= 0;
    return s / 4294967296;
  };
}

export class MushroomBody {
  /**
   * @param circuit  parsed data/out/mb_circuit.json
   * @param nFeatures number of clinical feature channels
   * @param nActions  size of the ACLS action space
   */
  constructor(circuit, nFeatures, nActions, opts = {}) {
    const { seed = 7, weights = 'real', output = null } = opts;
    this.p = { ...DEFAULTS, ...opts };
    // When an output pathway is supplied, the action stops being an input: the
    // Kenyon-cell code depends on the state alone and the action is whichever
    // descending channel wins. See web/descending.js.
    this.output = output;
    if (!Array.isArray(this.p.groups) || !this.p.groups.length) {
      throw new Error('MushroomBody needs `groups`: pass GROUP_SPANS from acls-engine.js');
    }
    this.circuit = circuit;
    this.nKC = circuit.kc.length;
    this.nMBON = circuit.mbon.length;
    this.nDAN = circuit.dan.length;
    this.nFeatures = nFeatures;
    this.nActions = nActions;
    this.nInput = nFeatures + nActions;      // state channels + the candidate action
    this.rand = rng(seed);
    this.trials = 0;

    this._buildProjection();
    this._buildKCtoMBON(weights);
    this._buildCompartments();

    this.kActive = Math.max(8, Math.round(this.nKC * this.p.sparsity));
    this.last = null;                        // activity trace for the visualisation
  }

  /**
   * Projection-neuron layer. A Kenyon cell claw contacts one glomerulus -- one
   * clinical variable -- and carries a different weight for each state that
   * variable can be in. Sampling single channels instead makes a one-bit change
   * (airway in / not in) invisible to ~80% of the population; sampling whole
   * glomeruli makes every cell that watches that variable respond differently.
   */
  _buildProjection() {
    const SC = this.p.stateClaws, AC = this.p.actionClaws;
    const groups = this.p.groups;          // [[channelStart, channelCount], ...]
    const byChan = Array.from({ length: this.nInput }, () => []);

    for (let k = 0; k < this.nKC; k++) {
      // distinct glomeruli, sampled without replacement
      const picked = new Set();
      while (picked.size < Math.min(SC, groups.length)) {
        picked.add(Math.floor(this.rand() * groups.length));
      }
      for (const g of picked) {
        const [start, len] = groups[g];
        if (this.p.coding === 'channel') {
          // claw listens to one state of the variable, silent in every other
          const c = Math.floor(this.rand() * len);
          byChan[start + c].push(k, 0.5 + this.rand());
        } else {
          // claw follows the whole glomerulus, with its own weight per state
          for (let c = 0; c < len; c++) byChan[start + c].push(k, 0.25 + this.rand() * 1.5);
        }
      }
      // action claws stay channel-specific: a cell reports "this action", and the
      // multiplicative gate downstream needs cells that are silent for the others
      const seen = new Set();
      while (seen.size < Math.min(AC, this.nActions)) seen.add(Math.floor(this.rand() * this.nActions));
      for (const a of seen) byChan[this.nFeatures + a].push(k, 0.5 + this.rand());
    }

    this.chanStart = new Int32Array(this.nInput + 1);
    for (let c = 0; c < this.nInput; c++) {
      this.chanStart[c + 1] = this.chanStart[c] + byChan[c].length / 2;
    }
    this.chanKC = new Int32Array(this.chanStart[this.nInput]);
    this.chanW = new Float32Array(this.chanStart[this.nInput]);
    for (let c = 0, p = 0; c < this.nInput; c++) {
      for (let i = 0; i < byChan[c].length; i += 2, p++) {
        this.chanKC[p] = byChan[c][i];
        this.chanW[p] = byChan[c][i + 1];
      }
    }
    this._drive = new Float32Array(this.nKC);
    this._stateDrive = new Float32Array(this.nKC);
  }

  _addChannel(target, ch, x) {
    for (let p = this.chanStart[ch]; p < this.chanStart[ch + 1]; p++) {
      target[this.chanKC[p]] += x * this.chanW[p];
    }
  }

  /** Top-k by histogram threshold -- O(n), no sort. This is the APL bottleneck. */
  _topK(drive, k) {
    const NB = 128;
    let max = 0;
    for (let i = 0; i < drive.length; i++) if (drive[i] > max) max = drive[i];
    if (max <= 0) return [];
    const bins = new Int32Array(NB + 1);
    const scale = NB / max;
    for (let i = 0; i < drive.length; i++) {
      if (drive[i] > 0) bins[Math.min(NB, (drive[i] * scale) | 0)]++;
    }
    let cum = 0, cutBin = 0;
    for (let b = NB; b >= 0; b--) { cum += bins[b]; if (cum >= k) { cutBin = b; break; } }
    const thr = cutBin / scale;
    const out = [];
    for (let i = 0; i < drive.length && out.length < k; i++) if (drive[i] >= thr) out.push(i);
    return out;
  }

  /** Sparse KC -> MBON matrix, initialised from measured synapse counts. */
  _buildKCtoMBON(mode) {
    const edges = this.circuit.kc_mbon;
    const rowCount = new Int32Array(this.nKC);
    for (const [k] of edges) rowCount[k]++;

    this.rowStart = new Int32Array(this.nKC + 1);
    for (let k = 0; k < this.nKC; k++) this.rowStart[k + 1] = this.rowStart[k] + rowCount[k];
    const n = edges.length;
    this.colIdx = new Int32Array(n);
    this.w = new Float32Array(n);
    this.w0 = new Float32Array(n);

    const fill = new Int32Array(this.nKC);
    const real = edges.map((e) => e[2]);
    const mean = real.reduce((a, b) => a + b, 0) / real.length;
    // Own RNG so every weight mode sees the same draw sequence and stays comparable.
    const wr = rng(0xC0FFEE);
    let syn;
    switch (mode) {
      case 'shuffled':                       // same multiset, reassigned across all edges
        syn = this._shuffle(real.slice(), wr); break;
      case 'resampled':                      // same marginal distribution, drawn independently
        syn = real.map(() => real[Math.floor(wr() * real.length)]); break;
      case 'random':                         // uniform weights -- different distribution entirely
        syn = real.map(() => wr() * 2 * mean); break;
      default:
        syn = real;
    }

    edges.forEach(([k, m], i) => {
      const p = this.rowStart[k] + fill[k]++;
      this.colIdx[p] = m;
      const v = syn[i] / mean;
      this.w[p] = v;
      this.w0[p] = v;
    });
  }

  _shuffle(a, r = this.rand) {
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  /**
   * MBON polarity from transmitter identity, and which MBON compartments each
   * DAN population drives.
   */
  _buildCompartments() {
    this.isPAM = this.circuit.dan.map((d) => String(d.type).startsWith('PAM'));
    // Per-MBON dopamine drive from the reward and the punishment population.
    this.pamDrive = new Float32Array(this.nMBON);
    this.ppl1Drive = new Float32Array(this.nMBON);
    for (const [d, m, w] of this.circuit.dan_mbon) {
      (this.isPAM[d] ? this.pamDrive : this.ppl1Drive)[m] += w;
    }
    const norm = (v) => {
      const mx = Math.max(...v) || 1;
      for (let i = 0; i < v.length; i++) v[i] /= mx;
    };
    norm(this.pamDrive); norm(this.ppl1Drive);

    /*
     * Behavioural polarity comes from which dopamine population owns the
     * compartment, not from the MBON's own transmitter. MBON-g1pedc is
     * GABAergic yet drives approach, so transmitter identity gets the sign
     * backwards. The compartment rule is what makes a depression-only plasticity
     * rule work at all: punishment (PPL1) depresses approach compartments,
     * reward (PAM) depresses avoidance compartments.
     */
    this.mbonSign = new Float32Array(this.nMBON);
    for (let j = 0; j < this.nMBON; j++) {
      this.mbonSign[j] = this.ppl1Drive[j] > this.pamDrive[j] ? 1 : -1;
    }
  }

  /**
   * Sparse KC code for one (clinical state, candidate action) pair.
   * The state half of the drive is shared across candidate actions, so it is
   * computed once per decision and only the action channel is added per action.
   */
  primeState(features) {
    this._stateDrive.fill(0);
    for (let f = 0; f < this.nFeatures; f++) {
      if (features[f]) this._addChannel(this._stateDrive, f, features[f]);
    }
  }

  encode(features, action, primed = false) {
    if (!primed) this.primeState(features);
    /*
     * Kenyon cells are coincidence detectors: their spike threshold is high
     * enough that they need several claws driven at once. Modelling that as a
     * product rather than a sum means a KC reports "this context AND this
     * action", which is what makes codes for different actions separable.
     */
    this._drive.fill(0);
    this._addChannel(this._drive, this.nFeatures + action, 1);
    for (let k = 0; k < this.nKC; k++) {
      this._drive[k] = this._drive[k] > 0 ? this._drive[k] * this._stateDrive[k] : 0;
    }
    // APL global inhibition -> keep the top k.
    return this._topK(this._drive, this.kActive);
  }

  /**
   * MBON population response to a KC code, and the net approach drive it carries.
   *
   * `mbon` is what the cells actually do -- raw synaptic drive, which is what the
   * atlas draws. The decision, though, reads the *balance*: how far each
   * compartment has been pushed from its anatomical resting weight. A naive fly
   * has no innate preference between actions, and learning is what tips the
   * scale. Reading absolute weight instead bakes in a permanent anatomical bias
   * toward whichever action happens to land on heavier synapses, and with sparse
   * reward the recovery term keeps dragging the animal back to that bias.
   */
  readout(activeKC) {
    const mbon = new Float32Array(this.nMBON);
    const learned = new Float32Array(this.nMBON);
    for (const k of activeKC) {
      for (let p = this.rowStart[k]; p < this.rowStart[k + 1]; p++) {
        const j = this.colIdx[p];
        mbon[j] += this.w[p];
        learned[j] += this.w[p] - this.w0[p];
      }
    }
    let value = 0;
    for (let j = 0; j < this.nMBON; j++) value += this.mbonSign[j] * learned[j];
    // `learned` is what actually leaves the mushroom body: the raw sums differ by
    // under 4% between actions, because averaging over ~78 random Kenyon cells
    // washes the difference out. The deviation from baseline is the signal.
    return { mbon, learned, value: value / Math.max(1, activeKC.length) };
  }

  /**
   * The Kenyon-cell code for a state, with no action in it. Used when the action
   * comes out of the descending pathway instead of being scored from a menu.
   */
  stateCode(features) {
    this.primeState(features);
    return this._topK(this._stateDrive, this.kActive);
  }

  /**
   * Decide by letting the descending neurons compete, rather than by scoring ten
   * labelled options. One state, one Kenyon-cell code, one MBON population
   * response; the fixed MBON -> DN pathway turns that into channel drives and the
   * action is the channel that wins.
   */
  act(features, { greedy = false } = {}) {
    const code = this.stateCode(features);
    const { mbon } = this.readout(code);
    const out = this.output.drive(mbon);

    const temp = greedy ? 0.05 : Math.max(this.p.tempFloor, TEMP0 * Math.exp(-this.trials / 220));
    const logits = Array.from(out, (v) => v / temp);
    const mx = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - mx));
    const tot = exps.reduce((a, b) => a + b, 0);
    let r = this.rand() * tot, chosen = exps.length - 1;
    for (let i = 0; i < exps.length; i++) { r -= exps[i]; if (r <= 0) { chosen = i; break; } }

    // The strongest channel that is not the one taken -- what the dopamine has to
    // push against when the action was right.
    let rival = chosen === 0 ? 1 : 0;
    for (let i = 0; i < out.length; i++) if (i !== chosen && out[i] > out[rival]) rival = i;

    this.last = {
      action: chosen, values: Array.from(out), activeKC: code, mbon,
      learned: this.readout(code).learned, temp,
      blame: this.output.contribution(chosen),
      credit: this.output.contribution(rival),
      // The same responsibility signal one layer down, for when MBON -> relay is
      // plastic too. Snapshotted here because the relay state is overwritten by
      // the next decision.
      blameRelay: this.output.relayContribution(chosen),
      creditRelay: this.output.relayContribution(rival),
    };
    return chosen;
  }

  /** Evaluate every candidate action and pick one. */
  decide(features, allowed = null, { greedy = false } = {}) {
    const codes = [], values = new Float32Array(this.nActions);
    let mbonOfBest = null;
    this.primeState(features);
    for (let a = 0; a < this.nActions; a++) {
      const code = this.encode(features, a, true);
      const { mbon, learned, value } = this.readout(code);
      codes.push({ code, mbon, learned });
      values[a] = value;
    }
    const pool = allowed ?? [...Array(this.nActions).keys()];
    const temp = greedy ? 0.05 : Math.max(this.p.tempFloor, TEMP0 * Math.exp(-this.trials / 220));
    const logits = pool.map((a) => values[a] / temp);
    const mx = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - mx));
    const tot = exps.reduce((s, e) => s + e, 0);
    let r = this.rand() * tot, chosen = pool[pool.length - 1];
    for (let i = 0; i < pool.length; i++) { r -= exps[i]; if (r <= 0) { chosen = pool[i]; break; } }

    mbonOfBest = codes[chosen];
    this.last = {
      action: chosen, values: Array.from(values),
      activeKC: mbonOfBest.code, mbon: mbonOfBest.mbon,
      learned: mbonOfBest.learned,
      temp,
    };
    return chosen;
  }

  /**
   * Dopamine-gated plasticity. Reward recruits PAM, error recruits PPL1; in both
   * cases the dopamine depresses KC->MBON synapses of the Kenyon cells that were
   * active, inside the compartments that population innervates.
   */
  learn(correct) {
    const trace = this.last;
    if (!trace) return null;
    const dopamine = correct ? this.pamDrive : this.ppl1Drive;
    let touched = 0, totalDelta = 0;

    /**
     * With the action gone from the Kenyon-cell code there is one code per state,
     * so dopamine can no longer be made action-specific by the code alone: which
     * compartment it reaches has to carry that. `blame` is how much each MBON
     * pushed the channel that was taken, `credit` how much it pushed the runner-up.
     * Wrong -- depress what drove the action, so it drops. Right -- depress what
     * drove its rival, so the action wins by more. Both directions are depression,
     * which is the only thing this synapse does.
     *
     * The anatomical warrant is the mushroom body's own MBON -> DAN feedback; the
     * exact form of the eligibility signal is a modelling assumption.
     */
    let elig = null;
    if (trace.blame) {
      const src = correct ? trace.credit : trace.blame;
      // The two arms must deliver the same total depression. Measured raw, PPL1 x
      // blame carries 48.9 where PAM x credit carries 3.0 -- a 16x asymmetry, and
      // early in training almost every trial is an error, so the punishment arm
      // crushes the weights into a corner the reward arm cannot pull them out of.
      // Lowering the learning rate cannot fix a ratio; it only freezes the animal
      // on its anatomical prior sooner, which is what the sweep showed.
      elig = new Float32Array(src.length);
      let tot = 0;
      for (let j = 0; j < src.length; j++) {
        const v = dopamine[j] * Math.max(0, src[j]);
        elig[j] = v; tot += v;
      }
      if (tot > 0) for (let j = 0; j < elig.length; j++) elig[j] /= tot;
    }

    for (const k of trace.activeKC) {
      for (let p = this.rowStart[k]; p < this.rowStart[k + 1]; p++) {
        const j = this.colIdx[p];
        const d = elig ? elig[j] * NORMALISED : dopamine[j];
        if (d > 0) {
          const delta = -this.p.lr * d * this.w[p];
          this.w[p] += delta;
          totalDelta += delta;
          touched++;
        }
        this.w[p] += this.p.recovery * (this.w0[p] - this.w[p]);   // anatomical recovery
      }
    }
    // The output pathway learns under the same dopamine, if it is plastic at all.
    const relay = trace.blameRelay
      ? this.output.learn(correct, trace.mbon, trace.blameRelay, trace.creditRelay)
      : null;

    this.trials++;
    return { touched, totalDelta, relay, dopamine: correct ? 'PAM' : 'PPL1' };
  }

  /**
   * How much each Kenyon cell's output has been driven away from its anatomical
   * baseline -- the engram. This is what the memory overlay draws.
   */
  engramPerKC(out = new Float32Array(this.nKC)) {
    for (let k = 0; k < this.nKC; k++) {
      let d = 0;
      for (let p = this.rowStart[k]; p < this.rowStart[k + 1]; p++) d += this.w0[p] - this.w[p];
      out[k] = d > 0 ? d : 0;
    }
    return out;
  }

  /** Total synaptic drive -- watch this to catch runaway depression. */
  weightMass() {
    let s = 0;
    for (let i = 0; i < this.w.length; i++) s += this.w[i];
    return s / this.w.length;
  }
}
