/**
 * Where does the answer get lost?
 *
 *   node pipeline/dn-ceiling.mjs [episodes]
 *
 * The descending-neuron architecture sits at 13.7% against 11.8% chance while the
 * action-menu architecture reaches 71.1%. Before changing anything else, find out
 * WHERE the information stops being there, by putting a plain linear decoder at
 * each stage and granting it perfect learning. This is not biology -- it is a
 * ceiling. Each line answers one question:
 *
 *   Kenyon-cell code   Does the state-only code distinguish states that need
 *                      different actions? If this is low, the 41.9% overlap
 *                      between state codes is the problem and nothing downstream
 *                      can help.
 *   MBON output        Does it survive the 5,177 -> 96 compression through the
 *                      anatomical KC->MBON weights? This is what a plastic relay
 *                      layer would have to read.
 *   channel drive      Does it survive the fixed MBON -> DN pathway? If the answer
 *                      is in the MBON output but not here, the fixed output map is
 *                      the bottleneck and making the relay plastic is the fix.
 *
 * Control: the menu architecture's per-action codes, measured the same way.
 */
import { readFileSync } from 'node:fs';
import { Megacode, ACTIONS, makeFeatureSpace } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';
import { CentralComplex } from '../web/central-complex.js';
import { DescendingReadout } from '../web/descending.js';

const EPISODES = Number(process.argv[2] ?? 1200);
const url = (n) => new URL(`../data/out/${n}`, import.meta.url);
const circuit = JSON.parse(readFileSync(url('mb_circuit_both.json')));
const cxCircuit = JSON.parse(readFileSync(url('cx_circuit.json')));
const dnCircuit = JSON.parse(readFileSync(url('dn_circuit.json')));
const space = makeFeatureSpace({ omitMemory: true, cxGroups: 8, cxLevels: 5 });

const out = new DescendingReadout(dnCircuit, ACTIONS.length);
const fly = new MushroomBody(circuit, space.FEATURES.length, ACTIONS.length,
                             { seed: 11, groups: space.GROUP_SPANS, stateClaws: 6, output: out });
const cx = new CentralComplex(cxCircuit, { readouts: 8, seed: 11 });

/** Play scenarios and record every stage of the untrained pathway. */
function collect(n) {
  const KC = [], MB = [], CH = [], MENU = [], Y = [];
  for (let e = 0; e < n; e++) {
    const mc = new Megacode(e);
    cx.reset(); out.reset();
    for (let s = 0; s < 60 && !mc.state.over; s++) {
      const acc = mc.correctActions();
      if (!acc.length) break;
      const f = space.encode(mc.state, mc.shockable, cx.out);

      const code = fly.stateCode(f);
      const sparse = new Float32Array(fly.nKC);
      for (const k of code) sparse[k] = 1;
      const { mbon, learned } = fly.readout(code);
      KC.push(sparse);
      MB.push(Float32Array.from(mbon));
      CH.push(Float32Array.from(out.drive(mbon)));

      // The menu architecture's representation of the same moment: the value it
      // would score for each action, which is what its softmax reads.
      const menu = new Float32Array(ACTIONS.length);
      fly.primeState(f);
      for (let a = 0; a < ACTIONS.length; a++) menu[a] = fly.readout(fly.encode(f, a, true)).value;
      MENU.push(menu);

      Y.push(new Set(acc));
      mc.step(acc[Math.floor(Math.random() * acc.length)]);
      cx.step(learned);
    }
  }
  return { KC, MB, CH, MENU, Y };
}

/** Multiclass linear decoder, delta rule. Scored as the task is: any accepted action counts. */
function decode(X, Y, label, passes = 20) {
  const D = X[0].length + 1, C = ACTIONS.length;
  const W = new Float32Array(D * C);
  const cut = Math.floor(X.length * 0.75);
  const score = (x, v) => {
    for (let c = 0; c < C; c++) {
      let s = W[c * D + x.length];
      for (let i = 0; i < x.length; i++) if (x[i] !== 0) s += W[c * D + i] * x[i];
      v[c] = s;
    }
    return v;
  };
  const v = new Float32Array(C);
  for (let pass = 0; pass < passes; pass++) {
    const lr = 0.05 / (1 + pass * 0.3);
    for (let n = 0; n < cut; n++) {
      score(X[n], v);
      for (let c = 0; c < C; c++) {
        const err = (Y[n].has(c) ? 1 : -1) - Math.tanh(v[c]);
        const x = X[n];
        for (let i = 0; i < x.length; i++) if (x[i] !== 0) W[c * D + i] += lr * err * x[i];
        W[c * D + x.length] += lr * err;
      }
    }
  }
  let right = 0;
  for (let n = cut; n < X.length; n++) {
    score(X[n], v);
    let a = 0;
    for (let c = 1; c < C; c++) if (v[c] > v[a]) a = c;
    if (Y[n].has(a)) right++;
  }
  const acc = right / (X.length - cut);
  console.log(`  ${label.padEnd(34)} ${(acc * 100).toFixed(1).padStart(5)}%   (${X[0].length} inputs)`);
  return acc;
}

console.log(`collecting from ${EPISODES} megacodes with an untrained fly...`);
const { KC, MB, CH, MENU, Y } = collect(EPISODES);
let chanceHits = 0;
for (const y of Y) chanceHits += y.size / ACTIONS.length;
console.log(`${KC.length} decisions. Chance is ${(chanceHits / Y.length * 100).toFixed(1)}%.\n`);
console.log('a linear decoder, granted perfect learning, reading each stage:\n');
decode(KC, Y, 'Kenyon-cell code (state only)');
decode(MB, Y, 'MBON output, anatomical weights');
decode(CH, Y, 'descending channel drive');
decode(MENU, Y, 'menu architecture, per-action values');
