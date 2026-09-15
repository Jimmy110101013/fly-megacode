/**
 * Is the history actually in the central complex, or does the mushroom body just
 * fail to read it?
 *
 *   node pipeline/cx-decodable.mjs [episodes]
 *
 * Runs scenarios, drives the reservoir exactly as the model does, and trains a
 * plain linear decoder on its readouts to recover the two things the task needs
 * and the observable features no longer carry: how many shocks have been given,
 * and how long ago epinephrine was. This is not biology, it is a ceiling -- if the
 * decoder can read it and the fly cannot, the state is there and the mushroom body
 * is the bottleneck. If the decoder cannot either, the reservoir is not holding it.
 */
import { readFileSync } from 'node:fs';
import { Megacode, ACTIONS, makeFeatureSpace } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';
import { CentralComplex } from '../web/central-complex.js';

const EPISODES = Number(process.argv[2] ?? 1500);
const READOUTS = 16;

const url = (n) => new URL(`../data/out/${n}`, import.meta.url);
const circuit = JSON.parse(readFileSync(url('mb_circuit_both.json')));
const cxCircuit = JSON.parse(readFileSync(url('cx_circuit.json')));
const space = makeFeatureSpace({ omitMemory: true, cxGroups: READOUTS, cxLevels: 5 });

const fly = new MushroomBody(circuit, space.FEATURES.length, ACTIONS.length,
                             { seed: 11, groups: space.GROUP_SPANS });
const cx = new CentralComplex(cxCircuit, { readouts: READOUTS, seed: 11 });

/** Collect (reservoir readout, label) pairs while the fly plays. */
function collect(n, seed0) {
  const X = [], shocks = [], epi = [];
  for (let e = 0; e < n; e++) {
    const mc = new Megacode(seed0 + e);
    cx.reset();
    for (let s = 0; s < 60 && !mc.state.over; s++) {
      const acc = mc.correctActions();
      if (!acc.length) break;
      X.push(Float32Array.from(cx.out));
      shocks.push(Math.min(3, mc.state.shocks));
      const dt = mc.state.t - mc.state.tLastEpi;
      epi.push(mc.state.epiCount === 0 ? 0 : dt >= 180 ? 1 : 2);
      const a = fly.decide(space.encode(mc.state, mc.shockable, cx.out));
      const res = mc.step(a);
      fly.learn(res.correct);
      cx.step(fly.last.learned);
    }
  }
  return { X, shocks, epi };
}

/** Multiclass linear decoder, delta rule, bias included. */
function decode(X, y, nClass, label) {
  const D = X[0].length + 1;
  const W = new Float32Array(D * nClass);
  const cut = Math.floor(X.length * 0.75);
  const score = (x) => {
    const v = new Float32Array(nClass);
    for (let c = 0; c < nClass; c++) {
      let s = W[c * D + X[0].length];
      for (let i = 0; i < x.length; i++) s += W[c * D + i] * x[i];
      v[c] = s;
    }
    return v;
  };
  for (let pass = 0; pass < 24; pass++) {
    const lr = 0.06 / (1 + pass * 0.25);
    for (let n = 0; n < cut; n++) {
      const v = score(X[n]);
      for (let c = 0; c < nClass; c++) {
        const err = (y[n] === c ? 1 : -1) - Math.tanh(v[c]);
        for (let i = 0; i < X[n].length; i++) W[c * D + i] += lr * err * X[n][i];
        W[c * D + X[n].length] += lr * err;
      }
    }
  }
  let right = 0;
  const base = new Array(nClass).fill(0);
  for (let n = cut; n < X.length; n++) {
    const v = score(X[n]);
    let a = 0;
    for (let c = 1; c < nClass; c++) if (v[c] > v[a]) a = c;
    if (a === y[n]) right++;
    base[y[n]]++;
  }
  const majority = Math.max(...base) / (X.length - cut);
  console.log(`  ${label.padEnd(26)} ${(right / (X.length - cut) * 100).toFixed(1).padStart(5)}%` +
              `   (always guessing the commonest class: ${(majority * 100).toFixed(1)}%)`);
}

console.log(`reservoir: ${cx.n} neurons, ${cx.nEdges} edges, ${READOUTS} readouts\n`);
const { X, shocks, epi } = collect(EPISODES, 4000);
console.log(`${X.length} decisions sampled; decoding from the ${READOUTS} readouts alone:\n`);
decode(X, shocks, 4, 'shocks given so far');
decode(X, epi, 3, 'epinephrine timing state');
