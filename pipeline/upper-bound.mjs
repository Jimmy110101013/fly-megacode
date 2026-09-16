/**
 * What is achievable from the feature vector at all?
 *
 *   node pipeline/upper-bound.mjs
 *
 * Trains a plain linear readout (one weight per feature per action, delta rule)
 * on exactly the features the fly sees. This is not biology -- it is a ceiling.
 * If the linear readout scores high, the features carry the answer and any
 * shortfall belongs to the mushroom body's code. If it scores low, the features
 * themselves are missing what the algorithm needs and no encoding can recover it.
 */
import { Megacode, ACTIONS, FEATURES, encodeState } from '../web/acls-engine.js';

const TRAIN = Number(process.argv[2] ?? 8000);
const EVAL = 600;
const LR = 0.05;

const nF = FEATURES.length, nA = ACTIONS.length;
const W = new Float32Array(nF * nA);

const score = (f) => {
  const v = new Float32Array(nA);
  for (let a = 0; a < nA; a++) {
    let s = 0;
    for (let i = 0; i < nF; i++) if (f[i]) s += f[i] * W[i * nA + a];
    v[a] = s;
  }
  return v;
};

function run(n, seed0, learn, eps) {
  let right = 0, total = 0;
  const per = new Map();
  for (let e = 0; e < n; e++) {
    const mc = new Megacode(seed0 + e);
    for (let s = 0; s < 60 && !mc.state.over; s++) {
      const acc = mc.correctActions(); if (!acc.length) break;
      const f = encodeState(mc.state, mc.shockable);
      const v = score(f);
      let a = 0;
      for (let i = 1; i < nA; i++) if (v[i] > v[a]) a = i;
      if (learn && Math.random() < eps) a = Math.floor(Math.random() * nA);
      const res = mc.step(a);
      if (learn) {
        // Delta rule: push the taken action toward its verdict.
        const target = res.correct ? 1 : -1;
        const err = target - Math.tanh(v[a]);
        for (let i = 0; i < nF; i++) if (f[i]) W[i * nA + a] += LR * err * f[i];
      }
      const rec = per.get(ACTIONS[a]) ?? [0, 0];
      rec[0] += res.correct ? 1 : 0; rec[1]++; per.set(ACTIONS[a], rec);
      right += res.correct ? 1 : 0; total++;
    }
  }
  return { acc: right / total, per };
}

for (let round = 0; round < 6; round++) run(TRAIN / 6, 1000 + round * 9999, true, 0.35 - round * 0.05);
const out = run(EVAL, 900000, false, 0);

console.log(`linear readout on the same ${nF} features, ${TRAIN} megacodes of training\n`);
// The two comparison figures used to be printed here as literals, which is how a
// number survives the measurement that made it false: they still read 12.5% and ~37%
// after the generator fix moved both. Whatever this script prints, it computes.
console.log(`  chance and the mushroom body's own score: see the README, measured by`);
console.log(`  chance.mjs and working-memory.mjs rather than typed in here`);
console.log(`  linear readout              ${(out.acc * 100).toFixed(1)}%   <- ceiling for these features\n`);
console.log('per action:');
[...out.per.entries()].sort((a, b) => b[1][1] - a[1][1]).forEach(([a, [r, n]]) =>
  console.log(`  ${a.padEnd(17)} ${String(n).padStart(5)} taken  ${((r / n) * 100).toFixed(0).padStart(3)}% right`));
