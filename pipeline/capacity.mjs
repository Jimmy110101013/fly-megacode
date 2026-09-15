/**
 * Is the fly short of brain, or short of a usable code?
 *
 *   node pipeline/capacity.mjs
 *
 * A: subsample the Kenyon cell population from 130 up to all 2,597, holding the
 *    5% sparsity rule. If capacity binds, accuracy climbs with cell count.
 * B: measure how much the KC code actually changes when the one feature that
 *    decides an action flips (access done / not done, airway in / not in).
 *    If the code barely moves, no number of neurons can fix it.
 */
import { readFileSync } from 'node:fs';
import { Megacode, ACTIONS, FEATURES, GROUP_SPANS, encodeState } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';

const circuit = JSON.parse(readFileSync(new URL('../data/out/mb_circuit.json', import.meta.url)));

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

/** A smaller mushroom body: keep n Kenyon cells and the synapses they own. */
function withKC(c, n, seed = 5) {
  if (n >= c.kc.length) return c;
  const r = rng(seed);
  const order = c.kc.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]];
  }
  const keep = order.slice(0, n);
  const map = new Map(keep.map((k, i) => [k, i]));
  return {
    ...c,
    kc: keep.map((i) => c.kc[i]),
    kc_mbon: c.kc_mbon.filter(([k]) => map.has(k)).map(([k, m, w]) => [map.get(k), m, w]),
  };
}

function trainEval(c, train = 4000, evalN = 400, opts = {}) {
  const fly = new MushroomBody(c, FEATURES.length, ACTIONS.length, { seed: 11, groups: GROUP_SPANS, ...opts });
  const pass = (n, s0, learn, greedy) => {
    let right = 0, total = 0;
    for (let e = 0; e < n; e++) {
      const mc = new Megacode(s0 + e);
      for (let s = 0; s < 60 && !mc.state.over; s++) {
        const acc = mc.correctActions(); if (!acc.length) break;
        const a = fly.decide(encodeState(mc.state, mc.shockable), null, { greedy });
        const res = mc.step(a);
        if (learn) fly.learn(res.correct);
        right += res.correct ? 1 : 0; total++;
      }
    }
    return right / total;
  };
  pass(train, 1000, true, false);
  return { acc: pass(evalN, 900000, false, true), fly };
}

console.log('A. does more brain help?  (5% sparsity held constant)\n');
console.log('   Kenyon cells   active/pattern   KC->MBON synapses   greedy accuracy');
for (const n of [130, 325, 650, 1300, 2597]) {
  const c = withKC(circuit, n);
  const { acc, fly } = trainEval(c);
  console.log(`   ${String(n).padStart(10)}   ${String(fly.kActive).padStart(13)}   ` +
              `${String(fly.w.length).padStart(17)}   ${(acc * 100).toFixed(1).padStart(6)}%`);
}

console.log('\nB. does the code even notice the feature that decides the answer?\n');
const fly = new MushroomBody(circuit, FEATURES.length, ACTIONS.length, { seed: 11, groups: GROUP_SPANS });
const jac = (a, b) => { const s = new Set(b); return a.filter((x) => s.has(x)).length / a.length; };

const base = new Megacode(42);
Object.assign(base.state, { rhythm: 'VF', phase: 'cpr', shocks: 2, access: false,
                            airway: false, causeTreated: false, t: 200, tLastEpi: -Infinity });
const flips = [
  ['access done', (s) => { s.access = !s.access; }],
  ['airway in place', (s) => { s.airway = !s.airway; }],
  ['cause treated', (s) => { s.causeTreated = !s.causeTreated; }],
  ['rhythm VF -> asystole', (s) => { s.rhythm = s.rhythm === 'VF' ? 'asystole' : 'VF'; }],
];
for (const [name, flip] of flips) {
  const s1 = { ...base.state };
  const f1 = encodeState(s1, ['VF', 'pVT'].includes(s1.rhythm));
  const s2 = { ...base.state }; flip(s2);
  const f2 = encodeState(s2, ['VF', 'pVT'].includes(s2.rhythm));
  let ov = 0;
  for (let a = 0; a < ACTIONS.length; a++) {
    ov += jac(fly.encode(f1, a), fly.encode(f2, a));
  }
  ov /= ACTIONS.length;
  console.log(`   ${name.padEnd(24)} KC code overlap ${(ov * 100).toFixed(1).padStart(5)}%` +
              `   ${ov > 0.5 ? '<-- the fly can barely tell these apart' : ''}`);
}

console.log('\n   for reference, two different candidate ACTIONS in the same state:');
{
  const f = encodeState(base.state, true);
  let ov = 0, n = 0;
  for (let a = 0; a < ACTIONS.length; a++)
    for (let b = a + 1; b < ACTIONS.length; b++) { ov += jac(fly.encode(f, a), fly.encode(f, b)); n++; }
  console.log(`   ${'action vs action'.padEnd(24)} KC code overlap ${(ov / n * 100).toFixed(1).padStart(5)}%`);
}
