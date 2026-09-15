/**
 * How much of the descending architecture's gap is just which channel got which name?
 *
 *   node pipeline/dn-naming.mjs
 *
 * pipeline/dn-ceiling.mjs grants a linear decoder a full 10x10 remix of the channel
 * drives and reaches 67.1%. The fly is not allowed that: it takes argmax over the ten
 * channels, with action k nailed to the k-th descending type by anatomical rank. This
 * splits the difference into the part that is naming and the part that is mixing.
 */
import { readFileSync } from 'node:fs';
import { Megacode, ACTIONS, makeFeatureSpace } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';
import { CentralComplex } from '../web/central-complex.js';
import { DescendingReadout } from '../web/descending.js';

const url = (n) => new URL(`../data/out/${n}`, import.meta.url);
const circuit = JSON.parse(readFileSync(url('mb_circuit_both.json')));
const cxC = JSON.parse(readFileSync(url('cx_circuit.json')));
const dnC = JSON.parse(readFileSync(url('dn_circuit.json')));
const space = makeFeatureSpace({ omitMemory: true, cxGroups: 8, cxLevels: 5 });
const out = new DescendingReadout(dnC, ACTIONS.length);
const fly = new MushroomBody(circuit, space.FEATURES.length, ACTIONS.length,
  { seed: 11, groups: space.GROUP_SPANS, stateClaws: 6, output: out });
const cx = new CentralComplex(cxC, { readouts: 8, seed: 11 });

const winners = [], accepted = [];
for (let e = 0; e < 800; e++) {
  const mc = new Megacode(e); cx.reset(); out.reset();
  for (let s = 0; s < 60 && !mc.state.over; s++) {
    const acc = mc.correctActions(); if (!acc.length) break;
    const code = fly.stateCode(space.encode(mc.state, mc.shockable, cx.out));
    const { mbon, learned } = fly.readout(code);
    const d = out.drive(mbon);
    let b = 0; for (let i = 1; i < d.length; i++) if (d[i] > d[b]) b = i;
    winners.push(b); accepted.push(new Set(acc));
    mc.step(acc[0]); cx.step(learned);
  }
}
// count[channel][action] = how often, when this channel won, that action was accepted
const n = ACTIONS.length;
const count = Array.from({ length: n }, () => new Float64Array(n));
for (let i = 0; i < winners.length; i++) for (const a of accepted[i]) count[winners[i]][a]++;

const score = (perm) => {
  let hit = 0;
  for (let i = 0; i < winners.length; i++) if (accepted[i].has(perm[winners[i]])) hit++;
  return hit / winners.length;
};
const identity = Array.from({ length: n }, (_, i) => i);
// Greedy best assignment on the count matrix, then a few improvement passes.
let perm = new Array(n).fill(-1);
const usedC = new Set(), usedA = new Set();
for (let step = 0; step < n; step++) {
  let bc = -1, ba = -1, bv = -1;
  for (let c = 0; c < n; c++) if (!usedC.has(c)) for (let a = 0; a < n; a++) if (!usedA.has(a))
    if (count[c][a] > bv) { bv = count[c][a]; bc = c; ba = a; }
  perm[bc] = ba; usedC.add(bc); usedA.add(ba);
}
for (let pass = 0; pass < 200; pass++) {
  let improved = false;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const p = perm.slice(); [p[i], p[j]] = [p[j], p[i]];
    if (score(p) > score(perm)) { perm = p; improved = true; }
  }
  if (!improved) break;
}
console.log(`${winners.length} decisions from an untrained fly, chance 11.7%\n`);
console.log(`argmax, channel k named action k (what the model does)  ${(score(identity)*100).toFixed(1)}%`);
console.log(`argmax, best permutation of the ten names              ${(score(perm)*100).toFixed(1)}%`);
console.log(`full 10x10 remix (the ceiling measured earlier)          67.1%`);
console.log(`\ndistinct channels ever winning: ${new Set(winners).size}/10`);
const hist = new Array(n).fill(0); for (const w of winners) hist[w]++;
console.log(`win share per channel: ${hist.map(h => (h/winners.length*100).toFixed(0)+'%').join(' ')}`);
