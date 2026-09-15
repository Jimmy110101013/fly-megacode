/**
 * Does the measured mushroom-body wiring matter, or would any graph with the same
 * degrees do?
 *
 *   node pipeline/rewire-menu.mjs [seeds=5] [train=2500]
 *
 * The page's configuration (menu architecture, the fly holding its own history in the
 * central complex). pipeline/rewire-control.mjs asks the same question of the closed
 * descending-neuron architecture; this asks it of the one the page runs.
 *
 * Rewire: double-edge swap of targets on the edge list as extracted. kc_mbon is a
 * multigraph (89,315 rows, 62,261 unique pairs), so duplicates are NOT rejected --
 * rejecting them, as rewire-control.mjs does for a simple graph, refuses most swaps
 * and leaves the graph barely rewired. A row keeps its synapse count as it moves, and
 * every node's in- and out-degree is asserted unchanged.
 *
 *   measured   the page as it is
 *   KC->MBON   which Kenyon cell reaches which compartment is lost. Scored.
 *   DAN->MBON  which dopamine population owns which compartment is lost. NOT scored:
 *              the rewired fly spends most of its choices on one action and runs every
 *              megacode to the step cap, so its accuracy is graded on a different mix
 *              of situations and is not comparable. The mechanism is reported instead.
 *
 * Rule, fixed before the first run: the wiring carries structure iff measured - rewired
 * > 5 points AND measured wins on >= 4 of 5 seeds. A single arm spans ~13 points across
 * seeds. Chance is 12.5% (pipeline/chance.mjs).
 *
 * Caveat: the input -> Kenyon cell layer is modelled as random, so a KC's identity carries
 * no task meaning to destroy. What KC->MBON rewiring removes is which KCs converge on the
 * same compartment -- lobe and KC-class structure -- and multi-contact pairs.
 */
import { readFileSync } from 'node:fs';
import { Megacode, ACTIONS, makeFeatureSpace } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';
import { CentralComplex } from '../web/central-complex.js';

const SEEDS = Number(process.argv[2] ?? 5);
const TRAIN = Number(process.argv[3] ?? 2500);
const EVAL = 300;
const MAX_STEPS = 60;

const load = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
const circuit = load('../web/mb_circuit.json');
const cxCircuit = load('../web/cx_circuit.json');
const space = makeFeatureSpace({ omitMemory: true, cxGroups: 8, cxLevels: 5 });

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

/** Swap the targets of two rows with different sources and different targets. */
function rewire(edges, seed, rounds = 20) {
  const e = edges.map((x) => x.slice());
  const r = rng(seed);
  for (let t = 0; t < rounds * e.length; t++) {
    const i = (r() * e.length) | 0, j = (r() * e.length) | 0;
    if (i === j || e[i][0] === e[j][0] || e[i][1] === e[j][1]) continue;
    const b = e[i][1];
    e[i][1] = e[j][1]; e[j][1] = b;
  }
  return e;
}

function degreesMatch(a, b) {
  const count = (edges, k) => {
    const m = new Map();
    for (const x of edges) m.set(x[k], (m.get(x[k]) ?? 0) + 1);
    return m;
  };
  for (const k of [0, 1]) {
    const ma = count(a, k), mb = count(b, k);
    if (ma.size !== mb.size) return false;
    for (const [node, n] of ma) if (mb.get(node) !== n) return false;
  }
  return true;
}

const pairs = (edges) => new Set(edges.map((x) => `${x[0]}>${x[1]}`));
function overlap(a, b) {
  const pa = pairs(a), pb = pairs(b);
  let n = 0;
  for (const p of pa) if (pb.has(p)) n++;
  return n / pa.size;
}

/** How cleanly each MBON's dopamine comes from one population, and its polarity. */
function compartments(fly) {
  let mixed = 0;
  for (let j = 0; j < fly.nMBON; j++) {
    const p = fly.pamDrive[j], q = fly.ppl1Drive[j];
    if (p + q > 0 && Math.max(p, q) / (p + q) < 0.8) mixed++;
  }
  return { mixed, sign: Array.from(fly.mbonSign) };
}

function run(circ, seed) {
  const fly = new MushroomBody(circ, space.FEATURES.length, ACTIONS.length,
                               { seed, groups: space.GROUP_SPANS, stateClaws: 6 });
  const cx = new CentralComplex(cxCircuit, { readouts: 8, seed });
  const comp = compartments(fly);
  const pass = (n, s0, learn, greedy) => {
    const chose = new Array(ACTIONS.length).fill(0);
    let right = 0, total = 0, capped = 0;
    for (let e = 0; e < n; e++) {
      const mc = new Megacode(s0 + e);
      cx.reset();
      let s = 0;
      for (; s < MAX_STEPS && !mc.state.over; s++) {
        if (!mc.correctActions().length) break;
        const a = fly.decide(space.encode(mc.state, mc.shockable, cx.out), null, { greedy });
        const res = mc.step(a);
        if (learn) fly.learn(res.correct);
        cx.step(fly.last.learned);
        chose[a]++;
        right += res.correct ? 1 : 0; total++;
      }
      if (s === MAX_STEPS) capped++;
    }
    return { acc: right / total, lockOn: Math.max(...chose) / total, capped: capped / n };
  };
  pass(TRAIN, 1000, true, false);
  const ev = pass(EVAL, 900000, false, true);
  if (cx.broken()) throw new Error('central complex went non-finite');
  return { ...ev, ...comp };
}

const seeds = Array.from({ length: SEEDS }, (_, s) => 11 + s * 101);
// Each arm hands MushroomBody a fresh circuit object; nothing mutates `circuit`, so the
// projection layer draws the same random sequence in every arm.
const arms = [
  ['measured', () => circuit],
  ['KC->MBON rewired', (seed) => ({ ...circuit, kc_mbon: rewire(circuit.kc_mbon, 0xC0DE ^ seed) })],
  ['DAN->MBON rewired', (seed) => ({ ...circuit, dan_mbon: rewire(circuit.dan_mbon, 0xDA7 ^ seed) })],
];

console.log('structure:');
for (const [key, salt] of [['kc_mbon', 0xC0DE], ['dan_mbon', 0xDA7]]) {
  const orig = circuit[key], r1 = rewire(orig, salt ^ 11), r2 = rewire(orig, salt ^ 112);
  if (!degreesMatch(orig, r1) || !degreesMatch(orig, r2)) throw new Error(`${key}: rewire changed a degree`);
  const om = overlap(orig, r1), rr = overlap(r1, r2);
  if (Math.abs(om - rr) > 0.05) throw new Error(`${key}: not randomised (${om.toFixed(3)} vs ${rr.toFixed(3)})`);
  console.log(`  ${key.padEnd(9)} degrees preserved; pair overlap with measured ${om.toFixed(3)}, ` +
              `between two rewires ${rr.toFixed(3)}`);
}
console.log(`\n${SEEDS} seeds, ${TRAIN} megacodes training, ${EVAL} unseen evaluated greedy. Chance 12.5%.\n`);

const results = {};
for (const [name, make] of arms) {
  results[name] = seeds.map((seed) => run(make(seed), seed));
  const accs = results[name].map((r) => r.acc * 100);
  const mean = accs.reduce((x, y) => x + y, 0) / accs.length;
  const span = `${Math.min(...accs).toFixed(0)}-${Math.max(...accs).toFixed(0)}`;
  if (name === 'DAN->MBON rewired') {
    const ref = results.measured[0].sign;
    const r = results[name];
    const avg = (k) => r.reduce((s, x) => s + x[k], 0) / r.length;
    console.log(`${name.padEnd(20)} not scored (see header)`);
    console.log(`  most-chosen action ${(avg('lockOn') * 100).toFixed(0)}% of choices, megacodes at step cap ` +
                `${(avg('capped') * 100).toFixed(0)}%  (measured: ${(results.measured.reduce((s, x) => s + x.lockOn, 0) / SEEDS * 100).toFixed(0)}%, ` +
                `${(results.measured.reduce((s, x) => s + x.capped, 0) / SEEDS * 100).toFixed(0)}%)`);
    console.log(`  MBONs with mixed PAM/PPL1 input ${r.map((x) => x.mixed).join(' ')} of ${ref.length} ` +
                `(measured ${results.measured[0].mixed}); polarity flipped ` +
                `${r.map((x) => x.sign.filter((v, j) => v !== ref[j]).length).join(' ')}`);
  } else {
    console.log(`${name.padEnd(20)} ${accs.map((a) => a.toFixed(1).padStart(5)).join(' ')}   ` +
                `mean ${mean.toFixed(1)}% (${span})`);
  }
}

const M = results.measured.map((r) => r.acc * 100);
const R = results['KC->MBON rewired'].map((r) => r.acc * 100);
const d = M.reduce((s, x, i) => s + (x - R[i]), 0) / SEEDS;
const wins = M.filter((x, i) => x > R[i]).length;
const verdict = d > 5 && wins >= Math.ceil(SEEDS * 0.8) ? 'the wiring carries structure' : 'draw';
console.log(`\nKC->MBON: measured - rewired ${d >= 0 ? '+' : ''}${d.toFixed(1)} points, ` +
            `measured wins ${wins}/${SEEDS} -> ${verdict}`);
