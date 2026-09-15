/**
 * Do the measured KC->MBON synapse counts matter as initial weights?
 *
 *   node pipeline/weight-controls.mjs [seeds=5] [train=2500]
 *
 * Replaces the README's earlier weight-control table, which came from
 * `train.mjs <mode>`: one seed, training accuracy (exploration included), and an old
 * configuration (one hemisphere, the engine holding the history). None of the three
 * is how a number in the README is supposed to be measured.
 *
 * Here: the page's configuration (menu architecture, both hemispheres, the fly holding
 * its own history in the central complex), every mode on the same seeds and the same
 * scenarios, evaluated greedy on unseen megacodes. The graph is untouched in every
 * mode -- only the numbers on the 89,315 KC->MBON edges change. Rewiring the graph is
 * pipeline/rewire-menu.mjs.
 *
 *   real       measured synapse counts
 *   shuffled   the same multiset, reassigned across all edges
 *   resampled  drawn independently from the same marginal distribution
 *   random     uniform on [0, 2 x mean] -- a different distribution entirely
 *
 * MushroomBody gives the weight modes their own generator, so every mode consumes the
 * same random draws elsewhere. Without that, an earlier version showed a ten-point
 * advantage for the real counts that was nothing but different draws.
 *
 * Rule, fixed before the first run: a mode is worse than the real counts iff
 * real - mode > 5 points AND real wins on >= 4 of 5 seeds; better by the same margin
 * the other way; otherwise a draw. Chance is 12.5% (pipeline/chance.mjs).
 */
import { readFileSync } from 'node:fs';
import { Megacode, ACTIONS, makeFeatureSpace } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';
import { CentralComplex } from '../web/central-complex.js';

const SEEDS = Number(process.argv[2] ?? 5);
const TRAIN = Number(process.argv[3] ?? 2500);
const EVAL = 300;
const MAX_STEPS = 60;
const MODES = ['real', 'shuffled', 'resampled', 'random'];

const load = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
const circuit = load('../web/mb_circuit.json');
const cxCircuit = load('../web/cx_circuit.json');
const space = makeFeatureSpace({ omitMemory: true, cxGroups: 8, cxLevels: 5 });

function evaluate(fly, cx, n, s0, learn, greedy) {
  let right = 0, total = 0;
  for (let e = 0; e < n; e++) {
    const mc = new Megacode(s0 + e);
    cx.reset();
    for (let s = 0; s < MAX_STEPS && !mc.state.over; s++) {
      if (!mc.correctActions().length) break;
      const a = fly.decide(space.encode(mc.state, mc.shockable, cx.out), null, { greedy });
      const res = mc.step(a);
      if (learn) fly.learn(res.correct);
      cx.step(fly.last.learned);
      right += res.correct ? 1 : 0; total++;
    }
  }
  return right / total;
}

function run(weights, seed) {
  const make = () => [
    new MushroomBody(circuit, space.FEATURES.length, ACTIONS.length,
                     { seed, weights, groups: space.GROUP_SPANS, stateClaws: 6 }),
    new CentralComplex(cxCircuit, { readouts: 8, seed }),
  ];
  // The untrained score comes from a separate, identical animal. Measuring it on the
  // one that then trains would consume its random draws and adapt its reservoir gain
  // first, and the trained result would no longer match rewire-menu.mjs's measured arm.
  const before = evaluate(...make(), EVAL, 900000, false, true);
  const [fly, cx] = make();
  evaluate(fly, cx, TRAIN, 1000, true, false);
  const after = evaluate(fly, cx, EVAL, 900000, false, true);
  if (cx.broken()) throw new Error('central complex went non-finite');
  return { before: before * 100, after: after * 100 };
}

const seeds = Array.from({ length: SEEDS }, (_, s) => 11 + s * 101);
console.log(`menu architecture, page configuration. ${SEEDS} seeds, ${TRAIN} megacodes training, ` +
            `${EVAL} unseen evaluated greedy. Chance 12.5%.\n`);
console.log('initial weights   untrained        after training: per seed             mean (range)');

const results = {};
for (const mode of MODES) {
  const runs = seeds.map((seed) => run(mode, seed));
  results[mode] = runs;
  const after = runs.map((r) => r.after), before = runs.map((r) => r.before);
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  console.log(`${mode.padEnd(17)} ${mean(before).toFixed(1).padStart(5)}%          ` +
              `${after.map((x) => x.toFixed(1).padStart(5)).join(' ')}   ` +
              `${mean(after).toFixed(1)}% (${Math.min(...after).toFixed(0)}-${Math.max(...after).toFixed(0)})`);
}

console.log('');
const real = results.real.map((r) => r.after);
for (const mode of MODES.slice(1)) {
  const m = results[mode].map((r) => r.after);
  const d = real.reduce((s, x, i) => s + (x - m[i]), 0) / SEEDS;
  const realWins = real.filter((x, i) => x > m[i]).length;
  const modeWins = real.filter((x, i) => x < m[i]).length;
  const need = Math.ceil(SEEDS * 0.8);
  const verdict = d > 5 && realWins >= need ? 'worse than real'
    : -d > 5 && modeWins >= need ? 'better than real' : 'draw';
  console.log(`real - ${mode.padEnd(9)} ${d >= 0 ? '+' : ''}${d.toFixed(1)} points, real wins ${realWins}/${SEEDS} -> ${verdict}`);
}
