/**
 * Does the fly still learn when nobody hands it a menu of actions?
 *
 *   node pipeline/no-menu.mjs [seeds] [train]
 *
 * Same mushroom body, same connectome, same central complex holding the history,
 * same one-bit dopamine. The only thing that changes is where the action comes from:
 *
 *   menu        the action is an INPUT. One Kenyon-cell code per candidate action,
 *               all ten scored, softmax over the scores. This is the architecture
 *               every number in the README so far belongs to.
 *   descending  the action is an OUTPUT. One code for the state, one MBON response,
 *               and the fixed MBON -> DN pathway decides which channel wins.
 *
 * The descending version is strictly less expressive -- a fixed final layer cannot
 * do what per-action coincidence coding does -- so a lower score is a result about
 * what the fly's own output wiring supports, not a regression.
 */
import { readFileSync } from 'node:fs';
import { Megacode, ACTIONS, makeFeatureSpace } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';
import { CentralComplex } from '../web/central-complex.js';
import { DescendingReadout } from '../web/descending.js';

const SEEDS = Number(process.argv[2] ?? 3);
const TRAIN = Number(process.argv[3] ?? 2500);
const EVAL = 300;

const url = (n) => new URL(`../data/out/${n}`, import.meta.url);
const circuit = JSON.parse(readFileSync(url('mb_circuit_both.json')));
const cxCircuit = JSON.parse(readFileSync(url('cx_circuit.json')));
const dnCircuit = JSON.parse(readFileSync(url('dn_circuit.json')));

const space = makeFeatureSpace({ omitMemory: true, cxGroups: 8, cxLevels: 5 });

const CONDITIONS = [
  ['menu',        'scoring ten labelled options',      null],
  ['fixed relay', 'descending, nothing downstream learns', { plastic: false }],
  ['plastic, da', 'descending, MBON->relay plastic',   { plastic: true, gate: 'da' }],
  ['plastic, DAN', 'same, gated on PAM/PPL1 only',     { plastic: true, gate: 'dan' }],
];

function run(mode, opts, seed) {
  const output = opts ? new DescendingReadout(dnCircuit, ACTIONS.length, opts) : null;
  // The descending arm learns at a tenth the rate. It reads absolute MBON drive
  // through a fixed pathway with large anatomical weights, so the same depression
  // that nudges a menu score swings a channel ranking outright; 0.006 is where the
  // rate sweep in the commit history puts it.
  const fly = new MushroomBody(circuit, space.FEATURES.length, ACTIONS.length,
                               { seed, groups: space.GROUP_SPANS, stateClaws: 6, output,
                                 lr: output ? 0.006 : 0.06 });
  const cx = new CentralComplex(cxCircuit, { readouts: 8, seed });

  const episode = (e, greedy) => {
    const mc = new Megacode(e);
    cx.reset();
    if (output) output.reset();
    let right = 0, total = 0;
    for (let s = 0; s < 60 && !mc.state.over; s++) {
      const acc = mc.correctActions();
      if (!acc.length) break;
      const f = space.encode(mc.state, mc.shockable, cx.out);
      const a = output ? fly.act(f, { greedy }) : fly.decide(f, null, { greedy });
      const res = mc.step(a);
      fly.learn(res.correct);
      cx.step(fly.last.learned);
      right += res.correct ? 1 : 0; total++;
    }
    return { right, total };
  };

  // Calibrate the descending channels on the naive animal, then freeze them, so
  // the gain is a fixed frame that learning moves within rather than a moving
  // target that absorbs it.
  if (output) {
    const w = fly.p.lr; fly.p.lr = 0;
    for (let e = 0; e < 200; e++) episode(700000 + seed * 1000 + e, false);
    fly.p.lr = w;
    output.freeze();
    fly.trials = 0;
  }
  for (let e = 0; e < TRAIN; e++) episode(seed * 100000 + e, false);
  let right = 0, total = 0;
  for (let e = 0; e < EVAL; e++) {
    const r = episode(900000 + seed * 10000 + e, true);
    right += r.right; total += r.total;
  }
  return right / Math.max(1, total);
}

console.log(`${SEEDS} seeds, ${TRAIN} training megacodes, ${EVAL} unseen evaluated greedy.`);
console.log('Chance is 12.5%. The bar for the plastic relay is 33.1% -- what the best');
console.log('permutation of the ten channel names already scores UNTRAINED, from');
console.log('pipeline/dn-naming.mjs. Below that, the plasticity is doing no work.\n');
console.log('condition        what changes                            accuracy');
for (const [mode, label, opts] of CONDITIONS) {
  const acc = [];
  for (let s = 0; s < SEEDS; s++) acc.push(run(mode, opts, 11 + s * 7));
  const mean = acc.reduce((a, b) => a + b, 0) / acc.length;
  console.log(`${mode.padEnd(16)} ${label.padEnd(40)} ${(mean * 100).toFixed(1).padStart(5)}%   `
            + `(${acc.map((a) => (a * 100).toFixed(0)).join('-')})`);
}
