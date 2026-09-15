/**
 * Can the fly hold its own state?
 *
 *   node pipeline/working-memory.mjs [seeds] [train]
 *
 * Three conditions, identical mushroom body and identical learning rule. The only
 * difference is who remembers what has already been done:
 *
 *   engine remembers  every feature, including shot count and drug timing. This is
 *                     the model as it has been all along, and it is a cheat: the
 *                     scenario engine is acting as the animal's working memory.
 *   nobody remembers  only what a bystander could read off the patient. Rhythm,
 *                     ETCO2, whether a line and a tube are in. No history at all.
 *   fly remembers     the same observable features, plus twelve readouts from a
 *                     central-complex reservoir the fly drives with its own MBON
 *                     output. Nothing in that reservoir is trained.
 */
import { readFileSync } from 'node:fs';
import { Megacode, ACTIONS, makeFeatureSpace } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';
import { CentralComplex } from '../web/central-complex.js';

const SEEDS = Number(process.argv[2] ?? 3);
const TRAIN = Number(process.argv[3] ?? 2500);
const EVAL = 300;
const MAX_STEPS = 60;

const url = (n) => new URL(`../data/out/${n}`, import.meta.url);
const circuit = JSON.parse(readFileSync(url('mb_circuit_both.json')));
const cxCircuit = JSON.parse(readFileSync(url('cx_circuit.json')));

function run(space, useCX, seed, { train, evalN }) {
  const fly = new MushroomBody(circuit, space.FEATURES.length, ACTIONS.length,
                               { seed, groups: space.GROUP_SPANS,
                                 stateClaws: useCX ? 6 : 5 });
  const cx = useCX ? new CentralComplex(cxCircuit, { readouts: 8, seed }) : null;

  const pass = (n, s0, learn, greedy) => {
    let right = 0, total = 0;
    for (let e = 0; e < n; e++) {
      const mc = new Megacode(s0 + e);
      cx?.reset();                       // a new patient starts with a clear head
      for (let s = 0; s < MAX_STEPS && !mc.state.over; s++) {
        const acc = mc.correctActions();
        if (!acc.length) break;
        const f = space.encode(mc.state, mc.shockable, cx ? cx.out : null);
        const a = fly.decide(f, null, { greedy });
        const res = mc.step(a);
        if (learn) fly.learn(res.correct);
        cx?.step(fly.last.learned);          // the mushroom body's output drives the CX
        right += res.correct ? 1 : 0; total++;
      }
    }
    return right / total;
  };
  pass(train, 1000, true, false);
  return { acc: pass(evalN, 900000, false, true), cx };
}

const conditions = [
  ['engine remembers', makeFeatureSpace(), false],
  ['nobody remembers', makeFeatureSpace({ omitMemory: true }), false],
  // Eight readouts, not sixteen: a Kenyon cell samples five glomeruli, so piling
  // on central-complex groups crowds out the seven observable ones and the code
  // stops carrying what the patient looks like.
  ['fly remembers', makeFeatureSpace({ omitMemory: true, cxGroups: 8, cxLevels: 5 }), true],
];

{
  const probe = new CentralComplex(cxCircuit, { readouts: 12, seed: 1 });
  const t0 = performance.now();
  for (let i = 0; i < 200; i++) probe.step(null);
  console.log(`central complex: ${probe.n} neurons, ${probe.nEdges} recurrent edges ` +
              `kept, ${probe.nDrive} MBON inputs, spectral radius normalised to ` +
              `${probe.p.rho}\n  ${((performance.now() - t0) / 200).toFixed(2)} ms per ` +
              `integration step\n`);
}

console.log(`${SEEDS} seeds, ${TRAIN} megacodes of training, ${EVAL} unseen evaluated ` +
            `greedy. Chance is 11.8%.\n`);
console.log('condition            channels   mean    range');
for (const [name, space, useCX] of conditions) {
  const runs = [];
  for (let s = 0; s < SEEDS; s++) {
    runs.push(run(space, useCX, 11 + s * 101, { train: TRAIN, evalN: EVAL }).acc);
  }
  const mean = runs.reduce((a, b) => a + b, 0) / runs.length;
  console.log(`${name.padEnd(20)}${String(space.FEATURES.length).padStart(8)}` +
              `${(mean * 100).toFixed(1).padStart(8)}%  ` +
              `${(Math.min(...runs) * 100).toFixed(0)}-${(Math.max(...runs) * 100).toFixed(0)}%`);
}
