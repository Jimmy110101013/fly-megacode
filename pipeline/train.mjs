/**
 * Headless check: does the fly actually learn ACLS?
 *
 *   node pipeline/train.mjs [scenarios] [weights: real|shuffled|random]
 *
 * Prints a learning curve plus the two diagnostics that catch a model which runs
 * but does not learn: KC code sparsity, and total synaptic mass (runaway
 * depression shows up as mass collapsing toward zero).
 */
import { readFileSync } from 'node:fs';
import { Megacode, ACTIONS, FEATURES, GROUP_SPANS, encodeState } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';

const N = Number(process.argv[2] ?? 600);
const MODE = process.argv[3] ?? 'real';
const MAX_STEPS = 60;

const circuit = JSON.parse(readFileSync(new URL('../data/out/mb_circuit.json', import.meta.url)));
const fly = new MushroomBody(circuit, FEATURES.length, ACTIONS.length, { seed: 11, weights: MODE, groups: GROUP_SPANS });

console.log(`fly: ${fly.nKC} KC, ${fly.nMBON} MBON, ${fly.nDAN} DAN, ` +
            `${fly.w.length} KC->MBON synapses, ${fly.kActive} active per pattern (${MODE})`);

const hist = [];
let overlapSum = 0, overlapN = 0, prevCode = null;

for (let ep = 0; ep < N; ep++) {
  const mc = new Megacode(1000 + ep);
  let right = 0, total = 0;
  for (let step = 0; step < MAX_STEPS && !mc.state.over; step++) {
    const feats = encodeState(mc.state, mc.shockable);
    const accepted = mc.correctActions();
    if (!accepted.length) break;
    const a = fly.decide(feats);
    const code = fly.last.activeKC;
    if (prevCode) {
      const set = new Set(prevCode);
      overlapSum += code.filter((k) => set.has(k)).length / code.length;
      overlapN++;
    }
    prevCode = code;
    const res = mc.step(a);
    fly.learn(res.correct);
    right += res.correct ? 1 : 0;
    total++;
  }
  hist.push(total ? right / total : 0);
}

const win = (arr, a, b) => {
  const s = arr.slice(a, b);
  return s.reduce((x, y) => x + y, 0) / (s.length || 1);
};
const B = Math.max(1, Math.floor(N / 12));
console.log('\nlearning curve (mean accuracy per block of %d scenarios)', B);
for (let i = 0; i < N; i += B) {
  const acc = win(hist, i, i + B);
  const bar = '#'.repeat(Math.round(acc * 50));
  console.log(`  ${String(i).padStart(5)}-${String(Math.min(N, i + B)).padStart(5)}  ` +
              `${(acc * 100).toFixed(1).padStart(5)}%  ${bar}`);
}
console.log(`\nfirst 10%%: ${(win(hist, 0, Math.floor(N * 0.1)) * 100).toFixed(1)}%`);
console.log(`last  10%%: ${(win(hist, Math.floor(N * 0.9), N) * 100).toFixed(1)}%`);
console.log(`\ndiagnostics`);
console.log(`  KC pattern overlap between consecutive decisions: ` +
            `${(overlapSum / Math.max(1, overlapN) * 100).toFixed(1)}%  (want well under 100%)`);
console.log(`  mean synaptic weight: ${fly.weightMass().toFixed(4)}  (anatomical baseline 1.0000)`);
console.log(`  softmax temperature now: ${fly.last.temp.toFixed(3)}`);
