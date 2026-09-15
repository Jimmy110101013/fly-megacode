/**
 * Where does the accuracy ceiling come from?
 *
 *   node pipeline/ceiling.mjs [trainScenarios]
 *
 * Trains, then evaluates the same fly three ways to separate the causes:
 *   exploring   -- what the live page shows: softmax with its temperature floor
 *   greedy      -- exploration switched off, learning frozen: what it actually knows
 *   per action  -- where the residual error lives
 */
import { readFileSync } from 'node:fs';
import { Megacode, ACTIONS, FEATURES, GROUP_SPANS, encodeState } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';

const TRAIN = Number(process.argv[2] ?? 2000);
const EVAL = 600;
const MAX_STEPS = 60;

const circuit = JSON.parse(readFileSync(new URL('../data/out/mb_circuit.json', import.meta.url)));

function run(fly, n, seed0, { learn, greedy }) {
  let right = 0, total = 0;
  const per = new Map();       // action -> [right, seen]
  const wantedInstead = new Map();
  for (let ep = 0; ep < n; ep++) {
    const mc = new Megacode(seed0 + ep);
    for (let s = 0; s < MAX_STEPS && !mc.state.over; s++) {
      const acc = mc.correctActions();
      if (!acc.length) break;
      const a = fly.decide(encodeState(mc.state, mc.shockable), null, { greedy });
      const res = mc.step(a);
      if (learn) fly.learn(res.correct);
      const rec = per.get(res.action) ?? [0, 0];
      rec[0] += res.correct ? 1 : 0; rec[1]++;
      per.set(res.action, rec);
      if (!res.correct) {
        for (const w of res.accepted) {
          const key = `${res.action} → ${ACTIONS[w]}`;
          wantedInstead.set(key, (wantedInstead.get(key) ?? 0) + 1);
        }
      }
      right += res.correct ? 1 : 0; total++;
    }
  }
  return { acc: right / total, total, per, wantedInstead };
}

const fly = new MushroomBody(circuit, FEATURES.length, ACTIONS.length, { seed: 11, groups: GROUP_SPANS });
run(fly, TRAIN, 1000, { learn: true, greedy: false });

const exploring = run(fly, EVAL, 900000, { learn: false, greedy: false });
const greedy = run(fly, EVAL, 900000, { learn: false, greedy: true });

console.log(`trained on ${TRAIN} megacodes, evaluated on ${EVAL} unseen ones\n`);
console.log(`  chance (engine accepts ~1.2 of 10)      11.8%`);
console.log(`  exploring  (softmax, temp floor ${fly.p.tempFloor})   ${(exploring.acc * 100).toFixed(1)}%`);
console.log(`  greedy     (exploration off)            ${(greedy.acc * 100).toFixed(1)}%`);
console.log(`  cost of keeping exploration on          ${((greedy.acc - exploring.acc) * 100).toFixed(1)} points\n`);

console.log('greedy accuracy per action taken:');
[...greedy.per.entries()]
  .sort((a, b) => b[1][1] - a[1][1])
  .forEach(([a, [r, n]]) => {
    const pct = (r / n) * 100;
    console.log(`  ${a.padEnd(17)} ${String(n).padStart(5)} taken  ` +
                `${pct.toFixed(0).padStart(3)}% right  ${'#'.repeat(Math.round(pct / 4))}`);
  });

console.log('\nmost common greedy mistakes (took → algorithm wanted):');
[...greedy.wantedInstead.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10)
  .forEach(([k, v]) => console.log(`  ${String(v).padStart(4)}×  ${k}`));
