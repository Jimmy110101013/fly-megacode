/**
 * The learning rule must move value in the right direction. If this fails, no
 * amount of training will help: reward and punishment are not differentiated.
 *
 *   node pipeline/test-plasticity.mjs
 */
import { existsSync, readFileSync } from 'node:fs';
import { Megacode, ACTIONS, FEATURES, GROUP_SPANS, encodeState } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';

// The freshly extracted single-hemisphere circuit if the pipeline has been run
// here, otherwise the bilateral circuit the page ships -- so this runs in CI
// against the same wiring a visitor's browser loads.
const extracted = new URL('../data/out/mb_circuit.json', import.meta.url);
const shipped = new URL('../web/mb_circuit.json', import.meta.url);
const circuit = JSON.parse(readFileSync(existsSync(extracted) ? extracted : shipped));

/** Value of one (state, action) pair before and after a single dopamine event. */
function probe(correct, seed = 42, action = 3) {
  const fly = new MushroomBody(circuit, FEATURES.length, ACTIONS.length, { seed: 11, groups: GROUP_SPANS });
  const mc = new Megacode(seed);
  const feats = encodeState(mc.state, mc.shockable);
  fly.primeState(feats);
  const code = fly.encode(feats, action, true);
  const before = fly.readout(code).value;
  fly.last = { action, activeKC: code, mbon: null, values: [], temp: 1 };
  fly.learn(correct);
  fly.primeState(feats);
  return fly.readout(fly.encode(feats, action, true)).value - before;
}

const checks = [];
const add = (name, pass, detail) => checks.push({ name, pass, detail });

// Averaged over several scenarios so a single odd state cannot decide the result.
let rSum = 0, pSum = 0, n = 0;
for (const seed of [42, 77, 101, 250, 313, 404, 512, 640]) {
  for (const a of [0, 3, 5, 8]) { rSum += probe(true, seed, a); pSum += probe(false, seed, a); n++; }
}
const rAvg = rSum / n, pAvg = pSum / n;

add('reward raises the value of the rewarded action', rAvg > 0, `delta = ${rAvg.toFixed(5)}`);
add('punishment lowers the value of the punished action', pAvg < 0, `delta = ${pAvg.toFixed(5)}`);
add('reward and punishment are differentiated', rAvg - pAvg > 1e-3, `gap = ${(rAvg - pAvg).toFixed(5)}`);

// A code that cannot tell two actions apart cannot support action selection.
{
  const fly = new MushroomBody(circuit, FEATURES.length, ACTIONS.length, { seed: 11, groups: GROUP_SPANS });
  const mc = new Megacode(42);
  const f = encodeState(mc.state, mc.shockable);
  fly.primeState(f);
  const codes = [...Array(ACTIONS.length).keys()].map((a) => fly.encode(f, a, true));
  let ov = 0, m = 0;
  for (let i = 0; i < codes.length; i++)
    for (let j = i + 1; j < codes.length; j++) {
      const s = new Set(codes[j]);
      ov += codes[i].filter((k) => s.has(k)).length / codes[i].length; m++;
    }
  const overlap = ov / m;
  add('KC codes for different actions are distinguishable', overlap < 0.45,
      `overlap = ${(overlap * 100).toFixed(1)}% (want < 45%)`);
}

let failed = 0;
for (const c of checks) {
  console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.name}  [${c.detail}]`);
  if (!c.pass) failed++;
}
console.log(failed ? `\n${failed} check(s) failed` : '\nall checks passed');
process.exit(failed ? 1 : 0);
