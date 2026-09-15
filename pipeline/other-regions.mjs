/**
 * Would another part of the fly brain have learned this?
 *
 *   node pipeline/other-regions.mjs [seeds] [train]
 *
 * Identical model, identical learning rule, identical task. The only thing that
 * changes is which circuit the weights come from: the mushroom body, or a
 * same-shaped stand-in built from another region by pipeline/extract_region.py.
 */
import { readdirSync, readFileSync } from 'node:fs';
import { Megacode, ACTIONS, FEATURES, GROUP_SPANS, encodeState } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';

const SEEDS = Number(process.argv[2] ?? 3);
const TRAIN = Number(process.argv[3] ?? 3000);
const EVAL = 300;

const dir = new URL('../data/out/', import.meta.url);
const load = (f) => JSON.parse(readFileSync(new URL(f, dir)));

function trial(circuit, seed) {
  const fly = new MushroomBody(circuit, FEATURES.length, ACTIONS.length,
                               { seed, groups: GROUP_SPANS });
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
  pass(TRAIN, 1000, true, false);
  return pass(EVAL, 900000, false, true);
}

const entries = [['mushroom body', 'mb_circuit.json']];
for (const f of readdirSync(dir).sort()) {
  if (f.startsWith('circuit_')) entries.push([f.slice(8, -5), f]);
}

console.log(`${SEEDS} seeds, ${TRAIN} megacodes of training, ` +
            `${EVAL} unseen megacodes evaluated greedy. Chance is 12.5% (pipeline/chance.mjs).\n`);
console.log('region              cells  outputs  synapses  reward/punish   mean   range');
for (const [name, file] of entries) {
  let c;
  try { c = load(file); } catch { continue; }
  const runs = [];
  let info = null;
  for (let s = 0; s < SEEDS; s++) {
    const fly = new MushroomBody(c, FEATURES.length, ACTIONS.length,
                                 { seed: 11 + s * 101, groups: GROUP_SPANS });
    info ??= { syn: fly.w.length,
               pam: fly.isPAM.filter(Boolean).length,
               ppl: fly.isPAM.filter((x) => !x).length };
    runs.push(trial(c, 11 + s * 101));
  }
  const mean = runs.reduce((a, b) => a + b, 0) / runs.length;
  console.log(
    `${name.padEnd(18)}${String(c.kc.length).padStart(6)}` +
    `${String(c.mbon.length).padStart(9)}${String(info.syn).padStart(10)}` +
    `${`${info.pam}/${info.ppl}`.padStart(15)}` +
    `${(mean * 100).toFixed(1).padStart(8)}%  ` +
    `${(Math.min(...runs) * 100).toFixed(0)}-${(Math.max(...runs) * 100).toFixed(0)}%`);
}
