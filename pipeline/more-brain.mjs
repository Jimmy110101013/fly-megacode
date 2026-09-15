/**
 * Does the fly need a bigger brain to learn this?
 *
 *   node pipeline/more-brain.mjs [seeds] [train]
 *
 * Compares half a mushroom body, one whole mushroom body, and both hemispheres
 * together -- the last one being real FlyWire wiring, not a synthetic scale-up.
 * Several seeds per configuration, because single-seed runs on this task swing by
 * thirty points and will tell you whatever you want to hear.
 */
import { readFileSync } from 'node:fs';
import { Megacode, ACTIONS, FEATURES, GROUP_SPANS, encodeState } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';

const SEEDS = Number(process.argv[2] ?? 5);
const TRAIN = Number(process.argv[3] ?? 4000);
const EVAL = 400;

const url = (n) => new URL(`../data/out/${n}`, import.meta.url);
const right = JSON.parse(readFileSync(url('mb_circuit.json')));
const both = JSON.parse(readFileSync(url('mb_circuit_both.json')));

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

function withKC(c, n, seed) {
  if (n >= c.kc.length) return c;
  const r = rng(seed);
  const order = c.kc.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]];
  }
  const keep = order.slice(0, n);
  const map = new Map(keep.map((k, i) => [k, i]));
  return { ...c, kc: keep.map((i) => c.kc[i]),
           kc_mbon: c.kc_mbon.filter(([k]) => map.has(k)).map(([k, m, w]) => [map.get(k), m, w]) };
}

function trial(c, seed) {
  const fly = new MushroomBody(c, FEATURES.length, ACTIONS.length, { seed, groups: GROUP_SPANS });
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
  return { acc: pass(EVAL, 900000, false, true), syn: fly.w.length, active: fly.kActive };
}

const configs = [
  ['half a mushroom body', () => withKC(right, 1300, 5), 1300],
  ['one mushroom body (right)', () => right, right.kc.length],
  ['both hemispheres', () => both, both.kc.length],
];

console.log(`${SEEDS} seeds each, ${TRAIN} megacodes of training, ` +
            `${EVAL} unseen megacodes evaluated greedy\n`);
console.log('                              KC   active  synapses    mean    range');
for (const [name, build, nkc] of configs) {
  const c = build();
  const runs = [];
  let info = null;
  for (let s = 0; s < SEEDS; s++) {
    const r = trial(c, 11 + s * 101);
    runs.push(r.acc); info = r;
  }
  const mean = runs.reduce((a, b) => a + b, 0) / runs.length;
  const lo = Math.min(...runs), hi = Math.max(...runs);
  console.log(`${name.padEnd(28)}${String(nkc).padStart(5)}` +
              `${String(info.active).padStart(9)}${String(info.syn).padStart(10)}` +
              `${(mean * 100).toFixed(1).padStart(8)}%  ` +
              `${(lo * 100).toFixed(0)}-${(hi * 100).toFixed(0)}%`);
}
console.log('\nchance on this task: 11.8%');
