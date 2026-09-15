/**
 * Does the fly's real relay wiring carry anything, or would any wiring do?
 *
 *   node pipeline/rewire-control.mjs [seeds] [train]
 *
 * Making MBON -> relay plastic gives the model 7,916 modifiable weights over 1,951
 * interneurons. That is enough machinery that it might learn the task through the
 * connectome rather than because of it. The control is a degree-preserving rewire:
 * every MBON keeps the same number of outgoing connections, every relay the same
 * number in and out, every descending neuron the same number of inputs -- only WHO
 * CONNECTS TO WHOM is destroyed.
 *
 * Two things are deliberately held fixed, so this changes one thing and not three:
 *   - the dopamine gate stays attached to the same relay INDEX, so the same cells
 *     are plastic in both arms
 *   - the ten output channels stay the same descending types, so the naming and
 *     the drive ranking do not move
 *
 * real wins    the measured wiring carries task-relevant structure
 * a draw       it is a 1,951-dimensional random projection and the connectome is
 *              contributing nothing here. That is the more publishable result.
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

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

/**
 * Double-edge swap on a weighted bipartite edge list [pre, post, w]. Picking two
 * edges and exchanging their targets leaves every out-degree and every in-degree
 * untouched; only the pairing moves. Self-pairs and duplicates are rejected so the
 * shuffled graph stays simple, like the measured one.
 */
function shuffle(edges, seed, rounds = 20) {
  const e = edges.map((x) => x.slice());
  const rand = rng(seed);
  const key = (a, b) => `${a}>${b}`;
  const seen = new Set(e.map((x) => key(x[0], x[1])));
  let swaps = 0;
  for (let pass = 0; pass < rounds * e.length; pass++) {
    const i = Math.floor(rand() * e.length), j = Math.floor(rand() * e.length);
    if (i === j) continue;
    const [a, b] = e[i], [c, d] = e[j];
    if (a === c || b === d) continue;
    if (seen.has(key(a, d)) || seen.has(key(c, b))) continue;
    seen.delete(key(a, b)); seen.delete(key(c, d));
    seen.add(key(a, d)); seen.add(key(c, b));
    e[i][1] = d; e[j][1] = b;
    swaps++;
  }
  return { edges: e, swaps };
}

/** A rewired copy. Dopamine gates and channel selection are carried over untouched. */
function rewire(dn, seed) {
  const a = shuffle(dn.mbon_relay, seed);
  const b = shuffle(dn.relay_dn, seed ^ 0x5bf03635);
  return { copy: { ...dn, mbon_relay: a.edges, relay_dn: b.edges }, swaps: a.swaps + b.swaps };
}

function run(dn, seed, plastic) {
  const output = new DescendingReadout(dn, ACTIONS.length,
                                       { plastic, gate: 'da' });
  const fly = new MushroomBody(circuit, space.FEATURES.length, ACTIONS.length,
                               { seed, groups: space.GROUP_SPANS, stateClaws: 6,
                                 output, lr: 0.006 });
  const cx = new CentralComplex(cxCircuit, { readouts: 8, seed });
  const ep = (e, greedy) => {
    const mc = new Megacode(e); cx.reset(); output.reset();
    let r = 0, t = 0;
    for (let s = 0; s < 60 && !mc.state.over; s++) {
      if (!mc.correctActions().length) break;
      const a = fly.act(space.encode(mc.state, mc.shockable, cx.out), { greedy });
      const res = mc.step(a);
      fly.learn(res.correct);
      cx.step(fly.last.learned);
      r += res.correct ? 1 : 0; t++;
    }
    return [r, t];
  };
  const keep = fly.p.lr; fly.p.lr = 0;
  for (let e = 0; e < 200; e++) ep(700000 + seed * 1000 + e, false);
  fly.p.lr = keep; output.freeze(); fly.trials = 0;
  for (let e = 0; e < TRAIN; e++) ep(seed * 100000 + e, false);
  let r = 0, t = 0;
  for (let e = 0; e < EVAL; e++) { const [x, y] = ep(900000 + seed * 10000 + e, true); r += x; t += y; }
  return r / t;
}

const probe = rewire(dnCircuit, 1);
console.log(`degree-preserving rewire: ${probe.swaps} accepted swaps over `
          + `${dnCircuit.mbon_relay.length} MBON->relay and ${dnCircuit.relay_dn.length} relay->DN edges`);
console.log(`${SEEDS} seeds, ${TRAIN} training megacodes, ${EVAL} unseen evaluated greedy.`);
console.log('Chance 11.8%; the untrained best-permutation bar is 33.1%.\n');
console.log('wiring            MBON->relay      accuracy');
for (const [label, mk] of [['measured', (s) => dnCircuit],
                           ['rewired', (s) => rewire(dnCircuit, 8000 + s).copy]]) {
  for (const plastic of [false, true]) {
    const acc = [];
    for (let s = 0; s < SEEDS; s++) acc.push(run(mk(s), 11 + s * 7, plastic));
    const mean = acc.reduce((x, y) => x + y, 0) / acc.length;
    console.log(`${label.padEnd(17)} ${(plastic ? 'plastic' : 'fixed').padEnd(16)} `
              + `${(mean * 100).toFixed(1).padStart(5)}%   (${acc.map((a) => (a * 100).toFixed(0)).join('-')})`);
  }
}
