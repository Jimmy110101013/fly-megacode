/**
 * Gate on the whole "no menu" experiment: can the measured MBON -> DN pathway
 * express ten different actions at all?
 *
 *   node pipeline/dn-separable.mjs [nActions]
 *
 * Replacing the action menu with a descending-neuron readout means the action is
 * whichever DN wins, and MBON -> DN is anatomy: fixed, never trained. So the
 * question comes before any learning rule. Granting the mushroom body PERFECT
 * learning, can each of the ten output channels be made to win? If not, the
 * architecture cannot express ten actions and no training schedule rescues it.
 *
 * Three admissible sets for the MBON activity vector m, in order of honesty:
 *
 *   free        m anywhere in R^96          -- pure upper bound, not achievable
 *   sign-only   m <= 0                      -- reading the learned deviation w - w0
 *   real        0 <= m_j <= M_j             -- what a Kenyon cell code can actually
 *                                              drive, given the anatomical baseline
 *                                              M_j and depression-only plasticity
 *
 * The third is the one that decides the experiment: a fly can only depress
 * KC->MBON, so it can lower an MBON's output from its anatomical baseline but
 * never push it above. Behaviour is released, not commanded.
 */
import { readFileSync } from 'node:fs';

const NACT = Number(process.argv[2] ?? 10);
const url = (n) => new URL(`../data/out/${n}`, import.meta.url);
const dn = JSON.parse(readFileSync(url('dn_circuit.json')));
const mb = JSON.parse(readFileSync(new URL('../web/mb_circuit.json', import.meta.url)));

const nMbon = dn.n_mbon;
const SPARSE = 39;                       // Kenyon cells active per state, from the model

/** Anatomical ceiling on each MBON's output under a sparse Kenyon-cell code. */
const ceiling = (() => {
  const tot = new Float64Array(nMbon);
  for (const [, j, w] of mb.kc_mbon) tot[j] += w;
  return Float64Array.from(tot, (t) => t * SPARSE / mb.kc.length);
})();

/**
 * Output channels. Left and right cells of the same descending type are the same
 * command neuron on two sides of the animal, so they are pooled: one action per
 * DN *type*, taken off the top of the anatomical drive ranking. Pooling also
 * avoids handing two channels a cosine of 0.94 and calling them different actions.
 */
const chosen = [];
const seen = new Set();
for (const j of dn.by_mb_drive) {
  const t = dn.neurons[j].t;
  if (t === 'unknown' || seen.has(t)) continue;
  seen.add(t); chosen.push(t);
  if (chosen.length === NACT) break;
}
const members = chosen.map((t) => dn.by_mb_drive.filter((j) => dn.neurons[j].t === t));

const W = members.map((js) => {
  const v = new Float64Array(nMbon);
  for (const [i, jj, w] of dn.mbon_dn) if (js.includes(jj)) v[i] += w;
  return Float64Array.from(v, (x) => x / js.length);       // mean over the pooled cells
});

/**
 * The relayed pathway: MBON -> interneuron -> DN, the way most of the mushroom
 * body's reach into the descending neurons actually runs. The relay layer is
 * anatomy, fixed like the rest, and it rectifies -- which is what lets the pathway
 * do something a single fixed matrix cannot.
 */
const nRelay = dn.n_relay;
const RELAY_SCALE = 0.02;                // keeps relay output in the same range as m
const relayIn = dn.mbon_relay;
const relayOut = members.map((js) => {
  const set = new Set(js);
  const v = new Float64Array(nRelay);
  for (const [c, j, w] of dn.relay_dn) if (set.has(j)) v[c] += w;
  return Float64Array.from(v, (x) => x / js.length);
});

/** DN drive for the chosen channels, direct plus relayed. Returns [d, relay]. */
function forward(m) {
  const r = new Float64Array(nRelay);
  for (const [i, c, w] of relayIn) r[c] += w * m[i];
  for (let c = 0; c < nRelay; c++) r[c] = r[c] > 0 ? r[c] * RELAY_SCALE : 0;   // rectify
  const d = W.map((w, k) => dot(w, m) + dot(relayOut[k], r));
  return [d, r];
}

/** d(margin)/dm for channel j against channel k, through the rectified relay. */
function grad(m, r, j, k) {
  const g = Float64Array.from(W[j], (w, i) => w - W[k][i]);
  const back = new Float64Array(nRelay);
  for (let c = 0; c < nRelay; c++) {
    if (r[c] > 0) back[c] = (relayOut[j][c] - relayOut[k][c]) * RELAY_SCALE;
  }
  for (const [i, c, w] of relayIn) g[i] += w * back[c];
  return g;
}

/** Same question as winnable(), but asked of the whole two-hop pathway. */
function winnableRelayed(j) {
  const m = Float64Array.from(ceiling, (c) => c * 0.5);
  const margin = () => {
    const [d] = forward(m);
    let worst = Infinity, worstK = -1;
    for (let k = 0; k < d.length; k++) {
      if (k === j) continue;
      if (d[j] - d[k] < worst) { worst = d[j] - d[k]; worstK = k; }
    }
    return { worst, worstK };
  };
  let step = 0.02;
  for (let it = 0; it < 4000; it++) {
    const { worst, worstK } = margin();
    if (worst > 1e-9 && it > 100) break;
    const [, r] = forward(m);
    const g = grad(m, r, j, worstK);
    const gn = norm(g) || 1;
    for (let i = 0; i < nMbon; i++) {
      m[i] = Math.min(ceiling[i], Math.max(0, m[i] + step * ceiling[i] * g[i] / gn));
    }
    step *= 0.9992;
  }
  const { worst } = margin();
  const [d] = forward(m);
  const scale = Math.max(...d.map(Math.abs)) || 1;
  return { ok: worst > 1e-9, margin: worst / scale };
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const norm = (a) => Math.sqrt(dot(a, a));

/** Is channel j the strict winner for some admissible m? Ascend the worst margin. */
function winnable(j, mode) {
  const m = new Float64Array(nMbon);
  for (let i = 0; i < nMbon; i++) {
    m[i] = mode === 'free' ? 0.01 : mode === 'sign' ? -0.01 : ceiling[i] * 0.5;
  }
  const project = (i) => {
    if (mode === 'sign' && m[i] > 0) m[i] = 0;
    if (mode === 'real') m[i] = Math.min(ceiling[i], Math.max(0, m[i]));
  };
  const margin = () => {
    let worst = Infinity, worstK = -1;
    for (let k = 0; k < W.length; k++) {
      if (k === j) continue;
      const g = dot(W[j], m) - dot(W[k], m);
      if (g < worst) { worst = g; worstK = k; }
    }
    return { worst, worstK };
  };
  let step = 0.5;
  for (let it = 0; it < 8000; it++) {
    const { worst, worstK } = margin();
    if (worst > 1e-9 && it > 100) break;
    for (let i = 0; i < nMbon; i++) { m[i] += step * (W[j][i] - W[worstK][i]); project(i); }
    if (mode !== 'real') { const n = norm(m); if (n > 1) for (let i = 0; i < nMbon; i++) m[i] /= n; }
    step *= 0.9995;
  }
  const { worst } = margin();
  const scale = mode === 'real' ? Math.max(...W.map((w) => Math.abs(dot(w, m)))) || 1 : norm(m) || 1;
  return { ok: worst > 1e-9, margin: worst / scale };
}

const live = new Set(dn.mbon_dn.map((e) => e[0]));
console.log(`MBON -> DN: ${dn.mbon_dn.length} edges at >= ${dn.min_syn} synapses; `
          + `${live.size} of ${nMbon} MBONs reach a descending neuron,`);
console.log(`${dn.by_mb_drive.length} descending neurons reached. Whatever the other `
          + `${nMbon - live.size} MBONs learn never leaves the brain.\n`);

console.log(`${NACT} output channels, one per descending type, by mushroom-body drive:`);
for (let k = 0; k < chosen.length; k++) {
  const inputs = W[k].reduce((s, w) => s + (w !== 0 ? 1 : 0), 0);
  const exc = W[k].reduce((s, w) => s + (w > 0 ? 1 : 0), 0);
  console.log(`  ${String(k).padStart(2)}  ${chosen[k].padEnd(10)} ${members[k].length} cell(s), `
            + `${inputs} MBON inputs (${exc} excitatory), |w| = ${norm(W[k]).toFixed(1)}`);
}

let worstPair = { c: -2, a: -1, b: -1 };
for (let a = 0; a < W.length; a++) for (let b = a + 1; b < W.length; b++) {
  const c = dot(W[a], W[b]) / ((norm(W[a]) * norm(W[b])) || 1);
  if (c > worstPair.c) worstPair = { c, a, b };
}
console.log(`\nmost similar pair: ${chosen[worstPair.a]} vs ${chosen[worstPair.b]}, `
          + `cosine ${worstPair.c.toFixed(3)}`);

const report = (label, res) => {
  const won = res.filter((r) => r.ok).length;
  console.log(`\n${label}`);
  console.log(`  ${won}/${NACT} channels reachable`);
  console.log(`  ${chosen.map((c, k) => `${c}:${res[k].ok ? res[k].margin.toFixed(2) : 'no'}`).join('  ')}`);
};

for (const [label, mode] of [['direct only, free        m anywhere in R^96', 'free'],
                             ['direct only, sign-only   m <= 0', 'sign'],
                             ['direct only, real        0 <= m <= anatomical ceiling', 'real']]) {
  report(label, chosen.map((_, j) => winnable(j, mode)));
}

const mbonsIntoRelay = new Set(relayIn.map((e) => e[0])).size;
console.log(`\nrelayed pathway: ${nRelay} interneurons, ${relayIn.length} MBON->relay edges `
          + `from ${mbonsIntoRelay} MBONs, ${dn.relay_dn.length} relay->DN edges`);
report('direct + relayed, real   0 <= m <= anatomical ceiling', chosen.map((_, j) => winnableRelayed(j)));
