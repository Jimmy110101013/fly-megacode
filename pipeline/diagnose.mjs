/**
 * Phase 1 evidence gathering. Instruments every boundary in the pipeline:
 *   engine -> features -> KC code -> MBON readout -> value -> dopamine -> weight change
 * Reports numbers only. No fixes here.
 */
import { readFileSync } from 'node:fs';
import { Megacode, ACTIONS, FEATURES, GROUP_SPANS, encodeState } from '../web/acls-engine.js';
import { MushroomBody } from '../web/mushroom-body.js';

const circuit = JSON.parse(readFileSync(new URL('../data/out/mb_circuit.json', import.meta.url)));
const fly = new MushroomBody(circuit, FEATURES.length, ACTIONS.length, { seed: 11, groups: GROUP_SPANS });
const jac = (a, b) => { const s = new Set(b); return a.filter((x) => s.has(x)).length / a.length; };

console.log('== boundary 1: engine -> accepted-action set ==');
{
  let sizes = [], feats = [];
  for (let e = 0; e < 200; e++) {
    const mc = new Megacode(500 + e);
    for (let i = 0; i < 20 && !mc.state.over; i++) {
      const acc = mc.correctActions(); if (!acc.length) break;
      sizes.push(acc.length);
      feats.push(encodeState(mc.state, mc.shockable).reduce((a, b) => a + b, 0));
      mc.step(acc[0]);
    }
  }
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  console.log(`  mean accepted actions: ${mean(sizes).toFixed(2)} of ${ACTIONS.length}`);
  console.log(`  => uniform-random accuracy would be ${(mean(sizes) / ACTIONS.length * 100).toFixed(1)}%`);
  console.log(`  mean active features: ${mean(feats).toFixed(1)} of ${FEATURES.length}`);
}

console.log('\n== boundary 2: features -> KC code (is the code action-specific?) ==');
{
  const mc = new Megacode(42);
  const f = encodeState(mc.state, mc.shockable);
  fly.primeState(f);
  const codes = [...Array(ACTIONS.length).keys()].map((a) => fly.encode(f, a, true));
  let ov = 0, n = 0;
  for (let i = 0; i < codes.length; i++) {
    for (let j = i + 1; j < codes.length; j++) { ov += jac(codes[i], codes[j]); n++; }
  }
  console.log(`  overlap between codes for DIFFERENT ACTIONS, same state: ${(ov / n * 100).toFixed(1)}%`);
  console.log(`  (want low -- high means the fly cannot tell actions apart)`);

  const mc2 = new Megacode(99);
  const f2 = encodeState(mc2.state, mc2.shockable);
  const cA = fly.encode(f, 0), cB = fly.encode(f2, 0);
  console.log(`  overlap between codes for DIFFERENT STATES, same action: ${(jac(cA, cB) * 100).toFixed(1)}%`);
}

console.log('\n== boundary 3: KC code -> MBON -> value spread across actions ==');
{
  const mc = new Megacode(42);
  const f = encodeState(mc.state, mc.shockable);
  fly.primeState(f);
  const vals = [...Array(ACTIONS.length).keys()].map((a) => fly.readout(fly.encode(f, a, true)).value);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const sd = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length);
  console.log(`  values: ${vals.map((v) => v.toFixed(3)).join(' ')}`);
  console.log(`  mean ${mean.toFixed(3)}  sd ${sd.toFixed(4)}  sd/|mean| ${(sd / Math.abs(mean)).toFixed(4)}`);
  console.log(`  at temperature 0.25 a spread this size gives softmax ratio ` +
              `${Math.exp((Math.max(...vals) - Math.min(...vals)) / 0.25).toFixed(2)}x best:worst`);
}

console.log('\n== boundary 4: dopamine geometry (does reward differ from punishment?) ==');
{
  const sign = fly.mbonSign, pam = fly.pamDrive, ppl = fly.ppl1Drive;
  const nPos = sign.reduce((s, v) => s + (v > 0), 0);
  console.log(`  MBON polarity: ${nPos} cholinergic(+)  ${sign.length - nPos} GABA/glut(-)`);
  const cov = (v) => v.reduce((s, x) => s + (x > 0.01), 0);
  console.log(`  MBONs innervated: PAM ${cov(pam)}/${sign.length}   PPL1 ${cov(ppl)}/${sign.length}`);
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const nrm = (a) => Math.sqrt(dot(a, a));
  console.log(`  cosine(PAM pattern, PPL1 pattern) = ${(dot(pam, ppl) / (nrm(pam) * nrm(ppl))).toFixed(3)}`);
  console.log(`    -> near 1.0 means reward and punishment depress the SAME compartments`);
  console.log(`  signed drive  sum_j sign_j*PAM_j  = ${dot(sign, pam).toFixed(2)}   (want negative: reward disinhibits)`);
  console.log(`  signed drive  sum_j sign_j*PPL1_j = ${dot(sign, ppl).toFixed(2)}   (want positive: punishment suppresses)`);
}

console.log('\n== boundary 5: does one learning step move value the right way? ==');
{
  const test = (correct) => {
    const f2 = new MushroomBody(circuit, FEATURES.length, ACTIONS.length, { seed: 11, groups: GROUP_SPANS });
    const mc = new Megacode(42);
    const feats = encodeState(mc.state, mc.shockable);
    const a = 3;
    f2.primeState(feats);
    const before = f2.readout(f2.encode(feats, a, true)).value;
    f2.decide(feats);                       // populates the trace
    f2.last.activeKC = f2.encode(feats, a, true);
    f2.last.action = a;
    f2.learn(correct);
    f2.primeState(feats);
    const after = f2.readout(f2.encode(feats, a, true)).value;
    return { before, after, delta: after - before };
  };
  const r = test(true), w = test(false);
  console.log(`  after REWARD   : ${r.before.toFixed(4)} -> ${r.after.toFixed(4)}  delta ${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(5)}  (want +)`);
  console.log(`  after PUNISH   : ${w.before.toFixed(4)} -> ${w.after.toFixed(4)}  delta ${w.delta >= 0 ? '+' : ''}${w.delta.toFixed(5)}  (want -)`);
  console.log(`  differential   : ${(r.delta - w.delta).toFixed(5)}  (want clearly positive)`);
}
