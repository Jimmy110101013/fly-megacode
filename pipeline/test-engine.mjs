/**
 * The engine is the clinical source of truth. If it teaches the fly something
 * the guideline does not say, everything downstream is wrong.
 * Each check cites the box or bullet it comes from (refs/aha2025-acls.md).
 *
 *   node pipeline/test-engine.mjs
 */
import { Megacode, ACTIONS, A } from '../web/acls-engine.js';

const checks = [];
const add = (name, pass, detail = '') => checks.push({ name, pass, detail });
const names = (set) => set.map((i) => ACTIONS[i]).join(', ') || '(none)';

/** Build a megacode and force it into a chosen rhythm/phase. */
function at(patch) {
  const mc = new Megacode(1);
  Object.assign(mc.state, patch);
  return mc;
}

// Box 9: "Asystole/PEA. Give Epinephrine ASAP."
for (const rhythm of ['asystole', 'PEA']) {
  const mc = at({ rhythm, phase: 'cpr', access: false, epiCount: 0, tLastEpi: -Infinity });
  const acc = mc.correctActions();
  add(`Box 9: epinephrine is accepted ASAP in ${rhythm}, before access exists`,
      acc.includes(A.epinephrine), `accepted = ${names(acc)}`);
}

// Box 1 / 3: a shockable rhythm at a rhythm check is shocked.
for (const rhythm of ['VF', 'pVT']) {
  const mc = at({ rhythm, phase: 'rhythm_check' });
  add(`Box 3: ${rhythm} at a rhythm check -> shock`,
      mc.correctActions().includes(A.shock), `accepted = ${names(mc.correctActions())}`);
}

// Non-shockable at a rhythm check is not shocked.
for (const rhythm of ['asystole', 'PEA']) {
  const mc = at({ rhythm, phase: 'rhythm_check' });
  add(`${rhythm} at a rhythm check is NOT shocked`,
      !mc.correctActions().includes(A.shock), `accepted = ${names(mc.correctActions())}`);
}

// Box 6: in VF/pVT epinephrine belongs after the second shock, not before.
{
  const early = at({ rhythm: 'VF', phase: 'cpr', shocks: 1, access: true, tLastEpi: -Infinity });
  add('Box 6: epinephrine is not yet due in VF after only one shock',
      !early.correctActions().includes(A.epinephrine), `accepted = ${names(early.correctActions())}`);
  const due = at({ rhythm: 'VF', phase: 'cpr', shocks: 2, access: true, tLastEpi: -Infinity });
  add('Box 6: epinephrine is accepted in VF after the second shock',
      due.correctActions().includes(A.epinephrine), `accepted = ${names(due.correctActions())}`);
}

// Box 8: "Amiodarone or lidocaine" -- both acceptable for the first dose.
{
  const mc = at({ rhythm: 'VF', phase: 'cpr', shocks: 3, access: true, epiCount: 1, tLastEpi: 0, t: 60 });
  const acc = mc.correctActions();
  add('Box 8: after the third shock both amiodarone and lidocaine are accepted',
      acc.includes(A.amiodarone) && acc.includes(A.lidocaine), `accepted = ${names(acc)}`);
}

// Once a drug is chosen the second dose must stay on that drug.
{
  const mc = at({ rhythm: 'VF', phase: 'cpr', shocks: 3, access: true, epiCount: 1, tLastEpi: 0,
                  t: 60, antiarrhythmicDoses: 1, antiarrhythmicDrug: 'amiodarone' });
  const acc = mc.correctActions();
  add('second antiarrhythmic dose stays on the drug already started',
      acc.includes(A.amiodarone) && !acc.includes(A.lidocaine), `accepted = ${names(acc)}`);
}

// Only two antiarrhythmic doses exist in the algorithm.
{
  const mc = at({ rhythm: 'VF', phase: 'cpr', shocks: 5, access: true, epiCount: 2, tLastEpi: 0,
                  t: 60, antiarrhythmicDoses: 2, antiarrhythmicDrug: 'amiodarone' });
  const acc = mc.correctActions();
  add('no third antiarrhythmic dose is accepted',
      !acc.includes(A.amiodarone) && !acc.includes(A.lidocaine), `accepted = ${names(acc)}`);
}

// Epinephrine interval: 1 mg every 3 to 5 minutes, not more often.
{
  const mc = at({ rhythm: 'PEA', phase: 'cpr', access: true, epiCount: 1, tLastEpi: 60, t: 120 });
  add('epinephrine is not accepted again 1 minute after the last dose',
      !mc.correctActions().includes(A.epinephrine), `accepted = ${names(mc.correctActions())}`);
  const later = at({ rhythm: 'PEA', phase: 'cpr', access: true, epiCount: 1, tLastEpi: 60, t: 60 + 200 });
  add('epinephrine is accepted again after 3 minutes',
      later.correctActions().includes(A.epinephrine), `accepted = ${names(later.correctActions())}`);
}

// "Minimize interruptions in chest compressions": reassessing early is a deviation.
{
  const pending = at({ rhythm: 'VF', phase: 'cpr', shocks: 2, access: false });
  add('rhythm check is NOT accepted while the block still has work pending',
      !pending.correctActions().includes(A.rhythm_check),
      `accepted = ${names(pending.correctActions())}`);
  const done = at({ rhythm: 'VF', phase: 'cpr', shocks: 1, access: true, causeTreated: true,
                    tLastEpi: 0, t: 30 });
  add('rhythm check IS accepted once nothing else is outstanding',
      done.correctActions().includes(A.rhythm_check),
      `accepted = ${names(done.correctActions())}`);
}

// ROSC leaves the arrest algorithm.
{
  const mc = at({ rosc: true, rhythm: 'ROSC' });
  const acc = mc.correctActions();
  add('after ROSC the only accepted action is post-cardiac-arrest care',
      acc.length === 1 && acc[0] === A.post_arrest_care, `accepted = ${names(acc)}`);
}

let failed = 0;
for (const c of checks) {
  console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.name}${c.detail ? `\n        ${c.detail}` : ''}`);
  if (!c.pass) failed++;
}
console.log(failed ? `\n${failed} of ${checks.length} check(s) failed` : `\nall ${checks.length} checks passed`);
process.exit(failed ? 1 : 0);
