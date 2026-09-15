/**
 * What does blind guessing score on this task?
 *
 *   node pipeline/chance.mjs [megacodes=3000] [seeds=3]
 *
 * A guesser picks one of the ten actions uniformly at random at every step and plays
 * the megacode out under the real grader. Its accuracy is the chance line every other
 * script is compared against.
 *
 * This replaces two numbers that were typed in by hand in the first commit and never
 * computed: 11.8% in the scripts and README, 16% on the page. The engine has changed
 * since (the rhythm check can no longer be answered at every moment), and a constant
 * does not notice.
 *
 * Run with and without the 60-step cap the experiments use: if a random guesser hit the
 * cap, the capped figure would be graded on a different mix of situations. It does not
 * -- a guesser's megacode ends in ~10 steps -- so the two agree.
 */
import { Megacode, ACTIONS } from '../web/acls-engine.js';

const N = Number(process.argv[2] ?? 3000);
const SEEDS = Number(process.argv[3] ?? 3);

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

function guess(seed, cap) {
  const r = rng(seed);
  let right = 0, total = 0, accepted = 0, capped = 0, steps = 0;
  for (let e = 0; e < N; e++) {
    const mc = new Megacode(900000 + e);   // the same unseen scenarios the experiments evaluate on
    let s = 0;
    for (; s < cap && !mc.state.over; s++) {
      const ok = mc.correctActions();
      if (!ok.length) break;
      accepted += ok.length;
      right += mc.step((r() * ACTIONS.length) | 0).correct ? 1 : 0;
      total++;
    }
    if (s === cap) capped++;
    steps += s;
  }
  return { acc: right / total, accepted: accepted / total, capped: capped / N, steps: steps / N };
}

console.log(`uniform random guesser, ${SEEDS} seeds x ${N} megacodes\n`);
for (const [label, cap] of [['60-step cap', 60], ['no cap', Infinity]]) {
  const runs = Array.from({ length: SEEDS }, (_, s) => guess(s + 1, cap));
  const mean = (k) => runs.reduce((a, x) => a + x[k], 0) / runs.length;
  const accs = runs.map((x) => x.acc * 100);
  console.log(`${label.padEnd(12)} chance ${(mean('acc') * 100).toFixed(1)}% ` +
              `(${Math.min(...accs).toFixed(1)}-${Math.max(...accs).toFixed(1)}), ` +
              `grader accepts ${mean('accepted').toFixed(2)} of ${ACTIONS.length} per step, ` +
              `${mean('steps').toFixed(1)} steps per megacode, ${(mean('capped') * 100).toFixed(0)}% capped`);
}
