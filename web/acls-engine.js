/**
 * ACLS Megacode engine -- the teacher.
 *
 * Encodes the AHA 2025 Adult Cardiac Arrest Algorithm (Circulation, 22 Oct 2025).
 * See refs/aha2025-acls.md for the extracted source text and box flow.
 *
 * This module owns clinical correctness. The mushroom body decides; this grades.
 * Nothing here ever asks the fly what the right answer is.
 */

export const GUIDELINE_VERSION = 'AHA 2025 Guidelines for CPR & ECC, Part 9';

export const ACTIONS = [
  'shock',              // defibrillate
  'cpr',                // resume/continue 2 min of compressions
  'epinephrine',        // 1 mg IV/IO
  'amiodarone',         // 300 mg then 150 mg
  'lidocaine',          // 1-1.5 then 0.5-0.75 mg/kg
  'access',             // IV first, IO if IV fails
  'airway',             // advanced airway + waveform capnography
  'treat_cause',        // address the H or T in play
  'rhythm_check',       // pulse/rhythm check at the end of a cycle
  'post_arrest_care',   // after ROSC
];
export const A = Object.fromEntries(ACTIONS.map((a, i) => [a, i]));

export const REVERSIBLE_CAUSES = [
  'hypovolemia', 'hypoxia', 'hydrogen_ion', 'hypo_hyperkalemia', 'hypothermia',
  'tension_pneumothorax', 'tamponade', 'toxins', 'thrombosis_pulmonary',
  'thrombosis_coronary',
];

const SHOCKABLE = new Set(['VF', 'pVT']);
const CYCLE_SECONDS = 120;

/** Deterministic PRNG so a seed reproduces a whole megacode exactly. */
export function rng(seed) {
  let s = seed >>> 0 || 1;
  /*
   * Mix the seed before anything draws from it. A raw xorshift's first output is close
   * to monotonic in its seed, and Megacode's constructor spends that first draw on the
   * starting rhythm -- so consecutive seeds, which is how every script in pipeline/
   * builds its scenarios, produced VF and PEA only and never once pVT or asystole.
   * Every number this repository has published was measured on that half of the task.
   * Three lines of avalanche cost nothing and close the trap for good.
   */
  s = Math.imul(s ^ (s >>> 16), 0x45d9f3b) >>> 0;
  s = Math.imul(s ^ (s >>> 16), 0x45d9f3b) >>> 0;
  s = (s ^ (s >>> 16)) >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5;  s >>>= 0;
    return s / 4294967296;
  };
}

export class Megacode {
  constructor(seed = Date.now()) {
    this.rand = rng(seed);
    this.seed = seed;
    const r = this.rand();
    this.s = {
      rhythm: r < 0.4 ? 'VF' : r < 0.55 ? 'pVT' : r < 0.8 ? 'PEA' : 'asystole',
      t: 0,
      cycle: 0,
      phase: 'rhythm_check',   // rhythm_check -> shock? -> cpr -> interventions
      shocks: 0,
      epiCount: 0,
      tLastEpi: -Infinity,
      antiarrhythmicDoses: 0,
      antiarrhythmicDrug: null,
      access: false,
      airway: false,
      etco2: 8 + Math.floor(this.rand() * 8),
      cause: REVERSIBLE_CAUSES[Math.floor(this.rand() * REVERSIBLE_CAUSES.length)],
      causeTreated: false,
      rosc: false,
      over: false,
      cprSubsteps: 0,          // interventions taken inside the current 2-min block
    };
    this.log = [];
  }

  get state() { return this.s; }
  get shockable() { return SHOCKABLE.has(this.s.rhythm); }

  /**
   * Actions that the algorithm accepts right now, best first.
   * More than one may be acceptable (amiodarone OR lidocaine; which gap to fill
   * during a CPR block is not strictly ordered).
   */
  correctActions() {
    const s = this.s;
    if (s.over) return [];
    if (s.rosc) return [A.post_arrest_care];

    if (s.phase === 'rhythm_check') {
      return this.shockable ? [A.shock] : [A.cpr];
    }

    // Inside a 2-minute CPR block: fill the algorithm's gaps in priority order.
    const want = [];
    if (!s.access) want.push(A.access);
    // Asystole/PEA: epinephrine as soon as possible. VF/pVT: after the 2nd shock.
    // Box 9 puts "give epinephrine ASAP" ahead of securing access, so in a
    // non-shockable rhythm epinephrine must not be gated on access. In VF/pVT it
    // is not due until Box 6, by which point Box 4 has already called for access.
    const epiDue = s.t - s.tLastEpi >= 180;
    const epiAllowed = this.shockable ? s.shocks >= 2 : true;
    if (epiAllowed && epiDue && (s.access || !this.shockable)) want.push(A.epinephrine);
    // Antiarrhythmic after the 3rd shock, for VF/pVT only.
    if (this.shockable && s.shocks >= 3 && s.antiarrhythmicDoses < 2) {
      want.push(s.antiarrhythmicDrug === 'lidocaine' ? A.lidocaine : A.amiodarone);
      if (s.antiarrhythmicDoses === 0) want.push(A.lidocaine);
    }
    if (!s.airway && s.shocks >= 2) want.push(A.airway);
    if (!s.causeTreated) want.push(A.treat_cause);
    /*
     * Reassessing is only correct once this block's work is done. Accepting it at
     * every moment makes "check the rhythm" a permanently safe answer, and any
     * learner will collapse onto it and never learn the algorithm. It is also what
     * the guideline says: interrupting compressions early violates "minimize
     * interruptions in chest compressions".
     */
    if (!want.length) want.push(A.rhythm_check);
    return [...new Set(want)];
  }

  /** Apply an action, grade it, advance the clinical state. */
  step(actionIdx) {
    const s = this.s;
    const accepted = this.correctActions();
    const correct = accepted.includes(actionIdx);
    const action = ACTIONS[actionIdx];
    let note = '';

    switch (action) {
      case 'shock':
        if (this.shockable) {
          s.shocks++;
          note = `Shock ${s.shocks} · biphasic, per manufacturer (initial 120–200 J)`;
        } else {
          note = `${s.rhythm} is not shockable: no benefit, and compressions were interrupted`;
        }
        s.phase = 'cpr';
        break;

      case 'cpr':
        s.phase = 'cpr';
        note = 'Resume high-quality CPR · 100–120/min, full chest recoil';
        break;

      case 'epinephrine':
        s.epiCount++; s.tLastEpi = s.t;
        note = `Epinephrine 1 mg IV/IO (dose ${s.epiCount}) · every 3–5 min`;
        s.cprSubsteps++;
        break;

      case 'amiodarone':
      case 'lidocaine':
        s.antiarrhythmicDrug ??= action;
        s.antiarrhythmicDoses++;
        note = action === 'amiodarone'
          ? `Amiodarone ${s.antiarrhythmicDoses === 1 ? '300' : '150'} mg IV/IO`
          : `Lidocaine ${s.antiarrhythmicDoses === 1 ? '1–1.5' : '0.5–0.75'} mg/kg IV/IO`;
        s.cprSubsteps++;
        break;

      case 'access':
        s.access = true;
        note = 'IV access (2025: IV first; IO if IV fails or is not feasible)';
        s.cprSubsteps++;
        break;

      case 'airway':
        s.airway = true; s.etco2 += 2;
        note = 'Advanced airway + continuous waveform capnography · 1 breath every 6 s once placed';
        s.cprSubsteps++;
        break;

      case 'treat_cause':
        s.causeTreated = true;
        note = `Treat reversible cause: ${s.cause}`;
        s.cprSubsteps++;
        break;

      case 'rhythm_check':
        this._advanceCycle();
        note = `End of 2-minute cycle ${s.cycle}: reassess rhythm`;
        break;

      case 'post_arrest_care':
        s.over = true;
        note = 'ROSC: hand over to post-cardiac arrest care';
        break;
    }

    if (action !== 'rhythm_check' && action !== 'post_arrest_care') s.t += 15;
    if (s.t > 40 * 60 && !s.rosc) { s.over = true; note += ' · resuscitation past 40 minutes'; }

    const entry = { t: s.t, action, correct, note, accepted, rhythm: s.rhythm };
    this.log.push(entry);
    return { correct, accepted, note, action, done: s.over, state: { ...s } };
  }

  /** End of a 2-minute block: decide what the rhythm does next. */
  _advanceCycle() {
    const s = this.s;
    s.cycle++;
    s.t += CYCLE_SECONDS;
    s.cprSubsteps = 0;
    s.phase = 'rhythm_check';

    // ROSC odds rise with the interventions that actually change outcome.
    let p = 0.04;
    if (this.shockable) p += 0.09 * Math.min(s.shocks, 4);
    p += 0.05 * Math.min(s.epiCount, 3);
    if (s.causeTreated) p += 0.14;
    if (s.airway) p += 0.02;
    if (s.antiarrhythmicDoses > 0 && this.shockable) p += 0.05;
    p -= 0.012 * s.cycle;                      // downtime penalty

    if (this.rand() < Math.max(0.01, p)) {
      s.rhythm = 'ROSC'; s.rosc = true; s.etco2 = 35 + Math.floor(this.rand() * 10);
      return;
    }
    // Rhythm may degenerate or convert without ROSC.
    const r = this.rand();
    if (this.shockable) {
      if (r < 0.14) s.rhythm = 'asystole';
      else if (r < 0.26) s.rhythm = 'PEA';
    } else if (r < 0.10) {
      s.rhythm = 'VF';
    }
    s.etco2 = Math.max(5, s.etco2 - 1 + (s.causeTreated ? 2 : 0));
  }

  /** Compact, human-readable description of where we are. */
  describe() {
    const s = this.s;
    const mm = String(Math.floor(s.t / 60)).padStart(2, '0');
    const ss = String(s.t % 60).padStart(2, '0');
    return `${mm}:${ss} · ${s.rhythm} · shocks ${s.shocks} · epi ${s.epiCount}` +
           ` · ${s.access ? 'IV' : 'no access'} · ${s.airway ? 'advanced airway' : 'no airway'}` +
           ` · ETCO₂ ${s.etco2}`;
  }
}

/**
 * What the fly is shown. Each group is one clinical variable and behaves like an
 * olfactory glomerulus: exactly one of its channels is active at any moment, and
 * a Kenyon cell claw contacts the whole group rather than a single channel. That
 * is the real anatomy, and it is what lets a one-bit clinical distinction --
 * airway in or not -- actually change the Kenyon cell code.
 */
export const FEATURE_GROUPS = [
  { name: 'rhythm',         channels: ['VF', 'pVT', 'asystole', 'PEA', 'ROSC'] },
  { name: 'shockable',      channels: ['no', 'yes'] },
  { name: 'phase',          channels: ['rhythm_check', 'cpr'] },
  { name: 'shocks',         channels: ['0', '1', '2', '3plus'] },
  { name: 'epinephrine',    channels: ['none', 'due', 'recent'] },
  { name: 'antiarrhythmic', channels: ['0', '1', '2'] },
  { name: 'access',         channels: ['no', 'yes'] },
  { name: 'airway',         channels: ['no', 'yes'] },
  { name: 'cause_status',   channels: ['untreated', 'treated'] },
  { name: 'etco2',          channels: ['low', 'mid', 'high'] },
  { name: 'cycle',          channels: ['early', 'mid', 'late'] },
  { name: 'cause',          channels: REVERSIBLE_CAUSES },
];

/**
 * Which groups a bystander could read off the patient, and which are a record of
 * what has already been done. Access and airway stay observable -- you can see a
 * line and a tube -- but nothing shows you that this was the third shock.
 */
export const MEMORY_GROUPS = ['shocks', 'epinephrine', 'antiarrhythmic',
                              'cause_status', 'cycle'];

/**
 * Build a feature space. The default one gives the fly everything, including the
 * history, which means the engine is doing its remembering for it. Dropping the
 * memory groups and adding central-complex readout channels moves that job into
 * the animal.
 */
export function makeFeatureSpace({ omitMemory = false, cxGroups = 0, cxLevels = 3 } = {}) {
  const groups = FEATURE_GROUPS
    .filter((g) => !(omitMemory && MEMORY_GROUPS.includes(g.name)))
    .map((g) => ({ ...g }));
  for (let i = 0; i < cxGroups; i++) {
    groups.push({
      name: `cx${i}`, cx: i,
      channels: Array.from({ length: cxLevels }, (_, k) => `l${k}`),
    });
  }
  const features = groups.flatMap((g) => g.channels.map((c) => `${g.name}:${c}`));
  const spans = []; let at = 0;
  for (const g of groups) { spans.push([at, g.channels.length]); at += g.channels.length; }
  const ch = Object.fromEntries(features.map((f, i) => [f, i]));

  const encode = (s, shockable, cx = null) => {
    const v = new Float32Array(features.length);
    const on = (group, channel) => {
      const i = ch[`${group}:${channel}`];
      if (i !== undefined) v[i] = 1;
    };
    on('rhythm', s.rhythm);
    on('shockable', shockable ? 'yes' : 'no');
    on('phase', s.phase === 'rhythm_check' ? 'rhythm_check' : 'cpr');
    on('shocks', s.shocks === 0 ? '0' : s.shocks === 1 ? '1' : s.shocks === 2 ? '2' : '3plus');
    const dt = s.t - s.tLastEpi;
    on('epinephrine', s.epiCount === 0 ? 'none' : dt >= 180 ? 'due' : 'recent');
    on('antiarrhythmic', String(Math.min(s.antiarrhythmicDoses, 2)));
    on('access', s.access ? 'yes' : 'no');
    on('airway', s.airway ? 'yes' : 'no');
    on('cause_status', s.causeTreated ? 'treated' : 'untreated');
    on('etco2', s.etco2 < 10 ? 'low' : s.etco2 < 25 ? 'mid' : 'high');
    on('cycle', s.cycle <= 2 ? 'early' : s.cycle <= 5 ? 'mid' : 'late');
    on('cause', s.cause);
    // Central-complex readouts arrive as a level per channel, the same shape as
    // every other glomerulus, so the Kenyon cell code keeps its statistics.
    if (cx) {
      for (const g of groups) {
        if (g.cx === undefined) continue;
        const x = Math.max(-1, Math.min(1, cx[g.cx] ?? 0));
        const lvl = Math.min(cxLevels - 1, Math.floor(((x + 1) / 2) * cxLevels));
        on(g.name, `l${lvl}`);
      }
    }
    return v;
  };
  return { groups, FEATURES: features, GROUP_SPANS: spans, encode };
}

/** Flat channel list, plus the [start, length] span of each group. */
const DEFAULT_SPACE = makeFeatureSpace();
export const FEATURES = DEFAULT_SPACE.FEATURES;
export const GROUP_SPANS = DEFAULT_SPACE.GROUP_SPANS;
export const encodeState = (s, shockable) => DEFAULT_SPACE.encode(s, shockable);
