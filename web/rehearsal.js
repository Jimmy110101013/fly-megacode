/**
 * Rehearsal: one fixed patient per starting rhythm, run by the fly as it is right now.
 *
 * The fly holds no sequence anywhere. What its KC->MBON weights hold is what to do in
 * a state; the order comes out one step at a time -- it acts, the grader changes the
 * patient, the central complex keeps the record, and the next state asks again. So
 * the honest way to show "the order it has learned" is to let it run one.
 *
 * Nothing here trains or disturbs the live animal. Each action is the greedy pick
 * from memorySplit (reward - punishment, exactly what decide compares), so no draw is
 * taken from the fly's random generator and fly.last is never touched; the history
 * runs through a second central complex with the live one's gain calibration copied
 * in. Patients are fixed, so a change on screen can only come from learning, and the
 * fly never trains on them. One patient per rhythm is a sketch, not an evaluation: a
 * different patient with the same rhythm can take a different road.
 */
import { CentralComplex } from './central-complex.js';
import { fit, label, INK, MUTED } from './brainview.js';

const RHYTHMS = ['VF', 'pVT', 'PEA', 'asystole'];
const STEPS = 14;
const TICK = 0.25;            // re-run one row per tick, so no single frame pays for all four
export const SHORT = {
  shock: 'shock', cpr: 'CPR', epinephrine: 'epi', amiodarone: 'amio', lidocaine: 'lido',
  access: 'IV', airway: 'airway', treat_cause: 'cause', rhythm_check: 'check',
  post_arrest_care: 'post',
};

export class Rehearsal {
  constructor(canvas, { fly, cx, cxCircuit, space, Megacode, actions }) {
    Object.assign(this, { canvas, fly, cx, space, Megacode, actions });
    this.cx2 = new CentralComplex(cxCircuit, cx.p);   // same options, so the same reservoir
    this.split = new Float32Array(actions.length * 2);
    // Seeds spread by a large odd multiplier: the engine's xorshift gives near-identical
    // first draws for neighbouring seeds, and a run of consecutive seeds only ever
    // started in VF or PEA.
    this.patients = RHYTHMS.map((rhythm) => {
      let seed = null;
      for (let i = 1; i < 5000 && seed === null; i++) {
        const s = (Math.imul(0x9E3779B1, i) + 0xC0FFEE) >>> 0;
        if (new Megacode(s).state.rhythm === rhythm) seed = s;
      }
      return { rhythm, seed, steps: [], stopped: null, seenAt: -1 };
    });
    this.row = 0;
    this.clock = TICK;
  }

  /** Forget the cached runs; the next ticks re-run every patient. */
  reset() { for (const p of this.patients) p.seenAt = -1; }

  _run(p) {
    const { fly, space, split } = this;
    const n = this.actions.length, cx2 = this.cx2;
    const net = (a) => split[2 * a] - split[2 * a + 1];
    const m = new this.Megacode(p.seed);
    cx2.reset();
    cx2.mu.set(this.cx.mu); cx2.mad.set(this.cx.mad);
    p.steps = []; p.stopped = null;
    let rhythm = m.state.rhythm;
    for (let i = 0; i < STEPS && !m.state.over && m.correctActions().length; i++) {
      const f = space.encode(m.state, m.shockable, cx2.out);
      fly.memorySplit(f, split);
      let best = 0;
      for (let a = 1; a < n; a++) if (net(a) > net(best)) best = a;
      let second = -Infinity;
      for (let a = 0; a < n; a++) if (a !== best) second = Math.max(second, net(a));
      // A naive fly values every action at exactly zero. Taking the first action on a
      // tie would draw a confident "shock" that the animal never chose.
      if (net(best) - second < 1e-6) { p.stopped = 'no preference yet'; break; }
      const learned = fly.readout(fly.encode(f, best)).learned;
      const res = m.step(best);
      cx2.step(learned);
      const turned = m.state.rhythm !== rhythm ? m.state.rhythm : null;
      rhythm = m.state.rhythm;
      p.steps.push({ action: best, correct: res.correct, wanted: res.correct ? null : res.accepted, turned });
    }
  }

  render(dt) {
    this.clock += dt;
    if (this.clock >= TICK) {
      this.clock = 0;
      const p = this.patients[this.row];
      this.row = (this.row + 1) % this.patients.length;
      // Weights only change when the fly learns, so an unchanged trial count means an
      // unchanged answer -- a paused page does no work here.
      if (p.seed !== null && p.seenAt !== this.fly.trials) { this._run(p); p.seenAt = this.fly.trials; }
    }

    // Layout first, from the width alone, so the panel's height does not depend on
    // how far any run got and cannot feed back into the canvas size.
    const cell = this.canvas.parentElement;
    const width = cell.getBoundingClientRect().width;
    const pad = 16, narrow = width < 520;
    const labW = narrow ? 58 : 84, scoreW = narrow ? 46 : 64;
    const cellW = narrow ? 44 : 54, gap = 6, lineH = 44;
    const perLine = Math.max(1, Math.floor((width - pad * 2 - labW - scoreW + gap) / (cellW + gap)));
    const lines = Math.ceil((STEPS + 1) / perLine);
    const legend = ['the fly as it is now, greedy, learning nothing',
      'struck through: the guideline wanted what is written under it'];
    const twoLine = width < 760;                     // the joined legend is ~560 px of mono
    const top = twoLine ? 58 : 46;
    const need = Math.ceil(top + RHYTHMS.length * (lines * lineH + 10));
    if (Math.abs(cell.offsetHeight - need) > 2) cell.style.minHeight = need + 'px';

    const { ctx, w } = fit(this.canvas);
    ctx.clearRect(0, 0, w, cell.offsetHeight);
    label(ctx, 'Rehearsal · one fixed patient per starting rhythm', pad, 14);
    ctx.font = '500 10px ui-monospace, monospace';
    ctx.fillStyle = MUTED;
    if (twoLine) { ctx.fillText(legend[0], pad, 30); ctx.fillText(legend[1], pad, 43); }
    else ctx.fillText(legend.join(' · '), pad, 30);

    let y = top;
    for (const p of this.patients) {
      ctx.font = '700 11px ui-monospace, monospace';
      ctx.fillStyle = INK;
      ctx.fillText(p.rhythm, pad, y + 24);

      const x0 = pad + labW;
      p.steps.forEach((s, i) => {
        const x = x0 + (i % perLine) * (cellW + gap);
        const yl = y + Math.floor(i / perLine) * lineH;
        if (s.turned) {
          ctx.font = '600 9px ui-monospace, monospace';
          ctx.fillStyle = INK;
          ctx.fillText(`→ ${s.turned}`, x, yl + 8);
        }
        ctx.fillStyle = s.correct ? '#1e2a36' : 'rgba(30,42,54,0.35)';
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x, yl + 11, cellW, 20, 3); else ctx.rect(x, yl + 11, cellW, 20);
        ctx.fill();
        if (s.correct) { ctx.strokeStyle = 'rgba(220,229,238,0.35)'; ctx.lineWidth = 1; ctx.stroke(); }
        const name = SHORT[this.actions[s.action]];
        ctx.font = '600 10px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = s.correct ? INK : MUTED;
        ctx.fillText(name, x + cellW / 2, yl + 25);
        if (!s.correct) {
          const tw = ctx.measureText(name).width;
          ctx.strokeStyle = MUTED; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(x + cellW / 2 - tw / 2 - 2, yl + 21.5);
          ctx.lineTo(x + cellW / 2 + tw / 2 + 2, yl + 21.5); ctx.stroke();
          ctx.font = '500 9px ui-monospace, monospace';
          ctx.fillStyle = '#8fa3b6';
          let wanted = s.wanted.map((a) => SHORT[this.actions[a]]).join('/');
          while (wanted.length > 3 && ctx.measureText(wanted).width > cellW) wanted = wanted.slice(0, -2) + '…';
          ctx.fillText(wanted, x + cellW / 2, yl + 41);
        }
        ctx.textAlign = 'left';
      });
      if (p.stopped || p.seed === null) {
        const i = p.steps.length;
        ctx.font = 'italic 500 10px ui-monospace, monospace';
        ctx.fillStyle = MUTED;
        ctx.fillText(p.seed === null ? 'no patient found' : p.stopped,
          x0 + (i % perLine) * (cellW + gap), y + Math.floor(i / perLine) * lineH + 25);
      }

      const right = p.steps.filter((s) => s.correct).length;
      ctx.font = '600 10px ui-monospace, monospace';
      ctx.fillStyle = MUTED;
      ctx.textAlign = 'right';
      ctx.fillText(p.steps.length ? `✓ ${right} / ${p.steps.length}` : '—', w - pad, y + 24);
      ctx.textAlign = 'left';
      y += lines * lineH + 10;
    }
  }
}
