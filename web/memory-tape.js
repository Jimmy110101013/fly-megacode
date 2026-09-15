/**
 * Memory tape: what the central complex handed the mushroom body at every step of the
 * current patient.
 *
 * One row per readout cell. Each step is shaded by the level, 0 to 4, that actually
 * reached the Kenyon cells -- the one-hot channel acls-engine.js builds from the
 * readout, not the raw activity -- so the tape shows the fly's input, not a nicer
 * version of it. Marks above the rows say what was done at that step. The tape clears
 * with every new patient, because the reservoir does -- on that patient's first
 * decision, not when the case ends, or the most informative tape would never be seen.
 *
 * Expect the rows to climb together as the case goes on, several of them near copies
 * of each other; where shock history is held it shows as the smaller differences
 * between rows. Which eight cells are read is modelled (the reservoir's hub cells by
 * in-degree), not an anatomical output of the central complex.
 */
import { fit, label, MUTED } from './brainview.js';

const MAX_STEPS = 60;                 // the page ends a megacode at 60 decisions
const EVENTS = {
  shock: ['S', '#f0a93b'], epinephrine: ['E', '#dce5ee'],
  amiodarone: ['A', '#b39dff'], lidocaine: ['L', '#b39dff'], rhythm_check: ['·', '#7c8da0'],
};
const LOW = [20, 27, 35], HIGH = [140, 242, 255];      // the atlas's "holding state" cyan

export class MemoryTape {
  constructor(canvas, { readouts, levels }) {
    this.canvas = canvas;
    this.readouts = readouts;
    this.levels = levels;
    this.reset();
  }

  reset() { this.steps = []; }

  /** One decision: the readouts the Kenyon cells were given, and what was then done. */
  record(out, action, rhythm) {
    if (this.steps.length >= MAX_STEPS) return;
    const lv = Uint8Array.from({ length: this.readouts }, (_, i) => {
      // Same quantisation as makeFeatureSpace's encode, or the tape would lie.
      const x = Math.max(-1, Math.min(1, out[i] ?? 0));
      return Math.min(this.levels - 1, Math.floor(((x + 1) / 2) * this.levels));
    });
    this.steps.push({ lv, action, rhythm });
  }

  render() {
    const { ctx, w, h } = fit(this.canvas);
    ctx.clearRect(0, 0, w, h);
    const pad = 16, narrow = w < 640;
    label(ctx, narrow ? 'Memory tape · CX → KC, this patient'
      : 'Memory tape · what the central complex gave the mushroom body, this patient', pad, 14);
    ctx.font = '500 10px ui-monospace, monospace';
    ctx.fillStyle = MUTED;
    const l1 = 'rows: the 8 readout cells (which 8 is modelled) · shade: the level 0–4 the Kenyon cells receive';
    const l2 = 'S shock · E epinephrine · A/L antiarrhythmic · · rhythm check';
    if (narrow) { ctx.fillText('rows: 8 readout cells (choice modelled)', pad, 30); ctx.fillText('shade: level 0–4 received · S shock · E epi', pad, 43); }
    else { ctx.fillText(l1, pad, 30); ctx.fillText(l2, pad, 43); }

    const labW = narrow ? 22 : 30;
    const top = 66, bottom = h - 26;
    const rowH = (bottom - top) / this.readouts;
    const colW = (w - pad * 2 - labW) / MAX_STEPS;
    const x0 = pad + labW;

    ctx.font = '500 9px ui-monospace, monospace';
    for (let r = 0; r < this.readouts; r++) {
      ctx.fillStyle = MUTED;
      ctx.fillText(`r${r + 1}`, pad, top + r * rowH + rowH * 0.68);
      ctx.fillStyle = 'rgba(20,27,35,0.6)';
      ctx.fillRect(x0, top + r * rowH + 1, colW * MAX_STEPS, rowH - 2);
    }

    ctx.font = '500 9px ui-monospace, monospace';
    const counter = `${this.steps.length} / ${MAX_STEPS} decisions`;
    const counterX = w - pad - ctx.measureText(counter).width - 6;
    let prevRhythm = null, labelEnd = -Infinity;
    this.steps.forEach((s, i) => {
      const x = x0 + i * colW;
      for (let r = 0; r < this.readouts; r++) {
        const t = s.lv[r] / (this.levels - 1);
        ctx.fillStyle = `rgb(${LOW.map((v, k) => Math.round(v + (HIGH[k] - v) * t)).join(',')})`;
        ctx.fillRect(x + 0.5, top + r * rowH + 1, Math.max(1, colW - 1), rowH - 2);
      }
      const ev = EVENTS[s.action];
      if (ev) {
        ctx.fillStyle = ev[1];
        ctx.textAlign = 'center';
        ctx.font = '700 9px ui-monospace, monospace';
        ctx.fillText(ev[0], x + colW / 2, top - 5);
        ctx.textAlign = 'left';
        if (s.action === 'shock' || s.action === 'epinephrine') {
          ctx.globalAlpha = 0.35;
          ctx.fillRect(x + colW / 2 - 0.5, top, 1, bottom - top);
          ctx.globalAlpha = 1;
        }
      }
      if (s.rhythm !== prevRhythm) {
        ctx.fillStyle = MUTED;
        ctx.font = '500 9px ui-monospace, monospace';
        // A rhythm that changes back within a few narrow columns would print on top of
        // the last label; the tick still marks the change, the name is skipped.
        if (x >= labelEnd && x + ctx.measureText(s.rhythm).width <= counterX) {
          ctx.fillText(s.rhythm, x, bottom + 13);
          labelEnd = x + ctx.measureText(s.rhythm).width + 4;
        }
        ctx.fillRect(x, bottom + 1, 1, 3);
        prevRhythm = s.rhythm;
      }
    });

    ctx.fillStyle = MUTED;
    ctx.font = '500 9px ui-monospace, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(counter, w - pad, bottom + 13);
    ctx.textAlign = 'left';
  }
}
