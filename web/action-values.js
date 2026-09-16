/**
 * Why it chose that: the score the fly gave every action on the last decision.
 *
 * The mushroom body scores all ten candidates and commits to one, and until now the page
 * showed only the commitment. The scores are the fly's actual reasoning -- what it
 * believed about each option at that moment -- and they are already computed every
 * decision, in fly.last.values. This draws them.
 *
 * A bar per action, longest first. The bar the fly took is filled; the actions the
 * grader would have accepted carry a tick. When those disagree you can see by how much:
 * a near-tie that the softmax lost, or a confident mistake. The gap between first and
 * second is drawn as a number because it is what decides whether sampling departs from
 * the top pick at all -- at the annealed temperature of 0.06 the two coincide 88.5% of
 * the time, and nearly every departure happens where this gap is small.
 *
 * Values are the readout's own scale, not probabilities, so the axis is relative: the
 * longest bar is full width and everything else is drawn against it.
 */
import { fit, label, INK, MUTED, PAM, PPL1 } from './brainview.js';

const OK = '#43e08a';                 // the grader's colour elsewhere on the page

export class ActionValues {
  /** @param actions  the ACTIONS array, so labels match the rest of the page */
  constructor(canvas, { actions, pretty = {} }) {
    this.canvas = canvas;
    this.actions = actions;
    this.pretty = pretty;
    this.last = null;
  }

  reset() { this.last = null; }

  /**
   * One decision.
   * @param values    fly.last.values, one score per action
   * @param chosen    the action index taken
   * @param accepted  indices the grader would have accepted
   * @param correct   whether the taken action was one of them
   */
  record(values, chosen, accepted, correct) {
    this.last = {
      values: Array.from(values), chosen,
      accepted: new Set(accepted), correct,
    };
  }

  render() {
    const { ctx, w, h } = fit(this.canvas);
    ctx.clearRect(0, 0, w, h);
    const pad = 16;
    label(ctx, 'Why it chose that · the fly’s score for every action', pad, 14);

    if (!this.last) {
      ctx.font = '500 10px ui-monospace, monospace';
      ctx.fillStyle = MUTED;
      ctx.fillText('press Start, or Step, to see a decision', pad, 34);
      return;
    }

    const { values, chosen, accepted, correct } = this.last;
    const order = values.map((v, i) => i).sort((a, b) => values[b] - values[a]);
    const best = values[order[0]], second = values[order[1]] ?? best;
    const gap = best - second;

    ctx.font = '500 10px ui-monospace, monospace';
    ctx.fillStyle = MUTED;
    ctx.fillText('filled = taken · ✓ = the guideline would accept it · ' +
                 `lead over the runner-up ${gap.toFixed(3)}`, pad, 30);

    // Bars are relative to the widest score, since the readout has no natural unit.
    const labW = 108, tickW = 16;
    const top = 44, bottom = h - 10;
    const rowH = Math.min(22, (bottom - top) / this.actions.length);
    const x0 = pad + labW + tickW;
    const barW = w - pad - x0;
    const lo = Math.min(...values), hi = Math.max(...values);
    const span = hi - lo || 1;

    order.forEach((a, row) => {
      const y = top + row * rowH;
      const taken = a === chosen;
      const wanted = accepted.has(a);
      const colour = taken ? (correct ? PAM : PPL1) : (wanted ? OK : MUTED);

      ctx.font = `${taken ? '700' : '500'} 10px ui-monospace, monospace`;
      ctx.fillStyle = taken ? INK : MUTED;
      ctx.fillText(this.pretty[this.actions[a]] ?? this.actions[a], pad, y + rowH * 0.7);

      if (wanted) {
        ctx.fillStyle = OK;
        ctx.fillText('✓', pad + labW, y + rowH * 0.7);
      }

      const len = Math.max(1, ((values[a] - lo) / span) * barW);
      ctx.fillStyle = colour;
      ctx.globalAlpha = taken ? 1 : 0.35;
      ctx.fillRect(x0, y + 3, len, rowH - 8);
      ctx.globalAlpha = 1;
      // The outline keeps a losing-but-correct action visible when its bar is tiny.
      if (wanted && !taken) {
        ctx.strokeStyle = OK;
        ctx.lineWidth = 1;
        ctx.strokeRect(x0 + 0.5, y + 3.5, Math.max(len, barW) - 1, rowH - 9);
      }

      ctx.fillStyle = taken ? INK : MUTED;
      ctx.font = '500 9px ui-monospace, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(values[a].toFixed(3), w - pad - 2, y + rowH * 0.7);
      ctx.textAlign = 'left';
    });
  }
}
