/**
 * The fly's brain, drawn from the activity trace the model already produces.
 * Three panels: the Kenyon cell field, the MBON vote, and the learning curve.
 * Nothing here computes anything -- it reads `fly.last` and the run history.
 */

const INK = '#dce5ee';
const MUTED = '#7c8da0';
const AMBER = '#f0a93b';
const PAM = '#3bb0f0';
const PPL1 = '#ff4d6d';
const LINE = '#1c2530';

function fit(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio, 2);
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

const label = (ctx, text, x, y) => {
  ctx.fillStyle = MUTED;
  ctx.font = '600 10px ui-sans-serif, system-ui, sans-serif';
  ctx.letterSpacing = '0.09em';
  ctx.fillText(text.toUpperCase(), x, y);
  ctx.letterSpacing = '0em';
};

export class BrainView {
  constructor({ kcCanvas, mbonCanvas, curveCanvas }, fly) {
    this.kcCanvas = kcCanvas;
    this.mbonCanvas = mbonCanvas;
    this.curveCanvas = curveCanvas;
    this.fly = fly;
    this.heat = new Float32Array(fly.nKC);      // decaying trace per Kenyon cell
    this.danFlash = 0;
    this.danKind = 'PAM';
    this.cols = Math.ceil(Math.sqrt(fly.nKC * 2.1));
  }

  /** Called when the fly commits to an action and gets graded. */
  pulse(activeKC, correct) {
    for (const k of activeKC) this.heat[k] = 1;
    this.danFlash = 1;
    this.danKind = correct ? 'PAM' : 'PPL1';
  }

  decay(dt) {
    const f = Math.exp(-dt * 1.5);
    for (let i = 0; i < this.heat.length; i++) this.heat[i] *= f;
    this.danFlash = Math.max(0, this.danFlash - dt * 1.1);
  }

  render(history) {
    if (this.kcCanvas) this._kc();
    this._mbon();
    this._curve(history);
  }

  /* 2,597 Kenyon cells. Roughly 130 of them carry any given decision. */
  _kc() {
    const { ctx, w, h } = fit(this.kcCanvas);
    ctx.clearRect(0, 0, w, h);
    const pad = 16, top = 26;
    const cols = this.cols;
    const rows = Math.ceil(this.fly.nKC / cols);
    const cw = (w - pad * 2) / cols;
    const ch = (h - pad - top) / rows;
    const r = Math.max(0.8, Math.min(cw, ch) * 0.36);

    for (let i = 0; i < this.fly.nKC; i++) {
      const x = pad + (i % cols) * cw + cw / 2;
      const y = top + Math.floor(i / cols) * ch + ch / 2;
      const a = this.heat[i];
      if (a > 0.02) {
        ctx.globalAlpha = Math.min(1, a);
        ctx.fillStyle = AMBER;
        ctx.beginPath(); ctx.arc(x, y, r * (1 + a * 1.5), 0, 6.283); ctx.fill();
        ctx.globalAlpha = Math.min(0.35, a * 0.35);
        ctx.beginPath(); ctx.arc(x, y, r * (2.5 + a * 3), 0, 6.283); ctx.fill();
      } else {
        ctx.globalAlpha = 0.16;
        ctx.fillStyle = '#35414f';
        ctx.beginPath(); ctx.arc(x, y, r * 0.75, 0, 6.283); ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    label(ctx, `Kenyon cells · ${this.fly.nKC} 個 · ${this.fly.kActive} 個放電`, pad, 14);
  }

  /* The MBON vote. Compartment owner sets behavioural polarity, not transmitter. */
  _mbon() {
    const { ctx, w, h } = fit(this.mbonCanvas);
    ctx.clearRect(0, 0, w, h);
    const pad = 16, top = 30, bottom = h - 22;
    const act = this.fly.last?.mbon;
    const n = this.fly.nMBON;
    const bw = (w - pad * 2) / n;
    let max = 1e-6;
    if (act) for (let i = 0; i < n; i++) max = Math.max(max, act[i]);

    const mid = (top + bottom) / 2;
    ctx.strokeStyle = LINE;
    ctx.beginPath(); ctx.moveTo(pad, mid); ctx.lineTo(w - pad, mid); ctx.stroke();

    for (let j = 0; j < n; j++) {
      const v = act ? act[j] / max : 0;
      const approach = this.fly.mbonSign[j] > 0;
      const len = v * (bottom - top) * 0.46;
      const x = pad + j * bw;
      ctx.fillStyle = approach ? PPL1 : PAM;
      ctx.globalAlpha = 0.25 + v * 0.75;
      if (approach) ctx.fillRect(x + bw * 0.12, mid - len, bw * 0.76, len);
      else ctx.fillRect(x + bw * 0.12, mid, bw * 0.76, len);
    }
    ctx.globalAlpha = 1;

    // Dopamine wash over the population the verdict recruited.
    if (this.danFlash > 0) {
      ctx.globalAlpha = this.danFlash * 0.16;
      ctx.fillStyle = this.danKind === 'PAM' ? PAM : PPL1;
      ctx.fillRect(pad, top, w - pad * 2, bottom - top);
      ctx.globalAlpha = 1;
    }

    label(ctx, `MBON 投票 · ${n} 個輸出神經元`, pad, 14);
    ctx.font = '600 9px ui-sans-serif, system-ui, sans-serif';
    ctx.fillStyle = PPL1; ctx.fillText('趨近 · PPL1 區間', pad, top - 4);
    ctx.fillStyle = PAM;
    ctx.fillText('迴避 · PAM 區間', pad, bottom + 14);
    if (this.danFlash > 0.02) {
      ctx.fillStyle = this.danKind === 'PAM' ? PAM : PPL1;
      ctx.font = '700 10px ui-monospace, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${this.danKind} 多巴胺`, w - pad, top - 4);
      ctx.textAlign = 'left';
    }
  }

  /* Rolling accuracy against the chance line. */
  _curve(history) {
    const { ctx, w, h } = fit(this.curveCanvas);
    ctx.clearRect(0, 0, w, h);
    const pad = 16, top = 26, bottom = h - 18;
    const plotH = bottom - top;
    const y = (frac) => bottom - frac * plotH;

    ctx.strokeStyle = LINE;
    ctx.lineWidth = 1;
    for (const f of [0, 0.25, 0.5, 0.75, 1]) {
      ctx.beginPath(); ctx.moveTo(pad, y(f)); ctx.lineTo(w - pad, y(f)); ctx.stroke();
    }

    // Chance: the engine accepts ~1.6 of 10 actions, so blind guessing sits here.
    const chance = 0.159;
    ctx.strokeStyle = MUTED;
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(pad, y(chance)); ctx.lineTo(w - pad, y(chance)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = MUTED;
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillText('亂猜 16%', pad + 2, y(chance) - 4);

    if (history.length > 1) {
      const N = history.length;
      ctx.strokeStyle = AMBER;
      ctx.lineWidth = 2;
      ctx.beginPath();
      history.forEach((v, i) => {
        const px = pad + (i / (N - 1)) * (w - pad * 2);
        i ? ctx.lineTo(px, y(v)) : ctx.moveTo(px, y(v));
      });
      ctx.stroke();

      const last = history[N - 1];
      ctx.fillStyle = AMBER;
      ctx.beginPath(); ctx.arc(w - pad, y(last), 3.5, 0, 6.283); ctx.fill();
      ctx.font = '700 12px ui-monospace, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${(last * 100).toFixed(0)}%`, w - pad - 8, y(last) - 8);
      ctx.textAlign = 'left';
    }
    label(ctx, '近 60 次決策正確率', pad, 14);
  }
}
