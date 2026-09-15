/**
 * The fly's brain, drawn from the activity trace the model already produces.
 * Four panels: the Kenyon cell field, the synapses on the current decision's path,
 * the MBON vote, and the learning curve. Nothing here computes anything -- it reads
 * `fly.last`, `fly.w`, `fly.w0` and the run history.
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
  constructor({ kcCanvas, mbonCanvas, curveCanvas, synCanvas = null }, fly) {
    this.kcCanvas = kcCanvas;
    this.mbonCanvas = mbonCanvas;
    this.curveCanvas = curveCanvas;
    this.synCanvas = synCanvas;
    this.fly = fly;
    this.heat = new Float32Array(fly.nKC);      // decaying trace per Kenyon cell
    this.danFlash = 0;
    this.danKind = 'PAM';
    this.cols = Math.ceil(Math.sqrt(fly.nKC * 2.1));

    // Synapse panel: MBONs sit on one row per polarity, ordered by type so the
    // two hemispheres' copies of a compartment land next to each other.
    const order = (sign) => [...Array(fly.nMBON).keys()]
      .filter((j) => (fly.mbonSign[j] > 0) === sign)
      .sort((a, b) => String(fly.circuit.mbon[a].type).localeCompare(String(fly.circuit.mbon[b].type)) || a - b);
    this.approachRow = order(true);
    this.avoidRow = order(false);
    this.mbonSlot = new Float32Array(fly.nMBON);  // position along its row, 0..1
    const place = (row) => row.forEach((j, i) => { this.mbonSlot[j] = (i + 0.5) / row.length; });
    place(this.approachRow); place(this.avoidRow);
    this.syn = null;                              // what the last graded decision did
  }

  /**
   * Snapshot the path's weights before the verdict lands, so the panel can show
   * where this one dose of dopamine went. Must be called before `fly.learn`.
   */
  before(activeKC) {
    const { rowStart, w } = this.fly;
    let n = 0;
    for (const k of activeKC) n += rowStart[k + 1] - rowStart[k];
    const wBefore = new Float32Array(n);
    let i = 0;
    for (const k of activeKC) for (let p = rowStart[k]; p < rowStart[k + 1]; p++) wBefore[i++] = w[p];
    this.syn = { trace: activeKC, wBefore, delta: null, kind: null, valueBefore: this._pathValue(activeKC, wBefore) };
  }

  /** Called when the fly commits to an action and gets graded. */
  pulse(activeKC, correct) {
    for (const k of activeKC) this.heat[k] = 1;
    this.danFlash = 1;
    this.danKind = correct ? 'PAM' : 'PPL1';

    const s = this.syn;
    if (s && s.trace === activeKC) {
      const { rowStart, w } = this.fly;
      // Relative depression per synapse, rescaled so this decision's largest is 1.
      // One event moves a weight by about 1%, which no line width can show; the
      // flash shows *where* it went, the width shows what has accumulated.
      s.delta = new Float32Array(s.wBefore.length);
      let i = 0, mx = 1e-9, touched = 0;
      for (const k of activeKC) for (let p = rowStart[k]; p < rowStart[k + 1]; p++, i++) {
        const d = Math.max(0, (s.wBefore[i] - w[p]) / s.wBefore[i]);
        s.delta[i] = d; if (d > mx) mx = d; if (d > 1e-4) touched++;
      }
      for (let j = 0; j < s.delta.length; j++) s.delta[j] /= mx;
      s.kind = this.danKind;
      s.touched = touched;
      s.valueAfter = this._pathValue(activeKC, null);
    }
  }

  /** The quantity the decision reads: Σ sign · (w − w0), per Kenyon cell. */
  _pathValue(activeKC, snapshot) {
    const { rowStart, colIdx, w, w0, mbonSign } = this.fly;
    let v = 0, i = 0;
    for (const k of activeKC) for (let p = rowStart[k]; p < rowStart[k + 1]; p++, i++) {
      v += mbonSign[colIdx[p]] * ((snapshot ? snapshot[i] : w[p]) - w0[p]);
    }
    return v / Math.max(1, activeKC.length);
  }

  decay(dt) {
    const f = Math.exp(-dt * 1.5);
    for (let i = 0; i < this.heat.length; i++) this.heat[i] *= f;
    this.danFlash = Math.max(0, this.danFlash - dt * 1.1);
  }

  render(history) {
    if (this.kcCanvas) this._kc();
    if (this.synCanvas) this._syn();
    this._mbon();
    this._curve(history);
  }

  /*
   * Every KC->MBON synapse on the path the fly just took, drawn as a line whose
   * width is w / w0: full at the anatomical weight, thinning as dopamine depresses
   * it. The atlas does the opposite and never lets thickness carry state, because
   * there a line is traced neurite and thinning it would read as the anatomy
   * changing. Here a line is a synaptic weight, which is exactly what learning
   * changes. Do not unify the two conventions.
   *
   * Lines are coloured by the target compartment's *dominant* dopamine input, not
   * by which dopamine just arrived. Most compartments receive some of both, so a
   * PPL1 dose thins avoidance lines too, only less. Colouring by the arriving arm
   * would make the legend look contradicted on a third of the lines.
   *
   * There is no potentiation anywhere in the model. A correct answer looks like
   * strengthening only because it thins the avoidance side of the path.
   */
  _syn() {
    const { ctx, w, h } = fit(this.synCanvas);
    ctx.clearRect(0, 0, w, h);
    const fly = this.fly;
    const trace = fly.last?.activeKC;
    const pad = 16, top = 58, bottom = h - 30;
    const yA = top, yV = bottom, yK = (top + bottom) / 2;
    const span = w - pad * 2;
    label(ctx, 'Synapses on this path · KC→MBON', pad, 14);
    if (!trace || !trace.length) return;

    const kcs = [...trace].sort((a, b) => a - b);
    const kcX = new Map(kcs.map((k, i) => [k, pad + ((i + 0.5) / kcs.length) * span]));
    const mx = (j) => pad + this.mbonSlot[j] * span;
    const my = (j) => (fly.mbonSign[j] > 0 ? yA : yV);

    // Bucket by polarity and width so ~1,300 lines cost a couple of dozen strokes.
    const LEVELS = 8;
    const buckets = Array.from({ length: LEVELS * 2 }, () => []);
    const s = this.syn;
    const flashOn = s && s.trace === trace && s.delta && this.danFlash > 0.02;
    const glow = Array.from({ length: 5 }, () => []);
    let sumA = 0, nA = 0, sumV = 0, nV = 0, i = 0;
    for (const k of trace) {
      const x0 = kcX.get(k);
      for (let p = fly.rowStart[k]; p < fly.rowStart[k + 1]; p++, i++) {
        const j = fly.colIdx[p];
        const r = Math.min(1, Math.max(0, fly.w[p] / fly.w0[p]));
        const approach = fly.mbonSign[j] > 0;
        if (approach) { sumA += r; nA++; } else { sumV += r; nV++; }
        const lvl = Math.min(LEVELS - 1, Math.floor(r * LEVELS));
        const seg = [x0, yK, mx(j), my(j)];
        buckets[(approach ? 0 : LEVELS) + lvl].push(seg);
        if (flashOn && s.delta[i] > 0.15) glow[Math.min(4, Math.floor(s.delta[i] * 5))].push(seg);
      }
    }

    ctx.lineCap = 'round';
    buckets.forEach((segs, b) => {
      if (!segs.length) return;
      const approach = b < LEVELS;
      const r = ((b % LEVELS) + 0.5) / LEVELS;
      ctx.strokeStyle = approach ? PPL1 : PAM;
      ctx.globalAlpha = 0.05 + 0.4 * r * r;
      ctx.lineWidth = 0.25 + 2.4 * r;
      ctx.beginPath();
      for (const [x0, y0, x1, y1] of segs) { ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); }
      ctx.stroke();
    });
    // Where this decision's dopamine landed, brightest where it depressed most.
    if (flashOn) {
      ctx.strokeStyle = '#ffffff';
      glow.forEach((segs, g) => {
        if (!segs.length) return;
        ctx.globalAlpha = this.danFlash * (0.08 + 0.14 * g);
        ctx.lineWidth = 0.6 + 0.5 * g;
        ctx.beginPath();
        for (const [x0, y0, x1, y1] of segs) { ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); }
        ctx.stroke();
      });
    }
    ctx.globalAlpha = 1;

    // Nodes.
    ctx.fillStyle = AMBER;
    for (const x of kcX.values()) { ctx.beginPath(); ctx.arc(x, yK, 2.2, 0, 6.283); ctx.fill(); }
    for (const [row, y, col] of [[this.approachRow, yA, PPL1], [this.avoidRow, yV, PAM]]) {
      ctx.fillStyle = col;
      for (const j of row) { ctx.beginPath(); ctx.arc(mx(j), y, 2.6, 0, 6.283); ctx.fill(); }
    }

    // Row captions and the running balance.
    ctx.font = '600 9px ui-sans-serif, system-ui, sans-serif';
    ctx.fillStyle = PPL1;
    ctx.fillText(`approach MBONs · PPL1-dominant · mean w/w₀ ${(sumA / Math.max(1, nA)).toFixed(2)}`, pad, yA - 10);
    ctx.fillStyle = PAM;
    ctx.fillText(`avoidance MBONs · PAM-dominant · mean w/w₀ ${(sumV / Math.max(1, nV)).toFixed(2)}`, pad, yV + 18);

    // Second line, not uppercased: the label style would turn w/w0 into W/W0.
    ctx.font = '700 10px ui-monospace, monospace';
    if (flashOn) {
      ctx.fillStyle = s.kind === 'PAM' ? PAM : PPL1;
      ctx.fillText(`${s.kind} · ${s.touched.toLocaleString()} depressed · value ${s.valueBefore.toFixed(3)} → ${s.valueAfter.toFixed(3)}`, pad, 30);
    } else {
      ctx.fillStyle = MUTED;
      ctx.fillText(`width = w / w₀ · ${kcs.length} KC · ${i.toLocaleString()} synapses`, pad, 30);
    }
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
    label(ctx, `Kenyon cells · ${this.fly.nKC} · ${this.fly.kActive} firing`, pad, 14);
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

    label(ctx, `MBON vote · ${n} output neurons`, pad, 14);
    ctx.font = '600 9px ui-sans-serif, system-ui, sans-serif';
    ctx.fillStyle = PPL1; ctx.fillText('approach · PPL1 compartments', pad, top - 4);
    ctx.fillStyle = PAM;
    ctx.fillText('avoidance · PAM compartments', pad, bottom + 14);
    if (this.danFlash > 0.02) {
      ctx.fillStyle = this.danKind === 'PAM' ? PAM : PPL1;
      ctx.font = '700 10px ui-monospace, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${this.danKind} dopamine`, w - pad, top - 4);
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

    // Chance: a uniform random guesser under the real grader, measured by
    // pipeline/chance.mjs. It was once a hand-typed 16%, which no definition gives.
    const chance = 0.125;
    ctx.strokeStyle = MUTED;
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(pad, y(chance)); ctx.lineTo(w - pad, y(chance)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = MUTED;
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillText('chance 12.5%', pad + 2, y(chance) - 4);

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
    label(ctx, 'accuracy · last 60 decisions', pad, 14);
  }
}
