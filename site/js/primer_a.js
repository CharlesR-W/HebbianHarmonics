// The primer, part A (slides 1-4): title, resistor network, harmonic embedding (neuron space and input space,
// linked), the covariance rule on two neurons. twoNeurons() is shared with part B (the distance rule).
import { randomLayout } from './netview.js';
const NS = 'http://www.w3.org/2000/svg';
const FONT = 'var(--font-sans, Inter, sans-serif)';
const INCOL = ['var(--c-cat-0)', 'var(--c-cat-1)', 'var(--c-cat-2)'];

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function txt(parent, x, y, s, attrs = {}) {
  const t = el('text', Object.assign({ x, y, 'font-size': 14, fill: 'var(--fg)', style: `font-family:${FONT}` }, attrs), parent);
  t.textContent = s;
  return t;
}

// A small fixed network: 3 inputs (B0..B2) and 10 interior neurons. The topology is fixed; positions are seeded
// random (netview.randomLayout), so the drawing is in neuron space and its layout says nothing about function.
const NI = 10, NB = 3;
// Edges: [a, b, w]; interior indices 0..9, inputs as 'B0'..'B2'.
const EDGES = [['B0', 0, 1.6], [0, 1, 1], [0, 2, 1], [1, 2, 0.6], [1, 3, 1], [2, 3, 1],
  [1, 4, 1.2], [2, 5, 0.8], [3, 8, 1], [3, 9, 1], [4, 8, 0.7], [5, 9, 1.1], [4, 6, 1],
  [5, 7, 1], [6, 'B2', 1.6], [7, 'B1', 1.6], [8, 9, 0.8], [6, 8, 1], [7, 9, 1], ['B0', 1, 0.5],
  [4, 'B2', 0.6], [5, 'B1', 0.6]];
// layout({x0, x1, y0, y1}, seed) -> {bnd: [[x, y]] x3, int: [[x, y]] x10} in viewBox units (inputs take the first slots).
function layout(box, seed) {
  const w = box.x1 - box.x0, h = box.y1 - box.y0;
  const r = randomLayout(NB + NI, { seed, aspect: w / h });
  const at = (k) => [box.x0 + r[2 * k] * w, box.y0 + r[2 * k + 1] * h];
  return { bnd: Array.from({ length: NB }, (_, b) => at(b)), int: Array.from({ length: NI }, (_, i) => at(NB + i)) };
}
const posOf = (L, k) => (typeof k === 'string' ? L.bnd[+k[1]] : L.int[k]);

// Solve L_II H = A_IB (A_IB: conductances from interior to inputs) by Gaussian elimination.
function harmonic() {
  const n = NI, m = NB;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  const R = Array.from({ length: n }, () => new Array(m).fill(0));
  for (const [a, b, w] of EDGES) {
    const ai = typeof a === 'number', bi = typeof b === 'number';
    if (ai) L[a][a] += w;
    if (bi) L[b][b] += w;
    if (ai && bi) { L[a][b] -= w; L[b][a] -= w; }
    else if (ai) R[a][+b[1]] += w;
    else if (bi) R[b][+a[1]] += w;
  }
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(L[r][c]) > Math.abs(L[p][c])) p = r;
    [L[c], L[p]] = [L[p], L[c]]; [R[c], R[p]] = [R[p], R[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = L[r][c] / L[c][c];
      if (!f) continue;
      for (let k = c; k < n; k++) L[r][k] -= f * L[c][k];
      for (let k = 0; k < m; k++) R[r][k] -= f * R[c][k];
    }
  }
  return R.map((row, i) => row.map((v) => v / L[i][i]));
}

function drawEdges(svg, L, opacity = 0.45) {
  const g = el('g', {}, svg);
  return EDGES.map(([a, b, w]) => {
    const [x1, y1] = posOf(L, a), [x2, y2] = posOf(L, b);
    return el('line', { x1, y1, x2, y2, stroke: 'var(--c-edge, var(--mute))', 'stroke-width': 4 * w, 'stroke-linecap': 'round', opacity }, g);
  });
}
function drawInputs(svg, L, mid) {
  return L.bnd.map(([x, y], b) => {
    const r = el('rect', { x: x - 17, y: y - 17, width: 34, height: 34, rx: 5, fill: INCOL[b], stroke: INCOL[b], 'stroke-width': 3 }, svg);
    const right = x < mid;
    txt(svg, x + (right ? 23 : -23), y + 5, `input ${b + 1}`, { 'text-anchor': right ? 'start' : 'end', fill: 'var(--mute)' });
    return r;
  });
}

// Boxes leave room for the badge (top left) and the input labels; the embed slide keeps the right side for its bars.
const MODEL_BOX = { x0: 40, x1: 600, y0: 62, y1: 372 }, MODEL_SEED = 3;
const EMBED_BOX = { x0: 40, x1: 520, y0: 56, y1: 376 }, EMBED_SEED = 3;

function modelSlide(root, ctx, H) {
  const svg = root.querySelector('#primer-a-model-svg');
  const btn = root.querySelector('#primer-a-model-play');
  if (!svg) return;
  const L = layout(MODEL_BOX, MODEL_SEED);
  const lines = drawEdges(svg, L, 0.5);
  const sq = drawInputs(svg, L, 320);
  const circ = L.int.map(([x, y]) => el('circle', { cx: x, cy: y, r: 15, stroke: 'var(--fg)', 'stroke-width': 1.2, fill: 'var(--bg)' }, svg));
  const col = (v) => {
    const p = Math.round(Math.min(1, Math.abs(v)) * 100);
    return `color-mix(in srgb, ${v >= 0 ? '#e0662a' : '#2b6cc4'} ${p}%, var(--bg, #fff))`;
  };
  let playing = true, t = 0, last = null;
  btn?.addEventListener('click', () => {
    playing = !playing;
    btn.setAttribute('aria-pressed', String(playing));
    btn.textContent = playing ? 'Playing' : 'Paused';
  });
  const frame = (now) => {
    requestAnimationFrame(frame);
    const dt = last == null ? 0 : Math.min(0.05, (now - last) / 1000);
    last = now;
    if (!ctx.visible || !playing) return;
    t += dt;
    const g = [Math.sin(1.1 * t), Math.sin(0.7 * t + 2.0), Math.sin(1.6 * t + 4.1)];
    const u = H.map((h) => h[0] * g[0] + h[1] * g[1] + h[2] * g[2]);
    g.forEach((v, b) => sq[b].setAttribute('fill', col(v)));
    u.forEach((v, i) => circ[i].setAttribute('fill', col(v)));
    const val = (k) => (typeof k === 'string' ? g[+k[1]] : u[k]);
    EDGES.forEach(([a, b], e) => lines[e].setAttribute('opacity', (0.25 + 0.75 * Math.min(1, Math.abs(val(a) - val(b)))).toFixed(3)));
  };
  frame(performance.now());
}

function pie(parent, cx, cy, r, fr) {
  let a0 = -Math.PI / 2;
  fr.forEach((f, b) => {
    if (f < 1e-4) return;
    const a1 = a0 + 2 * Math.PI * f;
    if (f > 0.9999) { el('circle', { cx, cy, r, fill: INCOL[b] }, parent); return; }
    const large = a1 - a0 > Math.PI ? 1 : 0;
    el('path', { d: `M${cx},${cy} L${cx + r * Math.cos(a0)},${cy + r * Math.sin(a0)} A${r},${r} 0 ${large} 1 ${cx + r * Math.cos(a1)},${cy + r * Math.sin(a1)} Z`, fill: INCOL[b] }, parent);
    a0 = a1;
  });
  el('circle', { cx, cy, r, fill: 'none', stroke: 'var(--fg)', 'stroke-width': 1 }, parent);
}

function bars(parent, x, y, h, w, sel) {
  // Horizontal bar chart of a mixture: one row per input.
  const g = el('g', {}, parent);
  h.forEach((f, b) => {
    const yy = y + b * 24;
    txt(g, x, yy + 14, `input ${b + 1}`, { fill: 'var(--mute)', 'font-size': 13 });
    el('rect', { x: x + 62, y: yy + 2, width: w, height: 16, fill: 'var(--line)', rx: 3 }, g);
    el('rect', { x: x + 62, y: yy + 2, width: Math.max(0, f) * w, height: 16, fill: INCOL[b], rx: 3 }, g);
    txt(g, x + 70 + w, yy + 15, f.toFixed(2), { 'font-size': 13 });
  });
  if (sel != null) txt(g, x, y - 10, sel, { 'font-weight': 600 });
  return g;
}

// Slide 3: the same ten neurons as pies at random positions (neuron space) and as points in the triangle (input
// space), with linked hover.
const SX = { A: [280, 64], B: [70, 366], C: [490, 366] };
function embedSlide(root, H) {
  const svg = root.querySelector('#primer-a-embed-svg'), sxs = root.querySelector('#primer-a-simplex-svg');
  if (!svg || !sxs) return;
  const L = layout(EMBED_BOX, EMBED_SEED);
  drawEdges(svg, L, 0.3);
  drawInputs(svg, L, 280);
  // the triangle
  const C = [SX.A, SX.B, SX.C];
  el('path', { d: `M${C[0]}L${C[1]}L${C[2]}Z`, fill: 'none', stroke: 'var(--line)', 'stroke-width': 1.5 }, sxs);
  C.forEach(([x, y], b) => {
    el('rect', { x: x - 11, y: y - 11, width: 22, height: 22, rx: 4, fill: INCOL[b] }, sxs);
    txt(sxs, b === 0 ? x + 20 : x, b === 0 ? y + 5 : y + 30, `input ${b + 1}`, { 'text-anchor': b === 0 ? 'start' : b === 1 ? 'start' : 'end', fill: 'var(--mute)' });
  });
  const at = (h) => [h[0] * C[0][0] + h[1] * C[1][0] + h[2] * C[2][0], h[0] * C[0][1] + h[1] * C[1][1] + h[2] * C[2][1]];
  const readout = txt(sxs, 552, 24, '', { 'text-anchor': 'end', 'font-weight': 600 });
  const nodes = el('g', {}, svg), dots = el('g', {}, sxs);
  const ringN = el('circle', { r: 23, fill: 'none', stroke: 'var(--fg)', 'stroke-width': 2.5, opacity: 0 }, svg);
  const ringS = el('circle', { r: 17, fill: 'none', stroke: 'var(--fg)', 'stroke-width': 2.5, opacity: 0 }, sxs);
  const select = (i) => {
    const on = i != null;
    ringN.setAttribute('opacity', on ? 1 : 0); ringS.setAttribute('opacity', on ? 1 : 0);
    if (!on) { readout.textContent = 'hover a neuron'; return; }
    ringN.setAttribute('cx', L.int[i][0]); ringN.setAttribute('cy', L.int[i][1]);
    const [x, y] = at(H[i]); ringS.setAttribute('cx', x); ringS.setAttribute('cy', y);
    readout.textContent = `neuron ${i + 1}: h = (${H[i].map((v) => v.toFixed(2)).join(', ')})`;
  };
  const bind = (g, i) => {
    g.addEventListener('pointerenter', () => select(i));
    g.addEventListener('click', () => select(i));
  };
  L.int.forEach(([x, y], i) => {
    const g = el('g', { style: 'cursor:pointer' }, nodes);
    pie(g, x, y, 17, H[i]);
    txt(g, x, y - 22, String(i + 1), { 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--mute)' });
    el('circle', { cx: x, cy: y, r: 22, fill: 'transparent' }, g);
    bind(g, i);
  });
  H.forEach((h, i) => {
    const [x, y] = at(h);
    const g = el('g', { style: 'cursor:pointer' }, dots);
    pie(g, x, y, 11, h);
    txt(g, x + 14, y + 4, String(i + 1), { 'font-size': 12, fill: 'var(--mute)' });
    el('circle', { cx: x, cy: y, r: 15, fill: 'transparent' }, g);
    bind(g, i);
  });
  select(4);
}

// Two neurons and the synapse between them (slides 4 and 7). mode "cov": the covariance rule's drive h_i'Σh_j - θ;
// mode "dist": the distance rule's fixed point w = (b - D²/2)_+/α. Σ = I.
export function twoNeurons(svg, slider, mode) {
  if (!svg || !slider) return;
  const b = 0.15, alpha = 1, theta = 0.3;
  const hi = [0.7, 0.2, 0.1], far = [0.1, 0.2, 0.7];
  const P = [[150, 130], [490, 130]];
  const syn = el('g', {}, svg);
  P.forEach(([x, y]) => el('circle', { cx: x, cy: y, r: 30, fill: 'var(--c-interior, var(--wash))', stroke: 'var(--fg)', 'stroke-width': 1.5 }, svg));
  txt(svg, P[0][0], P[0][1] - 44, 'neuron i', { 'text-anchor': 'middle', 'font-weight': 600 });
  txt(svg, P[1][0], P[1][1] - 44, 'neuron j', { 'text-anchor': 'middle', 'font-weight': 600 });
  const dyn = el('g', {}, svg);
  const update = () => {
    const s = slider.value / 100;
    const hj = hi.map((v, k) => s * v + (1 - s) * far[k]);
    const D2 = hi.reduce((acc, v, k) => acc + (v - hj[k]) ** 2, 0);
    const cov = hi.reduce((acc, v, k) => acc + v * hj[k], 0);
    const strength = mode === 'cov' ? Math.max(0, cov - theta) / 0.25 : Math.max(0, b - D2 / 2) / alpha / b;
    syn.replaceChildren(); dyn.replaceChildren();
    const [x1, y1] = P[0], [x2, y2] = P[1];
    if (strength > 0) {
      el('line', { x1: x1 + 30, y1, x2: x2 - 30, y2, stroke: 'var(--fg)', 'stroke-width': 2 + 24 * Math.min(1, strength), 'stroke-linecap': 'round' }, syn);
      txt(dyn, 320, y1 - 28, mode === 'cov' ? 'grows' : 'wired', { 'text-anchor': 'middle', 'font-weight': 600 });
    } else {
      el('line', { x1: x1 + 30, y1, x2: 290, y2, stroke: 'var(--mute)', 'stroke-width': 2, 'stroke-dasharray': '6 6' }, syn);
      el('line', { x1: 350, y1, x2: x2 - 30, y2, stroke: 'var(--mute)', 'stroke-width': 2, 'stroke-dasharray': '6 6' }, syn);
      txt(dyn, 320, y1 + 5, '✕', { 'text-anchor': 'middle', 'font-size': 22, fill: 'var(--accent, #c0502a)' });
      txt(dyn, 320, y1 - 28, mode === 'cov' ? 'decays to the floor' : 'disconnected', { 'text-anchor': 'middle', 'font-weight': 600, fill: 'var(--accent, #c0502a)' });
    }
    pie(dyn, x1, y1, 22, hi);
    pie(dyn, x2, y2, 22, hj);
    bars(dyn, 40, 200, hi, 90, null);
    bars(dyn, 380, 200, hj, 90, null);
    if (mode === 'cov') {
      txt(dyn, 320, 320, `covariance h_i·Σh_j = ${cov.toFixed(3)}   (threshold θ = ${theta})`, { 'text-anchor': 'middle' });
      txt(dyn, 320, 346, `drive = ${(cov - theta >= 0 ? '+' : '−') + Math.abs(cov - theta).toFixed(3)}`, { 'text-anchor': 'middle', 'font-weight': 600 });
    } else {
      const w = Math.max(0, b - D2 / 2) / alpha;
      txt(dyn, 320, 320, `D² = ${D2.toFixed(3)}   (cut-off 2b = ${(2 * b).toFixed(2)})`, { 'text-anchor': 'middle' });
      txt(dyn, 320, 346, `fixed point w = ${w.toFixed(3)}`, { 'text-anchor': 'middle', 'font-weight': 600 });
    }
  };
  slider.addEventListener('input', update);
  update();
}

export function init(root, ctx) {
  const H = harmonic();
  modelSlide(root, ctx, H);
  embedSlide(root, H);
  twoNeurons(root.querySelector('#primer-a-rule-svg'), root.querySelector('#primer-a-rule-slider'), 'cov');
}
