// The talk, part A (slides 1-4): resistor network, harmonic embedding, the rule.
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

// A small fixed network: 3 inputs (B0..B2) on the rim, 10 interior neurons.
const BND = [[320, 34], [560, 350], [80, 350]];
const INT = [[320, 110], [245, 165], [395, 165], [320, 220], [205, 245], [435, 245],
  [160, 310], [480, 310], [275, 300], [365, 300]];
// Edges: [a, b, w]; interior indices 0..9, inputs as 'B0'..'B2'.
const EDGES = [['B0', 0, 1.6], [0, 1, 1], [0, 2, 1], [1, 2, 0.6], [1, 3, 1], [2, 3, 1],
  [1, 4, 1.2], [2, 5, 0.8], [3, 8, 1], [3, 9, 1], [4, 8, 0.7], [5, 9, 1.1], [4, 6, 1],
  [5, 7, 1], [6, 'B2', 1.6], [7, 'B1', 1.6], [8, 9, 0.8], [6, 8, 1], [7, 9, 1], ['B0', 1, 0.5],
  [4, 'B2', 0.6], [5, 'B1', 0.6]];
const pos = (k) => (typeof k === 'string' ? BND[+k[1]] : INT[k]);

// Solve L_II H = A_IB (A_IB: conductances from interior to inputs) by Gaussian elimination.
function harmonic() {
  const n = INT.length, m = BND.length;
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

function drawEdges(svg, opacity = 0.45) {
  const g = el('g', {}, svg);
  return EDGES.map(([a, b, w]) => {
    const [x1, y1] = pos(a), [x2, y2] = pos(b);
    return el('line', { x1, y1, x2, y2, stroke: 'var(--c-edge, var(--mute))', 'stroke-width': 1.5 + 2.5 * w, 'stroke-linecap': 'round', opacity }, g);
  });
}
function drawInputs(svg) {
  return BND.map(([x, y], b) => {
    const r = el('rect', { x: x - 17, y: y - 17, width: 34, height: 34, rx: 5, fill: INCOL[b], stroke: INCOL[b], 'stroke-width': 3 }, svg);
    txt(svg, x, y + (b === 0 ? -26 : 38), `input ${b + 1}`, { 'text-anchor': 'middle', fill: 'var(--mute)' });
    return r;
  });
}

function modelSlide(root, ctx, H) {
  const svg = root.querySelector('#talk-a-model-svg');
  const btn = root.querySelector('#talk-a-model-play');
  if (!svg) return;
  const lines = drawEdges(svg, 0.5);
  const sq = drawInputs(svg);
  const circ = INT.map(([x, y]) => el('circle', { cx: x, cy: y, r: 15, stroke: 'var(--fg)', 'stroke-width': 1.2, fill: 'var(--bg)' }, svg));
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

function embedSlide(root, H) {
  const svg = root.querySelector('#talk-a-embed-svg');
  if (!svg) return;
  drawEdges(svg, 0.3);
  drawInputs(svg);
  const nodes = el('g', {}, svg);
  const panel = el('g', {}, svg);
  let ring = null;
  const select = (i) => {
    panel.replaceChildren();
    bars(panel, 470, 66, H[i], 70, `neuron ${i + 1}`);
    ring?.remove();
    ring = el('circle', { cx: INT[i][0], cy: INT[i][1], r: 22, fill: 'none', stroke: 'var(--fg)', 'stroke-width': 2.5 }, svg);
  };
  INT.forEach(([x, y], i) => {
    const g = el('g', { style: 'cursor:pointer' }, nodes);
    pie(g, x, y, 17, H[i]);
    el('circle', { cx: x, cy: y, r: 22, fill: 'transparent' }, g);
    g.addEventListener('click', () => select(i));
    g.addEventListener('mouseenter', () => select(i));
  });
  select(4);
}

function ruleSlide(root) {
  const svg = root.querySelector('#talk-a-rule-svg');
  const slider = root.querySelector('#talk-a-rule-slider');
  if (!svg || !slider) return;
  const b = 0.15, alpha = 1;
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
    const w = Math.max(0, b - D2 / 2) / alpha;
    syn.replaceChildren(); dyn.replaceChildren();
    const [x1, y1] = P[0], [x2, y2] = P[1];
    if (w > 0) {
      el('line', { x1: x1 + 30, y1, x2: x2 - 30, y2, stroke: 'var(--fg)', 'stroke-width': 2 + 26 * (w / b), 'stroke-linecap': 'round' }, syn);
      txt(dyn, 320, y1 - 28, 'wired', { 'text-anchor': 'middle', 'font-weight': 600 });
    } else {
      el('line', { x1: x1 + 30, y1, x2: 290, y2, stroke: 'var(--mute)', 'stroke-width': 2, 'stroke-dasharray': '6 6' }, syn);
      el('line', { x1: 350, y1, x2: x2 - 30, y2, stroke: 'var(--mute)', 'stroke-width': 2, 'stroke-dasharray': '6 6' }, syn);
      txt(dyn, 320, y1 + 5, '✕', { 'text-anchor': 'middle', 'font-size': 22, fill: 'var(--accent, #c0502a)' });
      txt(dyn, 320, y1 - 28, 'disconnected', { 'text-anchor': 'middle', 'font-weight': 600, fill: 'var(--accent, #c0502a)' });
    }
    pie(dyn, x1, y1, 22, hi);
    pie(dyn, x2, y2, 22, hj);
    bars(dyn, 40, 200, hi, 90, null);
    bars(dyn, 380, 200, hj, 90, null);
    txt(dyn, 320, 320, `D² = ${D2.toFixed(3)}   (cut-off 2b = ${(2 * b).toFixed(2)})`, { 'text-anchor': 'middle' });
    txt(dyn, 320, 346, `w = ${w.toFixed(3)}`, { 'text-anchor': 'middle', 'font-weight': 600 });
  };
  slider.addEventListener('input', update);
  update();
}

export function init(root, ctx) {
  const H = harmonic();
  modelSlide(root, ctx, H);
  embedSlide(root, H);
  ruleSlide(root);
}
