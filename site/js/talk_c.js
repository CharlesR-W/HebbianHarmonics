// Part C of "The talk" (slides 7-10): readout, knobs, allocation. Static SVG, redrawn on input.
const NS = "http://www.w3.org/2000/svg";
const FONT = "var(--font-sans, Inter, sans-serif)";

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function text(parent, x, y, s, opts = {}) {
  const t = el("text", {
    x, y, "text-anchor": opts.anchor || "middle",
    style: `font-family:${FONT};font-size:${opts.size || 15}px;fill:${opts.fill || "var(--fg)"};font-weight:${opts.weight || 500}`,
  }, parent);
  t.textContent = s;
  return t;
}
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }

// Deterministic PRNG so the figure is stable.
function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

// Diverging colour for value v in [-1, 1] using --c-div-0..8.
function divColor(v) {
  const i = Math.max(0, Math.min(8, Math.round((v + 1) * 4)));
  return `var(--c-div-${i})`;
}
function seqColor(v) {
  const i = Math.max(0, Math.min(8, Math.round(v * 8)));
  return `var(--c-seq-${i})`;
}

function heatmap(svg, M, x, y, size, colorFn, label) {
  const n = M.length, c = size / n;
  const g = el("g", {}, svg);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    el("rect", { x: x + j * c, y: y + i * c, width: c + 0.3, height: c + 0.3, fill: colorFn(M[i][j]) }, g);
  }
  el("rect", { x, y, width: size, height: size, fill: "none", stroke: "var(--line)" }, g);
  text(svg, x + size / 2, y + size + 24, label, { size: 15, weight: 600 });
}

function arrow(svg, x1, x2, y) {
  el("line", { x1, y1: y, x2: x2 - 8, y2: y, stroke: "var(--mute)", "stroke-width": 2.5 }, svg);
  el("path", { d: `M${x2} ${y} l-11 -6 v12 z`, fill: "var(--mute)" }, svg);
}

function drawReadout(svg) {
  clear(svg);
  const n = 9; // inputs, three blocks of three
  const blk = (i) => Math.floor(i / 3);
  const r = rng(7);
  // True covariance: block structure, decreasing block strength.
  const strength = [1.0, 0.7, 0.45];
  const S = [];
  for (let i = 0; i < n; i++) {
    S.push([]);
    for (let j = 0; j < n; j++) {
      S[i].push(blk(i) === blk(j) ? strength[blk(i)] * (i === j ? 1 : 0.8) : 0.05);
    }
  }
  // Centred covariance P S P, P = I - 11^T/n (what the linear readout identifies).
  const rowM = S.map((row) => row.reduce((a, b) => a + b, 0) / n);
  const all = rowM.reduce((a, b) => a + b, 0) / n;
  const PSP = S.map((row, i) => row.map((v, j) => v - rowM[i] - rowM[j] + all));
  const maxAbs = Math.max(...PSP.flat().map(Math.abs));
  const norm = (M) => M.map((row) => row.map((v) => v / maxAbs));
  const truth = norm(PSP);
  const rec = truth.map((row) => row.map((v) => v + (r() - 0.5) * 0.08)); // sampling-level noise

  // Weights: 12 interior neurons, block-diagonal-ish with noise.
  const N = 12;
  const W = [];
  for (let i = 0; i < N; i++) {
    W.push([]);
    for (let j = 0; j < N; j++) {
      const same = Math.floor(i / 4) === Math.floor(j / 4);
      W[i].push(i === j ? 0 : same ? 0.6 + 0.4 * r() : (r() < 0.1 ? 0.25 * r() : 0));
    }
  }
  // Embeddings H: N x n, each row a distribution concentrated on its block.
  const H = [];
  for (let i = 0; i < N; i++) {
    const b = Math.floor(i / 4);
    const row = [];
    for (let j = 0; j < n; j++) row.push((blk(j) === b ? 1 : 0.08) * (0.6 + 0.4 * r()));
    const s = row.reduce((a, c) => a + c, 0);
    H.push(row.map((v) => v / s));
  }
  const hMax = Math.max(...H.flat());

  const sz = 170, y = 50;
  const xs = [20, 260, 520, 760];
  text(svg, xs[0] + sz / 2, 30, "learned", { size: 14, fill: "var(--mute)" });
  heatmap(svg, W, xs[0], y, sz, seqColor, "weights W");
  arrow(svg, xs[0] + sz + 12, xs[1] - 12, y + sz / 2);
  text(svg, (xs[0] + sz + xs[1]) / 2, y + sz / 2 - 12, "+ wiring", { size: 14, fill: "var(--mute)" });
  // H drawn as N x n (non-square): squash into the square slot.
  const g = el("g", {}, svg);
  const cw = sz / n, ch = sz / N;
  for (let i = 0; i < N; i++) for (let j = 0; j < n; j++) {
    el("rect", { x: xs[1] + j * cw, y: y + i * ch, width: cw + 0.3, height: ch + 0.3, fill: seqColor(H[i][j] / hMax) }, g);
  }
  el("rect", { x: xs[1], y, width: sz, height: sz, fill: "none", stroke: "var(--line)" }, g);
  text(svg, xs[1] + sz / 2, y + sz + 24, "embeddings H", { size: 15, weight: 600 });
  arrow(svg, xs[1] + sz + 12, xs[2] - 12, y + sz / 2);
  text(svg, (xs[1] + sz + xs[2]) / 2, y + sz / 2 - 12, "+ rule", { size: 14, fill: "var(--mute)" });
  text(svg, xs[2] + sz / 2, 30, "inferred", { size: 14, fill: "var(--mute)" });
  heatmap(svg, rec, xs[2], y, sz, divColor, "recovered PΣP");
  text(svg, (xs[2] + sz + xs[3]) / 2, y + sz / 2 + 8, "≈", { size: 34, fill: "var(--mute)" });
  text(svg, xs[3] + sz / 2, 30, "actual", { size: 14, fill: "var(--mute)" });
  heatmap(svg, truth, xs[3], y, sz, divColor, "true PΣP");
}

// Slide 8: 30 neurons on a line, positions = 1-D "response". Group by gaps > threshold(b).
function makeKnobs(svg, bIn, aIn) {
  const r = rng(3);
  const centres = [0.08, 0.22, 0.31, 0.5, 0.58, 0.77, 0.9];
  const base = [];
  for (let i = 0; i < 30; i++) {
    const c = centres[i % centres.length];
    base.push({ c, off: (r() - 0.5) * 0.09 });
  }
  function draw() {
    clear(svg);
    const b = bIn.value / 100, alpha = aIn.value / 100;
    // tightness: low alpha -> points pulled hard toward their group mean.
    const pts = base.map((p) => p.c + p.off).sort((x, y) => x - y);
    const thr = 0.005 + b * b * 0.35; // larger b: fuse across larger gaps (fewer groups)
    const groups = [[pts[0]]];
    for (let i = 1; i < pts.length; i++) {
      if (pts[i] - pts[i - 1] <= thr) groups[groups.length - 1].push(pts[i]);
      else groups.push([pts[i]]);
    }
    const pull = 1 - (0.15 + 0.8 * alpha); // alpha small -> pull ~0.85 (tight)
    const X0 = 40, X1 = 600;
    el("line", { x1: X0, y1: 170, x2: X1, y2: 170, stroke: "var(--line)", "stroke-width": 1.5 }, svg);
    text(svg, X0, 228, "neuron response (1-D toy)", { anchor: "start", size: 13, fill: "var(--mute)" });
    groups.forEach((gp, k) => {
      const m = gp.reduce((a, c) => a + c, 0) / gp.length;
      const col = `var(--c-cat-${k % 8})`;
      const xs = gp.map((v) => X0 + (X1 - X0) * (v + (m - v) * pull));
      const lo = Math.min(...xs), hi = Math.max(...xs);
      el("rect", { x: lo - 12, y: 135, width: hi - lo + 24, height: 70 - 0, rx: 14, fill: col, "fill-opacity": 0.12, stroke: col, "stroke-opacity": 0.5 }, svg);
      gp.forEach((v, i) => {
        el("circle", { cx: xs[i], cy: 155 + (i % 3) * 13, r: 6.5, fill: col, stroke: "var(--bg, #fff)", "stroke-width": 1.2 }, svg);
      });
    });
    const tightLbl = alpha < 0.35 ? "tight, sticky" : alpha < 0.7 ? "medium" : "loose, little memory";
    text(svg, 320, 60, `${groups.length} group${groups.length === 1 ? "" : "s"}`, { size: 30, weight: 700 });
    text(svg, 320, 92, `resolution from b · ${tightLbl} from α`, { size: 15, fill: "var(--mute)" });
    text(svg, 320, 260, "small b: many small groups   ·   large b: few big groups", { size: 14, fill: "var(--mute)" });
  }
  bIn.addEventListener("input", draw);
  aIn.addEventListener("input", draw);
  draw();
}

// Slide 9: allocation rules.
function makeAlloc(svg, ctl) {
  const lam = [3.2, 2.1, 1.5, 1.0, 0.65, 0.4, 0.22, 0.1];
  const BUDGET = 60;
  function roundTo(q) {
    const s = q.reduce((a, c) => a + c, 0) || 1;
    const raw = q.map((v) => (v / s) * BUDGET);
    const n = raw.map(Math.floor);
    let rem = BUDGET - n.reduce((a, c) => a + c, 0);
    const order = raw.map((v, i) => [v - Math.floor(v), i]).sort((a, b) => b[0] - a[0]);
    for (let k = 0; k < rem; k++) n[order[k][1]]++;
    return n;
  }
  const mu = 0.6;
  const rules = {
    prop: { n: roundTo(lam), note: "n ∝ λ: every mode gets some" },
    water: { n: roundTo(lam.map((l) => Math.max(0, l - mu))), note: "n ∝ (λ − μ)₊: weak modes get none", mu },
    equal: { n: roundTo(lam.map(() => 1)), note: "equal: maximal resolution, least depth" },
  };
  let cur = "prop";
  function draw() {
    clear(svg);
    const R = rules[cur];
    const X0 = 60, bw = 56, gap = 15, base = 150, scale = 36;
    text(svg, 20, 24, "mode energy λ", { anchor: "start", size: 14, fill: "var(--mute)" });
    lam.forEach((l, a) => {
      const x = X0 + a * (bw + gap);
      el("rect", { x, y: base - l * scale, width: bw, height: l * scale, fill: `var(--c-cat-${a})`, "fill-opacity": 0.85, rx: 3 }, svg);
      // neuron dots below
      const n = R.n[a];
      for (let k = 0; k < n; k++) {
        const col = k % 4, row = Math.floor(k / 4);
        el("circle", { cx: x + 9 + col * 12.5, cy: base + 22 + row * 12.5, r: 5, fill: `var(--c-cat-${a})` }, svg);
      }
      text(svg, x + bw / 2, base + 22 + Math.ceil(Math.max(n, 1) / 4) * 12.5 + 14, String(n), { size: 14, weight: 600 });
    });
    el("line", { x1: X0 - 8, y1: base, x2: X0 + 8 * (bw + gap), y2: base, stroke: "var(--mute)" }, svg);
    if (R.mu) {
      const y = base - R.mu * scale;
      el("line", { x1: X0 - 8, y1: y, x2: X0 + 8 * (bw + gap), y2: y, stroke: "var(--fg)", "stroke-dasharray": "5 4", "stroke-width": 1.5 }, svg);
      text(svg, X0 + 8 * (bw + gap) + 2, y + 5, "μ", { anchor: "start", size: 15 });
    }
    text(svg, 320, 330, `${R.note} · 60 neurons total`, { size: 15, fill: "var(--mute)" });
  }
  ctl.querySelectorAll("button[data-rule]").forEach((btn) => {
    btn.addEventListener("click", () => {
      cur = btn.dataset.rule;
      ctl.querySelectorAll("button[data-rule]").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      draw();
    });
  });
  draw();
}

export function init(root, ctx) {
  const q = (s) => root.querySelector(s);
  const ro = q("#talk-c-readout-svg");
  if (ro) drawReadout(ro);
  const ks = q("#talk-c-knobs-svg"), bIn = q("#talk-c-b"), aIn = q("#talk-c-alpha");
  if (ks && bIn && aIn) makeKnobs(ks, bIn, aIn);
  const al = q("#talk-c-alloc-svg"), ctl = q("#talk-c-alloc-ctl");
  if (al && ctl) makeAlloc(al, ctl);
}
