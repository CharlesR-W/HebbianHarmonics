// STYLE GUIDE (placeholder for Chapter 1). Renders every component with FAKE data and FAKE dynamics so the design
// system can be seen working. It does not use the engine. Chapter authors: read this as worked examples of the
// API in DESIGN.md, then replace the file.
import * as ui from "./ui.js";
import { heatmap, lineChart, scatter, barChart, barMultiples, imageGrid, filmstrip } from "./charts.js";
import { view3d, simplexVertices, baryToXYZ } from "./view3d.js";
import { P, catColor, boundaryColor, mixBary } from "./palette.js";

// deterministic fake randomness
function rng(seed) { let s = seed >>> 0 || 1; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }
function randn(r) { return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()); }

const TOURS = [
  { id: "collapse", label: "No pruning: one cluster", note: "Fake stand-in for $\\theta \\le 0$: every neuron drifts to one point.", params: { theta: -0.1 } },
  { id: "vertices", label: "Pruning: three clusters", note: "Fake stand-in for $\\theta$ above the input correlations: each neuron goes to a vertex.", params: { theta: 0.35 } },
  { id: "edge", label: "Two inputs correlate", note: "Fake stand-in for correlated $g_1, g_2$: their neurons meet at the edge midpoint.", params: { theta: 0.15 } },
];

export function init(panel, ctx) {
  const B = 3, verts = simplexVertices(B), bcol = [0, 1, 2].map(b => boundaryColor(b, B));
  const Sigma = [1, .2, .05, .2, .8, .05, .05, .05, .6];
  // a tour in the URL wins over individual parameters (tours are exact states)
  const tour0 = TOURS.find(x => x.id === ctx.param("tour", ""));
  let theta = tour0 ? tour0.params.theta : ctx.param("theta", 0.35), wmin = ctx.param("wmin", 0.01), gamma = ctx.param("gamma", 1);
  let N = ctx.param("n", 150), spread = ctx.param("spread", 0.25), seed = ctx.param("seed", 1);
  let H, H0, target, jitter, t = 0, lastMove = [];

  // ---------- fake dynamics (NOT the model)
  function initState() {
    const r = rng(seed); H = new Float32Array(N * B); target = new Float32Array(N * B); t = 0; lastMove = [];
    for (let i = 0; i < N; i++) {
      let s = 0; for (let b = 0; b < B; b++) { const v = Math.max(0.02, 1 / 3 + spread * randn(r) * 0.4); H[i * B + b] = v; s += v; }
      for (let b = 0; b < B; b++) H[i * B + b] /= s;
    }
    H0 = H.slice(); jitter = Float32Array.from({ length: N * B }, () => r()); retarget();
  }
  function retarget() {
    for (let i = 0; i < N; i++) {
      let k = 0; for (let b = 1; b < B; b++) if (H0[i * B + b] > H0[i * B + k]) k = b;
      const tg = theta <= 0 ? [1 / 3, 1 / 3, 1 / 3] : theta < 0.25 && k < 2 ? [.5, .5, 0] : verts.map((_, b) => (b === k ? 1 : 0));
      let s = 0; for (let b = 0; b < B; b++) { target[i * B + b] = tg[b] + 0.05 + 0.08 * jitter[i * B + b]; s += target[i * B + b]; }
      for (let b = 0; b < B; b++) target[i * B + b] /= s;
    }
  }
  const r = rng(7);
  function step(n) {
    let mv = 0;
    for (let s = 0; s < n; s++) {
      t += 0.05; mv = 0;
      for (let i = 0; i < N; i++) {
        let sum = 0;
        for (let b = 0; b < B; b++) { const k = i * B + b, d = 0.02 * (target[k] - H[k]) + 0.0015 * randn(r) * spread, v = Math.max(1e-4, H[k] + d); mv += Math.abs(v - H[k]); H[k] = v; sum += v; }
        for (let b = 0; b < B; b++) H[i * B + b] /= sum;
      }
    }
    lastMove.push([t, mv / N]);
    if (lastMove.length > 600) lastMove = lastMove.filter((_, i) => i % 2 === 0);
  }
  const lift = h => { const z = new Float32Array(N); for (let i = 0; i < N; i++) { let q = 0; for (let a = 0; a < B; a++) for (let b = 0; b < B; b++) q += h[i * B + a] * Sigma[a * B + b] * h[i * B + b]; z[i] = 0.9 * q; } return z; };
  // fake synapses: strongest pairs by embedding proximity, at most 5,000
  function edgesOf(h) {
    const cand = [];
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      let d = 0; for (let b = 0; b < B; b++) d += Math.abs(h[i * B + b] - h[j * B + b]);
      const w = Math.exp(-d * d / 0.02); if (w > 0.05) cand.push([i, j, w]);
    }
    for (let i = 0; i < N; i++) for (let b = 0; b < B; b++) { const w = h[i * B + b] ** 3; if (w > 0.05) cand.push([i, N + b, w * 0.8]); }
    cand.sort((a, b) => b[2] - a[2]); cand.length = Math.min(cand.length, 5000);
    const pairs = new Uint32Array(cand.length * 2), alpha = new Float32Array(cand.length);
    cand.forEach(([i, j, w], e) => { pairs[2 * e] = i; pairs[2 * e + 1] = j; alpha[e] = w; });
    return { pairs, alpha };
  }

  // ---------- the sim figure: view + transport left, controls + readouts right
  const L = ui.simLayout({ parent: ctx.slot("sim") });
  const v = view3d(L.view, { height: 470, autoRotate: false, tooltip: (i, kind) => kind === "anchor"
    ? ui.tipRows(`boundary neuron g${i + 1}`, [{ name: "clamped voltage", value: "input" }], null)
    : ui.tipRows(`interior neuron ${i}`, [0, 1, 2].map(b => ({ name: `h(g${b + 1})`, value: ui.fmtFixed(H[i * B + b], 3), color: bcol[b], shape: "square" }))) });
  v.setAnchors(verts.map((xyz, b) => ({ xyz, label: `g${"₁₂₃"[b]}`, color: bcol[b] })));
  const render = h => {
    const xyz = baryToXYZ(h, B, verts, null, lift(h));
    const colors = Array.from({ length: N }, (_, i) => mixBary([h[i * B], h[i * B + 1], h[i * B + 2]], bcol));
    v.setPoints(xyz, colors);
    const e = edgesOf(h); v.setEdges(e.pairs, e.alpha);
    stats.update({ t, n: N, syn: e.alpha.length, k: theta <= 0 ? 1 : theta < 0.25 ? 2 : 3 });
    return e;
  };
  let tick = 0, fpsT = performance.now(), fpsN = 0;
  const tp = ui.transport({
    parent: L.view, watch: ctx.slot("sim"), speeds: [1, 4, 16], speed: 4, autoplay: true, maxHistory: 300,
    onTick: speed => {
      step(speed); render(H); tick += 1; fpsN += 1;
      const now = performance.now(); if (now - fpsT > 1000) { stats.set("fps", fpsN * 1000 / (now - fpsT)); fpsN = 0; fpsT = now; }
      if (tick % 6 === 0) { drawCov(H); moveChart.update({ series: [{ name: "mean movement", x: lastMove.map(p => p[0]), y: lastMove.map(p => p[1]) }] }); }
      moveChart.setCursor(t);
      const conv = lastMove.length > 5 && lastMove[lastMove.length - 1][1] < 1e-4;
      return { t, H: H.slice(), done: conv && t > 30 };
    },
    onSeek: f => { render(f.H); drawCov(f.H); moveChart.setCursor(f.t); },
    onRestore: f => { H = f.H.slice(); t = f.t; lastMove = lastMove.filter(p => p[0] <= t); },
    onReset: () => { initState(); v.clearTrails(); render(H); drawCov(H); return { t, H: H.slice() }; },
  });

  const tours = ui.chips({ parent: L.side, label: "Tours", items: TOURS, active: ctx.param("tour", null), onPick: it => applyTour(it) });
  const thetaS = ui.slider({ label: "pruning threshold $\\theta$", min: -0.2, max: 0.6, step: 0.01, value: theta, fmt: x => ui.fmtFixed(x, 2),
    marks: [{ value: 0, label: "0" }, { value: 0.2, label: "ρ₁₂" }], help: "Synapses survive where $\\mathrm{Cov}(u_i,u_j) > \\theta$.",
    onInput: x => { theta = x; retarget(); changed({ theta: x }); } });
  const wminS = ui.slider({ label: "floor $w_{\\min}\\lambda/\\eta$", min: 1e-4, max: 0.1, log: true, value: wmin, onChange: x => { wmin = x; changed({ wmin: x }); } });
  const gammaS = ui.segmented({ label: "lateral gain $\\gamma$", options: [0, 0.5, 1, 2].map(x => ({ value: x, label: String(x) })), value: gamma, onChange: x => { gamma = x; changed({ gamma: x }); } });
  const nS = ui.slider({ label: "interior neurons $N$", min: 30, max: 300, step: 10, value: N, onChange: x => { N = x; changed({ n: x }); tp.reset(); } });
  const spreadS = ui.slider({ label: "initial spread", min: 0, max: 1, step: 0.05, value: spread, onChange: x => { spread = x; changed({ spread: x }); tp.reset(); } });
  const seedS = ui.select({ label: "seed", options: [1, 2, 3], value: seed, onChange: x => { seed = x; changed({ seed: x }); tp.reset(); } });
  const trailsT = ui.toggle({ label: "fading trails", value: false, onChange: on => v.setTrails(on ? 14 : 0) });
  const rotT = ui.toggle({ label: "auto-rotate", value: false, onChange: on => v.setAutoRotate(on) });
  ui.controls({ parent: L.side, title: "Parameters", primary: [thetaS, wminS, gammaS], advanced: [nS, spreadS, seedS, trailsT, rotT], advancedLabel: "More controls" });
  const stats = ui.stats({ parent: L.side, items: [
    { key: "k", label: "clusters", fmt: x => String(x) }, { key: "syn", label: "visible synapses", fmt: ui.fmtInt },
    { key: "t", label: "time", fmt: x => ui.fmt(x, 3) }, { key: "fps", label: "render rate (style guide)", unit: "fps", fmt: x => x.toFixed(0) }] });
  ui.legend({ parent: L.side, items: [
    { label: "interior neuron", color: P.interior, shape: "dot" }, { label: "boundary neuron", color: bcol[0], shape: "square" },
    { label: "synapse, opacity ∝ w", color: P.edge, shape: "line" }] });

  function changed(params) { tours.moved(); ctx.setParams({ ...params, tour: null }); }
  function applyTour(it) {
    theta = it.params.theta; thetaS.set(theta); retarget();
    ctx.setParams({ theta, tour: it.id });
    if (!tp.playing) tp.play();
  }
  ctx.onParams(p => { const id = p.get("tour"); const it = TOURS.find(x => x.id === id); if (it) { tours.set(id); applyTour(it); } else if (p.has("theta")) { theta = +p.get("theta"); thetaS.set(theta); retarget(); } });
  ctx.onHide(() => v.setAutoRotate(false));

  // ---------- linked heatmap and line chart
  const covBox = ui.h("div"), moveBox = ui.h("div");
  L.below.append(ui.h("div", {}, ui.h("div", { class: "chart-title", html: "Voltage covariance $H\\Sigma_g H^\\top$, sorted by cluster" }), covBox),
    ui.h("div", {}, ui.h("div", { class: "chart-title", text: "Mean movement per step" }), moveBox));
  ui.renderMath(L.below);
  const cov = heatmap(covBox, { map: "div", range: [-0.6, 0.6], zlabel: "$h_i^\\top\\Sigma_g h_j$ (fake)", size: 300, xlabel: "neuron j (sorted)", ylabel: "neuron i (sorted)" });
  function drawCov(h) {
    const n = h.length / B, idx = Array.from({ length: n }, (_, i) => i), arg = i => { let k = 0; for (let b = 1; b < B; b++) if (h[i * B + b] > h[i * B + k]) k = b; return k; };
    idx.sort((a, b) => arg(a) - arg(b) || h[b * B + arg(b)] - h[a * B + arg(a)]);
    const M = new Float32Array(n * n);
    for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) {
      const i = idx[a], j = idx[c]; let q = 0;
      for (let x = 0; x < B; x++) for (let y = 0; y < B; y++) q += h[i * B + x] * Sigma[x * B + y] * h[j * B + y];
      M[a * n + c] = q - 0.4;
    }
    const groups = []; let s = 0;
    for (let k = 0; k < B; k++) { const e = s + idx.filter(i => arg(i) === k).length; if (e > s) groups.push({ start: s, end: e, color: bcol[k], label: `g${k + 1}` }); s = e; }
    cov.update({ values: M, rows: n, cols: n, groups: { rows: groups } });
  }
  const moveChart = lineChart(moveBox, { series: [{ name: "mean movement", y: [1], x: [0] }], ylog: true, height: 230, xlabel: "time t", ylabel: "mean ‖Δh‖₁", legend: false });

  ctx.onShow(() => { if (!H) { initState(); const f = { t, H: H.slice() }; tp.record(f); render(H); drawCov(H); } });
  if (!H) { initState(); tp.record({ t, H: H.slice() }); render(H); drawCov(H); }

  galleries(ctx);
}

// ---------------------------------------------------------------- static galleries (fake data)
function galleries(ctx) {
  const r = rng(42);
  // big heatmap: 1200 x 1200, four blocks
  const n = 1200, M = new Float32Array(n * n), bl = [0, 380, 700, 980, 1200], blk = i => bl.findIndex((b, k) => i >= b && i < bl[k + 1]);
  const bi = Int8Array.from({ length: n }, (_, i) => blk(i));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { const v = (bi[i] === bi[j] ? 0.65 : 0.12) + 0.08 * (r() - 0.5); M[i * n + j] = M[j * n + i] = v; }
  heatmap(ctx.slot("hm-big"), { values: M, rows: n, cols: n, map: "seq", range: [0, 1], size: 560, zlabel: "similarity (fake)",
    groups: { rows: bl.slice(0, -1).map((s, k) => ({ start: s, end: bl[k + 1], label: `block ${k + 1}`, color: catColor(k) })) }, xlabel: "neuron j", ylabel: "neuron i" });

  // scatter: 300 points in 5 clusters + unassigned
  const x = [], y = [], cat = [];
  for (let i = 0; i < 300; i++) { const k = i % 6 === 5 ? -1 : i % 6; const a = k < 0 ? r() * 6.28 : k * 1.257; const rad = k < 0 ? 0.3 + r() * 0.8 : 0.8;
    x.push(rad * Math.cos(a) + 0.12 * randn(r)); y.push(rad * Math.sin(a) + 0.12 * randn(r)); cat.push(k); }
  const sc = scatter(ctx.slot("scatter"), { x, y, category: cat, equal: true, height: 300, xlabel: "map x", ylabel: "map y",
    anchors: [{ x: 1.25, y: 0, label: "g₁", color: P.cat[0] }, { x: -1.1, y: 0.6, label: "g₂", color: P.cat[1] }],
    tooltip: i => ui.tipRows(`neuron ${i}`, [{ name: "cluster", value: cat[i] < 0 ? "unassigned" : cat[i] + 1, color: catColor(cat[i]) }]),
    onSelect: i => sc.highlight(i) });

  // bars with overlay
  const labels16 = Array.from({ length: 16 }, (_, i) => `${i + 1}`);
  const part = labels16.map((_, i) => Math.max(0, Math.exp(-((i - 5) ** 2) / 6) - 0.15) * 0.5), eig = labels16.map((_, i) => 0.42 * Math.cos(i / 2.6) * Math.exp(-i / 14));
  barChart(ctx.slot("bars"), { labels: labels16, values: part, color: P.cat[0], name: "cluster part h", overlay: { values: eig, name: "eigenvector 1 of Σ" }, xlabel: "boundary neuron", ylabel: "weight", height: 230 });

  // small multiples
  const labels12 = Array.from({ length: 12 }, (_, i) => `${i + 1}`);
  barMultiples(ctx.slot("multi"), { labels: labels12, yrange: [-0.6, 0.6], valueName: "cluster part h", overlayName: "nearest eigenvector of Σ",
    panels: [0, 1, 2, 3].map(k => ({ title: `cluster ${k + 1}`, sub: `${[41, 33, 27, 19][k]} neurons`, values: labels12.map((_, i) => Math.max(0, 0.5 * Math.exp(-((i - 3 * k - 1) ** 2) / 2) - 0.05)),
      overlay: labels12.map((_, i) => 0.45 * Math.sin((i + 1) * (k + 1) * Math.PI / 13)) })) });

  // image grid with labelled rows
  const arrays = [], colors = [];
  for (let k = 0; k < 3; k++) for (let m = 0; m < 6; m++) {
    const a = new Float32Array(196), cx = 4 + 3 * k + randn(r), cy = 5 + m * 0.8;
    for (let p = 0; p < 196; p++) { const px = p % 14, py = (p / 14) | 0; a[p] = Math.exp(-((px - cx) ** 2 + (py - cy) ** 2) / 8) - 0.5 * Math.exp(-((px - 13 + cx) ** 2 + (py - 7) ** 2) / 10); }
    arrays.push(a); colors.push(catColor(k));
  }
  imageGrid(ctx.slot("images"), { arrays, w: 14, h: 14, map: "div", colors, tile: 40, rows: [0, 1, 2].map(k => ({ label: `cluster ${k + 1}`, count: 6 })),
    tooltip: i => ui.tipRows(`neuron ${i}`, [{ name: "cluster", value: ((i / 6) | 0) + 1, color: colors[i] }]) });

  // filmstrip
  const frames = Array.from({ length: 8 }, (_, f) => {
    const m = 40, F = new Float32Array(m * m), s = f / 7;
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) F[i * m + j] = (i === j ? 1 : ((i / 10 | 0) === (j / 10 | 0) ? 0.85 * s : -0.25 * s)) + 0.25 * (1 - s) * (r() - 0.5);
    return F;
  });
  filmstrip(ctx.slot("film"), { frames, rows: 40, cols: 40, map: "div", range: [-1, 1], size: 260, frameLabels: frames.map((_, f) => `step ${f * 250}`), sliderLabel: "snapshot", zlabel: "similarity (fake)" });

  // controls gallery
  const host = ctx.slot("controls");
  const grid = ui.h("div", { class: "sg-grid" }); host.append(grid);
  const col = (...xs) => { const c = ui.h("div", { class: "sg-col" }, ...xs); grid.append(c); return c; };
  col(ui.h("div", { class: "side-h", text: "Sliders" }),
    ui.slider({ label: "linear slider, $\\gamma$", min: 0, max: 2, step: 0.1, value: 1 }).el,
    ui.slider({ label: "log slider, learning rate $\\eta$", min: 1e-4, max: 1, log: true, value: 0.01, unit: "/step" }).el,
    ui.slider({ label: "with marks", min: 0, max: 1, step: 0.01, value: 0.3, marks: [{ value: 0.1, label: "across" }, { value: 0.5, label: "within" }] }).el);
  col(ui.h("div", { class: "side-h", text: "Choices" }),
    ui.segmented({ label: "segmented", options: ["mean-field", "minibatch"], value: "mean-field" }).el,
    ui.select({ label: "select", options: [{ value: "pow", label: "power-law spectrum" }, { value: "blocks", label: "planted blocks" }], value: "blocks" }).el,
    ui.toggle({ label: "toggle", value: true, help: "Help text under any control." }).el);
  const tiles = ui.stats({ items: [{ key: "bw", label: "Bures–Wasserstein", fmt: x => ui.fmtFixed(x, 3) }, { key: "cos", label: "Frobenius cosine", fmt: x => ui.fmtFixed(x, 3) },
    { key: "rank", label: "eff. rank of $H$" }, { key: "n", label: "in clusters", unit: "of 300", fmt: ui.fmtInt }] });
  tiles.update({ bw: 0.214, cos: 0.871, rank: 3.24, n: 212 });
  col(ui.h("div", { class: "side-h", text: "Readouts" }), tiles, ui.h("div", { class: "u-help", text: "Stat tiles: label, value, unit." }),
    ui.chips({ label: "Preset chips", items: [{ id: "a", label: "blocks", note: "A chip's note: two definitional lines." }, { id: "b", label: "power law" }], active: "a" }));

  // colour grammar
  const cg = ctx.slot("colours");
  const row = (name, ...kids) => cg.append(ui.h("div", { class: "sg-row" }, ui.h("div", { class: "sg-name", html: name }), ui.h("div", { class: "sg-body" }, ...kids)));
  row("categorical <code>P.cat</code>", ...P.cat.map((c, i) => ui.h("span", { class: "sg-chip", style: { "--sw": c } }, ui.h("i"), `${i}`)), ui.h("span", { class: "sg-chip", style: { "--sw": P.other } }, ui.h("i"), "other"));
  row("barycentric <code>mixBary</code>", ...[[1, 0, 0], [.5, .5, 0], [0, 1, 0], [0, .5, .5], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3]].map(hh => ui.h("span", { class: "sg-chip", style: { "--sw": mixBary(hh, P.cat.slice(0, 3)) } }, ui.h("i"), `(${hh.map(x => +x.toFixed(2)).join(", ")})`)));
  row("<code>seq</code>: magnitudes", ui.colorbar({ map: "seq", range: [0, 1], label: "counts, errors, phase diagrams", width: 260 }).el);
  row("<code>ink</code>: weights only", ui.colorbar({ map: "ink", range: [0, 1], label: "conductance $w_{ij}/\\max w$", width: 260 }).el);
  row("<code>div</code>: signed, zero-centred", ui.colorbar({ map: "div", range: [-1, 1], label: "covariance, similarity, loadings", width: 260 }).el);
  row("neurons and synapses", ui.legend({ items: [{ label: "interior neuron", color: P.interior, shape: "dot" }, { label: "boundary neuron", color: P.cat[0], shape: "square" },
    { label: "boundary neuron, $B > 8$", color: P.boundary, shape: "square" }, { label: "synapse", color: P.edge, shape: "line" }, { label: "eigenvector / reference", color: P.fg, shape: "dash" }] }).el);
  ui.renderMath(cg);
}
