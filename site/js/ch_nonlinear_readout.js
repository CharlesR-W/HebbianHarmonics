// Chapter "nonlinear": the distance readout of a rectifying network, live (N = 40) and from the sweep
// (data/sweeps2/readout.json, `nl`). Moved here from the readout chapter (ch_readout.js) on 2026-11-19, unchanged
// apart from the helpers it needs: what the weights state about the response geometry (exact on live edges, bounded
// elsewhere), with a classical MDS of the measured distances in response-function space and of the weights'
// completion (labelled an approximation).
import * as ui from "./ui.js";
import { heatmap, lineChart, hidpi, scale, niceTicks } from "./charts.js";
import { P, catColor, cmap, rgbToCss } from "./palette.js";
import * as O from "./onsite_engine.js";
import * as R from "./ch_readout_math.js";
import { netview } from "./netview.js";

const B = 12, ALPHA = 1, DT = 0.5, S_TRAIN = 256, S_EVAL = 4096, STOP = 1e-9, RES_TOL = 1e-6;
const SUBS = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉", "₁₀", "₁₁", "₁₂"];
const GLAB = SUBS.map(s => "g" + s);
const SIGMAS = { blocks: O.covBlocks({ sizes: [3, 3, 3, 3], within: 0.8, between: 0, variances: [2, 1, 0.5, 0.25] }).Sigma };
const seqColor = t => rgbToCss(cmap("seq")(t));
const rng = (a) => { const v = a.filter(Number.isFinite); return v.length ? [Math.min(...v), Math.max(...v)] : [NaN, NaN]; };
const fmtRange = ([a, b], f = x => ui.fmtFixed(x, 3)) => (!Number.isFinite(a) ? "n/a" : f(a) === f(b) ? f(a) : `${f(a)}–${f(b)}`);

// helpers
// space: optional badge kind ("input" | "function" | "neuron") shown beside the title (ui.panelHead)
function chartBox(parent, title, sub, space) {
  const body = ui.h("div");
  const head = space ? ui.panelHead(title, space) : ui.h("div", { class: "chart-title", html: title });
  const box = ui.h("div", { style: { minWidth: "0" } }, head, sub ? ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" }, html: sub }) : null, body);
  parent.append(box); ui.renderMath(box);
  return body;
}
const grid2 = (min = 280) => ui.h("div", { style: { display: "grid", gridTemplateColumns: `repeat(auto-fit, minmax(min(${min}px, 100%), 1fr))`, gap: "18px 28px" } });
function chatOf(G, S) {
  const C = new Float64Array(B * B);
  for (let s = 0; s < S; s++) for (let a = 0; a < B; a++) { const x = G[s * B + a]; for (let c = 0; c < B; c++) C[a * B + c] += x * G[s * B + c]; }
  for (let k = 0; k < B * B; k++) C[k] /= S;
  return C;
}
function liveEdges(W, N) { let m = 0; for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (W[i * N + j] > 1e-9) m++; return m; }
// components sorted by size (largest first) with the neuron order that groups them
function componentOrder(W, N, U, S) {
  const { labels, K } = O.components(W, N);
  const st = U ? O.clusterStats(U, S, N, labels, K) : null;
  const comps = Array.from({ length: K }, (_, k) => ({ k, members: [] }));
  for (let i = 0; i < N; i++) comps[labels[i]].members.push(i);
  comps.forEach(c => { c.size = c.members.length; c.diam = st ? st[c.k].diamMax : NaN; });
  comps.sort((a, b) => b.size - a.size || a.k - b.k);
  const order = []; const groups = [];
  comps.forEach(c => { const s = order.length; order.push(...c.members); if (c.size > 1) groups.push({ start: s, end: order.length }); });
  let diamMax = 0; comps.forEach(c => { if (c.size > 1 && c.diam > diamMax) diamMax = c.diam; });
  return { labels, K, comps, order, groups, largest: comps[0].size, diamMax };
}


// (4) nonlinear: partial distance geometry, live
const NL_DEF = { nb: 0.1, nseed: 1 };
const NL_TOURS = [
  { id: "frag", label: "Fragmented", params: { nb: 0.02 }, note: "$b = 0.02$: few edges; almost every pair is known only from below." },
  { id: "perc", label: "Percolating", params: { nb: 0.1 }, note: "$b = 0.1$: a giant component of chains; shortest paths bound most distances from above, loosely." },
  { id: "fused", label: "Fused", params: { nb: 0.3 }, note: "$b = 0.3$: nearly every pair is an edge, so nearly every distance is known exactly." },
];
export function nlFigure(ctx) {
  const slot = ctx.slot("nl"); slot.replaceChildren();
  const fig = slot.closest("figure");
  const N = 40, Sigma = SIGMAS.blocks;
  let p = { nb: Math.min(0.5, Math.max(0.005, ctx.param("nb", NL_DEF.nb))), nseed: Math.max(0, Math.min(9, Math.round(ctx.param("nseed", NL_DEF.nseed)))) };
  let tourActive = NL_TOURS.find(t => t.id === ctx.param("ntour", "")) || null;
  if (tourActive) p = { ...NL_DEF, ...tourActive.params };
  let sim = null, Ge = null, cur = null, tick = 0;

  const L = ui.simLayout({ parent: slot });
  const two = grid2(270); L.view.append(two);
  const bSc = chartBox(two, "(a) What the weights say, against the true $D_{ij}$", "training bank; one point per pair");
  const sc = lineChart(bSc, { series: [{ name: "x", y: [] }], height: 300, xlabel: "true D (training bank)", ylabel: "from the weights", xname: "true D" });
  const bSt = chartBox(two, "(b) What is known about each pair", "neurons sorted by component, largest first; hover a cell to find the pair in (c) and (d)");
  const hSt = heatmap(bSt, { map: t => (t < 0.25 ? cmap("seq")(0.92) : t < 0.75 ? cmap("seq")(0.5) : cmap("seq")(0.1)), range: [0, 2], size: 270, colorbar: false,
    xlabel: "neuron j", ylabel: "neuron i", tooltip: (a, c, v) => stTip(a, c, v), onHover: (a, c) => linkPair(a, c) });
  ui.legend({ parent: bSt, items: [
    { label: "live edge: $D$ exact", color: seqColor(0.92), shape: "square" },
    { label: "same component: $\\sqrt{2b} \\le D \\le$ shortest path", color: seqColor(0.5), shape: "square" },
    { label: "different components: $D \\ge \\sqrt{2b}$", color: seqColor(0.1), shape: "square" },
  ] });
  // (c) response-function space beside (d) neuron space, linked by hover
  const pair = ui.h("div", { class: "space-pair", style: { marginTop: "20px" } }); L.view.append(pair);
  const bMd = chartBox(pair, "(c) Largest component: classical MDS of response distances", "dots: measured $D$ · rings: completed from the weights (an approximation)", "function");
  const md = mdsChart(bMd, { height: 300, onHover: i => nv.setHighlight(i) });
  ui.synapseToggle({ parent: bMd.parentNode, label: "show synapses (live edges)", onChange: on => md.setSynapses(on) });
  const bNv = chartBox(pair, "(d) The same network: neurons, inputs and live synapses", "positions are random; colour = component, largest first");
  const COMP_COLORS = 8;
  const nv = netview(bNv, { N, B, seed: 3, aspect: 1.25, maxHeight: 320, minHeight: 240, radius: 6, edgeWidth: 3,
    tooltip: (i, kind) => nvTip(i, kind), onHover: (i, kind) => md.setHighlight(kind === "neuron" ? i : null) });
  ui.legend({ parent: bNv, items: [
    { label: "neuron, coloured by component (grey: singleton)", color: catColor(0), shape: "dot" },
    { label: "input $g_b$", color: P.boundary, shape: "square" },
    { label: "live synapse or attachment, width ∝ conductance", color: P.edge, shape: "line" },
  ] });
  const PAIRS = new Uint32Array(2 * (N * (N - 1) / 2 + N * B)), EW = new Float32Array(PAIRS.length / 2);
  { let e = 0; for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { PAIRS[2 * e] = i; PAIRS[2 * e + 1] = j; e++; }
    for (let i = 0; i < N; i++) for (let b = 0; b < B; b++) { PAIRS[2 * e] = i; PAIRS[2 * e + 1] = N + b; e++; } }
  function linkPair(a, c) {
    if (a == null || !cur) { nv.setHighlight(null); md.setHighlight(null); return; }
    const ids = a === c ? [cur.co.order[a]] : [cur.co.order[a], cur.co.order[c]];
    nv.setHighlight(ids); md.setHighlight(ids);
  }
  function nvTip(i, kind) {
    if (!cur || !sim) return null;
    if (kind === "input") {
      let m = 0, s = 0; for (let k = 0; k < N; k++) { const a = sim.net.A[k * B + i]; if (a > 0) { m++; s += a; } }
      return ui.tipRows(`input ${GLAB[i]}`, [{ name: "attached neurons", value: String(m) }, { name: "total attachment conductance", value: ui.fmt(s, 3) }]);
    }
    const rank = cur.co.comps.findIndex(c => c.members.includes(i)), comp = cur.co.comps[rank];
    let deg = 0, ws = 0; for (let j = 0; j < N; j++) if (j !== i && cur.W[i * N + j] > 1e-9) { deg++; ws += cur.W[i * N + j]; }
    return ui.tipRows(`neuron ${i}`, [
      { name: "component", value: comp.size > 1 ? `#${rank + 1} of ${cur.co.K}, ${comp.size} neurons` : "singleton", color: nodeColor(rank, comp.size), shape: "dot" },
      { name: "live synapses", value: String(deg) }, { name: "sum of live w", value: ui.fmt(ws, 3) },
      { name: "threshold tᵢ", value: ui.fmt(sim.net.t[i], 3) }]);
  }
  const nodeColor = (rank, size) => (size > 1 && rank < COMP_COLORS ? catColor(rank) : P.other);
  const tp = ui.transport({ parent: L.view, watch: fig, autoplay: true, maxHistory: 200, speeds: [{ value: 1, label: "×1" }, { value: 8, label: "×8" }], speed: 8,
    timeFmt: t => `step ${t}`, onTick: s => onTick(s), onSeek: f => show(f), onReset: () => restart() });
  const tours = ui.chips({ parent: L.side, label: "Presets", items: NL_TOURS, active: tourActive ? tourActive.id : null, onPick: t => { tourActive = t; p = { ...NL_DEF, ...t.params }; bS.set(p.nb); seedS.set(p.nseed); retrain(); } });
  const bS = ui.slider({ label: "fusion scale $b$", min: 0.005, max: 0.5, log: true, sig: 2, value: p.nb, fmt: x => ui.fmt(x, 2), onChange: x => { p.nb = x; moved(); retrain(); } });
  const seedS = ui.slider({ label: "seed", min: 0, max: 9, step: 1, value: p.nseed, fmt: x => String(x), onChange: x => { p.nseed = x; moved(); retrain(); } });
  ui.controls({ parent: L.side, title: "Rectifying network", primary: [bS, seedS] });
  const stats = ui.stats({ parent: L.side, items: [
    { key: "k", label: "components $K$", fmt: x => String(x) },
    { key: "fx", label: "pairs known exactly", fmt: x => ui.fmtPct(x, 1) },
    { key: "fu", label: "bounded both sides", fmt: x => ui.fmtPct(x, 1) },
    { key: "fl", label: "bounded below only", fmt: x => ui.fmtPct(x, 1) },
    { key: "ee", label: "live-entry error", fmt: x => ui.fmt(x, 2), help: "max |√(2(b − αw)) − D| over live edges, training bank" },
    { key: "ev", label: "vs evaluation bank", fmt: x => ui.fmtPct(x, 1), help: "max relative |√(2(b − αw)) − D_eval| over live edges: sampling" },
    { key: "sp", label: "shortest path / true $D$", fmt: x => ui.fmt(x, 3), help: "median over same-component non-edge pairs" },
    { key: "dis", label: "MDS disparity", fmt: x => ui.fmt(x, 2), help: "Procrustes disparity, completed vs true, largest component, 3-D classical MDS" },
  ] });

  function build() {
    const net = O.makeNet({ N, B, rng: O.makeRng(p.nseed), s: 3, klo: 3, khi: 0, sigmaT: 0.7 });
    const G = O.sampleInputs(Sigma, B, S_TRAIN, O.makeRng(1000 + p.nseed));
    Ge = O.sampleInputs(Sigma, B, S_EVAL, O.makeRng(2000 + p.nseed));
    sim = O.createSim({ net, G, S: S_TRAIN, b: p.nb, alpha: ALPHA, dt: DT });
    tick = 0;
  }
  function analyse(done) {
    const b = p.nb, W = sim.net.W, D2 = sim.D2;
    const geo = R.distanceGeometry(W, N, b, ALPHA);
    let nx = 0, nu = 0, nlo = 0, ee = 0;
    const pts = { x: [[], [], []], y: [[], [], []], ij: [[], [], []] }, ratios = [];
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const st = geo.status[i * N + j], D = Math.sqrt(Math.max(D2[i * N + j], 0));
      if (st === 0) { nx++; ee = Math.max(ee, Math.abs(geo.Dw[i * N + j] - D)); pts.x[0].push(D); pts.y[0].push(geo.Dw[i * N + j]); pts.ij[0].push([i, j]); }
      else if (st === 1) { nu++; const sp = geo.Dsp[i * N + j]; ratios.push(sp / Math.max(D, 1e-300)); pts.x[1].push(D); pts.y[1].push(sp); pts.ij[1].push([i, j]); }
      else { nlo++; pts.x[2].push(D); pts.y[2].push(Math.sqrt(2 * b)); pts.ij[2].push([i, j]); }
    }
    ratios.sort((a, c) => a - c);
    const np = N * (N - 1) / 2;
    const co = componentOrder(W, N, null, 0);
    const f = { t: sim.steps, b, geo, co, pts, fx: nx / np, fu: nu / np, fl: nlo / np, ee: nx ? ee : NaN, sp: ratios.length ? ratios[ratios.length >> 1] : NaN, r: sim.residual, done, D2: Float64Array.from(D2), W: Float64Array.from(W) };
    // MDS of the largest component (>= 3 members): true vs completed, 2-D for drawing, 3-D for the disparity
    const idx = co.comps[0].members;
    const n = idx.length;
    if (n >= 3) {
      const Dt = new Float64Array(n * n), Dc = new Float64Array(n * n);
      for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) { Dt[a * n + c] = D2[idx[a] * N + idx[c]]; Dc[a * n + c] = geo.Dsp[idx[a] * N + idx[c]] ** 2; }
      const mt = O.mds(Dt, n, 2), mc = O.mds(Dc, n, 2, mt.X);
      const live = [];
      for (let a = 0; a < n; a++) for (let c = a + 1; c < n; c++) { const w = W[idx[a] * N + idx[c]]; if (w > 1e-9) live.push(a, c, w); }
      f.mds = { idx, X: mt.X, Y: mc.X, n, bar: Math.sqrt(2 * b), live };
      if (n >= 4) f.dis = R.procrustesDisparity(O.mds(Dt, n, 3).X, O.mds(Dc, n, 3).X, n, 3);
    }
    if (done) {
      // evaluation bank: sampling error of the live entries and of the lower bound
      const Ue = O.equilibrate(sim.net, Ge, S_EVAL).U, De2 = O.dist2(Ue, S_EVAL, N);
      let ev = 0, lbOk = 0, lbN = 0;
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        const st = geo.status[i * N + j], De = Math.sqrt(Math.max(De2[i * N + j], 0));
        if (st === 0) ev = Math.max(ev, Math.abs(geo.Dw[i * N + j] - De) / Math.max(De, 1e-300));
        if (st === 2) { lbN++; if (De >= Math.sqrt(2 * b)) lbOk++; }
      }
      f.ev = nx ? ev : NaN; f.lb = lbN ? lbOk / lbN : NaN;
      if (f.mds && n >= 4) {
        const De = new Float64Array(n * n), Dc = new Float64Array(n * n);
        for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) { De[a * n + c] = De2[idx[a] * N + idx[c]]; Dc[a * n + c] = geo.Dsp[idx[a] * N + idx[c]] ** 2; }
        f.disE = R.procrustesDisparity(O.mds(De, n, 3).X, O.mds(Dc, n, 3).X, n, 3);
      }
    }
    return f;
  }
  function onTick(speed) {
    sim.stepFor(14, speed, STOP); tick++;
    const done = sim.lastChange < STOP;
    if (!done && tick % 2) return null;
    const f = analyse(done); show(f);
    if (done) { f.done = true; f.doneLabel = sim.residual < RES_TOL ? "stationary: r < 10⁻⁶" : "stopped"; }
    return f;
  }
  function restart() { build(); const f = analyse(false); show(f); return f; }
  function retrain() { writeURL(); tp.reset(); tp.play(); }
  function show(f) {
    cur = f;
    const all = [...f.pts.x[0], ...f.pts.x[1], ...f.pts.x[2], ...f.pts.y[1].filter(Number.isFinite)];
    const mx = Math.max(Math.sqrt(2 * f.b) * 1.2, ...all) * 1.04;
    sc.update({ xrange: [0, mx], yrange: [0, mx], series: [
      { name: "live edge: √(2(b − αw))", x: f.pts.x[0], y: f.pts.y[0], color: seqColor(0.92), points: true, r: 2.6 },
      { name: "same component: shortest path (upper bound)", x: f.pts.x[1], y: f.pts.y[1], color: seqColor(0.5), points: true, r: 2.6 },
      { name: "different components: √(2b) (lower bound)", x: f.pts.x[2], y: f.pts.y[2], color: seqColor(0.22), points: true, r: 2 },
      { name: "y = x", x: [0, mx], y: [0, mx], color: P.mute, dash: true, width: 1 },
    ] });
    md.update(f.mds || null);
    // neuron space: colour by component (largest first), live synapses and attachments, width ∝ conductance
    const rankOf = new Int32Array(N); f.co.comps.forEach((c, k) => c.members.forEach(i => { rankOf[i] = k; }));
    nv.setColors(Array.from({ length: N }, (_, i) => nodeColor(rankOf[i], f.co.comps[rankOf[i]].size)));
    { let e = 0, amax = 0; for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) EW[e++] = f.W[i * N + j] > 1e-9 ? f.W[i * N + j] : 0;
      for (let i = 0; i < N * B; i++) { const a = sim.net.A[i]; EW[e++] = a; if (a > amax) amax = a; }
      nv.setEdges(PAIRS, EW, { wmax: amax }); }
    stats.set("k", f.co.K, `largest component ${f.co.largest} of ${N}`);
    stats.set("fx", f.fx, `${Math.round(f.fx * N * (N - 1) / 2)} of ${N * (N - 1) / 2} pairs`);
    stats.set("fu", f.fu, "shortest path above, √(2b) below");
    stats.set("fl", f.fl, f.lb != null && Number.isFinite(f.lb) ? `evaluation bank obeys $D \\ge \\sqrt{2b}$: ${ui.fmtPct(f.lb, 1)}` : "only $D \\ge \\sqrt{2b}$");
    stats.set("ee", f.ee, `plasticity residual $r$ = ${ui.fmt(f.r, 2)}`);
    stats.set("ev", f.ev ?? NaN, f.done ? `${S_EVAL} independent inputs` : "computed at the fixed point");
    stats.set("sp", f.sp, "median, inside components");
    stats.set("dis", f.dis ?? NaN, f.disE != null ? `evaluation bank: ${ui.fmt(f.disE, 2)}` : f.mds ? "training bank" : "largest component too small");
    ui.renderMath(stats.el);
    const ord = f.co.order, V = new Float32Array(N * N);
    for (let a = 0; a < N; a++) for (let c = 0; c < N; c++) V[a * N + c] = a === c ? NaN : f.geo.status[ord[a] * N + ord[c]];
    hSt.update({ values: V, rows: N, cols: N });
  }
  function stTip(a, c, v) {
    if (!cur || a === c) return null;
    const i = cur.co.order[a], j = cur.co.order[c], g = cur.geo, k = i * N + j;
    const D = Math.sqrt(Math.max(cur.D2[k], 0)), lb = Math.sqrt(2 * cur.b);
    const rows = [{ name: "true D (training bank)", value: ui.fmt(D, 3) }];
    if (v === 0) rows.push({ name: "from the weight √(2(b − αw))", value: ui.fmt(g.Dw[k], 3) });
    else if (v === 1) rows.push({ name: "lower bound √(2b)", value: ui.fmt(lb, 3) }, { name: "upper bound (shortest path)", value: ui.fmt(g.Dsp[k], 3) });
    else rows.push({ name: "lower bound √(2b)", value: ui.fmt(lb, 3) });
    return ui.tipRows(`neurons ${i} and ${j}`, rows);
  }
  function moved() { if (!tourActive) return; tourActive = null; tours.moved(); }
  tp.record(restart());
  function writeURL() {
    ctx.setParams({ ntour: tourActive ? tourActive.id : null, nb: tourActive || p.nb === NL_DEF.nb ? null : p.nb, nseed: tourActive || p.nseed === NL_DEF.nseed ? null : p.nseed });
  }
  return {
    fromURL() {
      const t = NL_TOURS.find(x => x.id === ctx.param("ntour", ""));
      const q = t ? { ...NL_DEF, ...t.params } : { nb: ctx.param("nb", NL_DEF.nb), nseed: ctx.param("nseed", NL_DEF.nseed) };
      if (q.nb === p.nb && q.nseed === p.nseed) return;
      tourActive = t || null; if (t) tours.set(t.id); else tours.clear();
      p = q; bS.set(p.nb); seedS.set(p.nseed); retrain();
    },
  };
}

// small canvas: two 2-D configurations (dots and rings) joined by segments, equal aspect. Response-function space:
// no synapse lines unless setSynapses(true) (live edges between the true positions, opacity ∝ w / max w).
// onHover(global neuron index | null); setHighlight(i | [i] | null) rings neurons without calling onHover.
function mdsChart(el, { height = 300, onHover = null } = {}) {
  el.classList.add("chart");
  const cv = ui.h("canvas"); el.replaceChildren(cv);
  let d = null, W = 0, proj = null, syn = false, hl = new Set(), hov = null;
  function draw() {
    W = Math.floor(el.clientWidth); if (!W) return;
    const g = hidpi(cv, W, height); g.clearRect(0, 0, W, height);
    g.font = "11px Inter, system-ui, sans-serif"; g.fillStyle = P.mute;
    if (!d) { g.textAlign = "center"; g.fillText("largest component has fewer than 3 neurons", W / 2, height / 2); proj = null; return; }
    const { X, Y, n } = d;
    let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (const M of [X, Y]) for (let i = 0; i < n; i++) for (let k = 0; k < 2; k++) { lo[k] = Math.min(lo[k], M[i * 2 + k]); hi[k] = Math.max(hi[k], M[i * 2 + k]); }
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], 1e-9) * 1.1, cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2;
    const m = 14, side = Math.min(W - 2 * m, height - 2 * m - 16);
    const sx = scale("linear", cx - span / 2, cx + span / 2, (W - side) / 2, (W + side) / 2), sy = scale("linear", cy - span / 2, cy + span / 2, m + side, m);
    proj = { sx, sy };
    if (syn && d.live && d.live.length) {
      let wm = 0; for (let e = 2; e < d.live.length; e += 3) wm = Math.max(wm, d.live[e]);
      g.strokeStyle = P.edge; g.lineCap = "round";
      for (let e = 0; e < d.live.length; e += 3) {
        const a = d.live[e], c = d.live[e + 1], f = d.live[e + 2] / (wm || 1);
        g.globalAlpha = 0.15 + 0.6 * f; g.lineWidth = 0.5 + 2.2 * f;
        g.beginPath(); g.moveTo(sx(X[a * 2]), sy(X[a * 2 + 1])); g.lineTo(sx(X[c * 2]), sy(X[c * 2 + 1])); g.stroke();
      }
      g.globalAlpha = 1;
    }
    g.strokeStyle = "rgba(141,140,134,0.6)"; g.lineWidth = 1;
    for (let i = 0; i < n; i++) { g.beginPath(); g.moveTo(sx(X[i * 2]), sy(X[i * 2 + 1])); g.lineTo(sx(Y[i * 2]), sy(Y[i * 2 + 1])); g.stroke(); }
    g.strokeStyle = P.fg; g.lineWidth = 1.4;
    for (let i = 0; i < n; i++) { g.beginPath(); g.arc(sx(Y[i * 2]), sy(Y[i * 2 + 1]), 4.2, 0, 2 * Math.PI); g.stroke(); }
    g.fillStyle = catColor(0);
    for (let i = 0; i < n; i++) { g.beginPath(); g.arc(sx(X[i * 2]), sy(X[i * 2 + 1]), 3.2, 0, 2 * Math.PI); g.fill(); }
    // linked highlight: ring both positions of each highlighted neuron of this component
    g.strokeStyle = P.accent; g.lineWidth = 2;
    for (let i = 0; i < n; i++) if (hl.has(d.idx[i])) for (const M of [X, Y]) { g.beginPath(); g.arc(sx(M[i * 2]), sy(M[i * 2 + 1]), 7.5, 0, 2 * Math.PI); g.stroke(); }
    // scale bar: one edge range sqrt(2b)
    const x0 = W - m - (sx(cx + d.bar) - sx(cx)), x1 = W - m, yb = height - 8;
    g.strokeStyle = P.fg; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x0, yb); g.lineTo(x1, yb); g.moveTo(x0, yb - 3); g.lineTo(x0, yb + 3); g.moveTo(x1, yb - 3); g.lineTo(x1, yb + 3); g.stroke();
    g.fillStyle = P.mute; g.textAlign = "right"; g.fillText("√(2b)", x0 - 6, yb + 4);
    g.textAlign = "left"; g.fillText(`${n} neurons · equal aspect`, 6, height - 4);
  }
  const ro = new ResizeObserver(() => { if (Math.floor(el.clientWidth) !== W) draw(); }); ro.observe(el);
  cv.addEventListener("pointermove", ev => {
    if (!d || !proj) return;
    const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = -1, bd = 100;
    for (let i = 0; i < d.n; i++) for (const M of [d.X, d.Y]) { const dd = (proj.sx(M[i * 2]) - x) ** 2 + (proj.sy(M[i * 2 + 1]) - y) ** 2; if (dd < bd) { bd = dd; best = i; } }
    const gi = best < 0 ? null : d.idx[best];
    if (gi !== hov) { hov = gi; hl = new Set(gi == null ? [] : [gi]); draw(); onHover?.(gi); }
    if (best < 0) { ui.hideTip(); return; }
    const sh = Math.hypot(d.X[best * 2] - d.Y[best * 2], d.X[best * 2 + 1] - d.Y[best * 2 + 1]);
    ui.showTip(ev.clientX, ev.clientY, ui.tipRows(`neuron ${d.idx[best]}`, [
      { name: "true position", value: `(${ui.fmt(d.X[best * 2], 2)}, ${ui.fmt(d.X[best * 2 + 1], 2)})`, color: catColor(0), shape: "dot" },
      { name: "from the weights", value: `(${ui.fmt(d.Y[best * 2], 2)}, ${ui.fmt(d.Y[best * 2 + 1], 2)})`, color: P.fg, shape: "dot" },
      { name: "displacement", value: ui.fmt(sh, 2) }]));
  });
  cv.addEventListener("pointerleave", () => { ui.hideTip(); if (hov != null) { hov = null; hl = new Set(); draw(); onHover?.(null); } });
  return {
    update(x) { d = x; draw(); },
    setSynapses(on) { syn = !!on; draw(); },
    setHighlight(i) { hl = new Set(i == null ? [] : Array.isArray(i) ? i : [i]); draw(); },
  };
}

// =====================================================================================================================
// (5) nonlinear sweep table
export function nlTable(ctx, data) {
  const el = ctx.slot("nltable"); if (!el) return;
  const bs = [...new Set(data.nl.map(r => r.b))].sort((a, b) => a - b);
  const mean = (rs, k) => rs.reduce((a, r) => a + r[k], 0) / rs.length;
  const rows = bs.map(b => {
    const rs = data.nl.filter(r => r.b === b);
    return `<tr><td class="num">${ui.fmt(b, 2)}</td><td class="num" style="white-space:nowrap">${fmtRange(rng(rs.map(r => r.K)), x => String(x))}</td>` +
      `<td class="num" style="white-space:nowrap">${ui.fmtFixed(mean(rs, "frac_exact"), 3)}</td><td class="num">${ui.fmtFixed(mean(rs, "frac_upper_bounded"), 3)}</td><td class="num">${ui.fmtFixed(mean(rs, "frac_lower_only"), 3)}</td>` +
      `<td class="num" style="white-space:nowrap">${fmtRange(rng(rs.map(r => r.sp_ratio_median)), x => ui.fmtFixed(x, 2))}</td>` +
      `<td class="num" style="white-space:nowrap">${fmtRange(rng(rs.map(r => r.edge_rel_err_eval)), x => ui.fmtFixed(x, 3))}</td>` +
      `<td class="num" style="white-space:nowrap">${fmtRange(rng(rs.map(r => r.lower_bound_holds_eval_frac)), x => ui.fmtFixed(x, 3))}</td>` +
      `<td class="num" style="white-space:nowrap">${fmtRange(rng(rs.map(r => r.mds_procrustes_disparity)), x => ui.fmtFixed(x, 3))}</td>` +
      `<td class="num" style="white-space:nowrap">${fmtRange(rng(rs.map(r => r.mds_procrustes_disparity_eval)), x => ui.fmtFixed(x, 3))}</td></tr>`;
  }).join("");
  const conv = data.nl.every(r => r.converged), edgeMax = Math.max(...data.nl.map(r => r.edge_err));
  el.innerHTML = `<div class="table-scroll"><table class="data"><thead><tr><th>$b$</th><th>$K$</th><th>exact</th><th>both bounds</th><th>lower bound only</th><th>shortest path / $D$, median</th><th>live entry vs eval., max rel. error</th><th>lower bound holds, eval.</th><th>MDS disparity, train</th><th>MDS disparity, eval.</th></tr></thead><tbody>${rows}</tbody></table></div>` +
    `<p class="u-help">From <code>data/sweeps2/readout.json</code> (<code>nl</code>): $N = 40$, blocks, rectifying ($k_{lo} = 3$, $k_{hi} = 0$, $\\sigma_t = 0.7$), seeds 0–2; fractions are means over seeds, other columns min–max. ${conv ? "All runs converged" : "Not all runs converged"}; live entries match the training bank to ${ui.fmt(edgeMax, 2)}.</p>`;
  ui.renderMath(el);
}
