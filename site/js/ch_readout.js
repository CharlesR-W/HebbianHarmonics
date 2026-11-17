// Chapter "readout" (Q-readout, SPEC §3; findings G21–G23). Four figures:
//   live    a linear network trained in the page, P C_hat P read back from W and A (estimator in ch_readout_math.js)
//   sweep   readout.json: rank, cosines with baselines, weight noise, the three errors, across b (N = 40 / 80)
//   leak    readout.json leak arms: uniform vs heterogeneous quadratic leak
//   nl      a rectifying network trained in the page: the partial distance geometry the weights determine
import * as ui from "./ui.js";
import { heatmap, lineChart, hidpi, scale, niceTicks } from "./charts.js";
import { P, catColor, cmap, rgbToCss } from "./palette.js";
import * as O from "./onsite_engine.js";
import * as R from "./ch_readout_math.js";

const B = 12, NC = B - 1, ALPHA = 1, DT = 0.5, S_TRAIN = 256, S_EVAL = 4096, STOP = 1e-9, RES_TOL = 1e-6;
const SUBS = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉", "₁₀", "₁₁", "₁₂"];
const GLAB = SUBS.map(s => "g" + s);
const NOISES = [0, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2];
const fmtEps = e => (e === 0 ? "0" : e >= 1e-3 ? `${+(e * 100).toPrecision(2)}%` : ui.fmt(e, 1));
const SIGMAS = {
  blocks: O.covBlocks({ sizes: [3, 3, 3, 3], within: 0.8, between: 0, variances: [2, 1, 0.5, 0.25] }).Sigma,
  power: O.covPowerLaw({ B, alpha: 1, seed: 12345, normalize: true }),
};
const Q = R.contrastBasis(B), IDX = R.svecIndex(NC), DIM = IDX.p;
const fmt3 = x => (Number.isFinite(x) ? ui.fmtFixed(x, 3) : "n/a");
const seqColor = t => rgbToCss(cmap("seq")(t));
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
// log-axis labels as powers of ten, only every `every`-th decade (the grid keeps every decade)
const pow10 = (every = 1) => v => {
  const e = Math.round(Math.log10(v));
  if (Math.abs(v / 10 ** e - 1) > 1e-6) return ui.fmt(v, 2);
  if (e % every) return "";
  return e === 0 ? "1" : e === 1 ? "10" : "10" + String(e).split("").map(c => SUP[c]).join("");
};

export async function init(panel, ctx) {
  const dataP = fetch("data/sweeps2/readout.json").then(r => { if (!r.ok) throw new Error(`readout.json: ${r.status}`); return r.json(); });
  const live = liveFigure(ctx);
  const nl = nlFigure(ctx);
  const data = await dataP;
  sweepFigure(ctx, data);
  leakFigure(ctx, data);
  nlTable(ctx, data);
  ctx.onParams(() => { live.fromURL(); nl.fromURL(); });
}

// =====================================================================================================================
// helpers
function chartBox(parent, title, sub) {
  const body = ui.h("div");
  const box = ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", html: title }), sub ? ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" }, html: sub }) : null, body);
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

// =====================================================================================================================
// (1) live linear readout
const LIVE_DEF = { b: 0.05, eps: 0, n: 80, m: "ridge", s: "blocks", seed: 1 };
const LIVE_TOURS = [
  { id: "few", label: "Too few live edges", params: { b: 0.0069, eps: 0, n: 80, m: "ridge", s: "blocks", seed: 1 },
    note: "$N = 80$, $b = 0.0069$: a handful of live edges. Rank = number of live edges; compare the cosine with the identity baseline beneath it, and switch the method to the hinge." },
  { id: "sweet", label: "Just above percolation", params: { b: 0.05, eps: 0.01, n: 80, m: "ridge", s: "blocks", seed: 1 },
    note: "$N = 80$, $b = 0.05$, 1% weight noise: full rank while the network is still in many components, and the readout survives the noise." },
  { id: "fused", label: "Fully fused", params: { b: 0.44, eps: 0, n: 80, m: "ridge", s: "blocks", seed: 1 },
    note: "$N = 80$, $b = 0.44$: one component, every pair live, no noise. The readout is exact; note the noise gain." },
  { id: "fragile", label: "Fused, 1% noise", params: { b: 0.44, eps: 0.01, n: 80, m: "ridge", s: "blocks", seed: 1 },
    note: "The same network with every weight multiplied by $1 + 0.01\\,\\xi$: the readout, rebuilt from the noisy weights, collapses." },
  { id: "n40", label: "Half the neurons", params: { b: 0.05, eps: 0, n: 40, m: "ridge", s: "blocks", seed: 1 },
    note: "$N = 40$ at the sweet-spot $b = 0.05$: fewer live edges than the 66 the contrast space needs." },
];

function liveFigure(ctx) {
  const slot = ctx.slot("live"); slot.replaceChildren();
  const fig = slot.closest("figure");
  let p = readLive(ctx), tourActive = LIVE_TOURS.find(t => t.id === ctx.param("tour", "")) || null;
  if (tourActive) p = { ...LIVE_DEF, ...tourActive.params };
  let sim = null, G = null, Chat = null, Sigma = null, N = 0, trace = [], cur = null, tick = 0;
  let hmRange = 1;

  const L = ui.simLayout({ parent: slot });
  // (a)-(d) four matrices
  const mats = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(215px, 100%), 1fr))", gap: "12px 18px", marginBottom: "10px" } });
  L.view.append(mats);
  const mk = (title, sub) => chartBox(mats, title, sub);
  const hmOpts = (z) => ({ map: "div", range: [-1, 1], size: 225, rowLabels: GLAB, colLabels: GLAB, zlabel: z });
  const hTarget = heatmap(mk("(a) target $P\\hat C P$", "training bank, what the weights encode"), { ...hmOpts("$P\\hat C P$"), tooltip: (i, j, v) => matTip(i, j, v, "target") });
  const hRec = heatmap(mk("(b) read from the weights", "recovery after PSD projection"), { ...hmOpts("recovered"), tooltip: (i, j, v) => matTip(i, j, v, "rec") });
  const hErr = heatmap(mk("(c) recovery − target", "same colour range"), { ...hmOpts("error"), tooltip: (i, j, v) => matTip(i, j, v, "err") });
  const hPop = heatmap(mk("(d) population $P\\Sigma_g P$", "what a perfect sample would give"), { ...hmOpts("$P\\Sigma_g P$"), tooltip: (i, j, v) => matTip(i, j, v, "pop") });
  const tp = ui.transport({
    parent: L.view, watch: fig, autoplay: true, maxHistory: 300,
    speeds: [{ value: 1, label: "×1" }, { value: 4, label: "×4" }, { value: 16, label: "×16" }], speed: 1,
    timeFmt: t => `step ${t}`,
    onTick: sp => onTick(sp), onSeek: f => show(f), onReset: () => restart(),
  });

  // side: tours, controls, tiles
  const tours = ui.chips({ parent: L.side, label: "Tours", items: LIVE_TOURS, active: tourActive ? tourActive.id : null, onPick: t => applyTour(t) });
  const bS = ui.slider({ label: "fusion scale $b$", min: 0.003, max: 1, log: true, sig: 2, value: p.b, fmt: x => ui.fmt(x, 2),
    marks: [{ value: 0.0069, label: "few" }, { value: 0.05, label: "sweet" }, { value: 0.44, label: "fused" }],
    help: "Edges live iff $D^2 &lt; 2b$. Retrains from $W = 0$.", onChange: x => { p.b = x; moved(); retrain(); } });
  const epsS = ui.slider({ label: "weight noise $\\varepsilon$", min: 0, max: NOISES.length - 1, step: 1, value: NOISES.indexOf(p.eps) >= 0 ? NOISES.indexOf(p.eps) : 0,
    fmt: k => fmtEps(NOISES[k]), help: "Every weight times $1 + \\varepsilon\\xi$; the readout is rebuilt from the noisy weights.",
    onInput: k => { p.eps = NOISES[k]; moved(); reread(); } });
  const nS = ui.segmented({ label: "interior neurons $N$", options: [{ value: 40, label: "40" }, { value: 80, label: "80" }], value: p.n, onChange: v => { p.n = v; moved(); retrain(); } });
  const mS = ui.segmented({ label: "method", options: [{ value: "ridge", label: "ridge" }, { value: "hinge", label: "ridge + dead-pair hinge" }], value: p.m,
    onChange: v => { p.m = v; moved(); reread(); } });
  const sS = ui.segmented({ label: "inputs $\\Sigma_g$", options: [{ value: "blocks", label: "blocks" }, { value: "power", label: "power law" }], value: p.s, onChange: v => { p.s = v; moved(); retrain(); } });
  const seedS = ui.slider({ label: "seed (attachments, training bank)", min: 0, max: 9, step: 1, value: p.seed, fmt: x => String(x), onChange: x => { p.seed = x; moved(); retrain(); } });
  ui.controls({ parent: L.side, title: "Network and readout", primary: [bS, epsS, nS, mS], advanced: [sS, seedS] });
  const stats = ui.stats({ parent: L.side, items: [
    { key: "m", label: "live edges", fmt: x => ui.fmtInt(x), help: "Pairs with w > 1e-9: each is one equation." },
    { key: "rank", label: "rank of $\\mathcal M$", fmt: x => `${x} / ${DIM}`, help: "Singular values above 1e-8 of the largest; 66 = dimension of the symmetric contrast space." },
    { key: "cos", label: "cos vs $P\\hat C P$", fmt: fmt3, help: "Frobenius cosine of the (PSD-projected) recovery with the training-bank target." },
    { key: "cosS", label: "cos vs $P\\Sigma_g P$", fmt: fmt3, help: "Against the population matrix: limited by sampling." },
    { key: "fp", label: "fixed-point error", fmt: x => ui.fmt(x, 2), help: "max over live edges |2(b − αw) − dᵀĈd| = 2 max|dw/dt| on live edges (units of D²)." },
    { key: "samp", label: "sampling error", fmt: x => ui.fmt(x, 3), help: "‖PĈP − PΣP‖/‖PΣP‖: what no readout can remove." },
    { key: "inv", label: "inverse-problem error", fmt: x => ui.fmt(x, 2), help: "Relative error of the ridge recovery on the identifiable subspace (singular values > 1e-4 of the largest), with the noisy weights if noise is on." },
    { key: "gain", label: "noise gain $2\\alpha w/D^2$", fmt: x => ui.fmt(x, 3), help: "Median over live edges: relative error of a stored D² per relative weight error." },
  ] });

  // below: spectra, singular values, training trace, weights
  // the four linked charts sit under the matrices on wide screens and after the controls on phones, so the
  // tours and sliders stay next to the view they drive
  const ch4 = grid2(400); ch4.style.marginTop = "18px";
  const oneCol = window.matchMedia("(max-width: 900px)");
  const placeCh4 = () => (oneCol.matches ? L.below : L.view).append(ch4);
  const bSpec = chartBox(ch4, "(e) Eigenvalues on the contrast space", "decreasing order");
  const bSv = chartBox(ch4, "(f) Singular values of the live-edge operator $\\mathcal M$", "log scale; dashed: rank tolerances");
  const bTr = chartBox(ch4, "(g) Training: readout and fixed-point error", "log scale, against the Euler step");
  const bW = chartBox(ch4, "(h) Weights $w_{ij}$", "sorted by component; ink: 0 to $b/\\alpha$");
  const spec = lineChart(bSpec, { series: [{ name: "x", y: [] }], height: 210, xlabel: "index k", ylabel: "eigenvalue", xname: "k", xfmt: v => (Number.isInteger(v) ? String(v) : "") });
  const svc = lineChart(bSv, { series: [{ name: "x", y: [] }], height: 210, ylog: true, xlabel: "index", ylabel: "singular value", xname: "index", yfmt: pow10(3) });
  const trc = lineChart(bTr, { series: [{ name: "x", y: [] }], height: 210, ylog: true, xlabel: "step", ylabel: "value", xname: "step", yfmt: pow10(4) });
  const hW = heatmap(bW, { map: "ink", range: [0, 1], size: 250, xlabel: "neuron j", ylabel: "neuron i", zlabel: "$w_{ij}$", tooltip: (i, j, v) => wTip(i, j, v) });
  placeCh4(); oneCol.addEventListener("change", placeCh4);

  // ---------------------------------------------------------------- simulation
  function build() {
    N = p.n; Sigma = SIGMAS[p.s];
    const net = O.makeNet({ N, B, rng: O.makeRng(p.seed), s: 3 });
    G = O.sampleInputs(Sigma, B, S_TRAIN, O.makeRng(1000 + p.seed));
    Chat = chatOf(G, S_TRAIN);
    sim = O.createSim({ net, G, S: S_TRAIN, b: p.b, alpha: ALPHA, dt: DT });
    const T1 = R.fromContrast(R.toContrast(Chat, Q, B, NC), Q, B, NC), T2 = R.fromContrast(R.toContrast(Sigma, Q, B, NC), Q, B, NC);
    let mx = 0; for (let k = 0; k < B * B; k++) mx = Math.max(mx, Math.abs(T1[k]), Math.abs(T2[k]));
    hmRange = +(1.05 * mx).toPrecision(2);
    trace = []; tick = 0;
  }
  function readNow(withHinge, W = sim.net.W) {
    const t0 = performance.now();
    const ro = R.readout({ W, A: sim.net.A, N, B, b: p.b, alpha: ALPHA, Chat, Sigma, Q, idx: IDX, noise: p.eps, hinge: withHinge });
    ro.ms = performance.now() - t0;
    ro.hasHinge = withHinge;
    return ro;
  }
  function frameOf(ro, done) {
    const co = componentOrder(sim.net.W, N, sim.U, S_TRAIN);
    const Wf = new Float32Array(N * N);
    for (let a = 0; a < N; a++) for (let c = 0; c < N; c++) Wf[a * N + c] = sim.net.W[co.order[a] * N + co.order[c]];
    return { t: sim.steps, ro, co, Wf, W: Float64Array.from(sim.net.W), N, b: p.b, r: sim.residual, eq: sim.eqResidual, done };
  }
  function pushTrace(f) {
    const sc = f.ro.sc.ridge.Chat;
    trace.push({ t: f.t, oneMinus: f.ro.m ? Math.max(1 - sc.cos, 1e-16) : NaN, fp: f.ro.m ? Math.max(f.ro.errFixedPoint, 1e-18) : NaN, r: Math.max(f.r, 1e-18) });
  }
  function onTick(speed) {
    sim.stepFor(14, speed, STOP); tick++;
    const done = sim.lastChange < STOP;
    if (!done && tick % 3 !== 0) return null;
    const f = frameOf(readNow(done), done);
    pushTrace(f); cur = f; show(f);
    if (done) { f.done = true; f.doneLabel = sim.residual < RES_TOL ? "stationary: r < 10⁻⁶" : "stopped"; }
    return f;
  }
  function restart() {
    build();
    const f = frameOf(readNow(false), false); pushTrace(f); cur = f; show(f);
    return f;
  }
  function retrain() { writeURL(); tp.reset(); tp.play(); }
  function reread() {
    writeURL();
    if (!cur) return;
    // re-read the frame on display (which may be a scrubbed-back one), from its own stored weights
    cur.ro = readNow(cur.done || p.m === "hinge", cur.W);
    show(cur);
  }

  // ---------------------------------------------------------------- drawing
  function show(f) {
    cur = f;
    const ro = f.ro, useH = p.m === "hinge" && ro.hasHinge;
    const Y = useH ? ro.Yh : ro.Yr, sc = useH ? ro.sc.hinge : ro.sc.ridge;
    const T1 = R.fromContrast(ro.T1, Q, B, NC), T2 = R.fromContrast(ro.T2, Q, B, NC), Rc = R.fromContrast(Y, Q, B, NC);
    const E = Rc.map((v, k) => v - T1[k]);
    const rg = [-hmRange, hmRange];
    hTarget.update({ values: Float32Array.from(T1), rows: B, cols: B, range: rg });
    hRec.update({ values: Float32Array.from(Rc), rows: B, cols: B, range: rg });
    hErr.update({ values: Float32Array.from(E), rows: B, cols: B, range: rg });
    hPop.update({ values: Float32Array.from(T2), rows: B, cols: B, range: rg });
    f._mats = { T1, T2, Rc, E };
    // tiles
    const co = f.co;
    stats.set("m", ro.m, `of ${ui.fmtInt(ro.nPairs)} pairs · $K = ${co.K}$ components, largest ${co.largest}${co.diamMax > 0 ? `, max member-to-mean $D$ ${ui.fmt(co.diamMax, 2)}` : ""}`);
    stats.set("rank", ro.rank8, ro.m ? `${ro.rank8 === DIM ? "full rank" : `${DIM - ro.rank8} directions unmeasured`} · ${ro.rank4} above $10^{-4}\\sigma_{\\max}$ · target share there ${ui.fmtPct(ro.identFrac, 1)}` : "no live edge: nothing to read");
    const methodNote = p.m === "hinge" && !ro.hasHinge ? " (hinge once converged; ridge shown)" : "";
    stats.set("cos", ro.m ? sc.Chat.cos : NaN, ro.m ? `identity baseline ${fmt3(ro.sc.identity.Chat.cos)} · diagonal ${fmt3(ro.sc.diag.Chat.cos)}${methodNote}` : "");
    stats.set("cosS", ro.m ? sc.Sigma.cos : NaN, ro.m ? `Bures–Wasserstein / $\\sqrt{\\operatorname{tr}}$: ${ui.fmt(sc.Sigma.bures, 2)} (vs $P\\hat CP$: ${ui.fmt(sc.Chat.bures, 2)})` : "");
    stats.set("fp", ro.m ? ro.errFixedPoint : NaN, `plasticity residual $r$ = ${ui.fmt(f.r, 2)} · neuronal residual ${ui.fmt(f.eq, 2)}`);
    stats.set("samp", ro.errSampling, `$S = ${S_TRAIN}$ training inputs`);
    stats.set("inv", ro.m ? ro.errInverse : NaN, ro.m ? (ro.identFrac < 0.999 ? `plus ${ui.fmtPct(1 - ro.identFrac, 1)} of $\\lVert P\\hat CP\\rVert^2$ outside it` : "nothing of the target outside it") : "");
    stats.set("gain", ro.m ? ro.gainMedian : NaN, p.eps > 0 ? `× ε = ${fmtEps(p.eps)} → typical relative error of a stored $D^2$: ${ui.fmt(ro.gainMedian * p.eps, 2)}` : "set the weight noise to see its effect");
    ui.renderMath(stats.el);
    // spectra
    const ks = Array.from({ length: NC }, (_, k) => k + 1);
    const ev = M => R.eigvalsDesc(M, NC);
    spec.update({ x: ks, series: [
      { name: "target P Ĉ P", y: ev(ro.T1), color: P.fg, dots: true },
      { name: "population P Σ P", y: ev(ro.T2), color: P.mute, dash: true },
      { name: `recovered (${useH ? "hinge" : "ridge"})`, y: ev(Y), color: catColor(0), points: true, r: 4 },
      { name: "identity baseline cI", y: ev(ro.Yi), color: catColor(2), dash: true },
    ] });
    // singular values
    const sv = ro.sv.filter(x => x > 0);
    if (sv.length) {
      const top = sv[0];
      svc.update({ series: [{ name: "σ", y: sv, x: sv.map((_, k) => k + 1), color: catColor(0), points: true, r: 2.5 }], xrange: [0, DIM + 1],
        yrange: [top * 1e-13, top * 3], refs: [{ y: 1e-8 * top, label: "10⁻⁸ σmax" }, { y: 1e-4 * top, label: "10⁻⁴ σmax" }, { x: DIM, label: "66" }] });
    } else svc.update({ series: [{ name: "σ", y: [1], x: [1], points: true, r: 0 }], xrange: [0, DIM + 1], yrange: [1e-3, 10], refs: [{ x: DIM, label: "66 · no live edge" }] });
    // trace
    if (trace.length) {
      const xs = trace.map(d => d.t);
      trc.update({ series: [
        { name: "1 − cos vs P Ĉ P (ridge)", x: xs, y: trace.map(d => d.oneMinus), color: catColor(0) },
        { name: "fixed-point error", x: xs, y: trace.map(d => d.fp), color: catColor(1) },
        { name: "plasticity residual r", x: xs, y: trace.map(d => d.r), color: P.mute, dash: true },
      ], yrange: [1e-16, 10], refs: [{ y: RES_TOL, label: "10⁻⁶" }] });
      trc.setCursor(f.t);
    }
    // weights
    hW.update({ values: f.Wf, rows: f.N, cols: f.N, range: [0, f.b / ALPHA], groups: f.co.groups.length && f.co.groups.length <= 14 ? { rows: f.co.groups.map(g => ({ ...g, color: P.ink2 })) } : null });
  }
  function matTip(i, j, v, which) {
    const m = cur && cur._mats; if (!m) return null;
    const k = i * B + j;
    return ui.tipRows(`inputs ${GLAB[i]}, ${GLAB[j]}`, [
      { name: "target (PĈP)", value: ui.fmt(m.T1[k], 3) }, { name: "recovered", value: ui.fmt(m.Rc[k], 3) },
      { name: "recovered − target", value: ui.fmt(m.E[k], 2) }, { name: "population (PΣP)", value: ui.fmt(m.T2[k], 3) }]);
  }
  function wTip(a, c, v) {
    if (!cur) return null;
    const i = cur.co.order[a], j = cur.co.order[c];
    const rows = [{ name: "w", value: ui.fmt(v, 3) }];
    if (v > 1e-9) rows.push({ name: "stored D² = 2(b − αw)", value: ui.fmt(2 * (cur.b - ALPHA * v), 3) });
    else rows.push({ name: "dead: D² ≥ 2b", value: ui.fmt(2 * cur.b, 3) });
    rows.push({ name: "components", value: `${cur.co.labels[i] === cur.co.labels[j] ? "same" : "different"}` });
    return ui.tipRows(`neurons ${i} and ${j}`, rows);
  }

  // ---------------------------------------------------------------- state plumbing
  function syncControls() {
    bS.set(p.b); epsS.set(Math.max(0, NOISES.indexOf(p.eps))); nS.set(p.n); mS.set(p.m); sS.set(p.s); seedS.set(p.seed);
  }
  function moved() { if (!tourActive) return; tourActive = null; tours.moved(); }
  function writeURL() {
    const o = { tour: tourActive ? tourActive.id : null };
    for (const k of Object.keys(LIVE_DEF)) o[k] = tourActive || p[k] === LIVE_DEF[k] ? null : p[k];
    ctx.setParams(o);
  }
  function applyTour(t) { tourActive = t; p = { ...LIVE_DEF, ...t.params }; syncControls(); retrain(); }
  tp.record(restart());
  return {
    fromURL() {
      const t = LIVE_TOURS.find(x => x.id === ctx.param("tour", ""));
      if (t) { tours.set(t.id); applyTour(t); return; }
      const q = readLive(ctx), same = Object.keys(LIVE_DEF).every(k => q[k] === p[k]);
      if (same) return;
      tourActive = null; tours.clear(); p = q; syncControls(); retrain();
    },
  };
}
function readLive(ctx) {
  const d = LIVE_DEF, p = {};
  p.b = Math.min(1, Math.max(0.003, ctx.param("b", d.b)));
  const e = ctx.param("eps", d.eps); p.eps = NOISES.reduce((a, x) => (Math.abs(Math.log((x || 1e-9) / (e || 1e-9))) < Math.abs(Math.log((a || 1e-9) / (e || 1e-9))) ? x : a), 0);
  p.n = ctx.param("n", d.n) === 40 ? 40 : 80;
  p.m = ctx.param("m", d.m) === "hinge" ? "hinge" : "ridge";
  p.s = ctx.param("s", d.s) === "power" ? "power" : "blocks";
  p.seed = Math.max(0, Math.min(9, Math.round(ctx.param("seed", d.seed))));
  return p;
}

// =====================================================================================================================
// (2) the sweep
function runsOf(data, sigma, N, leak = "0") { return data.lin.filter(x => x.sigma === sigma && x.N === N && x.leak === leak); }
function agg(runs, get) {
  const nb = runs[0].runs.length, mean = [], lo = [], hi = [];
  for (let k = 0; k < nb; k++) {
    const v = runs.map(r => get(r.runs[k])).filter(x => x != null && Number.isFinite(x));
    if (!v.length) { mean.push(NaN); lo.push(NaN); hi.push(NaN); continue; }
    mean.push(v.reduce((a, b) => a + b, 0) / v.length); lo.push(Math.min(...v)); hi.push(Math.max(...v));
  }
  return { y: mean, band: [lo, hi] };
}
const cosOr0 = (r, key, tgt) => (r[`${key}_vs_${tgt}`] ? r[`${key}_vs_${tgt}`].cos : 0);
const rng = (a) => { const v = a.filter(Number.isFinite); return v.length ? [Math.min(...v), Math.max(...v)] : [NaN, NaN]; };
const fmtRange = ([a, b], f = x => ui.fmtFixed(x, 3)) => (!Number.isFinite(a) ? "n/a" : f(a) === f(b) ? f(a) : `${f(a)}–${f(b)}`);

function sweepFigure(ctx, data) {
  const slot = ctx.slot("sweep"); slot.replaceChildren();
  const bgrid = data.meta.config.b_grid;
  let st = { s: "blocks", n: 80, t: "Chat" };
  const ctl = ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "10px 24px", marginBottom: "12px" } });
  slot.append(ctl);
  ui.segmented({ parent: ctl, label: "inputs $\\Sigma_g$", options: [{ value: "blocks", label: "blocks" }, { value: "power", label: "power law" }], value: st.s, onChange: v => { st.s = v; draw(); } });
  ui.segmented({ parent: ctl, label: "$N$", options: [{ value: 40, label: "40" }, { value: 80, label: "80" }], value: st.n, onChange: v => { st.n = v; draw(); } });
  ui.segmented({ parent: ctl, label: "target in (b)", options: [{ value: "Chat", label: "$P\\hat C P$" }, { value: "Sigma", label: "$P\\Sigma_g P$" }], value: st.t, onChange: v => { st.t = v; draw(); } });
  const g = grid2(320); slot.append(g);
  const common = { x: bgrid, xlog: true, height: 230, xlabel: "fusion scale b", xname: "b" };
  const cA = lineChart(chartBox(g, "(a) Live edges, operator rank and components"), { ...common, series: [{ name: "x", y: [] }], ylog: true, ylabel: "count", yfmt: pow10(1) });
  const cB = lineChart(chartBox(g, "(b) Frobenius cosine with the target"), { ...common, series: [{ name: "x", y: [] }], yrange: [0, 1.02], ylabel: "cosine" });
  const cC = lineChart(chartBox(g, "(c) Weight noise: cosine with $P\\hat C P$ (ridge)"), { ...common, series: [{ name: "x", y: [] }], yrange: [0, 1.02], ylabel: "cosine" });
  const cD = lineChart(chartBox(g, "(d) The three errors"), { ...common, series: [{ name: "x", y: [] }], ylog: true, ylabel: "error", yfmt: pow10(3) });
  function draw() {
    const runs = runsOf(data, st.s, st.n), t = st.t;
    cA.update({ series: [
      { name: "live edges", ...agg(runs, r => r.n_live), color: catColor(0), dots: true },
      { name: "rank of M (tol 10⁻⁸)", ...agg(runs, r => r.rank_1e8), color: catColor(1), dots: true },
      { name: "components K", ...agg(runs, r => r.K), color: P.mute, dash: true },
    ], refs: [{ y: 66, label: "66" }], yrange: [0.7, 5000] });
    cB.update({ series: [
      { name: "ridge + PSD", ...agg(runs, r => cosOr0(r, "ridge_psd", t)), color: catColor(0), width: 2.5 },
      { name: "hinge + PSD", ...agg(runs, r => cosOr0(r, "hinge_psd", t)), color: catColor(1) },
      { name: "identity cI", ...agg(runs, r => cosOr0(r, "base_identity", t)), color: catColor(2), dash: true },
      { name: "diagonal (NNLS)", ...agg(runs, r => cosOr0(r, "base_diag", t)), color: catColor(3), dash: true },
      { name: "untrained network", ...agg(runs, r => cosOr0(r, "base_untrained", t)), color: catColor(4), dash: true },
      { name: "shuffled weights", ...agg(runs, r => cosOr0(r, "base_shuffled", t)), color: catColor(5), dash: true },
      { name: "identifiable share of ‖PĈP‖²", ...agg(runs, r => r.identifiable_frac), color: P.mute, dash: true, width: 1.5 },
    ] });
    cC.update({ series: [
      { name: "ε = 0", ...agg(runs, r => cosOr0(r, "ridge_psd", "Chat")), color: P.fg },
      { name: "ε = 10⁻⁶", ...agg(runs, r => r["noise1e-06_ridge_psd_vs_Chat_cos"] ?? 0), color: catColor(0) },
      { name: "ε = 10⁻⁴", ...agg(runs, r => r["noise0.0001_ridge_psd_vs_Chat_cos"] ?? 0), color: catColor(1) },
      { name: "ε = 10⁻²", ...agg(runs, r => r["noise0.01_ridge_psd_vs_Chat_cos"] ?? 0), color: catColor(3) },
    ] });
    cD.update({ series: [
      { name: "fixed point (abs., D² units)", ...agg(runs, r => (r.n_live ? Math.max(r.err_fixed_point, 1e-17) : null)), color: catColor(0), dots: true },
      { name: "sampling (rel.)", ...agg(runs, r => r.err_sampling), color: catColor(1), dots: true },
      { name: "inverse problem, identifiable subspace (rel.)", ...agg(runs, r => (r.n_live ? Math.max(r.err_inverse_identifiable, 1e-17) : null)), color: catColor(2), dots: true },
    ], yrange: [1e-12, 3] });
  }
  draw();
  sweepText(ctx, data);
}

function sweepText(ctx, data) {
  const el = ctx.slot("sweep-text"); if (!el) return;
  const bg = data.meta.config.b_grid, fb = b => ui.fmt(b, 3);
  // first b at which every seed has full rank, per (sigma, N)
  const firstFull = (sig, N) => {
    const runs = runsOf(data, sig, N);
    for (let k = 0; k < bg.length; k++) if (runs.every(r => r.runs[k].rank_1e8 === r.runs[k].dim)) return { k, b: bg[k], K: rng(runs.map(r => r.runs[k].K)), cos: rng(runs.map(r => r.runs[k].ridge_psd_vs_Chat.cos)), n: rng(runs.map(r => r.runs[k].n_live)) };
    return null;
  };
  const f80 = firstFull("blocks", 80), f40 = firstFull("blocks", 40), p80 = firstFull("power", 80), p40 = firstFull("power", 40);
  const all = data.lin.filter(x => x.leak === "0");
  const fullRuns = all.flatMap(x => x.runs.filter(r => r.rank_1e8 === r.dim).map(r => ({ ...r, sig: x.sigma })));
  const vsSig = sig => rng(fullRuns.filter(r => r.sig === sig).map(r => r.ridge_psd_vs_Sigma.cos));
  const samp = rng(all.flatMap(x => x.runs.map(r => r.err_sampling)));
  const fusedNoise = rng(all.flatMap(x => x.runs.filter(r => r.b >= 0.4).map(r => r["noise0.01_ridge_psd_vs_Chat_cos"])));
  const k83 = bg.findIndex(b => Math.abs(b - 0.0829) < 1e-3);
  const sweet = rng(runsOf(data, "blocks", 80).map(r => r.runs[k83]["noise0.01_ridge_psd_vs_Chat_cos"]));
  const sweetP = rng(runsOf(data, "power", 80).map(r => r.runs[k83]["noise0.01_ridge_psd_vs_Chat_cos"]));
  const lowRuns = all.flatMap(x => x.runs.filter(r => r.n_live > 0 && r.rank_1e8 < r.dim / 2));
  const lowBeat = lowRuns.filter(r => r.ridge_psd_vs_Chat.cos > r.base_identity_vs_Chat.cos).length;
  const lowHinge = lowRuns.filter(r => r.hinge_psd_vs_Chat.cos > r.base_identity_vs_Chat.cos).length;
  const fpMax = Math.max(...all.flatMap(x => x.runs.map(r => r.err_fixed_point)));
  const wellRuns = fullRuns.filter(r => r.rank_1e4 === r.dim), illRuns = fullRuns.filter(r => r.rank_1e4 < r.dim);
  const invWell = Math.max(...wellRuns.map(r => r.err_inverse_identifiable));
  const cosWell = rng(wellRuns.map(r => r.ridge_psd_vs_Chat.cos)), cosIll = rng(illRuns.map(r => r.ridge_psd_vs_Chat.cos));
  const rankDef = all.flatMap(x => x.runs.filter(r => r.n_live >= r.dim && r.rank_1e8 < r.dim).map(r => `${r.n_live} live edges, rank ${r.rank_1e8}`));
  const s = [];
  s.push(`Rank follows the live-edge count until it saturates at 66 (panel a)${rankDef.length ? `, though not always: ${rankDef.length === 1 ? "one run has" : `${rankDef.length} runs have`} ${rankDef.slice(0, 2).join(" and ")}` : ""}.`);
  if (f80 && f40) s.push(`On the block covariance every seed first reaches full rank at $b = ${fb(f80.b)}$ for $N = 80$, with the network still in ${fmtRange(f80.K, x => String(x))} components, but only at $b = ${fb(f40.b)}$ for $N = 40$, with ${fmtRange(f40.K, x => String(x))} components left.`);
  if (p80 && p40) s.push(`For the power-law covariance the same thresholds are $b = ${fb(p80.b)}$ (${fmtRange(p80.K, x => String(x))} components) and $b = ${fb(p40.b)}$.`);
  s.push(`Full rank makes $P\\hat C P$ identifiable; how much of it the ridge returns depends on conditioning. In the ${wellRuns.length} full-rank runs whose singular values all lie above $10^{-4}\\sigma_{\\max}$ the cosine with $P\\hat C P$ is ${fmtRange(cosWell)} (inverse-problem error at most ${ui.fmt(invWell, 2)}); in the ${illRuns.length} full-rank runs with weaker directions, which the ridge shrinks, it is ${fmtRange(cosIll)}. The fixed-point error is at most ${ui.fmt(fpMax, 2)} in every run. Against $P\\Sigma_g P$ the full-rank cosine is ${fmtRange(vsSig("blocks"))} (blocks) and ${fmtRange(vsSig("power"))} (power law), with a sampling error of ${fmtRange(samp)}.`);
  s.push(`Below half rank the ridge recovery beats the fitted identity in ${lowBeat} of ${lowRuns.length} runs and the sweep's hinge iterate in ${lowHinge}: with few live edges the weights say less than "the covariance is a multiple of the identity".`);
  s.push(`Under 1% weight noise (panel c) the fully fused networks ($b \\ge 0.44$) keep a cosine of only ${fmtRange(fusedNoise)}, while at $N = 80$, $b = ${fb(bg[k83])}$ it stays at ${fmtRange(sweet)} (blocks) and ${fmtRange(sweetP)} (power law).`);
  el.innerHTML = s.join(" ");
  ui.renderMath(el);
}

// =====================================================================================================================
// (3) quadratic leak
function leakFigure(ctx, data) {
  const slot = ctx.slot("leak"); slot.replaceChildren();
  const bgrid = data.meta.config.b_grid;
  let kind = "u";
  const ctl = ui.h("div", { style: { marginBottom: "12px" } }); slot.append(ctl);
  ui.segmented({ parent: ctl, label: "leak", options: [{ value: "u", label: "uniform $k_i = k$" }, { value: "h", label: "heterogeneous $k_i \\sim U(0, k)$" }], value: kind, onChange: v => { kind = v; draw(); } });
  const g = grid2(320); slot.append(g);
  const common = { x: bgrid, xlog: true, height: 220, xlabel: "fusion scale b", xname: "b" };
  const cR = lineChart(chartBox(g, "(a) Rank of the live-edge operator"), { ...common, series: [{ name: "x", y: [] }], ylabel: "rank", yrange: [0, 82] });
  const cC = lineChart(chartBox(g, "(b) Cosine: full $\\hat C$ (solid) and contrast part (dashed)"), { ...common, series: [{ name: "x", y: [] }], ylabel: "cosine", yrange: [0, 1.02] });
  const cO = lineChart(chartBox(g, "(c) $\\max |d^\\top \\mathbf 1|$ over pairs"), { ...common, series: [{ name: "x", y: [] }], ylog: true, ylabel: "max |dᵀ1|", yrange: [1e-17, 2], yfmt: pow10(4) });
  const cN = lineChart(chartBox(g, "(d) Cosine with $\\hat C$ under $10^{-4}$ weight noise"), { ...common, series: [{ name: "x", y: [] }], ylabel: "cosine", yrange: [0, 1.02] });
  function draw() {
    const arms = kind === "u" ? [["u0.1", "k = 0.1"], ["u1", "k = 1"]] : [["h0.2", "k = 0.2"], ["h1", "k = 1"]];
    const none = runsOf(data, "blocks", 40, "0");
    const A = arms.map(([leak, lab], i) => ({ leak, lab, runs: runsOf(data, "blocks", 40, leak), color: catColor(i) }));
    cR.update({ series: [...A.map(a => ({ name: a.lab, ...agg(a.runs, r => r.rank_1e8), color: a.color, dots: true })),
      { name: "no leak (contrast space)", ...agg(none, r => r.rank_1e8), color: P.mute, dash: true }], refs: [{ y: 66, label: "66" }, { y: 78, label: "78" }] });
    cC.update({ series: [...A.map(a => ({ name: `${a.lab}: full Ĉ`, ...agg(a.runs, r => cosOr0(r, "ridge_psd", "Chat")), color: a.color })),
      ...A.map(a => ({ name: `${a.lab}: contrast part`, ...agg(a.runs, r => r.ridge_psd_contrast_vs_Chat ? r.ridge_psd_contrast_vs_Chat.cos : 0), color: a.color, dash: true }))] });
    cO.update({ series: A.map(a => ({ name: a.lab, ...agg(a.runs, r => Math.max(r.ones_leak, 1e-17)), color: a.color, dots: true })) });
    cN.update({ series: A.map(a => ({ name: a.lab, ...agg(a.runs, r => r["noise0.0001_ridge_psd_vs_Chat_cos"] ?? 0), color: a.color })) });
  }
  draw();
  // text from the JSON
  const el = ctx.slot("leak-text"); if (!el) return;
  const pick = (leaks, f) => data.lin.filter(x => leaks.includes(x.leak)).flatMap(x => x.runs.filter(f));
  const uFull = pick(["u0.1", "u1"], r => r.b >= 0.19);
  const hFull = pick(["h0.2", "h1"], r => r.b >= 0.19);
  const uOnes = Math.max(...pick(["u0.1", "u1"], () => true).map(r => r.ones_leak));
  const hOnes0 = rng(pick(["h0.2", "h1"], r => r.b < 0.004).map(r => r.ones_leak));
  const hOnes1 = rng(pick(["h0.2", "h1"], r => r.b > 0.99).map(r => r.ones_leak));
  const hNoise = rng(pick(["h0.2", "h1"], r => r.b > 0.99).map(r => r["noise0.0001_ridge_psd_vs_Chat_cos"]));
  el.innerHTML = `With a uniform leak, $|d^\\top\\mathbf 1|$ never exceeds ${ui.fmt(uOnes, 2)}, the rank stops at ${fmtRange(rng(uFull.map(r => r.rank_1e8)), x => String(x))} of 78, and the full-$\\hat C$ cosine plateaus at ${fmtRange(rng(uFull.map(r => r.ridge_psd_vs_Chat.cos)))} while the contrast part is exact (${fmtRange(rng(uFull.map(r => r.ridge_psd_contrast_vs_Chat.cos)))}): the same blind subspace. With a heterogeneous leak the rank in these runs reaches ${fmtRange(rng(hFull.map(r => r.rank_1e8)), x => String(x))} and the full $\\hat C$ is recovered (cosine ${fmtRange(rng(hFull.map(r => r.ridge_psd_vs_Chat.cos)))} for $b \\ge 0.19$). But $|d^\\top\\mathbf 1|$ falls from ${fmtRange(hOnes0, x => ui.fmt(x, 2))} at the smallest $b$ to ${fmtRange(hOnes1, x => ui.fmt(x, 2))} at $b = 1$, and there a weight noise of $10^{-4}$ already brings the cosine down to ${fmtRange(hNoise)}.`;
  ui.renderMath(el);
}

// =====================================================================================================================
// (4) nonlinear: partial distance geometry, live
const NL_DEF = { nb: 0.1, nseed: 1 };
const NL_TOURS = [
  { id: "frag", label: "Fragmented", params: { nb: 0.02 }, note: "$b = 0.02$: few edges; almost every pair is known only from below." },
  { id: "perc", label: "Percolating", params: { nb: 0.1 }, note: "$b = 0.1$: a giant component of chains; shortest paths bound most distances from above, loosely." },
  { id: "fused", label: "Fused", params: { nb: 0.3 }, note: "$b = 0.3$: nearly every pair is an edge, so nearly every distance is known exactly." },
];
function nlFigure(ctx) {
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
  const bMd = chartBox(two, "(b) Largest component: classical MDS", "dots: true $D$ · rings: completed from the weights");
  const sc = lineChart(bSc, { series: [{ name: "x", y: [] }], height: 300, xlabel: "true D (training bank)", ylabel: "from the weights", xname: "true D" });
  const md = mdsChart(bMd, { height: 300 });
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
  const bSt = chartBox(two, "(c) What is known about each pair", "neurons sorted by component, largest first");
  const hSt = heatmap(bSt, { map: t => (t < 0.25 ? cmap("seq")(0.92) : t < 0.75 ? cmap("seq")(0.5) : cmap("seq")(0.1)), range: [0, 2], size: 270, colorbar: false,
    xlabel: "neuron j", ylabel: "neuron i", tooltip: (a, c, v) => stTip(a, c, v) });
  ui.legend({ parent: bSt, items: [
    { label: "live edge: $D$ exact", color: seqColor(0.92), shape: "square" },
    { label: "same component: $\\sqrt{2b} \\le D \\le$ shortest path", color: seqColor(0.5), shape: "square" },
    { label: "different components: $D \\ge \\sqrt{2b}$", color: seqColor(0.1), shape: "square" },
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
    const f = { t: sim.steps, b, geo, co, pts, fx: nx / np, fu: nu / np, fl: nlo / np, ee: nx ? ee : NaN, sp: ratios.length ? ratios[ratios.length >> 1] : NaN, r: sim.residual, done, D2: Float64Array.from(D2) };
    // MDS of the largest component (>= 3 members): true vs completed, 2-D for drawing, 3-D for the disparity
    const idx = co.comps[0].members;
    const n = idx.length;
    if (n >= 3) {
      const Dt = new Float64Array(n * n), Dc = new Float64Array(n * n);
      for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) { Dt[a * n + c] = D2[idx[a] * N + idx[c]]; Dc[a * n + c] = geo.Dsp[idx[a] * N + idx[c]] ** 2; }
      const mt = O.mds(Dt, n, 2), mc = O.mds(Dc, n, 2, mt.X);
      f.mds = { idx, X: mt.X, Y: mc.X, n, bar: Math.sqrt(2 * b) };
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

// small canvas: two 2-D configurations (dots and rings) joined by segments, equal aspect
function mdsChart(el, { height = 300 } = {}) {
  el.classList.add("chart");
  const cv = ui.h("canvas"); el.replaceChildren(cv);
  let d = null, W = 0, proj = null;
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
    g.strokeStyle = "rgba(141,140,134,0.6)"; g.lineWidth = 1;
    for (let i = 0; i < n; i++) { g.beginPath(); g.moveTo(sx(X[i * 2]), sy(X[i * 2 + 1])); g.lineTo(sx(Y[i * 2]), sy(Y[i * 2 + 1])); g.stroke(); }
    g.strokeStyle = catColor(1); g.lineWidth = 1.6;
    for (let i = 0; i < n; i++) { g.beginPath(); g.arc(sx(Y[i * 2]), sy(Y[i * 2 + 1]), 4.2, 0, 2 * Math.PI); g.stroke(); }
    g.fillStyle = catColor(0);
    for (let i = 0; i < n; i++) { g.beginPath(); g.arc(sx(X[i * 2]), sy(X[i * 2 + 1]), 3.2, 0, 2 * Math.PI); g.fill(); }
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
    if (best < 0) { ui.hideTip(); return; }
    const sh = Math.hypot(d.X[best * 2] - d.Y[best * 2], d.X[best * 2 + 1] - d.Y[best * 2 + 1]);
    ui.showTip(ev.clientX, ev.clientY, ui.tipRows(`neuron ${d.idx[best]}`, [
      { name: "true position", value: `(${ui.fmt(d.X[best * 2], 2)}, ${ui.fmt(d.X[best * 2 + 1], 2)})`, color: catColor(0), shape: "dot" },
      { name: "from the weights", value: `(${ui.fmt(d.Y[best * 2], 2)}, ${ui.fmt(d.Y[best * 2 + 1], 2)})`, color: catColor(1), shape: "dot" },
      { name: "displacement", value: ui.fmt(sh, 2) }]));
  });
  cv.addEventListener("pointerleave", () => ui.hideTip());
  return { update(x) { d = x; draw(); } };
}

// =====================================================================================================================
// (5) nonlinear sweep table
function nlTable(ctx, data) {
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
