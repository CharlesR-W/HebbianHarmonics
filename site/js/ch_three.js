// Chapter 1, "Three inputs": a live simulation of the user's rule on B = 3 clamped inputs and N interior neurons,
// drawn in the triangle of harmonic embeddings, with linked views. Model side (presets, tours, analysis) lives in
// ch_three_model.js so the tours can be verified headless with the same code.
import * as ui from "./ui.js";
import { heatmap, lineChart, hidpi, scale, niceTicks } from "./charts.js";
import { view3d, simplexVertices } from "./view3d.js";
import { P, boundaryColor, mixBary, colorOf } from "./palette.js";
import * as M from "./ch_three_model.js";

const B = 3, SUB = "₁₂₃";
const KEYS = ["s", "v1", "v2", "v3", "r12", "r13", "r23", "theta", "gamma", "wmin", "n", "seed", "ii", "ib", "mode", "batch", "jit"];
const SIGMA_KEYS = ["v1", "v2", "v3", "r12", "r13", "r23"];
const RESET_KEYS = ["n", "seed", "ii", "ib"];
const EDGE_CAP = 4000;          // at most this many synapses drawn (the strongest); see DESIGN.md section 9
const TRAIL = 10;
const VIEW_DEFAULTS = { view: "var", trails: true, syn: true };

export async function init(panel, ctx) {
  // ------------------------------------------------------------ state
  let { p, tour } = readParams(ctx);
  let opts = { view: ctx.param("view", VIEW_DEFAULTS.view), trails: ctx.param("trails", VIEW_DEFAULTS.trails), syn: ctx.param("syn", VIEW_DEFAULTS.syn) };
  const sim = M.createSim(p);
  let tourActive = tour, schedFired = false, prevLabels = null, cur = null, tickN = 0;
  let fpsT = performance.now(), fpsN = 0, fps = 0;
  const verts = simplexVertices(B), bcol = [0, 1, 2].map(b => boundaryColor(b, B));
  const fig = ctx.slot("sim").closest("figure");

  // ------------------------------------------------------------ layout
  const L = ui.simLayout({ parent: ctx.slot("sim") });
  const narrow = L.view.clientWidth > 0 && L.view.clientWidth < 600;   // phones: leave room for the input labels
  const v = view3d(L.view, { height: 540, yaw: -0.42, pitch: 0.78, center: [0, -0.1, 0.36], radius: 1.0, zoom: narrow ? 0.84 : 1, hull: false,
    pointSize: 4, edgeAlpha: 0.5, depthFade: 0.3, tooltip: (i, kind) => tipFor(i, kind) });
  const cornerEl = v.el.querySelector(".v3-corner");

  const tp = ui.transport({
    parent: L.view, watch: fig, speeds: [{ value: 0.25, label: "×¼" }, 1, 4, 16, 64].map(x => typeof x === "number" ? { value: x, label: `×${x}` } : x),
    speed: (tour && tour.speed) || 1, autoplay: true, maxHistory: 400,
    timeFmt: t => `t = ${ui.fmtFixed(t, 1)}`,
    onTick: speed => tick(speed),
    onSeek: f => { cur = f; show(f); },
    onReset: () => restart(),
  });

  // ------------------------------------------------------------ tours
  const GROUPS = [["basics", "Tours: pruning and parts"], ["inputs", "Tours: other inputs, the floor"], ["history", "Tours: history and noise"]];
  const chipRows = GROUPS.map(([g, label]) => ui.chips({ parent: L.side, label, items: M.TOURS.filter(t => t.group === g), active: tour && tour.group === g ? tour.id : null,
    onPick: t => { chipRows.forEach((c, k) => { if (GROUPS[k][0] !== t.group) c.clear(); }); applyTour(t); } }));

  // ------------------------------------------------------------ controls
  const presetS = ui.select({ label: "inputs $\\Sigma_g$", options: M.PRESETS.map(x => ({ value: x.id, label: x.label })), value: p.s,
    onChange: id => { if (id !== "custom") { Object.assign(p, M.withPreset(p, id)); syncSigmaSliders(); } else p.s = "custom"; liveChange(); } });
  const sigmaBox = ui.h("div", { style: { display: "grid", gap: "6px" } });
  const thetaS = ui.slider({ label: "pruning threshold $\\theta = -w_0/\\eta$", min: -0.2, max: 1.2, step: 0.01, value: p.theta,
    fmt: x => ui.fmtFixed(x, 2), marks: [{ value: 0, label: "0" }, { value: 1 / 3, label: "⅓" }, { value: 1, label: "1" }],
    help: "Your regulariser $\\rho(w) = w_0 - \\lambda w$ with $w_0 = -\\eta\\theta$: a constant pruning drift. A synapse grows only where $\\mathrm{Cov}(u_i,u_j)$ beats $\\theta$.",
    onInput: x => { p.theta = x; liveChange(); } });
  const kappaLine = ui.h("div", { class: "u-help", style: { marginTop: "-4px" } });
  function renderKappa() {
    const S = M.sigmaOf(p); let k0 = 0; for (let q = 0; q < 9; q++) k0 += S[q]; k0 /= 9;
    const tc = p.gamma * k0, below = p.theta < tc;
    kappaLine.innerHTML = `$\\kappa_0 = \\mathbf 1^\\top\\Sigma_g\\mathbf 1/B^2 = ${ui.fmt(k0, 3)}$ (mean covariance of two random starting neurons); collective threshold $\\gamma\\kappa_0 = ${ui.fmt(tc, 3)}$. ` +
      (below ? "<b>θ is below it</b>: the population binds early and lands on one part." : "θ is above it: neurons choose parts independently.");
    ui.renderMath(kappaLine);
  }
  const gammaS = ui.slider({ label: "lateral gain $\\gamma$", min: 0, max: 2, step: 0.05, value: p.gamma, fmt: x => ui.fmtFixed(x, 2),
    help: "$\\eta_{II} = \\gamma\\eta$. Because $w_0$ is shared, the lateral threshold becomes $\\theta/\\gamma$.",
    onInput: x => { p.gamma = x; liveChange(); } });
  const omegaLine = ui.h("div", { class: "u-help", style: { marginTop: "-4px" } });
  const renderOmega = () => { omegaLine.innerHTML = `Floor leak $\\omega = N w_{\\min} = ${ui.fmt(p.n * p.wmin, 2)}$; keep it well below a cluster's input conductance $c$ (the findings recommend $\\omega \\le 0.02$).`; ui.renderMath(omegaLine); };
  const viewS = ui.segmented({ label: "height", options: [{ value: "var", label: "variance $h^\\top\\Sigma_g h$" }, { value: "flat", label: "flat" }], value: opts.view,
    onChange: x => { opts.view = x; geometry(); if (cur) draw3d(cur); writeView(); } });

  const vS = [0, 1, 2].map(b => ui.slider({ label: `variance $\\Sigma_{${b + 1}${b + 1}}$`, min: 0.1, max: 3, step: 0.05, value: p[`v${b + 1}`], fmt: x => ui.fmtFixed(x, 2),
    onInput: x => { p[`v${b + 1}`] = x; sigmaEdited(); } }));
  const rS = [["r12", "1", "2"], ["r13", "1", "3"], ["r23", "2", "3"]].map(([k, a, c]) => ui.slider({ label: `correlation $\\rho_{${a}${c}}$`, min: -1, max: 1, step: 0.05, value: p[k], fmt: x => ui.fmtFixed(x, 2),
    marks: [{ value: 0, label: "" }], onInput: x => { p[k] = x; sigmaEdited(); } }));
  const wminS = ui.slider({ label: "floor $w_{\\min}$ (units of $\\eta/\\lambda$)", min: 1e-5, max: 1e-2, log: true, sig: 2, value: p.wmin, fmt: x => ui.fmt(x, 2),
    help: "Weights are clipped at $w_{\\min}$, so $L_{II}$ stays invertible. A floor synapse still conducts, so every neuron leaks towards the population mean.",
    onChange: x => { p.wmin = x; liveChange(); } });
  const nS = ui.slider({ label: "interior neurons $N$", min: 30, max: 200, step: 10, value: p.n, fmt: x => String(x), onChange: x => { p.n = x; resetChange(); } });
  const seedS = ui.slider({ label: "seed (initial weights, samples, jitter)", min: 0, max: 20, step: 1, value: p.seed, fmt: x => String(x), onChange: x => { p.seed = x; resetChange(); } });
  const iiS = ui.slider({ label: "initial lateral weights $\\sim U(0, s_{II})$", min: 0, max: 0.5, step: 0.01, value: p.ii, fmt: x => ui.fmtFixed(x, 2),
    help: "$s_{II} = 0$ starts every lateral synapse at the floor.", onChange: x => { p.ii = x; resetChange(); } });
  const ibS = ui.slider({ label: "initial input weights $\\sim \\mathrm{Exp}(s_{IB})$", min: 0.02, max: 2, log: true, sig: 2, value: p.ib, fmt: x => ui.fmt(x, 2),
    onChange: x => { p.ib = x; resetChange(); } });
  const modeS = ui.segmented({ label: "covariance", options: [{ value: "mf", label: "exact (mean-field)" }, { value: "mb", label: "minibatch" }], value: p.mode,
    onChange: x => { p.mode = x; batchS.el.style.display = x === "mb" ? "" : "none"; liveChange(); } });
  const batchS = ui.select({ label: "samples per step", options: [1, 4, 16, 64, 256].map(x => ({ value: x, label: String(x) })), value: p.batch,
    onChange: x => { p.batch = x; liveChange(); } });
  batchS.el.style.display = p.mode === "mb" ? "" : "none";
  const jitS = ui.select({ label: "private jitter $w \\leftarrow w\\,e^{\\sigma\\xi}$ per step", options: [{ value: 0, label: "off" }, { value: 1e-3, label: "σ = 10⁻³" }, { value: 1e-2, label: "σ = 10⁻²" }], value: p.jit,
    help: "Independent noise on every synapse above the floor. A run can sit, with a tiny residual, on an exactly symmetric state that is unstable; jitter lets it fall off. Shared minibatch noise cannot do this.",
    onChange: x => { p.jit = x; liveChange(); } });
  const trailsT = ui.toggle({ label: "fading trails", value: opts.trails, onChange: on => { opts.trails = on; v.setTrails(on ? TRAIL : 0); writeView(); } });
  const synT = ui.toggle({ label: "draw synapses", value: opts.syn, onChange: on => { opts.syn = on; if (cur) draw3d(cur); writeView(); } });
  v.setTrails(opts.trails ? TRAIL : 0);

  ui.controls({ parent: L.side, title: "Inputs", primary: [presetS, sigmaBox], advanced: [...vS, ...rS], advancedLabel: "Edit $\\Sigma_g$" });
  ui.controls({ parent: L.side, title: "Plasticity and view", primary: [thetaS, kappaLine, gammaS, viewS],
    advanced: [wminS, omegaLine, nS, seedS, iiS, ibS, modeS, batchS, jitS, trailsT, synT], advancedLabel: "More controls" });

  // ------------------------------------------------------------ readouts and legend
  const stats = ui.stats({ parent: L.side, items: [
    { key: "k", label: "clusters", fmt: x => String(x), help: "Single linkage in L1 on the embeddings, cut at the largest gap; dead neurons excluded." },
    { key: "t", label: "time $t$", fmt: x => ui.fmtFixed(x, 1) },
    { key: "r", label: "stationarity $r$", fmt: x => ui.fmt(x, 2), help: "Projected residual max |dw/dt| over synapses (floor synapses pushed down count 0), covariance units." },
    { key: "floor", label: "synapses at $w_{\\min}$", fmt: x => ui.fmtPct(x, 0) },
  ] });
  ui.legend({ parent: L.side, items: [
    { label: "interior neuron, colour mixes the inputs' colours by $h_i$", color: mixBary([0.55, 0.3, 0.15], bcol), shape: "dot" },
    { label: "dead neuron (every synapse at the floor)", color: P.other, shape: "dot" },
    { label: "input $g_1, g_2, g_3$ (clamped)", color: bcol[0], shape: "square" },
    { label: "synapse, opacity $w/\\max w$", color: P.edge, shape: "line" },
  ] });

  // ------------------------------------------------------------ linked views (below)
  // the linked views sit in a 2 x 2 grid under the 3D view, filling the column beside the tall control panel
  const mid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(max(270px, 40%), 1fr))", gap: "18px 28px", marginTop: "18px" } });
  // wide screens: under the 3D view, beside the tall control panel. Narrow screens (one column): after the
  // controls, so tours and sliders stay next to the view they drive.
  const oneCol = window.matchMedia("(max-width: 900px)");
  const placeMid = () => (oneCol.matches ? L.below : L.view).append(mid);
  placeMid(); oneCol.addEventListener("change", placeMid);
  const box = (title, sub, host = L.below) => { const b = ui.h("div"); host.append(ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", html: title }), sub ? ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" }, html: sub }) : null, b)); return b; };
  const covBox = box("(a) Voltage covariance $H\\Sigma_g H^\\top$", "neurons sorted by cluster; bands: cluster colour", mid);
  const riverBox = box("(b) Neurons per cluster over time", "stacked; colour: the cluster's mean embedding", mid);
  const resBox = box("(c) Stationarity residual $r$", "log scale; dashed: the tolerance for “converged”", mid);
  const floorBox = box("(d) Synapses at the floor $w_{\\min}$", "fraction of each kind", mid);
  ui.renderMath(mid);
  const cov = heatmap(covBox, { map: "div", range: [-1, 1], size: 270, zlabel: "$h_i^\\top\\Sigma_g h_j$", xlabel: "neuron j", ylabel: "neuron i",
    tooltip: (i, j, val) => covTip(i, j, val) });
  let covOrder = null;
  const river = riverChart(riverBox, { height: 230, tooltip: f => riverTip(f) });
  const resChart = lineChart(resBox, { series: [{ name: "r", x: [0], y: [1], color: P.fg, width: 1.5 }], ylog: true, yrange: [1e-8, 2], height: 230, xlabel: "time t", ylabel: "r = max |dw/dt|",
    refs: [{ y: M.RES_TOL, label: "10⁻⁵" }], legend: false, xname: "t", yfmt: pow10 });
  const floorChart = lineChart(floorBox, { series: [{ name: "lateral (I–I)", x: [0], y: [0], color: P.fg, width: 1.5 }, { name: "to inputs (I–B)", x: [0], y: [0], color: P.mute2, width: 1.5, dash: true }],
    yrange: [0, 1], height: 230, xlabel: "time t", ylabel: "fraction at the floor", yfmt: x => ui.fmtPct(x, 0), xname: "t" });

  // ------------------------------------------------------------ geometry: lift, mesh, anchors
  let kappa = 0;
  function geometry() {
    const S = sim.Sigma, md = Math.max(S[0], S[4], S[8]);
    kappa = opts.view === "var" ? 0.8 / md : 0;
    const pos = h => { const x = h[0] * verts[0][0] + h[1] * verts[1][0] + h[2] * verts[2][0], y = h[0] * verts[0][1] + h[1] * verts[1][1] + h[2] * verts[2][1]; return [x, y, kappa * M.quad(h, S)]; };
    // the input squares sit just outside their corners (a cluster at a corner would otherwise hide under them);
    // a short leader joins each square to its true corner
    const OUT = 1.16;
    v.setAnchors(verts.map((xyz, b) => ({ xyz: [OUT * xyz[0], OUT * xyz[1], kappa * S[b * B + b]], label: `g${SUB[b]}`, color: bcol[b], size: 6.5 })));
    const G = [], m = 6, faint = "rgba(141,140,134,0.30)", rim = "rgba(107,107,112,0.75)";
    for (let b = 0; b < B; b++) {
      const a = (b + 1) % B, c = (b + 2) % B;
      for (let j = 0; j < m; j++) {               // lines of constant h_b = j/m (j = 0 is the opposite edge)
        const hb = j / m, segs = Math.max(1, m - j);
        for (let s = 0; s < segs; s++) {
          const f0 = s / segs, f1 = (s + 1) / segs, h0 = [0, 0, 0], h1 = [0, 0, 0];
          h0[b] = hb; h0[a] = (1 - hb) * f0; h0[c] = (1 - hb) * (1 - f0);
          h1[b] = hb; h1[a] = (1 - hb) * f1; h1[c] = (1 - hb) * (1 - f1);
          G.push({ a: pos(h0), b: pos(h1), color: j === 0 ? rim : faint });
        }
      }
    }
    for (let b = 0; b < B; b++) G.push({ a: [verts[b][0], verts[b][1], kappa * S[b * B + b]], b: [OUT * verts[b][0], OUT * verts[b][1], kappa * S[b * B + b]], color: bcol[b] });
    if (kappa > 0) {
      const base = verts.map(x => [x[0], x[1], 0]);
      for (let b = 0; b < B; b++) {
        G.push({ a: base[b], b: base[(b + 1) % B], color: "rgba(141,140,134,0.55)", dash: true });
        G.push({ a: base[b], b: [verts[b][0], verts[b][1], kappa * S[b * B + b]], color: "rgba(141,140,134,0.55)", dash: true });
      }
    }
    v.setGuides(G);
    cornerText();
  }
  function cornerText() {
    const mode = p.mode === "mb" ? `minibatch covariance, ${p.batch} sample${p.batch > 1 ? "s" : ""} per step` : "mean-field: exact $\\Sigma_g$";
    v.setCorner(`${mode}<br>${opts.view === "var" ? "height: voltage variance $h_i^\\top\\Sigma_g h_i$ (dashed: the flat triangle)" : "flat: the triangle of embeddings"}`);
    ui.renderMath(cornerEl);
  }

  // ------------------------------------------------------------ drawing
  let pairsAll = null, pairsN = -1;
  function edgeIndex(N) {
    if (pairsN === N) return pairsAll;
    const nTri = N * (N - 1) / 2; pairsAll = new Uint32Array(2 * (nTri + N * B)); let k = 0;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { pairsAll[k++] = i; pairsAll[k++] = j; }
    for (let i = 0; i < N; i++) for (let b = 0; b < B; b++) { pairsAll[k++] = i; pairsAll[k++] = N + b; }
    pairsN = N; return pairsAll;
  }
  function draw3d(f) {
    const N = f.N, H = f.H, S = f.Sigma, xyz = new Float32Array(3 * N), colors = new Array(N), sizes = new Array(N);
    const r = N > 150 ? 3.4 : N > 80 ? 4 : 4.6;
    // a settled cluster's members coincide, so its dot is drawn larger: radius grows with sqrt(cluster size / N)
    const rad = new Float32Array(N).fill(r);
    if (f.robust) for (const c of f.clusters) if (!c.dead) for (const i of c.members) rad[i] = r * (0.8 + 2 * Math.sqrt(c.n / N));
    for (let i = 0; i < N; i++) {
      const h = [H[i * B], H[i * B + 1], H[i * B + 2]];
      xyz[3 * i] = h[0] * verts[0][0] + h[1] * verts[1][0] + h[2] * verts[2][0];
      xyz[3 * i + 1] = h[0] * verts[0][1] + h[1] * verts[1][1] + h[2] * verts[2][1];
      xyz[3 * i + 2] = kappa * M.quad(h, S);
      colors[i] = f.dead[i] ? P.other : mixBary(h, bcol); sizes[i] = f.dead[i] ? r * 0.75 : rad[i];
    }
    v.setPoints(xyz, colors, sizes);
    if (!opts.syn) { v.setEdges(new Uint32Array(0), new Float32Array(0)); return; }
    const W8 = f.W8, all = edgeIndex(N), hist = new Uint32Array(256);
    let thr = 6;                                   // alpha >= 6/255: fainter lines are invisible anyway
    for (let e = 0; e < W8.length; e++) hist[W8[e]]++;
    let count = 0; for (let q = 255; q >= thr; q--) count += hist[q];
    while (count > EDGE_CAP && thr < 255) { count -= hist[thr]; thr++; }
    const n = Math.min(count, EDGE_CAP), pairs = new Uint32Array(2 * n), alpha = new Float32Array(n);
    let k = 0;
    for (let e = 0; e < W8.length && k < n; e++) if (W8[e] >= thr) { pairs[2 * k] = all[2 * e]; pairs[2 * k + 1] = all[2 * e + 1]; alpha[k] = W8[e] / 255; k++; }
    v.setEdges(pairs, alpha);
  }
  const statCache = {};
  const setStat = (key, val, sub) => { const sig = `${val}|${sub}`; if (statCache[key] === sig) return; statCache[key] = sig; stats.set(key, val, sub); };
  function showStats(f) {
    const live = f.clusters.filter(c => !c.dead && !c.small);
    const sizes = live.length <= 6 ? live.map(c => c.n).join(" · ") : `${live.length} groups`;
    setStat("k", f.K, `${sizes}${f.nSmall ? ` · ${f.nSmall} straggler${f.nSmall > 1 ? "s" : ""}` : ""}${f.nDead ? ` · ${f.nDead} dead` : ""}${f.robust ? "" : " · not yet clustered"}`);
    setStat("t", Math.round(f.t * 10) / 10, `dt = ${M.DT}${fps && tp.playing ? ` · ${fps.toFixed(0)} fps` : ""}`);
    setStat("r", +f.resid.toPrecision(2), f.mode === "mb" ? "mean-field drift at these weights" : f.resid < M.RES_TOL ? "below 10⁻⁵: converged (or stuck on a symmetric state)" : p.jit ? "jitter on: never exactly still" : "still moving");
    setStat("floor", +((f.fII * (f.N * (f.N - 1) / 2) + f.fIB * f.N * B) / (f.N * (f.N - 1) / 2 + f.N * B)).toFixed(3), `lateral ${ui.fmtPct(f.fII, 0)} · input ${ui.fmtPct(f.fIB, 0)}`);
  }
  function drawCov(f) {
    const N = f.N, H = f.H, S = f.Sigma, md = Math.max(S[0], S[4], S[8]);
    const order = [], groups = [];
    f.clusters.forEach((c, k) => {
      const start = order.length; c.members.forEach(i => order.push(i));
      groups.push({ start, end: order.length, color: c.color, label: c.dead ? "dead" : `c${k + 1}` });
    });
    const HS = new Float64Array(N * B);
    for (let i = 0; i < N; i++) for (let a = 0; a < B; a++) { let s = 0; for (let b = 0; b < B; b++) s += H[i * B + b] * S[b * B + a]; HS[i * B + a] = s; }
    const Mx = new Float32Array(N * N);
    for (let a = 0; a < N; a++) { const i = order[a]; for (let c = 0; c < N; c++) { const j = order[c]; Mx[a * N + c] = HS[i * B] * H[j * B] + HS[i * B + 1] * H[j * B + 1] + HS[i * B + 2] * H[j * B + 2]; } }
    covOrder = order;
    // cluster bands only once the clustering is clear-cut (dozens of transient groups would be unreadable)
    cov.update({ values: Mx, rows: N, cols: N, range: [-md, md], groups: f.robust && groups.length <= 12 ? { rows: groups } : null });
  }
  function drawCharts(extra) {
    let fr = tp.frames;
    if (extra && fr[fr.length - 1] !== extra) fr = fr.concat([extra]);
    const xs = fr.map(f => f.t);
    const ys = fr.map(f => Math.max(f.resid, 1e-13)), top = Math.max(1, ...ys.filter(Number.isFinite)) * 1.5;
    // fixed lower end so the 10⁻⁶ tolerance line is always inside the plot
    resChart.update({ series: [{ name: "r", x: xs, y: ys, color: P.fg, width: 1.5 }], yrange: [Math.min(1e-8, ...ys.filter(Number.isFinite)), top] });
    floorChart.update({ series: [{ name: "lateral (I–I)", x: xs, y: fr.map(f => f.fII), color: P.fg, width: 1.5 }, { name: "to inputs (I–B)", x: xs, y: fr.map(f => f.fIB), color: P.mute2, width: 1.5, dash: true }] });
    river.update(fr);
    setCursors(cur ? cur.t : null);
  }
  function setCursors(t) { resChart.setCursor(t); floorChart.setCursor(t); river.setCursor(t); }
  function show(f) { draw3d(f); showStats(f); drawCov(f); setCursors(f.t); }

  function colourise(f) { for (const c of f.clusters) c.color = c.dead ? P.other : mixBary(c.part, bcol); return f; }

  // ------------------------------------------------------------ simulation loop
  let stepAcc = 0;
  function tick(speed) {
    stepAcc += speed; const n = Math.floor(stepAcc); stepAcc -= n;
    if (n === 0 && tp.playing) return null;           // ×¼: one step every fourth frame (the step button always steps)
    sim.advance(Math.max(1, n), 22);
    if (tourActive && tourActive.at && !schedFired && sim.time() >= tourActive.at.t) {
      schedFired = true; p.theta = tourActive.at.theta; sim.update({ theta: p.theta }); thetaS.set(p.theta);
      sim.s.resid = NaN;                                 // the last step's residual was for the old theta
      thetaS.el.style.transition = "background-color .4s"; thetaS.el.style.backgroundColor = "rgba(194,65,12,.10)"; setTimeout(() => { thetaS.el.style.backgroundColor = ""; }, 1600);
    }
    const a = colourise(M.analyse(sim, prevLabels)); prevLabels = a.labels; cur = a; tickN++;
    draw3d(a); showStats(a);
    fpsN++; const now = performance.now();
    if (now - fpsT > 1000) { fps = fpsN * 1000 / (now - fpsT); fpsN = 0; fpsT = now; }
    const pending = tourActive && tourActive.at && !schedFired;
    const done = p.mode === "mf" && !p.jit && a.resid < M.RES_TOL && !pending;
    // linked views are the expensive part of a frame (DOM / SVG / canvas repaints): refresh them a few times a second
    if (done || tickN % 12 === 0) drawCharts(a);
    if (done || tickN % 12 === 6) drawCov(a);
    if (done) { a.done = true; a.doneLabel = "converged: r < 10⁻⁵"; }
    return a;
  }
  function restart() {
    sim.reset(p); schedFired = false; prevLabels = null; tickN = 0;
    geometry(); v.clearTrails();
    const a = colourise(M.analyse(sim, null)); prevLabels = a.labels; cur = a;
    draw3d(a); showStats(a); drawCov(a); drawCharts(a);
    return a;
  }

  // ------------------------------------------------------------ control plumbing
  let lastSigmaKey = "";
  function liveChange() {
    untour();
    sim.update({ ...p }); sim.s.resid = NaN;
    const key = SIGMA_KEYS.map(k => p[k]).join(",");
    if (key !== lastSigmaKey) { lastSigmaKey = key; renderSigma(); geometry(); }
    renderKappa(); renderOmega();
    cornerText(); writeURL(); tp.play();
  }
  function resetChange() { untour(); renderOmega(); writeURL(); tp.reset(); tp.play(); }
  function sigmaEdited() { p.s = M.presetOf(p); presetS.set(p.s); liveChange(); }
  function syncSigmaSliders() { vS.forEach((s, b) => s.set(p[`v${b + 1}`])); rS[0].set(p.r12); rS[1].set(p.r13); rS[2].set(p.r23); }
  function syncAll() {
    presetS.set(p.s); syncSigmaSliders(); thetaS.set(p.theta); gammaS.set(p.gamma); wminS.set(p.wmin); nS.set(p.n); seedS.set(p.seed);
    iiS.set(p.ii); ibS.set(p.ib); modeS.set(p.mode); batchS.set(p.batch); jitS.set(p.jit); renderKappa(); renderOmega(); batchS.el.style.display = p.mode === "mb" ? "" : "none";
  }
  function untour() {
    if (!tourActive) return;
    tourActive = null; chipRows.forEach(c => c.moved());
  }
  function writeURL() {
    const d = M.DEFAULTS, o = { tour: tourActive ? tourActive.id : null };
    for (const k of KEYS) {
      if (tourActive) { o[k] = null; continue; }
      if (SIGMA_KEYS.includes(k) && p.s !== "custom") { o[k] = null; continue; }
      o[k] = p[k] === d[k] ? null : p[k];
    }
    ctx.setParams(o);
  }
  function writeView() {
    ctx.setParams({ view: opts.view === VIEW_DEFAULTS.view ? null : opts.view, trails: opts.trails === VIEW_DEFAULTS.trails ? null : opts.trails, syn: opts.syn === VIEW_DEFAULTS.syn ? null : opts.syn });
  }
  function applyTour(t) {
    tourActive = t; p = M.tourParams(t); syncAll(); renderSigma(); writeURL();
    tp.setSpeed(t.speed || 1); tp.reset(); tp.play();
  }
  ctx.onParams(params => {
    const r = readParams(ctx);
    if (r.tour) {
      chipRows.forEach((c, k) => { if (GROUPS[k][0] === r.tour.group) c.set(r.tour.id); else c.clear(); });
      applyTour(r.tour); return;
    }
    tourActive = null; chipRows.forEach(c => c.clear());
    p = r.p; syncAll(); renderSigma(); tp.reset(); tp.play();
  });
  ctx.onHide(() => v.setAutoRotate(false));

  // ------------------------------------------------------------ the 3x3 matrix
  function renderSigma() {
    const S = M.sigmaOf(p), md = Math.max(S[0], S[4], S[8], 1e-9);
    const req = [p.r12, p.r13, p.r23], got = [S[1] / Math.sqrt(S[0] * S[4]), S[2] / Math.sqrt(S[0] * S[8]), S[5] / Math.sqrt(S[4] * S[8])];
    const clamped = req.some((x, i) => Math.abs(x - got[i]) > 2e-3);
    const cs = { font: "12px/1 var(--sans)", fontVariantNumeric: "tabular-nums", display: "grid", placeItems: "center", height: "26px", borderRadius: "3px" };
    const cell = (x, head) => ui.h("span", { style: head ? { ...cs, color: P.mute } : { ...cs, background: colorOf("div", x, -md, md), color: Math.abs(x) / md > 0.62 ? "#fff" : P.fg } }, head || ui.fmt(x, 2));
    const grid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "22px repeat(3, 46px)", gap: "2px" } }, cell(0, " "), ...[0, 1, 2].map(b => cell(0, `g${SUB[b]}`)));
    for (let a = 0; a < B; a++) { grid.append(cell(0, `g${SUB[a]}`)); for (let b = 0; b < B; b++) grid.append(cell(S[a * B + b])); }
    const cbar = ui.colorbar({ map: "div", range: [-md, md], width: 120, fmt: x => ui.fmt(x, 2) });
    const note = ui.h("div", { class: "u-help", html: clamped ? `Infeasible correlations were clamped to the nearest valid (PSD) matrix: realised $\\rho = ${got.map(x => ui.fmt(x, 2)).join(", ")}$.` : "The covariance the clamped inputs are drawn from (synthetic)." });
    sigmaBox.replaceChildren(ui.h("div", { style: { display: "flex", gap: "14px", alignItems: "center", flexWrap: "wrap" } }, grid,
      ui.h("div", { style: { display: "grid", gap: "8px", justifyItems: "start" } }, ui.h("span", { class: "synthetic-tag", text: "synthetic Σ" }), cbar.el)), note);
    ui.renderMath(sigmaBox);
  }

  // ------------------------------------------------------------ tooltips
  function clusterOf(f, i) { return f.clusters.findIndex(c => c.members.includes(i)); }
  function tipFor(i, kind) {
    const f = cur; if (!f) return null;
    if (kind === "anchor") return ui.tipRows(`input g${SUB[i]} (boundary neuron)`, [{ name: "variance Σ" + SUB[i] + SUB[i], value: ui.fmt(f.Sigma[i * B + i], 3), color: bcol[i], shape: "square" }], "Its voltage is clamped to the stimulus.");
    const h = [f.H[i * B], f.H[i * B + 1], f.H[i * B + 2]], k = clusterOf(f, i), c = f.clusters[k];
    const rows = [0, 1, 2].map(b => ({ name: `h${SUB[b]} (share of g${SUB[b]})`, value: ui.fmtFixed(h[b], 3), color: bcol[b], shape: "square" }));
    rows.push({ name: "voltage variance hᵀΣh", value: ui.fmt(M.quad(h, f.Sigma), 3) });
    rows.push({ name: "cluster", value: c ? (c.dead ? `dead (${c.n})` : `c${k + 1}, ${c.n} neuron${c.n > 1 ? "s" : ""}${c.small ? " (straggler)" : ""}`) : "–", color: c ? c.color : null });
    if (c && !c.dead) rows.push({ name: "c (summed input conductance)", value: ui.fmt(c.c, 3) });
    return ui.tipRows(`interior neuron ${i}`, rows);
  }
  function covTip(a, b, val) {
    const f = cur; if (!f || !covOrder) return null;
    const i = covOrder[a], j = covOrder[b];
    return ui.tipRows(`neurons ${i} and ${j}`, [{ name: "Cov(uᵢ, uⱼ) = hᵢᵀΣhⱼ", value: ui.fmt(val, 3) }, { name: "θ / γ (lateral threshold)", value: f.gamma > 0 ? ui.fmt(f.theta / f.gamma, 3) : "∞" }]);
  }
  function riverTip(f) {
    const rows = f.clusters.slice(0, 8).map((c, k) => ({ name: c.dead ? "dead" : `c${k + 1} at h = (${c.part.map(x => ui.fmtFixed(x, 2)).join(", ")}), c = ${ui.fmt(c.c, 2)}`, value: `${c.n}`, color: c.color }));
    return ui.tipRows(`t = ${ui.fmtFixed(f.t, 1)}: ${f.K} cluster${f.K === 1 ? "" : "s"}`, rows, f.clusters.length > 8 ? `and ${f.clusters.length - 8} more` : null);
  }

  // ------------------------------------------------------------ go
  renderSigma(); geometry(); renderKappa(); renderOmega();
  tp.record(restart());
  ctx.onShow(() => { if (cur) v.render(); });
}

// 1e-4 -> "10⁻⁴" (log-axis ticks); anything else in the usual format
function pow10(x) {
  const e = Math.round(Math.log10(x));
  if (Math.abs(Math.log10(x) - e) > 1e-6) return ui.fmt(x, 2);
  return e === 0 ? "1" : "10" + String(e).replace("-", "⁻").replace(/\d/g, d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
}

// URL -> parameters (a tour in the URL wins over individual parameters)
function readParams(ctx) {
  const tour = M.tourById(ctx.param("tour", ""));
  if (tour) return { p: M.tourParams(tour), tour };
  const d = M.DEFAULTS, p = {};
  for (const k of KEYS) p[k] = ctx.param(k, d[k]);
  if (M.PRESETS.some(x => x.id === p.s && x.v)) Object.assign(p, M.withPreset(p, p.s)); else p.s = M.presetOf(p);
  p.n = Math.max(30, Math.min(200, Math.round(p.n / 10) * 10)); p.seed = Math.max(0, Math.round(p.seed)); if (![0, 1e-3, 1e-2].includes(p.jit)) p.jit = 0;
  p.mode = p.mode === "mb" ? "mb" : "mf"; if (![1, 4, 16, 64, 256].includes(p.batch)) p.batch = d.batch;
  p.wmin = Math.max(1e-5, Math.min(1e-2, p.wmin)); p.gamma = Math.max(0, Math.min(2, p.gamma)); p.theta = Math.max(-0.2, Math.min(1.2, p.theta));
  return { p, tour: null };
}

// ---------------------------------------------------------------- a small stacked-area chart (canvas)
// Neurons per cluster at each recorded frame, stacked; clusters ordered around the triangle so colours keep
// their place; dead neurons on top in the "other" grey. Hover shows the clusters at that time.
function riverChart(el, { height = 220, tooltip } = {}) {
  el.classList.add("chart");
  const cv = ui.h("canvas"), ov = ui.h("canvas", { style: { position: "absolute", left: "0", top: "0", pointerEvents: "none" } });
  el.replaceChildren(cv, ov);
  let frames = [], cursor = null, W = 0, X = null, m = { l: 42, r: 10, t: 8, b: 38 };
  const angle = c => c.dead ? 99 : Math.atan2(c.part[1] - c.part[2], (2 * c.part[0] - c.part[1] - c.part[2]) / Math.sqrt(3)) + (Math.abs(c.part[0] - 1 / 3) + Math.abs(c.part[1] - 1 / 3) < 0.05 ? 50 : 0);
  function draw() {
    W = Math.floor(el.clientWidth); if (!W) return;
    const g = hidpi(cv, W, height); g.clearRect(0, 0, W, height);
    const N = frames.length ? frames[frames.length - 1].N : 100, t1 = frames.length ? Math.max(frames[frames.length - 1].t, 1) : 1;
    X = scale("linear", 0, t1, m.l, W - m.r); const Y = scale("linear", 0, N, height - m.b, m.t);
    g.font = "11px Inter, system-ui, sans-serif"; g.fillStyle = P.mute; g.strokeStyle = P.line;
    // grid
    g.lineWidth = 1; g.strokeStyle = "#efeee9";
    for (const yv of [0, N / 2, N]) { g.beginPath(); g.moveTo(m.l, Math.round(Y(yv)) + .5); g.lineTo(W - m.r, Math.round(Y(yv)) + .5); g.stroke(); }
    // columns
    for (let k = 0; k < frames.length; k++) {
      const f = frames[k], x0 = X(f.t), x1 = k + 1 < frames.length ? X(frames[k + 1].t) : Math.max(x0 + 2, X(f.t) + 2);
      const cs = f.clusters.slice().sort((a, b) => angle(a) - angle(b));
      let acc = 0;
      for (const c of cs) {
        const y0 = Y(acc), y1 = Y(acc + c.n); acc += c.n;
        g.fillStyle = c.color || P.other;
        g.fillRect(x0, y1, Math.max(1, x1 - x0 + 0.6), y0 - y1);
      }
    }
    // axes
    g.strokeStyle = "#c9c8c1"; g.beginPath(); g.moveTo(m.l, height - m.b + .5); g.lineTo(W - m.r, height - m.b + .5); g.stroke();
    g.fillStyle = P.mute; g.textAlign = "right";
    for (const yv of [0, N / 2, N]) g.fillText(String(yv), m.l - 6, Y(yv) + 3.5);
    g.textAlign = "center";
    for (const xv of niceTicks(0, t1, Math.max(3, Math.round(W / 90)))) { g.fillText(ui.fmt(xv), X(xv), height - m.b + 15); g.beginPath(); g.moveTo(X(xv) + .5, height - m.b); g.lineTo(X(xv) + .5, height - m.b + 4); g.stroke(); }
    g.fillStyle = P.ink2 || "#3f3e3b"; g.font = "560 11.5px Inter, system-ui, sans-serif";
    g.fillText("time t", (m.l + W - m.r) / 2, height - 4);
    g.save(); g.translate(12, (m.t + height - m.b) / 2); g.rotate(-Math.PI / 2); g.fillText("neurons", 0, 0); g.restore();
    drawCursor();
  }
  let ovW = 0, o = null;
  function drawCursor() {
    if (!W) return;
    if (ovW !== W || !o) { o = hidpi(ov, W, height); ovW = W; }
    o.clearRect(0, 0, W, height);
    if (cursor == null || !frames.length || !X) return;
    o.strokeStyle = P.accent; o.globalAlpha = 0.8; o.lineWidth = 1.25; o.beginPath(); o.moveTo(X(cursor), m.t); o.lineTo(X(cursor), height - m.b); o.stroke();
  }
  const ro = new ResizeObserver(() => { if (Math.floor(el.clientWidth) !== W) draw(); }); ro.observe(el);
  cv.addEventListener("pointermove", ev => {
    if (!X || !frames.length || !tooltip) return;
    const r = cv.getBoundingClientRect(), t = X.inv(ev.clientX - r.left);
    let best = 0; for (let k = 0; k < frames.length; k++) if (frames[k].t <= t) best = k;
    ui.showTip(ev.clientX, ev.clientY, tooltip(frames[best]));
  });
  cv.addEventListener("pointerleave", () => ui.hideTip());
  return {
    update(fr) { frames = fr; draw(); },
    setCursor(t) { if (t === cursor) return; cursor = t; drawCursor(); },
  };
}
