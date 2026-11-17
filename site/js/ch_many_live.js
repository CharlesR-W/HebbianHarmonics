// Chapter "many": the live population sim. N neurons with s random fixed attachments to B inputs (up to 24),
// simulated with onsite_engine.js; b acts on the running network (continuation), every other control restarts
// from W = 0. Views: classical MDS of the responses with live edges (3D), log10(D^2/2b) sorted by component,
// the converged states recorded against b (cold / up / down), and a component table with diameters on both banks.
import * as ui from "./ui.js";
import { heatmap, lineChart } from "./charts.js";
import { view3d } from "./view3d.js";
import { P, catColor } from "./palette.js";
import * as O from "./onsite_engine.js";
import * as M from "./ch_many_model.js";

export const BR = { cold: catColor(0), up: catColor(1), down: catColor(2) };   // branch colours, whole chapter
export const BR_LABEL = { cold: "cold start (W = 0)", up: "continuation up", down: "continuation down" };

const KEYS = { sig: "sig", b: "b", a: "a", pot: "lp", n: "n", s: "s", seed: "seed" };   // state -> URL key
const compColor = (cs, i) => {
  const k = cs.labels[i], c = cs.st[k];
  if (c.size === 1) return P.interior;
  return cs.rank[k] < 8 ? catColor(cs.rank[k]) : P.other;
};

export function buildLive(slot, ctx, F) {
  const fig = slot.closest("figure");
  let p = { ...M.LIVE_DEFAULTS }, tour = null, path = null;
  readURL();

  // ------------------------------------------------------------ layout
  const L = ui.simLayout({ parent: slot });
  const v = view3d(L.view, { height: 440, hull: false, pointSize: 4.5, edgeAlpha: 0.55, depthFade: 0.3, pitch: 0.7, yaw: -0.6, tooltip: (i, kind) => (kind === "point" ? tipNeuron(i) : null) });
  const cornerEl = v.el.querySelector(".v3-corner");
  const tp = ui.transport({
    parent: L.view, watch: fig, speeds: [{ value: 1, label: "×1" }, { value: 10, label: "×10" }, { value: 200, label: "max" }], speed: 200,
    autoplay: true, maxHistory: 200, timeFmt: t => `t = ${ui.fmt(t, 3)}`,
    onTick: sp => tick(sp), onSeek: f => seek(f), onRestore: f => restore(f), onReset: () => reset(),
  });

  const tours = ui.chips({ parent: L.side, label: "Tours", items: M.LIVE_TOURS, active: tour ? tour.id : null, onPick: t => applyTour(t) });
  const sigS = ui.select({ label: "inputs $\\Sigma_g$", options: M.SIGMAS.map(s => ({ value: s.id, label: s.label })), value: p.sig, onChange: x => { p.sig = x; structural(); } });
  const bS = ui.slider({ label: "fusion scale $b$", min: 0.002, max: 2, log: true, sig: 2, value: p.b,
    marks: [{ value: 0.01, label: "0.01" }, { value: 0.1, label: "0.1" }, { value: 1, label: "1" }],
    help: "Acts on the running network: $W$ is kept, so moving $b$ is a continuation. An edge can live only where $D_{ij}^2 &lt; 2b$.",
    onInput: x => setB(x, true) });
  const aS = ui.segmented({ label: "stiffness $\\alpha$", options: [0.1, 1, 10].map(x => ({ value: x, label: String(x) })), value: p.a, onChange: x => { p.a = x; structural(); } });
  const potS = ui.segmented({ label: "potential", options: [{ value: "linear", label: "linear" }, { value: "rect", label: "leaky rectifier" }], value: p.pot, onChange: x => { p.pot = x; structural(); } });
  const nS = ui.slider({ label: "neurons $N$", min: 20, max: 60, step: 10, value: p.n, fmt: String, onChange: x => { p.n = x; structural(); } });
  const sS = ui.slider({ label: "attachments per neuron $s$", min: 1, max: 6, step: 1, value: p.s, fmt: String, onChange: x => { p.s = x; structural(); } });
  const seedS = ui.slider({ label: "seed (attachments, thresholds, banks)", min: 0, max: 20, step: 1, value: p.seed, fmt: String, onChange: x => { p.seed = x; structural(); } });
  ui.controls({ parent: L.side, primary: [sigS, bS, aS, potS], advanced: [nS, sS, seedS] });

  const stats = ui.stats({ parent: L.side, items: [
    { key: "k", label: "components $K$", fmt: String, help: "Connected components of the live graph (w > 1e-9), singletons included." },
    { key: "big", label: "largest", fmt: String },
    { key: "span", label: "widest span$/\\sqrt{2b}$", fmt: x => ui.fmt(x, 3), help: "Largest pairwise D inside one component, over the edge range sqrt(2b). Above 1: a chain." },
    { key: "J", label: "objective $J$", fmt: x => ui.fmt(x, 4), help: "J(W) = mean energy over the training bank + sum over pairs of (alpha/2 w^2 - b w); the rule descends it." },
    { key: "r", label: "plasticity residual $r$", fmt: x => ui.fmt(x, 2), help: "Projected max |dw/dt|. Fixed-point statements only below 1e-6." },
    { key: "eq", label: "neuronal residual", fmt: x => ui.fmt(x, 2), help: "max |dE/du| of the last equilibrium solve (training bank)." },
  ] });
  ui.legend({ parent: L.side, items: [
    { label: "neuron; colour: its component", color: catColor(0), shape: "dot" },
    { label: "singleton", color: P.interior, shape: "dot" },
    { label: "live edge", color: P.edge, shape: "line" },
  ] });

  // below: matrix, recorded fixed points, component table
  const mid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 300px), 1fr))", gap: "18px 28px", marginTop: "18px" } });
  const oneCol = window.matchMedia("(max-width: 900px)");
  const placeMid = () => (oneCol.matches ? L.below : L.view).append(mid);
  placeMid(); oneCol.addEventListener("change", placeMid);
  const box = (title, sub) => { const b = ui.h("div"); mid.append(ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", html: title }), ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" }, html: sub }), b)); return b; };
  const hmBox = box("(a) $\\log_{10} D_{ij}^2/2b$, sorted by component", "blue: within one edge's range; bands: components (grey: singletons)");
  const recBox = box("(b) Fixed points reached, against $b$", "each dot: a state with $r &lt; 10^{-6}$, coloured by how it was reached");
  const tabBox = box("(c) Components with two or more members", "diameters on the training bank / the evaluation bank");
  ui.renderMath(mid);
  const hm = heatmap(hmBox, { map: "div", range: [-2, 2], size: 300, zlabel: "$\\log_{10} D_{ij}^2/2b$", xlabel: "neuron j (sorted)", ylabel: "neuron i", tooltip: (i, j) => tipPair(i, j) });
  const rec = lineChart(recBox, { series: [], xlog: true, ylog: true, xrange: [0.0018, 2.4], yrange: [0.8, 80], xlabel: "fusion scale b", ylabel: "components K", height: 230, xname: "b" });
  const recList = ui.h("div", { class: "u-help", style: { marginTop: "4px" } }); recBox.append(recList);
  const tab = ui.h("div"); tabBox.append(tab);

  // ------------------------------------------------------------ state
  let sys = null, cur = null, prevX = null, tickN = 0, T = 0, branch = "cold", lastB = p.b;
  let points = [], pointsKey = "", order = null, Ueval = null, evalStats = null, evalFor = -1;
  const configKey = () => [p.sig, p.a, p.pot, p.n, p.s, p.seed].join("|");

  function analyse() {
    const sim = sys.sim, N = sim.net.N;
    const md = O.mds(sim.D2, N, 3, prevX); prevX = md.X;
    const key = new Float64Array(N); for (let i = 0; i < N; i++) key[i] = md.X[i * 3];
    const cs = M.components(sim, key);
    let wmax = 0; for (let k = 0; k < N * N; k++) wmax = Math.max(wmax, sim.net.W[k]);
    let span = 0; cs.st.forEach(c => { span = Math.max(span, c.span); });
    return { N, md, cs, wmax, spanR: span / Math.sqrt(2 * sim.b), b: sim.b, residual: sim.residual, eq: sim.eqResidual, J: M.objectiveJ(sim) };
  }
  function draw3d(a, W) {
    const N = a.N, xyz = new Float32Array(3 * N), cols = new Array(N), sizes = new Array(N);
    for (let i = 0; i < N; i++) {
      xyz[3 * i] = a.md.X[3 * i]; xyz[3 * i + 1] = a.md.X[3 * i + 1]; xyz[3 * i + 2] = a.md.X[3 * i + 2];
      cols[i] = compColor(a.cs, i); sizes[i] = a.cs.st[a.cs.labels[i]].size === 1 ? 3.4 : 4.8;
    }
    v.setPoints(xyz, cols, sizes);
    const pr = [], al = [];
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const w = W[i * N + j]; if (w > 1e-9) { pr.push(i, j); al.push(w / a.wmax); } }
    v.setEdges(Uint32Array.from(pr), Float32Array.from(al));
    v.setCorner(`classical MDS of $D_{ij}$, training bank<br>3 axes carry ${ui.fmtPct(a.md.explained, 0)} of the positive spectrum`);
    ui.renderMath(cornerEl);
  }
  const statCache = {};
  const setStat = (k, val, sub) => { const sig = `${val}|${sub}`; if (statCache[k] === sig) return; statCache[k] = sig; stats.set(k, val, sub); };
  function showStats(a) {
    const nSing = a.cs.st.filter(c => c.size === 1).length;
    setStat("k", a.cs.K, `${a.cs.multi} with ≥ 2 members · ${nSing} singleton${nSing === 1 ? "" : "s"}`);
    setStat("big", a.cs.largest, `of N = ${a.N}`);
    setStat("span", +a.spanR.toPrecision(3), a.spanR > 1 ? "a chain: wider than one edge's range" : "within one edge's range");
    setStat("J", +a.J.toPrecision(5), `b = ${ui.fmt(a.b, 3)}, α = ${p.a}`);
    setStat("r", +a.residual.toPrecision(2), a.residual < M.RES_TOL ? "below 10⁻⁶: fixed point" : "still moving");
    setStat("eq", +a.eq.toPrecision(2), "max |∂E/∂u|, exact pattern solve");
  }
  function drawMatrix(a, D2) {
    const N = a.N, cs = a.cs, ord = cs.order, vals = new Float32Array(N * N), b2 = 2 * a.b;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) vals[r * N + c] = r === c ? NaN : Math.log10(Math.max(D2[ord[r] * N + ord[c]], 1e-12) / b2);
    const groups = []; let start = 0;
    while (start < N) {
      const k = cs.labels[ord[start]]; let end = start; while (end < N && cs.labels[ord[end]] === k) end++;
      if (cs.st[k].size === 1) { let e2 = start; while (e2 < N && cs.st[cs.labels[ord[e2]]].size === 1) e2++; groups.push({ start, end: e2, color: P.interior, label: "singletons" }); start = e2; continue; }
      groups.push({ start, end, color: compColor(cs, ord[start]), label: cs.rank[k] < 8 ? `c${cs.rank[k] + 1}` : "" });
      start = end;
    }
    order = ord;
    hm.update({ values: vals, rows: N, cols: N, groups: { rows: groups } });
  }
  function drawTable(a) {
    const multi = a.cs.ids.filter(k => a.cs.st[k].size > 1).slice(0, 8);
    const ev = evalStats && evalFor === a.stamp ? evalStats : null;
    const b2 = Math.sqrt(2 * a.b);
    const head = ui.h("tr", {}, ...["", "n", "max D", "mean D", "span/√(2b)"].map((t, i) => ui.h("th", { class: i ? "num" : null, text: t })));
    const rows = multi.map(k => {
      const c = a.cs.st[k], e = ev ? ev[k] : null, pair = (x, y) => `${ui.fmt(x, 2)} / ${y == null ? "…" : ui.fmt(y, 2)}`;
      return ui.h("tr", {}, ui.h("td", {}, ui.h("span", { class: "swatch round", style: { "--sw": compColor(a.cs, c.members[0]) } })),
        ui.h("td", { class: "num", text: String(c.size) }), ui.h("td", { class: "num", text: pair(c.diamMax, e && e.diamMax) }),
        ui.h("td", { class: "num", text: pair(c.diamMean, e && e.diamMean) }), ui.h("td", { class: "num", text: pair(c.span / b2, e && e.span / b2) }));
    });
    const nSing = a.cs.st.filter(c => c.size === 1).length, more = a.cs.multi - multi.length;
    tab.replaceChildren(rows.length ? ui.h("div", { class: "table-scroll" }, ui.h("table", { class: "data" }, ui.h("thead", {}, head), ui.h("tbody", {}, ...rows))) : ui.h("p", { class: "u-help", text: "No component has two or more members." }),
      ui.h("p", { class: "u-help", text: `${more > 0 ? `and ${more} more multi-member components; ` : ""}${nSing} singleton${nSing === 1 ? "" : "s"}. D: RMS distance of a member to its component's mean response; span: largest D between two members.${ev ? "" : " Evaluation bank: computing…"}` }));
  }
  function evalBank(a) {
    const sim = sys.sim, N = sim.net.N;
    const e = O.equilibrate(sim.net, sys.Ge, sys.Se, { U0: Ueval && Ueval.length === sys.Se * N ? Ueval : null });
    Ueval = e.U;
    evalStats = M.componentsOn(Ueval, sys.Se, N, a.cs); evalFor = a.stamp;
  }
  function drawRecord() {
    const ser = ["cold", "up", "down"].map(br => { const q = points.filter(x => x.branch === br); return { name: `${BR_LABEL[br]}, live`, x: q.map(x => x.b), y: q.map(x => x.K), color: BR[br], points: true, r: 4 }; });
    const ref = sweepRef();
    rec.update({ series: [...ref, ...ser], yrange: [0.8, p.n * 1.6], cursor: sys ? sys.sim.b : null });
    const last = points.slice(-5).reverse();
    recList.innerHTML = last.length ? "Latest: " + last.map(x => `<span style="white-space:nowrap"><span class="swatch round" style="--sw:${BR[x.branch]}"></span> b = ${ui.fmt(x.b, 2)}: K = ${x.K}, J = ${ui.fmt(x.J, 4)}</span>`).join(" · ") : "No fixed point reached yet with these settings.";
    if (ref.length) recList.innerHTML += `<br>Thin lines: the precomputed sweep's seed means for this $\\Sigma_g$ family, potential and $\\alpha$ (B = ${F.meta.config.B}, N = ${F.meta.config.N}, s = ${F.meta.config.s}; different random draws).`;
    ui.renderMath(recList);
  }
  function sweepRef() {
    const sg = M.sigmaById(p.sig), fam = sg.family;
    if (sg.B !== F.meta.config.B || p.n !== F.meta.config.N || p.s !== F.meta.config.s || !F.meta.config.alphas.includes(p.a) || (fam !== "blocks" && fam !== "power")) return [];
    const rs = F.runs.filter(r => r.sigma === fam && r.pot === p.pot && r.alpha === p.a);
    if (!rs.length) return [];
    const bg = F.meta.config.b_grid;
    return ["cold", "up", "down"].map(br => ({ name: `sweep ${br}, seed mean`, x: bg, y: bg.map((_, k) => rs.reduce((s, r) => s + r[br].K[k], 0) / rs.length), color: BR[br], width: 1.1, dash: true }));
  }

  // ------------------------------------------------------------ loop
  let stamp = 0;
  function show(a, W, D2, heavy) {
    draw3d(a, W); showStats(a); cur = a;
    if (heavy) { drawMatrix(a, D2); drawTable(a); rec.setCursor(a.b); }
  }
  function tick(speed) {
    const sim = sys.sim;
    if (!(sim.residual < M.RES_TOL)) sim.stepFor(25, speed, 0);
    T = stepsT0 + (sim.steps - steps0) * sim.dt;
    const a = analyse(); a.stamp = ++stamp; tickN++;
    const done = a.residual < M.RES_TOL;
    if (done) evalBank(a);
    show(a, sim.net.W, sim.D2, done || tickN % 8 === 0);
    if (!done) return frame(a);
    if (pointsKey !== configKey()) { points = []; pointsKey = configKey(); }
    const last = points[points.length - 1];
    if (!last || last.b !== a.b || last.branch !== branch || last.K !== a.cs.K) points.push({ b: a.b, K: a.cs.K, J: a.J, branch });
    drawRecord();
    if (path && path.i < path.bs.length - 1) { path.i++; setB(path.bs[path.i], false); return frame(a); }
    path = null;
    const f = frame(a); f.done = true; f.doneLabel = "fixed point: r < 10⁻⁶"; return f;
  }
  let stepsT0 = 0, steps0 = 0;
  function frame(a) { return { t: T, W: Float64Array.from(sys.sim.net.W), b: sys.sim.b, branch }; }
  function seek(f) {
    // redraw a recorded moment: re-solve the equilibrium for that W (one solve) without touching the sim
    const net = O.cloneNet(sys.sim.net); net.W.set(f.W);
    const e = O.equilibrate(net, sys.sim.G, sys.sim.S), N = net.N, D2 = O.dist2(e.U, sys.sim.S, N);
    const fake = { net, U: e.U, S: sys.sim.S, D2, b: f.b, alpha: sys.sim.alpha, energy: () => O.energy(net, e.U, sys.sim.G, sys.sim.S) };
    const md = O.mds(D2, N, 3, prevX); prevX = md.X;
    const key = new Float64Array(N); for (let i = 0; i < N; i++) key[i] = md.X[i * 3];
    const cs = M.components(fake, key);
    let wmax = 0; for (let k = 0; k < N * N; k++) wmax = Math.max(wmax, f.W[k]);
    let span = 0; cs.st.forEach(c => { span = Math.max(span, c.span); });
    const dW = O.driftFromD2(f.W, D2, N, f.b, sys.sim.alpha);
    const a = { N, md, cs, wmax, spanR: span / Math.sqrt(2 * f.b), b: f.b, residual: O.projectedResidual(f.W, dW, N), eq: e.residual, J: M.objectiveJ(fake), stamp: ++stamp };
    evalStats = null; show(a, f.W, D2, true);
  }
  function restore(f) {
    sys.sim.net.W.set(f.W); sys.sim.setParams({ b: f.b }); sys.sim.refresh();
    p.b = f.b; lastB = f.b; branch = f.branch; bS.set(f.b); path = null;
    stepsT0 = f.t; steps0 = sys.sim.steps; T = f.t;
    writeURL();
  }
  function reset() {
    sys = M.buildLive(p); prevX = null; Ueval = null; evalStats = null; branch = "cold"; lastB = p.b;
    stepsT0 = 0; steps0 = sys.sim.steps; T = 0; tickN = 0;
    if (pointsKey !== configKey()) { points = []; pointsKey = configKey(); }
    const a = analyse(); a.stamp = ++stamp;
    show(a, sys.sim.net.W, sys.sim.D2, true); v.fit(); drawRecord();
    return frame(a);
  }

  // ------------------------------------------------------------ controls
  function setB(x, byHand) {
    if (byHand) { path = null; untour(); }
    if (x !== lastB) branch = x > lastB ? "up" : "down";
    lastB = x; p.b = x; bS.set(x);
    sys.sim.setParams({ b: x });
    writeURL();
    if (byHand) tp.play();
  }
  function structural() { untour(); path = null; writeURL(); tp.reset(); tp.play(); }
  function untour() { if (!tour) return; tour = null; tours.moved(); }
  function writeURL() {
    const o = { tour: tour ? tour.id : null };
    for (const [k, u] of Object.entries(KEYS)) o[u] = tour || p[k] === M.LIVE_DEFAULTS[k] ? null : p[k];
    ctx.setParams(o);
  }
  function syncControls() { sigS.set(p.sig); bS.set(p.b); aS.set(p.a); potS.set(p.pot); nS.set(p.n); sS.set(p.s); seedS.set(p.seed); }
  function applyTour(t) {
    tour = t; p = { ...M.LIVE_DEFAULTS, ...t.p };
    path = t.path ? { bs: t.path, i: 0 } : null;
    if (path) p.b = path.bs[0];
    syncControls(); writeURL(); tp.reset(); tp.play();
  }
  function readURL() {
    const t = M.liveTourById(ctx.param("tour", ""));
    if (t) { tour = t; p = { ...M.LIVE_DEFAULTS, ...t.p }; path = t.path ? { bs: t.path, i: 0 } : null; if (path) p.b = path.bs[0]; return; }
    tour = null; path = null;
    const d = M.LIVE_DEFAULTS;
    p = { sig: M.sigmaById(ctx.param("sig", d.sig)).id, b: Math.min(2, Math.max(0.002, ctx.param("b", d.b))), a: [0.1, 1, 10].includes(ctx.param("a", d.a)) ? ctx.param("a", d.a) : d.a,
      pot: ctx.param("lp", d.pot) === "rect" ? "rect" : "linear", n: Math.min(60, Math.max(20, Math.round(ctx.param("n", d.n) / 10) * 10)),
      s: Math.min(6, Math.max(1, Math.round(ctx.param("s", d.s)))), seed: Math.min(20, Math.max(0, Math.round(ctx.param("seed", d.seed)))) };
  }
  ctx.onParams(() => {
    if (!ctx.params.has("tour") && !Object.values(KEYS).some(k => ctx.params.has(k))) return;   // a link for another figure
    readURL(); syncControls();
    if (tour) tours.set(tour.id); else tours.clear();
    tp.reset(); tp.play();
  });
  ctx.onShow(() => { if (cur) v.render(); });

  // ------------------------------------------------------------ tooltips
  function tipNeuron(i) {
    const a = cur; if (!a || !sys) return null;
    const net = sys.sim.net, B = net.B, c = a.cs.st[a.cs.labels[i]], rows = [];
    rows.push({ name: "component", value: c.size === 1 ? "singleton" : `c${a.cs.rank[a.cs.labels[i]] + 1}, ${c.size} neurons`, color: compColor(a.cs, i) });
    if (c.size > 1) { let d = 0; for (let s = 0; s < sys.sim.S; s++) { const x = sys.sim.U[s * net.N + i] - c.proto[s]; d += x * x; } rows.push({ name: "D to its prototype", value: ui.fmt(Math.sqrt(d / sys.sim.S), 3) }); }
    const att = []; for (let b = 0; b < B; b++) if (net.A[i * B + b] > 0) att.push(`g${b + 1}: ${ui.fmt(net.A[i * B + b], 2)}`);
    rows.push({ name: "attachments", value: att.join(", ") });
    if (p.pot === "rect") rows.push({ name: "threshold tᵢ", value: ui.fmt(net.t[i], 3) });
    return ui.tipRows(`neuron ${i}`, rows);
  }
  function tipPair(r, c) {
    if (!order || !sys) return null;
    const i = order[r], j = order[c], N = sys.sim.net.N, D2 = sys.sim.D2[i * N + j], w = sys.sim.net.W[i * N + j];
    return ui.tipRows(`neurons ${i} and ${j}`, [{ name: "D²ᵢⱼ (training bank)", value: ui.fmt(D2, 3) }, { name: "2b", value: ui.fmt(2 * sys.sim.b, 3) },
      { name: "wᵢⱼ", value: w > 1e-9 ? ui.fmt(w, 3) : "0 (dead)" }, { name: "same component", value: cur && cur.cs.labels[i] === cur.cs.labels[j] ? "yes" : "no" }]);
  }

  // ------------------------------------------------------------ go
  tp.record(reset());
}
