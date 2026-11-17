// Chapter "idea": the front door. One live figure (a small network: drag an input and watch the voltages relax;
// one neuron's on-site potential and transfer curve; a pair's response functions; the slow rule with its fixed
// point; the envelope identity checked by finite differences), plus a few numbers read from fusion.json.
// Model side in ch_idea_model.js, drawing in ch_idea_draw.js.
import * as ui from "./ui.js";
import { P, catColor } from "./palette.js";
import * as O from "./onsite_engine.js";
import * as M from "./ch_idea_model.js";
import { miniPlot, networkView } from "./ch_idea_draw.js";

const { N, B, S } = M;
const VRANGE = [-2.5, 2.5];
const KEYS = ["g1", "g2", "g3", "b", "a", "pot", "k", "st", "seed"];

export async function init(panel, ctx) {
  fillSweepNumbers(ctx);           // not awaited: the figure must not wait for the fetch
  initFigure(ctx);
}

// ------------------------------------------------------------ numbers from the precomputed sweep
async function fillSweepNumbers(ctx) {
  const spans = ctx.article.querySelectorAll("[data-iv]");
  try {
    const r = await fetch("data/sweeps2/fusion.json");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const s = (await r.json()).summary;
    const v = { runs: ui.fmtInt(s.n_runs), conv: ui.fmtInt(s.n_converged), res: ui.fmt(s.max_residual, 2),
      dev: ui.fmt(s.max_dev_live_converged, 2), slack: ui.fmt(s.min_slack_dead_converged, 2), env: ui.fmt(s.max_envelope_err, 2) };
    spans.forEach(sp => { sp.textContent = v[sp.dataset.iv] ?? "n/a"; });
  } catch (e) {
    spans.forEach(sp => { sp.textContent = "(not loaded)"; });
  }
}

// ------------------------------------------------------------ the live figure
function readParams(ctx) {
  const tourId = ctx.param("tour", null), tour = M.TOURS.find(t => t.id === tourId) || null;
  const p = {};
  for (const k of KEYS) p[k] = ctx.param(k, M.DEFAULTS[k]);
  if (p.pot !== "lin") p.pot = "rect";
  p.b = clamp(p.b, ...M.LIMITS.b); p.a = clamp(p.a, ...M.LIMITS.a);
  for (const k of ["g1", "g2", "g3"]) p[k] = clamp(p[k], ...M.LIMITS.g);
  p.seed = Math.round(clamp(p.seed, 0, 99)); p.k = clamp(p.k, 0, 20); p.st = clamp(p.st, 0, 2);
  if (tour) Object.assign(p, tour.params);
  const i = Math.round(ctx.param("i", 0)) - 1, j = Math.round(ctx.param("j", 0)) - 1;
  const sel = i >= 0 && i < N && j >= 0 && j < N && i !== j ? [i, j] : null;
  return { p, tour, sel };
}
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

function initFigure(ctx) {
  let { p, tour, sel } = readParams(ctx);
  let m = M.build(p), sim = M.createSim(m, p);
  let g = [p.g1, p.g2, p.g3];
  let uStar = M.equilibriumOne(m.net, g), uDisp = Float64Array.from(uStar);
  let tSlow = 0, tau = 0, acc = 0, tickN = 0, env = null, relaxAnim = null;
  if (!sel) sel = M.closestPair(sim.D2);
  const fig = ctx.slot("sim").closest("figure");

  // ---------------------------------------------------------- layout
  const L = ui.simLayout({ parent: ctx.slot("sim") });
  const netBox = ui.h("div");
  L.view.append(netBox);
  let nv = makeNetwork();
  ui.colorbar({ parent: L.view, map: "div", range: VRANGE, label: "voltage: inputs $g_b$ (squares), neurons $u_i$ (circles)", ticks: [-2, -1, 0, 1, 2], fmt: x => ui.fmt(x, 2) });
  const tp = ui.transport({
    parent: L.view, watch: fig, autoplay: false, maxHistory: 300,
    speeds: [{ value: 0.25, label: "×¼" }, { value: 1, label: "×1" }, { value: 4, label: "×4" }, { value: 16, label: "×16" }], speed: 1,
    timeFmt: t => `slow time t = ${ui.fmt(t, 3)}`,
    onTick: sp => tick(sp),
    onSeek: f => { setW(f.W); tSlow = f.t; afterW(true); },
    onRestore: f => { setW(f.W); tSlow = f.t; afterW(true); },
    onReset: () => restart(),
  });

  const tours = ui.chips({ parent: L.side, label: "Tours", items: M.TOURS, active: tour ? tour.id : null, onPick: t => applyTour(t) });

  const gS = [0, 1, 2].map(b => ui.slider({ label: `input $g_${b + 1}$`, min: M.LIMITS.g[0], max: M.LIMITS.g[1], step: 0.05, value: g[b], fmt: x => ui.fmtFixed(x, 2),
    onInput: x => { g[b] = x; stimulusChanged(); } }));
  const btnRow = ui.h("div", { class: "u-chip-row" });
  const bDraw = ui.h("button", { type: "button", class: "u-chip", text: "draw g from p_g" });
  const bZero = ui.h("button", { type: "button", class: "u-chip", text: "relax again from u = 0" });
  btnRow.append(bDraw, bZero);
  const rng = O.makeRng(4242);
  bDraw.addEventListener("click", () => { for (let b = 0; b < B; b++) { g[b] = clamp(rng.normal(), ...M.LIMITS.g); gS[b].set(g[b]); } stimulusChanged(); });
  bZero.addEventListener("click", () => { uDisp.fill(0); startRelax(); });
  ui.controls({ parent: L.side, title: "Stimulus (fast dynamics)", primary: [...gS, btnRow] });

  const sB = ui.slider({ label: "fusion scale $b$", min: M.LIMITS.b[0], max: M.LIMITS.b[1], log: true, sig: 2, value: p.b, fmt: x => ui.fmt(x, 2),
    help: "An edge can live only where $D_{ij}^2 &lt; 2b$.", onInput: x => { p.b = x; slowParams(); } });
  const sA = ui.slider({ label: "stiffness $\\alpha$", min: M.LIMITS.a[0], max: M.LIMITS.a[1], log: true, sig: 2, value: p.a, fmt: x => ui.fmt(x, 2),
    help: "A live edge reaches at most $b/\\alpha$; each neuron's attachments total 1.", onInput: x => { p.a = x; slowParams(); } });
  const segP = ui.segmented({ label: "on-site potential", options: [{ value: "lin", label: "none (linear)" }, { value: "rect", label: "leaky rectifier" }], value: p.pot,
    onChange: v => { p.pot = v; potentialChanged(); } });
  const sK = ui.slider({ label: "slope below threshold $k_{lo}$", min: 0, max: 10, step: 0.5, value: p.k, fmt: x => ui.fmtFixed(x, 1),
    help: "$V_i(u) = \\tfrac{k_{lo}}{2}(u - t_i)_-^2$ ($k_{hi} = 0$). Used when the potential is the leaky rectifier.", onInput: x => { p.k = x; potentialChanged(); } });
  const sT = ui.slider({ label: "threshold spread $\\sigma_t$", min: 0, max: 1.5, step: 0.05, value: p.st, fmt: x => ui.fmtFixed(x, 2),
    help: "$t_i = \\sigma_t z_i$ with fixed $z_i \\sim \\mathcal N(0,1)$.", onInput: x => { p.st = x; potentialChanged(); } });
  const sSeed = ui.slider({ label: "seed (attachments, thresholds, bank)", min: 0, max: 40, step: 1, value: p.seed, fmt: x => String(x),
    onChange: x => { p.seed = x; rebuild(); } });
  ui.controls({ parent: L.side, title: "Learning (slow dynamics)", primary: [sB, sA, segP], advanced: [sK, sT, sSeed], advancedLabel: "More controls" });

  const stats = ui.stats({ parent: L.side, items: [
    { key: "fast", label: "relaxation $\\max_i |\\partial E/\\partial u_i|$", fmt: x => ui.fmt(x, 2), help: "For the stimulus shown, at the voltages drawn (current units). Zero at the equilibrium." },
    { key: "K", label: "components", fmt: String, help: "Connected components of the live graph (w > 1e-9), singletons included." },
    { key: "r", label: "plasticity residual $r$", fmt: x => ui.fmt(x, 2), help: "max |dw/dt| over pairs, projected (a dead edge pushed below zero counts 0). A fixed point is claimed below 1e-6." },
    { key: "eq", label: "neuronal residual (bank)", fmt: x => ui.fmt(x, 2), help: "max over neurons and the 256 training stimuli of |dE/du_i| after the exact equilibrium solve." },
    { key: "env", label: "envelope check", fmt: x => ui.fmt(x, 2), help: "max over the 28 pairs of |central difference of E*(W) in w_ij (step 1e-4) − D_ij²/2|." },
    { key: "w", label: "strongest edge $\\max w$", fmt: x => ui.fmt(x, 3), help: "Compare with b/α, the most an edge can reach, and with the attachment total 1." },
  ] });
  ui.legend({ parent: L.side, items: [
    { label: "input $g_b$, clamped (drag it); outline: its colour", color: catColor(0), shape: "square" },
    { label: "interior neuron, fill: its voltage $u_i$", color: P.interior, shape: "dot" },
    { label: "attachment $a_{ib}$ (fixed), width $\\propto a_{ib}$", color: catColor(1), shape: "line" },
    { label: "synapse $w_{ij}$ (learned), opacity $w_{ij}/\\max w$", color: P.edge, shape: "line" },
    { label: "selected neuron $i$ (solid ring), partner $j$ (dashed)", color: P.accent, shape: "ring" },
  ] });

  // ---------------------------------------------------------- linked panels
  // wide screens: the panels sit under the network, next to the controls; one column: after the controls
  const grid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 230px), 1fr))", gap: "18px 24px", marginTop: "22px" } });
  const oneCol = window.matchMedia("(max-width: 900px)");
  const placeGrid = () => (oneCol.matches ? L.below : L.view).append(grid);
  placeGrid(); oneCol.addEventListener("change", placeGrid);
  const box = (title) => {
    const t = ui.h("div", { class: "chart-title" }), b = ui.h("div");
    grid.append(ui.h("div", { style: { minWidth: "0" } }, t, b));
    return { t, b };
  };
  const pa = box(), pb = box(), pc = box(), pd = box(), pe = box();
  const plA = miniPlot(pa.b, { height: 200, xlabel: "voltage u", ylabel: "V(u)" });
  const plB = miniPlot(pb.b, { height: 200, xlabel: "input current I", ylabel: "voltage u" });
  const plC = miniPlot(pc.b, { height: 200, xlabel: "F_i(g)", ylabel: "F_j(g)" });
  const plD = miniPlot(pd.b, { height: 200, xlabel: "D²_ij", ylabel: "w_ij" });
  const plE = miniPlot(pe.b, { height: 200, xlabel: "D²_ij / 2 (local)", ylabel: "∂E*/∂w_ij (global)" });
  function titles() {
    const [i, j] = sel;
    pa.t.innerHTML = `(a) On-site potential of neuron ${i + 1}`;
    pb.t.innerHTML = `(b) Transfer curve of neuron ${i + 1}`;
    pc.t.innerHTML = `(c) Responses of neurons ${i + 1} and ${j + 1}`;
    pd.t.innerHTML = "(d) The fixed point, all 28 pairs";
    pe.t.innerHTML = "(e) Envelope identity, all 28 pairs";
  }

  // ---------------------------------------------------------- dynamics
  function makeNetwork() {
    netBox.replaceChildren();
    return networkView(netBox, { A: m.net.A, N, B, range: VRANGE, getG: b => g[b],
      onPick: i => pick(i),
      onDrag: (b, v) => { g[b] = clamp(+v.toFixed(2), ...M.LIMITS.g); gS[b].set(g[b]); stimulusChanged(true); },
      onDragEnd: () => writeUrl(),
      tip: i => neuronTip(i) });
  }
  function setW(W) { m.net.W.set(W); sim.refresh(); sim.lastChange = NaN; }
  function restart() {
    m.net.W.fill(0); sim.refresh(true); tSlow = 0; acc = 0; sim.lastChange = NaN;
    afterW(true);
    return frame();
  }
  const frame = () => ({ t: tSlow, W: Float64Array.from(m.net.W) });
  function tick(sp) {
    acc += sp; const n = Math.floor(acc); acc -= n;
    if (n > 0) { sim.step(n); tSlow += n * sim.dt; }
    tickN++;
    const done = sim.lastChange < M.STOP_TOL;          // NaN (no step since the last change) is never done
    afterW(done || tickN % 6 === 0);
    return { ...frame(), done, doneLabel: "fixed point" };
  }
  // W changed: the shown stimulus jumps to its new equilibrium (no relaxation animation while learning)
  function afterW(withEnvelope) {
    stopRelax();
    uStar = M.equilibriumOne(m.net, g); uDisp.set(uStar); tau = 0;
    if (withEnvelope) env = M.envelopeCheck(m.net, m.G, sim.D2);
    drawAll();
  }
  function stimulusChanged(noUrl) {
    uStar = M.equilibriumOne(m.net, g);
    drawSlow();                                        // panel (c) marks the new stimulus's equilibrium
    if (tp.playing || ui.reducedMotion()) { uDisp.set(uStar); drawFast(); } else startRelax();
    if (!noUrl) writeUrl();
    tours.moved(); ctx.setParams({ tour: null });
  }
  function startRelax() {
    stopRelax(); tau = 0;
    if (ui.reducedMotion()) { uDisp.set(uStar); drawFast(); return; }
    relaxAnim = ui.animate(fig, () => {
      M.relax(m.net, uDisp, g, 0.12); tau += 0.12;
      const res = M.maxAbs(M.gradE(m.net, uDisp, g));
      if (res < 1e-9 || tau > 60) { uDisp.set(uStar); stopRelax(); }
      drawFast();
    });
  }
  function stopRelax() { if (relaxAnim) { relaxAnim.stop(); relaxAnim = null; } }

  function slowParams() {
    sim.setParams({ b: p.b, alpha: p.a, dt: M.dtOf(p.a) }); sim.lastChange = NaN;
    tours.moved(); writeUrl(true);
    drawSlow();
  }
  function potentialChanged() {
    M.setPotential(m, p); sim.refresh(); sim.lastChange = NaN;
    tours.moved(); writeUrl(true);
    afterW(true);
  }
  function rebuild() {
    tp.pause(); stopRelax();
    m = M.build(p); sim = M.createSim(m, p);
    nv = makeNetwork();
    sel = M.closestPair(sim.D2); titles();
    tours.moved(); writeUrl(true);
    tp.reset();
  }
  function pick(i) {
    if (i === sel[0]) return;
    sel = [i, sel[0]]; titles();
    ctx.setParams({ i: sel[0] + 1, j: sel[1] + 1 });
    drawAll();
  }
  function pickPair(q) {
    const [i, j] = M.PAIRS[q]; sel = [i, j]; titles();
    ctx.setParams({ i: i + 1, j: j + 1 });
    drawAll();
  }
  function writeUrl(dropTour) {
    const o = {}; for (const k of KEYS) o[k] = k === "g1" ? g[0] : k === "g2" ? g[1] : k === "g3" ? g[2] : p[k];
    o.i = sel[0] + 1; o.j = sel[1] + 1;
    if (dropTour) o.tour = null;
    ctx.setParams(o);
  }

  // ---------------------------------------------------------- tours
  function applyTour(t) {
    tp.pause(); stopRelax();
    const seedChanged = t.params.seed != null && t.params.seed !== p.seed;
    Object.assign(p, t.params);
    if (t.params.g1 != null) { g = [p.g1, p.g2, p.g3]; gS.forEach((s, b) => s.set(g[b])); }
    sB.set(p.b); sA.set(p.a); segP.set(p.pot); sK.set(p.k); sT.set(p.st); sSeed.set(p.seed);
    if (seedChanged) { m = M.build(p); sim = M.createSim(m, p); nv = makeNetwork(); sel = M.closestPair(sim.D2); titles(); }
    else { M.setPotential(m, p); sim.setParams({ b: p.b, alpha: p.a, dt: M.dtOf(p.a) }); }
    tp.reset();
    ctx.setParams({ ...Object.fromEntries(KEYS.map(k => [k, k[0] === "g" && k.length === 2 ? g[+k[1] - 1] : p[k]])), i: sel[0] + 1, j: sel[1] + 1, tour: t.id });
    if (t.relax) { uDisp.fill(0); startRelax(); }
    if (t.run) { tp.setSpeed(t.speed || 1); tp.play(); }
  }
  ctx.onParams(() => {
    const r = readParams(ctx);
    if (r.tour) { tours.set(r.tour.id); applyTour(r.tour); return; }
    const seedChanged = r.p.seed !== p.seed;
    p = r.p; g = [p.g1, p.g2, p.g3]; if (r.sel) sel = r.sel;
    gS.forEach((s, b) => s.set(g[b])); sB.set(p.b); sA.set(p.a); segP.set(p.pot); sK.set(p.k); sT.set(p.st); sSeed.set(p.seed);
    if (seedChanged) rebuild(); else { M.setPotential(m, p); sim.setParams({ b: p.b, alpha: p.a, dt: M.dtOf(p.a) }); sim.refresh(); sim.lastChange = NaN; afterW(true); }
    titles();
  });
  ctx.onHide(() => { tp.pause(); stopRelax(); });

  // ---------------------------------------------------------- drawing
  function drawAll() { drawFast(); drawSlow(); }
  function drawFast() {
    nv.update({ u: uDisp, g, W: m.net.W, sel });
    const [i] = sel, net = m.net;
    stats.set("fast", M.maxAbs(M.gradE(net, uDisp, g)), tau > 0 || relaxAnim ? `fast time τ = ${ui.fmtFixed(tau, 1)}` : "at equilibrium");
    // (a) V_i(u)
    const us = [], Vs = [], Is = [], lin = [];
    const d = M.degree(net, i);
    for (let k = 0; k <= 160; k++) { const u = -3 + 6 * k / 160; us.push(u); Vs.push(M.Vof(net, i, u)); Is.push(d * u + M.dVof(net, i, u)); lin.push(u); }
    const ui_ = uDisp[i], Ii = M.current(net, uDisp, g, i), Istar = M.current(net, uStar, g, i);
    const Vmax = Math.max(0.5, M.Vof(net, i, -3)), hasV = net.klo[i] > 0 || net.khi[i] > 0;
    plA.draw({ note: `u${sub(i + 1)} = ${ui.fmtFixed(ui_, 3)}`, xr: [-3, 3], yr: [-0.04 * Vmax, Vmax * 1.04], layers: [
      ...(hasV ? [{ type: "vline", v: net.t[i], label: `t${sub(i + 1)}`, color: P.mute }] : []),
      { type: "line", x: us, y: Vs, color: P.fg },
      { type: "dots", x: [ui_], y: [M.Vof(net, i, ui_)], color: P.accent, r: 5, stroke: P.surface,
        tip: () => ui.tipRows(`neuron ${i + 1}`, [{ name: "u_i", value: ui.fmt(ui_, 3) }, { name: "V_i(u_i)", value: ui.fmt(M.Vof(net, i, ui_), 3) }, { name: "threshold t_i", value: ui.fmt(net.t[i], 3) }]) },
    ] });
    // (b) transfer curve: u against I = d_i u + V_i'(u); dashed: no potential, u = I/d_i
    plB.draw({ xr: [Math.min(...Is) * 1.04, Math.max(...Is) * 1.04], yr: [-3, 3], note: `leak d${sub(i + 1)} = ${ui.fmtFixed(d, 3)}` + (hasV ? `, threshold t${sub(i + 1)} = ${ui.fmtFixed(net.t[i], 2)}` : ", no potential"), layers: [
      ...(hasV ? [{ type: "hline", v: net.t[i], label: `t${sub(i + 1)}`, color: P.mute }] : []),
      { type: "line", x: lin.map(u => d * u), y: lin, color: P.mute2, dash: "5 4", width: 1.2 },
      { type: "line", x: Is, y: us, color: P.fg },
      { type: "dots", x: [Istar], y: [uStar[i]], color: P.fg, ring: true, r: 6 },
      { type: "dots", x: [Ii], y: [ui_], color: P.accent, r: 5, stroke: P.surface,
        tip: () => ui.tipRows(`neuron ${i + 1}`, [{ name: "input current I_i", value: ui.fmt(Ii, 3) }, { name: "voltage u_i", value: ui.fmt(ui_, 3) }, { name: "equilibrium u_i*", value: ui.fmt(uStar[i], 3) }]) },
    ] });
  }
  function drawSlow() {
    const net = m.net, [i, j] = sel, U = sim.U, D2 = sim.D2;
    // (c) the pair's responses over the training bank
    const xi = new Array(S), yj = new Array(S);
    let lo = Infinity, hi = -Infinity;
    for (let s = 0; s < S; s++) { xi[s] = U[s * N + i]; yj[s] = U[s * N + j]; lo = Math.min(lo, xi[s], yj[s]); hi = Math.max(hi, xi[s], yj[s]); }
    const pad = 0.06 * (hi - lo || 1);
    plC.draw({ xr: [lo - pad, hi + pad], yr: [lo - pad, hi + pad], equal: true,
      note: `D² = ${ui.fmt(D2[i * N + j], 3)},  w = ${ui.fmt(net.W[i * N + j], 3)}`, layers: [
        { type: "line", x: [lo - 2 * pad, hi + 2 * pad], y: [lo - 2 * pad, hi + 2 * pad], color: P.mute2, dash: "4 3", width: 1 },
        { type: "dots", x: xi, y: yj, color: P.cat[0], r: 2, opacity: 0.55 },
        { type: "dots", x: [uStar[i]], y: [uStar[j]], color: P.accent, r: 5, stroke: P.surface },
      ] });
    // (d) w against D^2, every pair, on the fixed-point line
    const xmax = 4 * p.b, wcap = p.b / p.a;
    const px = [], py = [], pcol = [], idx = [];
    let beyond = 0;
    M.PAIRS.forEach(([a, c], q) => { const d2 = D2[a * N + c]; if (d2 > xmax) { beyond++; return; } px.push(d2); py.push(net.W[a * N + c]); pcol.push(net.W[a * N + c] > 1e-9 ? P.cat[0] : P.other); idx.push(q); });
    const sq = M.PAIRS.findIndex(([a, c]) => (a === Math.min(i, j) && c === Math.max(i, j)));
    const sd2 = D2[i * N + j];
    plD.draw({ xr: [0, xmax], yr: [-0.04 * wcap, 1.08 * wcap], note: `b = ${ui.fmt(p.b, 2)}, α = ${ui.fmt(p.a, 2)}` + (beyond ? `; ${beyond} pair${beyond > 1 ? "s" : ""} beyond 4b not shown` : ""),
      layers: [
        { type: "vline", v: 2 * p.b, label: "2b", color: P.mute },
        { type: "line", x: [0, 2 * p.b, xmax], y: [wcap, 0, 0], color: P.fg, width: 1.4 },
        { type: "dots", x: px, y: py, color: pcol, r: 3.5, tip: k => pairTip(idx[k]), click: k => pickPair(idx[k]) },
        ...(sd2 <= xmax ? [{ type: "dots", x: [sd2], y: [net.W[i * N + j]], color: P.accent, ring: true, r: 7 }] : []),
      ] });
    // (e) envelope identity
    if (env) {
      let emax = 0; for (let q = 0; q < env.half.length; q++) emax = Math.max(emax, env.half[q], env.fd[q]);
      emax = emax * 1.05 || 1;
      plE.draw({ note: `max |difference| = ${ui.fmt(env.err, 2)}`, xr: [0, emax], yr: [0, emax], layers: [
        { type: "line", x: [0, emax], y: [0, emax], color: P.mute2, dash: "4 3", width: 1 },
        { type: "dots", x: Array.from(env.half), y: Array.from(env.fd), color: M.PAIRS.map(([a, c]) => net.W[a * N + c] > 1e-9 ? P.cat[0] : P.other), r: 3.5,
          tip: k => pairTip(k), click: k => pickPair(k) },
        ...(sq >= 0 ? [{ type: "dots", x: [env.half[sq]], y: [env.fd[sq]], color: P.accent, ring: true, r: 7 }] : []),
      ] });
    }
    // tiles
    const c = O.components(net.W, N);
    let wmax = 0; for (const w of net.W) wmax = Math.max(wmax, w);
    const fp = M.fixedPointErrors(net, D2, p.b, p.a);
    stats.set("K", c.K, `${fp.live} live edge${fp.live === 1 ? "" : "s"} of 28`);
    stats.set("r", sim.residual, sim.residual < M.CLAIM_TOL ? "fixed point (r &lt; 10⁻⁶)" : "not stationary");
    stats.set("eq", sim.eqResidual, "exact piecewise-linear solve");
    if (env) stats.set("env", env.err, `finite-difference step ${ui.fmt(M.FD_STEP, 1)}`);
    stats.set("w", wmax, `b/α = ${ui.fmt(wcap, 3)}`);
  }
  function pairTip(q) {
    const [a, c] = M.PAIRS[q], net = m.net, d2 = sim.D2[a * N + c], w = net.W[a * N + c];
    const rows = [{ name: "D²_ij", value: ui.fmt(d2, 4) }, { name: "w_ij", value: ui.fmt(w, 4) },
      { name: "fixed-point w", value: ui.fmt(Math.max(0, p.b - d2 / 2) / p.a, 4) }, { name: "dw/dt", value: ui.fmt(sim.dW[a * N + c], 3) }];
    if (env) rows.push({ name: "∂E*/∂w (finite diff.)", value: ui.fmt(env.fd[q], 6) }, { name: "D²/2", value: ui.fmt(env.half[q], 6) });
    return ui.tipRows(`pair ${a + 1}–${c + 1}`, rows, "Click to select this pair.");
  }
  function neuronTip(i) {
    const net = m.net;
    return ui.tipRows(`neuron ${i + 1}`, [
      { name: "voltage u_i", value: ui.fmt(uDisp[i], 3) }, { name: "threshold t_i", value: ui.fmt(net.t[i], 3) },
      { name: "input current I_i", value: ui.fmt(M.current(net, uDisp, g, i), 3) }, { name: "d_i (leak)", value: ui.fmt(M.degree(net, i), 3) },
      ...[0, 1, 2].map(b => ({ name: `attachment a_i${"₁₂₃"[b]}`, value: ui.fmt(net.A[i * B + b], 3), color: catColor(b), shape: "square" })),
    ], "Click to select it; the previous selection becomes the partner.");
  }
  const sub = n => String(n).split("").map(c => "₀₁₂₃₄₅₆₇₈₉"[+c]).join("");

  // ---------------------------------------------------------- start
  titles();
  tp.reset();
  if (tour) { tours.set(tour.id); applyTour(tour); }
  else { uDisp.fill(0); startRelax(); }
}
