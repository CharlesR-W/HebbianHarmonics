// Chapter "fusion": the live B = 3 network of linear neurons. Input space (the triangle) beside neuron space, with
// transport, two tours, two controls (b, alpha) and stat tiles. A second figure (hooks.fixedSlot) is driven by the
// same run: w against D^2 for every pair, on the fixed-point line, with the residual and the envelope check.
import * as ui from "./ui.js";
import { P, catColor } from "./palette.js";
import * as O from "./onsite_engine.js";
import * as M from "./ch_fusion_model.js";
import { graphView, wdScatter } from "./ch_fusion_draw.js";
import { netview } from "./netview.js";

const SUB = "₁₂₃";
const URLKEY = { b: "b", alpha: "a" };   // the only reader-facing settings; the network itself is fixed (linear, seed 0)

export function initSim(slot, ctx, hooks = {}) {
  let { p, tour } = readParams(ctx);
  const F = M.createFusion(p);
  let script = null, cur = null, tickN = 0, envDone = false;
  const fig = slot.closest("figure");

  // ------------------------------------------------------------ layout
  const L = ui.simLayout({ parent: slot });
  // left: input space (rows of H in the triangle) or response-function space (MDS of D); right: neuron space
  // (netview, seeded-random layout). Hovering a neuron in either, or a pair in (a) / (b), rings it in both.
  const pairWrap = ui.h("div", { class: "space-pair" });
  const gvCol = ui.h("div"), nvCol = ui.h("div");
  pairWrap.append(gvCol, nvCol); L.view.append(pairWrap);
  const synT = ui.synapseToggle({ checked: ctx.param("syn", "") === "1", onChange: on => { ctx.setParams({ syn: on ? "1" : null }); if (cur) drawView(cur); } });
  gvCol.append(ui.panelHead("Each neuron at its mixture of the inputs", "input"), ui.h("div", { style: { margin: "0 0 2px" } }, synT.el));
  const gvBox = ui.h("div"); gvCol.append(gvBox);
  const gv = graphView(gvBox, { height: 470, tooltip: (i, kind) => tipNeuron(i, kind), onHover: i => linkHover(i == null ? [] : [i]) });
  nvCol.append(ui.panelHead("The network: neurons at random places", "neuron"), ui.h("div", { class: "u-help", style: { margin: "0 0 2px" }, html: "edges: live synapses, width ∝ w<sub>ij</sub>" }));
  const nvBox = ui.h("div"); nvCol.append(nvBox);
  let nv = null, nvN = 0;
  function ensureNet(N) {
    if (nv && nvN === N) return;
    if (nv) nv.destroy();
    nvN = N;
    nv = netview(nvBox, { N, B: M.B, seed: 11, aspect: 1.05, maxHeight: 470, minHeight: 260, radius: 6, edgeWidth: 3.6, badge: false,
      inputColors: P.boundary, tooltip: (i, kind) => tipNeuron(i, kind === "input" ? "anchor" : "point"),
      onHover: (i, kind) => linkHover(i == null || kind !== "neuron" ? [] : [i]) });
  }
  function linkHover(list, pair = null) { gv.highlight(list, pair); nv?.setHighlight(list.length ? list : null); }
  const tp = ui.transport({
    parent: L.view, watch: fig, autoplay: true, maxHistory: 300,
    speeds: [{ value: 1, label: "×1" }, { value: 4, label: "×4" }, { value: 16, label: "×16" }, { value: 64, label: "×64" }], speed: 4,
    timeFmt: t => `t = ${ui.fmt(t, 3)}`,
    onTick: speed => tick(speed),
    onSeek: f => { cur = f; showAll(f); },
    onReset: () => restart(),
  });

  // ------------------------------------------------------------ tours and controls
  const tours = ui.chips({ parent: L.side, label: "Tours", items: M.TOURS, active: tour ? tour.id : null, onPick: t => applyTour(t) });
  const sB = ui.slider({ label: "fusion scale $b$", min: M.LIMITS.b[0], max: M.LIMITS.b[1], log: true, sig: 2, value: p.b, fmt: x => ui.fmt(x, 2),
    help: "Edges can live where $D_{ij}^2 &lt; 2b$: the range of fusion.", onInput: x => { F.setB(x); changed(); } });
  const sA = ui.slider({ label: "stiffness $\\alpha$", min: M.LIMITS.alpha[0], max: M.LIMITS.alpha[1], log: true, sig: 2, value: p.alpha, fmt: x => ui.fmt(x, 2),
    help: "A live edge reaches at most $b/\\alpha$ (attachments total 1).", onInput: x => { F.setAlpha(x); changed(); } });
  ui.controls({ parent: L.side, primary: [sB, sA] });

  const stats = ui.stats({ parent: L.side, items: [
    { key: "K", label: "components $K$", fmt: String, help: "Connected components of the live graph (w > 1e-9), lone neurons included." },
    { key: "L", label: "largest component", fmt: String, help: "Its span: the largest D between two members, in edge ranges sqrt(2b), on the training bank (evaluation bank in brackets)." },
    { key: "t", label: "time $t$", fmt: x => ui.fmt(x, 3) },
  ] });
  ui.legend({ parent: L.side, items: [
    { label: "neuron, colour: its component (largest first)", color: P.cat[0], shape: "dot" },
    { label: "alone (no live synapse)", color: P.other, shape: "dot" },
    { label: "input $g_1, g_2, g_3$ (clamped)", color: P.boundary, shape: "square" },
    { label: "live synapse, width ∝ $w_{ij}$", color: P.edge, shape: "line" },
  ] });

  // ------------------------------------------------------------ second figure: w against D^2, residual, envelope
  const fixedSlot = hooks.fixedSlot || L.below;
  const fRow = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 260px), 1fr))", gap: "14px 28px", alignItems: "start" } });
  const sBox = ui.h("div", { style: { minWidth: "0", gridColumn: "span 2" } }), fSide = ui.h("div", { style: { minWidth: "0" } });
  fRow.append(sBox, fSide); fixedSlot.append(fRow);
  const sc = wdScatter(sBox, { height: 300, tooltip: (i, j) => tipPair(i, j), onHover: pr => linkHover(pr || [], pr) });
  const fstats = ui.stats({ parent: fSide, cols: 1, items: [
    { key: "r", label: "residual $r = \\max|dw_{ij}/dt|$", fmt: x => ui.fmt(x, 2) },
    { key: "env", label: "envelope check", fmt: x => ui.fmt(x, 2), help: "Central difference (step 1e-4) of the mean equilibrium energy in w_ij, minus D_ij^2/2, for the strongest live edge and the nearest dead pair; computed at convergence." },
  ] });
  ui.legend({ parent: fSide, items: [
    { label: "live synapse, colour: its component", color: P.cat[0], shape: "dot" },
    { label: "dead pair ($w = 0$)", color: P.other, shape: "line" },
    { label: "fixed point $w = (b - D^2/2)_+/\\alpha$", color: P.fg, shape: "line" },
  ] });

  // ------------------------------------------------------------ analysis -> views
  const compColor = rank => (rank < 0 ? P.other : rank < 8 ? catColor(rank) : P.interior);
  function drawView(f) {
    const lay = f.xySimplex, N = f.N;
    const colors = [], sizes = [];
    for (let i = 0; i < N; i++) { colors.push(compColor(f.rank[i])); sizes.push(f.rank[i] < 0 ? 3.8 : 4.6); }
    const edges = [], pairs = [], ws = [];
    if (f.wmax > 0) for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const w = f.W[i * N + j]; if (w > 1e-9) { edges.push({ i, j, a: w / f.wmax }); pairs.push(i, j); ws.push(w); } }
    ensureNet(N);
    nv.setColors(colors); nv.setSizes(sizes.map(r => r / 4.6)); nv.setEdges(pairs.length ? pairs : null, ws, { wmax: f.wmax });
    gv.setCorner("Corners: the inputs. A neuron sits at its mixture $h_i$ of them.");
    gv.set({ xy: lay.xy, anchors: lay.anchors, anchorLabels: ["g₁", "g₂", "g₃"], colors, sizes, edges: synT.get() ? edges : [], mode: "simplex", scaleBar: null });
  }
  function drawPanels(f) {
    sc.set({ N: f.N, D2: f.D2, W: f.W, b: f.b, alpha: f.alpha, wmax: f.wmax, colorOf: i => compColor(f.rank[i]) });
  }
  const statCache = {};
  const setStat = (k, v, sub) => { const sig = `${v}|${sub}`; if (statCache[k] === sig) return; statCache[k] = sig; stats.set(k, v, sub); };
  function showStats(f) {
    setStat("K", f.K, `${f.kMulti} with ≥ 2 members · ${f.singles} alone`);
    const c0 = f.comps[0], sr = Math.sqrt(2 * f.b);
    setStat("L", f.largest, c0 ? `span ${ui.fmtFixed(c0.span / sr, 2)}${f.withEval ? ` (${ui.fmtFixed(c0.spanEval / sr, 2)})` : ""} edge ranges` : "every neuron alone");
    const fs = (k, v, sub) => { const sig = `${v}|${sub}`; if (statCache["f" + k] === sig) return; statCache["f" + k] = sig; fstats.set(k, v, sub); };
    fs("r", +f.r.toPrecision(2), f.r < M.CLAIM_TOL ? "below 10⁻⁶: at the fixed point" : "still moving");
    const env = f.env;
    fs("env", env ? +env.err.toPrecision(2) : null, env ? `|∂E*/∂w − D²/2| on ${env.dead ? "a live edge and a dead pair" : "a live edge"}` : f.r < M.CLAIM_TOL ? "no pair to check" : "computed at convergence");
    setStat("t", f.t, `dt = ${ui.fmt(M.dtOf(f.alpha), 2)}, ${ui.fmtInt(f.steps)} steps`);
  }
  function showAll(f) { drawView(f); showStats(f); drawPanels(f); }

  // ------------------------------------------------------------ the loop
  function frameNow(withEval) {
    const a = F.analyse(withEval);
    a.steps = F.sim.steps;
    a.env = F.envelope;
    return a;
  }
  function tick(speed) {
    if (script && F.done() && script.idx + 1 < script.list.length) {
      script.idx++; const b = script.list[script.idx];
      F.setB(b); sB.set(b); envDone = false;
    }
    if (!F.done()) { F.advance(speed >= 64 ? 30 : 18, speed); envDone = false; }
    const done = F.done() && (!script || script.idx + 1 >= script.list.length);
    if (done && !envDone) { F.envelopeCheck(); envDone = true; }
    tickN++;
    const a = frameNow(done || tickN % 12 === 0);
    cur = a;
    drawView(a); showStats(a);
    if (done || tickN % 5 === 0) drawPanels(a);
    if (done) { a.done = true; a.doneLabel = "fixed point: r < 10⁻⁹"; hooks.onFixedPoint?.(a); }
    return a;
  }
  function restart() {
    F.reset(); script = null; envDone = false; tickN = 0;
    gv.refit();
    const a = frameNow(true); cur = a;
    showAll(a);
    return a;
  }

  // ------------------------------------------------------------ plumbing
  function untour() { if (!tour) return; tour = null; tours.moved(); }
  function changed() { untour(); script = null; envDone = false; writeURL(); if (!tp.playing) tp.play(); }
  function syncAll() {
    const q = F.p; sB.set(q.b); sA.set(q.alpha);
  }
  function writeURL() {
    const o = { tour: tour ? tour.id : null };
    for (const k of Object.keys(URLKEY)) o[URLKEY[k]] = tour || F.p[k] === M.DEFAULTS[k] ? null : F.p[k];
    ctx.setParams(o);
  }
  function applyTour(t) {
    tour = t; F.build(M.tourParams(t)); syncAll(); writeURL();
    tp.reset();
    script = t.script ? { list: t.script, idx: 0 } : null;
    if (t.script) F.setB(t.script[0]);
    tp.setSpeed(F.p.alpha < 0.5 ? 64 : 4);
    tp.play();
  }
  ctx.onParams(params => {
    const r = readParams(ctx);
    if (r.tour) { tours.set(r.tour.id); applyTour(r.tour); return; }
    if (![...params.keys()].some(k => Object.values(URLKEY).includes(k))) return;
    tour = null; tours.clear(); p = r.p; F.build(p); syncAll(); tp.reset(); tp.play();
  });

  // ------------------------------------------------------------ tooltips
  function tipNeuron(i, kind) {
    const f = cur; if (!f) return null;
    if (kind === "anchor") return ui.tipRows(`input g${SUB[i]} (clamped)`, [{ name: "variance", value: ui.fmt(F.Sigma[i * M.B + i], 3), color: P.boundary, shape: "square" }]);
    const N = f.N, rk = f.rank[i], comp = f.comps.find(c => c.members.includes(i));
    let nl = 0, sw = 0; for (let j = 0; j < N; j++) { const w = f.W[i * N + j]; if (w > 1e-9) { nl++; sw += w; } }
    const rows = [{ name: "component", value: comp ? `${comp.size} neurons` : "alone", color: compColor(rk) }];
    for (let b = 0; b < M.B; b++) rows.push({ name: `attachment a${SUB[b]}`, value: ui.fmtFixed(F.net.A[i * M.B + b], 3) });
    rows.push({ name: "live edges", value: String(nl) }, { name: "Σⱼ wᵢⱼ", value: ui.fmt(sw, 3) });
    return ui.tipRows(`neuron ${i}`, rows);
  }
  function tipPair(i, j) {
    const f = cur; if (!f || i === j) return i === j && f ? ui.tipRows(`neuron ${i}`, [{ name: "D²", value: "0" }]) : null;
    const N = f.N, d = f.D2[i * N + j], w = f.W[i * N + j], th = Math.max(f.b - d / 2, 0) / f.alpha, dw = f.b - f.alpha * w - d / 2;
    return ui.tipRows(`neurons ${i} and ${j}`, [
      { name: "D²ᵢⱼ (training bank)", value: ui.fmt(d, 3) }, { name: "2b", value: ui.fmt(2 * f.b, 3) },
      { name: "wᵢⱼ", value: ui.fmt(w, 4) }, { name: "(b − D²/2)₊/α", value: ui.fmt(th, 4) },
      { name: "dw/dt", value: w <= 0 && dw < 0 ? `${ui.fmt(dw, 2)} (held at 0)` : ui.fmt(dw, 2) },
    ]);
  }

  // ------------------------------------------------------------ go
  syncAll();
  if (tour) { script = tour.script ? { list: tour.script, idx: 0 } : null; if (tour.script) F.setB(tour.script[0]); tp.setSpeed(F.p.alpha < 0.5 ? 64 : 4); }
  tp.record(restart());
  if (tour && tour.script) { script = { list: tour.script, idx: 0 }; F.setB(tour.script[0]); }
  ctx.onShow(() => { if (cur) drawView(cur); });
  return { get params() { return { ...F.p }; }, F };
}

function readParams(ctx) {
  const tour = M.tourById(ctx.param("tour", ""));
  if (tour) return { p: M.tourParams(tour), tour };
  const p = {};
  for (const k of Object.keys(URLKEY)) p[k] = ctx.param(URLKEY[k], M.DEFAULTS[k]);
  return { p: M.clampParams(p), tour: null };
}
