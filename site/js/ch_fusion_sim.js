// Chapter "fusion": the live B = 3 network. View (triangle or MDS) with live edges, transport, tours, four primary
// controls (b, alpha, k_lo, sigma_t) and the rest under "More controls", stat tiles, and four linked panels:
// (a) the D^2 matrix with the 2b contour, (b) w against D^2 on the fixed-point line, (c) both residuals over time,
// (d) components with their diameters and spans on both banks.
import * as ui from "./ui.js";
import { P, catColor } from "./palette.js";
import * as O from "./onsite_engine.js";
import * as M from "./ch_fusion_model.js";
import { graphView, matrixView, wdScatter, svgPlot, svgEl } from "./ch_fusion_draw.js";
import { netview } from "./netview.js";

const SUB = "₁₂₃";
const SUPS = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
// decade labels every fourth decade only (the shared lineChart labels every decade)
const logLabel = v => { const e = Math.round(Math.log10(v)); return e % 4 ? "" : e === 0 ? "1" : "10" + String(e).split("").map(c => SUPS[c]).join(""); };
const URLKEY = { b: "b", alpha: "a", klo: "k", st: "st", sig: "sig", s: "s", n: "n", seed: "seed" };

export function initSim(slot, ctx, hooks = {}) {
  let { p, tour } = readParams(ctx);
  let viewMode = ctx.param("view", "") === "mds" ? "mds" : "auto";
  const F = M.createFusion(p);
  let script = null, cur = null, tickN = 0, envDone = false, lastEval = null;
  const hist = { t: [], r: [], eq: [] };
  const fig = slot.closest("figure");

  // ------------------------------------------------------------ layout
  const L = ui.simLayout({ parent: slot });
  // left: input space (rows of H in the triangle) or response-function space (MDS of D); right: neuron space
  // (netview, seeded-random layout). Hovering a neuron in either, or a pair in (a) / (b), rings it in both.
  const pairWrap = ui.h("div", { class: "space-pair" });
  const gvCol = ui.h("div"), nvCol = ui.h("div");
  pairWrap.append(gvCol, nvCol); L.view.append(pairWrap);
  const gvBadge = ui.spaceBadge("input");
  const synT = ui.synapseToggle({ checked: ctx.param("syn", "") === "1", onChange: on => { ctx.setParams({ syn: on ? "1" : null }); if (cur) drawView(cur); } });
  const gvHead = ui.h("div", { class: "panel-head" }, ui.h("div", { class: "chart-title", text: "Response functions" }), gvBadge);
  gvCol.append(gvHead, ui.h("div", { style: { margin: "0 0 2px" } }, synT.el));
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
  const sK = ui.slider({ label: "nonlinearity $k_{lo}$", min: 0, max: 10, step: 0.5, value: p.klo, fmt: x => ui.fmtFixed(x, 1),
    marks: [{ value: 0, label: "linear" }, { value: 3, label: "default" }],
    help: "Extra stiffness below threshold ($k_{hi} = 0$): a leaky rectifier.", onInput: x => { F.setKlo(x); onPotential(); changed(); } });
  const sT = ui.slider({ label: "threshold spread $\\sigma_t$", min: 0, max: 1.5, step: 0.05, value: p.st, fmt: x => ui.fmtFixed(x, 2),
    help: "$t_i = \\sigma_t z_i$, fixed $z_i$. Acts only when $k_{lo} &gt; 0$.", onInput: x => { F.setSt(x); changed(); } });
  const selSig = ui.select({ label: "inputs $\\Sigma_g$", options: M.SIGMAS.map(s => ({ value: s.id, label: s.label })), value: p.sig,
    onChange: id => { F.setSigma(id); changed(); } });
  const segS = ui.segmented({ label: "attachment sparsity $s$ (inputs per neuron)", options: [1, 2, 3].map(v => ({ value: v, label: String(v) })), value: p.s,
    help: "$s = 1$ puts every neuron on a corner, so neurons on one corner start identical.", onChange: v => { p = { ...F.p, s: v }; rebuild(); } });
  const segN = ui.segmented({ label: "interior neurons $N$", options: [20, 30, 40].map(v => ({ value: v, label: String(v) })), value: p.n, onChange: v => { p = { ...F.p, n: v }; rebuild(); } });
  const sSeed = ui.slider({ label: "seed (attachments, thresholds, banks)", min: 0, max: 9, step: 1, value: p.seed, fmt: x => String(x), onChange: x => { p = { ...F.p, seed: x }; rebuild(); } });
  const segV = ui.segmented({ label: "drawing", options: [{ value: "auto", label: "triangle (linear)" }, { value: "mds", label: "MDS of $D$" }], value: viewMode,
    onChange: v => { viewMode = v; gv.refit(); if (cur) drawView(cur); ctx.setParams({ view: v === "mds" ? "mds" : null }); } });
  ui.controls({ parent: L.side, title: "Plasticity and neurons", primary: [sB, sA, sK, sT], advanced: [selSig, segS, segN, sSeed, segV], advancedLabel: "More controls" });

  const stats = ui.stats({ parent: L.side, items: [
    { key: "K", label: "components $K$", fmt: String, help: "Connected components of the live graph (w > 1e-9), singletons included." },
    { key: "L", label: "largest component", fmt: String, help: "Its span: the largest D between two members, in edge ranges sqrt(2b), on the training bank (evaluation bank in brackets)." },
    { key: "r", label: "plasticity residual $r$", fmt: x => ui.fmt(x, 2), help: "max |dw/dt| over pairs, projected (a dead edge pushed below zero counts 0)." },
    { key: "eq", label: "neuronal residual", fmt: x => ui.fmt(x, 2), help: "max over neurons and stimuli of |dE/du_i| after the last equilibrium solve (training bank)." },
    { key: "env", label: "envelope check", fmt: x => ui.fmt(x, 2), help: "Central difference (step 1e-4) of the mean equilibrium energy in w_ij, minus D_ij^2/2, for the strongest live edge and the nearest dead pair; computed at convergence." },
    { key: "t", label: "time $t$", fmt: x => ui.fmt(x, 3) },
  ] });
  ui.legend({ parent: L.side, items: [
    { label: "response $F_i$, colour: its component (largest first)", color: P.cat[0], shape: "dot" },
    { label: "singleton (no live edge)", color: P.other, shape: "dot" },
    { label: "input $g_1, g_2, g_3$ (clamped)", color: P.boundary, shape: "square" },
    { label: "live synapse, width ∝ $w_{ij}$", color: P.edge, shape: "line" },
  ] });

  // ------------------------------------------------------------ linked panels
  const box = (parent, title, sub) => {
    const b = ui.h("div");
    parent.append(ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", html: title }), sub ? ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" }, html: sub }) : null, b));
    return b;
  };
  const mid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 270px), 1fr))", gap: "18px 28px", marginTop: "16px" } });
  const oneCol = window.matchMedia("(max-width: 900px)");
  const placeMid = () => (oneCol.matches ? L.below.prepend(mid) : L.view.append(mid));
  placeMid(); oneCol.addEventListener("change", placeMid);
  const mBox = box(mid, "(a) Distances $D_{ij}^2$ and live edges", "orange: the contour $D^2 = 2b$; dots: live edges");
  const sBox = box(mid, "(b) $w_{ij}$ against $D_{ij}^2$", "line: the fixed point $w = (b - D^2/2)_+/\\alpha$");
  const rBox = box(L.below, "(c) Residuals over time", "log scale; dashed: the claim tolerance $10^{-6}$");
  const tBox = box(L.below, "(d) Components with at least two members", "diameter and span on the training bank / evaluation bank");
  ui.renderMath(mid); ui.renderMath(L.below);
  const mx = matrixView(mBox, { size: 300, tooltip: (i, j) => tipPair(i, j), onHover: pr => linkHover(pr || [], pr) });
  const cbar = ui.colorbar({ parent: mBox, map: "seq", range: [0, 1], label: "$D_{ij}^2$ (training bank), clipped at $4b$", width: 230 });
  const sc = wdScatter(sBox, { height: 270, tooltip: (i, j) => tipPair(i, j), onHover: pr => linkHover(pr || [], pr) });
  let rCursor = null;
  const rc = svgPlot(rBox, { height: 210, ylog: true, xrange: [0, 1], yrange: [1e-16, 10], xlabel: "time t", ylabel: "residual", yfmt: logLabel, render: () => {} });
  const rLeg = ui.legend({ parent: rBox, items: [{ label: "plasticity residual $r$", color: P.cat[0], shape: "line" }, { label: "neuronal residual", color: P.cat[1], shape: "line" }, { label: "claim tolerance $10^{-6}$", color: P.mute2, shape: "dash" }] });
  rLeg.el.classList.add("chart-legend");
  const table = ui.h("div", { class: "table-scroll" });
  tBox.append(table);

  // ------------------------------------------------------------ analysis -> views
  const compColor = rank => (rank < 0 ? P.other : rank < 8 ? catColor(rank) : P.interior);
  function useSimplex(f) { return viewMode !== "mds" && f.linear && f.xySimplex; }
  function drawView(f) {
    const simplex = useSimplex(f), lay = simplex ? f.xySimplex : f.xyMds, N = f.N;
    const colors = [], sizes = [];
    for (let i = 0; i < N; i++) { colors.push(compColor(f.rank[i])); sizes.push(f.rank[i] < 0 ? 3.8 : 4.6); }
    const edges = [], pairs = [], ws = [];
    if (f.wmax > 0) for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const w = f.W[i * N + j]; if (w > 1e-9) { edges.push({ i, j, a: w / f.wmax }); pairs.push(i, j); ws.push(w); } }
    ensureNet(N);
    nv.setColors(colors); nv.setSizes(sizes.map(r => r / 4.6)); nv.setEdges(pairs.length ? pairs : null, ws, { wmax: f.wmax });
    gvBadge.dataset.space = simplex ? "input" : "function";
    gvBadge.textContent = simplex ? "input space" : "response-function space";
    gv.setCorner(simplex
      ? "Linear neurons: rows $h_i$ of $H$ in the triangle of inputs."
      : `MDS of $D$ (training bank): the two drawn dimensions hold ${ui.fmtPct(f.xyMds.explained, 0)} of the positive spectrum.`);
    gv.set({ xy: lay.xy, anchors: lay.anchors, anchorLabels: ["g₁", "g₂", "g₃"], colors, sizes, edges: synT.get() ? edges : [], mode: simplex ? "simplex" : "mds",
      scaleBar: simplex ? null : { len: Math.sqrt(2 * f.b), label: "edge range √(2b)" } });
  }
  function orderOf(f) {
    const N = f.N, out = [], groups = [];
    const byRank = [...f.comps].sort((a, b) => a.rank - b.rank);
    const singles = [];
    for (let i = 0; i < N; i++) if (f.rank[i] < 0) singles.push(i);
    const along = mem => {
      if (mem.length < 3) return mem.slice();
      const n = mem.length, D = new Float64Array(n * n);
      for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) D[a * n + c] = f.D2[mem[a] * N + mem[c]];
      const X = O.mds(D, n, 1).X;
      return mem.map((i, k) => [i, X[k]]).sort((a, b) => a[1] - b[1]).map(x => x[0]);
    };
    for (const c of byRank) { const s = out.length; out.push(...along(c.members)); groups.push({ start: s, end: out.length, color: compColor(c.rank) }); }
    const s0 = out.length; out.push(...singles);
    if (singles.length) groups.push({ start: s0, end: out.length, color: P.other });
    return { order: out, groups };
  }
  function drawPanels(f) {
    const { order, groups } = orderOf(f);
    mx.set({ N: f.N, D2: f.D2, W: f.W, order, groups, b: f.b, wmax: f.wmax });
    cbar.set({ range: [0, 4 * f.b], ticks: [0, 2 * f.b, 4 * f.b], fmt: x => (x === 0 ? "0" : x === 2 * f.b ? `2b = ${ui.fmt(x, 2)}` : ui.fmt(x, 2)) });
    sc.set({ N: f.N, D2: f.D2, W: f.W, b: f.b, alpha: f.alpha, wmax: f.wmax, colorOf: i => compColor(f.rank[i]) });
    drawTable(f);
  }
  function drawTable(f) {
    const sr = Math.sqrt(2 * f.b), ev = f.withEval;
    const rows = f.comps.slice(0, 10).map(c => `<tr><td><span class="swatch round" style="--sw:${compColor(c.rank)}"></span></td><td class="num">${c.size}</td>` +
      `<td class="num">${ui.fmt(c.diamMax, 3)} / ${ev ? ui.fmt(c.diamMaxEval, 3) : "…"}</td><td class="num">${ui.fmtFixed(c.span / sr, 2)} / ${ev ? ui.fmtFixed(c.spanEval / sr, 2) : "…"}</td></tr>`).join("");
    const more = f.comps.length > 10 ? `<tr><td></td><td colspan="3">and ${f.comps.length - 10} more</td></tr>` : "";
    table.innerHTML = `<table class="data"><thead><tr><th></th><th class="num">size</th><th class="num">diam<sub>max</sub> train / eval</th><th class="num">span/√(2b) train / eval</th></tr></thead>` +
      `<tbody>${rows || `<tr><td></td><td colspan="3">none: every neuron is a singleton</td></tr>`}${more}</tbody></table>` +
      `<div class="u-help" style="margin-top:4px">${f.singles} singleton${f.singles === 1 ? "" : "s"}; edge range √(2b) = ${ui.fmt(sr, 3)}${ev ? "" : "; evaluation-bank values are refreshed every few frames and at convergence"}.</div>`;
  }
  function drawResiduals() {
    const ts = hist.t, t1 = Math.max(ts.length ? ts[ts.length - 1] : 1, 1e-9);
    rc.update({ xrange: [0, t1], render: ({ plot, X, Y, m, W, H }) => {
      const line = (ys, color) => {
        let d = "", pen = false;
        ts.forEach((t, k) => { const y = ys[k]; if (!(y > 0)) { pen = false; return; } d += (pen ? "L" : "M") + X(t).toFixed(1) + "," + Y(Math.max(1e-16, Math.min(10, y))).toFixed(1); pen = true; });
        svgEl("path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round" }, plot);
      };
      svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(1e-6), y2: Y(1e-6), stroke: P.mute2, "stroke-dasharray": "3 3" }, plot);
      line(hist.r, P.cat[0]); line(hist.eq, P.cat[1]);
      if (rCursor != null) svgEl("line", { x1: X(rCursor), x2: X(rCursor), y1: m.t, y2: H - m.b, stroke: P.accent, "stroke-width": 1.25, opacity: 0.8 }, plot);
    } });
  }
  const statCache = {};
  const setStat = (k, v, sub) => { const sig = `${v}|${sub}`; if (statCache[k] === sig) return; statCache[k] = sig; stats.set(k, v, sub); };
  function showStats(f) {
    setStat("K", f.K, `${f.kMulti} with ≥ 2 members · ${f.singles} singleton${f.singles === 1 ? "" : "s"}`);
    const c0 = f.comps[0], sr = Math.sqrt(2 * f.b);
    setStat("L", f.largest, c0 ? `span ${ui.fmtFixed(c0.span / sr, 2)}${f.withEval ? ` (${ui.fmtFixed(c0.spanEval / sr, 2)})` : ""} edge ranges` : "all singletons");
    setStat("r", +f.r.toPrecision(2), f.r < M.CLAIM_TOL ? "below 10⁻⁶: at the fixed point" : "still moving");
    setStat("eq", +f.eq.toPrecision(2), "max |∂E/∂uᵢ|, training bank");
    const env = f.env;
    setStat("env", env ? +env.err.toPrecision(2) : null, env ? `|∂E*/∂w − D²/2| on ${env.dead ? "a live edge and a dead pair" : "a live edge"}` : f.r < M.CLAIM_TOL ? "no pair to check" : "computed at convergence");
    setStat("t", f.t, `dt = ${ui.fmt(M.dtOf(f.alpha), 2)}, ${ui.fmtInt(f.steps)} steps`);
  }
  function showAll(f) { drawView(f); showStats(f); drawPanels(f); rCursor = f.t; drawResiduals(); }

  // ------------------------------------------------------------ the loop
  function frameNow(withEval) {
    const a = F.analyse(withEval);
    if (withEval) lastEval = a;
    a.steps = F.sim.steps;
    a.env = F.envelope;
    return a;
  }
  function pushHist(a) {
    hist.t.push(a.t); hist.r.push(a.r > 0 ? a.r : NaN); hist.eq.push(a.eq > 0 ? a.eq : NaN);
    if (hist.t.length > 1500) for (const k of ["t", "r", "eq"]) hist[k] = hist[k].filter((_, i) => i % 2 === 0 || i === hist[k].length - 1);
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
    cur = a; pushHist(a);
    drawView(a); showStats(a);
    if (done || tickN % 5 === 0) drawPanels(a);
    rCursor = a.t;
    if (done || tickN % 8 === 0) drawResiduals();
    if (done) { a.done = true; a.doneLabel = "fixed point: r < 10⁻⁹"; hooks.onFixedPoint?.(a); }
    return a;
  }
  function restart() {
    F.reset(); script = null; envDone = false; tickN = 0; lastEval = null;
    hist.t.length = hist.r.length = hist.eq.length = 0;
    gv.refit();
    const a = frameNow(true); cur = a; pushHist(a);
    showAll(a); drawResiduals();
    return a;
  }

  // ------------------------------------------------------------ plumbing
  function onPotential() { segV.setDisabled("auto", !F.linear()); if (!F.linear() && viewMode !== "mds") gv.refit(); }
  function untour() { if (!tour) return; tour = null; tours.moved(); }
  function changed() { untour(); script = null; envDone = false; writeURL(); if (!tp.playing) tp.play(); }
  function rebuild() { untour(); script = null; F.build({ ...F.p, ...p }); syncAll(); writeURL(); tp.reset(); tp.play(); }
  function syncAll() {
    const q = F.p; sB.set(q.b); sA.set(q.alpha); sK.set(q.klo); sT.set(q.st); selSig.set(q.sig); segS.set(q.s); segN.set(q.n); sSeed.set(q.seed);
    onPotential();
  }
  function writeURL() {
    const o = { tour: tour ? tour.id : null };
    for (const k of M.KEYS) o[URLKEY[k]] = tour || F.p[k] === M.DEFAULTS[k] ? null : F.p[k];
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
    const rows = [{ name: "component", value: comp ? `${comp.size} neurons` : "singleton", color: compColor(rk) }];
    for (let b = 0; b < M.B; b++) rows.push({ name: `attachment a${SUB[b]}`, value: ui.fmtFixed(F.net.A[i * M.B + b], 3) });
    if (!f.linear) rows.push({ name: "threshold tᵢ", value: ui.fmt(F.net.t[i], 3) });
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
  for (const k of M.KEYS) p[k] = ctx.param(URLKEY[k], M.DEFAULTS[k]);
  return { p: M.clampParams(p), tour: null };
}
