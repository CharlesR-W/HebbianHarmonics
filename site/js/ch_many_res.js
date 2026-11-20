// Chapter "many": the resolution figures.
//   activation from W = 0: planted pairs from resolution.json (p2);
//   a line of neurons along one mode: live sim (onsite_engine) plus the population sweep (pop).
import * as ui from "./ui.js";
import { lineChart, hidpi, scale } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as M from "./ch_many_model.js";
import { BR, BR_LABEL } from "./ch_many_live.js";

const sub = "₀₁₂₃₄₅₆₇";

export function buildResolution(ctx, R) {
  buildActivation(ctx, ctx.slot("act"), R);
  buildLine(ctx, ctx.slot("line"), R);
}

// ------------------------------------------------------------------ (2) activation from the uncoupled state
function buildActivation(ctx, slot, R) {
  const p2 = R.p2, rows = p2.rows_seed0, b = p2.b;
  let ax = ctx.param("ax", "hat") === "sig" ? "sig" : "hat";
  const seg = ui.segmented({ label: "horizontal axis", options: [{ value: "hat", label: "$\\hat\\lambda_a = v_a^\\top \\hat C v_a$ (training bank)" }, { value: "sig", label: "$\\lambda_a$ of $\\Sigma_g$" }], value: ax,
    onChange: x => { ax = x; ctx.setParams({ ax: x === "hat" ? null : x }); go(); } });
  const note = ui.h("p", { class: "u-help", style: { margin: "4px 0 6px" } });
  const box = ui.h("div");
  slot.append(ui.h("div", { style: { font: "13px/1.4 var(--sans)" } }, seg.el), note, box);
  const ch = lineChart(box, { series: [], xlog: true, ylog: true, xlabel: "variance λ of the input direction the pair differs along", ylabel: "δ = |Δh·vₐ| (attachment share)", height: 300, xname: "λ" });
  function go() {
    const xOf = r => (ax === "hat" ? r[2] : r[1]);
    const alive = rows.filter(r => r[7]), dead = rows.filter(r => !r[7]);
    const xs = rows.map(xOf), lo = Math.min(...xs) / 1.4, hi = Math.max(...xs) * 1.4;
    const line = Array.from({ length: 40 }, (_, k) => lo * Math.pow(hi / lo, k / 39));
    const wrong = rows.filter(r => r[7] !== (r[4] < Math.sqrt(2 * b / xOf(r)))).length;
    ch.update({ series: [
      { name: "edge alive at the fixed point", x: alive.map(xOf), y: alive.map(r => r[4]), color: catColor(0), points: true, r: 3.2 },
      { name: "edge dead", x: dead.map(xOf), y: dead.map(r => r[4]), color: P.mute2, points: true, r: 3.2 },
      { name: "δ = √(2b/λ)", x: line, y: line.map(l => Math.sqrt(2 * b / l)), color: P.fg, width: 1.4 },
    ], xrange: [lo, hi], yrange: [Math.min(...rows.map(r => r[4])) / 1.25, Math.max(...rows.map(r => r[4])) * 1.25] });
    note.innerHTML = `On this axis the threshold $\\delta = \\sqrt{2b/\\lambda}$ misclassifies <b>${wrong}</b> of the ${rows.length} seed-0 pairs shown (all ${R.meta.seeds.length} seeds: ${ax === "hat" ? p2.misclassified_by_Chat_threshold : p2.misclassified_by_Sigma_threshold} of ${p2.n_pairs}).`;
    ui.renderMath(note);
  }
  go();
}

// ------------------------------------------------------------------ (3b) a line of neurons along one mode
function buildLine(ctx, slot, R) {
  const fig = slot.closest("figure");
  let p = { ...M.LINE_DEFAULTS }, tour = null, path = null;
  readURL();
  const L = ui.simLayout({ parent: slot });
  // input space: positions along mode a of the attachments (top) and of the embedding h_i (bottom); arcs opt-in
  const head = ui.h("div", { class: "panel-head", style: { justifyContent: "space-between", gap: "4px 14px" } }, ui.spaceBadge("input", "input space · along direction a"));
  L.view.append(head);
  const view = rulers(L.view, { tooltip: i => tip(i) });
  ui.synapseToggle({ parent: head, label: "show synapses (live edges as arcs)", onChange: on => view.setArcs(on) });
  const tp = ui.transport({
    parent: L.view, watch: fig, speeds: [{ value: 1, label: "×1" }, { value: 10, label: "×10" }, { value: 400, label: "max" }], speed: 400,
    autoplay: true, maxHistory: 200, timeFmt: t => `t = ${ui.fmt(t, 3)}`,
    onTick: sp => tick(sp), onSeek: f => seek(f), onRestore: f => restore(f), onReset: () => reset(),
  });
  const tours = ui.chips({ parent: L.side, label: "Tours", items: M.LINE_TOURS, active: tour ? tour.id : null, onPick: t => applyTour(t) });
  const bS = ui.slider({ label: "fusion scale $b$", min: 2e-4, max: 0.05, log: true, sig: 2, value: p.b, marks: [{ value: 0.001, label: "0.001" }, { value: 0.01, label: "0.01" }],
    help: "Acts on the running network (continuation).", onInput: x => setB(x, true) });
  const aS = ui.segmented({ label: "stiffness $\\alpha$", options: [0.01, 0.1, 1].map(x => ({ value: x, label: String(x) })), value: p.a, onChange: x => { p.a = x; structural(); } });
  const mS = ui.select({ label: "input direction $a$", options: [1, 2, 3, 4, 5, 6, 7].map(a => ({ value: a, label: `${a}: variance ${ui.fmt(M.LAM8[a], 3)}` })), value: p.m, onChange: x => { p.m = x; structural(); } });
  ui.controls({ parent: L.side, primary: [bS, aS, mS] });
  const stats = ui.stats({ parent: L.side, items: [
    { key: "k", label: "components $K$", fmt: String },
    { key: "kp", label: "predicted", fmt: String, help: "Components of the uncoupled graph: pairs with λ̂ (x_i − x_j)² < 2b." },
    { key: "br", label: "widest bridged gap$/\\delta_{on}$", fmt: x => ui.fmt(x, 3), help: "Largest uncoupled gap x_{i+1} − x_i inside one component, over sqrt(2b/λ̂). Above 1: a component bridges a gap that the prediction would leave open." },
    { key: "sp", label: "widest span$/\\sqrt{2b}$", fmt: x => ui.fmt(x, 3), help: "Training bank; the evaluation bank (4096 samples) in the sub-line." },
    { key: "r", label: "residual $r$", fmt: x => ui.fmt(x, 2) },
  ] });
  ui.legend({ parent: L.side, items: [
    { label: "neuron, at $x_i$ and at $y_i$", color: P.interior, shape: "dot" },
    { label: "within $\\delta_{on}$ at $W = 0$ (predicted)", color: P.other, shape: "band" },
    { label: "component (bottom)", color: catColor(0), shape: "dot" },
    { label: "live edge (arc, with show synapses on)", color: P.edge, shape: "line" },
  ] });
  const popTitle = ui.h("div", { class: "chart-title" }), popBox = ui.h("div"), popNote = ui.h("div", { class: "u-help", style: { marginTop: "4px" } });
  const popWrap = ui.h("div", { style: { minWidth: "0", maxWidth: "760px", marginTop: "18px" } }, popTitle, popBox, popNote);
  const oneCol = window.matchMedia("(max-width: 900px)");
  const placePop = () => (oneCol.matches ? L.below : L.view).append(popWrap);
  placePop(); oneCol.addEventListener("change", placePop);
  const pop = lineChart(popBox, { series: [], xlog: true, ylog: true, xlabel: "fusion scale b", ylabel: "components K", height: 240, xname: "b" });

  let sys = null, cur = null, T = 0, t0 = 0, s0 = 0, lastB = p.b, branch = "cold";
  const done = [];
  function analyse() { const a = M.lineAnalysis(sys); a.residual = sys.sim.residual; a.eq = sys.sim.eqResidual; a.b = sys.sim.b; return a; }
  function show(a, W) {
    cur = a;
    view.draw({ x: sys.x, y: a.y, labels: a.labels, comps: a.comps, W, N: M.LINE_N, don: a.don, lh: sys.lh, b: a.b, xmax: M.XMAX });
    stats.set("k", a.K, `${a.comps.filter(c => c.size > 1).length} with ≥ 2 members`);
    stats.set("kp", a.Kpred, `δ_on = √(2b/λ̂) = ${ui.fmt(a.don, 3)}`);
    stats.set("br", +a.bridged.toPrecision(3), a.bridged > 1 ? "bridges a gap the prediction leaves open" : "no gap wider than δ_on bridged");
    stats.set("sp", +a.spanT.toPrecision(3), `evaluation bank: ${ui.fmt(a.spanE, 3)}`);
    stats.set("r", +a.residual.toPrecision(2), a.residual < M.RES_TOL ? "below 10⁻⁶: fixed point" : "still moving");
  }
  function tick(speed) {
    const sim = sys.sim;
    if (!(sim.residual < M.RES_TOL)) sim.stepFor(32, speed, 0);
    T = t0 + (sim.steps - s0) * sim.dt;
    const a = analyse(); show(a, sim.net.W);
    const f = { t: T, W: Float64Array.from(sim.net.W), b: sim.b, branch };
    if (!(a.residual < M.RES_TOL)) return f;
    const last = done[done.length - 1];
    if (!last || last.b !== a.b || last.K !== a.K || last.branch !== branch || last.m !== p.m || last.a !== p.a || last.seed !== p.seed) done.push({ b: a.b, K: a.K, branch, m: p.m, a: p.a, seed: p.seed });
    drawPop();
    if (path && path.i < path.bs.length - 1) { path.i++; setB(path.bs[path.i], false); return f; }
    path = null; f.done = true; f.doneLabel = "fixed point: r < 10⁻⁶"; return f;
  }
  function seek(f) {
    const W = sys.sim.net.W, keep = Float64Array.from(W), b0 = sys.sim.b;
    W.set(f.W); sys.sim.setParams({ b: f.b }); sys.sim.refresh();
    show(analyse(), W);
    W.set(keep); sys.sim.setParams({ b: b0 }); sys.sim.refresh();
  }
  function restore(f) { sys.sim.net.W.set(f.W); sys.sim.setParams({ b: f.b }); sys.sim.refresh(); p.b = f.b; lastB = f.b; branch = f.branch; bS.set(f.b); path = null; t0 = f.t; s0 = sys.sim.steps; writeURL(); }
  function reset() {
    sys = M.buildLine(p); T = 0; t0 = 0; s0 = sys.sim.steps; lastB = p.b; branch = "cold";
    const a = analyse(); show(a, sys.sim.net.W); drawPop();
    return { t: 0, W: Float64Array.from(sys.sim.net.W), b: sys.sim.b, branch };
  }
  function setB(x, byHand) {
    if (byHand) { path = null; untour(); }
    if (x !== lastB) branch = x > lastB ? "up" : "down";
    lastB = x; p.b = x; bS.set(x); sys.sim.setParams({ b: x }); writeURL();
    if (byHand) tp.play();
  }
  function structural() { untour(); path = null; writeURL(); tp.reset(); tp.play(); }
  function untour() { if (!tour) return; tour = null; tours.moved(); }
  function writeURL() {
    const d = M.LINE_DEFAULTS, o = { ltour: tour ? tour.id : null };
    for (const [k, u] of [["m", "lm"], ["b", "lb"], ["a", "la"]]) o[u] = tour || p[k] === d[k] ? null : p[k];
    ctx.setParams(o);
  }
  function applyTour(t) {
    tour = t; p = { ...M.LINE_DEFAULTS, ...t.p }; path = t.path ? { bs: t.path, i: 0 } : null; if (path) p.b = path.bs[0];
    bS.set(p.b); aS.set(p.a); mS.set(p.m); writeURL(); tp.reset(); tp.play();
  }
  function readURL() {
    const t = M.lineTourById(ctx.param("ltour", ""));
    if (t) { tour = t; p = { ...M.LINE_DEFAULTS, ...t.p }; path = t.path ? { bs: t.path, i: 0 } : null; if (path) p.b = path.bs[0]; return; }
    const d = M.LINE_DEFAULTS; tour = null; path = null;
    p = { m: Math.min(7, Math.max(1, Math.round(ctx.param("lm", d.m)))), b: Math.min(0.05, Math.max(2e-4, ctx.param("lb", d.b))),
      a: [0.01, 0.1, 1].includes(ctx.param("la", d.a)) ? ctx.param("la", d.a) : d.a, seed: d.seed };
  }
  ctx.onParams(() => {
    if (!["ltour", "lm", "lb", "la"].some(k => ctx.params.has(k))) return;
    readURL(); bS.set(p.b); aS.set(p.a); mS.set(p.m);
    if (tour) tours.set(tour.id); else tours.clear();
    tp.reset(); tp.play();
  });

  // the precomputed population sweep for the nearest (mode, alpha) it has
  const bgp = R.meta.config.population.b_grid;
  function drawPop() {
    const modes = [...new Set(R.pop.map(q => q.a))], alphas = [...new Set(R.pop.map(q => q.alpha))];
    const m = modes.includes(p.m) ? p.m : modes.reduce((x, y) => (Math.abs(M.LAM8[y] - M.LAM8[p.m]) < Math.abs(M.LAM8[x] - M.LAM8[p.m]) ? y : x));
    const al = alphas.reduce((x, y) => (Math.abs(Math.log(y / p.a)) < Math.abs(Math.log(x / p.a)) ? y : x));
    const ps = R.pop.filter(q => q.a === m && q.alpha === al);
    const agg = f => { const mean = [], lo = [], hi = []; bgp.forEach((_, k) => { const v = ps.map(q => f(q, k)); mean.push(v.reduce((s, x) => s + x, 0) / v.length); lo.push(Math.min(...v)); hi.push(Math.max(...v)); }); return { mean, lo, hi }; };
    const s = (name, a, color, extra = {}) => ({ name, x: bgp, y: a.mean, band: [a.lo, a.hi], color, width: 1.8, ...extra });
    const series = [s("predicted from the uncoupled positions", agg((q, k) => q.cold[k][1]), P.mute2, { dash: true }), s(BR_LABEL.cold, agg((q, k) => q.cold[k][0]), BR.cold),
      s(BR_LABEL.up, agg((q, k) => q.up[k][0]), BR.up), s(BR_LABEL.down, agg((q, k) => q.down[k][0]), BR.down)];
    const match = p.m === m && p.a === al;
    if (match) for (const br of ["cold", "up", "down"]) { const q = done.filter(d => d.m === p.m && d.a === p.a && d.seed === p.seed && d.branch === br); if (q.length) series.push({ name: `live, ${br}`, x: q.map(d => d.b), y: q.map(d => d.K), color: BR[br], points: true, r: 4.5 }); }
    pop.update({ series, cursor: p.b, xrange: [bgp[0] * 0.8, bgp[bgp.length - 1] * 1.25], yrange: [0.8, 40] });
    popTitle.innerHTML = `Precomputed: components against $b$ on a line along direction ${m}, $\\alpha = ${al}$ (${ps.length} seeds)`;
    popNote.innerHTML = match ? "Large dots: fixed points this live figure has reached at the current direction and α. Lines: seed means; bands: seed range." :
      `The sweep has directions ${modes.join(" and ")} and $\\alpha \\in \\{${alphas.join(", ")}\\}$; it shows the nearest. Lines: seed means; bands: seed range.`;
    ui.renderMath(popTitle); ui.renderMath(popNote);
  }

  function tip(i) {
    const a = cur; if (!a) return null;
    const k = a.labels[i], c = a.comps[k];
    return ui.tipRows(`neuron ${i}`, [{ name: "uncoupled xᵢ", value: ui.fmt(sys.x[i], 3) }, { name: "current yᵢ", value: ui.fmt(a.y[i], 3) },
      { name: "component", value: c.size === 1 ? "singleton" : `${c.size} neurons` }]);
  }
  tp.record(reset());
  if (tour) tp.play();
}

// two horizontal rulers: uncoupled positions x_i (top) and current positions y_i (bottom), joined by thin lines
function rulers(el, { tooltip }) {
  const cv = ui.h("canvas", { style: { display: "block", width: "100%", touchAction: "pan-y" } });
  el.append(ui.h("div", { class: "chart" }, cv));
  let st = null, Wd = 0, X = null, hit = [], arcs = false;
  const H = 282, yTop = 64, yBot = 196;
  function draw(s) {
    if (s) st = s;
    Wd = Math.floor(el.clientWidth); if (!Wd || !st) return;
    const g = hidpi(cv, Wd, H); g.clearRect(0, 0, Wd, H);
    const m = { l: 16, r: 16 };
    X = scale("linear", -st.xmax, st.xmax, m.l, Wd - m.r);
    const N = st.N, ranks = rankComps(st.comps);
    const colOf = i => { const k = st.labels[i]; return st.comps[k].size === 1 ? P.interior : ranks[k] < 8 ? catColor(ranks[k]) : P.other; };
    g.font = "11px Inter, system-ui, sans-serif";
    // axes
    for (const y0 of [yTop, yBot]) { g.strokeStyle = P.line || "#d9d8d2"; g.lineWidth = 1; g.beginPath(); g.moveTo(m.l, y0 + 0.5); g.lineTo(Wd - m.r, y0 + 0.5); g.stroke(); }
    g.fillStyle = P.ink2; g.font = "560 11.5px Inter, system-ui, sans-serif"; g.textAlign = "left";
    g.fillText("uncoupled position xᵢ along the input direction (W = 0)", m.l, 14);
    g.fillText("position yᵢ at the current W", m.l, yBot + 66);
    g.font = "11px Inter, system-ui, sans-serif"; g.fillStyle = P.mute;
    // edge-range bar
    const len = X(st.don) - X(0);
    const bx = Wd - m.r - Math.min(len, Wd * 0.6);
    g.strokeStyle = P.ink2; g.lineWidth = 1.5; g.beginPath(); g.moveTo(bx, 34); g.lineTo(bx + len, 34); g.moveTo(bx, 30); g.lineTo(bx, 38); g.moveTo(bx + len, 30); g.lineTo(bx + len, 38); g.stroke();
    g.textAlign = "right"; g.fillText("one edge's range δ_on = √(2b/λ̂)", bx - 8, 38);
    // uncoupled neighbours within range: bars on the top axis
    g.fillStyle = P.other;
    for (let i = 1; i < N; i++) if (st.lh * (st.x[i] - st.x[i - 1]) ** 2 < 2 * st.b) g.fillRect(X(st.x[i - 1]), yTop - 3, X(st.x[i]) - X(st.x[i - 1]), 6);
    // connectors
    g.strokeStyle = "rgba(107,107,112,0.35)"; g.lineWidth = 1;
    for (let i = 0; i < N; i++) { g.beginPath(); g.moveTo(X(st.x[i]), yTop + 5); g.lineTo(X(st.y[i]), yBot - 5); g.stroke(); }
    // component extents on the bottom axis
    st.comps.forEach((c, k) => { if (c.size < 2) return; g.fillStyle = ranks[k] < 8 ? catColor(ranks[k]) : P.other; g.globalAlpha = 0.25; g.fillRect(X(c.ylo) - 3, yBot - 4, X(c.yhi) - X(c.ylo) + 6, 8); g.globalAlpha = 1; });
    // live edges as arcs below the bottom axis
    let wmax = 0; for (let k = 0; k < N * N; k++) wmax = Math.max(wmax, st.W[k]);
    if (arcs && wmax > 0) for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const w = st.W[i * N + j]; if (w <= 1e-9) continue;
      const x1 = X(st.y[i]), x2 = X(st.y[j]), r = Math.max(Math.abs(x2 - x1) / 2, 0.5);
      g.strokeStyle = P.edge || "#3f3e3b"; g.globalAlpha = Math.max(0.04, 0.6 * w / wmax); g.lineWidth = 1;
      g.beginPath(); g.ellipse((x1 + x2) / 2, yBot + 6, r, Math.min(r, 28) * 0.9, 0, 0, Math.PI); g.stroke();
    }
    g.globalAlpha = 1;
    // points
    hit = [];
    for (let i = 0; i < N; i++) {
      for (const [xx, yy, col] of [[X(st.x[i]), yTop, P.interior], [X(st.y[i]), yBot, colOf(i)]]) {
        g.beginPath(); g.arc(xx, yy, 4.2, 0, 2 * Math.PI); g.fillStyle = col; g.fill(); g.lineWidth = 1; g.strokeStyle = P.surface; g.stroke();
        hit.push([xx, yy, i]);
      }
    }
    // tick labels
    g.fillStyle = P.mute; g.textAlign = "center";
    for (const v of [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]) { g.fillText(ui.fmt(v), X(v), yTop - 9); g.fillText(ui.fmt(v), X(v), yBot + 47); }
  }
  function rankComps(comps) { const ids = comps.map((_, k) => k).sort((a, b) => comps[b].size - comps[a].size || a - b); const r = new Array(comps.length); ids.forEach((k, q) => { r[k] = q; }); return r; }
  new ResizeObserver(() => { if (Math.floor(el.clientWidth) !== Wd) draw(); }).observe(el);
  cv.addEventListener("pointermove", ev => {
    const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
    let best = null, bd = 64; for (const [x, y, i] of hit) { const d = (x - mx) ** 2 + (y - my) ** 2; if (d < bd) { bd = d; best = i; } }
    if (best == null) { ui.hideTip(); return; }
    const c = tooltip(best); if (c) ui.showTip(ev.clientX, ev.clientY, c);
  });
  cv.addEventListener("pointerleave", () => ui.hideTip());
  return { draw, setArcs(on) { arcs = !!on; draw(); } };
}
