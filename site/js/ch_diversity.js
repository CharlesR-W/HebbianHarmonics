// Chapter "diversity": Q-diversity (T1) and Q-eigen (T3) of SPEC §3.
// Part A (live): the scalar-drive experiment in three arms (fig #diversity-arms), the fusion cap at three
// stiffnesses (#diversity-cap), a hand-made output stage (#diversity-outside-fig); from diversity.json: the sweep
// against b, the potential-strength table and the multi-mode table. Part B (live): the s ± c example
// (#diversity-sc-fig) and trained block presets (#diversity-blocks); from eigen.json: the preset table.
// Model side: ch_diversity_model.js; custom plots: ch_diversity_plots.js.
import * as ui from "./ui.js";
import { lineChart, heatmap, barChart } from "./charts.js";
import { P, catColor, colorOf } from "./palette.js";
import * as M from "./ch_diversity_model.js";
import { curvesPanel, planePlot, logBars } from "./ch_diversity_plots.js";

const f2 = x => ui.fmt(x, 2), f3 = x => ui.fmt(x, 3);
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
const sd = a => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, a.length - 1)); };
const tex = s => { const e = ui.h("span", { html: s }); ui.renderMath(e); return e; };
const rel = x => (x < 1e-6 ? "< 10⁻⁶" : f2(x));
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const pow10 = v => { const e = Math.log10(v); return Math.abs(e - Math.round(e)) < 1e-9 && Math.abs(e) >= 3 ? "10" + String(Math.round(e)).split("").map(c => SUP[c]).join("") : ui.fmt(v, 2); };
// a table as a wide figure (no card), with a definitional caption
function tableFigure(el, { title, table, caption, controls = null }) {
  const f = ui.figure({ size: "wide", card: false, title, html: caption });
  f.setCaption({ html: caption });
  if (controls) f.body.append(controls);
  f.body.append(ui.h("div", { class: "table-scroll" }, table));
  el.replaceChildren(f.el); ui.renderMath(f.el);
  return f;
}

export async function init(panel, ctx) {
  const runner = M.createRunner({ cap: 24 });
  const dataP = Promise.all(["diversity", "eigen"].map(n => fetch(`data/sweeps2/${n}.json`).then(r => { if (!r.ok) throw new Error(`data/sweeps2/${n}.json: HTTP ${r.status}`); return r.json(); })));
  for (const s of ["sweep", "strength", "multi", "eigtable"]) ctx.slot(s).replaceChildren(ui.h("p", { class: "u-help" }, "loading data/sweeps2 …"));

  const A = armsFigure(ctx, runner);
  capFigure(ctx, runner, A);
  outsideFigure(ctx, A);
  const SC = scFigure(ctx);
  const BK = blocksFigure(ctx, runner);
  ctx.onParams(() => { A.fromURL(); SC.fromURL(); BK.fromURL(); });

  let div, eig;
  try { [div, eig] = await dataP; }
  catch (e) { for (const s of ["sweep", "strength", "multi", "eigtable"]) ctx.slot(s).replaceChildren(ui.h("p", { class: "placeholder" }, `Could not load the sweep data (${e.message}).`)); return; }
  diversityData(ctx, div);
  eigenData(ctx, eig);
}

// ===================================================================================== Part A, figure 1
const ATOURS = [
  { id: "default", label: "Default", note: "$b = 0.03$, $\\alpha = 1$, $k_{lo} = 3$, $\\sigma_t = 0.7$: the sweep's default setting. Toggle the map to see the heterogeneous members collapse.", p: { b: 0.03, alpha: 1, klo: 3, khi: 0, st: 0.7, seed: 0 } },
  { id: "strong", label: "Strong fusion", note: "$b = 0.3$, $\\alpha = 0.1$: conductances up to $b/\\alpha = 3$; one tight component per arm.", p: { b: 0.3, alpha: 0.1, klo: 3, khi: 0, st: 0.7, seed: 0 } },
  { id: "weak", label: "Weak fusion", note: "$b = 0.03$, $\\alpha = 10$: conductances at most $b/\\alpha = 0.003$, so edges select similar neurons but barely pull them together.", p: { b: 0.03, alpha: 10, klo: 3, khi: 0, st: 0.7, seed: 0 } },
  { id: "stiff", label: "Stiff rectifier", note: "$k_{lo} = 30$, $\\sigma_t = 1.4$, $b = 0.03$, $\\alpha = 1$: the most diverse potential in the sweep.", p: { b: 0.03, alpha: 1, klo: 30, khi: 0, st: 1.4, seed: 0 } },
  { id: "fine", label: "Small $b$", note: "$b = 0.006$, $\\alpha = 1$: a fine fusion scale; in the heterogeneous arm many neurons stay single.", p: { b: 0.006, alpha: 1, klo: 3, khi: 0, st: 0.7, seed: 0 } },
];
const ADEF = { ...ATOURS[0].p, map: false };

function armsFigure(ctx, runner) {
  const slot = ctx.slot("arms"), fig = slot.closest("figure");
  const L = ui.simLayout({ parent: slot, side: 300 });
  const grid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: "10px 18px" } });
  L.view.append(grid);
  const panels = {};
  for (const arm of M.ARM_IDS) { const box = ui.h("div", { style: { minWidth: "0" } }); grid.append(box); panels[arm] = curvesPanel(box, { height: 240, xlabel: "z (the shared scalar)", ylabel: "uᵢ(z)" }); }
  const tableBox = ui.h("div", { class: "table-scroll", style: { marginTop: "6px" } });
  L.below.append(tableBox);

  let p = readA(ctx), tour = ATOURS.find(t => t.id === ctx.param("tour", "")) || null;
  if (tour) p = { ...p, ...tour.p };
  const listeners = new Set();
  const tours = ui.chips({ parent: L.side, label: "Tours", items: ATOURS, active: tour ? tour.id : null, onPick: t => { tour = t; p = { ...p, ...t.p }; sync(); changed(); } });
  const bS = ui.slider({ label: "fusion scale $b$", min: 0.003, max: 0.3, log: true, sig: 2, value: p.b, fmt: x => ui.fmt(x, 2),
    help: "An edge lives iff $D_{ij}^2 &lt; 2b$.", onChange: x => { p.b = x; moved(); } });
  const aS = ui.segmented({ label: "stiffness $\\alpha$", options: [0.1, 1, 10].map(v => ({ value: v, label: String(v) })), value: p.alpha, onChange: x => { p.alpha = x; moved(); } });
  const mapT = ui.toggle({ label: "map each member onto its prototype", value: p.map, help: "Draw $(u_i - a_i)/c_i$, the member's best affine image on $p$.", onChange: v => { p.map = v; writeURL(); drawAll(); } });
  const kloS = ui.slider({ label: "slope below threshold $k_{lo}$", min: 0, max: 30, step: 0.5, value: p.klo, fmt: x => ui.fmtFixed(x, 1), onChange: x => { p.klo = x; moved(); } });
  const khiS = ui.slider({ label: "slope above threshold $k_{hi}$", min: 0, max: 30, step: 0.5, value: p.khi, fmt: x => ui.fmtFixed(x, 1), onChange: x => { p.khi = x; moved(); } });
  const stS = ui.slider({ label: "threshold spread $\\sigma_t$ (heterogeneous arm)", min: 0, max: 2, step: 0.05, value: p.st, fmt: x => ui.fmtFixed(x, 2), onChange: x => { p.st = x; moved(); } });
  const seedS = ui.slider({ label: "seed", min: 0, max: 9, step: 1, value: p.seed, fmt: x => String(x), help: "Attachments, thresholds and both banks.", onChange: x => { p.seed = x; moved(); } });
  ui.controls({ parent: L.side, primary: [bS, aS, mapT], advanced: [kloS, khiS, stS, seedS], advancedLabel: "Potential and seed" });
  const status = ui.h("div", { class: "u-help", style: { marginTop: "8px" } });
  L.side.append(status);
  ui.legend({ parent: L.side, items: [
    { label: "member, threshold $t_i &lt; 0$", color: colorOf("div", -1, -1, 1), shape: "line" },
    { label: "member, threshold $t_i &gt; 0$", color: colorOf("div", 1, -1, 1), shape: "line" },
    { label: "member, $t_i = 0$ (homogeneous, linear)", color: P.mute, shape: "line" },
    { label: "prototype $p(z)$", color: P.fg, shape: "line" },
  ] });

  let jobs = {}, unsub = [];
  function sync() { bS.set(p.b); aS.set(p.alpha); mapT.set(p.map); kloS.set(p.klo); khiS.set(p.khi); stS.set(p.st); seedS.set(p.seed); }
  function moved() { if (tour) { tour = null; tours.moved(); } changed(); }
  function writeURL() {
    const o = { tour: tour ? tour.id : null };
    for (const k of ["b", "alpha", "klo", "khi", "st", "seed"]) o[k === "alpha" ? "a" : k] = tour || p[k] === ADEF[k] ? null : p[k];
    o.map = p.map ? 1 : null;
    ctx.setParams(o);
  }
  const specOf = (arm, alpha = p.alpha) => ({ arm, klo: p.klo, khi: p.khi, st: p.st, b: p.b, alpha, seed: p.seed });
  function changed() {
    writeURL();
    unsub.forEach(f => f()); unsub = [];
    jobs = {};
    for (const arm of M.ARM_IDS) {
      const j = M.jobA(runner, specOf(arm)); j.prio = 2; jobs[arm] = j;
      unsub.push(runner.when(j, () => drawAll()));
    }
    drawAll();
    listeners.forEach(f => f());
  }
  function drawAll() {
    const done = M.ARM_IDS.filter(a => jobs[a] && jobs[a].done).map(a => jobs[a].result);
    // one vertical range for all three panels
    let lo = Infinity, hi = -Infinity;
    for (const r of done) { const c = r.multi[0]; if (!c) continue; for (const ys of curvesOf(r, c)) for (const y of ys) { lo = Math.min(lo, y); hi = Math.max(hi, y); } for (const y of c.protoGrid) { lo = Math.min(lo, y); hi = Math.max(hi, y); } }
    const pad = (hi - lo) * 0.06 || 0.1, yr = lo < hi ? [lo - pad, hi + pad] : [-1, 1];
    for (const arm of M.ARM_IDS) {
      const j = jobs[arm], pn = panels[arm], name = M.ARM_LABEL[arm];
      if (!j || !j.done) { pn.update({ title: `${name}`, x: M.ZGRID, curves: [], proto: null, yrange: yr, empty: "learning …", note: "" }); continue; }
      const r = j.result, c = r.multi[0];
      if (!c) { pn.update({ title: `${name}: no component with two or more members`, x: M.ZGRID, curves: [], proto: null, yrange: yr, empty: "all singletons at this b", note: "" }); continue; }
      const stv = Math.max(p.st, 0.05);
      const colors = c.members.map(i => (arm === "hetero" && p.st > 0 ? colorOf("div", r.t[i], -2 * stv, 2 * stv) : P.mute));
      pn.update({
        title: `${name}: largest component, ${c.size} of ${M.NA} neurons`, x: M.ZGRID, curves: curvesOf(r, c), colors, proto: c.protoGrid, yrange: yr,
        note: `ε<sub>proto</sub> = ${f2(c.protoErrMean)} (largest member ${f2(c.protoErrMax)}) · ${p.map ? "mapped onto p" : "raw responses"}`,
        tooltip: (q, k) => { const i = c.members[q], mp = c.map[q]; return ui.tipRows(`neuron ${i}`, [
          { name: "threshold tᵢ", value: f3(r.t[i]) },
          { name: "uᵢ(z) at z = " + ui.fmtFixed(M.ZGRID[k], 2), value: f3(r.curves[i][k]) },
          { name: "fit uᵢ ≈ aᵢ + cᵢ p: aᵢ, cᵢ", value: `${f3(mp.a)}, ${f3(mp.c)}` },
          { name: "ε_proto,i (held out)", value: f2(c.protoErr[q]) },
          { name: "ε_aff,i (held out)", value: f2(r.affErr[i]) }]); },
      });
    }
    drawTable(done);
  }
  function curvesOf(r, c) {
    return c.members.map((i, q) => p.map ? r.curves[i].map(u => (u - c.map[q].a) / (c.map[q].c || 1)) : r.curves[i]);
  }
  function drawTable(done) {
    const by = Object.fromEntries(done.map(r => [r.spec.arm, r]));
    const cell = (arm, fn) => { const r = by[arm]; return ui.h("td", { class: "num" }, r ? fn(r) : "…"); };
    const rows = [
      ["components $K$ (with ≥ 2 members)", r => `${r.K} (${r.multi.length})`],
      ["largest component: size · hops · span$/\\sqrt{2b}$", r => r.multi[0] ? `${r.multi[0].size} · ${r.multi[0].hops} · ${f2(r.multi[0].span / Math.sqrt(2 * r.spec.b))}` : "–"],
      ["$\\varepsilon_{\\rm aff}$, mean over neurons (nonlinearity)", r => f2(r.affMean)],
      ["$\\varepsilon_{\\rm proto}$, mean · largest member (diversity)", r => r.multi.length ? `${f2(r.protoErr)} · ${f2(r.protoErrMax)}` : "–"],
      ["$d_{\\rm eff}$ · numerical rank (descriptive)", r => r.multi.length ? `${ui.fmtFixed(r.dEff, 3)} · ${ui.fmtFixed(r.rank, 1)}` : "–"],
      ["residuals $r$ · $\\max|\\partial E/\\partial u|$", r => `${f2(r.residual)} · ${f2(r.eqResidual)}${r.residual < M.CLAIM_TOL ? "" : " (not converged)"}`],
      ["Euler steps to $r &lt; 10^{-8}$", r => ui.fmtInt(r.steps)],
    ];
    const t = ui.h("table", { class: "data", style: { whiteSpace: "nowrap" } },
      ui.h("thead", {}, ui.h("tr", {}, ui.h("th", {}, "live, this seed"), ...M.ARM_IDS.map(a => ui.h("th", { class: "num" }, M.ARM_LABEL[a])))),
      ui.h("tbody", {}, ...rows.map(([lab, fn]) => ui.h("tr", {}, ui.h("td", { html: lab }), ...M.ARM_IDS.map(a => cell(a, fn))))));
    tableBox.replaceChildren(t); ui.renderMath(t);
  }
  // progress line, refreshed while the runner pumps
  function showStatus() {
    const pend = M.ARM_IDS.filter(a => jobs[a] && !jobs[a].done);
    if (!pend.length) {
      const rs = M.ARM_IDS.map(a => jobs[a].result), ok = rs.every(r => r.residual < M.CLAIM_TOL);
      status.innerHTML = ok ? `All three arms stationary: $r &lt; 10^{-6}$ (largest ${f2(Math.max(...rs.map(r => r.residual)))}).` : "At least one arm stopped before $r &lt; 10^{-6}$: its equilibrium claims do not hold (see the table).";
    } else status.innerHTML = "Learning from $W = 0$: " + pend.map(a => `${M.ARM_LABEL[a]} step ${ui.fmtInt(jobs[a].sim.steps)}, $r$ = ${f2(jobs[a].sim.residual)}`).join("; ");
    ui.renderMath(status);
  }
  let lastStatus = 0;
  ui.animate(fig, () => { runner.pump(14); const t = performance.now(); if (t - lastStatus > 250) { lastStatus = t; showStatus(); } });
  changed();
  return {
    get p() { return p; }, jobs: () => jobs, specOf, onChange: f => listeners.add(f),
    fromURL() {
      if (!["tour", "b", "a", "klo", "khi", "st", "seed", "map"].some(k => ctx.params.has(k))) return;   // a link that does not concern this figure
      const t = ATOURS.find(x => x.id === ctx.param("tour", ""));
      const np = readA(ctx); if (t) Object.assign(np, t.p);
      const same = ["b", "alpha", "klo", "khi", "st", "seed", "map"].every(k => np[k] === p[k]) && (t ? t.id : null) === (tour ? tour.id : null);
      if (same) return;
      p = np; tour = t || null; if (tour) tours.set(tour.id); else tours.clear(); sync(); changed();
    },
  };
}
function readA(ctx) {
  const cl = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const a = ctx.param("a", ADEF.alpha);
  return { b: cl(ctx.param("b", ADEF.b), 0.003, 0.3), alpha: [0.1, 1, 10].includes(a) ? a : 1, klo: cl(ctx.param("klo", ADEF.klo), 0, 30), khi: cl(ctx.param("khi", ADEF.khi), 0, 30),
    st: cl(ctx.param("st", ADEF.st), 0, 2), seed: Math.round(cl(ctx.param("seed", ADEF.seed), 0, 9)), map: ctx.param("map", false) };
}

// ===================================================================================== Part A, the cap
function capFigure(ctx, runner, A) {
  const slot = ctx.slot("cap"), fig = slot.closest("figure");
  const row = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "14px 28px" } });
  const boxA = ui.h("div", { style: { minWidth: "0" } }), boxB = ui.h("div", { style: { minWidth: "0" } });
  const tA = ui.h("div", { class: "chart-title", html: "(a) Live edges: $D_{ij}^2/2b$, cumulative" }), tB = ui.h("div", { class: "chart-title" });
  const cA = ui.h("div"), cB = ui.h("div");
  boxA.append(tA, cA); boxB.append(tB, cB); row.append(boxA, boxB);
  const tbl = ui.h("div", { class: "table-scroll", style: { marginTop: "10px" } });
  const status = ui.h("div", { class: "u-help" });
  slot.append(row, tbl, status);
  ui.renderMath(tA);
  const cdf = lineChart(cA, { series: [], xlog: true, xrange: [1e-6, 2], yrange: [0, 1], height: 230, xlabel: "D²ᵢⱼ / 2b (training bank)", ylabel: "fraction of live edges", xname: "D²/2b", refs: [{ x: 1, label: "cap" }], yfmt: x => ui.fmtFixed(x, 1) });
  const bars = logBars(cB, { height: 230 });
  const ALPHAS = [0.1, 1, 10];
  let jobs = [], unsub = [];
  function changed() {
    unsub.forEach(f => f()); unsub = [];
    jobs = ALPHAS.map(al => { const j = M.jobA(runner, A.specOf("hetero", al)); j.prio = Math.max(j.prio, 1); return j; });
    jobs.forEach(j => unsub.push(runner.when(j, draw)));
    draw();
  }
  function draw() {
    const done = jobs.filter(j => j.done).map(j => j.result);
    const series = [], rowsT = [];
    let xmin = 1;
    jobs.forEach((j, k) => {
      const al = ALPHAS[k], col = catColor(k);
      if (!j.done) { rowsT.push([al, null]); return; }
      const r = j.result, xs = r.edges.map(e => e.r).sort((a, b) => a - b);
      xs.forEach(x => { if (x > 0) xmin = Math.min(xmin, x); });
      if (xs.length) series.push({ name: `α = ${al}`, x: xs, y: xs.map((_, q) => (q + 1) / xs.length), color: col, width: 2 });
      rowsT.push([al, r]);
    });
    const lo = Math.pow(10, Math.floor(Math.log10(Math.max(1e-9, Math.min(xmin, 0.1)))));
    cdf.update({ series: series.length ? series : [{ name: "", x: [1], y: [0], color: P.line }], xrange: [lo, 2] });
    // (b) for the alpha of the first figure
    const a0 = A.p.alpha, jb = jobs[ALPHAS.indexOf(a0)];
    tB.innerHTML = `(b) Member-to-prototype distance and its shape part, $\\alpha = ${a0}$`; ui.renderMath(tB);
    if (jb && jb.done) {
      const r = jb.result, cs = r.multi.slice(0, 8), b2 = 2 * r.spec.b;
      const tot = cs.map(c => c.d2p / b2), shp = cs.map(c => c.shape / b2);
      const pos = [...tot, ...shp].filter(x => x > 0);
      const lo2 = pos.length ? Math.pow(10, Math.floor(Math.log10(Math.min(...pos) / 3))) : 1e-4, hi2 = pos.length ? Math.pow(10, Math.ceil(Math.log10(Math.max(2, ...pos)))) : 10;
      bars.update({ labels: cs.map((c, k) => `c${k + 1}`), series: [{ name: "D²(Fᵢ, p)/2b, member mean", values: tot, color: P.ink2 }, { name: "shape part (after the affine map)", values: shp, color: P.cat[0] }],
        ref: { y: 1, label: "one edge's reach" }, yrange: [Math.max(lo2, 1e-8), hi2], xlabel: "component (largest first)", empty: "no component with two or more members",
        tooltip: (i, s) => { const c = cs[i]; return ui.tipRows(`component c${i + 1}: ${c.size} neurons`, [
          { name: "mean D²(Fᵢ, p)/2b", value: f3(tot[i]), color: P.ink2, shape: "square" }, { name: "shape part /2b", value: f3(shp[i]), color: P.cat[0], shape: "square" },
          { name: "shape share", value: ui.fmtPct(shp[i] / tot[i], 1) }, { name: "hop diameter", value: String(c.hops) }, { name: "span/√(2b)", value: f2(c.span / Math.sqrt(b2)) }, { name: "ε_proto (held out)", value: f2(c.protoErrMean) }]); } });
    } else bars.update({ labels: [], series: [], empty: "learning …" });
    // the table
    const t = ui.h("table", { class: "data", style: { whiteSpace: "nowrap" } },
      ui.h("thead", {}, ui.h("tr", {}, ...["heterogeneous arm", "live edges", "max D²/2b", "median D²/2b", "shape share of D²(Fᵢ, p)", "ε_proto (held out)", "residual r"].map((s, i) => ui.h("th", { class: i ? "num" : "" }, s)))),
      ui.h("tbody", {}, ...rowsT.map(([al, r], k) => {
        const sw = ui.h("span", { class: "swatch", style: { "--sw": catColor(k) } });
        if (!r) return ui.h("tr", {}, ui.h("td", {}, sw, ` α = ${al}`), ui.h("td", { class: "num", colspan: 6 }, "learning …"));
        const xs = r.edges.map(e => e.r).sort((a, b) => a - b), med = xs.length ? xs[(xs.length - 1) >> 1] : NaN;
        const d2 = r.multi.reduce((a, c) => a + c.d2p * c.size, 0), sh = r.multi.reduce((a, c) => a + c.shape * c.size, 0);
        return ui.h("tr", {}, ui.h("td", {}, sw, ` α = ${al}`), ...[String(xs.length), xs.length ? f3(xs[xs.length - 1]) : "–", f2(med), d2 > 0 ? ui.fmtPct(sh / d2, 1) : "–", r.multi.length ? f2(r.protoErr) : "–", f2(r.residual)].map(s => ui.h("td", { class: "num" }, s)));
      })));
    tbl.replaceChildren(t);
    const pend = jobs.filter(j => !j.done);
    status.textContent = pend.length ? `Learning: ${pend.map(j => `α = ${j.spec.alpha}, step ${ui.fmtInt(j.sim.steps)}`).join("; ")} …` : `Settings: b = ${ui.fmt(A.p.b, 2)}, k_lo = ${A.p.klo}, k_hi = ${A.p.khi}, σ_t = ${A.p.st}, seed ${A.p.seed}.`;
  }
  ui.animate(fig, () => runner.pump(14));
  let lastS = 0; ui.animate(fig, () => { const t = performance.now(); if (t - lastS > 300 && jobs.some(j => !j.done)) { lastS = t; draw(); } });
  A.onChange(changed);
  changed();
}

// ===================================================================================== Part A, output stage
function outsideFigure(ctx, A) {
  const slot = ctx.slot("outside");
  const row = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "14px 28px", alignItems: "start" } });
  const left = ui.h("div", { style: { minWidth: "0" } }), right = ui.h("div", { style: { minWidth: "0", maxWidth: "340px" } });
  row.append(left, right); slot.append(row);
  const cp = curvesPanel(left, { height: 250, xlabel: "z (the shared scalar)", ylabel: "uᵢ(z) or rᵢ(z)" });
  let arm = ctx.param("oarm", "homo"); if (!M.ARM_IDS.includes(arm)) arm = "homo";
  let on = ctx.param("oon", true), off = Math.max(0, Math.min(2, ctx.param("ooff", 1)));
  const armS = ui.segmented({ label: "arm", options: M.ARM_IDS.map(a => ({ value: a, label: M.ARM_LABEL[a] })), value: arm, stack: true, onChange: v => { arm = v; ctx.setParams({ oarm: v === "homo" ? null : v }); draw(); } });
  const onT = ui.toggle({ label: "output stage $r_i = \\max(0, a_i u_i + c_i)$", value: on, onChange: v => { on = v; ctx.setParams({ oon: v ? null : 0 }); draw(); } });
  const offS = ui.slider({ label: "offset spread $\\sigma_{\\rm off}$", min: 0, max: 2, step: 0.05, value: off, fmt: x => ui.fmtFixed(x, 2), help: "In units of each member's own standard deviation.", onChange: x => { off = x; ctx.setParams({ ooff: x === 1 ? null : x }); draw(); } });
  ui.controls({ parent: right, primary: [armS, onT, offS] });
  const st = ui.stats({ parent: right, cols: 2, items: [
    { key: "pe", label: "$\\varepsilon_{\\rm proto}$ of the plotted functions", fmt: f2 },
    { key: "pu", label: "$\\varepsilon_{\\rm proto}$ of the learned $u_i$", fmt: f2 },
  ] });
  function draw() {
    const j = A.jobs()[arm];
    if (!j || !j.done) { cp.update({ title: `${M.ARM_LABEL[arm]} arm`, x: M.ZGRID, curves: [], proto: null, yrange: [-1, 1], empty: "waiting for the runs of the first figure …", note: "" }); st.update({ pe: NaN, pu: NaN }); return; }
    const r = j.result, c = r.multi[0];
    if (!c) { cp.update({ title: `${M.ARM_LABEL[arm]} arm`, x: M.ZGRID, curves: [], proto: null, yrange: [-1, 1], empty: "no component with two or more members at this b", note: "" }); st.update({ pe: NaN, pu: NaN }); return; }
    const o = M.outputStage(r, c, off, on);
    let lo = Infinity, hi = -Infinity; for (const ys of o.curves) for (const y of ys) { lo = Math.min(lo, y); hi = Math.max(hi, y); }
    const pad = (hi - lo) * 0.06 || 0.1;
    cp.update({ title: `${M.ARM_LABEL[arm]} arm, largest component (${c.size} neurons): ${on ? "rates after the hand-made stage" : "learned responses"}`, x: M.ZGRID, curves: o.curves, colors: on ? o.offs.map(x => colorOf("div", x, -2 * Math.max(off, 0.05), 2 * Math.max(off, 0.05))) : c.members.map(i => colorOf("div", r.t[i], -2 * Math.max(A.p.st, 0.05), 2 * Math.max(A.p.st, 0.05))), proto: o.protoGrid, yrange: [lo - pad, hi + pad],
      note: on ? `Not part of the model: the gains and offsets are random draws.${o.silent ? ` ${o.silent} member${o.silent > 1 ? "s" : ""} silent on every sample (left out).` : ""}` : "The learned responses themselves.",
      tooltip: (q, k) => ui.tipRows(`neuron ${c.members[q]}`, on ? [{ name: "gain aᵢ·sd(uᵢ)", value: f3(o.gains[q]) }, { name: "offset cᵢ", value: f3(o.offs[q]) }, { name: `rᵢ at z = ${ui.fmtFixed(M.ZGRID[k], 2)}`, value: f3(o.curves[q][k]) }] : [{ name: "threshold tᵢ", value: f3(r.t[c.members[q]]) }, { name: `uᵢ at z = ${ui.fmtFixed(M.ZGRID[k], 2)}`, value: f3(o.curves[q][k]) }]) });
    st.set("pe", o.protoErr, o.silent ? `${o.silent} silent, left out` : "held out, evaluation bank");
    st.set("pu", c.protoErrMean, "for comparison");
  }
  A.onChange(draw);
  // redraw when the first figure's runs finish
  let seen = "";
  ui.animate(slot.closest("figure"), () => { const js = A.jobs(), k = M.ARM_IDS.map(a => js[a] && js[a].done ? 1 : 0).join("") + JSON.stringify(A.p); if (k !== seen) { seen = k; draw(); } });
  draw();
}

// ===================================================================================== Part B, s ± c
const SCTOURS = [
  { id: "equal", label: "Equal sizes", note: "$n_A = n_B = 12$, $\\sigma_c^2 = 0.1$: $\\psi_1 = s$, $\\psi_2 = c$, not the prototypes. (At $\\sigma_c^2 = 1$ the two eigenvalues would coincide and no single pair is singled out.)", p: { sc2: 0.1, na: 12, leak: 0 } },
  { id: "unequal", label: "20 / 4", note: "$n_A = 20$, $n_B = 4$, $\\sigma_c^2 = 0.1$: $\\psi_1$ tilts toward $p_A$.", p: { sc2: 0.1, na: 20, leak: 0 } },
  { id: "unequal5", label: "20 / 4, $\\sigma_c^2 = 0.5$", note: "Less overlap between the prototypes: $\\psi_1$ swings further toward $p_A$.", p: { sc2: 0.5, na: 20, leak: 0 } },
  { id: "orth", label: "Orthogonal prototypes", note: "$\\sigma_c^2 = 1$, $n_A = 16$, $n_B = 8$: the prototypes are orthogonal, $\\psi_a$ are the prototypes and $q = \\nu n$.", p: { sc2: 1, na: 16, leak: 0 } },
  { id: "leak", label: "Leaky attachments", note: "$\\sigma_c^2 = 0.5$, $n_A = 16$, leak 0.2: members differ slightly; the identical-member formula is off.", p: { sc2: 0.5, na: 16, leak: 0.2 } },
];
function scFigure(ctx) {
  const slot = ctx.slot("sc"), NTOT = 24;
  const row = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "14px 28px", alignItems: "start" } });
  const left = ui.h("div", { style: { minWidth: "0" } }), right = ui.h("div", { style: { minWidth: "0" } });
  row.append(left, right); slot.append(row);
  left.append(ui.panelHead("The plane of response functions $x\\,s + y\\,c$", "function"));
  const planeBox = ui.h("div"); left.append(planeBox);
  const plane = planePlot(planeBox, { height: 400 });
  const readP = () => { const t = SCTOURS.find(x => x.id === ctx.param("sct", "")); if (t) return { p: { ...t.p }, t };
    return { p: { sc2: Math.max(0.02, Math.min(1, ctx.param("sc2", 0.1))), na: Math.round(Math.max(1, Math.min(NTOT - 1, ctx.param("na", 12)))), leak: ctx.param("leak", 0) > 0 ? 0.2 : 0 }, t: null }; };
  let { p, t: tour } = readP();
  const tours = ui.chips({ parent: right, label: "Tours", items: SCTOURS, active: tour ? tour.id : null, onPick: t => { tour = t; p = { ...t.p }; sync(); compute(); } });
  const scS = ui.slider({ label: "contrast variance $\\sigma_c^2 = \\operatorname{var} c$", min: 0.02, max: 1, log: true, sig: 2, value: p.sc2, fmt: x => ui.fmt(x, 2), marks: [{ value: 1, label: "1" }],
    onInput: x => { p.sc2 = x; moved(); } });
  const naS = ui.slider({ label: "cluster sizes $n_A$ / $n_B$", min: 1, max: NTOT - 1, step: 1, value: p.na, fmt: x => `${x} / ${NTOT - x}`, marks: [{ value: 12, label: "equal" }], onInput: x => { p.na = x; moved(); } });
  const lkS = ui.segmented({ label: "attachment leak", options: [{ value: 0, label: "none" }, { value: 0.2, label: "U(0, 0.2)" }], value: p.leak, onChange: v => { p.leak = v; moved(); } });
  ui.controls({ parent: right, primary: [scS, naS, lkS] });
  const qBox = ui.h("div"); right.append(ui.h("div", { class: "chart-title", style: { marginTop: "10px" } }, "Top two eigenvalues and two predictions"), qBox);
  const qc = barChart(qBox, { labels: ["q₁", "q₂"], series: [{ name: "q (evaluation bank)", values: [0, 0], color: P.ink2 }, { name: "ν n", values: [0, 0], color: P.cat[3] }, { name: "cluster Gram (6)", values: [0, 0], color: P.cat[2] }], height: 170, ylabel: "variance × neurons" });
  const st = ui.stats({ parent: right, cols: 2, items: [
    { key: "s", label: "$|\\cos(\\psi_1, s)|$", fmt: f3 }, { key: "p", label: "$|\\cos(\\psi_1, p_{\\rm larger})|$", fmt: f3 },
    { key: "rho", label: "prototype cosine $\\rho$", fmt: f3 }, { key: "comm", label: "$\\lVert[K, \\Sigma_g]\\rVert$ (relative)", fmt: rel },
    { key: "r", label: "stationarity residual $r$", fmt: f2 },
  ] });
  function sync() { scS.set(p.sc2); naS.set(p.na); lkS.set(p.leak); }
  function moved() { if (tour) { tour = null; tours.moved(); } schedule(); }
  let pendingRaf = 0;
  function schedule() { if (pendingRaf) return; pendingRaf = requestAnimationFrame(() => { pendingRaf = 0; compute(); }); }
  function compute() {
    ctx.setParams(tour ? { sct: tour.id, sc2: null, na: null, leak: null } : { sct: null, sc2: p.sc2 === 0.1 ? null : p.sc2, na: p.na === 12 ? null : p.na, leak: p.leak ? 1 : null });
    const r = M.computeSC({ sc2: p.sc2, nA: p.na, nB: NTOT - p.na, mix: p.leak });
    // the exact operator's ellipse: second-moment matrix of the neurons in isometric coordinates
    const sc = Math.sqrt(p.sc2); let m11 = 0, m12 = 0, m22 = 0;
    for (const [x, y] of r.neurons) { const Y = sc * y; m11 += x * x; m12 += x * Y; m22 += Y * Y; }
    const tr = m11 + m22, det = m11 * m22 - m12 * m12, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const l1 = tr / 2 + disc, l2 = Math.max(0, tr / 2 - disc), ang = Math.abs(m12) > 1e-14 ? Math.atan2(l1 - m11, m12) : (m11 >= m22 ? 0 : Math.PI / 2);
    plane.update({ sc2: p.sc2, neurons: r.neurons, groups: r.neurons.map((_, i) => (i < p.na ? 0 : 1)), proto: r.proto, psi: r.degenerate ? [] : r.psi, colors: [P.cat[0], P.cat[1]], n: [p.na, NTOT - p.na],
      ell: { axes: [Math.sqrt(l1 / r.N), Math.sqrt(l2 / r.N)], angle: ang } });
    qc.update({ series: [{ name: "q (evaluation bank)", values: [...r.q], color: P.ink2 }, { name: "ν n (sorted)", values: r.nun, color: P.cat[3] }, { name: "cluster Gram (6)", values: r.qGram, color: P.cat[2] }],
      tooltip: (i, s) => ui.tipRows(`q${i ? "₂" : "₁"}`, [{ name: "evaluation bank", value: f3(r.q[i]) }, { name: "exact (H Σ Hᵀ)", value: f3(r.qExact[i]) }, { name: "ν n", value: f3(r.nun[i]) }, { name: "cluster Gram (6)", value: f3(r.qGram[i]) }]) });
    if (r.degenerate) { st.set("s", NaN, "q₁ = q₂: any rotation"); st.set("p", NaN, "degenerate"); }
    else { st.set("s", r.cosPsiS[0], `|cos(ψ₂, c)| = ${f3(r.cosPsiC[1])}`); st.set("p", r.cosPsiProto[0][r.larger], `cluster ${r.larger ? "B" : "A"}, n = ${Math.max(p.na, NTOT - p.na)}`); }
    st.set("rho", r.rho, `theory (1−σ²)/(1+σ²) = ${f3((1 - p.sc2) / (1 + p.sc2))}`);
    st.set("comm", r.comm, "0 ⇒ ψ = Σ_g's modes");
    st.set("r", r.residual, `max|∂E/∂u| = ${f2(r.eqResidual)}; ${r.K} components, b = ${ui.fmt(r.b, 2)}`);
  }
  compute();
  return { fromURL() { if (!["sct", "sc2", "na", "leak"].some(k => ctx.params.has(k))) return; const n = readP(); if (JSON.stringify(n.p) === JSON.stringify(p) && (n.t ? n.t.id : null) === (tour ? tour.id : null)) return; p = n.p; tour = n.t; if (tour) tours.set(tour.id); else tours.clear(); sync(); compute(); } };
}

// ===================================================================================== Part B, blocks
const BTOURS = [
  { id: "uncorr", label: "Uncorrelated blocks", note: "Cross-block correlation 0, linear, $b = 0.1$: four orthogonal clusters; $\\psi \\approx$ prototypes $\\approx$ $\\Sigma_g$'s modes (close, not exact), $q \\approx \\nu n$.", p: { rx: 0, pot: "linear", b: 0.1 } },
  { id: "corr", label: "Correlated blocks", note: "Cross-block correlation 0.4, linear, $b = 0.1$: the same four clusters, overlapping prototypes; $\\psi$ = $\\Sigma_g$'s modes, not the prototypes.", p: { rx: 0.4, pot: "linear", b: 0.1 } },
  { id: "merge", label: "Correlated, $b = 0.23$", note: "Two of the four clusters have merged; the eigenfunctions still follow $\\Sigma_g$ more closely than the prototypes.", p: { rx: 0.4, pot: "linear", b: 0.23 } },
  { id: "rect", label: "Rectifying", note: "Uncorrelated blocks with the rectifying potential at $b = 0.1$: cluster counts shift and the eigenfunctions are partly nonlinear.", p: { rx: 0, pot: "rect", b: 0.1 } },
];
function blocksFigure(ctx, runner) {
  const slot = ctx.slot("blocks"), fig = slot.closest("figure");
  const L = ui.simLayout({ parent: slot, side: 300 });
  const grid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "14px 24px" } });
  const mk = title => { const b = ui.h("div", { style: { minWidth: "0" } }), t = ui.h("div", { class: "chart-title", html: title }), c = ui.h("div"); b.append(t, c); grid.append(b); ui.renderMath(t); return c; };
  const hA = mk("(a) $\\cos(\\psi_a, p_k)$: eigenfunctions against cluster prototypes"), hB = mk("(b) $|\\cos(\\psi_a, e_b)|$: against $\\Sigma_g$'s modes"), qBox = mk("(c) $q_a$ against $\\nu n$ and the cluster Gram (6)");
  L.view.append(grid);
  const hmA = heatmap(hA, { values: [[0]], map: "div", range: [-1, 1], size: 300, zlabel: "signed cosine", xlabel: "prototype (largest first)", ylabel: "eigenfunction" });
  const hmB = heatmap(hB, { values: [[0]], map: "seq", range: [0, 1], size: 300, zlabel: "|cosine|", xlabel: "Σ_g mode (by λ)", ylabel: "eigenfunction" });
  const qc = barChart(qBox, { labels: ["q₁"], series: [{ name: "q", values: [0] }], height: 190, ylabel: "variance × neurons" });
  const readP = () => { const t = BTOURS.find(x => x.id === ctx.param("bt", "")); if (t) return { p: { ...t.p, seed: 0 }, t };
    return { p: { rx: ctx.param("rx", 0) > 0 ? 0.4 : 0, pot: ctx.param("pot", "linear") === "rect" ? "rect" : "linear", b: Math.max(0.01, Math.min(0.5, ctx.param("bb", 0.1))), seed: Math.round(Math.max(0, Math.min(4, ctx.param("bseed", 0)))) }, t: null }; };
  let { p, t: tour } = readP();
  const tours = ui.chips({ parent: L.side, label: "Tours", items: BTOURS, active: tour ? tour.id : null, onPick: t => { tour = t; p = { ...t.p, seed: p.seed }; sync(); changed(); } });
  const rxS = ui.segmented({ label: "cross-block correlation", options: [{ value: 0, label: "0" }, { value: 0.4, label: "0.4" }], value: p.rx, onChange: v => { p.rx = v; moved(); } });
  const potS = ui.segmented({ label: "potential", options: [{ value: "linear", label: "linear" }, { value: "rect", label: "rectifying" }], value: p.pot, onChange: v => { p.pot = v; moved(); } });
  const bS = ui.slider({ label: "fusion scale $b$", min: 0.01, max: 0.5, log: true, sig: 2, value: p.b, fmt: x => ui.fmt(x, 2), onChange: x => { p.b = x; moved(); } });
  const seedS = ui.slider({ label: "seed", min: 0, max: 4, step: 1, value: p.seed, fmt: x => String(x), onChange: x => { p.seed = x; moved(); } });
  ui.controls({ parent: L.side, primary: [rxS, potS, bS], advanced: [seedS] });
  const st = ui.stats({ parent: L.below, cols: window.matchMedia("(max-width: 700px)").matches ? 2 : 5, items: [
    { key: "K", label: "components $K$", fmt: x => String(x) },
    { key: "diam", label: "max $D$ to prototype", fmt: f2 },
    { key: "pc", label: "prototype $|\\cos|$, mean", fmt: f2 },
    { key: "bp", label: "$\\psi$ vs prototype", fmt: f3 },
    { key: "be", label: "$\\psi$ vs $\\Sigma_g$ mode", fmt: f3 },
    { key: "en", label: "$q$ vs $\\nu n$, rel. error", fmt: f2 },
    { key: "eg", label: "$q$ vs Gram (6), rel. error", fmt: f2 },
    { key: "comm", label: "$\\lVert[K, \\Sigma_g]\\rVert$ (relative)", fmt: x => (Number.isFinite(x) ? f2(x) : "–") },
    { key: "r", label: "stationarity residual $r$", fmt: x => (typeof x === "number" ? f2(x) : x) },
  ] });
  function sync() { rxS.set(p.rx); potS.set(p.pot); bS.set(p.b); seedS.set(p.seed); }
  function moved() { if (tour) { tour = null; tours.moved(); } changed(); }
  let job = null, unsub = null;
  function changed() {
    ctx.setParams(tour ? { bt: tour.id, rx: null, pot: null, bb: null, bseed: p.seed || null } : { bt: null, rx: p.rx || null, pot: p.pot === "linear" ? null : p.pot, bb: p.b === 0.1 ? null : p.b, bseed: p.seed || null });
    if (unsub) unsub();
    job = M.jobB(runner, { rx: p.rx, pot: p.pot, b: p.b, seed: p.seed }); job.prio = 3;
    unsub = runner.when(job, draw);
    if (!job.done) st.set("r", "learning …", "");
  }
  function draw() {
    const r = job.result;
    const rows = Array.from({ length: r.rr }, (_, a) => `ψ${sub(a + 1)}`);
    hmA.update({ values: r.cosP, rows: r.rr, cols: r.cosP[0].length, rowLabels: rows, colLabels: r.cosP[0].map((_, k) => `p${sub(k + 1)}`),
      tooltip: (i, j, v) => ui.tipRows(`ψ${sub(i + 1)} and prototype p${sub(j + 1)} (${r.sizes[j]} neurons)`, [{ name: "cosine (sign fixed by ψ's best match)", value: f3(v) }]) });
    hmB.update({ values: r.cosE, rows: r.rr, cols: r.nModes, rowLabels: rows, colLabels: Array.from({ length: r.nModes }, (_, b) => `e${sub(b + 1)}`),
      tooltip: (i, j, v) => ui.tipRows(`ψ${sub(i + 1)} and Σ_g mode e${sub(j + 1)}`, [{ name: "|cosine|", value: f3(v) }, { name: "λ of the mode", value: f3(r.lam[j]) }]) });
    qc.update({ labels: Array.from(r.q, (_, a) => `q${sub(a + 1)}`), series: [{ name: "q (evaluation bank)", values: r.q, color: P.ink2 }, { name: "ν n (sorted)", values: r.nun, color: P.cat[3] }, { name: "cluster Gram (6)", values: r.qGram, color: P.cat[2] }] });
    st.set("K", r.K, `sizes ${r.sizes.slice(0, 6).join(", ")}${r.sizes.length > 6 ? ", …" : ""}`);
    st.set("diam", r.diamMax, `prototype sd ${f2(r.protoSd)}`);
    st.set("pc", r.protoCos, `max ${f3(r.protoCosMax)}`);
    st.set("bp", mean(r.bestP), "mean best |cos| over ψ");
    st.set("be", mean(r.bestE), "mean best |cos| over ψ");
    st.set("en", r.errNun, ""); st.set("eg", r.errGram, "");
    st.set("comm", r.comm ?? NaN, r.comm != null ? `spectral identity ${f2(r.identityErr)} · sampling ${f2(r.samplingErr)}` : "linear neurons only");
    st.set("r", r.residual, `max|∂E/∂u| = ${f2(r.eqResidual)}; ${r.residual < M.CLAIM_TOL ? `stationary (${ui.fmtInt(r.steps)} steps)` : "not converged"}`);
  }
  ui.animate(fig, () => runner.pump(14));
  changed();
  return { fromURL() { if (!["bt", "rx", "pot", "bb", "bseed"].some(k => ctx.params.has(k))) return; const n = readP(); if (JSON.stringify(n.p) === JSON.stringify(p) && (n.t ? n.t.id : null) === (tour ? tour.id : null)) return; p = n.p; tour = n.t; if (tour) tours.set(tour.id); else tours.clear(); sync(); changed(); } };
}
const sub = n => String(n).split("").map(d => "₀₁₂₃₄₅₆₇₈₉"[+d]).join("");

// ===================================================================================== diversity.json
function diversityData(ctx, d) {
  const fill = (name, s) => { const el = ctx.slot(name); if (el) { el.innerHTML = s; ui.renderMath(el); } };
  const cfg = d.meta.config, bg = cfg.b_grid, al = cfg.alphas;
  const runs = (arm, a) => d.rank1.filter(r => r.arm === arm && r.alpha === a);
  const seedMean = (arm, a, key) => bg.map((_, k) => mean(runs(arm, a).map(r => r[key][k])));
  const seedLo = (arm, a, key) => bg.map((_, k) => Math.min(...runs(arm, a).map(r => r[key][k])));
  const seedHi = (arm, a, key) => bg.map((_, k) => Math.max(...runs(arm, a).map(r => r[key][k])));
  fill("j-runs", `${d.meta.n_converged} of ${d.meta.n_runs}`);
  // ---- the sweep figure
  const slot = ctx.slot("sweep");
  const row = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "14px 28px" } });
  const bA = ui.h("div", { style: { minWidth: "0" } }), bB = ui.h("div", { style: { minWidth: "0" } });
  const tA = ui.h("div", { class: "chart-title", html: "(a) Nonlinearity of each neuron, $\\varepsilon_{\\rm aff}$" }), tB = ui.h("div", { class: "chart-title", html: "(b) Diversity within clusters, $\\varepsilon_{\\rm proto}$" });
  const cA = ui.h("div"), cB = ui.h("div"); bA.append(tA, cA); bB.append(tB, cB); row.append(bA, bB);
  slot.replaceChildren(row); ui.renderMath(row);
  const aff0 = mean(runs("hetero", 1).map(r => r.uncoupled.affine_err_mean));
  const ser = key => [...al.map((a, k) => ({ name: `heterogeneous, α = ${a}`, x: bg, y: seedMean("hetero", a, key), band: [seedLo("hetero", a, key), seedHi("hetero", a, key)], color: catColor(k), dots: true })),
    { name: "homogeneous, α = 1", x: bg, y: seedMean("homo", 1, key), color: P.mute, dots: true }];
  lineChart(cA, { series: ser("affine_err_mean"), xlog: true, height: 240, xlabel: "fusion scale b", ylabel: "ε_aff (mean over neurons)", xname: "b", yrange: [0, 0.14], refs: [{ y: aff0, label: "heterogeneous, W = 0" }], yfmt: x => ui.fmtFixed(x, 2) });
  const pe = ser("proto_err_w").map(s => ({ ...s, y: s.y.map(v => Math.max(v, 1e-7)), band: s.band ? s.band.map(a => a.map(v => Math.max(v, 1e-7))) : undefined }));
  lineChart(cB, { series: pe, xlog: true, ylog: true, height: 240, xlabel: "fusion scale b", ylabel: "ε_proto (members vs prototype)", xname: "b", yrange: [1e-7, 0.1], yfmt: pow10 });
  // ---- slots in the prose
  fill("j-aff-w0", `${f2(aff0)} ± ${ui.fmt(sd(runs("hetero", 1).map(r => r.uncoupled.affine_err_mean)), 1)}`);
  fill("j-aff-fused", f2(seedMean("hetero", 0.1, "affine_err_mean")[bg.length - 1]));
  const hetMeans = al.flatMap(a => seedMean("hetero", a, "proto_err_w"));
  fill("j-pe-max", f2(Math.max(...hetMeans)));
  fill("j-pe-max1", f2(Math.max(...d.rank1.filter(r => r.arm === "hetero").flatMap(r => r.proto_err_w))));
  const dm = al.flatMap(a => seedMean("hetero", a, "deff_w"));
  fill("j-deff", `${ui.fmtFixed(Math.min(...dm), 2)} and ${ui.fmtFixed(Math.max(...dm), 2)}`);
  fill("j-rank", ui.fmtFixed(Math.max(...d.rank1.filter(r => r.arm === "hetero").flatMap(r => r.rank_w)), 0));
  fill("j-pe-homo", f2(Math.max(...d.rank1.filter(r => r.arm === "homo").flatMap(r => r.proto_err_w))));
  fill("j-pe-lin", ui.fmt(Math.max(...d.rank1.filter(r => r.arm === "linear").flatMap(r => [...r.proto_err_w, ...r.affine_err_mean])), 1));
  const a01 = seedMean("hetero", 0.1, "proto_err_w"), kmax = a01.indexOf(Math.max(...a01));
  fill("j-a01-lo", f2(a01[kmax])); fill("j-a01-blo", ui.fmt(bg[kmax], 2));
  fill("j-a01-hi", f2(a01[bg.length - 1])); fill("j-a01-bhi", ui.fmt(bg[bg.length - 1], 2));
  const a10 = seedMean("hetero", 10, "proto_err_w").slice(2);
  fill("j-a10", `${f2(Math.min(...a10))} and ${f2(Math.max(...a10))}`); fill("j-a10-b", ui.fmt(bg[2], 2));
  // ---- strength table
  const S = d.strength, pots = [], sts = [];
  for (const r of S) { const k = `${r.klo}|${r.khi}`; if (!pots.includes(k)) pots.push(k); if (!sts.includes(r.sigma_t)) sts.push(r.sigma_t); }
  const grp = (k, s) => S.filter(r => `${r.klo}|${r.khi}` === k && r.sigma_t === s);
  const potLabel = k => { const [lo, hi] = k.split("|").map(Number); return hi === 0 ? `rectifier, $k_{lo} = ${lo}$${lo === 3 ? " (default)" : ""}` : `compressive, $k_{lo} = ${lo}$, $k_{hi} = ${hi}$`; };
  const th = (s, cls = "num", extra = {}) => ui.h("th", { class: cls, html: s, ...extra });
  const tS = ui.h("table", { class: "data", style: { whiteSpace: "nowrap" } },
    ui.h("thead", {}, ui.h("tr", {}, th("potential", ""), th("$\\sigma_t$"), th("$\\varepsilon_{\\rm aff}$"), th("$\\varepsilon_{\\rm proto}$, mean ± sd"), th("largest $\\varepsilon_{\\rm proto}$"), th("largest $d_{\\rm eff}$"), th("components $K$"))),
    ui.h("tbody", {}, ...pots.flatMap(k => sts.map((s, si) => {
      const g = grp(k, s);
      return ui.h("tr", {}, ui.h("td", { html: si === 0 ? potLabel(k) : "" }), ui.h("td", { class: "num" }, String(s)),
        ui.h("td", { class: "num" }, f2(mean(g.map(r => r.affine_err_mean)))),
        ui.h("td", { class: "num" }, `${f2(mean(g.map(r => r.proto_err_w)))} ± ${ui.fmt(sd(g.map(r => r.proto_err_w)), 1)}`),
        ui.h("td", { class: "num" }, f2(Math.max(...g.map(r => r.proto_err_w)))), ui.h("td", { class: "num" }, ui.fmtFixed(Math.max(...g.map(r => r.deff_w)), 3)),
        ui.h("td", { class: "num" }, g.map(r => r.K).join(", ")));
    }))));
  tableFigure(ctx.slot("strength"), { title: "Stronger and reshaped potentials, scalar drive at the default fusion setting", table: tS,
    caption: `<b>Plotted:</b> per potential and threshold spread $\\sigma_t$: $\\varepsilon_{\\rm aff}$ (mean over neurons) and $\\varepsilon_{\\rm proto}$ (size-weighted over components with at least two members), mean ± sd over seeds; the largest $\\varepsilon_{\\rm proto}$ and $d_{\\rm eff}$ in any single seed; the number of components $K$ for each seed. <b>Values:</b> heterogeneous thresholds, $b = ${cfg.default.b}$, $\\alpha = ${cfg.default.alpha}$, seeds ${[...new Set(S.map(r => r.seed))].join(", ")}, $N = ${cfg.N}$, $B = ${cfg.B}$; from <code>data/sweeps2/diversity.json</code> (<code>strength</code>).` });
  const g30 = grp("30|0", 0.7);
  fill("j-s30-aff", f2(mean(g30.map(r => r.affine_err_mean))));
  fill("j-s30-pe", `${f2(mean(g30.map(r => r.proto_err_w)))} ± ${ui.fmt(sd(g30.map(r => r.proto_err_w)), 1)}`);
  const best = S.reduce((a, r) => (r.proto_err_w > a.proto_err_w ? r : a), S[0]);
  fill("j-s-best", `${f2(best.proto_err_w)} ($k_{lo} = ${best.klo}$, $\\sigma_t = ${best.sigma_t}$)`);
  fill("j-s-deff", ui.fmtFixed(Math.max(...S.map(r => r.deff_w)), 2));
  const comp = sts.map(s => mean(grp("0|3", s).map(r => r.proto_err_w)));
  fill("j-s-comp", `$\\varepsilon_{\\rm proto}$ of ${f2(Math.min(...comp))} to ${f2(Math.max(...comp))}`);
  // ---- multi-mode table
  const mc = cfg.multi, mb = mc.b_grid, MM = d.multi;
  const mrows = [];
  for (const att of ["random", "block"]) for (const arm of ["hetero", "homo", "linear"]) {
    const g = MM.filter(r => r.attach === att && r.arm === arm && r.rho === 1);
    mrows.push({ att, arm, deff: mb.map((_, k) => mean(g.map(r => r.deff_w[k]))), pe: mb.map((_, k) => mean(g.map(r => r.proto_err_w[k]))), spl: mb.map((_, k) => mean(g.map(r => r.spline_err_mean[k]))), K: mb.map((_, k) => mean(g.map(r => r.K[k]))) });
  }
  const tM = ui.h("table", { class: "data", style: { whiteSpace: "nowrap" } },
    ui.h("thead", {}, ui.h("tr", {}, th("attachments", ""), th("arm", ""), ...mb.map(b => th(`$b = ${ui.fmt(b, 2)}$`)))),
    ui.h("tbody", {}, ...mrows.map(r => ui.h("tr", {}, ui.h("td", {}, r.arm === "hetero" ? (r.att === "random" ? "3 random inputs" : "3 inputs of one block") : ""), ui.h("td", {}, M.ARM_LABEL[r.arm]),
      ...mb.map((_, k) => ui.h("td", { class: "num", style: { whiteSpace: "nowrap" }, title: `components ${ui.fmtFixed(r.K[k], 1)} (seed mean); best single-scalar fit error ${f2(r.spl[k])}` }, `${ui.fmtFixed(r.deff[k], 2)} · ${f2(r.pe[k])}`))))));
  tableFigure(ctx.slot("multi"), { title: "Four input modes: within-cluster spread in the three arms", table: tM,
    caption: `<b>Plotted:</b> each cell is $d_{\\rm eff}$ · $\\varepsilon_{\\rm proto}$, size-weighted over components with at least two members, mean over seeds ${[...new Set(MM.map(r => r.seed))].join(", ")}; hover a cell for the mean number of components and the best single-scalar fit error (held-out relative error of a 1-D piecewise-linear fit of each response from the best single block scalar, mean over neurons). <b>Values:</b> four blocks of ${cfg.B / mc.blocks} inputs, each block driven by its own scalar, block variances ${mc.var.join(", ")}, no private noise ($\\rho = 1$), $\\alpha = ${mc.alpha}$, $N = ${cfg.N}$; from <code>data/sweeps2/diversity.json</code> (<code>multi</code>).` });
  const hb = mrows.find(r => r.att === "block" && r.arm === "hetero"), hr = mrows.find(r => r.att === "random" && r.arm === "hetero");
  const kb = mb.findIndex(b => b > 0.06); const kcut = kb < 0 ? mb.length : kb;
  fill("j-m-block-spl", ui.fmt(Math.max(...hb.spl.slice(0, kcut)), 1)); fill("j-m-block-b", ui.fmt(mb[kcut - 1], 2));
  fill("j-m-rand-spl0", f2(hr.spl[0])); fill("j-m-rand-spl1", f2(hr.spl[mb.length - 1]));
}

// ===================================================================================== eigen.json
function eigenData(ctx, d) {
  const fill = (name, s) => { const el = ctx.slot(name); if (el) { el.innerHTML = s; ui.renderMath(el); } };
  const pa = (sc2, nA, nB, mix = 0) => d.partA.find(r => r.sc2 === sc2 && r.nA === nA && r.nB === nB && r.mix === mix);
  const e = pa(0.1, 12, 12); fill("j-sc-eq", `${f3(e.cos_psi_proto_best[0])} ($\\psi_1$) and ${f2(e.cos_psi_proto_best[1])} ($\\psi_2$)`);
  fill("j-sc-20-01", f3(pa(0.1, 20, 4).cos_psi1_s)); fill("j-sc-20-05", f3(pa(0.5, 20, 4).cos_psi1_s));
  const o1 = pa(1, 16, 8), o2 = pa(1, 20, 4);
  fill("j-sc-orth", `exact spectrum (${o1.q_analytic.map(x => ui.fmt(x, 3)).join(", ")}) against $\\nu n$ = (${o1.nu_n.map(x => ui.fmt(x, 3)).join(", ")}) at 16/8, and (${o2.q_analytic.map(x => ui.fmt(x, 3)).join(", ")}) against (${o2.nu_n.map(x => ui.fmt(x, 3)).join(", ")}) at 20/4`);
  // ---- the preset table, with a b selector
  const cfg = d.meta.config, bg = cfg.b_grid;
  const slot = ctx.slot("eigtable");
  let k = bg.findIndex(b => Math.abs(b - 0.1046) < 1e-3); if (k < 0) k = 3;
  const sel = ui.segmented({ label: "fusion scale $b$", options: bg.map((b, i) => ({ value: i, label: ui.fmt(b, 2) })), value: k, onChange: v => { k = v; draw(); } });
  const PRE = { blocks_blockatt: ["uncorrelated blocks", "block"], blockscorr_blockatt: ["correlated blocks", "block"], blocks_randatt: ["uncorrelated blocks", "random"], power_randatt: ["power law", "random"] };
  const cols = [
    ["K_multi", "clusters", x => ui.fmtFixed(x, 1)], ["proto_cos_mean", "prototype |cos|", f2], ["psi_proto_cos_mean", "ψ vs prototype", f3], ["psi_sigma_cos_mean", "ψ vs Σ mode", f3],
    ["q_vs_nun_rel_err", "q vs νn", f2], ["q_vs_identical_rel_err", "q vs (6)", f2], ["commutator_rel", "commutator", f2], ["psi_nonlinear_frac", "ψ nonlinear", f2],
  ];
  const fig = tableFigure(slot, { title: "Trained networks in the sweep: eigenfunctions, prototypes and Σ modes", table: ui.h("table"), controls: sel.el,
    caption: `<b>Plotted:</b> mean over seeds ${d.meta.seeds.partB.join(", ")} (hover a cell for each seed) at the chosen $b$: components with at least two members; mean $|\\cos|$ between distinct multi-member prototypes; mean over the top $\\psi_a$ (one per multi-member component) of the best $|\\cos|$ with any prototype and with any $\\Sigma_g$ mode; relative errors $\\sum_a |q_a - \\hat q_a|/\\sum_a q_a$ against $\\nu n$ and against the cluster Gram (6); the relative commutator $\\lVert[H^\\top H, \\Sigma_g]\\rVert_F/(\\lVert H^\\top H\\rVert_F\\lVert\\Sigma_g\\rVert_F)$ (linear only); the held-out relative error of an affine fit of the top $\\psi_a$ from $g$ (rectifying only). <b>Values:</b> $\\alpha = ${cfg.alpha}$, $N = ${cfg.N}$, $B = ${cfg.B}$, evaluation bank ${cfg.S_eval}; correlated blocks: ${cfg.blockscorr}; from <code>data/sweeps2/eigen.json</code> (<code>runs</code>).` });
  const scroll = fig.body.querySelector(".table-scroll");
  function draw() {
    const rowsH = [];
    for (const pre of Object.keys(PRE)) for (const pot of ["linear", "rect"]) {
      const g = d.runs.filter(r => r.preset === pre && r.pot === pot);
      if (!g.length) continue;
      rowsH.push(ui.h("tr", {}, ui.h("td", {}, pot === "linear" ? PRE[pre][0] : ""), ui.h("td", {}, pot === "linear" ? PRE[pre][1] : ""), ui.h("td", {}, pot === "rect" ? "rectifying" : "linear"), ...cols.map(([key, , fm]) => {
        const v = g.map(r => r[key][k]).filter(x => x != null && Number.isFinite(x));
        return ui.h("td", { class: "num", title: v.length ? `seeds: ${v.map(x => ui.fmt(x, 3)).join(", ")}` : "" }, v.length ? fm(mean(v)) : "–");
      })));
    }
    scroll.replaceChildren(ui.h("table", { class: "data", style: { whiteSpace: "nowrap" } },
      ui.h("thead", {}, ui.h("tr", {}, ui.h("th", {}, "Σ_g"), ui.h("th", {}, "attachments"), ui.h("th", {}, "potential"), ...cols.map(([, l]) => ui.h("th", { class: "num" }, l)))),
      ui.h("tbody", {}, ...rowsH)));
  }
  draw();
}
