// Chapter "allocation", section (c): the added similarity-matching principle. A live projected-gradient optimiser
// (water-filling plot, per-neuron composition, counts per mode) and the sweep's three tables (c1, c2, c3) with a
// counts chart.
import * as ui from "./ui.js";
import { heatmap, barChart, lineChart, svgEl, scale, niceTicks } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as M from "./ch_allocation_model.js";

export const DEF_C = { spec: "lam8", nu: 0.1, ing: "none", b: 0.03, seed: 0 };
const KEYS_C = { spec: "cs", nu: "cn", ing: "ci", b: "cb", seed: "cz" };
export const TOURS_C = [
  { id: "law", label: "(c1) Mass law, no ingredient", note: "<code>lam8</code>, $\\nu = 0.1$: masses on the water-filling line; neurons mixed.", p: { spec: "lam8", nu: 0.1, ing: "none", b: 0.03, seed: 0 } },
  { id: "fusion-weak", label: "Weak fusion", note: "Fusion term with $b = 0.03$: the law is kept; some neurons purify.", p: { spec: "lam8", nu: 0.1, ing: "fusion", b: 0.03, seed: 0 } },
  { id: "fusion-strong", label: "Strong fusion", note: "Fusion term with $b = 0.1$: neurons merge, weak modes empty, the masses leave the law.", p: { spec: "lam8", nu: 0.1, ing: "fusion", b: 0.1, seed: 0 } },
  { id: "nonneg", label: "Non-negative, eigenbasis", note: "$x_{ia} \\ge 0$: nearly all neurons pure, but the counts over-populate weak modes and the masses move off the law.", p: { spec: "lam8", nu: 0.1, ing: "nonneg_eig", b: 0.03, seed: 0 } },
  { id: "rotated", label: "Non-negative, rotated", note: "$R^\\top x_i \\ge 0$ for a random rotation $R$: no pure neurons and $X^\\top X$ is not diagonal.", p: { spec: "lam8", nu: 0.1, ing: "nonneg_rotated", b: 0.03, seed: 0 } },
  { id: "nonneg-big", label: "Non-negative, $\\nu = 0.2$", note: "The largest budget: $\\mu &lt; 0$, every mode populated, and the pure counts land close to the law.", p: { spec: "lam8", nu: 0.2, ing: "nonneg_eig", b: 0.03, seed: 0 } },
];
const NUS = [0.05, 0.1, 0.2];

function clampC(p) {
  return {
    spec: M.SPECTRA[p.spec] ? p.spec : "lam8",
    nu: NUS.includes(+p.nu) ? +p.nu : 0.1,
    ing: M.INGREDIENTS.some(x => x.id === p.ing) ? p.ing : "none",
    b: Math.min(0.3, Math.max(0.003, +p.b || 0.03)),
    seed: Math.min(20, Math.max(0, Math.round(+p.seed || 0))),
  };
}

export function smFigure(ctx, slot) {
  const fig = slot.closest("figure");
  const readURL = () => {
    const t = TOURS_C.find(x => x.id === ctx.param("tc", ""));
    if (t) return { p: { ...t.p }, tour: t };
    const p = {}; for (const k in KEYS_C) p[k] = ctx.param(KEYS_C[k], DEF_C[k]);
    return { p: clampC(p), tour: null };
  };
  let { p, tour } = readURL();
  let st = null, cur = null, tickN = 0, countMax = 10;

  const L = ui.simLayout({ parent: slot });
  const tp = ui.transport({
    parent: L.view, watch: fig, autoplay: false, maxHistory: 200,
    speeds: [{ value: 40, label: "×1" }, { value: 400, label: "×10" }, { value: 4000, label: "×100" }], speed: 400,
    timeFmt: t => `iteration ${ui.fmtInt(t)}`,
    onTick: s => tick(s), onSeek: f => { cur = f; show(f, true); }, onReset: () => restart(),
  });
  const ttl = (t, s) => ui.h("div", {}, ui.h("div", { class: "chart-title", html: t }), ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" }, html: s }));
  const wfBox = ui.h("div", { class: "chart" }), compBox = ui.h("div"), cntBox = ui.h("div");
  L.view.append(
    ui.h("div", { style: { marginTop: "12px" } }, ttl("(a) Water-filling: $q_a^\\star = (\\lambda_a - \\mu)_+$ against the live masses", "grey: $\\lambda_a$ · dashed: $\\mu$ · blue: $q^\\star_a$ from $\\mu$ to $\\lambda_a$ · ink tick: $\\mu + q_a$ (live)"), wfBox),
    ui.h("div", { style: { marginTop: "14px" } }, ttl("(b) Composition of each neuron", "$x_{ia}^2/\\nu$: the share of neuron $i$'s variance on mode $a$"), compBox),
    ui.h("div", { style: { marginTop: "14px" } }, ttl("(c) Neurons per mode", "home mode $= \\arg\\max_a x_{ia}^2$; dots: the count if neurons were pure, $q_a^\\star/\\nu$"), cntBox));
  ui.renderMath(L.view);
  const comp = heatmap(compBox, { map: "seq", range: [0, 1], square: false, cellH: 14, size: 1000, zlabel: "$x_{ia}^2/\\nu$", xlabel: "neurons, grouped by home mode", ylabel: "mode a",
    tooltip: (a, c, v) => compTip(a, c, v) });
  const cnt = barChart(cntBox, { labels: ["1"], series: [{ name: "pure neurons (purity > 0.9)", values: [0], color: P.ink2 }, { name: "all neurons, by home mode", values: [0], color: P.mute2 }],
    overlay: { values: [0], name: "law if pure, q*ₐ/ν", color: P.accent }, ylabel: "neurons", xlabel: "mode a", height: 190 });
  let wfState = null;
  new ResizeObserver(() => { if (wfState) drawWF(wfState); }).observe(wfBox);

  // ---------------------------------------------------------------- side
  const tours = ui.chips({ parent: L.side, label: "Tours", items: TOURS_C, active: tour ? tour.id : null, onPick: t => applyTour(t) });
  const specS = ui.segmented({ label: "spectrum", options: [{ value: "lam8", label: "lam8 (N = 60)" }, { value: "pow16", label: "pow16 (N = 80)" }], value: p.spec, onChange: x => { p.spec = x; changed(); } });
  const nuS = ui.segmented({ label: "amplitude $\\nu = \\lVert x_i\\rVert^2$", options: NUS.map(v => ({ value: v, label: String(v) })), value: p.nu, onChange: x => { p.nu = x; changed(); } });
  const ingS = ui.select({ label: "added ingredient", options: M.INGREDIENTS.map(x => ({ value: x.id, label: x.label })), value: p.ing, onChange: x => { p.ing = x; syncB(); changed(); } });
  const bS = ui.slider({ label: "fusion scale $b$ (fusion only)", min: 0.003, max: 0.3, log: true, sig: 2, value: p.b, fmt: x => ui.fmt(x, 2),
    marks: [{ value: 0.01, label: "0.01" }, { value: 0.03, label: "0.03" }, { value: 0.1, label: "0.1" }], onChange: x => { p.b = x; changed(); } });
  const sS = ui.slider({ label: "random start (seed)", min: 0, max: 20, step: 1, value: p.seed, fmt: x => String(x), onChange: x => { p.seed = x; changed(); } });
  const syncB = () => bS.setDisabled(!(M.INGREDIENTS.find(x => x.id === p.ing) || {}).fusion);
  syncB();
  ui.controls({ parent: L.side, title: "Objective (eq. 5) and ingredient", primary: [specS, nuS, ingS, bS], advanced: [sS] });
  const stats = ui.stats({ parent: L.side, items: [
    { key: "dev", label: "mass-law deviation", fmt: x => ui.fmt(x, 2) },
    { key: "pure", label: "pure fraction", fmt: x => ui.fmtPct(x, 0) },
    { key: "cnt", label: "count error", fmt: x => ui.fmtFixed(x, 3) },
    { key: "exc", label: "objective excess", fmt: x => ui.fmt(x, 2) },
    { key: "mu", label: "water level $\\mu$", fmt: x => ui.fmtFixed(x, 3) },
    { key: "it", label: "stationarity", fmt: x => ui.fmt(x, 2) },
  ] });

  // ---------------------------------------------------------------- optimiser
  function restart() {
    st = M.makeSM(p); tickN = 0;
    const N = st.N; countMax = Math.max(10, Math.ceil(Math.max(...M.waterFill(st.lam, N * st.nu).q) / st.nu * 1.5));
    st.stationarity();
    const f = frameOf(); cur = f; show(f, true); return f;
  }
  function frameOf() {
    const conv = st.gradMap < 1e-10, out = st.it >= 20000;
    return { t: st.it, X: new Float64Array(st.X), gm: st.gradMap, done: conv || out, doneLabel: conv ? "stationary (< 10⁻¹⁰)" : "20,000 iterations" };
  }
  function tick(speed) {
    const t0 = performance.now(); let done = 0;
    const chunk = st.N > 60 || st.b > 0 ? 10 : 25;
    while (done < speed && st.it < 20000) { const n = Math.min(chunk, speed - done, 20000 - st.it); st.step(n); done += n; if (performance.now() - t0 > 16) break; }
    st.stationarity();
    const f = frameOf(); cur = f; tickN++;
    show(f, f.done || tickN % 3 === 0);
    return f;
  }
  function show(f, heavy) {
    const m = M.smMetrics({ ...st, X: f.X });
    drawWF({ m, lam: st.lam });
    compData = m;
    if (heavy) { drawComp(m); drawCounts(m); }
    stats.set("dev", m.qDev, `Σₐ|qₐ − q*ₐ|/(Nν); largest off-diagonal ${ui.fmt(m.off, 2)}`);
    stats.set("pure", m.pureFrac, "neurons with purity maxₐ x²ᵢₐ/ν > 0.9");
    stats.set("cnt", m.countDev, `Σₐ|nₐ − q*ₐ/ν|/N; vs the integer optimum ${ui.fmtFixed(m.countDevInt, 3)}${m.pureFrac < 0.9 ? "; neurons mixed, so counts are by argmax" : ""}`);
    stats.set("exc", Math.abs(m.excess) < 1e-12 ? 0 : m.excess, "½‖Σ − XᵀX‖² above its constrained minimum");
    stats.set("mu", m.mu, `budget Nν = ${ui.fmt(st.N * st.nu, 3)}; Σₐλₐ = ${ui.fmt(st.lam.reduce((a, b) => a + b, 0), 3)}`);
    stats.set("it", f.gm, `${f.gm < 1e-10 ? "below 10⁻¹⁰" : "projected-gradient residual"} at iteration ${ui.fmtInt(f.t)}`);
  }

  // ---------------------------------------------------------------- drawing
  function drawWF({ m, lam }) {
    wfState = { m, lam };
    const W = Math.floor(wfBox.clientWidth); if (!W) return;
    const Hh = 220, mg = { l: 42, r: 10, t: 10, b: 30 };
    wfBox.replaceChildren();
    const svg = svgEl("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}` }, wfBox);
    const Bn = lam.length, lo = Math.min(0, m.mu) - 0.1, hi = Math.max(...lam) * 1.06;
    const Y = scale("linear", lo, hi, Hh - mg.b, mg.t), bw = (W - mg.l - mg.r) / Bn;
    const txt = (x, y, s, a = {}) => { const t = svgEl("text", { x, y, "font-size": 11, fill: P.mute, "font-family": "Inter, system-ui, sans-serif", ...a }, svg); t.textContent = s; return t; };
    niceTicks(lo, hi, 5).forEach(v => { svgEl("line", { x1: mg.l, x2: W - mg.r, y1: Y(v), y2: Y(v), stroke: P.line }, svg); txt(mg.l - 6, Y(v) + 3.5, ui.fmt(v, 2), { "text-anchor": "end" }); });
    svgEl("line", { x1: mg.l, x2: W - mg.r, y1: Y(0), y2: Y(0), stroke: P.mute2 }, svg);
    const every = Math.max(1, Math.ceil(Bn / Math.floor((W - mg.l - mg.r) / 22)));
    lam.forEach((l, a) => {
      const x = mg.l + a * bw + bw * 0.14, w = bw * 0.72;
      svgEl("rect", { x, y: Y(Math.max(l, 0)), width: w, height: Math.abs(Y(0) - Y(l)), fill: P.wash, stroke: P.mute2, "stroke-width": 1 }, svg);
      if (l > m.mu) svgEl("rect", { x: x + w * 0.18, y: Y(l), width: w * 0.64, height: Y(m.mu) - Y(l), fill: P.seq[5], opacity: 0.85 }, svg);
      const yq = Y(m.mu + m.q[a]);
      svgEl("line", { x1: x - 2, x2: x + w + 2, y1: yq, y2: yq, stroke: P.fg, "stroke-width": 2.4 }, svg);
      if (a % every === 0) txt(mg.l + (a + 0.5) * bw, Hh - mg.b + 14, String(a + 1), { "text-anchor": "middle" });
      const hit = svgEl("rect", { x: mg.l + a * bw, y: mg.t, width: bw, height: Hh - mg.t - mg.b, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", ev => ui.showTip(ev.clientX, ev.clientY, ui.tipRows(`mode ${a + 1}`, [
        { name: "eigenvalue λₐ", value: ui.fmt(l, 3) }, { name: "predicted mass q*ₐ = (λₐ − μ)₊", value: ui.fmt(m.qPred[a], 3), color: P.seq[5], shape: "square" },
        { name: "live mass qₐ = (XᵀX)ₐₐ", value: ui.fmt(m.q[a], 3), color: P.fg }, { name: "count if pure, q*ₐ/ν", value: ui.fmt(m.law[a], 3) }])));
      hit.addEventListener("pointerleave", ui.hideTip);
    });
    svgEl("line", { x1: mg.l, x2: W - mg.r, y1: Y(m.mu), y2: Y(m.mu), stroke: P.accent, "stroke-width": 1.4, "stroke-dasharray": "5 4" }, svg);
    txt(W - mg.r - 2, Y(m.mu) - 5, `μ = ${ui.fmt(m.mu, 3)}`, { "text-anchor": "end", fill: P.accent, "font-weight": 600 });
    txt((mg.l + W - mg.r) / 2, Hh - 3, "mode a (eigenvalues in decreasing order)", { "text-anchor": "middle", fill: P.ink2, "font-weight": 560 });
  }
  let compData = null, compOrder = null;
  function drawComp(m) {
    const N = st.N, Bn = st.Bn;
    const order = [...Array(N).keys()].sort((i, j) => m.home[i] - m.home[j] || m.purity[j] - m.purity[i] || i - j);
    const vals = new Float32Array(Bn * N);
    for (let a = 0; a < Bn; a++) for (let c = 0; c < N; c++) vals[a * N + c] = m.comp[order[c] * Bn + a];
    const groups = []; let s = 0;
    while (s < N) { const h = m.home[order[s]]; let e = s; while (e < N && m.home[order[e]] === h) e++; groups.push({ start: s, end: e, label: String(h + 1) }); s = e; }
    compOrder = order;
    const colLabels = new Array(N).fill(""); groups.forEach(g => { if (g.end - g.start >= 2 || groups.length <= 12) colLabels[g.start] = g.label; });
    comp.update({ values: vals, rows: Bn, cols: N, rowLabels: st.lam.map((_, a) => String(a + 1)), colLabels, groups: { rows: null, cols: groups } });
  }
  function compTip(a, c) {
    if (!compData || !compOrder) return null;
    const i = compOrder[c], m = compData;
    return ui.tipRows(`neuron ${i}`, [{ name: `share on mode ${a + 1}, x²ᵢₐ/ν`, value: ui.fmtFixed(m.comp[i * st.Bn + a], 3) }, { name: "home mode", value: String(m.home[i] + 1) }, { name: "purity maxₐ x²ᵢₐ/ν", value: ui.fmtFixed(m.purity[i], 3) }]);
  }
  function drawCounts(m) {
    countMax = Math.max(countMax, ...m.counts.map(x => x + 2));
    cnt.update({ labels: st.lam.map((_, a) => String(a + 1)), series: [{ name: "pure neurons (purity > 0.9)", values: m.countsPure, color: P.ink2 }, { name: "all neurons, by home mode", values: m.counts, color: P.mute2 }],
      overlay: { values: m.law, name: "law if pure, q*ₐ/ν", color: P.accent }, yrange: [0, countMax] });
  }

  // ---------------------------------------------------------------- state plumbing
  function syncControls() { specS.set(p.spec); nuS.set(p.nu); ingS.set(p.ing); bS.set(p.b); sS.set(p.seed); syncB(); }
  function writeURL() {
    const o = { tc: tour ? tour.id : null };
    for (const k in KEYS_C) o[KEYS_C[k]] = tour || p[k] === DEF_C[k] ? null : p[k];
    ctx.setParams(o);
  }
  function changed() { if (tour) { tour = null; tours.moved(); } writeURL(); tp.reset(); tp.play(); }
  function applyTour(t) { tour = t; p = { ...t.p }; syncControls(); writeURL(); tp.reset(); tp.play(); }
  ctx.onParams(params => {
    if (![...params.keys()].some(k => k === "tc" || Object.values(KEYS_C).includes(k))) return;
    const r = readURL();
    if (r.tour) { tours.set(r.tour.id); applyTour(r.tour); return; }
    tour = null; tours.clear(); p = r.p; syncControls(); tp.reset(); tp.play();
  });
  tp.record(restart());
  if (tour) tp.play();
  return { tp };
}

// ---------------------------------------------------------------- the sweep: c1, c2, c3 and counts
export const ING_LABEL = {
  "none": "none", "fusion_b0.01": "fusion, b = 0.01", "fusion_b0.03": "fusion, b = 0.03", "fusion_b0.1": "fusion, b = 0.1",
  "nonneg_eig": "non-negative, eigenbasis", "nonneg_rotated": "non-negative, rotated", "nonneg_eig+fusion_b0.03": "non-negative + fusion 0.03",
};
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const sd = xs => { const m = mean(xs); return Math.sqrt(mean(xs.map(x => (x - m) ** 2))); };
const pm = (xs, f) => { const s = sd(xs); return `${f(mean(xs))}<span class="mute"> ± ${f(s)}</span>`; };
export function smSettings(data) {
  const keys = []; for (const r of data.c) { const k = `${r.spec}|${r.nu}`; if (!keys.includes(k)) keys.push(k); }
  return keys.map(k => { const [spec, nu] = k.split("|"); return { spec, nu: +nu, key: k }; });
}
export function smRows(data, spec, nu, ing) { return data.c.filter(r => r.spec === spec && r.nu === nu && (ing == null || r.ing === ing)); }

export function smTables(data, el) {
  const settings = smSettings(data);
  const ings = [...new Set(data.c.map(r => r.ing))];
  let sel = settings.findIndex(s => s.spec === "lam8" && s.nu === 0.1); if (sel < 0) sel = 0;
  let ingSel = "nonneg_eig";
  const f3 = x => ui.fmt(x, 2), ffix = x => ui.fmtFixed(x, 2), fsci = x => (x === 0 ? "0" : ui.fmt(x, 2));

  // c1 (independent of the selector)
  const c1 = settings.map(s => {
    const rs = smRows(data, s.spec, s.nu, "none");
    return `<tr><td>${s.spec}</td><td class="num">${s.nu}</td><td class="num">${ui.fmt(rs[0].mu, 3)}</td><td class="num">${fsci(Math.max(...rs.map(r => r.q_dev)))}</td><td class="num">${fsci(Math.max(...rs.map(r => r.offdiag)))}</td><td class="num">${fsci(Math.max(...rs.map(r => Math.abs(r.obj_main_excess))))}</td><td class="num">${fsci(Math.max(...rs.map(r => r.grad_map)))}</td><td class="num">${ui.fmtPct(mean(rs.map(r => r.pure_frac)), 0)}</td></tr>`;
  }).join("");
  const c1html = `<h4 style="margin:4px 0 6px">(c1) The mass law, no ingredient</h4><div class="table-scroll"><table class="data"><thead><tr><th>spectrum</th><th class="num">$\\nu$</th><th class="num">$\\mu$</th><th class="num">max mass-law dev.</th><th class="num">max off-diag.</th><th class="num">max obj. excess</th><th class="num">max stationarity</th><th class="num">pure fraction</th></tr></thead><tbody>${c1}</tbody></table></div>`;

  const seg = ui.segmented({ label: "setting for c2, c3 and the counts", options: settings.map((s, i) => ({ value: i, label: `${s.spec}, ν = ${s.nu}` })), value: sel, onChange: i => { sel = i; draw(); } });
  const c2 = ui.h("div"), c3 = ui.h("div"), cntHead = ui.h("div"), cntBox = ui.h("div");
  const ingSelect = ui.select({ label: "ingredient", options: ings.map(i => ({ value: i, label: ING_LABEL[i] || i })), value: ingSel, onChange: v => { ingSel = v; drawCounts(); } });
  el.replaceChildren(ui.h("div", { html: c1html }), ui.h("div", { style: { marginTop: "18px" } }, seg.el), c2, c3,
    ui.h("div", { style: { marginTop: "16px" } }, ui.h("h4", { style: { margin: "4px 0 6px" }, text: "Counts per mode in the sweep" }), ingSelect.el, cntHead, cntBox));
  ui.renderMath(el);
  const cntChart = lineChart(cntBox, { series: [{ name: "law", y: [0] }], height: 220, xlabel: "mode a", ylabel: "neurons", xname: "mode" });

  function draw() {
    const s = settings[sel];
    const rowsC2 = ings.map(ing => {
      const rs = smRows(data, s.spec, s.nu, ing); if (!rs.length) return "";
      const pf = mean(rs.map(r => r.pure_frac)), grey = pf < 0.9 ? ' style="color:var(--mute)"' : "";
      return `<tr><td>${ING_LABEL[ing] || ing}</td><td class="num">${pm(rs.map(r => r.pure_frac), x => ui.fmtFixed(x, 2))}</td><td class="num"${grey}>${pm(rs.map(r => r.count_dev), x => ui.fmtFixed(x, 2))}</td><td class="num"${grey}>${pm(rs.map(r => r.count_dev_integer), x => ui.fmtFixed(x, 2))}</td></tr>`;
    }).join("");
    c2.innerHTML = `<h4 style="margin:14px 0 6px">(c2) Specialisation: ${s.spec}, $\\nu = ${s.nu}$, $N = ${smRows(data, s.spec, s.nu)[0].N}$</h4><div class="table-scroll"><table class="data"><thead><tr><th>ingredient</th><th class="num">pure fraction</th><th class="num">count error vs law</th><th class="num">vs integer optimum</th></tr></thead><tbody>${rowsC2}</tbody></table></div>`;
    const rowsC3 = ings.map(ing => {
      const rs = smRows(data, s.spec, s.nu, ing); if (!rs.length) return "";
      return `<tr><td>${ING_LABEL[ing] || ing}</td><td class="num">${pm(rs.map(r => r.q_dev), x => fsci(x))}</td><td class="num">${pm(rs.map(r => r.obj_main_excess), fsci)}</td><td class="num">${fsci(Math.max(...rs.map(r => r.offdiag)))}</td><td class="num">${fsci(Math.max(...rs.map(r => r.grad_map)))}</td></tr>`;
    }).join("");
    c3.innerHTML = `<h4 style="margin:14px 0 6px">(c3) Cost to the law: ${s.spec}, $\\nu = ${s.nu}$</h4><div class="table-scroll"><table class="data"><thead><tr><th>ingredient</th><th class="num">mass-law deviation</th><th class="num">objective excess</th><th class="num">max off-diag.</th><th class="num">max stationarity</th></tr></thead><tbody>${rowsC3}</tbody></table></div>`;
    ui.renderMath(c2); ui.renderMath(c3);
    drawCounts();
  }
  function drawCounts() {
    const s = settings[sel], rs = smRows(data, s.spec, s.nu, ingSel);
    if (!rs.length) return;
    const x = rs[0].n_from_law.map((_, a) => a + 1);
    const series = [{ name: "law q*ₐ/ν", x, y: rs[0].n_from_law, color: P.accent, width: 2 }, { name: "integer optimum (pure)", x, y: rs[0].n_integer_fill, color: P.ink2, dash: true, width: 1.5 }];
    rs.forEach((r, k) => series.push({ name: `start ${r.seed}: nₐ (purity ${ui.fmtPct(r.pure_frac, 0)} pure)`, x, y: r.counts, points: true, r: 3.6, color: catColor(k) }));
    cntChart.update({ series });
    cntHead.innerHTML = `<p class="u-help" style="margin:4px 0">${ING_LABEL[ingSel] || ingSel}, ${s.spec}, $\\nu = ${s.nu}$: mean pure fraction ${ui.fmtPct(mean(rs.map(r => r.pure_frac)), 0)}, count error ${ui.fmtFixed(mean(rs.map(r => r.count_dev)), 3)} (mean over starts).</p>`;
    ui.renderMath(cntHead);
  }
  draw();
}
