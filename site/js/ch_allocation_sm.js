// Chapter "allocation", section 3: the added similarity-matching objective. A live projected-gradient optimiser
// (water-filling against the live masses; neurons per mode against the count the law would give) and the sweep
// as small multiples (pure fraction against count error, one panel per budget).
import * as ui from "./ui.js";
import { barChart, svgEl, scale, niceTicks } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as M from "./ch_allocation_model.js";

// Live options: the ingredient (the fusion term's b is folded into the choice) and the budget nu.
const OPTS = [
  { id: "none", label: "none", ing: "none", b: 0.03 },
  { id: "fusion-weak", label: "fusion term, weak ($b = 0.03$)", ing: "fusion", b: 0.03 },
  { id: "fusion-strong", label: "fusion term, strong ($b = 0.1$)", ing: "fusion", b: 0.1 },
  { id: "nonneg_eig", label: "non-negative, eigenbasis", ing: "nonneg_eig", b: 0.03 },
  { id: "nonneg_rotated", label: "non-negative, rotated basis", ing: "nonneg_rotated", b: 0.03 },
];
const NUS = [0.05, 0.1, 0.2];
const DEF_C = { opt: "none", nu: 0.1 };
const SPEC = "lam8";

export function smFigure(ctx, slot) {
  const fig = slot.closest("figure");
  const readURL = () => {
    const o = ctx.param("ci", DEF_C.opt), nu = +ctx.param("cn", DEF_C.nu);
    return { opt: OPTS.some(x => x.id === o) ? o : DEF_C.opt, nu: NUS.includes(nu) ? nu : DEF_C.nu };
  };
  let p = readURL();
  let st = null, cur = null, tickN = 0, countMax = 10;

  const L = ui.simLayout({ parent: slot });
  const tp = ui.transport({
    parent: L.view, watch: fig, autoplay: true, maxHistory: 200,
    speeds: [{ value: 40, label: "×1" }, { value: 400, label: "×10" }, { value: 4000, label: "×100" }], speed: 400,
    timeFmt: t => `iteration ${ui.fmtInt(t)}`,
    onTick: s => tick(s), onSeek: f => { cur = f; show(f, true); }, onReset: () => restart(),
  });
  const ttl = (t, s) => ui.h("div", {}, ui.h("div", { class: "chart-title", html: t }), ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" }, html: s }));
  const wfBox = ui.h("div", { class: "chart" }), cntBox = ui.h("div");
  L.view.append(
    ui.h("div", { style: { marginTop: "12px" } }, ttl("(a) Variance per mode: the law and the optimiser", "grey bar: eigenvalue $\\lambda_a$ · dashed: water level $\\mu$ · blue: the law's mass, from $\\mu$ up to $\\lambda_a$ · black tick: $\\mu$ + the optimiser's mass $q_a$"), wfBox),
    ui.h("div", { style: { marginTop: "16px" } }, ttl("(b) Neurons per mode", "each neuron counted on the mode holding most of its variance; dots: the count if neurons were pure, $q_a/\\nu$"), cntBox));
  ui.renderMath(L.view);
  const cnt = barChart(cntBox, { labels: ["1"], series: [{ name: "pure neurons (> 90% on one mode)", values: [0], color: P.ink2 }, { name: "all neurons", values: [0], color: P.mute2 }],
    overlay: { values: [0], name: "law if neurons were pure", color: P.accent }, ylabel: "neurons", xlabel: "mode a (largest eigenvalue first)", height: 200 });
  let wfState = null;
  new ResizeObserver(() => { if (wfState) drawWF(wfState); }).observe(wfBox);

  // ---------------------------------------------------------------- side
  const ingS = ui.select({ label: "added ingredient", options: OPTS.map(x => ({ value: x.id, label: x.label.replace(/\$b = ([0-9.]+)\$/, "b = $1") })), value: p.opt, onChange: x => { p.opt = x; changed(); } });
  const nuS = ui.segmented({ label: "variance per neuron $\\nu$", options: NUS.map(v => ({ value: v, label: String(v) })), value: p.nu, onChange: x => { p.nu = x; changed(); } });
  ui.controls({ parent: L.side, title: "Objective and ingredient", primary: [ingS, nuS] });
  const stats = ui.stats({ parent: L.side, items: [
    { key: "dev", label: "masses off the law", fmt: x => ui.fmt(x, 2) },
    { key: "pure", label: "pure neurons", fmt: x => ui.fmtPct(x, 0) },
    { key: "cnt", label: "count error", fmt: x => ui.fmtFixed(x, 2) },
  ] });

  // ---------------------------------------------------------------- optimiser
  function restart() {
    const o = OPTS.find(x => x.id === p.opt);
    st = M.makeSM({ spec: SPEC, nu: p.nu, ing: o.ing, b: o.b, seed: 0 }); tickN = 0;
    countMax = Math.max(10, Math.ceil(Math.max(...M.waterFill(st.lam, st.N * st.nu).q) / st.nu * 1.5));
    st.stationarity();
    const f = frameOf(); cur = f; show(f, true); return f;
  }
  function frameOf() {
    const conv = st.gradMap < 1e-10, out = st.it >= 20000;
    return { t: st.it, X: new Float64Array(st.X), gm: st.gradMap, done: conv || out, doneLabel: conv ? "stationary" : "20,000 iterations" };
  }
  function tick(speed) {
    const t0 = performance.now(); let done = 0;
    const chunk = st.b > 0 ? 10 : 25;
    while (done < speed && st.it < 20000) { const n = Math.min(chunk, speed - done, 20000 - st.it); st.step(n); done += n; if (performance.now() - t0 > 16) break; }
    st.stationarity();
    const f = frameOf(); cur = f; tickN++;
    show(f, f.done || tickN % 3 === 0);
    return f;
  }
  function show(f, heavy) {
    const m = M.smMetrics({ ...st, X: f.X });
    drawWF({ m, lam: st.lam });
    if (heavy) drawCounts(m);
    stats.set("dev", m.qDev, "Σₐ|qₐ − q*ₐ|/(Nν), q* the law");
    stats.set("pure", m.pureFrac, "neurons with more than 90% of their variance on one mode");
    stats.set("cnt", m.countDev, `Σₐ|nₐ − q*ₐ/ν|/N${m.pureFrac < 0.5 ? "; most neurons are mixed" : ""}`);
  }

  // ---------------------------------------------------------------- drawing
  function drawWF({ m, lam }) {
    wfState = { m, lam };
    const W = Math.floor(wfBox.clientWidth); if (!W) return;
    const Hh = 210, mg = { l: 42, r: 10, t: 10, b: 30 };
    wfBox.replaceChildren();
    const svg = svgEl("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}` }, wfBox);
    const Bn = lam.length, lo = Math.min(0, m.mu) - 0.1, hi = Math.max(...lam, ...m.q.map(q => m.mu + q)) * 1.06;   // the axis covers every tick
    const Y = scale("linear", lo, hi, Hh - mg.b, mg.t), bw = (W - mg.l - mg.r) / Bn;
    const txt = (x, y, s, a = {}) => { const t = svgEl("text", { x, y, "font-size": 11, fill: P.mute, "font-family": "Inter, system-ui, sans-serif", ...a }, svg); t.textContent = s; return t; };
    niceTicks(lo, hi, 5).forEach(v => { svgEl("line", { x1: mg.l, x2: W - mg.r, y1: Y(v), y2: Y(v), stroke: P.line }, svg); txt(mg.l - 6, Y(v) + 3.5, ui.fmt(v, 2), { "text-anchor": "end" }); });
    svgEl("line", { x1: mg.l, x2: W - mg.r, y1: Y(0), y2: Y(0), stroke: P.mute2 }, svg);
    lam.forEach((l, a) => {
      const x = mg.l + a * bw + bw * 0.14, w = bw * 0.72;
      svgEl("rect", { x, y: Y(Math.max(l, 0)), width: w, height: Math.abs(Y(0) - Y(l)), fill: P.wash, stroke: P.mute2, "stroke-width": 1 }, svg);
      if (l > m.mu) svgEl("rect", { x: x + w * 0.18, y: Y(l), width: w * 0.64, height: Y(m.mu) - Y(l), fill: P.seq[5], opacity: 0.85 }, svg);
      const yq = Y(m.mu + m.q[a]);
      svgEl("line", { x1: x - 2, x2: x + w + 2, y1: yq, y2: yq, stroke: P.fg, "stroke-width": 2.4 }, svg);
      txt(mg.l + (a + 0.5) * bw, Hh - mg.b + 14, String(a + 1), { "text-anchor": "middle" });
      const hit = svgEl("rect", { x: mg.l + a * bw, y: mg.t, width: bw, height: Hh - mg.t - mg.b, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", ev => ui.showTip(ev.clientX, ev.clientY, ui.tipRows(`mode ${a + 1}`, [
        { name: "eigenvalue λₐ", value: ui.fmt(l, 3) }, { name: "law: (λₐ − μ)₊", value: ui.fmt(m.qPred[a], 3), color: P.seq[5], shape: "square" },
        { name: "optimiser: qₐ", value: ui.fmt(m.q[a], 3), color: P.fg }])));
      hit.addEventListener("pointerleave", ui.hideTip);
    });
    svgEl("line", { x1: mg.l, x2: W - mg.r, y1: Y(m.mu), y2: Y(m.mu), stroke: P.accent, "stroke-width": 1.4, "stroke-dasharray": "5 4" }, svg);
    txt(W - mg.r - 2, Y(m.mu) - 5, `μ = ${ui.fmt(m.mu, 3)}`, { "text-anchor": "end", fill: P.accent, "font-weight": 600 });
    txt((mg.l + W - mg.r) / 2, Hh - 3, "mode a (largest eigenvalue first)", { "text-anchor": "middle", fill: P.ink2, "font-weight": 560 });
  }
  function drawCounts(m) {
    countMax = Math.max(countMax, ...m.counts.map(x => x + 2));
    cnt.update({ labels: st.lam.map((_, a) => String(a + 1)), series: [{ name: "pure neurons (> 90% on one mode)", values: m.countsPure, color: P.ink2 }, { name: "all neurons", values: m.counts, color: P.mute2 }],
      overlay: { values: m.law, name: "law if neurons were pure", color: P.accent }, yrange: [0, countMax] });
  }

  // ---------------------------------------------------------------- state plumbing
  function writeURL() { ctx.setParams({ ci: p.opt === DEF_C.opt ? null : p.opt, cn: p.nu === DEF_C.nu ? null : p.nu }); }
  function changed() { writeURL(); tp.reset(); tp.play(); }
  ctx.onParams(params => {
    if (![...params.keys()].some(k => k === "ci" || k === "cn")) return;
    p = readURL(); ingS.set(p.opt); nuS.set(p.nu); tp.reset(); tp.play();
  });
  tp.record(restart());
  return { tp };
}

// ---------------------------------------------------------------- the sweep: small multiples, one per budget
// x: pure fraction, y: count error (mean over the four random starts); filled marker: masses on the law.
const SWEEP_ING = [
  { ing: "none", label: "none" },
  { ing: "fusion_b0.03", label: "fusion, weak (b = 0.03)" },
  { ing: "fusion_b0.1", label: "fusion, strong (b = 0.1)" },
  { ing: "nonneg_eig", label: "non-negative, eigenbasis" },
  { ing: "nonneg_rotated", label: "non-negative, rotated" },
];
export const LAW_TOL = 1e-6;
export function smRows(data, spec, nu, ing) { return data.c.filter(r => r.spec === spec && r.nu === nu && (ing == null || r.ing === ing)); }
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;

export function smSweep(data, el) {
  const nus = [...new Set(data.c.filter(r => r.spec === SPEC).map(r => r.nu))].sort((a, b) => a - b);
  const pts = nus.map(nu => SWEEP_ING.map((s, k) => {
    const rs = smRows(data, SPEC, nu, s.ing);
    return rs.length ? { ...s, k, nu, pure: mean(rs.map(r => r.pure_frac)), cnt: mean(rs.map(r => r.count_dev)), qdev: mean(rs.map(r => r.q_dev)), mu: rs[0].mu, n: rs.length } : null;
  }).filter(Boolean));
  const ymax = Math.max(0.5, ...pts.flat().map(q => q.cnt)) * 1.08;
  const row = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "8px 14px" } });
  const legendHost = ui.h("div");
  el.replaceChildren(row, legendHost);
  const panels = nus.map((nu, j) => { const box = ui.h("div", { style: { minWidth: "0" } }); row.append(box); return { box, nu, pts: pts[j] }; });
  function draw() {
    for (const pn of panels) {
      const W = Math.floor(pn.box.clientWidth); if (!W) continue;
      const Hh = 230, mg = { l: 40, r: 10, t: 24, b: 34 };
      pn.box.replaceChildren();
      const svg = svgEl("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}` }, pn.box);
      const X = scale("linear", -0.04, 1.04, mg.l, W - mg.r), Y = scale("linear", 0, ymax, Hh - mg.b, mg.t);
      const txt = (x, y, s, a = {}) => { const t = svgEl("text", { x, y, "font-size": 11, fill: P.mute, "font-family": "Inter, system-ui, sans-serif", ...a }, svg); t.textContent = s; return t; };
      [0, 0.5, 1].forEach(v => { svgEl("line", { x1: X(v), x2: X(v), y1: mg.t, y2: Hh - mg.b, stroke: P.line }, svg); txt(X(v), Hh - mg.b + 14, v === 0 ? "0" : v === 1 ? "100%" : "50%", { "text-anchor": "middle" }); });
      niceTicks(0, ymax, 4).forEach(v => { svgEl("line", { x1: mg.l, x2: W - mg.r, y1: Y(v), y2: Y(v), stroke: P.line }, svg); txt(mg.l - 6, Y(v) + 3.5, ui.fmt(v, 2), { "text-anchor": "end" }); });
      txt((mg.l + W - mg.r) / 2, Hh - 3, "pure neurons", { "text-anchor": "middle", fill: P.ink2, "font-weight": 560 });
      txt(0, 0, "count error", { "text-anchor": "middle", fill: P.ink2, "font-weight": 560, transform: `translate(11,${(mg.t + Hh - mg.b) / 2}) rotate(-90)` });
      const ttl = txt(mg.l, 14, "", { fill: P.fg, "font-weight": 600, "font-size": 12 });
      ttl.textContent = "";
      svgEl("tspan", {}, ttl).textContent = "variance per neuron ";
      svgEl("tspan", { "font-family": "'Source Serif 4', Georgia, serif", "font-style": "italic", "font-size": 13.5 }, ttl).textContent = "ν";
      svgEl("tspan", {}, ttl).textContent = ` = ${pn.nu}`;
      // the goal: every neuron pure, counts equal to the law
      const gx = X(1), gy = Y(0);
      svgEl("path", { d: `M${gx - 7},${gy - 7}L${gx + 7},${gy + 7}M${gx - 7},${gy + 7}L${gx + 7},${gy - 7}`, stroke: P.fg, "stroke-width": 2 }, svg);
      txt(gx - 9, gy - 8, "goal", { "text-anchor": "end", fill: P.fg, "font-weight": 600 });
      for (const q of pn.pts) {
        const on = q.qdev < LAW_TOL, col = catColor(q.k);
        const c = svgEl("circle", { cx: X(q.pure), cy: Y(q.cnt), r: 6.5, fill: on ? col : P.surface, stroke: col, "stroke-width": 2.4 }, svg);
        c.addEventListener("pointermove", ev => ui.showTip(ev.clientX, ev.clientY, ui.tipRows(`${q.label}, ν = ${q.nu}`, [
          { name: "pure neurons (mean)", value: ui.fmtPct(q.pure, 0) }, { name: "count error (mean)", value: ui.fmtFixed(q.cnt, 3) },
          { name: "masses off the law (mean)", value: ui.fmt(q.qdev, 2) }], `mean over ${q.n} random starts`)));
        c.addEventListener("pointerleave", ui.hideTip);
      }
    }
  }
  ui.legend({ parent: legendHost, items: [...SWEEP_ING.map((s, k) => ({ label: s.label, color: catColor(k), shape: "dot" })),
    { label: "filled: masses on the law · hollow: off it", color: P.mute, shape: "ring" },
    { label: "× goal: all neurons pure, counts on the law", color: P.fg, shape: "line" }] });
  new ResizeObserver(() => draw()).observe(row);
  draw();
  return pts;
}
