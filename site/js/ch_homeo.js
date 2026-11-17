// Chapter "homeo" (Homeostasis): what the homeostatic knobs do to the clustering and to the number of neurons each
// part receives. A live explorer (real engine, ch_homeo_model.js) and precomputed maps (ch_homeo_maps.js).
import * as ui from "./ui.js";
import { heatmap, barMultiples, imageGrid, svgEl, scale, niceTicks, logTicks, hidpi } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as M from "./ch_homeo_model.js";
import { initMaps } from "./ch_homeo_maps.js";

const f2 = x => ui.fmtFixed(x, 2), f3 = x => ui.fmt(x, 3);
const SUBS = "₀₁₂₃₄₅₆₇₈₉", sub = n => String(n).replace(/\d/g, d => SUBS[d]);

export async function init(panel, ctx) {
  const host = ctx.slot("sim");
  host.append(ui.h("p", { class: "mute", text: "loading the covariances…" }));
  const data = await fetch("data/homeo/covs.json").then(r => r.json());
  host.replaceChildren();
  const live = initLive(ctx, data);
  await initMaps(ctx, data, { load: (params, why) => live.load(params, why) });
}

// ======================================================================================== live explorer
function initLive(ctx, data) {
  let { p, tour } = readParams(ctx, data);
  let xMode = ctx.param("x", "lam") === "c" ? "c" : "lam";
  const sim = M.createSim(data, p);
  let tourActive = tour, prevLabels = null, cur = null, tickN = 0, partsSig = "";
  let fpsT = performance.now(), fpsN = 0, fps = 0;
  const fig = ctx.slot("sim").closest("figure");

  // ------------------------------------------------------------ layout
  const L = ui.simLayout({ parent: ctx.slot("sim"), side: 330 });
  const box = (parent, title, subText) => {
    const b = ui.h("div");
    parent.append(ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", html: title }),
      subText ? ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" }, html: subText }) : null, b));
    return b;
  };
  const modeLine = ui.h("div", { class: "u-help", style: { margin: "0 0 4px" } });
  L.view.append(modeLine);
  const allocBox = box(L.view, "(a) Neurons per cluster against the energy of its part",
    "log–log; one dot per cluster; hollow circles under the axis: stable parts that got no neurons");
  const xSeg = ui.segmented({ options: [{ value: "lam", label: "energy $\\lambda_{\\max}(\\Sigma_{SS})$" }, { value: "c", label: "supra-threshold $c_k$" }], value: xMode,
    onChange: x => { xMode = x; ctx.setParams({ x: x === "lam" ? null : x }); if (cur) alloc.update(cur, xMode); } });
  xSeg.el.style.margin = "4px 0 0";
  allocBox.before(xSeg.el);
  const alloc = allocPlot(allocBox, { height: 330, tooltip: true });
  const tp = ui.transport({
    parent: L.view, watch: fig, speeds: [1, 4, 16, 64].map(x => ({ value: x, label: `×${x}` })), speed: 16, autoplay: true, maxHistory: 300,
    timeFmt: t => `t = ${ui.fmtFixed(t, 1)}`,
    onTick: speed => tick(speed),
    onSeek: f => { cur = f; show(f, true); },
    onReset: () => restart(),
  });
  const row = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(max(260px, 40%), 1fr))", gap: "14px 26px", marginTop: "16px" } });
  L.view.append(row);
  const stripBox = box(row, "(b) Parts that exist at each $\\theta$", "step line: number of stable parts; dot: clusters populated now; click to set $\\theta$");
  const riverBox = box(row, "(c) Neurons per cluster over time", "stacked; colour = cluster (by size); grey: small clusters and dead neurons");
  const partsBox = box(L.view, "(d) The clusters' parts", "member mean of $h_i$ over the inputs, one panel per cluster (largest first), then stable parts with no neurons");
  partsBox.parentElement.style.marginTop = "16px";
  const strip = thetaStrip(stripBox, { height: 190, onPick: th => { thetaS.set(th); p.theta = th; liveChange(); } });
  const river = riverChart(riverBox, { height: 190 });
  ui.renderMath(L.view);

  // ------------------------------------------------------------ tours
  const GROUPS = [["f", "Tours: the filter and the floor"], ["m", "More tours"], ["n", "With private noise"]];
  const chipSets = GROUPS.map(([g, label]) => ui.chips({ parent: L.side, label, items: M.TOURS.filter(t => t.group === g), active: tour && tour.group === g ? tour.id : null,
    onPick: t => { chipSets.forEach(c => { if (c !== chipSets[GROUPS.findIndex(x => x[0] === t.group)]) c.clear(); }); applyTour(t); } }));

  // ------------------------------------------------------------ controls
  const covIds = Object.keys(data.covs);
  const covS = ui.select({ label: "input covariance $\\Sigma_g$", options: covIds.map(id => ({ value: id, label: (data.covs[id].real ? "real: " : "synthetic: ") + data.covs[id].label })), value: p.cov,
    onChange: id => { p.cov = id; const c = M.covOf(data, id); p.theta = clampTheta(p.theta, c); makeThetaSlider(); resetChange(); } });
  const sigmaBox = ui.h("div", { style: { display: "grid", gap: "6px" } });
  let thetaS = null;
  const thetaHost = ui.h("div");
  function thetaRange(c) { return [-0.2, Math.min(1.6, Math.ceil((c.stats.var_max + 0.05) * 100) / 100)]; }
  function clampTheta(th, c) { const [lo, hi] = thetaRange(c); return Math.max(lo, Math.min(hi, Math.round(th * 100) / 100)); }
  function makeThetaSlider() {
    const c = M.covOf(data, p.cov), [lo, hi] = thetaRange(c);
    const s = ui.slider({ label: "pruning threshold $\\theta = -w_0/\\eta$", min: lo, max: hi, step: 0.01, value: p.theta, fmt: f2,
      marks: thetaMarks(c, p.gamma, lo, hi),
      help: "Sets which parts exist (the step line in (b)). Ticks: $\\kappa_0$ and $\\gamma\\kappa_0$; the smallest and largest covariance (cov) and variance (var) of $\\Sigma_g$.",
      onInput: x => { p.theta = x; liveChange(); } });
    if (thetaS) thetaS.el.replaceWith(s.el); else thetaHost.append(s.el);
    thetaS = s;
  }
  function refreshMarks() {
    const c = M.covOf(data, p.cov), [lo, hi] = thetaRange(c), mk = thetaS.el.querySelector(".u-marks");
    if (!mk) return;
    mk.replaceChildren(...thetaMarks(c, p.gamma, lo, hi).map(m => {
      const pct = (m.value - lo) / (hi - lo) * 100;
      return ui.h("span", { class: "u-mark", style: { left: `calc(${pct}% + ${(0.5 - pct / 100) * 14}px)` }, title: m.title || "" }, ui.h("i"), m.label ? ui.h("b", { html: m.label }) : null);
    }));
  }
  makeThetaSlider();
  const gammaS = ui.slider({ label: "lateral gain $\\gamma = \\eta_{II}/\\eta$", min: 0, max: 3, step: 0.05, value: p.gamma, fmt: f2,
    help: "Not part of $\\rho$: the lateral learning rate. The population tends to bind into one blob when $\\theta < \\gamma\\kappa_0$ (a heuristic crossover, accurate to 0.03–0.08 where tested).",
    onInput: x => { p.gamma = x; refreshMarks(); liveChange(); } });
  const omegaS = ui.slider({ label: "floor leak $\\omega = N w_{\\min}\\lambda/\\eta$", min: 1e-4, max: 1, log: true, sig: 2, value: p.omega, fmt: x => ui.fmt(x, 2),
    help: "", onChange: x => { p.omega = x; liveChange(); } });
  const omegaHelp = omegaS.el.querySelector(".u-help") || ui.h("div", { class: "u-help" });
  if (!omegaHelp.parentNode) omegaS.el.append(omegaHelp);
  const initS = ui.select({ label: "initial input weights $W_{IB}(0)$ (mean 0.2)", options: M.INITS.map(x => ({ value: x.id, label: x.label })), value: p.init,
    help: "The spread of the starting weights, not their scale, sets how steeply allocation follows energy.", onChange: x => { p.init = x; resetChange(); } });
  const nS = ui.slider({ label: "interior neurons $N$", min: 60, max: 250, step: 10, value: p.n, fmt: String, onChange: x => { p.n = x; resetChange(); } });
  const lamS = ui.segmented({ label: "decay $\\lambda$ (with $\\eta = 1$)", options: [0.25, 0.5, 1, 2, 4].map(x => ({ value: x, label: String(x) })), value: p.lam,
    help: "Weights, $w_{\\min}$, the initial weights and the time step all scale with $1/\\lambda$; watch the max weight and the clock change and nothing else.",
    onChange: x => { p.lam = x; resetChange(); } });
  const seedS = ui.slider({ label: "seed (initial weights, samples, noise)", min: 1, max: 20, step: 1, value: p.seed, fmt: String, onChange: x => { p.seed = x; resetChange(); } });
  const iiS = ui.slider({ label: "initial lateral weights $\\sim U(0, s_{II})$", min: 0, max: 1, step: 0.01, value: p.ii, fmt: f2,
    help: "$s_{II} = 0$: lateral synapses start at the floor (the sweeps' default).", onChange: x => { p.ii = x; resetChange(); } });
  const batchS = ui.segmented({ label: "covariance seen by plasticity", options: M.BATCHES.map(x => ({ value: x, label: x ? `batch ${x}` : "exact" })), value: p.batch, stack: true,
    help: "Minibatch: every neuron sees the same samples (shared noise).", onChange: x => { p.batch = x; liveChange(); } });
  const noiseS = ui.select({ label: "private noise (independent per synapse)", options: M.NOISES.map(x => ({ value: x.id, label: x.label })), value: p.noise,
    help: "Input noise: each synapse estimates its covariance from its own samples, $1/\\sigma^2$ per unit time. Additive: $w \\mathrel{+}= \\sigma\\sqrt{dt}\\,\\xi$. With noise, a neuron counts as in a part when its embedding has cosine $\\ge 0.9$ with it, otherwise as in transit.",
    onChange: x => { p.noise = x; liveChange(); } });

  ui.controls({ parent: L.side, title: "Inputs", primary: [covS, sigmaBox] });
  ui.controls({ parent: L.side, title: "Knobs", primary: [thetaHost, gammaS, omegaS, initS],
    advanced: [noiseS, batchS, nS, lamS, seedS, iiS], advancedLabel: "More controls ($N$, $\\lambda$, seed, noise)" });

  // ------------------------------------------------------------ readouts
  const stats = ui.stats({ parent: L.side, items: [
    { key: "k", label: "clusters $K$", fmt: String, help: `Clusters with at least max(2, 2% N) members (single linkage in L1, cut at the largest gap), dead neurons excluded.` },
    { key: "ns", label: "parts that exist", fmt: String, help: "Stable parts at this θ, from data/homeo/covs.json." },
    { key: "beta", label: "allocation $\\beta$", fmt: x => Number.isFinite(x) ? ui.fmtFixed(x, 2) : "–", help: "Least-squares slope of log n against log energy over the clusters in (a)." },
    { key: "dead", label: "dead neurons", fmt: String },
    { key: "t", label: "time $t$", fmt: x => ui.fmtFixed(x, 1) },
    { key: "r", label: "stationarity $r$", fmt: x => ui.fmt(x, 2), help: "Projected residual max |dw/dt| (floor synapses pushed down count 0), covariance units." },
  ] });
  const regimeBox = ui.h("div", { style: { borderLeft: `3px solid ${P.line}`, padding: "4px 0 4px 10px", margin: "10px 0 6px", fontSize: "13px", lineHeight: "1.45" } });
  L.side.append(regimeBox);

  // ------------------------------------------------------------ the Sigma panel
  let sigmaHM = null, imageBox = null;
  function renderSigma() {
    const c = M.covOf(data, p.cov), B = c.B, S = c._S;
    let m = 0; for (const x of S) m = Math.max(m, Math.abs(x));
    const tag = ui.h("span", { class: c.real ? "synthetic-tag" : "synthetic-tag", text: c.real ? "real data: MNIST" : "synthetic Σ" });
    if (c.real) tag.style.filter = "hue-rotate(180deg)";
    const hmHost = ui.h("div");
    const st = c.stats;
    const desc = ui.h("div", { class: "u-help", html: `${c.desc}. $\\kappa_0 = \\mathbf 1^\\top\\Sigma_g\\mathbf 1/B^2 = ${f3(st.kappa0)}$; covariances ${f2(st.offdiag_min)} to ${f2(st.offdiag_max)}; variances ${f2(st.var_min)} to ${f2(st.var_max)}.` });
    sigmaBox.replaceChildren(ui.h("div", { style: { display: "flex", gap: "8px", alignItems: "center" } }, tag), hmHost, desc);
    const groups = c.groups ? groupRuns(c.groups) : null;
    sigmaHM = heatmap(hmHost, { values: Float32Array.from(S), rows: B, cols: B, map: "div", range: [-m, m], size: 200, zlabel: "$\\Sigma_g$",
      groups: groups ? { rows: groups } : null, xlabel: c.image ? "pixel" : "input", ylabel: c.image ? "pixel" : "input",
      tooltip: (i, j, v) => ui.tipRows(`inputs g${sub(i + 1)}, g${sub(j + 1)}`, [{ name: i === j ? "variance" : "covariance", value: f3(v) }]) });
    ui.renderMath(sigmaBox);
    // real data: the covariance as images, under the sim
    L.below.replaceChildren();
    if (c.image) {
      const im = c.image, w = im.w;
      imageBox = ui.h("div");
      const wrap = ui.h("div", { style: { gridColumn: "1 / -1", minWidth: "0" } });
      L.below.append(wrap);
      wrap.append(ui.h("div", { class: "chart-title", html: "(e) The real covariance as images: mean digit, pixel variances $\\Sigma_{bb}$, and the top eight eigenvectors of $\\Sigma_g$" }), imageBox);
      const vari = Float32Array.from({ length: B }, (_, b) => S[b * B + b]);
      imageGrid(imageBox, { arrays: [Float32Array.from(im.mean_image), vari], w, h: w, map: "seq", range: "tile", tile: 64, labels: ["mean", "variance"],
        tooltip: i => ui.tipRows(i ? "pixel variance (scaled)" : "mean pixel intensity", []) });
      const eg = ui.h("div", { style: { marginTop: "8px" } }); imageBox.append(eg);
      imageGrid(eg, { arrays: im.eigvecs.map(v => Float32Array.from(v)), w, h: w, map: "div", range: "tile", tile: 64, labels: im.eigvecs.map((_, k) => `v${sub(k + 1)}`),
        tooltip: k => ui.tipRows(`eigenvector ${k + 1} of Σ_g`, [{ name: "eigenvalue", value: f3(im.eigvals[k]) }], "signed (red +, blue −), its own symmetric scale") });
      wrap.append(ui.h("div", { class: "u-help", html: `Computed by <code>hebb/homeo_cov.py</code> from the ${ui.fmtInt(im.n_images)} MNIST training images, average-pooled $28\\times28 \\to 7\\times7$; one global scale so the mean variance is 1. Eigenvectors are signed and global; compare the clusters' parts in (d), which are non-negative and local.` }));
      ui.renderMath(L.below);
    }
  }

  // ------------------------------------------------------------ drawing
  function colourise(a) {
    a.major.forEach((g, k) => { g.color = catColor(k); g.rank = k; });
    a.minor.forEach(g => { g.color = P.other; g.rank = -1; });
    return a;
  }
  const statCache = {};
  const setStat = (key, val, s) => { const sig = `${val}|${s}`; if (statCache[key] === sig) return; statCache[key] = sig; stats.set(key, val, s); };
  function showStats(a) {
    const sizes = a.major.length <= 6 ? a.major.map(g => g.n).join(" · ") : `${a.major.length} clusters`;
    setStat("k", a.K, `${sizes}${a.minor.length ? ` · +${a.minor.length} small` : ""}${a.robust ? "" : " · not yet clustered"}`);
    const c = sim.c;
    setStat("ns", a.nStable, a.tie ? `marginal here: ${a.tie[0]} just below, ${a.tie[1]} just above` : c.method === "enum" ? "exact enumeration of supports" : "lone-neuron flow, checked exactly");
    const fit = xMode === "c" ? a.fitC : a.fitL;
    setStat("beta", a.K >= 3 && fit ? +fit.beta.toFixed(2) : NaN, a.K >= 3 && fit ? `over ${fit.n} clusters, r = ${ui.fmtFixed(fit.r, 2)}` : a.K >= 3 ? "equal energies: undefined" : "needs ≥ 3 clusters");
    setStat("dead", a.nDead, a.noise !== "none" ? `of ${a.N} · in transit ${a.transit} (${ui.fmtPct(a.transit / a.N, 0)})` : `of ${a.N}`);
    setStat("t", Math.round(a.t * 10) / 10, `dt = ${ui.fmt(M.DT / a.lam, 2)}${fps && tp.playing ? ` · ${fps.toFixed(0)} fps` : ""} · max w ${ui.fmt(a.wmax, 2)}`);
    setStat("r", +a.resid.toPrecision(2), a.mode !== "mf" || a.noise !== "none" ? "mean-field drift at these weights" : a.resid < M.RES_TOL ? "converged (below 10⁻⁶)" : "still moving");
  }
  function showRegime() {
    const c = sim.c, pt = M.partsAt(c, p.theta), rg = M.regime(c, p, pt.n);
    const om = p.omega > 0.02 ? ` <b>Floor:</b> $\\omega = ${ui.fmt(p.omega, 2)}$ is above the recommended 0.02; clusters leak towards the population mean (strength $\\sim\\omega/c$), and at high $\\theta$ the network can die.` : "";
    regimeBox.innerHTML = `<b>Regime: ${rg.name}.</b> ${rg.text}${om}`;
    ui.renderMath(regimeBox);
  }
  function showMode() {
    const wmin = M.wminOf(p);
    modeLine.innerHTML = `${p.batch ? `minibatch: plasticity sees the covariance of ${p.batch} Gaussian samples per step (shared by all neurons)` : "mean-field: plasticity sees the exact $\\Sigma_g$"}${p.noise !== "none" ? `, plus private ${M.NOISES.find(x => x.id === p.noise).label}` : ""} · $N = ${p.n}$, $w_{\\min} = \\omega/N = ${ui.fmt(wmin, 2)}$${p.lam !== 1 ? `/$\\lambda$` : ""} · Euler $dt = ${ui.fmt(M.DT / p.lam, 2)}$`;
    ui.renderMath(modeLine);
    omegaHelp.innerHTML = `A synapse at the floor still conducts $w_{\\min}$; together they pull every cluster towards the population mean. Here $w_{\\min} = ${ui.fmt(wmin, 2)}$ for $N = ${p.n}$.`;
    ui.renderMath(omegaHelp);
  }
  function drawParts(a) {
    const B = a.B, c = sim.c, maxP = c.image ? 12 : 8;
    const sig = a.major.map(g => `${g.n}:${g.part}`).join(",") + "|" + a.empty.map(q => q.k).join(",") + "|" + a.theta;
    if (sig === partsSig) return; partsSig = sig;
    const shown = a.major.slice(0, maxP), empties = a.empty.slice(0, Math.max(0, (c.image ? 12 : 10) - shown.length));
    const xname = g => `λ = ${ui.fmt(g.lam, 3)}, c = ${ui.fmt(g.c, 3)}`;
    if (c.image) {
      const arrays = [...shown.map(g => Float32Array.from(g.h)), ...empties.map(q => Float32Array.from(M.partVec(q, B)))];
      const rows = []; if (shown.length) rows.push({ label: "clusters", count: shown.length }); if (empties.length) rows.push({ label: "empty parts", count: empties.length });
      if (!arrays.length) { partsBox.replaceChildren(ui.h("p", { class: "mute", text: "no live clusters" })); return; }
      partsBox.replaceChildren(); const gb = ui.h("div"); partsBox.append(gb);
      imageGrid(gb, { arrays, w: c.image.w, h: c.image.w, map: "seq", range: "tile", tile: 58, rows,
        labels: [...shown.map(g => String(g.n)), ...empties.map(() => "0")], colors: [...shown.map(g => g.color), ...empties.map(() => P.other)],
        tooltip: i => i < shown.length ? ui.tipRows(`cluster c${i + 1}: ${shown[i].n} neurons`, [{ name: "energy λmax(Σ_SS)", value: f3(shown[i].lam) }, { name: "c (supra-threshold)", value: f3(shown[i].c) }, { name: "inputs in support", value: String(shown[i].supp.length) }], shown[i].part >= 0 ? "at a stable part" : "not (yet) at a stable part")
          : ui.tipRows("stable part with no neurons", [{ name: "energy λmax(Σ_SS)", value: f3(empties[i - shown.length].lam) }, { name: "c", value: f3(empties[i - shown.length].c) }, { name: "support size", value: String(empties[i - shown.length].S.length) }]) });
      return;
    }
    const labels = Array.from({ length: B }, (_, b) => String(b + 1));
    const panels = [...shown.map((g, k) => ({ title: `c${k + 1}: ${g.n} neuron${g.n === 1 ? "" : "s"}`, values: Array.from(g.h), color: g.color, sub: xname(g) })),
      ...empties.map(q => ({ title: "no neurons", values: Array.from(M.partVec(q, B)), color: P.other, sub: `stable part · λ = ${ui.fmt(q.lam, 3)}` }))];
    if (!panels.length) { partsBox.replaceChildren(ui.h("p", { class: "mute", text: "no live clusters and no stable part" })); return; }
    barMultiples(partsBox, { labels, panels, yrange: [0, 1], height: 70, panelWidth: B > 12 ? 190 : 150, valueName: "h (share of each input)", labelEvery: B > 12 ? 4 : B > 8 ? 2 : 1 });
  }
  function show(a, full) {
    alloc.update(a, xMode); showStats(a);
    strip.update({ c: sim.c, gamma: a.gamma, theta: a.theta, K: a.K, range: thetaRange(sim.c) });
    if (full) { drawParts(a); river.update(tp.frames); river.setCursor(a.t); }
  }

  // ------------------------------------------------------------ simulation loop
  function tick(speed) {
    const t0 = performance.now();
    let n = 0; while (n < speed && performance.now() - t0 < 24) n += sim.advance(1);
    const a = colourise(M.analyse(sim, prevLabels)); prevLabels = a.labels; cur = a; tickN++;
    fpsN++; const now = performance.now();
    if (now - fpsT > 1000) { fps = fpsN * 1000 / (now - fpsT); fpsN = 0; fpsT = now; }
    const done = !sim.noisy() && a.resid < M.RES_TOL;
    const capped = sim.noisy() && a.t > 1000 / p.lam;
    show(a, false);
    if (done || capped || tickN % 8 === 0) { drawParts(a); river.update(tp.frames.concat([a])); river.setCursor(a.t); }
    if (done) { a.done = true; a.doneLabel = "converged: r < 10⁻⁶"; }
    if (capped) { a.done = true; a.doneLabel = "stopped at t = 1000 (noisy run)"; }
    return a;
  }
  function restart() {
    sim.reset(p); prevLabels = null; tickN = 0; partsSig = "";
    const a = colourise(M.analyse(sim, null)); prevLabels = a.labels; cur = a;
    show(a, false); drawParts(a); river.update([a]); river.setCursor(a.t);
    return a;
  }

  // ------------------------------------------------------------ control plumbing
  function liveChange() {
    untour(); sim.update({ ...p }); sim.s.resid = NaN; partsSig = "";
    showRegime(); showMode(); writeURL();
    if (cur) strip.update({ c: sim.c, gamma: p.gamma, theta: p.theta, K: cur.K, range: thetaRange(sim.c) });
    tp.play();
  }
  function resetChange() { untour(); renderSigma(); showRegime(); showMode(); writeURL(); tp.reset(); tp.play(); }
  function syncAll() {
    covS.set(p.cov); makeThetaSlider(); gammaS.set(p.gamma); omegaS.set(p.omega); initS.set(p.init); nS.set(p.n); lamS.set(p.lam);
    seedS.set(p.seed); iiS.set(p.ii); batchS.set(p.batch); noiseS.set(p.noise);
  }
  function untour() { if (!tourActive) return; tourActive = null; chipSets.forEach(c => c.moved()); }
  function writeURL() {
    const d = M.DEFAULTS, o = { tour: tourActive ? tourActive.id : null };
    for (const k of M.KEYS) o[k] = tourActive || p[k] === d[k] ? null : p[k];
    ctx.setParams(o);
  }
  function applyTour(t) {
    tourActive = t; p = M.tourParams(t); syncAll(); renderSigma(); showRegime(); showMode(); writeURL();
    tp.reset(); tp.play();
  }
  function applyParams(np) {
    tourActive = null; chipSets.forEach(c => c.clear());
    p = np; syncAll(); renderSigma(); showRegime(); showMode(); tp.reset(); tp.play();
  }
  ctx.onParams(() => {
    const r = readParams(ctx, data);
    if (r.tour) { GROUPS.forEach(([g], k) => { if (g === r.tour.group) chipSets[k].set(r.tour.id); else chipSets[k].clear(); }); applyTour(r.tour); return; }
    applyParams(r.p);
  });

  // ------------------------------------------------------------ go
  renderSigma(); showRegime(); showMode();
  tp.record(restart());
  return {
    // load a parameter state from a precomputed figure (phase diagram click) and bring the explorer into view
    load(params, why) {
      const np = { ...M.DEFAULTS, ...params };
      applyParams(np); writeURL();
      fig.scrollIntoView({ behavior: ui.reducedMotion() ? "auto" : "smooth", block: "start" });
      if (why) { modeLine.insertAdjacentHTML("afterbegin", `<span style="color:${P.accent}">${why} · </span>`); ui.renderMath(modeLine); }
    },
  };
}

// URL -> parameters (a tour in the URL wins)
function readParams(ctx, data) {
  const tour = M.tourById(ctx.param("tour", ""));
  if (tour) return { p: M.tourParams(tour), tour };
  const d = M.DEFAULTS, p = {};
  for (const k of M.KEYS) p[k] = ctx.param(k, d[k]);
  if (!data.covs[p.cov]) p.cov = d.cov;
  const c = M.covOf(data, p.cov), hi = Math.min(1.6, Math.ceil((c.stats.var_max + 0.05) * 100) / 100);
  p.theta = Math.max(-0.2, Math.min(hi, Math.round(p.theta * 100) / 100));
  p.gamma = Math.max(0, Math.min(3, p.gamma)); p.omega = Math.max(1e-4, Math.min(1, p.omega));
  if (!M.INITS.some(x => x.id === p.init)) p.init = d.init;
  p.n = Math.max(60, Math.min(250, Math.round(p.n / 10) * 10)); p.seed = Math.max(1, Math.round(p.seed));
  if (![0.25, 0.5, 1, 2, 4].includes(p.lam)) p.lam = 1;
  if (!M.BATCHES.includes(p.batch)) p.batch = 0; if (!M.NOISES.some(x => x.id === p.noise)) p.noise = "none";
  p.ii = Math.max(0, Math.min(1, p.ii));
  return { p, tour: null };
}

// slider ticks for theta: kappa0, gamma*kappa0, min/max covariance, min/max variance (labels merged when close)
function thetaMarks(c, gamma, lo, hi) {
  const st = c.stats, raw = [
    { value: st.kappa0, label: "κ₀", title: `κ₀ = ${f3(st.kappa0)}: mean covariance of two random initial embeddings` },
    { value: gamma * st.kappa0, label: "γκ₀", title: `γκ₀ = ${f3(gamma * st.kappa0)}: collective threshold` },
    { value: st.offdiag_min, label: "cov↓", title: `smallest covariance ${f3(st.offdiag_min)}: below it, exactly one part` },
    { value: st.offdiag_max, label: "cov↑", title: `largest covariance ${f3(st.offdiag_max)}` },
    { value: st.var_min, label: "var↓", title: `smallest variance ${f3(st.var_min)}` },
    { value: st.var_max, label: "var↑", title: `largest variance ${f3(st.var_max)}: above it nothing exists` },
  ].filter(m => m.value >= lo && m.value <= hi);
  raw.sort((a, b) => a.value - b.value);
  // a label only when it is clear of the previous labelled tick; coincident ticks share one label ("κ₀=γκ₀")
  let last = null;
  for (const m of raw) {
    if (last && (m.value - last.value) / (hi - lo) < 0.17) { last.title += "; " + m.title; m.label = ""; }
    else last = m;
  }
  return raw;
}
function groupRuns(lab) {
  const out = []; let s = 0;
  for (let i = 1; i <= lab.length; i++) if (i === lab.length || lab[i] !== lab[s]) { out.push({ start: s, end: i, label: `${lab[s] + 1}` }); s = i; }
  return out;
}

// ======================================================================================== custom charts
// (a) allocation plot: neurons per cluster against the energy of its part, log-log, with the fitted exponent and the
// proportional reference; empty stable parts drawn on a "0" row under the axis.
function allocPlot(el, { height = 320 } = {}) {
  el.classList.add("chart");
  const hostEl = ui.h("div", { style: { position: "relative" } }); el.replaceChildren(hostEl);
  const lg = ui.legend({ items: [
    { label: "cluster (colour: size rank)", color: P.cat[0], shape: "dot" },
    { label: "stable part, no neurons", color: P.mute2, shape: "ring" },
    { label: "fit $n \\propto x^{\\beta}$", color: P.fg, shape: "line" },
    { label: "proportional, $\\beta = 1$", color: P.mute2, shape: "dash" },
  ] });
  lg.el.classList.add("chart-legend"); el.append(lg.el); ui.renderMath(lg.el);
  let a = null, mode = "lam", W = 0, pts = [];
  function draw() {
    W = Math.floor(el.clientWidth); if (!W || !a) return;
    hostEl.replaceChildren();
    const Hh = height, m = { l: 50, r: 14, t: 12, b: 60 };
    const svg = svgEl("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "neurons per cluster against part energy" }, hostEl);
    const xOf = g => mode === "c" ? g.c : g.lam;
    const groups = [...a.major, ...a.minor].filter(g => xOf(g) > 0);
    const xs = [...a.parts.map(q => mode === "c" ? q.c : q.lam), ...groups.map(xOf)].filter(x => x > 0);
    let x0 = xs.length ? Math.min(...xs) : 0.5, x1 = xs.length ? Math.max(...xs) : 2;
    if (x1 / x0 < 3) { const g = Math.sqrt(x0 * x1); x0 = g / 1.8; x1 = g * 1.8; } else { x0 /= 1.25; x1 *= 1.25; }
    const X = scale("log", x0, x1, m.l, W - m.r), Y = scale("log", 0.8, a.N * 1.3, Hh - m.b, m.t);
    const gGrid = svgEl("g", { class: "grid" }, svg), ax = svgEl("g", { class: "axis" }, svg);
    const yt = logTicks(1, a.N).filter(v => v <= a.N);
    const xt = xTicks(x0, x1);
    yt.forEach(v => { svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, gGrid); text(svg, m.l - 6, Y(v) + 3.5, ui.fmt(v), { "text-anchor": "end" }); });
    xt.forEach(v => { svgEl("line", { x1: X(v), x2: X(v), y1: m.t, y2: Hh - m.b }, gGrid); text(svg, X(v), Hh - m.b + 34, ui.fmt(v, 2), { "text-anchor": "middle" }); });
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Hh - m.b + .5, y2: Hh - m.b + .5 }, ax);
    // the "0" row
    const y0 = Hh - m.b + 16;
    text(svg, m.l - 6, y0 + 3.5, "0", { "text-anchor": "end" });
    svgEl("line", { x1: m.l, x2: W - m.r, y1: y0, y2: y0, stroke: P.line, "stroke-dasharray": "2 3" }, svg);
    text(svg, (m.l + W - m.r) / 2, Hh - 6, mode === "c" ? "supra-threshold eigenvalue c of the part (log)" : "energy of the part: top eigenvalue of Σ on its support (log)", { "text-anchor": "middle", class: "ttl" });
    text(svg, 0, 0, "neurons n (log)", { "text-anchor": "middle", class: "ttl", transform: `translate(12,${(m.t + Hh - m.b) / 2}) rotate(-90)` });
    const fit = mode === "c" ? a.fitC : a.fitL;
    const clip = svgEl("clipPath", { id: "hh-alloc-clip" }, svgEl("defs", {}, svg));
    svgEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: Hh - m.t - m.b }, clip);
    const pl = svgEl("g", { "clip-path": "url(#hh-alloc-clip)" }, svg);
    if (fit && a.K >= 3) {
      const line = (slope, attrs) => { const ya = Math.exp(fit.my + slope * (Math.log(x0) - fit.mx)), yb = Math.exp(fit.my + slope * (Math.log(x1) - fit.mx)); svgEl("line", { x1: X(x0), x2: X(x1), y1: Y(ya), y2: Y(yb), ...attrs }, pl); };
      line(1, { stroke: P.mute2, "stroke-width": 1.3, "stroke-dasharray": "5 4" });
      line(fit.beta, { stroke: P.fg, "stroke-width": 1.6 });
      const lab = text(svg, m.l + 10, m.t + 16, `β = ${ui.fmtFixed(fit.beta, 2)}  (fit over ${fit.n} clusters, r = ${ui.fmtFixed(fit.r, 2)})`, { class: "ttl" });
      lab.setAttribute("style", "font-weight:600");
    } else {
      text(svg, m.l + 10, m.t + 16, a.K >= 3 ? "β undefined: every cluster's part has the same energy" : a.K >= 1 ? `β needs at least 3 clusters (K = ${a.K})` : "no live clusters", { fill: P.mute });
    }
    pts = [];
    for (const q of a.empty) {
      const xv = mode === "c" ? q.c : q.lam; if (!(xv > 0)) continue;
      svgEl("circle", { cx: X(xv), cy: y0, r: 4.5, fill: P.surface, stroke: P.mute2, "stroke-width": 1.5 }, svg);
      pts.push({ x: X(xv), y: y0, tip: () => ui.tipRows("stable part with no neurons", [{ name: "energy λmax(Σ_SS)", value: f3(q.lam) }, { name: "c (supra-threshold)", value: f3(q.c) }, { name: "inputs in support", value: String(q.S.length) }]) });
    }
    for (const g of [...a.minor].reverse().concat([...a.major].reverse())) {
      const xv = xOf(g); if (!(xv > 0)) continue;
      const big = g.rank >= 0;
      svgEl("circle", { cx: X(xv), cy: Y(g.n), r: big ? 6 : 3.5, fill: g.color, stroke: P.surface, "stroke-width": 1.5, opacity: big ? 0.95 : 0.8 }, svg);
      pts.push({ x: X(xv), y: Y(g.n), tip: () => ui.tipRows(g.rank >= 0 ? `cluster c${g.rank + 1}` : "small cluster", [
        { name: "neurons n", value: String(g.n), color: g.color, shape: "dot" }, { name: "energy λmax(Σ_SS)", value: f3(g.lam) }, { name: "c = (hᵀΣh − θ)/|h|²", value: f3(g.c) },
        { name: "inputs in support", value: String(g.supp.length) }], g.part >= 0 ? "sits at a stable part (cos > 0.98)" : `not at a stable part (best cos ${ui.fmtFixed(g.cos, 3)})`) });
    }
    const hit = svgEl("rect", { x: 0, y: 0, width: W, height: Hh, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", ev => {
      const r = svg.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
      let best = null, bd = 14 * 14; for (const q of pts) { const d = (q.x - mx) ** 2 + (q.y - my) ** 2; if (d < bd) { bd = d; best = q; } }
      if (best) ui.showTip(ev.clientX, ev.clientY, best.tip()); else ui.hideTip();
    });
    hit.addEventListener("pointerleave", () => ui.hideTip());
  }
  const ro = new ResizeObserver(() => { if (Math.floor(el.clientWidth) !== W) draw(); }); ro.observe(el);
  return { update(an, md) { a = an; mode = md || mode; draw(); } };
}
function xTicks(x0, x1) {
  const out = [], span = Math.log10(x1 / x0), mant = span > 1.6 ? [1, 2, 5] : [1, 1.5, 2, 3, 5, 7];
  for (let e = Math.floor(Math.log10(x0)) - 1; e <= Math.ceil(Math.log10(x1)); e++) for (const k of mant) { const v = k * Math.pow(10, e); if (v >= x0 && v <= x1) out.push(v); }
  return out;
}
function text(parent, x, y, s, attrs = {}) { const t = svgEl("text", { x, y, ...attrs }, parent); t.textContent = s; return t; }

// (b) the theta axis: number of stable parts n(theta) as a step line, regimes shaded, the current theta and K
function thetaStrip(el, { height = 180, onPick } = {}) {
  el.classList.add("chart");
  const hostEl = ui.h("div"); el.replaceChildren(hostEl);
  let st = null, W = 0;
  function draw() {
    W = Math.floor(el.clientWidth); if (!W || !st) return;
    hostEl.replaceChildren();
    const { c, gamma, theta, K, range } = st, [lo, hi] = range, Hh = height, m = { l: 34, r: 10, t: 30, b: 34 };
    const rows = c.parts.filter(r => r.theta >= lo - 1e-9 && r.theta <= hi + 1e-9);
    const nmax = Math.max(K, ...rows.map(r => r.n), 1);
    const X = scale("linear", lo, hi, m.l, W - m.r), Y = scale("linear", 0, nmax * 1.08, Hh - m.b, m.t);
    const svg = svgEl("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "number of stable parts against theta" }, hostEl);
    const k0 = c.stats.kappa0 * gamma, vmax = c.stats.var_max;
    if (gamma > 0 && k0 > lo) { const xb = X(Math.min(k0, hi)); svgEl("rect", { x: m.l, y: m.t, width: Math.max(0, xb - m.l), height: Hh - m.t - m.b, fill: P.seq[1], opacity: 0.55 }, svg); if (xb - m.l > 110) text(svg, m.l + 4, m.t + 11, "collective (heuristic)", { fill: P.ink2, "font-size": 10.5 }); else if (xb - m.l > 54) text(svg, m.l + 4, m.t + 11, "collective", { fill: P.ink2, "font-size": 10.5 }); }
    if (vmax < hi) { const xa = X(vmax); svgEl("rect", { x: xa, y: m.t, width: W - m.r - xa, height: Hh - m.t - m.b, fill: P.wash }, svg); if (W - m.r - xa > 40) text(svg, xa + 4, m.t + 11, "extinct", { fill: P.ink2, "font-size": 10.5 }); }
    const g = svgEl("g", { class: "grid" }, svg);
    const yt = niceTicks(0, nmax, 3).filter(v => Number.isInteger(v));
    yt.forEach(v => { svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); text(svg, m.l - 5, Y(v) + 3.5, String(v), { "text-anchor": "end" }); });
    niceTicks(lo, hi, Math.max(3, Math.round(W / 80))).forEach(v => text(svg, X(v), Hh - m.b + 14, ui.fmt(v, 2), { "text-anchor": "middle" }));
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Hh - m.b + .5, y2: Hh - m.b + .5, stroke: "#c9c8c1" }, svg);
    text(svg, (m.l + W - m.r) / 2, Hh - 4, "θ", { "text-anchor": "middle", class: "ttl" });
    // reference levels, labels staggered on two rows above the plot
    const marks = [[c.stats.offdiag_min, "cov↓"], [c.stats.offdiag_max, "cov↑"], [c.stats.var_min, "var↓"], [vmax, "var↑"], [c.stats.kappa0, "κ₀"], [k0, "γκ₀"]]
      .filter(([v]) => v >= lo && v <= hi).sort((x, y) => x[0] - y[0]);
    let lastX = [-99, -99, -99];
    marks.forEach(([v, lab]) => {
      const xv = X(v), dashed = lab.startsWith("κ") || lab.startsWith("γ");
      svgEl("line", { x1: xv, x2: xv, y1: m.t - 4, y2: Hh - m.b, stroke: dashed ? P.ink2 : P.mute2, "stroke-width": 1, "stroke-dasharray": dashed ? "4 3" : "1.5 2.5" }, svg);
      let rowI = lastX.findIndex(x => xv - x > 30); if (rowI < 0) return;      // no room: the tick stays, the label goes
      lastX[rowI] = xv;
      text(svg, xv, m.t - 6 - rowI * 10, lab, { "text-anchor": "middle", "font-size": 10, fill: P.ink2 });
    });
    // n(theta) step line
    // step line; a grid value exactly on a correlation level (marginal parts, `tie`) is left as a gap
    let d = "", pen = false;
    rows.forEach(r => {
      if (r.tie) { pen = false; return; }
      const xa = X(r.theta - 0.005 < lo ? lo : r.theta - 0.005), xb = X(r.theta + 0.005 > hi ? hi : r.theta + 0.005), y = Y(r.n);
      d += (pen ? "L" : "M") + xa.toFixed(1) + "," + y.toFixed(1) + "L" + xb.toFixed(1) + "," + y.toFixed(1); pen = true;
    });
    svgEl("path", { d, fill: "none", stroke: P.fg, "stroke-width": 1.6, "stroke-linejoin": "round" }, svg);
    // now
    const xt = X(theta);
    svgEl("line", { x1: xt, x2: xt, y1: m.t, y2: Hh - m.b, stroke: P.accent, "stroke-width": 1.4 }, svg);
    svgEl("circle", { cx: xt, cy: Y(K), r: 5, fill: P.cat[0], stroke: P.surface, "stroke-width": 1.5 }, svg);
    const hit = svgEl("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: Hh, fill: "transparent", style: "cursor:pointer" }, svg);
    const thOf = ev => { const r = svg.getBoundingClientRect(); return Math.max(lo, Math.min(hi, Math.round(X.inv(ev.clientX - r.left) * 100) / 100)); };
    hit.addEventListener("pointermove", ev => {
      const th = thOf(ev), row = rows.find(r => Math.abs(r.theta - th) < 1e-6);
      ui.showTip(ev.clientX, ev.clientY, ui.tipRows(`θ = ${f2(th)}`, [{ name: "stable parts", value: row ? String(row.n) : "–" }], row && row.tie ? `exactly on a level: ${row.tie[0]} just below, ${row.tie[1]} just above` : "click to set θ"));
    });
    hit.addEventListener("pointerleave", () => ui.hideTip());
    hit.addEventListener("click", ev => onPick && onPick(thOf(ev)));
  }
  const ro = new ResizeObserver(() => { if (Math.floor(el.clientWidth) !== W) draw(); }); ro.observe(el);
  return { update(s) { st = s; draw(); } };
}

// (c) neurons per cluster over time, stacked (canvas), adapted from chapter 1's river
function riverChart(el, { height = 190 } = {}) {
  el.classList.add("chart");
  const cv = ui.h("canvas"), ov = ui.h("canvas", { style: { position: "absolute", left: "0", top: "0", pointerEvents: "none" } });
  el.style.position = "relative"; el.replaceChildren(cv, ov);
  let frames = [], cursor = null, W = 0, X = null; const m = { l: 40, r: 10, t: 8, b: 34 };
  function draw() {
    W = Math.floor(el.clientWidth); if (!W) return;
    const g = hidpi(cv, W, height); g.clearRect(0, 0, W, height);
    const N = frames.length ? frames[frames.length - 1].N : 100, t1 = frames.length ? Math.max(frames[frames.length - 1].t, 1) : 1;
    X = scale("linear", 0, t1, m.l, W - m.r); const Y = scale("linear", 0, N, height - m.b, m.t);
    g.font = "11px Inter, system-ui, sans-serif"; g.lineWidth = 1; g.strokeStyle = "#efeee9";
    for (const yv of [0, N / 2, N]) { g.beginPath(); g.moveTo(m.l, Math.round(Y(yv)) + .5); g.lineTo(W - m.r, Math.round(Y(yv)) + .5); g.stroke(); }
    for (let k = 0; k < frames.length; k++) {
      const f = frames[k], x0 = X(f.t), x1 = k + 1 < frames.length ? X(frames[k + 1].t) : x0 + 2;
      let acc = 0;
      const col = (n, c, alpha = 1) => { const y0 = Y(acc), y1 = Y(acc + n); acc += n; g.globalAlpha = alpha; g.fillStyle = c; g.fillRect(x0, y1, Math.max(1, x1 - x0 + 0.6), y0 - y1); g.globalAlpha = 1; };
      for (const gr of f.major) col(gr.n, gr.color);
      let small = 0; for (const gr of f.minor) small += gr.n;
      if (small) col(small, P.other, 0.8);
      if (f.transit) col(f.transit, P.other, 0.55);
      if (f.nDead) col(f.nDead, P.other, 0.35);
    }
    g.strokeStyle = "#c9c8c1"; g.beginPath(); g.moveTo(m.l, height - m.b + .5); g.lineTo(W - m.r, height - m.b + .5); g.stroke();
    g.fillStyle = P.mute; g.textAlign = "right";
    for (const yv of [0, N / 2, N]) g.fillText(String(yv), m.l - 6, Y(yv) + 3.5);
    g.textAlign = "center";
    for (const xv of niceTicks(0, t1, Math.max(3, Math.round(W / 90)))) g.fillText(ui.fmt(xv), X(xv), height - m.b + 15);
    g.fillStyle = P.ink2; g.font = "560 11.5px Inter, system-ui, sans-serif";
    g.fillText("time t", (m.l + W - m.r) / 2, height - 3);
    g.save(); g.translate(11, (m.t + height - m.b) / 2); g.rotate(-Math.PI / 2); g.fillText("neurons", 0, 0); g.restore();
    drawCursor();
  }
  let o = null, ovW = 0;
  function drawCursor() {
    if (!W) return;
    if (ovW !== W || !o) { o = hidpi(ov, W, height); ovW = W; }
    o.clearRect(0, 0, W, height);
    if (cursor == null || !frames.length || !X) return;
    o.strokeStyle = P.accent; o.globalAlpha = .8; o.lineWidth = 1.25; o.beginPath(); o.moveTo(X(cursor), m.t); o.lineTo(X(cursor), height - m.b); o.stroke();
  }
  const ro = new ResizeObserver(() => { if (Math.floor(el.clientWidth) !== W) draw(); }); ro.observe(el);
  cv.addEventListener("pointermove", ev => {
    if (!X || !frames.length) return;
    const r = cv.getBoundingClientRect(), t = X.inv(ev.clientX - r.left);
    let f = frames[0]; for (const q of frames) if (q.t <= t) f = q;
    const rows = f.major.slice(0, 8).map((g, k) => ({ name: `c${k + 1} (λ = ${ui.fmt(g.lam, 3)})`, value: String(g.n), color: g.color }));
    if (f.transit) rows.push({ name: "in transit (noise)", value: String(f.transit), color: P.other });
    if (f.nDead) rows.push({ name: "dead", value: String(f.nDead), color: P.other });
    ui.showTip(ev.clientX, ev.clientY, ui.tipRows(`t = ${ui.fmtFixed(f.t, 1)}: K = ${f.K}`, rows));
  });
  cv.addEventListener("pointerleave", () => ui.hideTip());
  return { update(fr) { frames = fr; draw(); }, setCursor(t) { if (t === cursor) return; cursor = t; drawCursor(); } };
}
