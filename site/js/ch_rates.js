// Chapter "rates" (Rectified neurons): linear circuit, rectified output. Live: the dual activation with the
// empirical scatter (engine closed form for tau = 0) and a linear-vs-rectified ring simulation. Precomputed:
// the MNIST exports in data/nonlin (parts across q, training film, probes, readout bias, samples).
import * as ui from "./ui.js";
import { lineChart } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as E from "./engine.js";
import * as D from "./ch_rates_data.js";
import { ringFigure } from "./ch_rates_ring.js";
import { partsFigure, filmFigure, samplesFigure } from "./ch_rates_digits.js";

const TAU_COL = { 0: catColor(0), 0.5: catColor(1), 1: catColor(2) };
const f2 = x => (Number.isFinite(x) ? ui.fmtFixed(x, 2) : "n/a");
const f3 = x => (Number.isFinite(x) ? ui.fmtFixed(x, 3) : "n/a");
const rg = ([a, b], f = f2) => (!Number.isFinite(a) ? "n/a" : f(a) === f(b) ? f(a) : a < 0 ? `${f(a)} to ${f(b)}` : `${f(a)}–${f(b)}`);

export async function init(panel, ctx) {
  // the live ring figure needs no data
  ringFigure(ctx.slot("ring"), ctx);

  const [index, dual, synth] = await Promise.all([D.json("index.json"), D.json("dual.json"), D.json("synth_ring.json")]);
  const S = D.summarise(index);
  ringPython(ctx, synth);
  dualFigure(ctx.slot("dual"), dual, index, ctx);
  partsTable(ctx.slot("parts-table"), S);
  scoreTable(ctx.slot("score"), S);
  probeChart(ctx.slot("probe"), index, S);
  readoutChart(ctx.slot("readout"), index);
  [ctx.slot("parts-table"), ctx.slot("score"), ctx.article.querySelector('[data-n="ring-python"]')].forEach(el => el && ctx.renderMath(el));
  // image-heavy figures last
  await partsFigure(ctx.slot("parts"), index);
  filmFigure(ctx.slot("film"), index, ctx);
  await samplesFigure(ctx.slot("samples"), index);
}

// ------------------------------------------------------------------ the dual activation, with the empirical scatter
const DUAL_RUNS = [
  { key: "A_rect_t0", label: "rectified τ = 0, digits, q = 0.980" },
  { key: "A_rect_t0_q0.946", label: "rectified τ = 0, digits, q = 0.946" },
  { key: "A_rect_t1", label: "rectified τ = 1, digits, q = 0.980" },
  { key: "A_lin", label: "linear rule (rates post hoc, τ = 0)" },
  { key: "Bown_rect_t0", label: "rectified τ = 0 trained on Gaussian pixels (measured on digits)" },
];
const FAMS = [{ value: "bb", label: "pixel pairs" }, { value: "ib", label: "neuron–pixel" }, { value: "ii", label: "neuron pairs" }];

function dualFigure(host, dual, index, ctx) {
  const byName = Object.fromEntries(index.runs.map(r => [r.name, r]));
  let run = DUAL_RUNS.some(o => o.key === ctx.param("dual", "")) ? ctx.param("dual", "") : "A_rect_t0";
  let fam = FAMS.some(o => o.value === ctx.param("fam", "")) ? ctx.param("fam", "") : "ib";
  host.replaceChildren();
  const grid = ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "16px 32px", alignItems: "flex-start" } });
  const chartBox = ui.h("div", { style: { minWidth: 0, flex: "1 1 480px" } }), side = ui.h("div", { style: { minWidth: 0, flex: "0 1 300px" } });
  grid.append(chartBox, side); host.append(grid);

  // curves: tau = 0 from the engine's closed form; 0.5 and 1 tabulated in dual.json
  const rho = dual.rho, k1 = E.reluCorr(1);
  const k0 = rho.map(x => E.reluCorr(x) / k1);
  const curves = [
    { name: "k̃₀ (engine, closed form)", x: rho, y: k0, color: TAU_COL[0], width: 2.2 },
    { name: "k̃₀.₅", x: rho, y: dual.ktilde["0.5"], color: TAU_COL[0.5], width: 2.2 },
    { name: "k̃₁", x: rho, y: dual.ktilde["1"], color: TAU_COL[1], width: 2.2 },
    { name: "identity (linear rule)", x: [-1, 1], y: [-1, 1], color: P.mute2, dash: true, width: 1.5 },
  ];
  const chart = lineChart(chartBox, { series: curves, xrange: [-1, 1], yrange: [-0.55, 1.02], height: 380, xlabel: "voltage correlation ρ", ylabel: "rate correlation", xname: "ρ", yfmt: f2, xfmt: f2 });

  const selRun = ui.select({ stack: true, label: "network", options: DUAL_RUNS.map(o => ({ value: o.key, label: o.label })), value: run,
    onChange: v => { run = v; ctx.setParams({ dual: v }); draw(); } });
  const segFam = ui.segmented({ stack: true, label: "pairs", options: FAMS, value: fam, onChange: v => { fam = v; ctx.setParams({ fam: v }); draw(); } });
  Object.assign(selRun.select.style, { width: "100%", maxWidth: "100%" });
  ui.controls({ parent: side, primary: [selRun, segFam] });
  const stats = ui.stats({ parent: side, cols: 2, items: [
    { key: "mad", label: "mean |rate corr − k̃(ρ)|", fmt: f3 },
    { key: "fro", label: "‖E − P‖ / ‖E‖", fmt: f2 },
    { key: "vi", label: "rate variance / Gaussian, interior (median)", fmt: f2 },
    { key: "vp", label: "rate variance / Gaussian, pixels (median)", fmt: f2 },
  ] });
  const note = ui.h("div", { class: "u-help", style: { marginTop: "8px" } });
  side.append(note);

  function draw() {
    const sc = dual.scatter[run], tau = sc.tau, r = byName[sc.run], g = r.gap, pts = sc[fam];
    const col = TAU_COL[tau] || P.fg;
    const dots = { name: `pairs (${FAMS.find(f => f.value === fam).label}), τ = ${tau}`, x: pts.rho, y: pts.rate_corr, points: true, r: 2, color: col };
    chart.update({ series: [dots, ...curves] });
    stats.update({ mad: g[fam].rate_corr_mean_abs_gap, fro: g[fam].rel_fro_gap, vi: g.var_ratio_interior_q10_50_90[1], vp: g.var_ratio_pixel_q10_50_90[1] });
    const one = r.final.n_live_clusters === 1;
    note.innerHTML = `Run <code>${sc.run}</code>; its curve is $\\tilde k_{${tau}}$. ` +
      (fam === "ii" && one ? "One cluster holds every neuron, so every neuron pair sits at ρ = 1 (one dot)." :
        fam === "bb" ? "Pixel pairs do not depend on the network, only on τ." : "");
    ui.renderMath(note);
  }
  draw();
  ctx.onParams(p => {
    const a = p.get("dual"), b = p.get("fam");
    if (a && a !== run && DUAL_RUNS.some(o => o.key === a)) { run = a; selRun.set(a); }
    if (b && b !== fam && FAMS.some(o => o.value === b)) { fam = b; segFam.set(b); }
    draw();
  });
}

// ------------------------------------------------------------------ ring: the Python runs (synth_ring.json)
function ringPython(ctx, synth) {
  const el = ctx.article.querySelector('[data-n="ring-python"]');
  if (!el) return;
  // a cluster counts as live when its part's own (rate) variance exceeds theta: the condition for any of its synapses
  // to be active. The dead pool (members drifting to the centroid of their neighbours) fails it.
  const live = r => r.energy.filter(e => e > r.theta + 1e-9).length;
  const cell = (rule, tau, rs) => synth.runs.filter(r => r.rule === rule && (rule === "linear" || +r.tau === tau) && Math.abs(r.rho_star - rs) < 1e-6).sort((a, b) => a.seed - b.seed).map(live).join(", ");
  const rhos = [...new Set(synth.runs.map(r => r.rho_star))].filter(x => x >= 0.3 && x <= 0.8).sort((a, b) => a - b);
  el.innerHTML = `<p>The same comparison in Python (<code>synth_ring.json</code>, two numpy seeds), counting live clusters (a cluster is live when its part's own rate variance exceeds $\\theta$; 0 = extinct):</p>
<div class="table-scroll"><table class="data"><thead><tr><th>$\\rho^\\ast$</th>${rhos.map(r => `<th class="num">${r}</th>`).join("")}</tr></thead><tbody>
${[["linear rule", "linear", 0], ["rectified, $\\tau = 0$", "rect", 0], ["rectified, $\\tau = 1$", "rect", 1]].map(([lab, rule, tau]) => `<tr><td>${lab}</td>${rhos.map(rs => `<td class="num">${cell(rule, tau, rs)}</td>`).join("")}</tr>`).join("")}
</tbody></table></div>`;
}

// ------------------------------------------------------------------ tables
function partsTable(tbody, S) {
  tbody.replaceChildren(...S.partRows.map(r => ui.h("tr", {},
    ui.h("td", { html: r.label }),
    ui.h("td", { class: "num", text: r.cos.map(f3).join(" / ") }),
    ui.h("td", { class: "num", text: rg(r.skew) }),
    ui.h("td", { class: "num", text: rg(r.kurt), style: { whiteSpace: "nowrap" } }),
    ui.h("td", { class: "num", text: rg(r.excess) }))));
}

function scoreTable(tbody, S) {
  const p = S.probe;
  const ro = S.readout.map(x => `τ = ${x.tau}: ${x.ratio.map(f2).join(" / ")} (excess ${x.excess.map(f2).join(" / ")})`).join("<br>");
  const rows = [
    ["adds", "<b>Access to higher moments</b>, without a kernel lift", "The rule converges to a different part on digits; on Gaussian pixels with the same covariance it does not.",
      `cosine to the linear part ${rg(S.cosRect)}; Gaussian control at its own $q$: ${rg(S.cosSurr, f3)}`],
    ["adds", "<b>A skewed, heavy-tailed part</b> (projection pursuit)", "Two vertical bars, the sides of a “0”, instead of the central stroke.",
      `skewness ${rg(S.skewRect)} against ${rg(S.skewLin)} for the linear rule; 28×28: still a different part`],
    ["adds", "<b>Survives to higher θ</b>", "Rate covariances of skewed parts exceed the Gaussian value, so lateral binding is stronger.",
      `$\\tau = 1$, $q = 0.987$: ${ui.fmtInt(S.t1Alive987)} of 980 alive; linear rule: ${rg(S.linDead987, ui.fmtInt)} dead; Gaussian control at the digits' θ: ${ui.fmtInt(S.surrDead967)} dead already at $q = 0.967$`],
    ["adds", "<b>Different fixed points for Gaussian inputs</b>", "More and narrower clusters at the same pairwise threshold, from the convexity of $k_\\tau$; no new information.",
      "synthetic ring: see the live figure and the Python runs below it"],
    ["not", "<b>More clusters on real data</b>", "The dense population binds into one blob on the loudest part and stays there.",
      `live clusters at $q = 0.980$: rectified ${rg(S.rectLiveAtQ98, String)}, linear ${rg(S.linLiveAtQ98, String)}; most major clusters in any 14×14 rectified run: ${S.rectMajorMax}`],
    ["not", "<b>Diverse members</b>", "Members of the cluster are exact clones computing one function.",
      `mean $L_1$ distance to the centroid ≤ ${ui.fmt(S.cloneL1, 2)}`],
    ["not", "<b>A better code</b>", "Linear probes on the rates are worse than on the pixels, the untrained network or random features.",
      `rectified (1 cluster) ${rg(p.rectOne)}; linear with ≥ 3 clusters ${rg(p.linMany)}, with 1 cluster ${rg(p.linOne)}; pixels ${f3(p.pixels)}; untrained rates ${f2(p.untrained)}; random ReLU ${f2(p.randrelu)}`],
    ["not", "<b>A test of the allocation law</b>", "Neurons per cluster against energy needs two or more major clusters; the rectified rule never gave them.", "—"],
    ["not", "<b>An unbiased readout on real data</b>", "Inverting $k_\\tau$ overstates the learned part's energy by exactly the measured higher-order excess.",
      `${ro}<br>Gaussian-trained nets: ${rg(S.readoutSurr, f3)}; linear rule: ${rg(S.readoutLin)}`],
  ];
  const tag = k => (k === "adds" ? '<span class="verdict holds">adds</span>' : '<span class="verdict fails">does not add</span>');
  tbody.replaceChildren(...rows.map(([k, what, how, ev]) => ui.h("tr", {},
    ui.h("td", { html: `${tag(k)}<br>${what}` }), ui.h("td", { html: how }), ui.h("td", { html: ev, style: { fontSize: "13.5px" } }))));
}

// ------------------------------------------------------------------ probes against cluster count
function probeChart(host, index, S) {
  const nets = index.runs.filter(r => r.res === 14 && r.input === "pixels");
  const grp = [
    { name: "linear rule", f: r => r.rule === "linear", dx: -0.18, color: P.fg },
    { name: "rectified τ = 0", f: r => r.rule === "rect" && r.tau === 0, dx: 0, color: TAU_COL[0] },
    { name: "rectified τ = 0.5", f: r => r.rule === "rect" && r.tau === 0.5, dx: 0.12, color: TAU_COL[0.5] },
    { name: "rectified τ = 1", f: r => r.rule === "rect" && r.tau === 1, dx: 0.24, color: TAU_COL[1] },
  ];
  const series = grp.map(g => { const rs = nets.filter(g.f); return { name: g.name, x: rs.map(r => r.final.n_live_clusters + g.dx), y: rs.map(r => r.probes.trained_rate_t0_all), points: true, r: 3.5, color: g.color }; });
  host.replaceChildren();
  lineChart(host, { series, xrange: [-0.5, 8.5], yrange: [0, 1], height: 300, xlabel: "live clusters", ylabel: "probe test accuracy", xname: "live clusters",
    xfmt: x => String(Math.round(x)), yfmt: f3,
    refs: [{ y: S.probe.randrelu, label: `random ReLU ${f2(S.probe.randrelu)} · pixels ${f3(S.probe.pixels)} (lower line)` }, { y: S.probe.pixels }, { y: S.probe.untrained, label: `untrained network ${f2(S.probe.untrained)}` }] });
}

// ------------------------------------------------------------------ readout bias against the measured excess
function readoutChart(host, index) {
  const g1 = index.runs.filter(r => r.res === 14 && r.gamma === 1);
  const pts = (f, xOf) => { const rs = g1.filter(r => f(r) && D.mainPart(r)?.energy_ratio_gauss_both != null); return { x: rs.map(xOf), y: rs.map(r => D.mainPart(r).energy_ratio_gauss_both) }; };
  const series = [
    ...[0, 0.5, 1].map(tau => ({ name: `rectified τ = ${tau}, digits`, ...pts(r => r.exp === "A_rect" && r.tau === tau, r => D.mainPart(r).rate_var_over_gauss), points: true, r: 3.5, color: TAU_COL[tau] })),
    { name: "rectified, Gaussian pixels", ...pts(r => r.exp === "B_rect" || r.exp === "Bown_rect", () => 1), points: true, r: 3.5, color: P.fg },
    { name: "unbiased: y = x", x: [0.8, 5], y: [0.8, 5], color: P.mute2, dash: true, width: 1.5 },
  ];
  host.replaceChildren();
  lineChart(host, { series, xrange: [0.8, 5], yrange: [0.8, 5], height: 300, xlabel: "measured rate-variance excess", ylabel: "recovered energy ratio", xname: "excess", xfmt: f2, yfmt: f2 });
}
