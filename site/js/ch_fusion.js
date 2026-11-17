// Chapter "fusion" (Chapter 1, the centrepiece): the model as needed here, the two-neuron calibration, the live
// B = 3 network, and a b-scan of that network. Numbers quoted in the prose are read from data/sweeps2/fusion.json
// at load time or computed live.
// Modules: ch_fusion_model.js (model side, no DOM), ch_fusion_draw.js (purpose-built charts),
// ch_fusion_pair.js, ch_fusion_sim.js, ch_fusion_scan.js (the three figures).
import * as ui from "./ui.js";
import { initPair } from "./ch_fusion_pair.js";
import { initSim } from "./ch_fusion_sim.js";
import { initScan } from "./ch_fusion_scan.js";

export async function init(panel, ctx) {
  let J = null;
  try { const r = await fetch("data/sweeps2/fusion.json"); if (r.ok) J = await r.json(); } catch (e) { J = null; }

  const pair = initPair(ctx.slot("pair"), ctx, { loop: J ? J.pair.loop : null });
  const sim = initSim(ctx.slot("sim"), ctx);
  const scan = initScan(ctx.slot("scan"), ctx.slot("scan-text"), () => sim.params);
  ctx.onParams(params => pair.apply(params));
  ctx.onParams(() => scan.rerunIfChanged());

  if (J) fillFromSweep(J, ctx);
  else for (const k of ["g1", "pair-spec", "sweep-text"]) { const el = ctx.slot(k); if (el) el.textContent = "(The precomputed sweep data/sweeps2/fusion.json could not be loaded.)"; }
}

const f3 = x => ui.fmt(x, 3), f2 = x => ui.fmt(x, 2);

function fillFromSweep(J, ctx) {
  const c = J.meta.config, s = J.summary;
  // G1: the identity over the whole sweep
  ctx.slot("g1").innerHTML = `The same identities hold in the precomputed sweep (<code>data/sweeps2/fusion.json</code>; $N = ${c.N}$, $B = ${c.B}$, $s = ${c.s}$, two covariances, linear and rectifying neurons, $\\alpha \\in \\{${c.alphas.join(", ")}\\}$, ${c.b_grid.length} values of $b$, cold start and continuation up and down): ${s.n_converged} of ${s.n_runs} runs converged, the largest final residual was ${f2(s.max_residual)}, live edges satisfied $\\alpha w = b - D^2/2$ to ${f2(s.max_dev_live_converged)}, dead pairs had $D^2 - 2b \\ge ${f2(s.min_slack_dead_converged)}$, and a central difference of $\\mathbb E_g[E^\\ast]$ in $w_{ij}$ matched $D_{ij}^2/2$ to ${f2(s.max_envelope_err)}.`;

  // G2: the SPEC example and the continuation loop
  const ex = J.pair.spec_example, unit = ex.find(e => e.D02 === 2 && e.b === 0.2), half = ex.find(e => e.D02 === 1), lp = J.pair.loop;
  if (unit && half) {
    const fused = half.fixed_points.filter(f => f[1] && f[0] > 0).map(f => f3(f[0]));
    const ends = unit.sims.map(r => `${f3(r.w)} (from ${f3(r.w0)})`).join(", ");
    ctx.slot("pair-spec").innerHTML = `The example in SPEC.md ($b = ${half.b}$, $\\alpha = ${half.alpha}$, with $w = 0$ and $w \\approx ${fused.join(", ")}$ both stable) holds for $D_0^2 = ${half.D02}$, where $b_{sn} = ${f3(half.b_sn)}$. For two unit-variance inputs, $D_0^2 = ${unit.D02}$, so $b_{sn} = ${f3(unit.b_sn)} &gt; ${unit.b}$ and the only fixed point is $w = 0$: the engine ends at $w$ = ${ends} (<code>pair.spec_example</code>). Run as a continuation in $b$ at $\\alpha = ${lp.alpha}$, $D_0^2 = ${lp.D02}$, on a grid of step ${lp.grid_step}, the pair switches on at $b = ${f3(lp.b_switch_up)}$ (analytic $D_0^2/2 = ${f3(lp.b_on)}$) and off at $b = ${f3(lp.b_switch_down)}$ (analytic $b_{sn} = ${f3(lp.b_sn)}$) (<code>pair.loop</code>, drawn in panel (b) when $\\alpha$ and $D_0^2$ match).`;
    ui.renderMath(ctx.slot("pair-spec"));
  }

  // G3-G5 at B = 12: seed means for blocks / linear / alpha = 1, and the largest chain, hysteresis and nestedness
  const bg = c.b_grid;
  const sel = J.runs.filter(r => r.sigma === "blocks" && r.pot === "linear" && r.alpha === 1);
  const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const meanAt = (field, k) => mean(sel.map(r => r.cold[field][k]));
  const Kmean = bg.map((_, k) => meanAt("K", k)), Kmulti = bg.map((_, k) => meanAt("K_multi", k));
  const kPeak = Kmulti.indexOf(Math.max(...Kmulti));
  let sr = 0, srE = 0;
  for (const r of J.runs.filter(r => r.alpha === 1)) r.cold.span_over_range.forEach((v, k) => { if (r.cold.converged[k] && v > sr) { sr = v; srE = r.cold.span_over_range_eval[k]; } });
  const shown = bg.map((b, k) => k).filter(k => bg[k] >= 0.012 && bg[k] <= 0.36);
  const hystB = alpha => {
    const rs = J.runs.filter(r => r.sigma === "blocks" && r.pot === "linear" && r.alpha === alpha);
    const ks = bg.map((_, k) => k).filter(k => rs.some(r => r.part_diff_up_down[k] > 0));
    return ks.length ? [bg[ks[0]], bg[ks[ks.length - 1]]] : null;
  };
  const h01 = hystB(0.1), h1 = hystB(1);
  let nest = 0; for (const r of J.runs) for (const v of r.nest_cold) nest = Math.max(nest, v);
  const nSeeds = sel.length;
  ctx.slot("sweep-text").innerHTML = `The precomputed sweep at $B = ${c.B}$ and $N = ${c.N}$ (<code>fusion.json</code>) says the same with more inputs and seeds. There the inputs come in four correlated blocks, so four clusters would have been the natural answer. With random $s = ${c.s}$ attachments, linear neurons and $\\alpha = 1$, the seed-mean component count (${nSeeds} seeds) at $b$ = ${shown.map(k => f2(bg[k])).join(", ")} is ${shown.map(k => ui.fmtFixed(Kmean[k], 1)).join(", ")}. It passes through 4 without stopping there, the seed-mean number of components with more than one member peaks at ${ui.fmtFixed(Kmulti[kPeak], 1)} (at $b = ${f3(bg[kPeak])}$), and at $\\alpha = 1$ the widest component in any converged cold-start run spans ${f2(sr)} edge ranges on the training bank (${f2(srE)} on the evaluation bank). Up and down continuations disagree on the partition for $b$ from ${h01 ? `${f3(h01[0])} to ${f3(h01[1])}` : "none"} at $\\alpha = 0.1$, and from ${h1 ? `${f3(h1[0])} to ${f3(h1[1])}` : "none"} at $\\alpha = 1$ (field <code>part_diff_up_down</code>, any seed). Refinement is nested away from percolation, but near it up to ${f2(nest)} of the pairs together at one $b$ are split at the next larger $b$ (<code>nest_cold</code>, all runs).`;
  ui.renderMath(ctx.slot("g1")); ui.renderMath(ctx.slot("sweep-text"));
}
