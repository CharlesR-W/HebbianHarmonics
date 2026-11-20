// Chapter "readout" (Reading the weights). One linear network is trained in the page (distance rule, B = 12 block
// inputs) and every live figure reads it:
//   idea   W -> H -> recovered P C P next to the true one                         (fig 1, the N control lives here)
//   law    every pair: w against the measured D^2, with the fixed-point law          (fig 2)
//   film   recovered against true entries at several training steps (+ heat-map alternative)
//   blind  C = shared part + P C P, with a live retraining check                    (fig 5)
//   rule   distance rule vs covariance rule (data/readout/cov_rule.json) + sensitivity to the assumptions
// The figure against N is in ch_readout_n.js; the estimator in ch_readout_math.js.
import * as ui from "./ui.js";
import { heatmap } from "./charts.js";
import { P } from "./palette.js";
import * as O from "./onsite_engine.js";
import * as R from "./ch_readout_math.js";
import { lawPlot, matPanel } from "./ch_readout_draw.js";
import { nFigure } from "./ch_readout_n.js";

const B = 12, NC = B - 1, ALPHA = 1, DT = 0.5, S = 256, STOP = 1e-9, BRULE = 0.05, SEED = 1;
const NS = [40, 80, 160];
const SNAP = [1, 2, 4, 8, 16];
const GLAB = Array.from({ length: B }, (_, k) => "g" + String(k + 1).split("").map(c => "₀₁₂₃₄₅₆₇₈₉"[c]).join(""));
const SIGMA = O.covBlocks({ sizes: [3, 3, 3, 3], within: 0.8, between: 0, variances: [2, 1, 0.5, 0.25] }).Sigma;
const Q = R.contrastBasis(B), IDX = R.svecIndex(NC);
const fmtE = x => ui.fmt(x, 2);
const tick = () => new Promise(r => setTimeout(r, 0));

// ---------------------------------------------------------------------------------------------- helpers
function chatOf(G) {
  const C = new Float64Array(B * B);
  for (let s = 0; s < S; s++) for (let a = 0; a < B; a++) { const x = G[s * B + a]; for (let c = 0; c < B; c++) C[a * B + c] += x * G[s * B + c]; }
  for (let k = 0; k < B * B; k++) C[k] /= S;
  return C;
}
const toB = Y => R.fromContrast(Y, Q, B, NC);                 // contrast coordinates -> B x B (P C P)
const project = C => toB(R.toContrast(C, Q, B, NC));
const absMax = (...Ms) => { let m = 0; for (const M of Ms) for (const v of M) m = Math.max(m, Math.abs(v)); return m; };
const nice = x => +(x).toPrecision(2);
function readW(run, W, extra = {}) {
  return R.readout({ W, A: run.A, N: run.N, B, b: BRULE, alpha: ALPHA, Chat: run.Chat, Sigma: SIGMA, Q, idx: IDX, hinge: false, ...extra });
}
// neuron order: by connected component (components ordered by the input their members listen to most), then by
// each neuron's own strongest input
function neuronOrder(W, H, N) {
  const { labels, K } = O.components(W, N);
  const argmax = i => { let bi = 0; for (let a = 1; a < B; a++) if (H[i * B + a] > H[i * B + bi]) bi = a; return bi; };
  const comps = Array.from({ length: K }, () => ({ members: [], mean: new Float64Array(B) }));
  for (let i = 0; i < N; i++) { const c = comps[labels[i]]; c.members.push(i); for (let a = 0; a < B; a++) c.mean[a] += H[i * B + a]; }
  comps.forEach(c => { let bi = 0; for (let a = 1; a < B; a++) if (c.mean[a] > c.mean[bi]) bi = a; c.key = bi; c.lead = c.mean[bi] / c.members.length; });
  comps.sort((x, y) => x.key - y.key || y.members.length - x.members.length || y.lead - x.lead);
  const order = [];
  comps.forEach(c => { c.members.sort((i, j) => argmax(i) - argmax(j) || H[j * B + argmax(j)] - H[i * B + argmax(i)]); order.push(...c.members); });
  return { order, K };
}

// ---------------------------------------------------------------------------------------------- training
async function train(N, onProgress) {
  const net = O.makeNet({ N, B, rng: O.makeRng(SEED), s: 3 });
  const G = O.sampleInputs(SIGMA, B, S, O.makeRng(1000 + SEED));
  const Chat = chatOf(G);
  const sim = O.createSim({ net, G, S, b: BRULE, alpha: ALPHA, dt: DT });
  const snaps = [];
  let t0 = performance.now();
  while (sim.steps < 5000) {
    sim.step(1);
    if (SNAP.includes(sim.steps)) snaps.push({ step: sim.steps, W: Float64Array.from(sim.net.W), drift: sim.residual });
    if (sim.lastChange < STOP) break;
    if (performance.now() - t0 > 40) { onProgress?.(sim.steps); await tick(); t0 = performance.now(); }
  }
  const W = Float64Array.from(sim.net.W);
  const run = { N, A: net.A, G, Chat, W, D2: Float64Array.from(sim.D2), steps: sim.steps, residual: sim.residual, eqResidual: sim.eqResidual };
  run.H = O.harmonic(O.makeNetFrom({ A: net.A, N, B, W }));
  run.ro = readW(run, W);
  run.truth = toB(run.ro.T1);
  run.rec = toB(run.ro.Yr);
  run.snaps = snaps.filter(s => s.step < sim.steps);
  return run;
}

// ---------------------------------------------------------------------------------------------- init
export async function init(panel, ctx) {
  const covP = fetch("data/readout/cov_rule.json").then(r => { if (!r.ok) throw new Error(`cov_rule.json: ${r.status}`); return r.json(); });
  const sweepP = fetch("data/sweeps2/readout.json").then(r => { if (!r.ok) throw new Error(`readout.json: ${r.status}`); return r.json(); });
  const nP = nFigure(ctx);

  let N = NS.includes(ctx.param("n", 80)) ? ctx.param("n", 80) : 80;
  let job = 0, current = null;
  const figs = [];
  const idea = ideaFigure(ctx, n => retrain(n));
  figs.push(idea, lawFigure(ctx), filmFigure(ctx), blindFigure(ctx));
  const rule = ruleFigure(ctx, covP);
  figs.push(rule);
  ctx.article.querySelectorAll("[data-v=b]").forEach(e => { e.textContent = String(BRULE); });

  async function retrain(n) {
    N = n; const my = ++job;
    ctx.setParams({ n: n === 80 ? null : n });
    idea.busy(true, `training ${n} neurons…`);
    const run = await train(n, st => { if (my === job) idea.busy(true, `training ${n} neurons… step ${st}`); });
    if (my !== job) return;
    current = run;
    idea.busy(false);
    for (const f of figs) { f.update(run); await tick(); if (my !== job) return; }
    for (const f of figs) { if (f.after) { await f.after(run, () => my !== job); if (my !== job) return; } }
    await sensitivityText(ctx, run, () => my !== job);
  }
  ctx.onParams(p => { const n = +(p.get("n") || 80); if (NS.includes(n) && n !== N) { idea.setN(n); retrain(n); } });
  idea.setN(N);
  const first = retrain(N);
  sweepText(ctx, await sweepP);
  await Promise.all([first, nP, covP]);
}

// ---------------------------------------------------------------------------------------------- fig 1: the idea
function ideaFigure(ctx, onN) {
  const slot = ctx.slot("idea"); slot.replaceChildren();
  const nSeg = ui.segmented({ label: "neurons $N$", options: NS.map(v => ({ value: v, label: String(v) })), value: 80, onChange: v => onN(v) });
  const status = ui.h("span", { class: "ro-status" });
  slot.append(ui.h("div", { class: "ro-controls" }, nSeg.el, status));
  const row = ui.h("div", { class: "ro-row" }); slot.append(row);
  const arrow = (label) => ui.h("div", { class: "ro-arrow", html: `<span>${label}</span><b>→</b>` });
  const hW = matPanel(row, { kicker: "learned", title: "weights $W$", cls: "ro-w", map: "ink", range: [0, BRULE / ALPHA], size: 280, xlabel: "neuron j", ylabel: "neuron i", zlabel: "weight $w_{ij}$", ticks: [0, BRULE / ALPHA],
    tooltip: (i, j, v) => ui.tipRows(null, [{ name: "neurons", value: `${st.order[i] + 1}, ${st.order[j] + 1}` }, { name: "weight", value: ui.fmt(v, 3) }]) });
  row.append(arrow("+ fixed input<br>wiring $A$"));
  const hH = matPanel(row, { kicker: "computed", title: "mixtures $H$", cls: "ro-h", map: "seq", range: [0, 1], size: 132, square: false, cellH: 3.5, colLabels: GLAB, xlabel: "input", zlabel: "$h_{ib}$", ticks: [0, 1],
    tooltip: (i, b, v) => ui.tipRows(`neuron ${st.order[i] + 1}`, [{ name: `share of ${GLAB[b]}`, value: ui.fmt(v, 3) }]) });
  row.append(arrow("+ rule (1),<br>solve (3)"));
  const covOpts = { map: "div", size: 230, rowLabels: GLAB, colLabels: GLAB, zlabel: "second moment, contrast part" };
  const hRec = matPanel(row, { kicker: "inferred", title: "recovered", cls: "ro-c", ...covOpts, tooltip: (i, j, v) => covTip(i, j, v) });
  row.append(arrow("compare"));
  const hTrue = matPanel(row, { kicker: "actual", title: "true, of the training inputs", cls: "ro-c", ...covOpts, tooltip: (i, j, v) => covTip(i, j, v) });
  row.append(ui.h("p", { class: "ro-note", text: "These two maps compare differences between inputs: shifts shared by all inputs are removed (the contrast part, explained below)." }));
  const stat = ui.h("p", { class: "ro-stat" }); slot.append(stat);
  ui.renderMath(slot);
  let st = { order: [] }, run = null;
  function covTip(i, j, v) {
    if (!run) return null;
    return ui.tipRows(`${GLAB[i]}, ${GLAB[j]}`, [{ name: "recovered", value: ui.fmt(run.rec[i * B + j], 4) }, { name: "true", value: ui.fmt(run.truth[i * B + j], 4) }]);
  }
  return {
    setN(n) { nSeg.set(n); },
    busy(on, text = "") { status.textContent = on ? text : ""; slot.classList.toggle("ro-busy", on); },
    update(r) {
      run = r;
      const N = r.N, { order } = neuronOrder(r.W, r.H, N); st.order = order;
      const Wv = new Float32Array(N * N), Hv = new Float32Array(N * B);
      for (let a = 0; a < N; a++) { for (let c = 0; c < N; c++) Wv[a * N + c] = r.W[order[a] * N + order[c]]; for (let k = 0; k < B; k++) Hv[a * B + k] = r.H[order[a] * B + k]; }
      hW.update({ values: Wv, rows: N, cols: N });
      hH.update({ values: Hv, rows: N, cols: B, cellH: 280 / N });
      const m = nice(1.05 * absMax(r.truth));
      hRec.update({ values: Float32Array.from(r.rec), rows: B, cols: B, range: [-m, m] });
      hTrue.update({ values: Float32Array.from(r.truth), rows: B, cols: B, range: [-m, m] });
      const ro = r.ro, err = ro.sc.ridge.Chat.rel;
      hRec.setTitle(`recovered <span class="ro-err">error ${fmtE(err)}</span>`);
      stat.innerHTML = `${N} neurons · <b>${ui.fmtInt(ro.m)}</b> live synapses, of ${ui.fmtInt(N * (N - 1) / 2)} pairs · <b>${ro.rank8}</b> independent equations for the 66 unknowns · `
        + `error of the recovered second moment <b class="ro-big">${fmtE(err)}</b>`;
      ctx.article.querySelectorAll("[data-v=samp]").forEach(e => { e.textContent = ui.fmtPct(ro.errSampling, 1); });
    },
  };
}

// ---------------------------------------------------------------------------------------------- fig 2: the law
function lawFigure(ctx) {
  const slot = ctx.slot("law"); slot.replaceChildren();
  const box = ui.h("div"), stat = ui.h("p", { class: "ro-stat" }); slot.append(box, stat);
  const lp = lawPlot(box);
  return {
    update(r) {
      const N = r.N, n = N * (N - 1) / 2, D2 = new Float64Array(n), w = new Float64Array(n), live = new Uint8Array(n), pi = new Int32Array(n), pj = new Int32Array(n);
      let k = 0;
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++, k++) { D2[k] = r.D2[i * N + j]; w[k] = r.W[i * N + j]; live[k] = w[k] > 1e-9 ? 1 : 0; pi[k] = i; pj[k] = j; }
      lp.update({ D2, w, live, pi, pj, b: BRULE, alpha: ALPHA });
      let dev = 0, margin = Infinity, nl = 0;
      for (let q = 0; q < n; q++) { if (live[q]) { nl++; dev = Math.max(dev, Math.abs(w[q] - (BRULE - D2[q] / 2) / ALPHA)); } else margin = Math.min(margin, D2[q] - 2 * BRULE); }
      stat.innerHTML = `${ui.fmtInt(nl)} live synapses, largest vertical weight deviation from the line <b>${ui.fmt(dev, 2)}</b> · ${ui.fmtInt(n - nl)} pruned pairs, the closest one at $D^2 = 2b + ${ui.fmt(margin, 2)}$`;
      ui.renderMath(stat);
    },
  };
}

// ---------------------------------------------------------------------------------------------- fig 3: filmstrip
function filmFrames(r) {
  const frames = r.snaps.map(s => {
    const ro = readW(r, s.W);
    return { step: s.step, rec: toB(ro.Yr), err: ro.sc.ridge.Chat.rel, drift: s.drift, m: ro.m };
  });
  frames.push({ step: r.steps, rec: r.rec, err: r.ro.sc.ridge.Chat.rel, drift: r.residual, m: r.ro.m, final: true });
  return frames;
}
let framesCache = null;
const framesOf = r => (framesCache && framesCache.run === r ? framesCache.frames : (framesCache = { run: r, frames: filmFrames(r) }).frames);

function filmFigure(ctx) {
  const slot = ctx.slot("film"); slot.replaceChildren();
  const row = ui.h("div", { class: "ro-film" }); slot.append(row);
  return {
    update(r) {
      row.replaceChildren();
      const fr = framesOf(r), m = nice(1.05 * absMax(r.truth));
      const cells = [...fr.map(f => ({ M: f.rec, head: f.final ? `step ${f.step} <span class="ro-mute">(converged)</span>` : `step ${f.step}`, foot: `error <b>${fmtE(f.err)}</b><br><span class="ro-mute">drift ${fmtE(f.drift)}</span>` })),
        { M: r.truth, head: "true", foot: "&nbsp;<br>&nbsp;", last: true }];
      cells.forEach(c => {
        const cell = ui.h("div", { class: "ro-film-cell" + (c.last ? " ro-film-true" : "") });
        cell.append(ui.h("div", { class: "ro-film-head", html: c.head }));
        const box = ui.h("div", { style: { width: c.last ? "140px" : "118px" } }); cell.append(box);
        cell.append(ui.h("div", { class: "ro-film-foot", html: c.foot }));
        row.append(cell);
        heatmap(box, { values: Float32Array.from(c.M), rows: B, cols: B, map: "div", range: [-m, m], size: c.last ? 116 : 116, colorbar: !!c.last, zlabel: "second moment, contrast part", tooltip: (i, j, v) => ui.tipRows(`${GLAB[i]}, ${GLAB[j]}`, [{ name: "value", value: ui.fmt(v, 3) }]) });
      });
    },
  };
}

// ---------------------------------------------------------------------------------------------- fig 5: blind part
function blindFigure(ctx) {
  const slot = ctx.slot("blind"); slot.replaceChildren();
  const row = ui.h("div", { class: "ro-row3" }); slot.append(row);
  const opts = { map: "div", size: 230, rowLabels: GLAB, colLabels: GLAB, zlabel: "second moment" };
  const hC = matPanel(row, { kicker: "everything", title: "second moment of the inputs $\\hat C$", ...opts });
  row.append(ui.h("div", { class: "ro-op", text: "=" }));
  const hS = matPanel(row, { kicker: "invisible", title: "part shared by all inputs", ...opts });
  row.append(ui.h("div", { class: "ro-op", text: "+" }));
  const hP = matPanel(row, { kicker: "recorded", title: "the rest, $P\\hat C P$", ...opts });
  const stat = ui.h("p", { class: "ro-stat" }); slot.append(stat);
  return {
    update(r) {
      const C = r.Chat, PC = project(C), Sh = C.map((v, k) => v - PC[k]);
      const m = nice(1.05 * absMax(C, Sh, PC));
      hC.update({ values: Float32Array.from(C), rows: B, cols: B, range: [-m, m] });
      hS.update({ values: Float32Array.from(Sh), rows: B, cols: B, range: [-m, m] });
      hP.update({ values: Float32Array.from(PC), rows: B, cols: B, range: [-m, m] });
      stat.innerHTML = `<span class="ro-mute">Check: retraining the same ${r.N} neurons on the same inputs plus one random signal added to all twelve…</span>`;
    },
    // retrain with g -> g + z 1 (z a fresh random signal) and compare the weights
    async after(r, stale) {
      const net = O.makeNet({ N: r.N, B, rng: O.makeRng(SEED), s: 3 });
      const G2 = Float64Array.from(r.G), z = O.makeRng(77);
      for (let s = 0; s < S; s++) { const v = 1.5 * z.normal(); for (let a = 0; a < B; a++) G2[s * B + a] += v; }
      const sim = O.createSim({ net, G: G2, S, b: BRULE, alpha: ALPHA, dt: DT });
      let t0 = performance.now();
      while (sim.steps < 5000) { sim.step(1); if (sim.lastChange < STOP) break; if (performance.now() - t0 > 30) { await tick(); if (stale()) return; t0 = performance.now(); } }
      let d = 0, wmax = 0; for (let k = 0; k < r.W.length; k++) { d = Math.max(d, Math.abs(sim.net.W[k] - r.W[k])); wmax = Math.max(wmax, r.W[k]); }
      const C2 = chatOf(G2); let dc = 0; for (let k = 0; k < B * B; k++) dc = Math.max(dc, Math.abs(C2[k] - r.Chat[k]));
      stat.innerHTML = `Check: retraining the same ${r.N} neurons on the same inputs plus one random signal added to all twelve changes the inputs' second moment by up to <b>${ui.fmt(dc, 2)}</b> per entry, `
        + `but no weight by more than <b>${ui.fmt(d, 2)}</b> (the largest weight is ${ui.fmt(wmax, 2)}).`;
    },
  };
}

// ---------------------------------------------------------------------------------------------- fig 6: the rule
function ruleFigure(ctx, covP) {
  const slot = ctx.slot("rule"); slot.replaceChildren();
  const grid = ui.h("div", { class: "ro-rule" }); slot.append(grid);
  const opts = { map: "div", size: 200, rowLabels: GLAB, colLabels: GLAB, zlabel: "second moment, contrast part" };
  const rowHead = (title, setup) => {
    const d = ui.h("div", { class: "ro-rule-head" }, ui.h("div", { class: "ro-title", html: title }), ui.h("div", { class: "ro-rule-setup", html: setup }), ui.h("div", { class: "ro-rule-sub" }));
    grid.append(d); ui.renderMath(d); return d;
  };
  const hD = rowHead("Distance rule (1)", "");
  const dRec = matPanel(grid, { kicker: "recovered", title: "", ...opts });
  const dTrue = matPanel(grid, { kicker: "true", title: "", ...opts });
  const hC = rowHead("Covariance rule with pruning", "");
  const cRec = matPanel(grid, { kicker: "recovered", title: "", ...opts });
  const cTrue = matPanel(grid, { kicker: "true", title: "", ...opts });
  const pct = v => ui.fmtPct(v, 1);
  covP.then(d => {
    const s = d.summary, cfg = d.meta.config;
    const put = (k, v) => ctx.article.querySelectorAll(`[data-cv=${k}]`).forEach(e => { e.textContent = v; });
    const span = ([lo, hi], f = String) => (f(lo) === f(hi) ? f(lo) : `${f(lo)}–${f(hi)}`);
    put("theta", String(cfg.theta)); put("N", String(cfg.N)); put("seeds", String(cfg.seeds.length));
    put("rank", span(s.rank)); put("K", span(s.K)); put("rel", span(s.relerr_PCP, pct));
    hC.children[1].innerHTML = `${cfg.N} neurons; every synapse plastic, input synapses included; θ = ${cfg.theta}; trained on the population covariance`;
  }).catch(e => { console.warn(e); });
  return {
    async update(r) {
      const d = await covP;
      const PS = d.PSigmaP.flat(), PC = d.example.PCP.flat();
      const m = nice(1.05 * absMax(r.truth, PS));
      dRec.update({ values: Float32Array.from(r.rec), rows: B, cols: B, range: [-m, m] });
      dTrue.update({ values: Float32Array.from(r.truth), rows: B, cols: B, range: [-m, m] });
      cRec.update({ values: Float32Array.from(PC), rows: B, cols: B, range: [-m, m] });
      cTrue.update({ values: Float32Array.from(PS), rows: B, cols: B, range: [-m, m] });
      const ex = d.runs.find(u => u.seed === d.example.seed);
      hD.children[1].innerHTML = `the network above: ${r.N} neurons, each wired to 3 inputs by fixed wires; trained on the 256 samples`;
      hD.lastChild.innerHTML = `error <b>${fmtE(r.ro.sc.ridge.Chat.rel)}</b> · equations span ${r.ro.rank8} of the 66 contrasts`;
      hC.lastChild.innerHTML = `error <b>${pct(ex.relerr_PCP)}</b> · equations span ${ex.rank} independent combinations of the 78 numbers; ${ex.K} groups`;
    },
  };
}

// the sensitivity sentence: a fixed 80-neuron example (the default network), computed once
let sensDone = false;
async function sensitivityText(ctx, r, stale) {
  if (sensDone) return;
  const r80 = r.N === 80 ? r : await train(80);
  if (stale()) return;
  const e = 0.1, put = (k, v) => ctx.article.querySelectorAll(`[data-sens=${k}]`).forEach(x => { x.textContent = v; });
  put("b", ui.fmtPct(readW(r80, r80.W, { b: BRULE * (1 + e) }).sc.ridge.Chat.rel)); await tick();
  put("a", ui.fmtPct(readW(r80, r80.W, { alpha: ALPHA * (1 + e) }).sc.ridge.Chat.rel)); await tick();
  put("n", ui.fmtPct(readW(r80, r80.W, { noise: e }).sc.ridge.Chat.rel));
  const { labels, K } = O.components(r80.W, 80), sz = new Array(K).fill(0); labels.forEach(l => { sz[l]++; });
  put("K", `${K} pieces, the largest of ${Math.max(...sz)} neurons`);
  sensDone = true;
}

// numbers for the noise sentence in fig 6's details (precomputed sweep, N = 80, blocks, no leak)
function sweepText(ctx, d) {
  const arms = d.lin.filter(e => e.sigma === "blocks" && e.N === 80 && (e.leak ?? "0") === "0");
  const at = b0 => arms.map(e => e.runs.find(x => Math.abs(x.b - b0) / b0 < 0.02)?.["noise0.01_ridge_psd_vs_Chat_cos"]).filter(Number.isFinite);
  const rng = v => { if (!v.length) return "n/a"; const lo = ui.fmtFixed(Math.min(...v), 3), hi = ui.fmtFixed(Math.max(...v), 3); return lo === hi ? lo : `${lo}–${hi}`; };
  const put = (k, v) => ctx.article.querySelectorAll(`[data-sv=${k}]`).forEach(e => { e.textContent = v; });
  put("good", rng(at(0.0829))); put("bad", rng(at(0.4361)));
  const ks = b0 => arms.map(e => e.runs.find(x => Math.abs(x.b - b0) / b0 < 0.02)?.K).filter(Number.isFinite);
  const kText = v => (v.every(k => k === 1) ? "one piece each" : `${v.join(", ")} pieces`);
  put("goodK", ks(0.0829).join(", ")); put("badK", kText(ks(0.4361)));
}
