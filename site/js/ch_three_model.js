// Chapter 1 ("three"): model side, no DOM. Presets, tours, the simulation wrapper and the per-frame analysis.
// Kept separate from ch_three.js so the tours can be checked headless with exactly the page's code
// (import this module in node and call runTour for each entry of TOURS).
import * as E from "./engine.js";

export const B = 3;
export const DT = 0.05;            // Euler step (lambda = eta = 1). ENGINE.md: 0.05 when the transient is the subject
export const LAM = 1, ETA = 1;
export const RES_TOL = 1e-6;       // stationarity tolerance for "converged" (covariance units, eta = 1)

// Input covariance presets. corr order follows the UI: rho12, rho13, rho23 (g1..g3).
export const PRESETS = [
  { id: "indep", label: "independent", v: [1, 1, 1], r: [0, 0, 0] },
  { id: "pair", label: "g₁ and g₂ correlated", v: [1, 1, 1], r: [0.6, 0, 0] },
  { id: "anti", label: "anticorrelated", v: [2 / 3, 2 / 3, 2 / 3], r: [-0.5, -0.5, -0.5] },
  { id: "loud", label: "g₁ louder", v: [2, 1, 1], r: [0, 0, 0] },
  { id: "custom", label: "custom", v: null, r: null },
];

export const DEFAULTS = { s: "indep", v1: 1, v2: 1, v3: 1, r12: 0, r13: 0, r23: 0, theta: 0.4, gamma: 1, wmin: 1e-4,
  n: 100, seed: 1, ii: 0, ib: 0.2, mode: "mf", batch: 16 };

// A tour is an exact state: DEFAULTS overridden by `params` (a preset fills v*, r*), a fresh start, optionally a
// scheduled change of theta at time `at.t`, and a playback speed. Every claim in a note was checked headless with
// runTour (all seeds quoted) and in the page; `expect` records the outcome the note describes.
export const TOURS = [
  { id: "collapse", group: "basics", label: "No pruning", params: { s: "indep", theta: 0 },
    note: "Independent inputs, $\\theta = 0$: no constant pruning drift. Every lateral synapse grows, the whole interior becomes one blob, and it settles at the centroid, $h_i \\approx (\\tfrac13, \\tfrac13, \\tfrac13)$ for every $i$.",
    expect: { K: 1, where: ["centroid"] } },
  { id: "corner", group: "basics", label: "Weak pruning", params: { s: "indep", theta: 0.2 },
    note: "$\\theta = 0.2$. Still one blob, because early on most pairs of neurons covary by more than $0.2$; but the blob does not stay in the middle: it slides, as one body, to a single corner. Which corner depends on the seed; the other two inputs get no neurons at all.",
    expect: { K: 1, where: ["corner"] } },
  { id: "corners", group: "basics", label: "Three corners", params: { s: "indep", theta: 0.4 },
    note: "$\\theta = 0.4$, above the typical initial covariance of two neurons ($\\tfrac13$ for these inputs). Lateral synapses between neurons near different corners are pruned before a blob can form, and the interior splits into three clusters, one per input. Their sizes are set by where the neurons started.",
    expect: { K: 3, where: ["corner", "corner", "corner"] } },
  { id: "pair", group: "basics", label: "A correlated pair", params: { s: "pair", theta: 0.45 },
    note: "$\\rho_{12} = 0.6$, $\\theta = 0.45$ (below $\\rho_{12}$). Two clusters: one at $g_3$ and one at the <em>midpoint</em> of the $g_1$–$g_2$ edge, $h = (\\tfrac12, \\tfrac12, 0)$. That embedding is a shared, non-negative part, not an eigenvector of $\\Sigma_g$.",
    expect: { K: 2, where: ["midpoint12", "corner3"] } },
  { id: "anti", group: "basics", label: "Anticorrelated, θ = 0", params: { s: "anti", theta: 0, seed: 5 },
    note: "$\\Sigma_g = I - \\mathbf 1\\mathbf 1^\\top/3$ (every pair correlated $-\\tfrac12$) and <em>no</em> pruning drift. Three corner clusters anyway: lateral synapses between neurons near different corners see negative covariance and stay at the floor. Not a new mechanism, though: these dynamics are <em>identical</em> to independent inputs at $\\theta = \\tfrac13$ (see the text), right on the threshold. Of seeds 1–8, seeds 1, 2, 3 and 7 give two clusters.",
    expect: { K: 3, where: ["corner", "corner", "corner"] } },
  { id: "history", group: "subtle", label: "Lower θ afterwards", params: { s: "indep", theta: 0.4 }, at: { t: 20, theta: 0.1 },
    note: "Starts as the three-corners tour; at $t = 20$, $\\theta$ drops to $0.1$ (watch the slider). The three clusters stay, because corners covary by $0 < 0.1$. A fresh start at $\\theta = 0.1$ gives one cluster: the fixed point you reach depends on the path.",
    expect: { K: 3, where: ["corner", "corner", "corner"] } },
  { id: "slide", group: "subtle", label: "Raise θ past ρ₁₂", params: { s: "pair", theta: 0.45 }, at: { t: 20, theta: 0.7 },
    note: "Starts as the correlated-pair tour; at $t = 20$, $\\theta$ rises to $0.7$, above $\\rho_{12}$, so the midpoint is no longer a stable part. The cluster does not split: its members are identical, covary by their full variance, and stay wired together. The whole cluster slides to $g_1$ or $g_2$.",
    expect: { K: 2, where: ["corner", "corner3"] } },
  { id: "seed", group: "subtle", label: "Same θ, other seed", params: { s: "indep", theta: 0.35, seed: 2 },
    note: "$\\theta = 0.35$, just above $\\tfrac13$, seed 2: two clusters, and one input gets no neurons at all. Seeds 1 and 3–8 give three clusters with the same parameters. Near the threshold even the number of clusters is set by the initial weights.",
    expect: { K: 2, where: ["corner", "corner"] } },
  { id: "loud", group: "subtle", label: "One louder input", params: { s: "loud", theta: 0.6 },
    note: "$\\mathrm{Var}\\,g_1 = 2$, the others $1$, $\\theta = 0.6$: three corner clusters, and the louder input gets about half of the neurons (seeds 1–4: 55, 55, 42 and 54 of 100).",
    expect: { K: 3, where: ["corner1", "corner", "corner"] } },
  { id: "noise", group: "subtle", label: "Minibatch noise", params: { s: "indep", theta: 0.5, mode: "mb", batch: 16, seed: 1 }, speed: 4,
    note: "Independent inputs, $\\theta = 0.5$, but each step sees the covariance of a minibatch of 16 samples instead of the exact $\\Sigma_g$. The noise does not blur clusters, since every neuron sees the same samples; now and then it makes two clusters covary by more than $\\theta$, they wire together, and merged clusters never separate. Three becomes two, then one.",
    expect: { K: 1, where: ["corner"] } },
  { id: "lateral", group: "subtle", label: "Wired-up start", params: { s: "indep", theta: 0.4, ii: 0.3 },
    note: "As the three-corners tour, but every lateral synapse starts at a random weight up to $0.3$ instead of at the floor. The interior starts as one blob and stays one, ending at a single corner.",
    expect: { K: 1, where: ["corner"] } },
];

export const tourById = id => TOURS.find(t => t.id === id) || null;

// Full parameter set for a tour (defaults + preset + tour params).
export function tourParams(t) {
  const p = { ...DEFAULTS, ...t.params };
  return withPreset(p, p.s);
}
export function withPreset(p, sid) {
  const pr = PRESETS.find(x => x.id === sid);
  if (!pr || !pr.v) return { ...p, s: "custom" };
  return { ...p, s: sid, v1: pr.v[0], v2: pr.v[1], v3: pr.v[2], r12: pr.r[0], r13: pr.r[1], r23: pr.r[2] };
}
// Which preset (if any) matches the six numbers
export function presetOf(p) {
  const close = (a, b) => Math.abs(a - b) < 1e-3;
  const hit = PRESETS.find(x => x.v && close(x.v[0], p.v1) && close(x.v[1], p.v2) && close(x.v[2], p.v3) && close(x.r[0], p.r12) && close(x.r[1], p.r13) && close(x.r[2], p.r23));
  return hit ? hit.id : "custom";
}

export function sigmaOf(p) {
  return E.cov3({ variances: [p.v1, p.v2, p.v3], corr: [p.r12, p.r13, p.r23] });
}

// ---------------------------------------------------------------- the simulation
export function createSim(p0) {
  const sim = { p: null, s: null, Sigma: null, P: null, sampler: null };

  sim.reset = p => {
    sim.p = { ...p };
    sim.s = E.createState({ N: p.n, B, seed: p.seed, iiScale: p.ii, ibScale: p.ib });
    sim.Sigma = sigmaOf(p);
    sim.P = { ...E.defaultParams(), theta: p.theta, gamma: p.gamma, wmin: p.wmin, lam: LAM, eta: ETA, dt: DT };
    sim.sampler = null;
    sim.s.resid = NaN;
  };
  // live parameter changes (no restart): theta, gamma, wmin, Sigma, mode, batch
  sim.update = p => {
    const old = sim.p; sim.p = { ...old, ...p };
    sim.P.theta = sim.p.theta; sim.P.gamma = sim.p.gamma; sim.P.wmin = sim.p.wmin;
    if (["v1", "v2", "v3", "r12", "r13", "r23"].some(k => sim.p[k] !== old[k])) { sim.Sigma = sigmaOf(sim.p); sim.sampler = null; }
  };
  sim.time = () => sim.s.t * DT;
  // advance n steps (or until the time budget in ms runs out, at least one step); returns the steps taken
  sim.advance = (n, budgetMs = Infinity) => {
    const t0 = budgetMs < Infinity ? performance.now() : 0;
    let k = 0;
    while (k < n) {
      if (sim.p.mode === "mb") {
        if (!sim.sampler) sim.sampler = E.makeSampler(sim.Sigma, B, sim.p.seed + 1000);
        E.stepStochastic(sim.s, sim.sampler, sim.P, sim.p.batch);
      } else E.step(sim.s, sim.Sigma, sim.P);
      k++;
      if (budgetMs < Infinity && performance.now() - t0 > budgetMs) break;
    }
    return k;
  };
  // Stationarity residual in covariance units. Mean-field: the step's own value (weights the last step started
  // from). Minibatch, or before the first step: the exact mean-field drift at the current weights (recomputed).
  sim.residual = () => {
    if (sim.p.mode === "mb" || !Number.isFinite(sim.s.resid)) return E.stationarity(sim.s, sim.Sigma, sim.P).r / ETA;
    return sim.s.resid / ETA;
  };
  sim.reset(p0);
  return sim;
}

// ---------------------------------------------------------------- per-frame analysis
// Everything needed to draw one moment (positions, synapses, clusters), detached from the engine buffers.
// H is the harmonic embedding of the weights the last step started from (engine convention; after `residual()`
// in minibatch mode, of the current weights), W8 the current weights as 8-bit fractions of their maximum.
export function analyse(sim, prevLabels = null) {
  const { s, Sigma } = sim, N = s.N, wmin = sim.P.wmin;
  const resid = sim.residual();
  const H = s.H;
  // dead neurons: every synapse (to inputs and laterally) at the floor; they carry no learned structure
  const floorTol = wmin * (1 + 1e-3);
  const dead = new Uint8Array(N);
  let nFloorII = 0, nII = 0, nFloorIB = 0, nIB = 0, wmax = 0;
  for (let i = 0; i < N; i++) {
    let alive = false;
    for (let b = 0; b < B; b++) { if (!s.Mib[i * B + b]) continue; const w = s.Wib[i * B + b]; nIB++; if (w <= floorTol) nFloorIB++; else alive = true; if (w > wmax) wmax = w; }
    for (let j = 0; j < N; j++) {
      if (j === i || !s.Mii[i * N + j]) continue;
      const w = s.Wii[i * N + j]; if (w > floorTol) alive = true;
      if (j > i) { nII++; if (w <= floorTol) nFloorII++; if (w > wmax) wmax = w; }
    }
    dead[i] = alive ? 0 : 1;
  }
  const nTri = N * (N - 1) / 2, W8 = new Uint8Array(nTri + N * B);
  const inv = wmax > 0 ? 255 / wmax : 0;
  let k = 0;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) W8[k++] = Math.round(s.Wii[i * N + j] * inv);
  for (let q = 0; q < N * B; q++) W8[k++] = Math.round(s.Wib[q] * inv);
  // clusters: single linkage in L1 with the cut at the largest gap (engine.clustersAuto); dead neurons split off
  const cl = E.clustersAuto(H, N, B);
  let labels = cl.labels;
  if (prevLabels && prevLabels.length === N) labels = E.relabel(prevLabels, labels, N);
  const byLab = new Map();
  for (let i = 0; i < N; i++) {
    const key = dead[i] ? "dead" : labels[i];
    let c = byLab.get(key);
    if (!c) { c = { id: key, n: 0, part: [0, 0, 0], dead: !!dead[i], members: [] }; byLab.set(key, c); }
    c.n++; c.members.push(i); for (let b = 0; b < B; b++) c.part[b] += H[i * B + b];
  }
  const all = [...byLab.values()];
  for (const c of all) { for (let b = 0; b < B; b++) c.part[b] /= c.n; c.energy = quad(c.part, Sigma); }
  const live = all.filter(c => !c.dead).sort((a, b) => b.n - a.n);
  const deadC = all.find(c => c.dead) || null;
  return {
    t: sim.time(), N, H: Float32Array.from(H.subarray(0, N * B)), W8, wmax, labels: Int32Array.from(labels), dead,
    clusters: deadC ? [...live, deadC] : live, K: live.length, nDead: deadC ? deadC.n : 0, robust: cl.robust, gap: cl.gap,
    resid, fII: nII ? nFloorII / nII : 0, fIB: nIB ? nFloorIB / nIB : 0,
    theta: sim.P.theta, gamma: sim.P.gamma, Sigma: Float64Array.from(Sigma), mode: sim.p.mode,
  };
}

export function quad(h, S) { let q = 0; for (let a = 0; a < B; a++) for (let b = 0; b < B; b++) q += h[a] * S[a * B + b] * h[b]; return q; }

// Where is a part? "corner<b>", "midpoint<ab>", "centroid" or "interior" (1-based input numbers).
export function placeOf(part, tol = 0.06) {
  const srt = part.map((x, b) => [x, b]).sort((x, y) => y[0] - x[0]);
  if (srt[0][0] > 1 - tol) return `corner${srt[0][1] + 1}`;
  if (Math.abs(srt[0][0] - 0.5) < tol && Math.abs(srt[1][0] - 0.5) < tol) return `midpoint${[srt[0][1] + 1, srt[1][1] + 1].sort().join("")}`;
  if (part.every(x => Math.abs(x - 1 / 3) < tol)) return "centroid";
  return "interior";
}

// Does an analysis match a tour's expectation? (`where` entries match by prefix, largest cluster first.)
export function matches(a, expect) {
  if (a.K !== expect.K) return false;
  const live = a.clusters.filter(c => !c.dead).map(c => placeOf(c.part));
  return expect.where.every((w, i) => live[i] && live[i].startsWith(w));
}

// Headless run of a tour (for verification): returns the final analysis and a K time series.
export function runTour(t, { tMax = 200, every = 1, seed = null } = {}) {
  const p = tourParams(t); if (seed != null) p.seed = seed;
  const sim = createSim(p);
  let fired = !t.at, firedAt = 0, prev = null; const Kseq = [];
  const steps = Math.round(tMax / DT), ev = Math.max(1, Math.round(every / DT));
  for (let k = 0; k < steps; k++) {
    sim.advance(1);
    if (!fired && sim.time() >= t.at.t) { sim.update({ theta: t.at.theta }); fired = true; firedAt = k; }
    if (k % ev === 0) { const a = analyse(sim, prev); prev = a.labels; Kseq.push(a.K); }
    if (p.mode === "mf" && fired && k > firedAt + 2 && sim.s.resid / ETA < RES_TOL) break;
  }
  const a = analyse(sim, prev);
  return { a, Kseq, t: sim.time() };
}
