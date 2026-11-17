// Chapter 1 ("three"): model side, no DOM. Presets, tours, the simulation wrapper and the per-frame analysis.
// Kept separate from ch_three.js so the tours can be checked headless with exactly the page's code
// (import this module in node and call runTour for each entry of TOURS).
import * as E from "./engine.js";

export const B = 3;
export const DT = 0.05;            // Euler step (lambda = eta = 1). ENGINE.md: 0.05 when the transient is the subject
export const LAM = 1, ETA = 1;
export const RES_TOL = 1e-5;       // stationarity tolerance for "converged" (docs/findings.md conventions)
export const JIT_SEED = 3000;      // private-jitter RNG stream offset (seed + JIT_SEED)

// Input covariance presets. corr order follows the UI: rho12, rho13, rho23 (g1..g3).
export const PRESETS = [
  { id: "indep", label: "independent, equal variances", v: [1, 1, 1], r: [0, 0, 0] },
  { id: "pair", label: "g₁ and g₂ correlated (0.6)", v: [1, 1, 1], r: [0.6, 0, 0] },
  { id: "pos", label: "all positively correlated (0.3)", v: [1, 1, 1], r: [0.3, 0.3, 0.3] },
  { id: "anti", label: "anticorrelated (I − 11ᵀ/3)", v: [2 / 3, 2 / 3, 2 / 3], r: [-0.5, -0.5, -0.5] },
  { id: "tilt", label: "unequal variances 1, 1.5, 2.5", v: [1, 1.5, 2.5], r: [0, 0, 0] },
  { id: "custom", label: "custom", v: null, r: null },
];

// Defaults follow docs/findings.md ("What the explorer should let the reader discover").
export const DEFAULTS = { s: "indep", v1: 1, v2: 1, v3: 1, r12: 0, r13: 0, r23: 0, theta: 0.4, gamma: 1, wmin: 1e-4,
  n: 100, seed: 0, ii: 0.03, ib: 0.2, mode: "mf", batch: 16, jit: 0 };

// A tour is an exact state: DEFAULTS overridden by `params` (a preset fills v*, r*), a fresh start, optionally a
// scheduled change of theta at time `at.t`, and a playback speed. The first twelve are docs/findings.md's verified
// Chapter 1 tours (site/data/sweeps/tours.json, same ids and parameters, seed 0 of the three verified there); the
// last three are this chapter's own, checked with runTour. `expect` records the outcome the note describes.
const F = { n: 90, seed: 0, ii: 0, ib: 0.2, wmin: 1e-4, gamma: 1 };     // tours.json common parameters
export const TOURS = [
  { id: "neutral-identity", group: "basics", label: "No pruning", params: { ...F, s: "indep", theta: 0 },
    note: "Independent inputs and $\\theta = 0$. For a lone neuron every point of the triangle is then a fixed point; lateral coupling makes consensus instead: all 90 neurons end on one point, the population's starting mean, near but not at the centroid.",
    expect: { K: 1, where: [""] } },
  { id: "corners-collective", group: "basics", label: "θ below κ₀", params: { ...F, s: "indep", theta: 0.2 },
    note: "$\\theta = 0.2$: the only stable parts are now the three corners. But $\\theta$ is below the collective threshold $\\gamma\\kappa_0 = \\tfrac13$, so lateral synapses grow first, the population binds into one blob, and it lands on <em>one</em> corner together.",
    expect: { K: 1, where: ["corner"] } },
  { id: "three-corners", group: "basics", label: "Three corners", params: { ...F, s: "indep", theta: 0.4 },
    note: "$\\theta = 0.4 &gt; \\gamma\\kappa_0$: neurons choose independently, and the interior splits into one cluster per corner. The sizes (40, 30, 20 for this seed) come from where the neurons started: any split would be a fixed point.",
    expect: { K: 3, where: ["corner", "corner", "corner"] } },
  { id: "edge-midpoint", group: "basics", label: "A correlated pair", params: { ...F, s: "pair", theta: 0.5 },
    note: "$\\rho_{12} = 0.6$, $\\theta = 0.5$. Two clusters: one at $g_3$, one at the <em>midpoint</em> of the $g_1$–$g_2$ edge, $h = (\\tfrac12, \\tfrac12, 0)$: a shared non-negative part, not an eigenvector of $\\Sigma_g$.",
    expect: { K: 2, where: ["midpoint12", "corner3"] } },
  { id: "midpoint-gated", group: "basics", label: "θ above ρ₁₂", params: { ...F, s: "pair", theta: 0.65 },
    note: "$\\theta = 0.65 &gt; \\rho_{12}$: the midpoint is no longer a stable part, but the pair's population is still bound (its self-covariance $0.8 &gt; \\theta/\\gamma$) and picks <em>one</em> of $g_1$, $g_2$ together. The other gets nothing.",
    expect: { K: 2, where: ["corner", "corner3"] } },
  { id: "midpoint-splits", group: "basics", label: "… with γ = ½", params: { ...F, s: "pair", theta: 0.65, gamma: 0.5 },
    note: "The same, with lateral gain $\\gamma = 0.5$: the lateral threshold $\\theta/\\gamma = 1.3$ exceeds every covariance, nothing binds, and all three corners are populated.",
    expect: { K: 3, where: ["corner", "corner", "corner"] } },
  { id: "consensus", group: "inputs", label: "All correlated", params: { ...F, s: "pos", theta: 0.1 },
    note: "Every pair correlated $0.3$ and $\\theta = 0.1$, below every covariance: $\\Sigma_g - \\theta\\mathbf 1\\mathbf 1^\\top$ is positive, so there is exactly one part, its full-support Perron vector, here the centroid.",
    expect: { K: 1, where: ["centroid"] } },
  { id: "anticorrelated", group: "inputs", label: "Anticorrelated", params: { ...F, s: "anti", theta: 0.02 },
    note: "$\\Sigma_g = I - \\mathbf 1\\mathbf 1^\\top/3$ (every pair correlated $-\\tfrac12$) and almost no pruning drift, $\\theta = 0.02$: three corner clusters. Negative covariances prune by themselves; the text shows this is the same mechanism as drift. (Close to the threshold: in this page's random start, seed 2 gives two clusters.)",
    expect: { K: 3, where: ["corner", "corner", "corner"] } },
  { id: "winner-takes-all", group: "inputs", label: "Loudest wins", params: { ...F, s: "tilt", theta: 0.2 },
    note: "Variances $1, 1.5, 2.5$, $\\theta = 0.2$, below the collective threshold: the bound population goes, all 90 neurons, to the loudest input $g_3$.",
    expect: { K: 1, where: ["corner3"] } },
  { id: "independent-tilt", group: "inputs", label: "… with γ = 0", params: { ...F, s: "tilt", theta: 0.2, gamma: 0 },
    note: "The same without lateral plasticity ($\\gamma = 0$): neurons choose independently, every input is populated, and louder inputs get far more neurons (66, 18 and 6 for this seed: a steep power of the input's energy).",
    expect: { K: 3, where: ["corner3", "corner2", "corner1"] } },
  { id: "partial-extinction", group: "inputs", label: "θ above a variance", params: { ...F, s: "tilt", theta: 1.2 },
    note: "$\\theta = 1.2$ exceeds $\\mathrm{Var}\\,g_1 = 1$: no neuron can covary with $g_1$ by more than $\\theta$, so that corner is not a part and gets nothing. The others share the population (67 and 23 for this seed).",
    expect: { K: 2, where: ["corner3", "corner2"] } },
  { id: "floor-too-high", group: "inputs", label: "Floor too high", params: { ...F, s: "indep", theta: 0.6, wmin: 1e-2 },
    note: "$w_{\\min} = 10^{-2}$, so the floor conductance $\\omega = N w_{\\min} = 0.9$ is comparable to a cluster's input conductance. Every neuron leaks to the centroid before its embedding can concentrate, and at $\\theta = 0.6$ the whole network dies.",
    expect: { K: 0, where: [] } },
  { id: "history", group: "history", label: "Lower θ afterwards", params: { ...F, s: "indep", theta: 0.4 }, at: { t: 20, theta: 0.1 },
    note: "Starts as the three-corners tour; at $t = 20$, $\\theta$ drops to $0.1$ (watch the slider). The three clusters stay, because corners covary by $0 &lt; \\theta$, while a fresh start at $\\theta = 0.1$ gives one cluster. Which fixed point you reach depends on the path.",
    expect: { K: 3, where: ["corner", "corner", "corner"] } },
  { id: "noise", group: "history", label: "Minibatch noise", params: { ...F, s: "indep", theta: 0.5, mode: "mb", batch: 16, n: 100, seed: 1 }, speed: 4,
    note: "Independent inputs, $\\theta = 0.5$, but each step sees the covariance of a minibatch of 16 samples instead of the exact $\\Sigma_g$. Every neuron sees the same samples, so clusters stay points; now and then the noise pushes two clusters' covariance above $\\theta$, they wire together, and merged clusters never separate. Three becomes two, then one.",
    expect: { K: 1, where: ["corner"] } },
  { id: "wired", group: "history", label: "Wired-up start", params: { ...F, s: "indep", theta: 0.4, ii: 0.3 },
    note: "As the three-corners tour, but every lateral synapse starts at a random weight up to $0.3$ instead of at the floor. The interior starts as one blob, stays one, and lands on a single corner: what matters is initial lateral conductance relative to input conductance.",
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
    sim.sampler = null; sim.jrng = E.makeRng(p.seed + JIT_SEED);
    sim.s.resid = NaN;
  };
  // private multiplicative jitter (docs/findings.md F8): w <- w exp(sigma xi), independently per synapse, each step.
  // It breaks exact symmetries (clones, symmetric states) that the deterministic dynamics can never leave.
  function jitter(sig) {
    const { N, Wii, Wib, Mii, Mib } = sim.s, r = sim.jrng;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (Mii[i * N + j]) { const w = Wii[i * N + j] * Math.exp(sig * r.normal()); Wii[i * N + j] = w; Wii[j * N + i] = w; }
    for (let k = 0; k < N * B; k++) if (Mib[k]) Wib[k] *= Math.exp(sig * r.normal());
  }
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
      if (sim.p.jit > 0) jitter(sim.p.jit);
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
  // c = a member's summed input conductance (lambda/eta = 1); at a fixed point c = (h'Σh − θ)/|h|² (findings F1)
  for (const c of all) { let sw = 0; for (const i of c.members) for (let b = 0; b < B; b++) sw += s.Wib[i * B + b]; c.c = sw / c.n; }
  // K counts populated clusters, >= max(2, 2% N) members (findings.md convention); smaller ones are stragglers
  const minPop = Math.max(2, Math.ceil(0.02 * N));
  const live = all.filter(c => !c.dead).sort((a, b) => b.n - a.n);
  for (const c of live) c.small = c.n < minPop;
  const deadC = all.find(c => c.dead) || null;
  return {
    t: sim.time(), N, H: Float32Array.from(H.subarray(0, N * B)), W8, wmax, labels: Int32Array.from(labels), dead,
    clusters: deadC ? [...live, deadC] : live, K: live.filter(c => !c.small).length, nSmall: live.filter(c => c.small).reduce((a, c) => a + c.n, 0), nDead: deadC ? deadC.n : 0, robust: cl.robust, gap: cl.gap,
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
  const live = a.clusters.filter(c => !c.dead && !c.small).map(c => placeOf(c.part));
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
    if (p.mode === "mf" && !p.jit && fired && k > firedAt + 2 && sim.s.resid / ETA < RES_TOL) break;
  }
  const a = analyse(sim, prev);
  return { a, Kseq, t: sim.time() };
}
