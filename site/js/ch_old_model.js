// Chapter "old" (the v1 covariance rule, condensed): model side, no DOM. Presets, tours, the simulation wrapper
// and the per-frame analysis, on the v1 engine (engine.js, parity-tested against hebb/core.py). Kept separate
// from ch_old.js so the tours can be checked headless with exactly the page's code (import this module in node
// and call runTour for each entry of TOURS).
import * as E from "./engine.js";

export const B = 3;
export const DT = 0.05;            // Euler step (lambda = eta = 1), as in the v1 sweeps
export const LAM = 1, ETA = 1;
export const RES_TOL = 1e-5;       // stationarity tolerance for "converged" (docs/v1/findings.md conventions)

// Input covariance presets. corr order follows the UI: rho12, rho13, rho23 (g1..g3).
export const PRESETS = [
  { id: "indep", label: "independent, equal variances", v: [1, 1, 1], r: [0, 0, 0] },
  { id: "pair", label: "g₁ and g₂ correlated (0.6)", v: [1, 1, 1], r: [0.6, 0, 0] },
  { id: "pos", label: "all positively correlated (0.3)", v: [1, 1, 1], r: [0.3, 0.3, 0.3] },
  { id: "tilt", label: "unequal variances 1, 1.5, 2.5", v: [1, 1.5, 2.5], r: [0, 0, 0] },
  { id: "custom", label: "custom", v: null, r: null },
];

// Defaults: the v1 tours' common parameters (docs/v1/findings.md, "What the explorer should let the reader
// discover"): N = 90, lateral weights starting at the floor, W_IB ~ Exp(0.2), w_min = 1e-4, gamma = 1.
export const DEFAULTS = { s: "indep", v1: 1, v2: 1, v3: 1, r12: 0, r13: 0, r23: 0, theta: 0.4, gamma: 1, wmin: 1e-4,
  n: 90, seed: 0, ii: 0, ib: 0.2 };

// A tour is an exact state: DEFAULTS overridden by `params` (a preset fills v*, r*) and a fresh start. Ids and
// parameters follow the v1 findings' verified Chapter 1 tours (seed 0 of the three verified there), plus
// "three-corners-s1" (the same with seed 1). The counts in the notes are this page's engine, checked with runTour.
export const TOURS = [
  { id: "neutral-identity", label: "No pruning", params: { s: "indep", theta: 0 },
    note: "Independent inputs and $\\theta = 0$: nothing is pruned. Every lateral synapse grows and all 90 neurons end on one point, the population's starting mean, near the centroid.",
    expect: { K: 1, where: [""] } },
  { id: "corners-collective", label: "θ below γκ₀", params: { s: "indep", theta: 0.2 },
    note: "$\\theta = 0.2$: the three corners are now the only stable parts. But $\\theta$ is below the heuristic crossover $\\gamma\\kappa_0 = \\tfrac13$, so lateral synapses grow first, the population binds into one blob, and the blob lands on <em>one</em> corner.",
    expect: { K: 1, where: ["corner"] } },
  { id: "three-corners", label: "Pruning", params: { s: "indep", theta: 0.4 },
    note: "$\\theta = 0.4$, above $\\gamma\\kappa_0$: neurons choose independently and the interior splits into one cluster per corner, 40, 30 and 20 neurons for this seed. Within each cluster the members are clones (see the spread tile).",
    expect: { K: 3, where: ["corner", "corner", "corner"] } },
  { id: "three-corners-s1", label: "… another seed", params: { s: "indep", theta: 0.4, seed: 1 },
    note: "The same parameters with a different random start (seed 1): again three corner clusters, now 35, 31 and 24 neurons. Both runs converge; the split is whatever the start gave.",
    expect: { K: 3, where: ["corner", "corner", "corner"] } },
  { id: "edge-midpoint", label: "A correlated pair", params: { s: "pair", theta: 0.5 },
    note: "$\\rho_{12} = 0.6$, $\\theta = 0.5$, between the cross-correlation and zero. Two clusters: one at $g_3$, one at the midpoint of the $g_1$–$g_2$ edge, $h = (\\tfrac12, \\tfrac12, 0)$, the non-negative part shared by the pair. The signed eigenvector $(1, -1, 0)/\\sqrt2$ of $\\Sigma_g$ has no cluster.",
    expect: { K: 2, where: ["midpoint12", "corner3"] } },
  { id: "midpoint-splits", label: "θ above ρ₁₂, γ = ½", params: { s: "pair", theta: 0.65, gamma: 0.5 },
    note: "The same inputs at $\\theta = 0.65 &gt; \\rho_{12}$ (and lateral gain $\\gamma = 0.5$, so nothing binds): the midpoint is no longer a part, and the neurons sit on the three corners. The corners $g_1$ and $g_2$ are not eigenvectors of this $\\Sigma_g$.",
    expect: { K: 3, where: ["corner", "corner", "corner"] } },
  { id: "independent-tilt", label: "Unequal inputs", params: { s: "tilt", theta: 0.2, gamma: 0 },
    note: "Variances $1, 1.5, 2.5$ and no lateral plasticity ($\\gamma = 0$): every input is populated, but the louder ones get far more neurons (66, 18 and 6 for this seed). With $\\gamma = 1$ the bound population goes entirely to $g_3$.",
    expect: { K: 3, where: ["corner3", "corner2", "corner1"] } },
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

// ---------------------------------------------------------------- the simulation (mean-field: exact Sigma_g)
export function createSim(p0) {
  const sim = { p: null, s: null, Sigma: null, P: null };

  sim.reset = p => {
    sim.p = { ...p };
    sim.s = E.createState({ N: p.n, B, seed: p.seed, iiScale: p.ii, ibScale: p.ib });
    sim.Sigma = sigmaOf(p);
    sim.P = { ...E.defaultParams(), theta: p.theta, gamma: p.gamma, wmin: p.wmin, lam: LAM, eta: ETA, dt: DT };
    sim.s.resid = NaN;
  };
  // live parameter changes (no restart): theta, gamma, wmin, Sigma
  sim.update = p => {
    const old = sim.p; sim.p = { ...old, ...p };
    sim.P.theta = sim.p.theta; sim.P.gamma = sim.p.gamma; sim.P.wmin = sim.p.wmin;
    if (["v1", "v2", "v3", "r12", "r13", "r23"].some(k => sim.p[k] !== old[k])) sim.Sigma = sigmaOf(sim.p);
  };
  sim.time = () => sim.s.t * DT;
  // advance n steps (or until the time budget in ms runs out, at least one step); returns the steps taken
  sim.advance = (n, budgetMs = Infinity) => {
    const t0 = budgetMs < Infinity ? performance.now() : 0;
    let k = 0;
    while (k < n) {
      E.step(sim.s, sim.Sigma, sim.P);
      k++;
      if (budgetMs < Infinity && performance.now() - t0 > budgetMs) break;
    }
    return k;
  };
  // Stationarity residual in covariance units: the last step's own value (weights that step started from);
  // before the first step, or after a parameter change, the drift at the current weights (recomputed).
  sim.residual = () => {
    if (!Number.isFinite(sim.s.resid)) return E.stationarity(sim.s, sim.Sigma, sim.P).r / ETA;
    return sim.s.resid / ETA;
  };
  sim.reset(p0);
  return sim;
}

// ---------------------------------------------------------------- per-frame analysis
// Everything needed to draw one moment (positions, synapses, clusters), detached from the engine buffers.
// H is the harmonic embedding of the weights the last step started from (engine convention), W8 the current
// weights as 8-bit fractions of their maximum.
export function analyse(sim, prevLabels = null) {
  const { s, Sigma } = sim, N = s.N, wmin = sim.P.wmin;
  const resid = sim.residual();
  const H = s.H;
  // dead neurons: every synapse (to inputs and laterally) at the floor; they carry no learned structure
  const floorTol = wmin * (1 + 1e-3);
  const dead = new Uint8Array(N);
  let wmax = 0;
  for (let i = 0; i < N; i++) {
    let alive = false;
    for (let b = 0; b < B; b++) { if (!s.Mib[i * B + b]) continue; const w = s.Wib[i * B + b]; if (w > floorTol) alive = true; if (w > wmax) wmax = w; }
    for (let j = 0; j < N; j++) {
      if (j === i || !s.Mii[i * N + j]) continue;
      const w = s.Wii[i * N + j]; if (w > floorTol) alive = true;
      if (j > i && w > wmax) wmax = w;
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
  // spread: the largest L1 distance from a member's embedding to its cluster's mean embedding
  for (const c of all) { let m = 0; for (const i of c.members) { let d = 0; for (let b = 0; b < B; b++) d += Math.abs(H[i * B + b] - c.part[b]); if (d > m) m = d; } c.spread = m; }
  // K counts populated clusters, >= max(2, 2% N) members (findings.md convention); smaller ones are stragglers
  const minPop = Math.max(2, Math.ceil(0.02 * N));
  const live = all.filter(c => !c.dead).sort((a, b) => b.n - a.n);
  for (const c of live) c.small = c.n < minPop;
  const deadC = all.find(c => c.dead) || null;
  const big = live.filter(c => !c.small);
  return {
    t: sim.time(), N, H: Float32Array.from(H.subarray(0, N * B)), W8, wmax, labels: Int32Array.from(labels), dead,
    clusters: deadC ? [...live, deadC] : live, K: big.length, nSmall: live.filter(c => c.small).reduce((a, c) => a + c.n, 0), nDead: deadC ? deadC.n : 0, robust: cl.robust, gap: cl.gap,
    spread: big.length ? Math.max(...big.map(c => c.spread)) : NaN,
    resid, theta: sim.P.theta, gamma: sim.P.gamma, Sigma: Float64Array.from(Sigma),
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

// Headless run of a tour (for verification): returns the final analysis.
export function runTour(t, { tMax = 200, seed = null } = {}) {
  const p = tourParams(t); if (seed != null) p.seed = seed;
  const sim = createSim(p);
  const steps = Math.round(tMax / DT);
  for (let k = 0; k < steps; k++) {
    sim.advance(1);
    if (k > 2 && sim.s.resid / ETA < RES_TOL) break;
  }
  return { a: analyse(sim, null), t: sim.time() };
}
