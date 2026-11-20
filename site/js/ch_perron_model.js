// Chapter "perron": the three-input simulation, model side (no DOM). A thin wrapper around the v1 covariance-rule
// engine (engine.js, parity-tested against hebb/core.py), the per-run analysis (groups of identical copies), and a
// cooperative runner that integrates many small runs without blocking the page. Runs under node too.
import * as E from "./engine.js";

export const B = 3;
export const DT = 0.05;            // Euler step (lambda = eta = 1)
export const RES_TOL = 1e-7;       // stop when the projected stationarity residual falls below this
export const MAX_STEPS = 6000;     // ... or after this many steps (t = 300)

// Defaults of every three-input run on the page: N neurons, lateral synapses starting at the floor,
// input synapses W_IB ~ Exp(0.2), floor w_min = 1e-4, lateral gain gamma = 0.25.
export const DEFAULTS = { n: 60, seed: 0, gamma: 0.25, wmin: 1e-4, ii: 0, ib: 0.2 };

// Input covariances used on the page: variances v (g1, g2, g3) and correlations r = (rho12, rho13, rho23).
export const SIGMAS = {
  pair: { v: [1, 1, 1], r: [0.6, 0, 0], label: "g₁, g₂ correlated 0.6" },
  tilt: { v: [1, 1.6, 1], r: [0.6, 0, 0], label: "same, g₂ louder (variance 1.6)" },
  tri: { v: [1, 1, 1], r: [0.6, 0.3, 0.45], label: "all three correlated" },
};
export const sigmaOf = s => E.cov3({ variances: s.v, corr: s.r });

export function createSim(p) {
  const P = { ...E.defaultParams(), theta: p.theta, gamma: p.gamma, wmin: p.wmin, lam: 1, eta: 1, dt: DT };
  const s = E.createState({ N: p.n, B, seed: p.seed, iiScale: p.ii, ibScale: p.ib });
  return { p, P, s, Sigma: p.Sigma, steps: 0,
    step() { E.step(s, p.Sigma, P); this.steps++; },
    done() { return this.steps >= MAX_STEPS || (this.steps > 50 && s.resid < RES_TOL); } };
}

// Groups at the current state: single linkage on the embeddings in L1, cut at the largest gap (engine.clustersAuto).
// Returns {H (Float32Array N*3), groups: [{n, h: [3], members}] largest first, nDead}. A neuron is dead when every
// synapse it has is at the floor.
export function analyse(sim) {
  const { s } = sim, N = s.N, H = s.H, floorTol = sim.P.wmin * (1 + 1e-3);
  const dead = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    let alive = false;
    for (let b = 0; b < B && !alive; b++) if (s.Wib[i * B + b] > floorTol) alive = true;
    for (let j = 0; j < N && !alive; j++) if (j !== i && s.Wii[i * N + j] > floorTol) alive = true;
    dead[i] = alive ? 0 : 1;
  }
  const { labels } = E.clustersAuto(H, N, B);
  const by = new Map();
  for (let i = 0; i < N; i++) {
    if (dead[i]) continue;
    let g = by.get(labels[i]);
    if (!g) { g = { n: 0, h: [0, 0, 0], members: [] }; by.set(labels[i], g); }
    g.n++; g.members.push(i); for (let b = 0; b < B; b++) g.h[b] += H[i * B + b];
  }
  const groups = [...by.values()].sort((a, b) => b.n - a.n);
  for (const g of groups) for (let b = 0; b < B; b++) g.h[b] /= g.n;
  return { H: Float32Array.from(H.subarray(0, N * B)), groups, nDead: dead.reduce((a, x) => a + x, 0), steps: sim.steps, t: sim.steps * DT, resid: s.resid };
}

// ------------------------------------------------------------------ cooperative runner (browser and node)
// run(p, {record}) -> Promise of {final: analyse(...), path: [Float32Array N*3 every `record` steps] | null}.
// Runs are queued one after another and integrated in slices of ~20 ms so the page stays responsive.
// Results are cached by their parameters.
const cache = new Map();
const queue = [];
let busy = false;
const later = typeof requestAnimationFrame === "function" ? f => setTimeout(f, 0) : f => setImmediate(f);
const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

export function run(p, { record = 0 } = {}) {
  const key = JSON.stringify({ ...p, Sigma: Array.from(p.Sigma), record });
  if (cache.has(key)) return cache.get(key);
  const pr = new Promise(resolve => { queue.push({ p, record, resolve }); pump(); });
  cache.set(key, pr);
  return pr;
}
function pump() {
  if (busy || !queue.length) return;
  busy = true;
  const job = queue.shift(), sim = createSim(job.p), path = job.record ? [] : null;
  const slice = () => {
    const t0 = now();
    while (!sim.done() && now() - t0 < 20) {
      sim.step();      // afterwards s.H is the embedding of the weights this step started from
      if (path && (sim.steps - 1) % job.record === 0) path.push(Float32Array.from(sim.s.H.subarray(0, sim.s.N * B)));
    }
    if (!sim.done()) { later(slice); return; }
    const final = analyse(sim);
    if (path) path.push(final.H);
    job.resolve({ final, path });
    busy = false; later(pump);
  };
  later(slice);
}
