// Chapter "homeo" (Homeostasis): model side, no DOM. Parameters, tours, the simulation wrapper and the per-frame
// analysis (clusters, their energies, the allocation fit, the regime). Kept apart from ch_homeo.js so the tours can
// be verified headless in node with exactly the page's code (see verifyTour at the bottom).
//
// Covariances and their stable parts come from data/homeo/covs.json (hebb/homeo_cov.py): the page and the sweeps
// then use bit-identical matrices, and the parts table is exact (enumeration of supports for B <= 16).
import * as E from "./engine.js";

export const DT = 0.05;              // Euler step in units of 1/lambda (the transient is the subject: ENGINE.md)
export const ETA = 1;
export const RES_TOL = 1e-6;         // "converged" (covariance units)
export const IB_MEAN = 0.2;          // mean initial input weight (units of eta/lambda), as in every sweep

// Initial input-weight distributions, all with mean 0.2 (alloc_limit.json / alloc_ingredients.json recipes).
export const INITS = [
  { id: "exp", label: "exponential (default)", logsd: Math.PI / Math.sqrt(6), short: "Exp" },
  { id: "unif", label: "uniform U(0, 0.4)", logsd: null, short: "uniform" },
  { id: "ln0.5", label: "log-normal, log-sd 0.5", logsd: 0.5, short: "log-sd 0.5" },
  { id: "ln1", label: "log-normal, log-sd 1", logsd: 1, short: "log-sd 1" },
  { id: "ln2", label: "log-normal, log-sd 2", logsd: 2, short: "log-sd 2" },
  { id: "ln3", label: "log-normal, log-sd 3", logsd: 3, short: "log-sd 3" },
];
export const BATCHES = [0, 8, 32, 128, 512];          // 0 = mean-field (exact Sigma)
// Private (per-synapse, independent) noise, the three models of findings_private_noise.md F11; sigma per sqrt(1/lambda).
//   inp: each synapse estimates Cov from its own samples (1/sigma^2 per unit time): Var per step sigma^2 dt (Cxx Cyy + Cxy^2)
//   add: w += sigma sqrt(dt) xi;  mul: w <- w exp(sigma sqrt(dt) xi); all clipped at the floor afterwards
export const NOISES = [
  { id: "none", label: "none" },
  { id: "inp:0.2", label: "input noise σ = 0.2" }, { id: "inp:0.3", label: "input noise σ = 0.3" },
  { id: "inp:0.4", label: "input noise σ = 0.4" }, { id: "inp:0.6", label: "input noise σ = 0.6" },
  { id: "add:0.3", label: "additive σ = 0.3" }, { id: "add:0.7", label: "additive σ = 0.7" },
];
export const noiseOf = id => { if (!id || id === "none") return null; const [kind, s] = id.split(":"); return { kind, sigma: +s }; };

export const DEFAULTS = { cov: "graded6", theta: 0.4, gamma: 1, omega: 0.02, init: "exp", n: 150, lam: 1, seed: 1,
  ii: 0, batch: 0, noise: "none" };
export const KEYS = Object.keys(DEFAULTS);
export const RESET_KEYS = ["cov", "init", "n", "lam", "seed", "ii"];      // these restart the run

// Tours. The first five are the verified "Chapter 4" tours of docs/findings.md (tours.json); the rest are mine.
// Every note was checked by running verifyTour in node for seeds 1-3 (outcomes recorded in `expect`) and in the
// page. Numbers in notes are what the JS engine gives for the seed the tour uses.
export const TOURS = [
  { id: "filter-steep", group: "f", label: "Steep filter", params: { cov: "graded6", theta: 0.05, gamma: 0, omega: 0.02, n: 200 },
    note: "Six independent inputs, variances 0.5 to 3; no lateral plasticity ($\\gamma = 0$); $\\theta = 0.05$. Nearly everything goes to the loudest input: 171–177 of 200 neurons on $v = 3$, 20–25 on $v = 2$, at most 4 elsewhere (seeds 1–3; fitted $\\beta \\approx 5$). All six parts exist; four get nobody." },
  { id: "filter-soft", group: "f", label: "Soft filter", params: { cov: "graded6", theta: 0.4, gamma: 0, omega: 0.02, n: 200 },
    note: "Same, $\\theta = 0.4$, just below the smallest variance. Five inputs populated, roughly 101–114, 46–53, 24–25, 7–18 and 3–11 neurons from loud to quiet (seeds 1–3); fitted $\\beta$ 1.6–2.2. The $v = 0.5$ part exists ($c = 0.1$) but gets nobody." },
  { id: "filter-collective", group: "f", label: "Collective", params: { cov: "graded6", theta: 0.2, gamma: 1, omega: 0.02, n: 200 },
    note: "Lateral plasticity on ($\\gamma = 1$), $\\theta = 0.2$, below the heuristic crossover $\\gamma\\kappa_0 = 0.243$: the population binds before anyone chooses, and 195–198 of 200 neurons land on the loudest input (seeds 1–3). Six parts exist; one is populated." },
  { id: "floor-leak", group: "f", label: "Floor too high", params: { cov: "block12", theta: 0.65, gamma: 1, omega: 0.15, n: 150 },
    note: "<code>block12</code>, $\\theta = 0.65$ (twelve single-input parts exist), floor $\\omega = N w_{\\min} = 0.15$ ($w_{\\min} = 10^{-3}$): every neuron dies (seeds 1–3)." },
  { id: "floor-leak-ok", group: "f", label: "Floor lowered", params: { cov: "block12", theta: 0.65, gamma: 1, omega: 0.0015, n: 150 },
    note: "The same with $\\omega = 0.0015$ ($w_{\\min} = 10^{-5}$): twelve clusters, one per input, of 3 to 21 neurons (seeds 1–3)." },
  { id: "lambda-inert", group: "m", label: "λ is inert", params: { cov: "graded6", theta: 0.4, gamma: 0, omega: 0.02, n: 200, lam: 4 },
    note: "The soft filter with decay $\\lambda = 4$ instead of 1. Exactly the same clusters and sizes (seed 1: 101, 53, 25, 18, plus 3), but every weight is 4 times smaller (max 0.65 instead of 2.59) and the clock runs 4 times faster (converged at $t = 7.5$ instead of 30)." },
  { id: "spread-narrow", group: "m", label: "Narrow start", params: { cov: "graded6", theta: 0.2, gamma: 0, omega: 0.02, n: 200, init: "ln0.5" },
    note: "$\\theta = 0.2$, $\\gamma = 0$, initial input weights log-normal with log-sd 0.5 (narrower than the default exponential): 188–194 of 200 neurons on the loudest input (seeds 1–3). With the default start it is 129, 45, 17, 7 (seed 1)." },
  { id: "spread-wide", group: "m", label: "Wide start", params: { cov: "graded6", theta: 0.2, gamma: 0, omega: 0.02, n: 200, init: "ln3" },
    note: "Same $\\theta$, initial weights with log-sd 3: all six inputs populated, from 98–114 on $v = 3$ down to 8–11 on $v = 0.5$, fitted $\\beta \\approx 1.25$ (seeds 1–3). Nothing but the spread of the starting weights changed." },
  { id: "mnist", group: "m", label: "MNIST parts", params: { cov: "mnist49", theta: 0.6, gamma: 1, omega: 0.02, n: 150 },
    note: "Real covariance: MNIST digits at $7\\times7$, $\\theta = 0.6$, $\\gamma = 1$. Eight parts exist, each a small set of 1–5 pixels that brighten together; four are populated, the two most energetic ($\\lambda \\approx 5.3$) with 56–76 neurons each and two quieter ones with 5–15 (seeds 1–3). Compare (d) with the eigenvectors in (e)." },
  { id: "noise-wta", group: "n", label: "Noise: winner", params: { cov: "graded6", theta: 0.2, gamma: 0, omega: 0.02, n: 200, noise: "inp:0.2" },
    note: "Private input noise $\\sigma = 0.2$ (each synapse estimates its covariance from its own 25 samples per unit time), $\\gamma = 0$. Neurons now hop, and the allocation forgets the start; the law it settles to is winner-take-all: by $t \\approx 200$, about 186–192 of 200 neurons sit on $v = 3$ and 7–13 are in transit (seeds 1–3; the numbers fluctuate)." },
  { id: "noise-graded", group: "n", label: "Noise: graded", params: { cov: "graded6", theta: 0.2, gamma: 0, omega: 0.02, n: 200, noise: "inp:0.4" },
    note: "The same with $\\sigma = 0.4$: a graded allocation, but about half the neurons (roughly 90–120 of 200) are between parts at any moment; of the rest, roughly 70–95 sit on $v = 3$ and up to about 17 on $v = 2$ around $t = 200$ (seeds 1–3; the numbers fluctuate). That is the price of the graded law (findings F14, F17)." },
  { id: "noise-bound", group: "n", label: "Noise + lateral", params: { cov: "planted_energy", theta: 0.35, gamma: 1, omega: 0.02, n: 120, noise: "inp:0.3" },
    note: "Four planted blocks of unequal energy, $\\gamma = 1$, $\\theta = 0.35$ (above the heuristic crossover $\\gamma\\kappa_0 = 0.234$), input noise $\\sigma = 0.3$: all 120 neurons end in one cluster on the loudest block (seeds 1–3, $t = 200$). Without noise the same start populates all four blocks (67–72, 31–36, 9–18, 3–4)." },
];
export const tourById = id => TOURS.find(t => t.id === id) || null;
export const tourParams = t => ({ ...DEFAULTS, ...t.params });

// ---------------------------------------------------------------- covariances (from covs.json)
export function covOf(data, id) {
  const c = data.covs[id];
  if (!c) return null;
  if (!c._S) {
    const B = c.B, S = new Float64Array(B * B);
    for (let a = 0; a < B; a++) for (let b = 0; b < B; b++) S[a * B + b] = c.Sigma[a][b];
    c._S = S;
    c._grid = new Map(c.parts.map(r => [r.theta.toFixed(2), r]));
  }
  return c;
}
// Stable parts at theta (exact table on the 0.01 grid; nearest grid row otherwise, flagged)
export function partsAt(c, theta) {
  const key = (Math.round(theta * 100) / 100).toFixed(2);
  const row = c._grid.get(key);
  if (row) return { ...row, exact: Math.abs(theta - row.theta) < 1e-9 };
  const lo = c.parts[0], hi = c.parts[c.parts.length - 1];
  if (theta > hi.theta) return { theta, n: 0, parts: [], exact: false, beyond: true };
  return { ...(theta < lo.theta ? lo : hi), exact: false };
}
// Dense part vector (length B) from a table entry
export function partVec(part, B) { const v = new Float64Array(B); part.S.forEach((b, i) => { v[b] = part.h[i]; }); return v; }

// ---------------------------------------------------------------- the simulation
export function wminOf(p) { return p.omega * ETA / (p.lam * p.n); }

export function createSim(data, p0) {
  const sim = { p: null, s: null, c: null, Sigma: null, P: null, sampler: null, jrng: null };
  sim.reset = p => {
    sim.p = { ...p };
    sim.c = covOf(data, p.cov); sim.Sigma = sim.c._S;
    const B = sim.c.B, N = p.n, sc = ETA / p.lam;      // initial weights in units of eta/lambda, so lambda is inert
    sim.s = E.createState({ N, B, seed: p.seed, iiScale: p.ii * sc, ibScale: IB_MEAN * sc });
    if (p.init !== "exp") {
      const rng = E.makeRng(p.seed * 7919 + 17), W = sim.s.Wib, ini = INITS.find(x => x.id === p.init);
      for (let k = 0; k < N * B; k++) {
        if (!sim.s.Mib[k]) continue;
        if (p.init === "unif") W[k] = 2 * IB_MEAN * rng.uniform() * sc;
        else { const s = ini.logsd; W[k] = IB_MEAN * Math.exp(s * rng.normal() - s * s / 2) * sc; }
      }
    }
    sim.P = { ...E.defaultParams(), theta: p.theta, gamma: p.gamma, wmin: wminOf(p), lam: p.lam, eta: ETA, dt: DT / p.lam };
    sim.sampler = null; sim.jrng = E.makeRng(p.seed * 104729 + 3);
    sim.s.resid = NaN;
  };
  // live changes (no restart): theta, gamma, omega (-> wmin), batch, noise
  sim.update = p => {
    sim.p = { ...sim.p, ...p };
    sim.P.theta = sim.p.theta; sim.P.gamma = sim.p.gamma; sim.P.wmin = wminOf(sim.p);
  };
  sim.time = () => sim.s.t * sim.P.dt;
  sim.advance = (n, budgetMs = Infinity) => {
    const t0 = budgetMs < Infinity ? performance.now() : 0, s = sim.s, N = s.N, B = s.B;
    let k = 0;
    while (k < n) {
      if (sim.p.batch > 0) {
        if (!sim.sampler) sim.sampler = E.makeSampler(sim.Sigma, B, sim.p.seed + 1000);
        E.stepStochastic(s, sim.sampler, sim.P, sim.p.batch, true);
      } else E.step(s, sim.Sigma, sim.P);
      const nz = noiseOf(sim.p.noise);
      if (nz) privateNoise(sim, nz);
      k++;
      if (budgetMs < Infinity && performance.now() - t0 > budgetMs) break;
    }
    return k;
  };
  sim.noisy = () => sim.p.batch > 0 || !!noiseOf(sim.p.noise);
  // residual in covariance units: the step's own value in clean mean-field; else the exact drift at these weights
  sim.residual = () => {
    if (sim.noisy() || !Number.isFinite(sim.s.resid)) return E.stationarity(sim.s, sim.Sigma, sim.P).r / ETA;
    return sim.s.resid / ETA;
  };
  sim.reset(p0);
  return sim;
}

// One step of private noise on every potential synapse (after the deterministic step). Units: weights in eta/lambda,
// sigma per sqrt(1/lambda), so the rescaled dynamics do not depend on lambda (findings_private_noise.md conventions).
function privateNoise(sim, { kind, sigma }) {
  const s = sim.s, N = s.N, B = s.B, H = s.H, S = sim.Sigma, r = sim.jrng, wmin = sim.P.wmin, lam = sim.p.lam, g = sim.P.gamma;
  const amp = (ETA / lam) * sigma * Math.sqrt(lam * sim.P.dt), Wib = s.Wib, Wii = s.Wii;
  if (kind === "mul") {
    const a = sigma * Math.sqrt(lam * sim.P.dt);
    for (let q = 0; q < N * B; q++) if (s.Mib[q]) Wib[q] = Math.max(wmin, Wib[q] * Math.exp(a * r.normal()));
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (s.Mii[i * N + j]) { const w = Math.max(wmin, Wii[i * N + j] * Math.exp(a * r.normal())); Wii[i * N + j] = w; Wii[j * N + i] = w; }
    return;
  }
  if (kind === "add") {
    for (let q = 0; q < N * B; q++) if (s.Mib[q]) Wib[q] = Math.max(wmin, Wib[q] + amp * r.normal());
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (s.Mii[i * N + j]) { const w = Math.max(wmin, Wii[i * N + j] + amp * r.normal()); Wii[i * N + j] = w; Wii[j * N + i] = w; }
    return;
  }
  // input (covariance-estimation) noise: Isserlis variance of a sample covariance
  if (!sim._HS || sim._HS.length !== N * B) { sim._HS = new Float64Array(N * B); sim._q = new Float64Array(N); }
  const HS = sim._HS, q = sim._q;
  for (let i = 0; i < N; i++) {
    let qi = 0;
    for (let a = 0; a < B; a++) { let t = 0; for (let b = 0; b < B; b++) t += H[i * B + b] * S[b * B + a]; HS[i * B + a] = t; qi += t * H[i * B + a]; }
    q[i] = qi;
  }
  for (let i = 0; i < N; i++) for (let b = 0; b < B; b++) {
    const k = i * B + b; if (!s.Mib[k]) continue;
    const v = q[i] * S[b * B + b] + HS[k] * HS[k];
    Wib[k] = Math.max(wmin, Wib[k] + amp * Math.sqrt(Math.max(v, 0)) * r.normal());
  }
  if (g > 0) for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    if (!s.Mii[i * N + j]) continue;
    let c = 0; for (let b = 0; b < B; b++) c += HS[i * B + b] * H[j * B + b];
    const w = Math.max(wmin, Wii[i * N + j] + amp * g * Math.sqrt(Math.max(q[i] * q[j] + c * c, 0)) * r.normal());
    Wii[i * N + j] = w; Wii[j * N + i] = w;
  }
}

// ---------------------------------------------------------------- analysis
export const minSize = N => Math.max(2, Math.ceil(0.02 * N));     // the sweeps' K counts clusters this big

function topEig(Sigma, B, S) {
  const m = S.length; if (m === 1) return Sigma[S[0] * B + S[0]];
  const A = new Float64Array(m * m);
  for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) A[a * m + b] = Sigma[S[a] * B + S[b]];
  return E.eigh(A, m).values[0];
}

// Least-squares slope of log y on log x (the allocation exponent), with its points
export function fitBeta(xs, ys) {
  const pts = xs.map((x, i) => [x, ys[i]]).filter(([x, y]) => x > 0 && y > 0);
  if (pts.length < 2) return null;
  const lx = pts.map(p => Math.log(p[0])), ly = pts.map(p => Math.log(p[1]));
  const mx = lx.reduce((a, b) => a + b) / lx.length, my = ly.reduce((a, b) => a + b) / ly.length;
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < lx.length; i++) { sxx += (lx[i] - mx) ** 2; sxy += (lx[i] - mx) * (ly[i] - my); syy += (ly[i] - my) ** 2; }
  if (sxx < 1e-9) return null;
  const beta = sxy / sxx, r = syy > 0 ? sxy / Math.sqrt(sxx * syy) : 1;
  return { beta, r, mx, my, n: pts.length };
}

// Everything needed to draw one moment, detached from the engine buffers.
export function analyse(sim, prevLabels = null) {
  const { s, Sigma, c } = sim, N = s.N, B = s.B, wmin = sim.P.wmin, theta = sim.P.theta;
  const resid = sim.residual();
  const H = s.H;
  const dead = new Uint8Array(N);
  let nDead = 0, wmax = 0;
  for (let i = 0; i < N; i++) {
    let m = 0; for (let b = 0; b < B; b++) { const w = s.Wib[i * B + b]; if (w > m) m = w; }
    if (m > wmax) wmax = m;
    if (m <= 1.5 * wmin) { dead[i] = 1; nDead++; }
  }
  const pt = partsAt(c, theta), parts = pt.parts, pvec = parts.map(q => partVec(q, B));
  const pnorm = pvec.map(v => Math.hypot(...v));
  const byLab = new Map(), add = (key, i) => {
    let g = byLab.get(key);
    if (!g) { g = { id: key, n: 0, h: new Float64Array(B), members: [] }; byLab.set(key, g); }
    g.n++; g.members.push(i); for (let b = 0; b < B; b++) g.h[b] += H[i * B + b];
  };
  let cl, labels, transit = 0;
  if (noiseOf(sim.p.noise) && parts.length) {
    // private noise blurs clusters into clouds: assign each neuron to a part's core (cos >= 0.9), else "in transit"
    // (the measure of findings_private_noise.md)
    labels = new Int32Array(N).fill(-1); cl = { robust: true };
    for (let i = 0; i < N; i++) {
      if (dead[i]) continue;
      let hn = 0; for (let b = 0; b < B; b++) hn += H[i * B + b] ** 2; hn = Math.sqrt(hn);
      let best = -1, bc = 0;
      pvec.forEach((v, k) => { let d = 0; for (let b = 0; b < B; b++) d += v[b] * H[i * B + b]; d /= pnorm[k] * hn; if (d > bc) { bc = d; best = k; } });
      if (bc >= 0.9) { labels[i] = best; add(best, i); } else transit++;
    }
  } else {
    cl = E.clustersAuto(H, N, B); labels = cl.labels;
    if (prevLabels && prevLabels.length === N) labels = E.relabel(prevLabels, labels, N);
    for (let i = 0; i < N; i++) if (!dead[i]) add(labels[i], i);
  }
  const groups = [...byLab.values()];
  for (const g of groups) {
    for (let b = 0; b < B; b++) g.h[b] /= g.n;
    let E2 = 0, nn = 0;
    for (let a = 0; a < B; a++) { let t = 0; for (let b = 0; b < B; b++) t += Sigma[a * B + b] * g.h[b]; E2 += g.h[a] * t; nn += g.h[a] * g.h[a]; }
    g.E = E2; g.c = (E2 - theta) / nn;                          // findings F1: c = (h'Sigma h - theta)/|h|^2
    // nearest stable part (cosine > 0.98 = matched, as in s1_parts.py)
    let best = -1, bc = 0; const hn = Math.sqrt(nn);
    pvec.forEach((v, k) => { let d = 0; for (let b = 0; b < B; b++) d += v[b] * g.h[b]; d /= (pnorm[k] * hn); if (d > bc) { bc = d; best = k; } });
    g.cos = bc; g.part = typeof g.id === "number" && noiseOf(sim.p.noise) && parts.length ? g.id : bc > 0.98 ? best : -1;
    if (g.part >= 0) { g.lam = parts[g.part].lam; g.supp = parts[g.part].S; }
    else {
      const mx = Math.max(...g.h); g.supp = []; for (let b = 0; b < B; b++) if (g.h[b] > 0.05 * mx) g.supp.push(b);
      g.lam = topEig(Sigma, B, g.supp);
    }
  }
  groups.sort((a, b) => b.n - a.n || b.lam - a.lam);
  const ms = minSize(N), major = groups.filter(g => g.n >= ms), minor = groups.filter(g => g.n < ms);
  const populated = new Set(major.filter(g => g.part >= 0).map(g => g.part));
  const empty = parts.map((q, k) => ({ ...q, k })).filter(q => !populated.has(q.k) && !groups.some(g => g.part === q.k));
  const fitL = fitBeta(major.map(g => g.lam), major.map(g => g.n));
  const fitC = fitBeta(major.map(g => g.c), major.map(g => g.n));
  return {
    t: sim.time(), N, B, H: Float32Array.from(H.subarray(0, N * B)), labels: Int32Array.from(labels), dead, nDead,
    groups, major, minor, K: major.length, robust: cl.robust, nStable: pt.n, tie: pt.tie || null, exact: pt.exact,
    parts, empty, fitL, fitC, resid, wmax, theta, gamma: sim.P.gamma, omega: sim.p.omega, lam: sim.p.lam,
    mode: sim.p.batch > 0 ? `minibatch ${sim.p.batch}` : "mf", noise: sim.p.noise, transit,
  };
}

// The regime the reader is in (findings F1 corollaries, F3, F5). Returns {id, name, text}.
export function regime(c, p, nStable) {
  const st = c.stats, k0 = st.kappa0, th = p.theta, g = p.gamma;
  const f = x => (Math.round(x * 1000) / 1000).toString().replace("-", "−");
  if (th >= st.var_max) return { id: "extinct", name: "extinction", text: `θ is above every input variance (max ${f(st.var_max)}): no part exists and every neuron dies.` };
  if (nStable === 0) return { id: "extinct", name: "no stable part", text: "No part is strictly stable at this exact θ (it sits on a correlation level); move θ slightly." };
  const extra = th < st.offdiag_min ? ` θ is below every covariance (min ${f(st.offdiag_min)}), so exactly one part exists, with full support.`
    : (th > st.offdiag_max && th < st.var_min ? ` θ lies between the largest covariance (${f(st.offdiag_max)}) and the smallest variance (${f(st.var_min)}): the parts are exactly the ${c.B} single inputs.` : "");
  if (g * k0 > th) return { id: "collective", name: "collective", text: `θ < γκ₀ = ${f(g * k0)} (the heuristic crossover): lateral synapses grow before neurons can choose, so the population tends to bind into one blob and land in one part, the loudest.${extra}` };
  if (nStable === 1) return { id: "one", name: "one part", text: `Only one stable part exists at this θ, so every surviving neuron ends up in it.${extra}` };
  const why = g > 0 ? `θ > γκ₀ = ${f(g * k0)} (the heuristic crossover)` : "No lateral plasticity (γ = 0)";
  return { id: "independent", name: "independent", text: `${why}: neurons pick parts on their own; in the families tested, louder parts get more of them (n ∝ λ^β, β set by θ and the initial spread).${extra}` };
}

// ---------------------------------------------------------------- headless verification (node)
export function runToFixedPoint(data, p, { maxSteps = 60000, every = 200 } = {}) {
  const sim = createSim(data, p);
  let steps = 0, a = null;
  while (steps < maxSteps) {
    sim.advance(every); steps += every;
    if (!sim.noisy() && sim.s.resid / ETA < RES_TOL) break;
  }
  a = analyse(sim);
  return { a, steps, sim };
}
