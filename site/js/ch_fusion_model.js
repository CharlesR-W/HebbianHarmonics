// Chapter "fusion", model side (no DOM, so it runs headless under node for checking the tours):
// the two-neuron calibration (analytic drift, fixed points, bistable window, an engine run), and the live
// B = 3 network of linear neurons wrapped around onsite_engine.createSim with the analysis each frame needs
// (components sorted by size, diameters and spans on the training and the evaluation bank, positions in the
// simplex, the envelope check).
import * as O from "./onsite_engine.js";

export const B = 3, S_TRAIN = 256, S_EVAL = 1024;
export const CLAIM_TOL = 1e-6;     // equilibrium claims need the plasticity residual below this (SPEC §1)
export const STOP_TOL = 1e-9;      // runs pause here (as the sweeps: max|dW|/dt < 1e-9)
export const dtOf = alpha => Math.min(0.5, 1 / alpha);   // as hebb/sweeps2/fusion.py

// ---------------------------------------------------------------- input statistics (synthetic)
export const SIGMAS = [
  { id: "iid", label: "independent, unit variances", make: () => O.covDiag([1, 1, 1]) },
];
export const sigmaById = id => SIGMAS.find(s => s.id === id) || SIGMAS[0];

// ---------------------------------------------------------------- parameters, tours
export const DEFAULTS = { b: 0.015, alpha: 1, sig: "iid", s: 3, n: 30, seed: 0 };
export const LIMITS = { b: [0.001, 1], alpha: [0.03, 10] };

export function clampParams(p) {
  const q = { ...DEFAULTS, ...p };
  for (const k of ["b", "alpha"]) { const x = Number(q[k]); q[k] = Math.min(LIMITS[k][1], Math.max(LIMITS[k][0], Number.isFinite(x) ? x : DEFAULTS[k])); }
  q.s = [1, 2, 3].includes(+q.s) ? +q.s : 3;
  q.n = [20, 30, 40].includes(+q.n) ? +q.n : 30;
  q.seed = Math.max(0, Math.min(9, Math.round(+q.seed || 0)));
  q.sig = sigmaById(q.sig).id;
  return q;
}

// A tour is an exact state. `script` (optional) is a list of b values visited by continuation: the run goes to
// its fixed point at each one before moving on, carrying W.
export const TOURS = [
  { id: "chains", label: "Chains at percolation", params: { b: 0.015 },
    note: "$b = 0.015$, $\\alpha = 1$, from $W = 0$: the live graph is about to percolate. Compare the span of the largest component with the edge range $\\sqrt{2b}$ (tiles)." },
  { id: "hysteresis", label: "Hysteresis", params: { b: 0.3, alpha: 0.1 }, script: [0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.04, 0.03],
    note: "$\\alpha = 0.1$: from $W = 0$ at $b = 0.3$ (one blob), then $b$ is lowered step by step to $0.03$, each time to the fixed point, carrying $W$. Then press ↺ and play to restart from $W = 0$ at the same $b$." },
];
export const tourById = id => TOURS.find(t => t.id === id) || null;
export const tourParams = t => clampParams({ ...DEFAULTS, ...t.params });

// ---------------------------------------------------------------- the two-neuron calibration (analytic)
// Two linear neurons, neuron k attached to input k with conductance 1, inputs with E(g1 - g2)^2 = D0^2:
// u1 - u2 = (g1 - g2)/(1 + 2w), so D^2(w) = D0^2/(1 + 2w)^2 and dw/dt = b - alpha w - D^2(w)/2.
export const pairD2 = (w, D02) => D02 / ((1 + 2 * w) * (1 + 2 * w));
export const pairDrift = (w, { b, alpha, D02 }) => b - alpha * w - 0.5 * pairD2(w, D02);
export const pairDriftDw = (w, { alpha, D02 }) => -alpha + 2 * D02 / Math.pow(1 + 2 * w, 3);

// The bistable window b_sn < b < b_on = D0^2/2 (G2): with c = D0^2/2, kappa = alpha (deg = 1),
// saddle-node at y* = 1 + 2w* = (4c/kappa)^(1/3), b_sn = c/y*^2 + kappa (y* - 1)/2; non-empty iff c > kappa/4.
export function pairWindow({ alpha, D02 }) {
  const c = D02 / 2, k = alpha, bon = c;
  if (!(c > k / 4)) return { bon, bsn: null, ysn: null, wsn: null, nonempty: false };
  const y = Math.cbrt(4 * c / k), bsn = c / (y * y) + k * (y - 1) / 2;
  return { bon, bsn, ysn: y, wsn: (y - 1) / 2, nonempty: true };
}

// Fixed points of the projected flow on w >= 0: w = 0 when the drift there is <= 0, and the interior roots
// (bracketed on a fine grid in y = 1 + 2w over (1, 1 + 2b/alpha], then bisected). Stability: sign of f'.
export function pairFixedPoints(p) {
  const out = [], f0 = pairDrift(0, p);
  if (f0 <= 0) out.push({ w: 0, stable: f0 < 0, boundary: true });
  const wmax = Math.max(p.b, 0) / p.alpha;
  if (wmax <= 0) return out;
  const n = 4000, f = w => pairDrift(w, p);
  const grid = i => wmax * Math.pow(i / n, 2);        // denser near w = 0
  let a = grid(1e-9 * n), fa = f(a);
  for (let i = 1; i <= n; i++) {
    const c = grid(i), fc = f(c);
    if (fa === 0 && a > 0) out.push({ w: a, stable: pairDriftDw(a, p) < 0 });
    else if (fa * fc < 0) {
      let lo = a, hi = c, flo = fa;
      for (let k = 0; k < 80; k++) { const m = 0.5 * (lo + hi), fm = f(m); if (flo * fm <= 0) hi = m; else { lo = m; flo = fm; } }
      const w = 0.5 * (lo + hi);
      out.push({ w, stable: pairDriftDw(w, p) < 0 });
    }
    a = c; fa = fc;
  }
  return out;
}

// Branches of fixed points against b (for the bifurcation panel): w = 0 (stable for b < c) and the interior
// curve parametrised by y = 1 + 2w: b(y) = c/y^2 + kappa (y - 1)/2, stable where db/dy > 0 (y > y*).
export function pairBranches({ alpha, D02 }, bmax) {
  const c = D02 / 2, k = alpha, win = pairWindow({ alpha, D02 });
  const stable = [], unstable = [];
  const ystar = win.nonempty ? win.ysn : 1;
  const ymax = 1 + 2 * (bmax + c) / k;
  const n = 400;
  if (win.nonempty) for (let i = 0; i <= n; i++) {          // unstable part, y in (1, y*)
    const y = 1 + (ystar - 1) * i / n, b = c / (y * y) + k * (y - 1) / 2;
    unstable.push([b, (y - 1) / 2]);
  }
  for (let i = 0; i <= n; i++) {                              // stable part, y in [y*, ymax], denser near y*
    const y = ystar + (ymax - ystar) * Math.pow(i / n, 2), b = c / (y * y) + k * (y - 1) / 2;
    if (b > bmax * 1.05) break;
    stable.push([b, (y - 1) / 2]);
  }
  return { c, stable, unstable, window: win };
}

// The same pair in the engine: N = 2, B = 2, A = I, a bank of the four rows (+-1, +-1) sqrt(D0^2/2), so that the
// empirical second moment is exactly (D0^2/2) I and D0^2 = E(g1 - g2)^2 exactly (as hebb/sweeps2/fusion.py).
export function makePairSim({ b, alpha, D02 }, w0, dt = 0.05) {
  const s = Math.sqrt(D02 / 2), G = new Float64Array([s, s, s, -s, -s, s, -s, -s]);
  const net = O.makeNetFrom({ N: 2, B: 2, A: new Float64Array([1, 0, 0, 1]), t: 0, klo: 0, khi: 0, W: new Float64Array([0, w0, w0, 0]) });
  return O.createSim({ net, G, S: 4, b, alpha, dt });
}
// advance a pair sim for at most `budgetMs`; returns true when it has stopped (|dw|/dt < tol) or hit maxSteps
export function advancePair(sim, budgetMs, tol = 1e-10, maxSteps = 400000) {
  const t0 = performance.now();
  while (sim.steps < maxSteps) {
    for (let k = 0; k < 200; k++) { sim.step(1); if (sim.lastChange < tol || sim.steps >= maxSteps) return true; }
    if (performance.now() - t0 > budgetMs) return false;
  }
  return true;
}

// ---------------------------------------------------------------- the live network
// Seed streams: attachments and threshold offsets from rng(1000 + seed); training bank from rng(2000 + seed);
// evaluation bank from rng(3000 + seed). Changing Sigma keeps the underlying standard-normal draws (G = Z Sigma^1/2).
export function createFusion(p0) {
  const F = { p: clampParams(p0) };
  let Sigma = null;

  function banks() {
    Sigma = sigmaById(F.p.sig).make();
    F.G = O.sampleInputs(Sigma, B, S_TRAIN, O.makeRng(2000 + F.p.seed));
    F.Geval = O.sampleInputs(Sigma, B, S_EVAL, O.makeRng(3000 + F.p.seed));
    F.Sigma = Sigma; F.Ueval = null;
  }
  F.build = (p = F.p) => {
    F.p = clampParams(p);
    const { n, s, seed } = F.p;
    // linear neurons (klo = khi = 0). sigmaT: 1 only keeps the random stream, hence the attachments, as before;
    // thresholds have no effect without an on-site potential.
    const net = O.makeNet({ N: n, B, rng: O.makeRng(1000 + seed), s, klo: 0, khi: 0, sigmaT: 1 });
    banks();
    F.net = net; F.N = n;
    F.sim = O.createSim({ net, G: F.G, S: S_TRAIN, b: F.p.b, alpha: F.p.alpha, dt: dtOf(F.p.alpha) });
    F.t = 0; F.envelope = null;
    return F;
  };
  F.reset = () => {                       // cold start: W = 0, same network and banks
    F.net.W.fill(0); F.sim.refresh(true); F.t = 0; F.Ueval = null; F.envelope = null;
    return F;
  };
  // parameter changes that keep W (continuation)
  F.setB = b => { F.p.b = b; F.sim.setParams({ b }); F.envelope = null; };
  F.setAlpha = a => { F.p.alpha = a; F.sim.setParams({ alpha: a, dt: dtOf(a) }); F.envelope = null; };
  F.done = () => F.sim.residual < STOP_TOL;

  // steps until budgetMs is used (at most maxSteps); returns the number of steps
  F.advance = (budgetMs, maxSteps) => {
    const k = F.sim.stepFor(budgetMs, maxSteps, STOP_TOL);
    F.t += k * F.sim.dt;
    return k;
  };

  // evaluation-bank responses (warm-started), for population claims
  F.evalResponses = () => {
    const e = O.equilibrate(F.net, F.Geval, S_EVAL, { U0: F.Ueval && F.Ueval.length === S_EVAL * F.N ? F.Ueval : null });
    F.Ueval = e.U; F.evalResidual = e.residual;
    return F.Ueval;
  };

  // one frame of analysis. withEval: also the evaluation-bank diameters and spans.
  F.analyse = (withEval = false) => {
    const N = F.N, sim = F.sim, W = sim.net.W, D2 = sim.D2;
    const { labels, K } = O.components(W, N);
    const cs = O.clusterStats(sim.U, S_TRAIN, N, labels, K);
    // rank components: multi-member by size (ties: lowest member index), singletons last
    const order = cs.map((c, k) => k).sort((a, b) => (cs[b].size - cs[a].size) || (cs[a].members[0] - cs[b].members[0]));
    const rankOf = new Int32Array(K).fill(-1);
    let kMulti = 0;
    for (const k of order) if (cs[k].size > 1) rankOf[k] = kMulti++;
    let D2e = null, cse = null;
    if (withEval) {
      const Ue = F.evalResponses();
      D2e = O.dist2(Ue, S_EVAL, N);
      cse = O.clusterStats(Ue, S_EVAL, N, labels, K);
    }
    const comps = [];
    for (const k of order) {
      const c = cs[k];
      let span = 0, spanE = 0;
      const m = c.members;
      for (let a = 0; a < m.length; a++) for (let q = a + 1; q < m.length; q++) {
        const i = m[a], j = m[q];
        span = Math.max(span, D2[i * N + j]);
        if (D2e) spanE = Math.max(spanE, D2e[i * N + j]);
      }
      comps.push({ k, rank: rankOf[k], members: Array.from(m), size: c.size, diamMax: c.diamMax, diamMean: c.diamMean,
        span: Math.sqrt(span), diamMaxEval: cse ? cse[k].diamMax : NaN, spanEval: D2e ? Math.sqrt(spanE) : NaN });
    }
    const rank = new Int32Array(N);
    for (let i = 0; i < N; i++) rank[i] = rankOf[labels[i]];
    let nLive = 0, wmax = 0;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const w = W[i * N + j]; if (w > 1e-9) { nLive++; if (w > wmax) wmax = w; } }
    const multi = comps.filter(c => c.size > 1);
    return {
      t: F.t, b: F.p.b, alpha: F.p.alpha, N, K, kMulti, singles: K - kMulti, largest: comps.length ? comps[0].size : 0,
      comps: multi, labels: Int32Array.from(labels), rank, nLive, wmax,
      W: Float64Array.from(W), D2: Float64Array.from(D2),
      r: sim.residual, eq: sim.eqResidual, evalResidual: withEval ? F.evalResidual : NaN, withEval,
      xySimplex: F.layoutSimplex(),
    };
  };

  // positions: the B = 3 simplex (rows of H)
  F.layoutSimplex = () => {
    const H = O.harmonic(F.net), N = F.N, xy = new Float64Array(2 * N);
    const V = [[0, 1], [-Math.sqrt(3) / 2, -0.5], [Math.sqrt(3) / 2, -0.5]];
    for (let i = 0; i < N; i++) for (let b = 0; b < B; b++) { xy[2 * i] += H[i * B + b] * V[b][0]; xy[2 * i + 1] += H[i * B + b] * V[b][1]; }
    return { xy, H, anchors: V.map(v => v.slice()) };
  };
  // the envelope theorem, numerically: a central difference of the mean equilibrium energy E*(W) in w_ij,
  // against D_ij^2/2, for the strongest live edge and the closest dead pair
  F.envelopeCheck = (h = 1e-4) => {
    const N = F.N, W = F.net.W, D2 = F.sim.D2;
    let li = -1, lj = -1, lw = 0, di = -1, dj = -1, dd = Infinity;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const w = W[i * N + j];
      if (w > lw) { lw = w; li = i; lj = j; }
      if (w === 0 && D2[i * N + j] < dd) { dd = D2[i * N + j]; di = i; dj = j; }
    }
    const Estar = Wx => { const n2 = O.cloneNet(F.net); n2.W = Wx; const e = O.equilibrate(n2, F.G, S_TRAIN, { U0: F.sim.U }); return O.energy(n2, e.U, F.G, S_TRAIN); };
    const one = (i, j) => {
      if (i < 0) return null;
      const Wp = Float64Array.from(W), Wm = Float64Array.from(W);
      Wp[i * N + j] += h; Wp[j * N + i] += h; Wm[i * N + j] -= h; Wm[j * N + i] -= h;
      const fd = (Estar(Wp) - Estar(Wm)) / (2 * h), half = D2[i * N + j] / 2;
      return { i, j, w: W[i * N + j], fd, half, err: Math.abs(fd - half) };
    };
    F.envelope = { live: one(li, lj), dead: one(di, dj) };
    F.envelope.err = Math.max(F.envelope.live ? F.envelope.live.err : 0, F.envelope.dead ? F.envelope.dead.err : 0);
    return F.envelope;
  };

  return F.build(F.p);
}

// ---------------------------------------------------------------- scan in b (cold starts and continuation)
// Cold: every b from W = 0. Down: from the fixed point at the largest b, lowering b along the grid, carrying W.
export const SCAN_B = [0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3];
export const SCAN_ALPHAS = [1, 0.1];
export function summarise(F) {
  const a = F.analyse(false), sr = Math.sqrt(2 * F.p.b);
  let spanMax = 0, diamMax = 0;
  for (const c of a.comps) { spanMax = Math.max(spanMax, c.span); diamMax = Math.max(diamMax, c.diamMax); }
  return { K: a.K, kMulti: a.kMulti, largest: a.largest, spanRange: spanMax / sr, diamMax, labels: a.labels, r: a.r, steps: F.sim.steps };
}
// nestedness (as hebb.onsite_analysis.pair_violation): of the pairs together at the smaller b, the fraction that
// are apart at the larger b (0 = the finer partition refines the coarser one)
export function pairViolation(labSmall, labLarge) {
  const N = labSmall.length; let bad = 0, tot = 0;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (labSmall[i] === labSmall[j]) { tot++; if (labLarge[i] !== labLarge[j]) bad++; }
  return tot ? bad / tot : 0;
}

// The scan as a generator: yields after every chunk of work (budgetMs) so a page can spread it over frames.
// Result: {alpha: {cold: [summary per SCAN_B], down: [...]}} filled in place; progress in res.done / res.total.
export function* scanJob(p, res, budgetMs = 25, capSteps = 40000) {
  res.total = SCAN_ALPHAS.length * 2 * SCAN_B.length; res.done = 0;
  for (const alpha of SCAN_ALPHAS) {
    const cold = [], down = new Array(SCAN_B.length);
    res[alpha] = { cold, down };
    for (const b of SCAN_B) {
      const F = createFusion({ ...p, b, alpha });
      while (!F.done() && F.sim.steps < capSteps) { F.advance(budgetMs, 1e9); yield; }
      cold.push({ b, ...summarise(F), converged: F.sim.residual < CLAIM_TOL }); res.done++; yield;
    }
    const top = SCAN_B.length - 1, F = createFusion({ ...p, b: SCAN_B[top], alpha });
    for (let q = top; q >= 0; q--) {
      F.setB(SCAN_B[q]); const s0 = F.sim.steps;
      while (!F.done() && F.sim.steps - s0 < capSteps) { F.advance(budgetMs, 1e9); yield; }
      down[q] = { b: SCAN_B[q], ...summarise(F), converged: F.sim.residual < CLAIM_TOL }; res.done++; yield;
    }
  }
}
