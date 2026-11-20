// Chapter "allocation": model side, DOM-free so that node can test it (tests: run this file's functions against
// the numbers in data/sweeps2/allocation.json). Three pieces, one per section of the chapter:
//   (a) the base model with block attachments (hebb/sweeps2/allocation.py a_unit), simulated with onsite_engine.js;
//   (b) plastic attachments under the energy rule (b_unit): projected gradient flow on the rows of A;
//   (c) the added similarity-matching principle (sm_solve): projected gradient on N x B response vectors.
import * as O from "./onsite_engine.js";

export const RHO = 0.8;                 // within-block correlation (sigma_blocks)
export const BLK = 4, PER = 3, B = BLK * PER;
export const RES_TOL = 1e-6, RUN_TOL = 1e-9;

// ---------------------------------------------------------------- shared
// Four equal blocks of three inputs, within-block correlation rho, block variances vars, no cross-block
// correlation (onsite_analysis.sigma_blocks).
export function sigmaBlocks(vars, rho = RHO) {
  const S = new Float64Array(B * B);
  for (let k = 0; k < BLK; k++) for (let a = 0; a < PER; a++) for (let c = 0; c < PER; c++) {
    const i = k * PER + a, j = k * PER + c;
    S[i * B + j] = vars[k] * (a === c ? 1 : rho);
  }
  return S;
}
export const blockOf = b => Math.floor(b / PER);

// standard-normal bank Z (S x B); a bank with covariance Sigma is Z Sigma^{1/2}. Keeping Z fixed while a block's
// variance changes only rescales that block's columns, so the sampling noise does not change with the slider.
export function normalBank(S, seed) {
  const rng = O.makeRng(seed), Z = new Float64Array(S * B);
  for (let i = 0; i < Z.length; i++) Z[i] = rng.normal();
  return Z;
}
export function colour(Z, S, Sigma) {
  const R = O.sqrtm(Sigma, B), G = new Float64Array(S * B);
  for (let s = 0; s < S; s++) for (let j = 0; j < B; j++) {
    let x = 0; for (let k = 0; k < B; k++) x += Z[s * B + k] * R[k * B + j];
    G[s * B + j] = x;
  }
  return G;
}

// ---------------------------------------------------------------- (a) base model, block attachments
// Each neuron attached to the three inputs of ONE block, Dirichlet(1) shares, total conductance 1
// (onsite_analysis.block_attachments with s = 3 = block size). counts[k] neurons on block k; each block draws from
// its own stream so that changing one block's count leaves the other blocks' attachments untouched.
export function blockAttachments(counts, seed) {
  const N = counts.reduce((a, b) => a + b, 0), A = new Float64Array(N * B), home = new Int32Array(N);
  let i = 0;
  for (let k = 0; k < BLK; k++) {
    const rng = O.makeRng(1000 * (seed + 1) + k);
    for (let m = 0; m < counts[k]; m++, i++) {
      const e = [0, 0, 0].map(() => rng.exponential(1)), s = e[0] + e[1] + e[2];
      for (let a = 0; a < PER; a++) A[i * B + k * PER + a] = e[a] / s;
      home[i] = k;
    }
  }
  return { A, home, N };
}

// Build the live base-model simulation: linear neurons (V = 0), alpha = 1, dt = min(0.5, 1/alpha) as the sweep.
export function makeBase({ v1 = 1, n1 = 12, b = 0.02, seed = 0, S = 256 }) {
  const vars = [v1, 1, 1, 1], counts = [n1, 12, 12, 12];
  const Sigma = sigmaBlocks(vars);
  const { A, home, N } = blockAttachments(counts, seed);
  const G = colour(normalBank(S, 5000 + seed), S, Sigma);
  const net = O.makeNetFrom({ N, B, A, t: 0, klo: 0, khi: 0 });
  const sim = O.createSim({ net, G, S, b, alpha: 1, dt: 0.5 });
  return { sim, Sigma, vars, counts, home, N, b, S };
}

// Per-block analysis in the Sigma_g metric (linear case: population claims use Sigma_g analytically).
// For a neuron whose embedding h lies on one block's three inputs, D^2 to any other such point is
// v (1 - rho) |dh|^2 (dh sums to 0), so the block's simplex drawn as an equilateral triangle of side
// sqrt(2 v (1 - rho)) is an exact isometry; the drawing uses it.
export function analyseBase(base) {
  const { sim, Sigma, home, N } = base, W = sim.net.W;
  const H = O.harmonic(sim.net);
  const { labels, K } = O.components(W, N);
  const quad = (x) => { let s = 0; for (let a = 0; a < B; a++) { if (!x[a]) continue; for (let c = 0; c < B; c++) s += x[a] * Sigma[a * B + c] * x[c]; } return s; };
  const comps = Array.from({ length: K }, () => []);
  for (let i = 0; i < N; i++) comps[labels[i]].push(i);
  // components with members from more than one block (never seen in the sweep; reported if they occur)
  let mixed = 0, offBlock = 0;
  for (const c of comps) if (new Set(c.map(i => home[i])).size > 1) mixed++;
  for (let i = 0; i < N; i++) { let m = 0; for (let a = 0; a < B; a++) if (blockOf(a) !== home[i]) m += Math.abs(H[i * B + a]); offBlock = Math.max(offBlock, m); }
  const blocks = [];
  for (let k = 0; k < BLK; k++) {
    const members = []; for (let i = 0; i < N; i++) if (home[i] === k) members.push(i);
    const cs = comps.filter(c => home[c[0]] === k && c.every(i => home[i] === k));
    const stats = cs.map(c => {
      const hb = new Float64Array(B); for (const i of c) for (let a = 0; a < B; a++) hb[a] += H[i * B + a] / c.length;
      let dmax = 0, dsum = 0, span = 0;
      for (const i of c) { const d = new Float64Array(B); for (let a = 0; a < B; a++) d[a] = H[i * B + a] - hb[a]; const x = Math.sqrt(Math.max(0, quad(d))); dmax = Math.max(dmax, x); dsum += x; }
      for (let p = 0; p < c.length; p++) for (let q = p + 1; q < c.length; q++) { const d = new Float64Array(B); for (let a = 0; a < B; a++) d[a] = H[c[p] * B + a] - H[c[q] * B + a]; span = Math.max(span, Math.sqrt(Math.max(0, quad(d)))); }
      return { members: c, size: c.length, diamMax: dmax, diamMean: dsum / c.length, span };
    }).sort((x, y) => y.size - x.size || x.members[0] - y.members[0]);
    blocks.push({ k, v: base.vars[k], members, comps: stats, K: stats.length });
  }
  return { H, labels, K, blocks, mixed, offBlock, residual: sim.residual, eqResidual: sim.eqResidual, lastChange: sim.lastChange, steps: sim.steps };
}

// ---------------------------------------------------------------- (b) plastic attachments
// Euclidean projection of each row of V (n x m) onto {a >= 0, sum a = z}.
export function projSimplexRow(v, z = 1) {
  const u = Array.from(v).sort((p, q) => q - p);
  let css = 0, rho = 0, tau = 0;
  for (let k = 0; k < u.length; k++) { css += u[k]; const t = (css - z) / (k + 1); if (u[k] - t > 0) { rho = k + 1; tau = t; } }
  const out = new Float64Array(v.length);
  for (let k = 0; k < v.length; k++) out[k] = Math.max(v[k] - tau, 0);
  return out;
}

export const VARS_B = [2, 1, 0.5, 0.25];

// Joint projected gradient flow on J over the rows of A (fixed degree 1) and, optionally, the lateral W
// (alpha = 1): dA_ib = -alphaA a_ib - 1/2 E[(u_i - g_b)^2], dW_ij = b - w_ij - 1/2 D_ij^2.
export function makePlastic({ alphaA = 1, lateral = false, b = 0.1, seed = 0, N = 40, S = 256 }) {
  const Sigma = sigmaBlocks(VARS_B);
  const rng = O.makeRng(7000 + seed);
  const A0 = new Float64Array(N * B);
  for (let i = 0; i < N; i++) { let s = 0; const e = []; for (let a = 0; a < B; a++) { e.push(rng.exponential(1)); s += e[a]; } for (let a = 0; a < B; a++) A0[i * B + a] = e[a] / s; }
  const net = O.makeNetFrom({ N, B, A: A0, t: new Float64Array(N), klo: 0, khi: 0 });   // linear neurons
  const A = net.A;                                     // makeNetFrom copies; the flow edits the net's own array
  const G = colour(normalBank(S, 7200 + seed), S, Sigma);
  const dt = Math.min(0.2, 1 / alphaA);
  const st = { net, G, S, N, Sigma, alphaA, lateral, b, dt, steps: 0, res: Infinity, U: null, eqRes: 0 };
  const G2 = new Float64Array(B);
  for (let s = 0; s < S; s++) for (let a = 0; a < B; a++) G2[a] += G[s * B + a] ** 2 / S;
  st.step = () => {
    const r = O.equilibrate(net, G, S, st.U ? { U0: st.U } : {});
    const U = r.U; st.U = U; st.eqRes = r.residual;
    let ch = 0;
    if (lateral) {
      const D2 = O.dist2(U, S, N), dW = O.driftFromD2(net.W, D2, N, b, 1);
      for (let p = 0; p < N * N; p++) { const w = Math.max(net.W[p] + dt * dW[p], 0); ch = Math.max(ch, Math.abs(w - net.W[p])); net.W[p] = w; }
    }
    // D2B[i, b] = E_s (u_i - g_b)^2 on the training bank
    const U2 = new Float64Array(N), UG = new Float64Array(N * B);
    for (let s = 0; s < S; s++) for (let i = 0; i < N; i++) {
      const u = U[s * N + i]; U2[i] += u * u / S;
      for (let a = 0; a < B; a++) UG[i * B + a] += u * G[s * B + a] / S;
    }
    for (let i = 0; i < N; i++) {
      const v = new Float64Array(B);
      for (let a = 0; a < B; a++) v[a] = A[i * B + a] + dt * (-alphaA * A[i * B + a] - 0.5 * (U2[i] + G2[a] - 2 * UG[i * B + a]));
      const row = projSimplexRow(v, 1);
      for (let a = 0; a < B; a++) { ch = Math.max(ch, Math.abs(row[a] - A[i * B + a])); A[i * B + a] = row[a]; }
    }
    st.res = ch / dt; st.steps++;
    return st.res;
  };
  return st;
}

export function plasticSummary(st) {
  const { net, N } = st, A = net.A;
  const mass = [], home = new Int32Array(N), maj = [0, 0, 0, 0], argm = [0, 0, 0, 0], share = [0, 0, 0, 0];
  let purity = 0, support = 0;
  for (let i = 0; i < N; i++) {
    const m = [0, 0, 0, 0]; let tot = 0;
    for (let a = 0; a < B; a++) { m[blockOf(a)] += A[i * B + a]; tot += A[i * B + a]; if (A[i * B + a] > 1e-6) support++; }
    for (let k = 0; k < BLK; k++) { m[k] /= tot; share[k] += m[k] / N; }
    let best = 0; for (let k = 1; k < BLK; k++) if (m[k] > m[best]) best = k;
    home[i] = best; argm[best]++; if (m[best] > 0.5) maj[best]++; purity += m[best] / N;
    mass.push(m);
  }
  return { mass, home, maj, argm, share, purity, support: support / N };
}

// ---------------------------------------------------------------- (c) similarity matching
export const SPECTRA = { lam8: [3.0, 2.0, 1.5, 1.0, 0.6, 0.3, 0.1, 0.05], pow16: Array.from({ length: 16 }, (_, k) => 4 / (k + 1)) };
export const NS = { lam8: 60, pow16: 80 };
export const INGREDIENTS = [
  { id: "none", label: "none", fusion: 0, nonneg: null },
  { id: "fusion", label: "fusion term", fusion: 1, nonneg: null },
  { id: "nonneg_eig", label: "non-negative, eigenbasis", fusion: 0, nonneg: "eig" },
  { id: "nonneg_rotated", label: "non-negative, rotated basis", fusion: 0, nonneg: "rot" },
  { id: "nonneg_eig+fusion", label: "non-negative + fusion", fusion: 1, nonneg: "eig" },
];

// q_a = (lam_a - mu)_+ with sum q = total (mu may be negative); bisection as onsite_analysis.water_fill
export function waterFill(lam, total) {
  let lo = Math.min(...lam) - total, hi = Math.max(...lam), mu = 0;
  const sum = m => lam.reduce((s, l) => s + Math.max(l - m, 0), 0);
  for (let it = 0; it < 200; it++) { mu = 0.5 * (lo + hi); if (sum(mu) > total) lo = mu; else hi = mu; }
  mu = 0.5 * (lo + hi);
  return { q: lam.map(l => Math.max(l - mu, 0)), mu };
}
// best counts for pure neurons: argmin 1/2 sum (lam_a - n_a nu)^2 with sum n = N (greedy is exact)
export function integerFill(lam, N, nu) {
  const n = lam.map(() => 0);
  for (let t = 0; t < N; t++) {
    let best = 0, bg = -Infinity;
    lam.forEach((l, a) => { const g = (l - n[a] * nu) ** 2 - (l - (n[a] + 1) * nu) ** 2; if (g > bg) { bg = g; best = a; } });
    n[best]++;
  }
  return n;
}

// The live optimiser: projected gradient (lr 0.02) on
//   f(X) = 1/2 ||Sigma - X^T X||_F^2 - 1/(2 alpha) sum_{i<j} (b - |x_i - x_j|^2/2)_+^2      (alpha = 1)
// with rows on the sphere |x_i|^2 = nu, optionally also R^T x_i >= 0. Sigma = diag(lam): the eigenbasis is the
// coordinate basis. Same scheme as hebb/sweeps2/allocation.py sm_solve; the random init and the rotation R come
// from this page's generator, not numpy's, so runs match the sweep in distribution, not sample for sample.
export function makeSM({ spec = "lam8", nu = 0.1, ing = "none", b = 0.03, seed = 0, lr = 0.02 }) {
  const lam = SPECTRA[spec], Bn = lam.length, N = NS[spec];
  const ingr = INGREDIENTS.find(x => x.id === ing) || INGREDIENTS[0];
  const bf = ingr.fusion ? b : 0;
  const R = ingr.nonneg === "rot" ? O.randomOrthogonal(Bn, O.makeRng(777)) : null;   // columns orthonormal
  const nonneg = !!ingr.nonneg;
  const rng = O.makeRng(9000 + seed);
  const X = new Float64Array(N * Bn);
  for (let p = 0; p < X.length; p++) X[p] = rng.normal();
  const Gm = new Float64Array(Bn * Bn), grad = new Float64Array(N * Bn), Y = new Float64Array(N * Bn), Xn = new Float64Array(N * Bn);
  const sn = Math.sqrt(nu);
  function project(Z, out) {
    for (let i = 0; i < N; i++) {
      const o = i * Bn;
      if (nonneg) {
        // y = R^T z clipped at 0 (R = I in the eigenbasis); a row with no positive entry goes to its best axis
        let any = false, bi = 0, bv = -Infinity;
        for (let c = 0; c < Bn; c++) {
          let y = 0;
          if (R) { for (let a = 0; a < Bn; a++) y += Z[o + a] * R[a * Bn + c]; } else y = Z[o + c];
          if (y > bv) { bv = y; bi = c; }
          Y[o + c] = Math.max(y, 0); if (y > 0) any = true;
        }
        if (!any) Y[o + bi] = 1;
        for (let a = 0; a < Bn; a++) {
          let x = 0;
          if (R) { for (let c = 0; c < Bn; c++) x += Y[o + c] * R[a * Bn + c]; } else x = Y[o + a];
          out[o + a] = x;
        }
      } else for (let a = 0; a < Bn; a++) out[o + a] = Z[o + a];
      let nn = 0; for (let a = 0; a < Bn; a++) nn += out[o + a] ** 2;
      const s = sn / Math.sqrt(nn);
      for (let a = 0; a < Bn; a++) out[o + a] *= s;
    }
    return out;
  }
  function gradient(Xc, g) {
    Gm.fill(0);
    for (let i = 0; i < N; i++) for (let a = 0; a < Bn; a++) { const x = Xc[i * Bn + a]; if (!x) continue; for (let c = a; c < Bn; c++) Gm[a * Bn + c] += x * Xc[i * Bn + c]; }
    for (let a = 0; a < Bn; a++) { for (let c = 0; c < a; c++) Gm[a * Bn + c] = Gm[c * Bn + a]; Gm[a * Bn + a] -= lam[a]; }
    for (let i = 0; i < N; i++) for (let a = 0; a < Bn; a++) { let s = 0; for (let c = 0; c < Bn; c++) s += Xc[i * Bn + c] * Gm[c * Bn + a]; g[i * Bn + a] = 2 * s; }
    if (bf > 0) for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      let d2 = 0; for (let a = 0; a < Bn; a++) d2 += (Xc[i * Bn + a] - Xc[j * Bn + a]) ** 2;
      const w = bf - d2 / 2; if (w <= 0) continue;
      for (let a = 0; a < Bn; a++) { const d = w * (Xc[i * Bn + a] - Xc[j * Bn + a]); g[i * Bn + a] += d; g[j * Bn + a] -= d; }
    }
    return g;
  }
  project(X, X);
  const st = { lam, N, Bn, nu, ing: ingr.id, b: bf, X, it: 0, lr, gradMap: NaN };
  st.step = (n = 1) => {
    for (let k = 0; k < n; k++) {
      gradient(X, grad);
      for (let p = 0; p < X.length; p++) Xn[p] = X[p] - lr * grad[p];
      project(Xn, X); st.it++;
    }
  };
  // stationarity: max |X - P(X - lr grad)| / lr (the sweep's grad_map)
  st.stationarity = () => {
    gradient(X, grad);
    for (let p = 0; p < X.length; p++) Xn[p] = X[p] - lr * grad[p];
    const P = project(Xn, new Float64Array(X.length));
    let m = 0; for (let p = 0; p < X.length; p++) m = Math.max(m, Math.abs(X[p] - P[p]));
    return (st.gradMap = m / lr);
  };
  return st;
}

export function smMetrics(st) {
  const { X, N, Bn, nu, lam } = st;
  const M = new Float64Array(Bn * Bn);
  for (let i = 0; i < N; i++) for (let a = 0; a < Bn; a++) for (let c = 0; c < Bn; c++) M[a * Bn + c] += X[i * Bn + a] * X[i * Bn + c];
  const q = lam.map((_, a) => M[a * Bn + a]);
  let off = 0; for (let a = 0; a < Bn; a++) for (let c = 0; c < Bn; c++) if (a !== c) off = Math.max(off, Math.abs(M[a * Bn + c]));
  const { q: qp, mu } = waterFill(lam, N * nu);
  const comp = new Float64Array(N * Bn), purity = new Float64Array(N), home = new Int32Array(N);
  const counts = lam.map(() => 0), countsPure = lam.map(() => 0);
  for (let i = 0; i < N; i++) {
    let bi = 0, bv = -1;
    for (let a = 0; a < Bn; a++) { const x = X[i * Bn + a] ** 2 / nu; comp[i * Bn + a] = x; if (x > bv) { bv = x; bi = a; } }
    purity[i] = bv; home[i] = bi; counts[bi]++; if (bv > 0.9) countsPure[bi]++;
  }
  const law = qp.map(x => x / nu), nInt = integerFill(lam, N, nu);
  let qDev = 0, cDev = 0, cDevI = 0, fMain = 0, fStar = 0;
  for (let a = 0; a < Bn; a++) { qDev += Math.abs(q[a] - qp[a]); cDev += Math.abs(counts[a] - law[a]); cDevI += Math.abs(counts[a] - nInt[a]); fStar += 0.5 * (lam[a] - qp[a]) ** 2; }
  for (let a = 0; a < Bn; a++) for (let c = 0; c < Bn; c++) fMain += 0.5 * ((a === c ? lam[a] : 0) - M[a * Bn + c]) ** 2;
  let pure = 0; for (let i = 0; i < N; i++) if (purity[i] > 0.9) pure++;
  return { q, qPred: qp, mu, off, comp, purity, home, counts, countsPure, law, nInt,
    qDev: qDev / (N * nu), countDev: cDev / N, countDevInt: cDevI / N, excess: fMain - fStar, pureFrac: pure / N };
}
