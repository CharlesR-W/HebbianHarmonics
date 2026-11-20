// Chapter "basics": the model side (no DOM, so every claim on the page can be checked under node).
// - The small network of figures 1-4 (the same ten neurons and three inputs as the talk): its harmonic embedding,
//   what happens to it when one synapse is changed, voltages relaxing after a stimulus changes.
// - Gaussian samples of the inputs for the covariance figure.
// - The covariance rule (figure 5) runs on engine.js (parity-tested against hebb/core.py), with the perron
//   chapter's parameter names; see ch_basics.js.
// - The distance rule (figure 6) on a small linear network with exact expectations over a given input covariance:
//   H from the weights, D^2, the averaged equilibrium energy E*(W) and the objective J(W).

export const B = 3;

// ------------------------------------------------------------ small dense linear algebra
/** Solve M X = R for symmetric positive definite M (n x n, row-major, copied) and R (n x m). Returns X. */
export function solveSPD(M, n, R, m) {
  const L = Float64Array.from(M), X = Float64Array.from(R);
  for (let j = 0; j < n; j++) {
    let s = L[j * n + j];
    for (let k = 0; k < j; k++) s -= L[j * n + k] * L[j * n + k];
    if (!(s > 0)) throw new Error("solveSPD: matrix not positive definite");
    const d = Math.sqrt(s); L[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let t = L[i * n + j];
      for (let k = 0; k < j; k++) t -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = t / d;
    }
  }
  for (let c = 0; c < m; c++) {
    for (let i = 0; i < n; i++) { let t = X[i * m + c]; for (let k = 0; k < i; k++) t -= L[i * n + k] * X[k * m + c]; X[i * m + c] = t / L[i * n + i]; }
    for (let i = n - 1; i >= 0; i--) { let t = X[i * m + c]; for (let k = i + 1; k < n; k++) t -= L[k * n + i] * X[k * m + c]; X[i * m + c] = t / L[i * n + i]; }
  }
  return X;
}

/** Harmonic embedding H = (diag(d) - W)^{-1} A, d_i = sum_j W_ij + sum_b A_ib. W: N x N symmetric, A: N x B. */
export function harmonic(W, A, N, nB = B) {
  const M = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    let d = 0;
    for (let j = 0; j < N; j++) if (j !== i) { d += W[i * N + j]; M[i * N + j] = -W[i * N + j]; }
    for (let b = 0; b < nB; b++) d += A[i * nB + b];
    M[i * N + i] = d;
  }
  return solveSPD(M, N, A, nB);
}

/** (L_II)^{-1}: the Green's function with the inputs grounded. */
export function green(W, A, N, nB = B) {
  const I = new Float64Array(N * N); for (let i = 0; i < N; i++) I[i * N + i] = 1;
  const M = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    let d = 0;
    for (let j = 0; j < N; j++) if (j !== i) { d += W[i * N + j]; M[i * N + j] = -W[i * N + j]; }
    for (let b = 0; b < nB; b++) d += A[i * nB + b];
    M[i * N + i] = d;
  }
  return solveSPD(M, N, I, N);
}

export const quadForm = (x, S, n = B) => { let q = 0; for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) q += x[a] * S[a * n + c] * x[c]; return q; };
export const rowOf = (H, i, n = B) => Array.from(H.subarray ? H.subarray(i * n, i * n + n) : H.slice(i * n, i * n + n));

// ------------------------------------------------------------ the small network (figures 1-4)
// Ten interior neurons and three inputs; the talk's wiring (talk_a.js) with the synapses onto the inputs twice as
// strong, so the mixtures spread further across the triangle. Edge [a, b, w]: interior 0..9, inputs "B0".."B2".
export const NS = 10;
export const EDGES = [["B0", 0, 3.2], [0, 1, 1], [0, 2, 1], [1, 2, 0.6], [1, 3, 1], [2, 3, 1],
  [1, 4, 1.2], [2, 5, 0.8], [3, 8, 1], [3, 9, 1], [4, 8, 0.7], [5, 9, 1.1], [4, 6, 1],
  [5, 7, 1], [6, "B2", 3.2], [7, "B1", 3.2], [8, 9, 0.8], [6, 8, 1], [7, 9, 1], ["B0", 1, 1.0],
  [4, "B2", 1.2], [5, "B1", 1.2]];

/** The small network as matrices {W (NS x NS), A (NS x 3)}; `extra` = {i, j, w} overrides one lateral synapse. */
export function smallNet(extra = null) {
  const W = new Float64Array(NS * NS), A = new Float64Array(NS * B);
  for (const [a, b, w] of EDGES) {
    const ai = typeof a === "number", bi = typeof b === "number";
    if (ai && bi) { W[a * NS + b] = w; W[b * NS + a] = w; }
    else if (ai) A[a * B + +b[1]] += w;
    else A[b * B + +a[1]] += w;
  }
  if (extra) { W[extra.i * NS + extra.j] = extra.w; W[extra.j * NS + extra.i] = extra.w; }
  return { W, A };
}
export const lateralW = (i, j) => { const { W } = smallNet(); return W[i * NS + j]; };

/** The harmonic embedding of the small network with synapse (i, j) set to w (or the original network). */
export function smallH(extra = null) { const { W, A } = smallNet(extra); return harmonic(W, A, NS); }

/** Effective resistance between interior neurons i and j with the inputs grounded, at the network with w_ij = w0.
 *  Strengthening w_ij to w shrinks h_i - h_j to (h_i - h_j)/(1 + (w - w0) R) and moves every neuron k along the
 *  same direction (Sherman-Morrison; checked numerically in tests below). */
export function effRes(i, j, w0) {
  const { W, A } = smallNet({ i, j, w: w0 });
  const G = green(W, A, NS);
  return G[i * NS + i] + G[j * NS + j] - 2 * G[i * NS + j];
}

/** Embedding as w_ij runs over `ws`: array of H (NS x 3), plus the limit w -> infinity (the pair merged). */
export function pullPath(i, j, ws) {
  const Hs = ws.map(w => smallH({ i, j, w }));
  return { Hs, Hinf: smallH({ i, j, w: 1e7 }) };
}

/** Relaxation: tau du/dt = sum_j w_ij (u_j - u_i) + sum_b a_ib (g_b - u_i) (the downhill flow of the energy),
 *  integrated with explicit Euler substeps. In place; `rate` = 1/tau. */
export function makeRelaxer(rate) {
  const { W, A } = smallNet();
  const deg = new Float64Array(NS);
  for (let i = 0; i < NS; i++) { let d = 0; for (let j = 0; j < NS; j++) d += W[i * NS + j]; for (let b = 0; b < B; b++) d += A[i * B + b]; deg[i] = d; }
  const dmax = Math.max(...deg), hmax = 0.5 / (2 * dmax * rate);
  const du = new Float64Array(NS);
  return function relax(u, g, dt) {
    const n = Math.max(1, Math.ceil(dt / hmax)), h = dt / n;
    for (let s = 0; s < n; s++) {
      for (let i = 0; i < NS; i++) {
        let I = -deg[i] * u[i];
        for (let j = 0; j < NS; j++) I += W[i * NS + j] * u[j];
        for (let b = 0; b < B; b++) I += A[i * B + b] * g[b];
        du[i] = rate * I;
      }
      for (let i = 0; i < NS; i++) u[i] += h * du[i];
    }
    return u;
  };
}
/** Slowest relaxation rate of the small network (smallest eigenvalue of L_II), by inverse iteration. */
export function slowestRate() {
  const { W, A } = smallNet(), G = green(W, A, NS);
  let v = new Float64Array(NS).fill(1), lam = 0;
  for (let it = 0; it < 200; it++) {
    const x = new Float64Array(NS);
    for (let i = 0; i < NS; i++) for (let j = 0; j < NS; j++) x[i] += G[i * NS + j] * v[j];
    const nrm = Math.hypot(...x); lam = nrm / Math.hypot(...v); v = x.map(t => t / nrm);
  }
  return 1 / lam;
}

// ------------------------------------------------------------ random numbers and input samples
export function rng(seed) {
  let a = seed >>> 0;
  const uni = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const normal = () => { let u = 0; while (u <= 1e-12) u = uni(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * uni()); };
  return { uni, normal };
}
/** Lower Cholesky factor of a 3 x 3 covariance (PSD, tiny ridge for safety). */
export function chol3(S) {
  const L = new Float64Array(9);
  for (let j = 0; j < 3; j++) {
    let s = S[j * 3 + j] + 1e-12; for (let k = 0; k < j; k++) s -= L[j * 3 + k] ** 2;
    L[j * 3 + j] = Math.sqrt(Math.max(s, 0));
    for (let i = j + 1; i < 3; i++) { let t = S[i * 3 + j]; for (let k = 0; k < j; k++) t -= L[i * 3 + k] * L[j * 3 + k]; L[i * 3 + j] = L[j * 3 + j] > 0 ? t / L[j * 3 + j] : 0; }
  }
  return L;
}
/** n zero-mean Gaussian input vectors with covariance S (n x 3, row-major); same z's for a given seed. */
export function sampleInputs(S, n, seed) {
  const r = rng(seed), L = chol3(S), G = new Float64Array(3 * n);
  for (let s = 0; s < n; s++) {
    const z = [r.normal(), r.normal(), r.normal()];
    for (let i = 0; i < 3; i++) { let v = 0; for (let k = 0; k <= i; k++) v += L[i * 3 + k] * z[k]; G[3 * s + i] = v; }
  }
  return G;
}
export function sigma3(r12 = 0, r13 = 0, r23 = 0) { return Float64Array.from([1, r12, r13, r12, 1, r23, r13, r23, 1]); }

// ------------------------------------------------------------ the distance rule (figure 6)
// Linear network, N interior neurons, fixed attachments A (N x 3, each row Dirichlet(1) shares, total 1), plastic
// lateral W. Expectations are exact over a given Sigma (zero-mean inputs): D^2_ij = (h_i - h_j)' Sigma (h_i - h_j).
//   E*(W) = E_g min_u E_W(u; g) = 1/2 [ sum_{i<j} w_ij D^2_ij + sum_{i,b} a_ib (h_i - e_b)' Sigma (h_i - e_b) ]
//   J(W)  = E*(W) + sum_{i<j} (alpha/2 w_ij^2 - b w_ij),   dJ/dw_ij = D^2_ij/2 + alpha w_ij - b   (envelope theorem)
export function dirichletAttach(N, seed) {
  const r = rng(seed), A = new Float64Array(N * B);
  for (let i = 0; i < N; i++) {
    let s = 0; for (let b = 0; b < B; b++) { const e = -Math.log(Math.max(1e-300, r.uni())); A[i * B + b] = e; s += e; }
    for (let b = 0; b < B; b++) A[i * B + b] /= s;
  }
  return A;
}
export function dist2(H, N, S) {
  const D2 = new Float64Array(N * N), d = [0, 0, 0];
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    for (let b = 0; b < B; b++) d[b] = H[i * B + b] - H[j * B + b];
    const q = quadForm(d, S); D2[i * N + j] = q; D2[j * N + i] = q;
  }
  return D2;
}
export function meanEnergy(W, A, H, N, S) {
  const D2 = dist2(H, N, S);
  let e = 0;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) e += W[i * N + j] * D2[i * N + j];
  const d = [0, 0, 0];
  for (let i = 0; i < N; i++) for (let b = 0; b < B; b++) {
    if (!A[i * B + b]) continue;
    for (let c = 0; c < B; c++) d[c] = H[i * B + c] - (c === b ? 1 : 0);
    e += A[i * B + b] * quadForm(d, S);
  }
  return { E: e / 2, D2 };
}
export function synapseCost(W, N, b, alpha) {
  let c = 0; for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const w = W[i * N + j]; c += alpha / 2 * w * w - b * w; }
  return c;
}
/** Run the distance rule (projected Euler) from W = 0. Returns frames {t, H, E, cost, J, live} at every `every` steps. */
export function runDistance({ N, A, S, b, alpha, dt, T, every = 1 }) {
  const W = new Float64Array(N * N), frames = [];
  const steps = Math.round(T / dt);
  for (let k = 0; k <= steps; k++) {
    const H = harmonic(W, A, N), { E, D2 } = meanEnergy(W, A, H, N, S), cost = synapseCost(W, N, b, alpha);
    if (k % every === 0 || k === steps) {
      let live = 0; for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (W[i * N + j] > 0) live++;
      frames.push({ t: k * dt, H, E, cost, J: E + cost, live, W: Float64Array.from(W) });
    }
    if (k === steps) break;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const w = Math.max(0, W[i * N + j] + dt * (b - alpha * W[i * N + j] - D2[i * N + j] / 2));
      W[i * N + j] = w; W[j * N + i] = w;
    }
  }
  return frames;
}
