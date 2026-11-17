// onsite_engine.js — browser engine for the v2 model (SPEC.md §1): a conductance network
// with convex two-slope on-site potentials and the distance-based plasticity rule.
// Pure ES module: no DOM, no imports; runs in node and browsers.
// Mirrors hebb/onsite.py (the numpy reference); cross-checked by tests/test_onsite_parity.py.
//
//     E_W(u; g) = 1/2 sum_{i<j} w_ij (u_i-u_j)^2 + 1/2 sum_{i,b} a_ib (u_i-g_b)^2 + sum_i V_i(u_i)
//     V_i(u)    = klo_i/2 (u-t_i)_-^2 + khi_i/2 (u-t_i)_+^2
//     dw_ij/dt  = b - alpha w_ij - 1/2 E_g[(u_i-u_j)^2],   projected at w >= 0
//
// Conventions: every matrix is a Float64Array in row-major order with its shape passed
// explicitly.  A is N×B (A[i*B+b]), W is N×N (symmetric, zero diagonal), the input bank G
// is S×B (one sample per row), responses U are S×N (U[s*N+i]), like the reference.
// Expectations are plain (uncentred) means over the S rows of G.  See ONSITE_ENGINE.md.

export const VERSION = '2.0.0';

// ───────────────────────────── PRNG ─────────────────────────────
// mulberry32 (bit-identical to engine.js makeRng and hebb.core.JsRng).

export function makeRng(seed = 0) {
  let a = seed | 0;
  function uniform() {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  const exponential = (scale = 1) => -scale * Math.log(1 - uniform());
  return {
    uniform,
    normal() { const u1 = uniform(), u2 = uniform(); return Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2); },
    exponential,
    int(n) { return Math.floor(uniform() * n); },
    /** Unit-variance Laplace: random sign (one uniform) times Exp(1) (one uniform), / sqrt 2. */
    laplace() { const sg = uniform() < 0.5 ? -1 : 1; return sg * exponential(1) * Math.SQRT1_2; },
  };
}

const asRng = r => (r && typeof r.uniform === 'function' ? r : makeRng(r | 0));

// ─────────────────────────── linear algebra ───────────────────────────

/** In-place Cholesky of the n×n SPD matrix in M (lower triangle overwritten with L;
 *  the strict upper triangle is left as is).  Returns false if not numerically SPD. */
export function cholesky(M, n) {
  for (let i = 0; i < n; i++) {
    const ri = i * n;
    for (let j = 0; j <= i; j++) {
      const rj = j * n;
      let s = M[ri + j];
      for (let k = 0; k < j; k++) s -= M[ri + k] * M[rj + k];
      if (i === j) { if (!(s > 0)) return false; M[ri + i] = Math.sqrt(s); }
      else M[ri + j] = s / M[rj + j];
    }
  }
  return true;
}

/** Solve (L L^T) x = y in place for one vector x of length n (x[off .. off+n)). */
export function cholSolveVec(L, n, x, off = 0) {
  for (let i = 0; i < n; i++) {
    const ri = i * n;
    let s = x[off + i];
    for (let k = 0; k < i; k++) s -= L[ri + k] * x[off + k];
    x[off + i] = s / L[ri + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    const ri = i * n;
    const xi = x[off + i] / L[ri + i];
    x[off + i] = xi;
    for (let k = 0; k < i; k++) x[off + k] -= L[ri + k] * xi;
  }
  return x;
}

/** Solve the SPD system M X = R for an n×m right-hand side (row-major, column c is one
 *  system).  M is not modified.  Throws if M is not SPD. */
export function solveSPD(M, n, R, m) {
  const L = Float64Array.from(M.subarray(0, n * n));
  if (!cholesky(L, n)) throw new Error('solveSPD: matrix not SPD');
  const X = Float64Array.from(R.subarray(0, n * m)), col = new Float64Array(n);
  for (let c = 0; c < m; c++) {
    for (let i = 0; i < n; i++) col[i] = X[i * m + c];
    cholSolveVec(L, n, col);
    for (let i = 0; i < n; i++) X[i * m + c] = col[i];
  }
  return X;
}

/** Symmetric eigendecomposition (Householder + implicit QL, EISPACK tred2/tql2 as in
 *  JAMA; same code as engine.js).  Returns {values (descending), vectors} with
 *  vectors[i*n + k] = component i of eigenvector k (numpy's column convention). */
export function eigh(A, n) {
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) V[i * n + j] = 0.5 * (A[i * n + j] + A[j * n + i]);
  const d = new Float64Array(n), e = new Float64Array(n);
  if (n > 0) { tred2(V, d, e, n); tql2(V, d, e, n); }
  const idx = Array.from({ length: n }, (_, k) => k).sort((x, y) => d[y] - d[x]);
  const values = new Float64Array(n), vectors = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    values[k] = d[idx[k]];
    for (let i = 0; i < n; i++) vectors[i * n + k] = V[i * n + idx[k]];
  }
  return { values, vectors };
}

function tred2(V, d, e, n) {
  for (let j = 0; j < n; j++) d[j] = V[(n - 1) * n + j];
  for (let i = n - 1; i > 0; i--) {
    let scale = 0, h = 0;
    for (let k = 0; k < i; k++) scale += Math.abs(d[k]);
    if (scale === 0) {
      e[i] = d[i - 1];
      for (let j = 0; j < i; j++) { d[j] = V[(i - 1) * n + j]; V[i * n + j] = 0; V[j * n + i] = 0; }
    } else {
      for (let k = 0; k < i; k++) { d[k] /= scale; h += d[k] * d[k]; }
      let f = d[i - 1], g = Math.sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g; h -= f * g; d[i - 1] = f - g;
      for (let j = 0; j < i; j++) e[j] = 0;
      for (let j = 0; j < i; j++) {
        f = d[j]; V[j * n + i] = f; g = e[j] + V[j * n + j] * f;
        for (let k = j + 1; k <= i - 1; k++) { g += V[k * n + j] * d[k]; e[k] += V[k * n + j] * f; }
        e[j] = g;
      }
      f = 0;
      for (let j = 0; j < i; j++) { e[j] /= h; f += e[j] * d[j]; }
      const hh = f / (h + h);
      for (let j = 0; j < i; j++) e[j] -= hh * d[j];
      for (let j = 0; j < i; j++) {
        f = d[j]; g = e[j];
        for (let k = j; k <= i - 1; k++) V[k * n + j] -= (f * e[k] + g * d[k]);
        d[j] = V[(i - 1) * n + j]; V[i * n + j] = 0;
      }
    }
    d[i] = h;
  }
  for (let i = 0; i < n - 1; i++) {
    V[(n - 1) * n + i] = V[i * n + i]; V[i * n + i] = 1;
    const h = d[i + 1];
    if (h !== 0) {
      for (let k = 0; k <= i; k++) d[k] = V[k * n + i + 1] / h;
      for (let j = 0; j <= i; j++) {
        let g = 0;
        for (let k = 0; k <= i; k++) g += V[k * n + i + 1] * V[k * n + j];
        for (let k = 0; k <= i; k++) V[k * n + j] -= g * d[k];
      }
    }
    for (let k = 0; k <= i; k++) V[k * n + i + 1] = 0;
  }
  for (let j = 0; j < n; j++) { d[j] = V[(n - 1) * n + j]; V[(n - 1) * n + j] = 0; }
  V[(n - 1) * n + n - 1] = 1;
  e[0] = 0;
}

function tql2(V, d, e, n) {
  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0;
  let f = 0, tst1 = 0;
  const eps = 2 ** -52;
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
    let m = l;
    while (m < n) { if (Math.abs(e[m]) <= eps * tst1) break; m++; }
    if (m > l) {
      let iter = 0;
      do {
        if (++iter > 200) throw new Error('eigh: no convergence');
        let g = d[l];
        let p = (d[l + 1] - g) / (2 * e[l]);
        let r = Math.hypot(p, 1); if (p < 0) r = -r;
        d[l] = e[l] / (p + r); d[l + 1] = e[l] * (p + r);
        const dl1 = d[l + 1];
        let h = g - d[l];
        for (let i = l + 2; i < n; i++) d[i] -= h;
        f += h;
        p = d[m];
        let c = 1, c2 = c, c3 = c, s = 0, s2 = 0;
        const el1 = e[l + 1];
        for (let i = m - 1; i >= l; i--) {
          c3 = c2; c2 = c; s2 = s;
          g = c * e[i]; h = c * p; r = Math.hypot(p, e[i]);
          e[i + 1] = s * r; s = e[i] / r; c = p / r;
          p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          for (let k = 0; k < n; k++) {
            const ki = k * n + i;
            h = V[ki + 1];
            V[ki + 1] = s * V[ki] + c * h;
            V[ki] = c * V[ki] - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p; d[l] = c * p;
      } while (Math.abs(e[l]) > eps * tst1);
    }
    d[l] += f; e[l] = 0;
  }
}

/** V f(Λ) V^T for symmetric A (n×n). */
export function symFn(A, n, fn) {
  const { values, vectors } = eigh(A, n);
  const out = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    const fk = fn(values[k]); if (fk === 0) continue;
    for (let i = 0; i < n; i++) {
      const v = vectors[i * n + k] * fk; if (v === 0) continue;
      for (let j = 0; j < n; j++) out[i * n + j] += v * vectors[j * n + k];
    }
  }
  return out;
}
/** Symmetric PSD square root (negative eigenvalues clipped to 0). */
export const sqrtm = (A, n) => symFn(A, n, x => Math.sqrt(Math.max(x, 0)));

// ─────────────────────────── covariances ───────────────────────────
// Ported from engine.js (same recipes and argument names) plus rank-one and diagonal.

function clampPSD(A, n, unitDiag) {
  const { values } = eigh(A, n);
  if (values[n - 1] >= 0) return Float64Array.from(A.subarray(0, n * n));
  const M = symFn(A, n, x => Math.max(x, 0));
  if (unitDiag) {
    const s = new Float64Array(n);
    for (let i = 0; i < n; i++) s[i] = M[i * n + i] > 0 ? 1 / Math.sqrt(M[i * n + i]) : 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i * n + j] *= s[i] * s[j];
    for (let i = 0; i < n; i++) M[i * n + i] = 1;
  }
  return M;
}

function scaleByVariances(R, n, v) {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i * n + j] = R[i * n + j] * Math.sqrt(v[i] * v[j]);
  return out;
}

/** 3×3 covariance from variances [v0,v1,v2] and correlations [r01, r02, r12]
 *  (correlation matrix clamped to PSD, unit diagonal kept). */
export function cov3({ variances = [1, 1, 1], corr = [0, 0, 0] } = {}) {
  const [a, b, c] = corr;
  const R = clampPSD(Float64Array.from([1, a, b, a, 1, c, b, c, 1]), 3, true);
  return scaleByVariances(R, 3, variances);
}

/** Planted blocks: sizes [n1, n2, ...]; correlation `within` inside a block (scalar or
 *  per-block array), `between` across blocks; per-block variance (scalar or array).
 *  Returns {Sigma, B, labels}. */
export function covBlocks({ sizes = [4, 4, 4], within = 0.5, between = 0.1, variances = 1 } = {}) {
  const B = sizes.reduce((x, y) => x + y, 0);
  const labels = new Int32Array(B);
  let o = 0; sizes.forEach((sz, k) => { for (let q = 0; q < sz; q++) labels[o++] = k; });
  const wv = k => (Array.isArray(within) ? within[k] : within);
  const vv = k => (Array.isArray(variances) ? variances[k] : variances);
  const R = new Float64Array(B * B);
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++)
    R[i * B + j] = i === j ? 1 : labels[i] === labels[j] ? wv(labels[i]) : between;
  const v = Array.from(labels, k => vv(k));
  return { Sigma: scaleByVariances(clampPSD(R, B, true), B, v), B, labels };
}

/** Hierarchical blocks: branching [b1, b2, ...] (B = product), corr[l] = correlation of two
 *  inputs whose deepest common ancestor is at level l (0 = root; last = siblings). */
export function covHierarchical({ branching = [3, 4], corr = [0.1, 0.5], variance = 1 } = {}) {
  const B = branching.reduce((x, y) => x * y, 1), D = branching.length;
  const digits = i => { const d = []; for (let l = D - 1; l >= 0; l--) { d[l] = i % branching[l]; i = Math.floor(i / branching[l]); } return d; };
  const dg = Array.from({ length: B }, (_, i) => digits(i));
  const R = new Float64Array(B * B);
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) {
    if (i === j) { R[i * B + j] = 1; continue; }
    let l = 0; while (l < D && dg[i][l] === dg[j][l]) l++;
    R[i * B + j] = corr[l];
  }
  const labels = Int32Array.from(dg, d => d[0]);
  const S = clampPSD(R, B, true);
  for (let k = 0; k < B * B; k++) S[k] *= variance;
  return { Sigma: S, B, labels };
}

/** Random orthogonal matrix: modified Gram–Schmidt on a Gaussian matrix drawn row-major. */
export function randomOrthogonal(B, rng) {
  rng = asRng(rng);
  const Q = new Float64Array(B * B);
  for (let k = 0; k < B * B; k++) Q[k] = rng.normal();
  for (let k = 0; k < B; k++) {
    for (let j = 0; j < k; j++) {
      let d = 0; for (let i = 0; i < B; i++) d += Q[i * B + j] * Q[i * B + k];
      for (let i = 0; i < B; i++) Q[i * B + k] -= d * Q[i * B + j];
    }
    let nr = 0; for (let i = 0; i < B; i++) nr += Q[i * B + k] * Q[i * B + k];
    nr = Math.sqrt(nr); for (let i = 0; i < B; i++) Q[i * B + k] /= nr;
  }
  return Q;
}

function normalizeMeanDiag(S, B) {
  let t = 0; for (let b = 0; b < B; b++) t += S[b * B + b];
  const f = B / t; for (let k = 0; k < B * B; k++) S[k] *= f;
  return S;
}

/** Covariance with a given spectrum `lambdas` (length B) in basis Q (B×B, column k is
 *  eigenvector k; default identity). */
export function covFromSpectrum(lambdas, B, Q = null) {
  const S = new Float64Array(B * B);
  for (let k = 0; k < B; k++) {
    const lam = lambdas[k];
    if (!Q) { S[k * B + k] = lam; continue; }
    for (let i = 0; i < B; i++) { const q = Q[i * B + k] * lam; for (let j = 0; j < B; j++) S[i * B + j] += q * Q[j * B + k]; }
  }
  for (let i = 0; i < B; i++) for (let j = i + 1; j < B; j++) { const v = 0.5 * (S[i * B + j] + S[j * B + i]); S[i * B + j] = v; S[j * B + i] = v; }
  return S;
}

/** Power-law spectrum lambda_k ∝ k^-alpha (k = 1..B) in a random orthogonal basis drawn
 *  from makeRng(seed) (same matrix as engine.js covPowerLaw).  nonneg: add c·11^T with
 *  the smallest c making all off-diagonals >= 0.  normalize: mean variance 1. */
export function covPowerLaw({ B, alpha = 1, seed = 0, nonneg = false, normalize = true }) {
  const Q = randomOrthogonal(B, makeRng(seed));
  const S = covFromSpectrum(Array.from({ length: B }, (_, k) => Math.pow(k + 1, -alpha)), B, Q);
  if (nonneg) {
    let mn = Infinity; for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) if (i !== j) mn = Math.min(mn, S[i * B + j]);
    const c = Math.max(0, -mn); for (let k = 0; k < B * B; k++) S[k] += c;
  }
  return normalize ? normalizeMeanDiag(S, B) : S;
}

/** Rank one: Sigma = scale · v v^T (v defaults to the all-ones vector).  The scalar-drive
 *  input of Q-diversity. */
export function covRankOne({ B, v = null, scale = 1 }) {
  const S = new Float64Array(B * B);
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) S[i * B + j] = scale * (v ? v[i] * v[j] : 1);
  return S;
}

/** Diagonal covariance with the given variances. */
export function covDiag(variances) {
  const B = variances.length, S = new Float64Array(B * B);
  for (let b = 0; b < B; b++) S[b * B + b] = variances[b];
  return S;
}

// ─────────────────────────── inputs and network ───────────────────────────

/** S zero-mean inputs (S×B) with covariance Sigma: G = Z Sigma^{1/2}, Z drawn row-major
 *  from rng.  kind = 'gauss' (standard normal Z) or 'laplace' (independent unit-variance
 *  Laplace Z: same covariance, heavy tails).  As hebb.onsite.sample_inputs, but with
 *  this module's PRNG. */
export function sampleInputs(Sigma, B, S, rng, kind = 'gauss') {
  rng = asRng(rng);
  const root = sqrtm(Sigma, B), Z = new Float64Array(B), G = new Float64Array(S * B);
  const draw = kind === 'laplace' ? rng.laplace : kind === 'gauss' ? rng.normal : null;
  if (!draw) throw new Error(`sampleInputs: unknown kind ${kind}`);
  for (let s = 0; s < S; s++) {
    for (let b = 0; b < B; b++) Z[b] = draw();
    const o = s * B;
    for (let c = 0; c < B; c++) { const z = Z[c]; if (z === 0) continue; const rc = c * B; for (let b = 0; b < B; b++) G[o + b] += z * root[rc + b]; }
  }
  return G;
}

/** Each neuron attached to s distinct random inputs (partial Fisher–Yates with rng.int),
 *  total conductance deg, shares Dirichlet(1) (normalised rng.exponential draws) or equal.
 *  Draw order per neuron: s ints, then s exponentials (if !equal). */
export function makeAttachments(N, B, s, rng, deg = 1, equal = false) {
  rng = asRng(rng);
  if (s > B || s < 1) throw new Error('makeAttachments: need 1 <= s <= B');
  const A = new Float64Array(N * B), perm = new Int32Array(B), e = new Float64Array(s);
  for (let i = 0; i < N; i++) {
    for (let b = 0; b < B; b++) perm[b] = b;
    for (let q = 0; q < s; q++) { const r = q + rng.int(B - q); const x = perm[q]; perm[q] = perm[r]; perm[r] = x; }
    let tot = 0;
    for (let q = 0; q < s; q++) { e[q] = equal ? 1 : rng.exponential(1); tot += e[q]; }
    for (let q = 0; q < s; q++) A[i * B + perm[q]] = deg * e[q] / tot;
  }
  return A;
}

const vecN = (x, N) => (typeof x === 'number' ? new Float64Array(N).fill(x) : Float64Array.from(x));

/** Network object {N, B, A, t, klo, khi, W}; scalars broadcast; arrays are copied.
 *  W defaults to zeros. */
export function makeNetFrom({ A, N, B, t = 0, klo = 0, khi = 0, W = null }) {
  return { N, B, A: Float64Array.from(A), t: vecN(t, N), klo: vecN(klo, N), khi: vecN(khi, N),
    W: W ? Float64Array.from(W) : new Float64Array(N * N) };
}

/** Random network as hebb.onsite.make_net: attachments, then thresholds t = sigmaT · normal. */
export function makeNet({ N, B, rng = 0, s = 3, deg = 1, klo = 0, khi = 0, sigmaT = 0, equal = false }) {
  rng = asRng(rng);
  const A = makeAttachments(N, B, s, rng, deg, equal);
  const t = new Float64Array(N);
  for (let i = 0; i < N; i++) t[i] = sigmaT * rng.normal();
  return makeNetFrom({ A, N, B, t, klo, khi });
}

export function cloneNet(net) { return makeNetFrom(net); }

/** Node degrees d_i = sum_j w_ij + sum_b a_ib. */
function degrees(net, out = new Float64Array(net.N)) {
  const { N, B, A, W } = net;
  for (let i = 0; i < N; i++) {
    let d = 0;
    for (let j = 0; j < N; j++) if (j !== i) d += W[i * N + j];
    for (let b = 0; b < B; b++) d += A[i * B + b];
    out[i] = d;
  }
  return out;
}

/** L = diag(d) - W (N×N). */
export function laplacian(net) {
  const { N, W } = net, d = degrees(net), L = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) L[i * N + j] = i === j ? d[i] : -W[i * N + j];
  return L;
}

/** Input currents R = G A^T (S×N). */
export function inputCurrents(net, G, S, out = new Float64Array(S * net.N)) {
  const { N, B, A } = net;
  for (let s = 0; s < S; s++) {
    const g = s * B, o = s * N;
    for (let i = 0; i < N; i++) { let r = 0; const a = i * B; for (let b = 0; b < B; b++) r += A[a + b] * G[g + b]; out[o + i] = r; }
  }
  return out;
}

/** Mean energy over the bank (U: S×N, G: S×B). */
export function energy(net, U, G, S) {
  const { N, B, A, W, t, klo, khi } = net;
  let e = 0;
  for (let s = 0; s < S; s++) {
    const o = s * N, g = s * B;
    for (let i = 0; i < N; i++) {
      const ui = U[o + i];
      for (let j = i + 1; j < N; j++) { const w = W[i * N + j]; if (w !== 0) { const d = ui - U[o + j]; e += 0.5 * w * d * d; } }
      for (let b = 0; b < B; b++) { const a = A[i * B + b]; if (a !== 0) { const d = ui - G[g + b]; e += 0.5 * a * d * d; } }
      const x = ui - t[i];
      e += 0.5 * (x < 0 ? klo[i] : khi[i]) * x * x;
    }
  }
  return e / S;
}

/** Max over samples and neurons of |dE/du| = |L u + V'(u) - R| (the node-equation
 *  residual; units of current).  R optional (input currents). */
export function nodeResidual(net, U, G, S, R = null) {
  const { N, W, t, klo, khi } = net;
  R = R || inputCurrents(net, G, S);
  const d = degrees(net);
  let mx = 0;
  for (let s = 0; s < S; s++) {
    const o = s * N;
    for (let i = 0; i < N; i++) {
      let g = d[i] * U[o + i] - R[o + i];
      const ri = i * N;
      for (let j = 0; j < N; j++) { const w = W[ri + j]; if (w !== 0 && j !== i) g -= w * U[o + j]; }
      const x = U[o + i] - t[i];
      g += (x < 0 ? klo[i] : khi[i]) * x;
      const a = Math.abs(g); if (a > mx || a !== a) mx = a;
    }
  }
  return mx;
}

// ─────────────────────────── equilibrium ───────────────────────────
//
// FAST SOLVER (default).  The node equations are piecewise linear: on each "activity
// pattern" (which neurons sit above threshold) they are the linear SPD system
// (L + diag k_P) u = R + k_P t.  Semismooth Newton = solve with the pattern of the
// current iterate, then check whether the solution has the same pattern; if so it is the
// exact equilibrium (to roundoff).  Three exact speedups over the per-sample reference:
//  1. The live graph splits into connected components, which decouple exactly; each is
//     solved alone (isolated neurons in closed form).
//  2. Within a component, samples that share an activity pattern share one Cholesky
//     factor (a fused cluster with spread thresholds typically shows O(n) patterns, not S).
//  3. Warm start from the previous equilibrium (the live loop changes W by dt·dW per
//     step), so almost every sample's pattern is already right and Newton stops after
//     one solve.  Samples whose pattern changed take further Newton steps, with the
//     reference's backtracking safeguard on max|dE/du|.
// A sample is accepted when its pattern is self-consistent, or when max|dE/du| <= tol.
// The achieved residual (max |dE/du| over the whole bank) is always returned.

function componentList(W, N) {
  // Coupling components of W != 0 (the solver must respect every nonzero conductance).
  const lab = new Int32Array(N).fill(-1), comps = [], stack = [];
  for (let i = 0; i < N; i++) {
    if (lab[i] >= 0) continue;
    const c = comps.length, mem = [i]; lab[i] = c; stack.push(i);
    while (stack.length) {
      const j = stack.pop(), r = j * N;
      for (let k = 0; k < N; k++) if (lab[k] < 0 && W[r + k] !== 0 && k !== j) { lab[k] = c; mem.push(k); stack.push(k); }
    }
    mem.sort((x, y) => x - y);
    comps.push(mem);
  }
  return comps;
}

/**
 * Equilibrium responses U (S×N) for every row of G.
 * opts: {U0 (warm start, S×N), R (input currents, S×N), tol = 1e-10, maxIter = 100,
 *        exact = false (use the slow per-sample reference path), out (S×N buffer; may alias U0)}.
 * Returns {U, residual (max |dE/du| over the bank), iters (max Newton iterations over
 *          components), factorizations (fresh Cholesky factorisations), updates (rank-1
 *          factor updates), components (coupling components of W != 0)}.
 */
export function equilibrate(net, G, S, opts = {}) {
  if (opts.exact) return equilibrateExact(net, G, S, opts);
  const { N, B, A, W, t, klo, khi } = net;
  const tol = opts.tol ?? 1e-10, maxIter = opts.maxIter ?? 100;
  const R = opts.R || inputCurrents(net, G, S);
  const U0 = opts.U0 || null;
  const U = opts.out || new Float64Array(S * N);
  const dAll = degrees(net);
  const asum = new Float64Array(N);
  for (let i = 0; i < N; i++) { let a = 0; for (let b = 0; b < B; b++) a += A[i * B + b]; asum[i] = a; }
  const comps = componentList(W, N);
  let resid = 0, itersMax = 0, nFact = 0, nUpd = 0;

  for (const mem of comps) {
    const n = mem.length;
    if (n === 1) {
      // Isolated neuron: d u + V'(u) = r with d = a_i.  u >= t iff r >= d t.
      const i = mem[0], d = dAll[i], ti = t[i], kl = klo[i], kh = khi[i];
      if (!(d + Math.min(kl, kh) > 0)) throw new Error(`equilibrate: neuron ${i} has no attachment and no potential`);
      for (let s = 0; s < S; s++) {
        const r = R[s * N + i];
        const k = r - d * ti >= 0 ? kh : kl;
        const u = (r + k * ti) / (d + k);
        U[s * N + i] = u;
        const x = u - ti;
        const g = Math.abs(d * u + (x < 0 ? kl : kh) * x - r);
        if (g > resid || g !== g) resid = g;
      }
      itersMax = Math.max(itersMax, 1);
      continue;
    }
    const r = solveComponent(mem, n, net, R, S, U0, U, dAll, tol, maxIter);
    nFact += r.nFact; nUpd += r.nUpd; itersMax = Math.max(itersMax, r.iters);
    if (r.resid > resid || r.resid !== r.resid) resid = r.resid;
  }
  return { U, residual: resid, iters: itersMax, factorizations: nFact, updates: nUpd, components: comps.length };
}

function solveComponent(mem, n, net, R, S, U0, U, dAll, tol, maxIter) {
  const { N, W, t, klo, khi } = net;
  // Local Laplacian, thresholds, slopes; currents gathered per sample.
  const Lc = new Float64Array(n * n);
  for (let p = 0; p < n; p++) {
    const ip = mem[p];
    Lc[p * n + p] = dAll[ip];
    for (let q = 0; q < n; q++) if (q !== p) Lc[p * n + q] = -W[ip * N + mem[q]];
  }
  const tc = new Float64Array(n), kl = new Float64Array(n), kh = new Float64Array(n);
  const nl = [];
  for (let p = 0; p < n; p++) { const i = mem[p]; tc[p] = t[i]; kl[p] = klo[i]; kh[p] = khi[i]; if (kl[p] !== kh[p]) nl.push(p); }
  const Rc = new Float64Array(S * n), Uc = new Float64Array(S * n);
  for (let s = 0; s < S; s++) for (let p = 0; p < n; p++) Rc[s * n + p] = R[s * N + mem[p]];
  const J = new Float64Array(n * n), kv = new Float64Array(n);
  let nFact = 0;
  const factor = k => {
    J.set(Lc);
    for (let p = 0; p < n; p++) J[p * n + p] += k[p];
    nFact++;
    if (!cholesky(J, n)) throw new Error('equilibrate: component matrix not SPD (a neuron with no attachment?)');
  };
  const solveInto = (k, s, dst, off) => {           // dst[off..] = J^{-1}(R_s + k t)
    for (let p = 0; p < n; p++) dst[off + p] = Rc[s * n + p] + k[p] * tc[p];
    cholSolveVec(J, n, dst, off);
  };
  // Gradient max-norm of the node equations at u (length n, offset off).
  const gmax = (u, off, s) => {
    let mx = 0;
    for (let p = 0; p < n; p++) {
      let g = -Rc[s * n + p];
      const rp = p * n;
      for (let q = 0; q < n; q++) g += Lc[rp + q] * u[off + q];
      const x = u[off + p] - tc[p];
      g += (x < 0 ? kl[p] : kh[p]) * x;
      const a = Math.abs(g); if (a > mx || a !== a) mx = a;
    }
    return mx;
  };

  if (nl.length === 0) {
    // Linear on this component (klo = khi): a single factor solves every sample exactly.
    factor(kl);
    for (let s = 0; s < S; s++) solveInto(kl, s, Uc, s * n);
  } else {
    if (U0) for (let s = 0; s < S; s++) for (let p = 0; p < n; p++) Uc[s * n + p] = U0[s * N + mem[p]];
    else {
      for (let p = 0; p < n; p++) kv[p] = 0.5 * (kl[p] + kh[p]);
      factor(kv);
      for (let s = 0; s < S; s++) solveInto(kv, s, Uc, s * n);
    }
  }

  let iters = 0, nUpd = 0;
  if (nl.length > 0) {
    let active = Array.from({ length: S }, (_, s) => s);
    const m = nl.length, nw = Math.ceil(m / 32);
    // Activity pattern of the nonlinear members, packed in 32-bit words.
    const pattern = (u, off, out) => {
      out.fill(0);
      for (let q = 0; q < m; q++) { const p = nl[q]; if (u[off + p] - tc[p] >= 0) out[q >> 5] |= 1 << (q & 31); }
      return out;
    };
    const keyOf = w => { let k = ''; for (let i = 0; i < nw; i++) k += String.fromCharCode(w[i] & 0xffff, w[i] >>> 16); return k; };
    const same = (a, b) => { for (let i = 0; i < nw; i++) if (a[i] !== b[i]) return false; return true; };
    const hamming = (a, b) => { let h = 0; for (let i = 0; i < nw; i++) h += popcount(a[i] ^ b[i]); return h; };
    const ufull = new Float64Array(n), cand = new Float64Array(n), wtmp = new Uint32Array(nw);
    const cur = new Uint32Array(nw), xw = new Float64Array(n);
    // Rank-1 updates are used instead of a fresh factor when the next pattern differs in at
    // most maxFlips neurons; a fresh factor is forced after maxChain accumulated updates.
    const maxFlips = Math.max(2, n >> 2), maxChain = 2 * n;
    while (active.length && iters < maxIter) {
      iters++;
      const groups = new Map();
      for (const s of active) {
        pattern(Uc, s * n, wtmp);
        const key = keyOf(wtmp);
        let g = groups.get(key); if (!g) { g = { words: Uint32Array.from(wtmp), list: [] }; groups.set(key, g); } g.list.push(s);
      }
      // Greedy nearest-neighbour chain through the patterns, from the most populous one.
      const todo = [...groups.values()];
      let start = 0; for (let i = 1; i < todo.length; i++) if (todo[i].list.length > todo[start].list.length) start = i;
      const order = [todo[start]]; todo[start] = todo[todo.length - 1]; todo.pop();
      while (todo.length) {
        const last = order[order.length - 1].words;
        let best = 0, bd = Infinity;
        for (let i = 0; i < todo.length; i++) { const h = hamming(last, todo[i].words); if (h < bd) { bd = h; best = i; if (h <= 1) break; } }
        order.push(todo[best]); todo[best] = todo[todo.length - 1]; todo.pop();
      }
      const next = [];
      let have = false, chain = 0;
      for (const { words, list } of order) {
        const d = have ? hamming(cur, words) : Infinity;
        let fresh = !(d <= maxFlips && chain + d <= maxChain);
        if (!fresh) {
          for (let q = 0; q < m && !fresh; q++) {
            if (((cur[q >> 5] ^ words[q >> 5]) >>> (q & 31) & 1) === 0) continue;
            const p = nl[q], knew = (words[q >> 5] >>> (q & 31) & 1) ? kh[p] : kl[p];
            const delta = knew - kv[p]; kv[p] = knew; nUpd++; chain++;
            if (!cholRank1(J, n, p, delta, xw)) fresh = true;
          }
        }
        if (fresh) {
          for (let p = 0; p < n; p++) kv[p] = kl[p];
          for (let q = 0; q < m; q++) { const p = nl[q]; if (words[q >> 5] >>> (q & 31) & 1) kv[p] = kh[p]; }
          factor(kv); chain = 0;
        }
        cur.set(words); have = true;
        for (const s of list) {
          const off = s * n;
          solveInto(kv, s, ufull, 0);
          if (same(pattern(ufull, 0, wtmp), words)) { Uc.set(ufull, off); continue; }   // self-consistent: exact
          const gnew = gmax(ufull, 0, s);
          if (gnew <= tol) { Uc.set(ufull, off); continue; }
          const gold = gmax(Uc, off, s);
          if (gold <= tol) continue;                                        // already converged
          let step = 1, accepted = gnew <= gold * (1 + 1e-12);
          if (!accepted) {
            for (let h = 0; h < 30 && !accepted; h++) {
              step *= 0.5;
              for (let p = 0; p < n; p++) cand[p] = Uc[off + p] + step * (ufull[p] - Uc[off + p]);
              accepted = gmax(cand, 0, s) <= gold * (1 + 1e-12);
            }
            Uc.set(cand, off);
          } else Uc.set(ufull, off);
          next.push(s);
        }
      }
      active = next;
    }
  }
  let resid = 0;
  for (let s = 0; s < S; s++) {
    const g = gmax(Uc, s * n, s); if (g > resid || g !== g) resid = g;
    for (let p = 0; p < n; p++) U[s * N + mem[p]] = Uc[s * n + p];
  }
  return { nFact, nUpd, iters, resid };
}

function popcount(x) {
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return Math.imul((x + (x >>> 4)) & 0x0F0F0F0F, 0x01010101) >>> 24;
}

/** In-place rank-1 update of a lower Cholesky factor: L L^T + delta e_p e_p^T.
 *  x is an n-length workspace.  Returns false if the result would not be SPD. */
export function cholRank1(L, n, p, delta, x) {
  if (delta === 0) return true;
  x.fill(0, p, n);
  x[p] = Math.sqrt(Math.abs(delta));
  const sg = delta > 0 ? 1 : -1;
  for (let k = p; k < n; k++) {
    const xk = x[k]; if (xk === 0) continue;
    const kk = k * n + k, Lkk = L[kk];
    const r2 = Lkk * Lkk + sg * xk * xk;
    if (!(r2 > 0)) return false;
    const r = Math.sqrt(r2), c = r / Lkk, s = xk / Lkk, ss = sg * s, ic = 1 / c;
    L[kk] = r;
    for (let i = k + 1, ik = kk + n; i < n; i++, ik += n) {
      const li = (L[ik] + ss * x[i]) * ic;
      L[ik] = li;
      x[i] = c * x[i] - s * li;
    }
  }
  return true;
}

/**
 * Slow exact path, a line-by-line port of hebb.onsite.equilibrate: per-sample semismooth
 * Newton on the full N×N system (no components, no pattern sharing), the same cold start
 * (L + diag kbar) u = R + kbar t, and the same backtracking on max|dE/du|.  For tests.
 */
export function equilibrateExact(net, G, S, opts = {}) {
  const { N, t, klo, khi } = net;
  const tol = opts.tol ?? 1e-10, maxIter = opts.maxIter ?? 100;
  const R = opts.R || inputCurrents(net, G, S);
  const L = laplacian(net);
  const U = opts.out || new Float64Array(S * N);
  let nonlin = false; for (let i = 0; i < N; i++) if (klo[i] !== 0 || khi[i] !== 0) nonlin = true;
  const J = new Float64Array(N * N), kv = new Float64Array(N);
  let nFact = 0;
  const factor = k => { J.set(L); for (let i = 0; i < N; i++) J[i * N + i] += k[i]; nFact++; if (!cholesky(J, N)) throw new Error('equilibrateExact: not SPD'); };
  if (!nonlin) {
    factor(kv);
    for (let s = 0; s < S; s++) { for (let i = 0; i < N; i++) U[s * N + i] = R[s * N + i]; cholSolveVec(J, N, U, s * N); }
    return { U, residual: nodeResidual(net, U, G, S, R), iters: 1, factorizations: nFact, components: 1 };
  }
  if (opts.U0) { if (opts.U0 !== U) U.set(opts.U0.subarray(0, S * N)); }
  else {
    for (let i = 0; i < N; i++) kv[i] = 0.5 * (klo[i] + khi[i]);
    factor(kv);
    for (let s = 0; s < S; s++) { for (let i = 0; i < N; i++) U[s * N + i] = R[s * N + i] + kv[i] * t[i]; cholSolveVec(J, N, U, s * N); }
  }
  const grad = (u, off, s, g) => {           // g = L u + k x - r ; returns max|g|
    let mx = 0;
    for (let i = 0; i < N; i++) {
      let v = -R[s * N + i];
      for (let j = 0; j < N; j++) v += L[i * N + j] * u[off + j];
      const x = u[off + i] - t[i];
      v += (x < 0 ? klo[i] : khi[i]) * x;
      if (g) g[i] = v;
      const a = Math.abs(v); if (a > mx || a !== a) mx = a;
    }
    return mx;
  };
  const g = new Float64Array(N), dU = new Float64Array(N), cand = new Float64Array(N);
  let iters = 0;
  for (let s = 0; s < S; s++) {
    const off = s * N;
    let gn = grad(U, off, s, g);
    let it = 0;
    for (; it < maxIter && gn > tol; it++) {
      for (let i = 0; i < N; i++) kv[i] = U[off + i] - t[i] < 0 ? klo[i] : khi[i];
      factor(kv);
      dU.set(g); cholSolveVec(J, N, dU);
      let step = 1, gnew = 0;
      for (let h = 0; h < 30; h++) {
        for (let i = 0; i < N; i++) cand[i] = U[off + i] - step * dU[i];
        gnew = grad(cand, 0, s, null);
        if (!(gnew > gn * (1 + 1e-12))) break;
        step *= 0.5;
      }
      for (let i = 0; i < N; i++) U[off + i] = cand[i];
      gn = grad(U, off, s, g);
    }
    iters = Math.max(iters, it);
  }
  return { U, residual: nodeResidual(net, U, G, S, R), iters, factorizations: nFact, components: 1 };
}

/** Linear zero-potential response matrix H (N×B), u = H g; rows sum to 1. */
export function harmonic(net) {
  return solveSPD(laplacian(net), net.N, net.A, net.B);
}

// ─────────────────────────── geometry and the rule ───────────────────────────

/** D2[i,j] = mean over the bank of (U_i - U_j)^2 (N×N, exact zero diagonal).
 *  Computed from differences (no cancellation).  With V (S×M): the N×M cross matrix. */
export function dist2(U, S, N, V = null, M = N, out = null) {
  const UT = new Float64Array(N * S);
  for (let s = 0; s < S; s++) for (let i = 0; i < N; i++) UT[i * S + s] = U[s * N + i];
  if (V) {
    const VT = new Float64Array(M * S);
    for (let s = 0; s < S; s++) for (let j = 0; j < M; j++) VT[j * S + s] = V[s * M + j];
    out = out || new Float64Array(N * M);
    for (let i = 0; i < N; i++) for (let j = 0; j < M; j++) {
      let a = 0; const oi = i * S, oj = j * S;
      for (let s = 0; s < S; s++) { const d = UT[oi + s] - VT[oj + s]; a += d * d; }
      out[i * M + j] = a / S;
    }
    return out;
  }
  out = out || new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    out[i * N + i] = 0;
    const oi = i * S;
    for (let j = i + 1; j < N; j++) {
      let a = 0; const oj = j * S;
      for (let s = 0; s < S; s++) { const d = UT[oi + s] - UT[oj + s]; a += d * d; }
      out[i * N + j] = out[j * N + i] = a / S;
    }
  }
  return out;
}

/** dW = b - alpha W - D2/2 with zero diagonal, from a given D2. */
export function driftFromD2(W, D2, N, b, alpha, out = new Float64Array(N * N)) {
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) out[i * N + j] = i === j ? 0 : b - alpha * W[i * N + j] - 0.5 * D2[i * N + j];
  return out;
}

/** Projected residual max |dW| where edges at w <= 0 with negative drift count 0. */
export function projectedResidual(W, dW, N) {
  let r = 0;
  for (let k = 0; k < N * N; k++) { const v = W[k] <= 0 && dW[k] < 0 ? 0 : Math.abs(dW[k]); if (v > r || v !== v) r = v; }
  return r;
}

/** {dW, U, D2}: drift of the rule at the equilibrium U (computed if not given). */
export function drift(net, G, S, b, alpha, U = null, solverOpts = {}) {
  U = U || equilibrate(net, G, S, solverOpts).U;
  const D2 = dist2(U, S, net.N);
  return { dW: driftFromD2(net.W, D2, net.N, b, alpha), U, D2 };
}

/** Projected stationarity residual max |dw/dt| (as hebb.onsite.residual). */
export function residual(net, G, S, b, alpha, U = null, solverOpts = {}) {
  const { dW } = drift(net, G, S, b, alpha, U, solverOpts);
  return projectedResidual(net.W, dW, net.N);
}

/** One projected Euler step of W, in place.  Returns the U used. */
export function step(net, G, S, b, alpha, dt, U = null, solverOpts = {}) {
  const { dW, U: Uu } = drift(net, G, S, b, alpha, U, solverOpts);
  const W = net.W;
  for (let k = 0; k < W.length; k++) W[k] = Math.max(W[k] + dt * dW[k], 0);
  return Uu;
}

/** Integrate to stationarity, as hebb.onsite.run (warm-started equilibria, stop when
 *  max|ΔW|/dt < tol).  Returns {steps, residual, U, eqResidual}. */
export function run(net, G, S, b, alpha, { dt = 0.2, T = 5000, tol = 1e-8, callback = null, solver = {} } = {}) {
  const R = inputCurrents(net, G, S), N = net.N;
  let U = null, n = 0, eqRes = 0;
  const Wold = new Float64Array(N * N);
  for (n = 0; n < T; n++) {
    const e = equilibrate(net, G, S, { ...solver, R, U0: U });
    U = e.U; eqRes = Math.max(eqRes, e.residual);
    Wold.set(net.W);
    step(net, G, S, b, alpha, dt, U);
    if (callback) callback(n, net, U);
    let mx = 0; for (let k = 0; k < N * N; k++) mx = Math.max(mx, Math.abs(net.W[k] - Wold[k]));
    if (mx / dt < tol) break;
  }
  const e = equilibrate(net, G, S, { ...solver, R, U0: U });
  return { steps: Math.min(n + 1, T), residual: residual(net, G, S, b, alpha, e.U), U: e.U, eqResidual: Math.max(eqRes, e.residual) };
}

// ─────────────────────────── live simulation ───────────────────────────

/**
 * Stateful live loop.  Invariant after every call: sim.U is the equilibrium at the
 * current sim.net.W, and sim.D2, sim.dW, sim.residual are consistent with it.
 *   sim.step(n)  n projected Euler steps (one equilibrium solve per step, warm-started).
 *   sim.stepFor(ms, maxSteps, stopTol)  as many steps as fit in a frame budget.
 *   sim.refresh() re-solve after you edit net (W, A, t, klo, khi) or G by hand.
 *   sim.setParams({b, alpha, dt})  cheap (no re-solve).
 *   sim.setInputs(G, S)  new bank (re-solve, warm-started if S unchanged).
 * Fields: net, G, S, b, alpha, dt, U, D2, dW, residual (projected max|dw/dt|),
 *   eqResidual (max |dE/du| of the last solve), lastChange (max|ΔW|/dt of the last step),
 *   steps, lastSolve {iters, factorizations, updates, components}.
 */
export function createSim({ net, G, S, b = 0.1, alpha = 1, dt = 0.2, solver = {} }) {
  const N = net.N;
  const sim = {
    net, G, S, b, alpha, dt, solver,
    U: null, D2: new Float64Array(N * N), dW: new Float64Array(N * N),
    residual: NaN, eqResidual: NaN, lastChange: NaN, steps: 0, lastSolve: null,
    _R: null,
  };
  sim.refresh = (cold = false) => {
    sim._R = inputCurrents(sim.net, sim.G, sim.S, sim._R && sim._R.length === sim.S * N ? sim._R : undefined);
    const U0 = !cold && sim.U && sim.U.length === sim.S * N ? sim.U : null;
    const e = equilibrate(sim.net, sim.G, sim.S, { ...sim.solver, R: sim._R, U0, out: U0 || new Float64Array(sim.S * N) });
    sim.U = e.U; sim.eqResidual = e.residual;
    sim.lastSolve = { iters: e.iters, factorizations: e.factorizations, updates: e.updates || 0, components: e.components };
    dist2(sim.U, sim.S, N, null, N, sim.D2);
    sim._drift();
    return sim;
  };
  sim._drift = () => {
    driftFromD2(sim.net.W, sim.D2, N, sim.b, sim.alpha, sim.dW);
    sim.residual = projectedResidual(sim.net.W, sim.dW, N);
  };
  sim.step = (n = 1) => {
    const W = sim.net.W;
    for (let k = 0; k < n; k++) {
      let mx = 0;
      for (let q = 0; q < N * N; q++) {
        const w = Math.max(W[q] + sim.dt * sim.dW[q], 0);
        const c = Math.abs(w - W[q]); if (c > mx) mx = c;
        W[q] = w;
      }
      sim.lastChange = mx / sim.dt;
      sim.steps++;
      const e = equilibrate(sim.net, sim.G, sim.S, { ...sim.solver, R: sim._R, U0: sim.U, out: sim.U });
      sim.eqResidual = e.residual;
      sim.lastSolve = { iters: e.iters, factorizations: e.factorizations, updates: e.updates || 0, components: e.components };
      dist2(sim.U, sim.S, N, null, N, sim.D2);
      sim._drift();
    }
    return sim;
  };
  /** Step until `budgetMs` of wall time is used (at least one step, at most maxSteps), for
   *  requestAnimationFrame loops.  Stops early once lastChange < stopTol.  Returns steps. */
  sim.stepFor = (budgetMs = 12, maxSteps = Infinity, stopTol = 0) => {
    const t0 = performance.now();
    let k = 0;
    do { sim.step(1); k++; } while (k < maxSteps && sim.lastChange >= stopTol && performance.now() - t0 < budgetMs);
    return k;
  };
  sim.setParams = ({ b, alpha, dt } = {}) => {
    if (b !== undefined) sim.b = b;
    if (alpha !== undefined) sim.alpha = alpha;
    if (dt !== undefined) sim.dt = dt;
    sim._drift();
    return sim;
  };
  sim.setInputs = (G, S) => { const cold = S !== sim.S; sim.G = G; sim.S = S; sim._R = null; return sim.refresh(cold); };
  sim.energy = () => energy(sim.net, sim.U, sim.G, sim.S);
  sim.components = (eps = 1e-9) => components(sim.net.W, N, eps);
  return sim.refresh(true);
}

// ─────────────────────────── analysis ───────────────────────────

/** Connected components of the live graph (w > eps).  Labels ordered by first member
 *  (as hebb.onsite.components).  Returns {labels: Int32Array(N), K}. */
export function components(W, N, eps = 1e-9) {
  const lab = new Int32Array(N).fill(-1);
  let c = 0;
  const stack = [];
  for (let i = 0; i < N; i++) {
    if (lab[i] >= 0) continue;
    lab[i] = c; stack.push(i);
    while (stack.length) {
      const j = stack.pop();
      for (let k = 0; k < N; k++) if (W[j * N + k] > eps && lab[k] < 0) { lab[k] = c; stack.push(k); }
    }
    c++;
  }
  return { labels: lab, K: c };
}

/** Per component k = 0..K-1: {members (Int32Array), size, proto (S, mean member
 *  response), diamMax, diamMean (D of members to the prototype), dEff ((Tr C)^2/Tr C^2
 *  of the bank-centred member responses)}.  As hebb.onsite.cluster_stats. */
export function clusterStats(U, S, N, labels, K = null) {
  if (K === null) { K = 0; for (let i = 0; i < N; i++) K = Math.max(K, labels[i] + 1); }
  const mean = new Float64Array(N);
  for (let s = 0; s < S; s++) for (let i = 0; i < N; i++) mean[i] += U[s * N + i];
  for (let i = 0; i < N; i++) mean[i] /= S;
  const out = [];
  for (let k = 0; k < K; k++) {
    const idx = []; for (let i = 0; i < N; i++) if (labels[i] === k) idx.push(i);
    const m = idx.length, proto = new Float64Array(S);
    for (let s = 0; s < S; s++) { let a = 0; for (const i of idx) a += U[s * N + i]; proto[s] = a / m; }
    let dmax = 0, dsum = 0;
    for (const i of idx) {
      let a = 0; for (let s = 0; s < S; s++) { const d = U[s * N + i] - proto[s]; a += d * d; }
      const D = Math.sqrt(a / S); dmax = Math.max(dmax, D); dsum += D;
    }
    const C = new Float64Array(m * m);
    for (let s = 0; s < S; s++) for (let p = 0; p < m; p++) {
      const xp = U[s * N + idx[p]] - mean[idx[p]];
      for (let q = p; q < m; q++) C[p * m + q] += xp * (U[s * N + idx[q]] - mean[idx[q]]);
    }
    let tr = 0, tr2 = 0;
    for (let p = 0; p < m; p++) for (let q = p; q < m; q++) {
      const c = C[p * m + q] / S;
      if (p === q) { tr += c; tr2 += c * c; } else tr2 += 2 * c * c;
    }
    out.push({ members: Int32Array.from(idx), size: m, proto, diamMax: m ? dmax : 0, diamMean: m ? dsum / m : 0, dEff: tr2 > 0 ? tr * tr / tr2 : 0 });
  }
  return out;
}

/** Population response operator on the bank (hebb.onsite.population_modes):
 *  C = Uc^T Uc / S (N×N, Uc bank-centred); q = eigenvalues (descending, clipped at 0);
 *  V (N×N, columns = eigenvectors); Psi (S×r) unit-norm eigenfunctions
 *  psi_a = q_a^{-1/2} Uc V_a for the r modes with q_a > 1e-12 q_0; dEff = (sum q)^2/sum q^2. */
export function populationModes(U, S, N) {
  const mean = new Float64Array(N);
  for (let s = 0; s < S; s++) for (let i = 0; i < N; i++) mean[i] += U[s * N + i];
  for (let i = 0; i < N; i++) mean[i] /= S;
  const Uc = new Float64Array(S * N);
  for (let s = 0; s < S; s++) for (let i = 0; i < N; i++) Uc[s * N + i] = U[s * N + i] - mean[i];
  const C = new Float64Array(N * N);
  for (let s = 0; s < S; s++) {
    const o = s * N;
    for (let i = 0; i < N; i++) { const x = Uc[o + i]; if (x === 0) continue; const ri = i * N; for (let j = i; j < N; j++) C[ri + j] += x * Uc[o + j]; }
  }
  for (let i = 0; i < N; i++) for (let j = i; j < N; j++) { const v = C[i * N + j] / S; C[i * N + j] = v; C[j * N + i] = v; }
  const { values, vectors } = eigh(C, N);
  const q = values.map(v => Math.max(v, 0));
  const thr = 1e-12 * Math.max(q[0] || 0, 1e-300);
  let r = 0; while (r < N && q[r] > thr) r++;
  const Psi = new Float64Array(S * r);
  for (let s = 0; s < S; s++) for (let a = 0; a < r; a++) {
    let v = 0; for (let i = 0; i < N; i++) v += Uc[s * N + i] * vectors[i * N + a];
    Psi[s * r + a] = v / Math.sqrt(q[a]);
  }
  let sq = 0, sq2 = 0; for (const v of q) { sq += v; sq2 += v * v; }
  return { q, V: vectors, Psi, r, dEff: sq2 > 0 ? sq * sq / sq2 : 0, C };
}

/** Classical MDS of a squared-distance matrix D2 (n×n): Gram K = -1/2 J D2 J, top `dim`
 *  eigenpairs, X[i*dim + k] = v_k[i] sqrt(max(λ_k, 0)).  If `prev` (n×dim) is given, the
 *  embedding is rotated/reflected onto it (orthogonal Procrustes) so animations don't
 *  flip.  Returns {X, values (top dim eigenvalues of K), explained (positive mass captured),
 *  negMass (|negative eigenvalue mass| / total |mass|: 0 for exactly Euclidean D2)}. */
export function mds(D2, n, dim = 2, prev = null) {
  const K = new Float64Array(n * n), rm = new Float64Array(n);
  let tm = 0;
  for (let i = 0; i < n; i++) { let a = 0; for (let j = 0; j < n; j++) a += D2[i * n + j]; rm[i] = a / n; tm += a; }
  tm /= n * n;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * n + j] = -0.5 * (D2[i * n + j] - rm[i] - rm[j] + tm);
  const { values, vectors } = eigh(K, n);
  const X = new Float64Array(n * dim);
  for (let k = 0; k < Math.min(dim, n); k++) {
    const sc = Math.sqrt(Math.max(values[k], 0));
    for (let i = 0; i < n; i++) X[i * dim + k] = vectors[i * n + k] * sc;
  }
  let pos = 0, neg = 0, top = 0;
  for (let k = 0; k < n; k++) { if (values[k] > 0) { pos += values[k]; if (k < dim) top += values[k]; } else neg -= values[k]; }
  if (prev) procrustes(X, prev, n, dim);
  return { X, values: values.slice(0, dim), explained: pos > 0 ? top / pos : 1, negMass: pos + neg > 0 ? neg / (pos + neg) : 0 };
}

/** Rotate/reflect X (n×dim) in place by the orthogonal R minimising ||X R - P||_F.
 *  Works for rank-deficient X^T P (e.g. one-dimensional geometry): with X^T P = U Σ V^T,
 *  R = U V^T, where singular vectors with σ ≈ 0 are completed to an orthonormal basis
 *  (any completion is optimal; it only moves directions in which X or P has no spread). */
export function procrustes(X, P, n, dim) {
  const M = new Float64Array(dim * dim);                      // M = X^T P
  for (let i = 0; i < n; i++) for (let a = 0; a < dim; a++) for (let c = 0; c < dim; c++) M[a * dim + c] += X[i * dim + a] * P[i * dim + c];
  const MtM = new Float64Array(dim * dim);
  for (let a = 0; a < dim; a++) for (let c = 0; c < dim; c++) { let v = 0; for (let k = 0; k < dim; k++) v += M[k * dim + a] * M[k * dim + c]; MtM[a * dim + c] = v; }
  const { values, vectors: V } = eigh(MtM, dim);              // V: right singular vectors
  if (!(values[0] > 0)) return X;
  const U = new Float64Array(dim * dim);                      // columns: left singular vectors
  let r = 0;
  for (let k = 0; k < dim; k++) {
    const sk = Math.sqrt(Math.max(values[k], 0));
    if (!(sk > 1e-9 * Math.sqrt(values[0]))) break;
    for (let a = 0; a < dim; a++) { let v = 0; for (let c = 0; c < dim; c++) v += M[a * dim + c] * V[c * dim + k]; U[a * dim + k] = v / sk; }
    r++;
  }
  // Complete U (columns r..dim-1) by Gram–Schmidt on the standard basis.
  for (let e = 0, k = r; k < dim && e < dim; e++) {
    for (let a = 0; a < dim; a++) U[a * dim + k] = a === e ? 1 : 0;
    for (let j = 0; j < k; j++) {
      let d = 0; for (let a = 0; a < dim; a++) d += U[a * dim + j] * U[a * dim + k];
      for (let a = 0; a < dim; a++) U[a * dim + k] -= d * U[a * dim + j];
    }
    let nr = 0; for (let a = 0; a < dim; a++) nr += U[a * dim + k] ** 2;
    if (nr < 1e-12) continue;
    nr = Math.sqrt(nr); for (let a = 0; a < dim; a++) U[a * dim + k] /= nr;
    k++;
  }
  const R = new Float64Array(dim * dim);                      // R = U V^T
  for (let a = 0; a < dim; a++) for (let c = 0; c < dim; c++) { let v = 0; for (let k = 0; k < dim; k++) v += U[a * dim + k] * V[c * dim + k]; R[a * dim + c] = v; }
  const row = new Float64Array(dim);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < dim; c++) { let v = 0; for (let a = 0; a < dim; a++) v += X[i * dim + a] * R[a * dim + c]; row[c] = v; }
    for (let c = 0; c < dim; c++) X[i * dim + c] = row[c];
  }
  return X;
}

/** Analytic two-neuron calibration (SPEC Q-fusion): with unit attachments to two
 *  uncorrelated unit-variance inputs, D2(w) = D0^2/(1+2w)^2 (linear case). */
export function twoNeuronD2(w, D02 = 2) { return D02 / ((1 + 2 * w) * (1 + 2 * w)); }
