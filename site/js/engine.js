// engine.js — HebbianHarmonics simulation engine (SPEC.md §1, §3, §7).
// Pure ES module: no DOM, no dependencies; runs in node and browsers.
// Mirrored by hebb/core.py and cross-checked by tests/test_engine_parity.py.
//
// Conventions: every matrix is a Float64Array in row-major order with its shape
// passed explicitly (H is N×B: H[i*B + b]).  Interior–interior weights Wii (N×N,
// symmetric, zero diagonal), interior–boundary weights Wib (N×B), masks Mii/Mib in
// {0,1} mark potential synapses (non-potential synapses are exactly 0, not wmin).
// Params: {theta, lam, eta, gamma, wmin, dt}; w0 = -eta*theta is shared by both
// edge types, eta_IB = eta, eta_II = gamma*eta.  See ENGINE.md for the API reference.

export const VERSION = '1.0.0';

// ───────────────────────────── PRNG ─────────────────────────────
// mulberry32: 32-bit state, period 2^32.  Mirrored bit-exactly by core.JsRng.
export function makeRng(seed = 0) {
  let a = seed | 0;
  function uniform() {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  return {
    uniform,
    normal() { const u1 = uniform(), u2 = uniform(); return Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2); },
    exponential(scale = 1) { return -scale * Math.log(1 - uniform()); },
    int(n) { return Math.floor(uniform() * n); },
  };
}

// ─────────────────────────── linear algebra ───────────────────────────

/** In-place-able Cholesky A = L L^T (lower triangle of `out`; upper triangle untouched).
 *  Returns out, or null if A is not numerically SPD. */
export function cholesky(A, n, out = A) {
  if (out !== A) out.set(A.subarray(0, n * n));
  for (let i = 0; i < n; i++) {
    const ri = i * n;
    for (let j = 0; j <= i; j++) {
      const rj = j * n;
      let s = out[ri + j];
      for (let k = 0; k < j; k++) s -= out[ri + k] * out[rj + k];
      if (i === j) { if (!(s > 0)) return null; out[ri + i] = Math.sqrt(s); }
      else out[ri + j] = s / out[rj + j];
    }
  }
  return out;
}

/** Solve (L L^T) X = R for X (n×m) given the Cholesky factor L (lower). X may alias R. */
export function cholSolve(L, n, R, m, X = new Float64Array(n * m)) {
  if (X !== R) X.set(R.subarray(0, n * m));
  for (let i = 0; i < n; i++) {           // forward: L Y = R
    const ri = i * n, xi = i * m;
    for (let k = 0; k < i; k++) {
      const l = L[ri + k]; if (l === 0) continue;
      const xk = k * m;
      for (let c = 0; c < m; c++) X[xi + c] -= l * X[xk + c];
    }
    const d = 1 / L[ri + i];
    for (let c = 0; c < m; c++) X[xi + c] *= d;
  }
  for (let i = n - 1; i >= 0; i--) {      // backward: L^T X = Y
    const xi = i * m;
    for (let k = i + 1; k < n; k++) {
      const l = L[k * n + i]; if (l === 0) continue;
      const xk = k * m;
      for (let c = 0; c < m; c++) X[xi + c] -= l * X[xk + c];
    }
    const d = 1 / L[i * n + i];
    for (let c = 0; c < m; c++) X[xi + c] *= d;
  }
  return X;
}

/** C (m×n) = A (m×k) · B (k×n). */
export function matmul(A, Bm, m, k, n, out = new Float64Array(m * n)) {
  out.fill(0, 0, m * n);
  for (let i = 0; i < m; i++) {
    const ai = i * k, oi = i * n;
    for (let p = 0; p < k; p++) {
      const a = A[ai + p]; if (a === 0) continue;
      const bp = p * n;
      for (let j = 0; j < n; j++) out[oi + j] += a * Bm[bp + j];
    }
  }
  return out;
}

/** C (m×n) = A^T B with A (k×m), B (k×n). */
export function matmulAtB(A, Bm, k, m, n, out = new Float64Array(m * n)) {
  out.fill(0, 0, m * n);
  for (let p = 0; p < k; p++) {
    const ap = p * m, bp = p * n;
    for (let i = 0; i < m; i++) {
      const a = A[ap + i]; if (a === 0) continue;
      const oi = i * n;
      for (let j = 0; j < n; j++) out[oi + j] += a * Bm[bp + j];
    }
  }
  return out;
}

/** C (m×n) = A B^T with A (m×k), B (n×k). */
export function matmulABt(A, Bm, m, k, n, out = new Float64Array(m * n)) {
  for (let i = 0; i < m; i++) {
    const ai = i * k;
    for (let j = 0; j < n; j++) {
      const bj = j * k; let s = 0;
      for (let p = 0; p < k; p++) s += A[ai + p] * Bm[bj + p];
      out[i * n + j] = s;
    }
  }
  return out;
}

export function transpose(A, m, n, out = new Float64Array(m * n)) {
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[j * m + i] = A[i * n + j];
  return out;
}

/** Symmetric eigendecomposition (Householder tridiagonalisation + implicit QL;
 *  EISPACK tred2/tql2 as in JAMA).  Returns {values (descending), vectors} with
 *  vectors[i*n + k] = component i of eigenvector k (numpy's column convention). */
export function eigh(A, n) {
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) V[i * n + j] = 0.5 * (A[i * n + j] + A[j * n + i]);
  const d = new Float64Array(n), e = new Float64Array(n);
  tred2(V, d, e, n);
  tql2(V, d, e, n);
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

/** V f(Λ) V^T for symmetric A. */
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
export const sqrtm = (A, n) => symFn(A, n, x => Math.sqrt(Math.max(x, 0)));
/** Moore–Penrose pseudo-inverse of a symmetric matrix (eigenvalues below rcond·max|λ| dropped). */
export function pinvSym(A, n, rcond = 1e-12) {
  const { values } = eigh(A, n);
  let mx = 0; for (const v of values) mx = Math.max(mx, Math.abs(v));
  return symFn(A, n, x => (Math.abs(x) > rcond * mx ? 1 / x : 0));
}
/** Nearest PSD matrix in Frobenius norm (clip negative eigenvalues). Also returns the clipped mass. */
export function psdProject(A, n) {
  const { values } = eigh(A, n);
  let neg = 0, tot = 0;
  for (const v of values) { tot += Math.abs(v); if (v < 0) neg -= v; }
  const M = symFn(A, n, x => Math.max(x, 0));
  M.negMass = neg; M.negFrac = tot > 0 ? neg / tot : 0;
  return M;
}

// ─────────────────────────────── model ───────────────────────────────

export function defaultParams() {
  return { theta: 0.3, lam: 1.0, eta: 1.0, gamma: 1.0, wmin: 1e-3, dt: 0.05 };
}

function withWork(s) {
  const { N, B } = s;
  s.L = new Float64Array(N * N);
  s.H = new Float64Array(N * B);
  s.HS = new Float64Array(N * B);
  s.Shat = new Float64Array(B * B);
  s.G = null;
  s.t = 0;
  s.resid = NaN;   // projected stationarity residual at the weights the last step started from
  return s;
}

/** Load explicit arrays (copied). Use this to start JS from a Python-built state. */
export function loadState({ N, B, Wii, Wib, Mii, Mib }) {
  const f = (x, n) => { const a = new Float64Array(n); a.set(x); return a; };
  return withWork({ N, B, Wii: f(Wii, N * N), Wib: f(Wib, N * B), Mii: f(Mii, N * N), Mib: f(Mib, N * B) });
}

/** Random start, same recipe as core.init_state but with the seeded mulberry32 PRNG
 *  (bit-for-bit mirrored by core.create_state).  Draw order: Mii (i<j), Mib, the
 *  rescue draws for neurons with no boundary synapse, Wii (i<j), Wib. */
export function createState({ N, B, seed = 0, p = 1, iiScale = 0, ibScale = 0.2 }) {
  const rng = makeRng(seed);
  const Wii = new Float64Array(N * N), Wib = new Float64Array(N * B);
  const Mii = new Float64Array(N * N), Mib = new Float64Array(N * B);
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (rng.uniform() < p) { Mii[i * N + j] = 1; Mii[j * N + i] = 1; }
  for (let k = 0; k < N * B; k++) Mib[k] = rng.uniform() < p ? 1 : 0;
  for (let i = 0; i < N; i++) {
    let s = 0; for (let b = 0; b < B; b++) s += Mib[i * B + b];
    if (s === 0) Mib[i * B + rng.int(B)] = 1;
  }
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    const w = rng.uniform() * iiScale * Mii[i * N + j];
    Wii[i * N + j] = w; Wii[j * N + i] = w;
  }
  for (let k = 0; k < N * B; k++) Wib[k] = rng.exponential(ibScale) * Mib[k];
  return withWork({ N, B, Wii, Wib, Mii, Mib });
}

export function cloneState(s) {
  const c = loadState(s); c.t = s.t; c.resid = s.resid; c.H.set(s.H); return c;
}

/** H = L_II^{-1} W_IB for arbitrary weights (no clipping).  Throws if L_II is not SPD. */
export function harmonicFrom(Wii, Wib, N, B, L = new Float64Array(N * N), H = new Float64Array(N * B)) {
  for (let i = 0; i < N; i++) {
    const ri = i * N; let deg = 0;
    for (let j = 0; j < N; j++) deg += Wii[ri + j];
    for (let b = 0; b < B; b++) deg += Wib[i * B + b];
    for (let j = 0; j < i; j++) L[ri + j] = -Wii[ri + j];
    L[ri + i] = deg;
  }
  if (!cholesky(L, N)) throw new Error('harmonic: L_II not positive definite (a neuron component has no conductance to the boundary; is wmin > 0?)');
  return cholSolve(L, N, Wib, B, H);
}

/** H for the state's current weights (written into state.H and returned). */
export function harmonic(s) { return harmonicFrom(s.Wii, s.Wib, s.N, s.B, s.L, s.H); }

// ── rectified-rate mode (params.nonlin = 'relu') ──
// Kirchhoff stays linear (u = H g), plasticity sees rates r = relu(u - E[u]).  For zero-mean
// Gaussian (x, y) with variances vi, vj and covariance c (rho = c / sqrt(vi vj), phi = acos rho):
//   Cov(relu x, relu y) = sqrt(vi vj)/(2π) · (sin φ + (π − φ) cos φ − 1)      (arc-cosine kernel)
// ranging from −sqrt(vi vj)/(2π) (rho = −1) through 0 (rho = 0) to vi(π−1)/(2π) (rho = 1, vi = vj).

/** Normalised dual activation k(rho) = (sqrt(1−rho²) + (π − acos rho)·rho − 1)/(2π); increasing on [−1, 1]. */
export function reluCorr(rho) {
  if (rho > 1) rho = 1; else if (rho < -1) rho = -1;
  return (Math.sqrt(Math.max(0, 1 - rho * rho)) + (Math.PI - Math.acos(rho)) * rho - 1) / (2 * Math.PI);
}
/** Inverse of reluCorr (Newton with bisection safeguard; dk/drho = (π − acos rho)/(2π)).
 *  Targets outside [k(−1), k(1)] are clamped to rho = ∓1. */
export function reluCorrInv(t) {
  const lo0 = -1 / (2 * Math.PI), hi0 = (Math.PI - 1) / (2 * Math.PI);
  if (!(t > lo0)) return -1;
  if (t >= hi0) return 1;
  let lo = -1, hi = 1, x = Math.max(-1, Math.min(1, 4 * t));
  for (let it = 0; it < 60; it++) {
    const f = reluCorr(x) - t;
    if (f > 0) hi = x; else lo = x;
    const d = (Math.PI - Math.acos(x)) / (2 * Math.PI);
    let xn = d > 1e-12 ? x - f / d : 0.5 * (lo + hi);
    if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
    if (Math.abs(xn - x) < 1e-15) { x = xn; break; }
    x = xn;
  }
  return x;
}
/** Cov(relu(x), relu(y)) for zero-mean jointly Gaussian x, y with Var x = vi, Var y = vj, Cov = c. */
export function dualRelu(vi, vj, c) {
  const s = Math.sqrt(vi * vj);
  if (!(s > 0)) return 0;
  return s * reluCorr(c / s);
}
/** Inverse of dualRelu in its covariance argument: the c with dualRelu(vi, vj, c) = k (clamped to |c| <= sqrt(vi vj)). */
export function dualReluInv(vi, vj, k) {
  const s = Math.sqrt(vi * vj);
  if (!(s > 0)) return 0;
  return s * reluCorrInv(k / s);
}

// Shared kernel: drift f = w0 - lam*w + eta_ij*Cov at the (floor-clipped) weights.
// Cov is the voltage covariance (nonlin 'linear'), its arc-cosine transform ('relu'), or,
// if `rates` is given, the minibatch covariance of sampled rates.
// apply=true: Euler update exactly as core.step.  Returns projected residual max|f|
// (edges at the floor with f<0 count 0).
function kernel(s, Sigma, P, apply, rates = null) {
  const { N, B, Wii, Wib, Mii, Mib, L, H, HS } = s;
  const wmin = P.wmin, dt = P.dt, lam = P.lam, eta = P.eta, ge = P.gamma * P.eta, w0 = -P.eta * P.theta;
  const relu = (P.nonlin || 'linear') === 'relu';
  if (apply) {
    for (let k = 0; k < N * N; k++) { const f = wmin * Mii[k]; if (Wii[k] < f) Wii[k] = f; }
    for (let k = 0; k < N * B; k++) { const f = wmin * Mib[k]; if (Wib[k] < f) Wib[k] = f; }
    harmonicFrom(Wii, Wib, N, B, L, H);
  } else {
    // stationarity of the current weights without modifying them (floor-clipped copy semantics)
    const Wc = s._Wc || (s._Wc = new Float64Array(N * N)), Wbc = s._Wbc || (s._Wbc = new Float64Array(N * B));
    for (let k = 0; k < N * N; k++) { const f = wmin * Mii[k]; Wc[k] = Wii[k] < f ? f : Wii[k]; }
    for (let k = 0; k < N * B; k++) { const f = wmin * Mib[k]; Wbc[k] = Wib[k] < f ? f : Wib[k]; }
    harmonicFrom(Wc, Wbc, N, B, L, H);
  }
  const Wi = apply ? Wii : s._Wc, Wb = apply ? Wib : s._Wbc;
  let v = null, Rt = null, Rbt = null, nb = 0, den = 1;
  if (rates) {
    ({ Rt, Rbt, nb } = sampleRates(s, rates.G, rates.batch, rates.center));
    den = nb - 1;
  } else {
    HS.fill(0);
    for (let i = 0; i < N; i++) {
      const hi = i * B;
      for (let b = 0; b < B; b++) {
        const h = H[hi + b], sb = b * B;
        for (let c = 0; c < B; c++) HS[hi + c] += h * Sigma[sb + c];
      }
    }
    if (relu) {
      v = s._v || (s._v = new Float64Array(N));
      for (let i = 0; i < N; i++) { let a = 0; for (let c = 0; c < B; c++) a += HS[i * B + c] * H[i * B + c]; v[i] = a; }
    }
  }
  let rII = 0, rIB = 0;
  for (let i = 0; i < N; i++) {
    const ri = i * N, hi = i * B;
    if (apply) Wii[ri + i] = 0;
    for (let j = i + 1; j < N; j++) {
      if (Mii[ri + j] === 0) { if (apply) { Wii[ri + j] = 0; Wii[j * N + i] = 0; } continue; }
      let cov = 0;
      if (rates) { const a = i * nb, b2 = j * nb; for (let r = 0; r < nb; r++) cov += Rt[a + r] * Rt[b2 + r]; cov /= den; }
      else {
        const hj = j * B;
        for (let c = 0; c < B; c++) cov += HS[hi + c] * H[hj + c];
        if (relu) cov = dualRelu(v[i], v[j], cov);
      }
      const w = Wi[ri + j];
      const f = (w0 - lam * w) + ge * cov;
      const af = (w <= wmin && f < 0) ? 0 : Math.abs(f);
      if (af > rII) rII = af;
      if (apply) {
        let nw = w + dt * f;
        if (nw < wmin) nw = wmin;
        Wii[ri + j] = nw; Wii[j * N + i] = nw;
      }
    }
  }
  for (let i = 0; i < N; i++) for (let b = 0; b < B; b++) {
    const k = i * B + b;
    if (Mib[k] === 0) { if (apply) Wib[k] = 0; continue; }
    let cov;
    if (rates) { cov = 0; const a = i * nb, c2 = b * nb; for (let r = 0; r < nb; r++) cov += Rt[a + r] * Rbt[c2 + r]; cov /= den; }
    else cov = relu ? dualRelu(v[i], Sigma[b * B + b], HS[k]) : HS[k];
    const w = Wb[k];
    const f = (w0 - lam * w) + eta * cov;
    const af = (w <= wmin && f < 0) ? 0 : Math.abs(f);
    if (af > rIB) rIB = af;
    if (apply) { const nw = w + dt * f; Wib[k] = nw < wmin ? wmin : nw; }
  }
  const r = Math.max(rII, rIB);
  return { r, rII, rIB };
}

// Rates from sampled voltages: u = H g (rows of G are samples g), r = relu(u − mu) with mu = 0
// (known zero mean) or the batch mean (center); then centred by the batch mean of the rates.
// Returns transposed, centred rate matrices Rt (N×batch) and Rbt (B×batch).
function sampleRates(s, G, batch, center) {
  const { N, B, H } = s;
  if (batch < 2) throw new Error("nonlin 'relu' stochastic step needs batch >= 2 (rate covariance uses the batch mean)");
  if (!s._Rt || s._Rt.length < N * batch) s._Rt = new Float64Array(N * batch);
  if (!s._Rbt || s._Rbt.length < B * batch) s._Rbt = new Float64Array(B * batch);
  const Rt = s._Rt, Rbt = s._Rbt;
  for (let i = 0; i < N; i++) {
    const a = i * batch; let mu = 0;
    for (let r = 0; r < batch; r++) { let u = 0; for (let c = 0; c < B; c++) u += G[r * B + c] * H[i * B + c]; Rt[a + r] = u; mu += u; }
    mu = center ? mu / batch : 0;
    let m = 0;
    for (let r = 0; r < batch; r++) { const x = Rt[a + r] - mu; const y = x > 0 ? x : 0; Rt[a + r] = y; m += y; }
    m /= batch;
    for (let r = 0; r < batch; r++) Rt[a + r] -= m;
  }
  for (let b = 0; b < B; b++) {
    const a = b * batch; let mu = 0;
    for (let r = 0; r < batch; r++) mu += G[r * B + b];
    mu = center ? mu / batch : 0;
    let m = 0;
    for (let r = 0; r < batch; r++) { const x = G[r * B + b] - mu; const y = x > 0 ? x : 0; Rbt[a + r] = y; m += y; }
    m /= batch;
    for (let r = 0; r < batch; r++) Rbt[a + r] -= m;
  }
  return { Rt, Rbt, nb: batch };
}

/** One Euler step, arithmetic identical to core.step (nonlin 'linear') or core.step_nl.
 *  Mutates state; returns state.H, which is H of the weights *before* this update (what
 *  the update used).  Sets state.resid = projected stationarity residual of those weights. */
export function step(s, Sigma, P) {
  const res = kernel(s, Sigma, P, true);
  s.t++; s.resid = res.r; s.residII = res.rII; s.residIB = res.rIB;
  return s.H;
}

/** n mean-field steps. Returns state.H (H of the weights before the last update). */
export function advance(s, Sigma, P, nSteps) {
  for (let k = 0; k < nSteps; k++) step(s, Sigma, P);
  return s.H;
}

/** Projected stationarity residual (SPEC §1) of the current weights: max |dw/dt| over
 *  potential edges, edges at the floor with negative drift count 0.  Does not modify
 *  the weights; overwrites state.H with H of the (floor-clipped) current weights.
 *  Returns {r, rII, rIB}, in weight-per-time units; r/eta is in covariance units. */
export function stationarity(s, Sigma, P) { return kernel(s, Sigma, P, false); }

/** Second-moment estimate of a batch G (batch×B) into out (B×B).
 *  center=false: G^T G / batch (zero mean known: sampler is zero-mean, works for batch=1).
 *  center=true:  sample covariance with the batch mean removed, divided by (batch-1). */
export function sampleCov(G, batch, B, out = new Float64Array(B * B), center = false) {
  out.fill(0, 0, B * B);
  let mean = null;
  if (center) {
    mean = new Float64Array(B);
    for (let r = 0; r < batch; r++) for (let b = 0; b < B; b++) mean[b] += G[r * B + b] / batch;
  }
  for (let r = 0; r < batch; r++) {
    const gr = r * B;
    for (let b = 0; b < B; b++) {
      const x = center ? G[gr + b] - mean[b] : G[gr + b];
      for (let c = b; c < B; c++) out[b * B + c] += x * (center ? G[gr + c] - mean[c] : G[gr + c]);
    }
  }
  const den = center ? batch - 1 : batch;
  for (let b = 0; b < B; b++) for (let c = b; c < B; c++) { const v = out[b * B + c] / den; out[b * B + c] = v; out[c * B + b] = v; }
  return out;
}

/** Step with an explicit batch of samples G (batch×B).  nonlin 'linear': Sigma is replaced by
 *  sampleCov(G, center).  nonlin 'relu': rates are computed from the sampled voltages u = H g
 *  (and boundary rates from g) and their minibatch covariance drives plasticity (batch >= 2). */
export function stepSamples(s, G, batch, P, center = false) {
  if ((P.nonlin || 'linear') === 'relu') {
    const res = kernel(s, null, P, true, { G, batch, center });
    s.t++; s.resid = res.r; s.residII = res.rII; s.residIB = res.rIB;
    return s.H;
  }
  sampleCov(G, batch, s.B, s.Shat, center);
  return step(s, s.Shat, P);
}

/** Stochastic step: draw `batch` samples from `sampler` and step with their covariance. */
export function stepStochastic(s, sampler, P, batch = 1, center = false) {
  if (!s.G || s.G.length < batch * s.B) s.G = new Float64Array(batch * s.B);
  sampler.draw(s.G, batch);
  return stepSamples(s, s.G, batch, P, center);
}

/** Zero-mean Gaussian sampler with covariance Sigma (B×B, PSD; rank-deficient OK).
 *  Uses the symmetric square root's eigen-factor F = V sqrt(Λ+), g = F z. */
export function makeSampler(Sigma, B, seed = 0) {
  const { values, vectors } = eigh(Sigma, B);
  const F = new Float64Array(B * B);
  for (let b = 0; b < B; b++) for (let k = 0; k < B; k++) F[b * B + k] = vectors[b * B + k] * Math.sqrt(Math.max(values[k], 0));
  const rng = makeRng(seed), z = new Float64Array(B);
  return {
    B, F,
    draw(G, batch) {
      for (let r = 0; r < batch; r++) {
        for (let k = 0; k < B; k++) z[k] = rng.normal();
        for (let b = 0; b < B; b++) { let v = 0; const fb = b * B; for (let k = 0; k < B; k++) v += F[fb + k] * z[k]; G[r * B + b] = v; }
      }
      return G;
    },
  };
}

/** Euler stability guidance. The linear part w' = -lam w + ... is monotone for dt*lam < 1
 *  (stable for < 2); weights stay below max(w(0), gamma_or_1*eta*(max diag Sigma - theta)/lam).
 *  Returns {dtMax, wBound}. */
export function dtAdvice(Sigma, B, P) {
  let mx = -Infinity; for (let b = 0; b < B; b++) mx = Math.max(mx, Sigma[b * B + b]);
  const g = Math.max(1, P.gamma);
  return { dtMax: 1 / P.lam, wBound: Math.max(0, g * P.eta * (mx - P.theta) / P.lam) };
}

// ─────────────────────────── covariance generators ───────────────────────────

/** Clamp a symmetric matrix to PSD only if it has a negative eigenvalue; for a matrix
 *  with unit diagonal (correlation), the diagonal is restored to 1 afterwards. */
export function clampPSD(A, n, { unitDiag = false } = {}) {
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

/** 3×3 covariance from variances [v0,v1,v2] and correlations [r01, r02, r12],
 *  correlation matrix clamped to PSD (unit diagonal kept). */
export function cov3({ variances = [1, 1, 1], corr = [0, 0, 0] } = {}) {
  const [a, b, c] = corr;
  const R = clampPSD(Float64Array.from([1, a, b, a, 1, c, b, c, 1]), 3, { unitDiag: true });
  return scaleByVariances(R, 3, variances);
}

/** Planted blocks: sizes [n1, n2, ...]; correlation `within` inside a block (scalar or
 *  per-block array), `between` across blocks; per-block variance (scalar or array).
 *  Returns {Sigma, B, labels}. Correlation clamped to PSD if needed. */
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
  return { Sigma: scaleByVariances(clampPSD(R, B, { unitDiag: true }), B, v), B, labels };
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
  const S = clampPSD(R, B, { unitDiag: true });
  for (let k = 0; k < B * B; k++) S[k] *= variance;
  return { Sigma: S, B, labels };
}

/** Haar-ish random orthogonal matrix: modified Gram–Schmidt on the columns of a Gaussian
 *  matrix drawn row-major from rng.normal(). Column k is basis vector k. */
export function randomOrthogonal(B, rng) {
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

/** Random covariance with power-law spectrum lambda_k ∝ k^-alpha (k = 1..B) in a random
 *  orthogonal basis.  nonneg: add c·11^T with the smallest c making all off-diagonal
 *  entries >= 0 (a rank-one, all-positive component: exactly the direction the
 *  contrast-subspace score projects out).  normalize: mean variance 1. */
export function covPowerLaw({ B, alpha = 1, seed = 0, nonneg = false, normalize = true }) {
  const Q = randomOrthogonal(B, makeRng(seed));
  const S = new Float64Array(B * B);
  for (let k = 0; k < B; k++) {
    const lam = Math.pow(k + 1, -alpha);
    for (let i = 0; i < B; i++) { const q = Q[i * B + k] * lam; for (let j = 0; j < B; j++) S[i * B + j] += q * Q[j * B + k]; }
  }
  for (let i = 0; i < B; i++) for (let j = i + 1; j < B; j++) { const v = 0.5 * (S[i * B + j] + S[j * B + i]); S[i * B + j] = v; S[j * B + i] = v; }
  if (nonneg) {
    let mn = Infinity; for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) if (i !== j) mn = Math.min(mn, S[i * B + j]);
    const c = Math.max(0, -mn); for (let k = 0; k < B * B; k++) S[k] += c;
  }
  return normalize ? normalizeMeanDiag(S, B) : S;
}

/** Wishart: X (dof×B) standard normal, Sigma = X^T X / dof (rank min(dof, B)). */
export function covWishart({ B, dof = B, seed = 0, normalize = true }) {
  const rng = makeRng(seed), X = new Float64Array(dof * B);
  for (let k = 0; k < dof * B; k++) X[k] = rng.normal();
  const S = matmulAtB(X, X, dof, B, B);
  for (let k = 0; k < B * B; k++) S[k] /= dof;
  return normalize ? normalizeMeanDiag(S, B) : S;
}

export function toCorrelation(S, B) {
  const out = new Float64Array(B * B);
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) out[i * B + j] = S[i * B + j] / Math.sqrt(S[i * B + i] * S[j * B + j]);
  return out;
}

// ─────────────────────────────── analysis ───────────────────────────────

function l1(H, B, i, j) { let d = 0; const a = i * B, b = j * B; for (let c = 0; c < B; c++) d += Math.abs(H[a + c] - H[b + c]); return d; }

function compactLabels(root, N) {
  const map = new Map(), labels = new Int32Array(N);
  for (let i = 0; i < N; i++) { const r = root(i); if (!map.has(r)) map.set(r, map.size); labels[i] = map.get(r); }
  return { labels, K: map.size };
}

/** Mirror of core.clusters: greedy leader clustering in L1 (order-dependent; quick look only).
 *  Returns {labels, K, centroids (K×B, the leader embeddings)}. */
export function clustersLeader(H, N, B, tol = 0.1) {
  const leaders = [], labels = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    let found = -1;
    for (let k = 0; k < leaders.length; k++) if (l1(H, B, i, leaders[k]) < tol) { found = k; break; }
    if (found < 0) { found = leaders.length; leaders.push(i); }
    labels[i] = found;
  }
  const centroids = new Float64Array(leaders.length * B);
  leaders.forEach((i, k) => centroids.set(H.subarray(i * B, i * B + B), k * B));
  return { labels, K: leaders.length, centroids };
}

/** Minimum spanning tree of the embeddings under L1 distance (Prim, O(N^2 B)).
 *  Returns {parent (Int32Array, -1 at the root), weight (Float64Array)}; edge i–parent[i]. */
export function mstL1(H, N, B) {
  const parent = new Int32Array(N).fill(-1), weight = new Float64Array(N);
  const best = new Float64Array(N).fill(Infinity), inT = new Uint8Array(N);
  if (N === 0) return { parent, weight };
  best[0] = 0;
  for (let it = 0; it < N; it++) {
    let u = -1, bu = Infinity;
    for (let i = 0; i < N; i++) if (!inT[i] && best[i] < bu) { bu = best[i]; u = i; }
    inT[u] = 1; weight[u] = it === 0 ? 0 : bu;
    for (let v = 0; v < N; v++) if (!inT[v]) { const d = l1(H, B, u, v); if (d < best[v]) { best[v] = d; parent[v] = u; } }
  }
  return { parent, weight };
}

function cutMST(mst, N, tol) {
  const par = new Int32Array(N); for (let i = 0; i < N; i++) par[i] = i;
  const find = x => { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; };
  for (let i = 0; i < N; i++) if (mst.parent[i] >= 0 && mst.weight[i] < tol) { const a = find(i), b = find(mst.parent[i]); if (a !== b) par[Math.max(a, b)] = Math.min(a, b); }
  return compactLabels(find, N);
}

/** Single-linkage clusters at a fixed L1 tolerance: i~j if a chain of embeddings with
 *  consecutive L1 distance < tol connects them.  Order-independent.  Labels are numbered
 *  by smallest member index.  Returns {labels, K}. */
export function clustersL1(H, N, B, tol = 0.1, mst = null) {
  return cutMST(mst || mstL1(H, N, B), N, tol);
}

/** Headline cluster definition (default).  Single linkage in L1, with the cut placed in
 *  the largest multiplicative gap of the MST merge heights among heights >= minSep.
 *  If the merge heights are all below minSep → one cluster.  `robust` = gap ratio >=
 *  minGap (otherwise the state is not clearly clustered; K is then the count at tol=fallbackTol).
 *  Returns {labels, K, tol, gap, robust, heights (sorted MST weights)}. */
export function clustersAuto(H, N, B, { minSep = 0.05, minGap = 3, fallbackTol = 0.1, floor = 1e-9 } = {}) {
  const mst = mstL1(H, N, B);
  const hs = Array.from(mst.weight.subarray(0, N)).filter((_, i) => mst.parent[i] >= 0).sort((a, b) => a - b);
  let bestR = 0, tol = Infinity;
  for (let k = 0; k < hs.length; k++) {
    if (hs[k] < minSep) continue;
    const lo = k > 0 ? Math.max(hs[k - 1], floor) : floor;
    const r = hs[k] / lo;
    if (r > bestR) { bestR = r; tol = Math.sqrt(hs[k] * lo); }
  }
  if (hs.length === 0 || hs[hs.length - 1] < minSep) { const c = cutMST(mst, N, Infinity); return { ...c, tol: Infinity, gap: Infinity, robust: true, heights: Float64Array.from(hs) }; }
  const robust = bestR >= minGap;
  const c = cutMST(mst, N, robust ? tol : fallbackTol);
  return { ...c, tol: robust ? tol : fallbackTol, gap: bestR, robust, heights: Float64Array.from(hs) };
}

/** Number of single-linkage clusters at each tolerance in `tols` (for robustness plots). */
export function clusterCountCurve(H, N, B, tols) {
  const mst = mstL1(H, N, B);
  return tols.map(t => { let K = 1; for (let i = 0; i < N; i++) if (mst.parent[i] >= 0 && mst.weight[i] >= t) K++; return K; });
}

/** Connected components of the active lateral-synapse graph (Wii > wmin(1+eps)).
 *  Early in training (Wii at the floor) every neuron is its own component. */
export function clustersSynapse(Wii, Mii, N, wmin, eps = 1e-6) {
  const par = new Int32Array(N); for (let i = 0; i < N; i++) par[i] = i;
  const find = x => { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; };
  const thr = wmin * (1 + eps);
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++)
    if (Mii[i * N + j] && Wii[i * N + j] > thr) { const a = find(i), b = find(j); if (a !== b) par[Math.max(a, b)] = Math.min(a, b); }
  return compactLabels(find, N);
}

/** Per-cluster summary.  part = member mean of h (the representative); energy = part^T Σ part;
 *  spread = mean L1 distance of members to the part.  Returns {sizes, parts (K×B), energy, spread}. */
export function clusterStats(H, N, B, labels, K, Sigma) {
  const sizes = new Int32Array(K), parts = new Float64Array(K * B), energy = new Float64Array(K), spread = new Float64Array(K);
  for (let i = 0; i < N; i++) { const k = labels[i]; sizes[k]++; for (let b = 0; b < B; b++) parts[k * B + b] += H[i * B + b]; }
  for (let k = 0; k < K; k++) for (let b = 0; b < B; b++) parts[k * B + b] /= sizes[k];
  for (let i = 0; i < N; i++) { const k = labels[i]; let d = 0; for (let b = 0; b < B; b++) d += Math.abs(H[i * B + b] - parts[k * B + b]); spread[k] += d / sizes[k]; }
  for (let k = 0; k < K; k++) {
    let e = 0; for (let b = 0; b < B; b++) { let s = 0; for (let c = 0; c < B; c++) s += Sigma[b * B + c] * parts[k * B + c]; e += parts[k * B + b] * s; }
    energy[k] = e;
  }
  return { sizes, parts, energy, spread };
}

/** Relabel `labels` to maximise overlap with `prev` (greedy, largest overlaps first), so
 *  cluster colours stay stable across frames.  New clusters get fresh ids.  JS-only helper. */
export function relabel(prev, labels, N) {
  if (!prev) return labels;
  const cnt = new Map();
  for (let i = 0; i < N; i++) { const key = labels[i] + ',' + prev[i]; cnt.set(key, (cnt.get(key) || 0) + 1); }
  const pairs = [...cnt.entries()].map(([k, c]) => { const [a, b] = k.split(',').map(Number); return [c, a, b]; }).sort((x, y) => y[0] - x[0] || x[1] - y[1]);
  const mapNew = new Map(), usedOld = new Set();
  for (const [, a, b] of pairs) if (!mapNew.has(a) && !usedOld.has(b)) { mapNew.set(a, b); usedOld.add(b); }
  let next = 0; for (let i = 0; i < N; i++) next = Math.max(next, prev[i] + 1);
  const out = new Int32Array(N);
  for (let i = 0; i < N; i++) { const a = labels[i]; if (!mapNew.has(a)) { while (usedOld.has(next)) next++; mapNew.set(a, next); usedOld.add(next); } out[i] = mapNew.get(a); }
  return out;
}

/** Permutation of 0..N-1 sorted by cluster label then index (for sorted similarity matrices). */
export function orderByCluster(labels, N) {
  return Int32Array.from(Array.from({ length: N }, (_, i) => i).sort((a, b) => labels[a] - labels[b] || a - b));
}

/** Effective rank of H: participation ratio of its squared singular values,
 *  (Σσ²)² / Σσ⁴ = ||H||_F^4 / ||H^T H||_F^2.  Uncentred: K orthogonal clusters → K. */
export function effectiveRank(H, N, B) {
  const G = matmulAtB(H, H, N, B, B);
  let f2 = 0, g2 = 0;
  for (let b = 0; b < B; b++) f2 += G[b * B + b];
  for (let k = 0; k < B * B; k++) g2 += G[k] * G[k];
  return g2 > 0 ? (f2 * f2) / g2 : 0;
}

/** Similarity matrix H Σ H^T (N×N, the voltage covariance). */
export function similarity(H, N, B, Sigma, out = new Float64Array(N * N)) {
  const HS = matmul(H, Sigma, N, B, B);
  for (let i = 0; i < N; i++) for (let j = i; j < N; j++) {
    let s = 0; for (let c = 0; c < B; c++) s += HS[i * B + c] * H[j * B + c];
    out[i * N + j] = s; out[j * N + i] = s;
  }
  return out;
}

/** PCA of the embeddings to 3D.  Basis = top-3 eigenvectors of the centred covariance of
 *  the rows of H.  Orientation: if `prev` (a previous result) is given, the basis is
 *  rotated within the new top-3 subspace by the orthogonal Procrustes fit to prev.basis
 *  (no flips/jumps across frames; axes then need not be exactly the principal axes);
 *  otherwise each axis is signed so its largest-|.| component is positive.
 *  Returns {coords (N×3), basis (B×3), mean (B), variance (3, along the returned axes)}. */
export function pca3(H, N, B, prev = null) {
  const mean = new Float64Array(B);
  for (let i = 0; i < N; i++) for (let b = 0; b < B; b++) mean[b] += H[i * B + b] / N;
  const X = new Float64Array(N * B);
  for (let i = 0; i < N; i++) for (let b = 0; b < B; b++) X[i * B + b] = H[i * B + b] - mean[b];
  const Cv = matmulAtB(X, X, N, B, B); for (let k = 0; k < B * B; k++) Cv[k] /= N;
  const { vectors } = eigh(Cv, B);
  let basis = new Float64Array(B * 3);
  const kk = Math.min(3, B);
  for (let b = 0; b < B; b++) for (let k = 0; k < kk; k++) basis[b * 3 + k] = vectors[b * B + k];
  let aligned = false;
  if (prev && prev.basis && prev.basis.length === B * 3) {
    const K = matmulAtB(basis, prev.basis, B, 3, 3);          // new^T prev
    const KtK = matmulAtB(K, K, 3, 3, 3);
    const { values } = eigh(KtK, 3);
    if (values[2] > 1e-10 * Math.max(values[0], 1e-300)) {
      const Mi = symFn(KtK, 3, x => 1 / Math.sqrt(x));
      const R = matmul(K, Mi, 3, 3, 3);
      basis = matmul(basis, R, B, 3, 3);
      aligned = true;
    }
  }
  if (!aligned) for (let k = 0; k < kk; k++) {
    let bi = 0; for (let b = 1; b < B; b++) if (Math.abs(basis[b * 3 + k]) > Math.abs(basis[bi * 3 + k]) + 1e-12) bi = b;
    if (basis[bi * 3 + k] < 0) for (let b = 0; b < B; b++) basis[b * 3 + k] = -basis[b * 3 + k];
  }
  const coords = matmul(X, basis, N, B, 3);
  const variance = new Float64Array(3);
  for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) variance[k] += coords[i * 3 + k] ** 2 / N;
  return { coords, basis, mean, variance };
}

/** Project any B-vectors (rows of X, n×B), e.g. the boundary vertices e_b, with a pca3 result. */
export function projectPCA(pca, X, n, B) {
  const Y = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { let s = 0; for (let b = 0; b < B; b++) s += (X[i * B + b] - pca.mean[b]) * pca.basis[b * 3 + k]; Y[i * 3 + k] = s; }
  return Y;
}

// ─────────────────────────── weight-space readout (§3, §7) ───────────────────────────

/** Fixed-point equations on active edges.  Each equation reads
 *     coef · h_i^T C x = rhs          (x = h_j for I–I with coef = gamma, x = e_b for I–B with coef = 1)
 *  rhs = thetaRhs + (lam/eta)·w + (1/eta)·dw   (dw = observed dw/dt; 0 for the equilibrium readout).
 *  Active: potential edge with w > wmin(1+eps) (or, if `activeII/activeIB` arrays are given, those).
 *  Returns {ii: Int32Array [i0,j0,i1,j1,...], iiR, ib: Int32Array [i0,b0,...], ibR, gamma}. */
export function readoutEquations(Wii, Wib, Mii, Mib, N, B, o) {
  const thr = o.wmin * (1 + (o.eps ?? 1e-6)), kap = o.lam / o.eta, th = o.thetaRhs ?? o.theta;
  const useII = (o.use ?? 'both') !== 'ib' && o.gamma !== 0, useIB = (o.use ?? 'both') !== 'ii';
  const ii = [], iiR = [], ib = [], ibR = [];
  if (useII) for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    const k = i * N + j;
    if (!Mii[k]) continue;
    const act = o.activeII ? o.activeII[k] > thr : Wii[k] > thr;
    if (!act) continue;
    ii.push(i, j); iiR.push(th + kap * Wii[k] + (o.dWii ? o.dWii[k] / o.eta : 0));
  }
  if (useIB) for (let i = 0; i < N; i++) for (let b = 0; b < B; b++) {
    const k = i * B + b;
    if (!Mib[k]) continue;
    const act = o.activeIB ? o.activeIB[k] > thr : Wib[k] > thr;
    if (!act) continue;
    ib.push(i, b); ibR.push(th + kap * Wib[k] + (o.dWib ? o.dWib[k] / o.eta : 0));
  }
  return { ii: Int32Array.from(ii), iiR: Float64Array.from(iiR), ib: Int32Array.from(ib), ibR: Float64Array.from(ibR), gamma: o.gamma };
}

// Measurement operator A: Sym(B) (+ optional scalar tau) -> R^m, and its adjoint.
function makeOperator(H, N, B, eq, fitTheta, wII = 1, wIB = 1) {
  const mII = eq.ii.length / 2, mIB = eq.ib.length / 2, g = eq.gamma;
  const HC = new Float64Array(N * B), T = new Float64Array(N * B);
  const n = B * B + 1;
  function A(x, y) {           // y[0..mII) II, y[mII..) IB
    matmul(H, x, N, B, B, HC);
    const tau = fitTheta ? x[B * B] : 0;
    for (let e = 0; e < mII; e++) {
      const i = eq.ii[2 * e] * B, j = eq.ii[2 * e + 1] * B; let s = 0;
      for (let c = 0; c < B; c++) s += HC[i + c] * H[j + c];
      y[e] = g * s - tau;
    }
    for (let e = 0; e < mIB; e++) y[mII + e] = HC[eq.ib[2 * e] * B + eq.ib[2 * e + 1]] - tau;
    return y;
  }
  function At(r, out) {        // out = A^T W r  (W = family weights)
    T.fill(0); let st = 0;
    for (let e = 0; e < mII; e++) {
      const re = wII * r[e]; st += re; if (re === 0) continue;
      const i = eq.ii[2 * e] * B, j = eq.ii[2 * e + 1] * B, cf = g * re;
      for (let c = 0; c < B; c++) T[i + c] += cf * H[j + c];
    }
    for (let e = 0; e < mIB; e++) { const re = wIB * r[mII + e]; st += re; T[eq.ib[2 * e] * B + eq.ib[2 * e + 1]] += re; }
    matmulAtB(H, T, N, B, B, out);
    for (let b = 0; b < B; b++) for (let c = b + 1; c < B; c++) { const v = 0.5 * (out[b * B + c] + out[c * B + b]); out[b * B + c] = v; out[c * B + b] = v; }
    out[B * B] = fitTheta ? -st : 0;
    return out;
  }
  // trace of A^T W A restricted to the C block (for the relative ridge)
  let tr = 0;
  const nrm = new Float64Array(N);
  for (let i = 0; i < N; i++) { let s = 0; for (let c = 0; c < B; c++) s += H[i * B + c] ** 2; nrm[i] = s; }
  for (let e = 0; e < mII; e++) {
    const i = eq.ii[2 * e], j = eq.ii[2 * e + 1]; let d = 0;
    for (let c = 0; c < B; c++) d += H[i * B + c] * H[j * B + c];
    tr += wII * g * g * (nrm[i] * nrm[j] + d * d) / 2;
  }
  for (let e = 0; e < mIB; e++) { const i = eq.ib[2 * e], b = eq.ib[2 * e + 1]; tr += wIB * (nrm[i] + H[i * B + b] ** 2) / 2; }
  return { A, At, n, m: mII + mIB, mII, mIB, trace: tr };
}

// P C P with P = I - 11^T/B
export function project1(C, B, out = new Float64Array(B * B)) {
  const r = new Float64Array(B); let m = 0;
  for (let b = 0; b < B; b++) { let s = 0; for (let c = 0; c < B; c++) s += C[b * B + c]; r[b] = s / B; m += s; }
  m /= B * B;
  for (let b = 0; b < B; b++) for (let c = 0; c < B; c++) out[b * B + c] = C[b * B + c] - r[b] - r[c] + m;
  return out;
}

/** Masked ridge least squares readout (SPEC §3/§7.3), from H and weights.
 *  opts: theta, lam, eta, gamma, wmin (assumed model parameters);
 *    use: 'both' | 'ii' | 'ib' (edge families; gamma = 0 drops I–I, which then carries no C information);
 *    thetaMode: 'known' (rhs uses theta) | 'fit' (theta is an extra unpenalised unknown; with
 *       gamma != 1 and both families it is separately identifiable) | 'project' (theta assumed 0,
 *       ridge only on P C P so the 1-directions absorb theta; exact only for gamma = 1; returns P C P);
 *    ridge (relative to the mean eigenvalue of the normal operator, default 1e-6), tol, maxIter,
 *    x0 (warm-start C), psd (clip negative eigenvalues of the result), eps (active threshold),
 *    dWii/dWib (observed dw/dt for the derivative-aware transient readout), activeII/activeIB,
 *    weightII/weightIB (family weights).
 *  Returns {C (B×B), theta, iters, relResidual, rmsII, rmsIB, mII, mIB, alpha, negMass, warnings}. */
export function readoutLS(H, Wii, Wib, Mii, Mib, N, B, opts) {
  const o = { eps: 1e-6, use: 'both', thetaMode: 'known', ridge: 1e-6, tol: 1e-10, maxIter: 2000, psd: false, weightII: 1, weightIB: 1, ...opts };
  const eq = readoutEquations(Wii, Wib, Mii, Mib, N, B, { ...o, thetaRhs: o.thetaMode === 'known' ? o.theta : 0 });
  return solveEquations(H, N, B, eq, o);
}

/** Solve given equations (from readoutEquations, or custom) for symmetric C by ridge CG.
 *  Same options as readoutLS (thetaMode decides only how theta/1-directions are handled;
 *  the rhs is taken as given). */
export function solveEquations(H, N, B, eq, opts) {
  const o = { use: 'both', thetaMode: 'known', ridge: 1e-6, tol: 1e-10, maxIter: 2000, psd: false, weightII: 1, weightIB: 1, ...opts };
  const warnings = [];
  if (o.gamma === 0 && o.use !== 'ib') warnings.push('gamma = 0: I–I edges carry no information about C; using I–B only');
  if (o.thetaMode === 'project' && o.gamma !== 1) warnings.push('thetaMode "project" assumes gamma = 1 (unknown theta is a pure 11^T shift only then); use "fit"');
  const fit = o.thetaMode === 'fit';
  const op = makeOperator(H, N, B, eq, fit, o.weightII, o.weightIB);
  const n = op.n, m = op.m, p = B * (B + 1) / 2;
  const alpha = o.ridge * op.trace / p;
  const eps2 = o.thetaMode === 'project' ? 1e-4 : 1;
  const rhs = new Float64Array(m); rhs.set(eq.iiR, 0); rhs.set(eq.ibR, op.mII);
  const y = new Float64Array(m), tmp = new Float64Array(n), Pc = new Float64Array(B * B);
  const Op = (x, out) => {
    op.A(x, y); op.At(y, out);
    if (o.thetaMode === 'project') {
      project1(x, B, Pc);
      for (let k = 0; k < B * B; k++) out[k] += alpha * (Pc[k] + eps2 * (x[k] - Pc[k]));
    } else for (let k = 0; k < B * B; k++) out[k] += alpha * x[k];
    if (!fit) out[B * B] = 0;
    return out;
  };
  const b = op.At(rhs, new Float64Array(n));
  const x = new Float64Array(n);
  if (o.x0) x.set(o.x0.subarray(0, B * B));
  if (fit && o.theta0 !== undefined) x[B * B] = o.theta0;
  const r = new Float64Array(n), d = new Float64Array(n), q = new Float64Array(n);
  Op(x, tmp); for (let k = 0; k < n; k++) r[k] = b[k] - tmp[k];
  d.set(r);
  let rr = dot(r, r, n), bn = Math.sqrt(dot(b, b, n)) || 1, it = 0;
  while (it < o.maxIter && Math.sqrt(rr) > o.tol * bn) {
    Op(d, q);
    const a = rr / dot(d, q, n);
    for (let k = 0; k < n; k++) { x[k] += a * d[k]; r[k] -= a * q[k]; }
    const rn = dot(r, r, n), be = rn / rr; rr = rn;
    for (let k = 0; k < n; k++) d[k] = r[k] + be * d[k];
    it++;
  }
  // equation residuals
  op.A(x, y);
  let sII = 0, sIB = 0;
  for (let e = 0; e < op.mII; e++) sII += (y[e] - rhs[e]) ** 2;
  for (let e = op.mII; e < m; e++) sIB += (y[e] - rhs[e]) ** 2;
  let C = x.slice(0, B * B);
  if (o.thetaMode === 'project') C = project1(C, B);
  let negMass = 0;
  if (o.psd) { const P = psdProject(C, B); negMass = P.negMass; C = new Float64Array(P); }
  return {
    C, theta: o.thetaMode === 'fit' ? x[B * B] : o.thetaMode === 'known' ? o.theta : NaN,
    iters: it, relResidual: Math.sqrt(rr) / bn, rmsII: op.mII ? Math.sqrt(sII / op.mII) : 0, rmsIB: op.mIB ? Math.sqrt(sIB / op.mIB) : 0,
    mII: op.mII, mIB: op.mIB, alpha, negMass, warnings,
  };
}

function dot(a, b, n) { let s = 0; for (let k = 0; k < n; k++) s += a[k] * b[k]; return s; }

/** Readout from a state: computes H from the weights, then readoutLS. */
export function readoutState(s, P, opts = {}) {
  const H = harmonicFrom(s.Wii, s.Wib, s.N, s.B);
  return readoutLS(H, s.Wii, s.Wib, s.Mii, s.Mib, s.N, s.B, { theta: P.theta, lam: P.lam, eta: P.eta, gamma: P.gamma, wmin: P.wmin, ...opts });
}

/** Readout under nonlin 'relu' (equilibrium, parameters known).  Active-edge equations are
 *  gamma·dualRelu(v_i, v_j, h_i^T C h_j) = rhs (I–I) and dualRelu(v_i, C_bb, (H C)_ib) = rhs (I–B)
 *  with v_i = h_i^T C h_i.  The variances are NOT observed (no self-edges), so this alternates:
 *  v from the current C → invert the dual activation per edge (dualReluInv) → linear ridge LS
 *  (solveEquations, warm-started) → repeat.  Initial C: the linear readout of 3×rhs (the dual
 *  activation has slope 1/4 at rho = 0 and ~1/3 near rho = 1).  thetaMode must be 'known'
 *  (theta enters non-linearly).  opts as readoutLS plus outer (default 50), outerTol (1e-10).
 *  Returns readoutLS's fields plus {outer, change, rmsRateII, rmsRateIB} (residuals in rate units). */
export function readoutRelu(H, Wii, Wib, Mii, Mib, N, B, opts) {
  const o = { eps: 1e-6, use: 'both', ridge: 1e-6, tol: 1e-12, maxIter: 5000, psd: false, outer: 50, outerTol: 1e-10, ...opts, thetaMode: 'known' };
  const eq = readoutEquations(Wii, Wib, Mii, Mib, N, B, { ...o, thetaRhs: o.theta });
  return solveRelu(H, N, B, eq, o);
}

export function solveRelu(H, N, B, eq, opts) {
  const o = { outer: 50, outerTol: 1e-10, tol: 1e-12, maxIter: 5000, ...opts, thetaMode: 'known' };
  const mII = eq.ii.length / 2, mIB = eq.ib.length / 2, g = eq.gamma;
  const kII = Float64Array.from(eq.iiR, x => x / g), kIB = eq.ibR;
  const lin = { ii: eq.ii, ib: eq.ib, gamma: 1, iiR: new Float64Array(mII), ibR: new Float64Array(mIB) };
  for (let e = 0; e < mII; e++) lin.iiR[e] = 3 * kII[e];
  for (let e = 0; e < mIB; e++) lin.ibR[e] = 3 * kIB[e];
  let res = solveEquations(H, N, B, lin, { ...o, psd: false, gamma: 1 });
  let C = res.C, change = Infinity, t = 0;
  const v = new Float64Array(N), HC = new Float64Array(N * B), tiny = 1e-12;
  const variances = C => {
    matmul(H, C, N, B, B, HC);
    for (let i = 0; i < N; i++) { let a = 0; for (let c = 0; c < B; c++) a += HC[i * B + c] * H[i * B + c]; v[i] = Math.max(a, tiny); }
  };
  for (t = 0; t < o.outer; t++) {
    variances(C);
    for (let e = 0; e < mII; e++) lin.iiR[e] = dualReluInv(v[eq.ii[2 * e]], v[eq.ii[2 * e + 1]], kII[e]);
    for (let e = 0; e < mIB; e++) { const b = eq.ib[2 * e + 1]; lin.ibR[e] = dualReluInv(v[eq.ib[2 * e]], Math.max(C[b * B + b], tiny), kIB[e]); }
    res = solveEquations(H, N, B, lin, { ...o, psd: false, gamma: 1, x0: C });
    let d = 0, n = 0;
    for (let k = 0; k < B * B; k++) { d += (res.C[k] - C[k]) ** 2; n += res.C[k] ** 2; }
    C = res.C; change = Math.sqrt(d / (n || 1));
    if (change < o.outerTol) { t++; break; }
  }
  // residuals in rate units
  variances(C);
  let sII = 0, sIB = 0;
  for (let e = 0; e < mII; e++) {
    const i = eq.ii[2 * e], j = eq.ii[2 * e + 1]; let c = 0;
    for (let q = 0; q < B; q++) c += HC[i * B + q] * H[j * B + q];
    sII += (g * dualRelu(v[i], v[j], c) - eq.iiR[e]) ** 2;
  }
  for (let e = 0; e < mIB; e++) { const i = eq.ib[2 * e], b = eq.ib[2 * e + 1]; sIB += (dualRelu(v[i], Math.max(C[b * B + b], tiny), HC[i * B + b]) - eq.ibR[e]) ** 2; }
  let negMass = 0;
  if (o.psd) { const P = psdProject(C, B); negMass = P.negMass; C = new Float64Array(P); }
  return { ...res, C, negMass, outer: t, change, rmsRateII: mII ? Math.sqrt(sII / mII) : 0, rmsRateIB: mIB ? Math.sqrt(sIB / mIB) : 0 };
}

/** Derivative-aware (transient) readout from two consecutive Euler snapshots prev → next
 *  (one step of size P.dt): uses h(prev), w(prev) and dw/dt = (next - prev)/dt, which makes
 *  the equations exact away from equilibrium.  Active = edges whose update was not clipped. */
export function readoutTransient(prev, next, P, opts = {}) {
  const { N, B } = prev;
  const Wc = new Float64Array(N * N), Wbc = new Float64Array(N * B);
  for (let k = 0; k < N * N; k++) Wc[k] = prev.Mii[k] ? Math.max(prev.Wii[k], P.wmin) : 0;
  for (let k = 0; k < N * B; k++) Wbc[k] = prev.Mib[k] ? Math.max(prev.Wib[k], P.wmin) : 0;
  const H = harmonicFrom(Wc, Wbc, N, B);
  const dWii = new Float64Array(N * N), dWib = new Float64Array(N * B);
  for (let k = 0; k < N * N; k++) dWii[k] = (next.Wii[k] - Wc[k]) / P.dt;
  for (let k = 0; k < N * B; k++) dWib[k] = (next.Wib[k] - Wbc[k]) / P.dt;
  return readoutLS(H, Wc, Wbc, prev.Mii, prev.Mib, N, B, {
    theta: P.theta, lam: P.lam, eta: P.eta, gamma: P.gamma, wmin: P.wmin, ...opts,
    dWii, dWib, activeII: next.Wii, activeIB: next.Wib,
  });
}

/** The note's naive formula C = H^+ R H^{+T} with ridge pseudo-inverse
 *  H^+ = (H^T H + alpha I)^{-1} H^T, alpha = ridge·tr(H^T H)/B.  R_ij = (theta + (lam/eta) w_ij)/gamma
 *  for i != j (all potential edges, clipped ones at wmin; w = 0 where there is no synapse);
 *  the unobserved diagonal R_ii is filled with max_j R_ij.  Returns null if gamma = 0. */
export function readoutNaive(H, Wii, Mii, N, B, { theta, lam, eta, gamma, ridge = 1e-6 }) {
  if (gamma === 0) return null;
  const R = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    let mx = -Infinity;
    for (let j = 0; j < N; j++) if (j !== i) { const v = (theta + (lam / eta) * Wii[i * N + j]) / gamma; R[i * N + j] = v; if (v > mx) mx = v; }
    R[i * N + i] = N > 1 ? mx : theta / gamma;
  }
  void Mii;
  const K = matmulAtB(H, H, N, B, B);
  let tr = 0; for (let b = 0; b < B; b++) tr += K[b * B + b];
  const alpha = ridge * tr / B;
  for (let b = 0; b < B; b++) K[b * B + b] += alpha;
  cholesky(K, B);
  const Xt = cholSolve(K, B, transpose(H, N, B), N);   // X^T = K^{-1} H^T  (B×N)
  const X = transpose(Xt, B, N);                        // N×B
  const RX = matmul(R, X, N, N, B);
  const C = matmulAtB(X, RX, N, B, B);
  for (let b = 0; b < B; b++) for (let c = b + 1; c < B; c++) { const v = 0.5 * (C[b * B + c] + C[c * B + b]); C[b * B + c] = v; C[c * B + b] = v; }
  return C;
}

/** Spectrum of the measurement operator per edge family on Sym(B) (orthonormal basis
 *  E_bb, (E_bc+E_cb)/√2).  Forms the p×p Gram (p = B(B+1)/2) explicitly: O(m p^2); fine for
 *  B ≲ 20 live, use core.measurement_spectrum offline for larger B.
 *  Returns {ii: {rank, cond, eig}, ib: {...}, both: {...}, p}; rank uses eig > rtol·max. */
export function measurementSpectrum(H, eq, N, B, rtol = 1e-10) {
  const p = B * (B + 1) / 2, mII = eq.ii.length / 2, mIB = eq.ib.length / 2, g = eq.gamma;
  const pi = [], pj = [], sc = [];
  for (let b = 0; b < B; b++) for (let c = b; c < B; c++) { pi.push(b); pj.push(c); sc.push(b === c ? 1 : Math.SQRT2); }
  const row = new Float64Array(p);
  const Gii = new Float64Array(p * p), Gib = new Float64Array(p * p);
  const acc = (G) => { for (let u = 0; u < p; u++) { const a = row[u]; if (a === 0) continue; for (let v = 0; v < p; v++) G[u * p + v] += a * row[v]; } };
  for (let e = 0; e < mII; e++) {
    const i = eq.ii[2 * e] * B, j = eq.ii[2 * e + 1] * B;
    // <a_e, basis_u> with a_e = gamma*sym(h_i h_j^T)
    for (let u = 0; u < p; u++) { const b = pi[u], c = pj[u]; row[u] = g * (b === c ? H[i + b] * H[j + b] : (H[i + b] * H[j + c] + H[i + c] * H[j + b]) / 2 * sc[u]); }
    acc(Gii);
  }
  for (let e = 0; e < mIB; e++) {
    const i = eq.ib[2 * e] * B, bb = eq.ib[2 * e + 1];
    for (let u = 0; u < p; u++) { const b = pi[u], c = pj[u]; let v = 0; if (b === c) v = b === bb ? H[i + b] : 0; else v = ((c === bb ? H[i + b] : 0) + (b === bb ? H[i + c] : 0)) / 2 * sc[u]; row[u] = v; }
    acc(Gib);
  }
  const summ = G => {
    const { values } = eigh(G, p); const mx = Math.max(values[0], 0);
    let rank = 0, mn = Infinity; for (const v of values) if (v > rtol * mx && mx > 0) { rank++; mn = Math.min(mn, v); }
    return { rank, cond: rank ? Math.sqrt(mx / mn) : Infinity, eig: values };
  };
  const Gb = new Float64Array(p * p); for (let k = 0; k < p * p; k++) Gb[k] = Gii[k] + Gib[k];
  return { ii: summ(Gii), ib: summ(Gib), both: summ(Gb), p };
}

// ─────────────────────────────── metrics (§3, §7.4) ───────────────────────────────

function frob(A, Bm, n) { let s = 0; for (let k = 0; k < n * n; k++) s += A[k] * Bm[k]; return s; }

/** Compare an estimate Chat with the true Sigma (both B×B).
 *  setting: 'known' (no projection, scale fixed to 1) | 'scale' (best scale fitted) |
 *           'contrast' (P·P with P = I - 11^T/B on both, best scale; "contrast-subspace score",
 *            also discards 1a^T + a1^T terms).
 *  Returns {setting, scale, cosine, relErr, bw, negMass, negFrac, subspace, k, eigHat, eigTrue}:
 *   cosine  = <A,S>_F / (|A| |S|)
 *   relErr  = |s A - S|_F / |S|_F
 *   bw      = BW(s A+, S)^2 / tr S, A+ = A with negative eigenvalues clipped (mass reported);
 *             for 'scale'/'contrast' s minimises it (closed form 1 - F^2/(tr A+ tr S), F = fidelity);
 *             all-zero A+ gives 1.
 *   subspace = |U_k^T V_k|_F^2 / k for the top-k eigenvectors (mean squared cos of principal angles). */
export function compareCov(Chat, Sigma, B, { setting = 'contrast', k = 3 } = {}) {
  let A = Float64Array.from(Chat.subarray(0, B * B)), S = Float64Array.from(Sigma.subarray(0, B * B));
  for (let b = 0; b < B; b++) for (let c = b + 1; c < B; c++) { const v = 0.5 * (A[b * B + c] + A[c * B + b]); A[b * B + c] = v; A[c * B + b] = v; }
  if (setting === 'contrast') { A = project1(A, B); S = project1(S, B); }
  const aa = frob(A, A, B), ss = frob(S, S, B), as = frob(A, S, B);
  const cosine = aa > 0 && ss > 0 ? as / Math.sqrt(aa * ss) : 0;
  const scale = setting === 'known' ? 1 : (aa > 0 ? as / aa : 0);
  let rel = 0; for (let q = 0; q < B * B; q++) rel += (scale * A[q] - S[q]) ** 2;
  const relErr = Math.sqrt(rel / ss);
  const Ap = psdProject(A, B);
  const trS = trace(S, B), trA = trace(Ap, B);
  const Sh = sqrtm(S, B);
  const M = matmul(matmul(Sh, Ap, B, B, B), Sh, B, B, B);
  let F = 0; for (const v of eigh(M, B).values) F += Math.sqrt(Math.max(v, 0));
  let bw;
  if (trA <= 0) bw = 1;
  else if (setting === 'known') bw = (trA + trS - 2 * F) / trS;
  else bw = Math.max(0, 1 - (F * F) / (trA * trS));
  const eA = eigh(A, B), eS = eigh(S, B);
  const kk = Math.min(k, B);
  let ov = 0;
  for (let a = 0; a < kk; a++) for (let c = 0; c < kk; c++) {
    let d = 0; for (let b = 0; b < B; b++) d += eA.vectors[b * B + a] * eS.vectors[b * B + c];
    ov += d * d;
  }
  const eigHat = eA.values.map(v => v * scale);
  return { setting, scale, cosine, relErr, bw, negMass: Ap.negMass, negFrac: Ap.negFrac, subspace: ov / kk, k: kk, eigHat, eigTrue: eS.values };
}

function trace(A, B) { let t = 0; for (let b = 0; b < B; b++) t += A[b * B + b]; return t; }

/** Baseline estimates: identity, diag(Sigma), and the rank-k truncation of Sigma.
 *  (The "untrained network" baseline is readoutState() on the initial state.) */
export function baselines(Sigma, B, k = 3) {
  const I = new Float64Array(B * B), D = new Float64Array(B * B);
  for (let b = 0; b < B; b++) { I[b * B + b] = 1; D[b * B + b] = Sigma[b * B + b]; }
  const { values, vectors } = eigh(Sigma, B);
  const T = new Float64Array(B * B);
  for (let q = 0; q < Math.min(k, B); q++) for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) T[i * B + j] += values[q] * vectors[i * B + q] * vectors[j * B + q];
  return { identity: I, diag: D, rankK: T };
}
