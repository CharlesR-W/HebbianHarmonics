// Chapter "readout": the estimator (no DOM), so it can be checked in node against hebb/sweeps2/readout.py.
//
// Linear, zero potential. At the fixed point every live edge gives D_ij^2 = 2(b - alpha w_ij), and
// D_ij^2 = d^T C_hat d with d = h_i - h_j, H = (diag(d) - W)^{-1} A computed from the weights. H 1 = 1, so d is a
// contrast and only P C P is visible. We work in an orthonormal contrast basis Q (B x (B-1), DCT-II columns
// 1..B-1, as hebb.onsite_analysis.contrast_basis for B = 12), with unknown Y = Q^T C Q and the measurement
// operator Y -> (c_e^T Y c_e), c_e = Q^T d_e, written on the orthonormal svec basis (off-diagonals scaled by
// sqrt 2). Recovery as the reference: masked ridge with rho = 1e-10 sv_max^2 (min-norm on the identifiable
// subspace), an optional squared hinge for dead pairs (d^T Y d >= 2b), PSD projection. The singular values come
// from a Householder QR followed by a one-sided Jacobi SVD, so ranks at a relative tolerance of 1e-8 are
// meaningful (the normal equations would square the condition number).
import * as O from "./onsite_engine.js";

const SQ2 = Math.SQRT2;

// ---------------------------------------------------------------- small dense helpers
export function contrastBasis(B) {
  const n = B - 1, Q = new Float64Array(B * n);
  for (let k = 1; k < B; k++) {
    let nr = 0;
    for (let i = 0; i < B; i++) { const v = Math.cos(Math.PI * (i + 0.5) * k / B); Q[i * n + k - 1] = v; nr += v * v; }
    nr = Math.sqrt(nr); for (let i = 0; i < B; i++) Q[i * n + k - 1] /= nr;
  }
  return Q;
}
export function svecIndex(n) {
  const ia = [], ib = [];
  for (let a = 0; a < n; a++) for (let b = a; b < n; b++) { ia.push(a); ib.push(b); }
  return { ia: Int32Array.from(ia), ib: Int32Array.from(ib), p: ia.length, n };
}
export function svec(Y, idx) {
  const { ia, ib, p, n } = idx, s = new Float64Array(p);
  for (let k = 0; k < p; k++) s[k] = Y[ia[k] * n + ib[k]] * (ia[k] === ib[k] ? 1 : SQ2);
  return s;
}
export function smat(s, idx) {
  const { ia, ib, p, n } = idx, Y = new Float64Array(n * n);
  for (let k = 0; k < p; k++) { const v = ia[k] === ib[k] ? s[k] : s[k] / SQ2; Y[ia[k] * n + ib[k]] = v; Y[ib[k] * n + ia[k]] = v; }
  return Y;
}
// Q^T C Q (n x n) for C (B x B), and Q Y Q^T (B x B) back
export function toContrast(C, Q, B, n) {
  const T = new Float64Array(B * n), Y = new Float64Array(n * n);
  for (let i = 0; i < B; i++) for (let k = 0; k < n; k++) { let v = 0; for (let j = 0; j < B; j++) v += C[i * B + j] * Q[j * n + k]; T[i * n + k] = v; }
  for (let a = 0; a < n; a++) for (let k = 0; k < n; k++) { let v = 0; for (let i = 0; i < B; i++) v += Q[i * n + a] * T[i * n + k]; Y[a * n + k] = v; }
  return Y;
}
export function fromContrast(Y, Q, B, n) {
  const T = new Float64Array(B * n), C = new Float64Array(B * B);
  for (let i = 0; i < B; i++) for (let k = 0; k < n; k++) { let v = 0; for (let a = 0; a < n; a++) v += Q[i * n + a] * Y[a * n + k]; T[i * n + k] = v; }
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) { let v = 0; for (let k = 0; k < n; k++) v += T[i * n + k] * Q[j * n + k]; C[i * B + j] = v; }
  return C;
}
export function psd(Y, n) {
  const S = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) S[i * n + j] = 0.5 * (Y[i * n + j] + Y[j * n + i]);
  return O.symFn(S, n, x => Math.max(x, 0));
}
export function eigvalsDesc(Y, n) { return Array.from(O.eigh(Float64Array.from(Y), n).values); }
const frob = A => { let s = 0; for (const v of A) s += v * v; return Math.sqrt(s); };
export function froCos(A, Bm) { let s = 0; for (let k = 0; k < A.length; k++) s += A[k] * Bm[k]; return s / (frob(A) * frob(Bm) + 1e-300); }
export function relErr(A, T) { let s = 0; for (let k = 0; k < A.length; k++) s += (A[k] - T[k]) ** 2; return Math.sqrt(s) / (frob(T) + 1e-300); }
const trace = (A, n) => { let t = 0; for (let i = 0; i < n; i++) t += A[i * n + i]; return t; };
const matmul = (A, Bm, n) => { const C = new Float64Array(n * n); for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) { const a = A[i * n + k]; if (a === 0) continue; for (let j = 0; j < n; j++) C[i * n + j] += a * Bm[k * n + j]; } return C; };
/** Bures–Wasserstein distance between PSD matrices (as hebb.onsite_analysis.bures). */
export function bures(A, Bm, n) {
  const rA = O.symFn(A, n, x => Math.sqrt(Math.max(x, 0)));
  const M = matmul(matmul(rA, Bm, n), rA, n);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = 0.5 * (M[i * n + j] + M[j * n + i]); M[i * n + j] = v; M[j * n + i] = v; }
  const cross = trace(O.symFn(M, n, x => Math.sqrt(Math.max(x, 0))), n);
  return Math.sqrt(Math.max(trace(A, n) + trace(Bm, n) - 2 * cross, 0));
}
/** {cos, rel_err, bures_rel} of a (PSD) recovery Y against the target T, both n x n. */
export function score(Y, T, n) { return { cos: froCos(Y, T), rel: relErr(Y, T), bures: bures(Y, T, n) / Math.sqrt(Math.max(trace(T, n), 1e-300)) }; }

function cholSolve(Mx, n, rhs) {
  const L = Float64Array.from(Mx);
  if (!O.cholesky(L, n)) {
    let t = 0; for (let i = 0; i < n; i++) t += Math.abs(Mx[i * n + i]);
    for (let i = 0; i < n; i++) L[i * n + i] += 1e-12 * (t / n || 1);
    for (let tries = 0; tries < 6; tries++) {
      const L2 = Float64Array.from(L); if (O.cholesky(L2, n)) return O.cholSolveVec(L2, n, Float64Array.from(rhs));
      for (let i = 0; i < n; i++) L[i * n + i] += 1e-8 * Math.pow(100, tries) * (t / n || 1);
    }
    return new Float64Array(n);
  }
  return O.cholSolveVec(L, n, Float64Array.from(rhs));
}

// ---------------------------------------------------------------- SVD of the measurement operator
/** Columns-major (p columns of length m) matrix Mc and right-hand side y (m). Returns {sv (desc), V (p x p,
 *  column k = right singular vector k, row-major V[i*p+k]), uty (U^T y, aligned with sv), rank}. */
export function svdOperator(Mc, m, p, y) {
  let r, Ac, z;
  if (m > p) {
    // Householder QR, column-major: reduce to R (p x p) and z = (Q^T y)[0:p]
    const A = Float64Array.from(Mc), yy = Float64Array.from(y), v = new Float64Array(m);
    for (let k = 0; k < p; k++) {
      const ck = k * m;
      let nr = 0; for (let i = k; i < m; i++) nr += A[ck + i] * A[ck + i];
      nr = Math.sqrt(nr);
      if (nr === 0) continue;
      const alpha = A[ck + k] > 0 ? -nr : nr;
      for (let i = k; i < m; i++) v[i] = A[ck + i];
      v[k] -= alpha;
      let vv = 0; for (let i = k; i < m; i++) vv += v[i] * v[i];
      if (vv === 0) continue;
      for (let j = k; j < p; j++) {
        const cj = j * m; let s = 0;
        for (let i = k; i < m; i++) s += v[i] * A[cj + i];
        s = 2 * s / vv;
        for (let i = k; i < m; i++) A[cj + i] -= s * v[i];
      }
      let s = 0; for (let i = k; i < m; i++) s += v[i] * yy[i];
      s = 2 * s / vv; for (let i = k; i < m; i++) yy[i] -= s * v[i];
    }
    r = p; Ac = new Float64Array(p * p); z = yy.slice(0, p);
    for (let j = 0; j < p; j++) for (let i = 0; i <= j; i++) Ac[j * p + i] = A[j * m + i];
  } else { r = m; Ac = Float64Array.from(Mc); z = Float64Array.from(y); }
  // one-sided Jacobi (Hestenes) on the p columns of Ac (length r)
  const V = new Float64Array(p * p); for (let i = 0; i < p; i++) V[i * p + i] = 1;
  const nrm = new Float64Array(p);
  const colNorm = j => { let s = 0; const c = j * r; for (let i = 0; i < r; i++) s += Ac[c + i] * Ac[c + i]; return s; };
  for (let j = 0; j < p; j++) nrm[j] = colNorm(j);
  let maxN = 0; for (let j = 0; j < p; j++) maxN = Math.max(maxN, nrm[j]);
  const tiny = 1e-300 + maxN * 1e-34;
  for (let sweep = 0; sweep < 80; sweep++) {
    let rot = 0;
    for (let i = 0; i < p - 1; i++) for (let j = i + 1; j < p; j++) {
      const a = nrm[i], b = nrm[j];
      if (a <= tiny || b <= tiny) continue;
      const ci = i * r, cj = j * r;
      let g = 0; for (let k = 0; k < r; k++) g += Ac[ci + k] * Ac[cj + k];
      if (Math.abs(g) <= 1e-15 * Math.sqrt(a * b)) continue;
      rot++;
      const zeta = (b - a) / (2 * g);
      const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
      const c = 1 / Math.sqrt(1 + t * t), s = c * t;
      for (let k = 0; k < r; k++) { const x = Ac[ci + k], w = Ac[cj + k]; Ac[ci + k] = c * x - s * w; Ac[cj + k] = s * x + c * w; }
      for (let k = 0; k < p; k++) { const x = V[k * p + i], w = V[k * p + j]; V[k * p + i] = c * x - s * w; V[k * p + j] = s * x + c * w; }
      nrm[i] = colNorm(i); nrm[j] = colNorm(j);
    }
    if (!rot) break;
  }
  const sv0 = Array.from(nrm, x => Math.sqrt(x));
  const order = sv0.map((_, k) => k).sort((a, b) => sv0[b] - sv0[a]);
  const sv = order.map(k => sv0[k]);
  const Vs = new Float64Array(p * p), aty = new Float64Array(p);
  order.forEach((k, q) => {
    for (let i = 0; i < p; i++) Vs[i * p + q] = V[i * p + k];
    let s = 0; for (let i = 0; i < r; i++) s += Ac[k * r + i] * z[i];
    aty[q] = s;                                     // = sigma_q * (u_q^T z)
  });
  return { sv, V: Vs, aty };
}

// ---------------------------------------------------------------- the readout
/** One-shot readout from a weight matrix.
 *  opts: {W, A, N, B, b, alpha, Chat, Sigma, Q?, idx?, noise = 0, noiseSeed = 8, hinge = true, eps = 1e-9}
 *  Returns everything the page shows; matrices in contrast coordinates are n x n (n = B - 1). */
export function readout(o) {
  const { N, B, b, alpha } = o, eps = o.eps ?? 1e-9, n = B - 1;
  const Q = o.Q || contrastBasis(B), idx = o.idx || svecIndex(n), p = idx.p;
  const T1 = toContrast(o.Chat, Q, B, n), T2 = toContrast(o.Sigma, Q, B, n), s1 = svec(T1, idx);
  const W0 = o.W;
  // live edges, from the clean weights (multiplicative noise keeps zeros at zero)
  const live = [], dead = [];
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) (W0[i * N + j] > eps ? live : dead).push(i, j);
  const m = live.length / 2, md = dead.length / 2;
  // noisy weights (symmetric multiplicative noise, fixed draw), as the reference
  let W = W0;
  if (o.noise > 0) {
    const rng = O.makeRng(o.noiseSeed ?? 8);
    W = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const v = W0[i * N + j] * (1 + o.noise * rng.normal()); W[i * N + j] = v; W[j * N + i] = v; }
  }
  const H0 = O.harmonic(O.makeNetFrom({ A: o.A, N, B, W: W0 }));
  const H = o.noise > 0 ? O.harmonic(O.makeNetFrom({ A: o.A, N, B, W })) : H0;
  let onesLeak = 0;
  const rowsOf = (Hm, list, cnt) => {
    const Mc = new Float64Array(p * cnt), c = new Float64Array(n), q = new Float64Array(cnt), dsq = new Float64Array(cnt * B);
    for (let e = 0; e < cnt; e++) {
      const i = list[2 * e], j = list[2 * e + 1];
      let s1d = 0;
      for (let a = 0; a < B; a++) { const d = Hm[i * B + a] - Hm[j * B + a]; dsq[e * B + a] = d * d; s1d += d; }
      if (Math.abs(s1d) > onesLeak) onesLeak = Math.abs(s1d);
      let qq = 0;
      for (let k = 0; k < n; k++) { let v = 0; for (let a = 0; a < B; a++) v += (Hm[i * B + a] - Hm[j * B + a]) * Q[a * n + k]; c[k] = v; qq += v * v; }
      q[e] = qq;
      for (let k = 0; k < p; k++) Mc[k * cnt + e] = c[idx.ia[k]] * c[idx.ib[k]] * (idx.ia[k] === idx.ib[k] ? 1 : SQ2);
    }
    return { Mc, q, dsq };
  };
  const yOf = Wm => { const y = new Float64Array(m); for (let e = 0; e < m; e++) y[e] = 2 * (b - alpha * Wm[live[2 * e] * N + live[2 * e + 1]]); return y; };
  const y = yOf(W), y0 = o.noise > 0 ? yOf(W0) : y;
  const R = rowsOf(H, live, m);
  const R0 = o.noise > 0 ? rowsOf(H0, live, m) : R;
  // fixed-point error (clean weights): max |2(b - alpha w) - d^T C_hat d| over live edges
  let fp = 0;
  for (let e = 0; e < m; e++) { let v = 0; for (let k = 0; k < p; k++) v += R0.Mc[k * m + e] * s1[k]; fp = Math.max(fp, Math.abs(y0[e] - v)); }
  // noise gain: relative change of the measured D^2 per unit relative weight noise, 2 alpha w / D^2
  const gains = []; for (let e = 0; e < m; e++) gains.push(2 * alpha * W0[live[2 * e] * N + live[2 * e + 1]] / Math.max(y0[e], 1e-300));
  gains.sort((a, c) => a - c);
  // SVD and ridge
  let sv = [], V = null, s = new Float64Array(p), rho = 0, rank8 = 0, rank4 = 0;
  if (m > 0) {
    const d = svdOperator(R.Mc, m, p, y); sv = d.sv; V = d.V;
    rho = 1e-10 * Math.max(sv[0] * sv[0], 1e-300);
    for (let k = 0; k < p; k++) { const c = d.aty[k] / (sv[k] * sv[k] + rho); if (!c) continue; for (let i = 0; i < p; i++) s[i] += V[i * p + k] * c; }
    rank8 = sv.filter(x => x > 1e-8 * sv[0]).length; rank4 = sv.filter(x => x > 1e-4 * sv[0]).length;
  }
  const Yridge = smat(s, idx);
  // identifiable-subspace error (singular values > 1e-4 of the largest) and the target's share in it
  let errInv = NaN, identFrac = 0;
  if (m > 0) {
    let num = 0, den = 0, tot = 0;
    for (let k = 0; k < rank4; k++) { let pt = 0, ph = 0; for (let i = 0; i < p; i++) { pt += V[i * p + k] * s1[i]; ph += V[i * p + k] * s[i]; } num += (ph - pt) ** 2; den += pt * pt; }
    for (const v of s1) tot += v * v;
    errInv = Math.sqrt(num) / Math.max(Math.sqrt(den), 1e-300); identFrac = den / tot;
  }
  // dead-pair hinge: min ||M x - y||^2 + rho ||x||^2 + sum_dead (2b - r.x)_+^2, semismooth Newton from the ridge
  let Yhinge = Yridge, hingeIters = 0;
  if (o.hinge !== false && md > 0) {
    const D = rowsOf(H, dead, md).Mc;          // column-major p x md
    const G0 = new Float64Array(p * p), h0 = new Float64Array(p);
    for (let a = 0; a < p; a++) for (let c = a; c < p; c++) { let v = 0; const ca = a * m, cc = c * m; for (let e = 0; e < m; e++) v += R.Mc[ca + e] * R.Mc[cc + e]; G0[a * p + c] = v; G0[c * p + a] = v; }
    for (let a = 0; a < p; a++) { let v = 0; for (let e = 0; e < m; e++) v += R.Mc[a * m + e] * y[e]; h0[a] = v; }
    let yy = 0; for (const v of y) yy += v * v;
    const two = 2 * b, Dx = new Float64Array(md);
    const evalF = x => {
      let f = yy, g = 0;
      for (let a = 0; a < p; a++) { let v = 0; for (let c = 0; c < p; c++) v += G0[a * p + c] * x[c]; f += x[a] * v - 2 * h0[a] * x[a] + rho * x[a] * x[a]; }
      for (let e = 0; e < md; e++) { let v = 0; for (let k = 0; k < p; k++) v += D[k * md + e] * x[k]; Dx[e] = v; if (v < two) g += (two - v) ** 2; }
      return f + g;
    };
    let x = Float64Array.from(s), f = evalF(x);
    const Dd = new Float64Array(md), Gd = new Float64Array(p);
    for (let it = 0; it < 200; it++) {
      hingeIters = it + 1;
      const act = []; for (let e = 0; e < md; e++) if (Dx[e] < two) act.push(e);
      const g = new Float64Array(p), Hs = Float64Array.from(G0);
      for (let a = 0; a < p; a++) { let v = 0; for (let c = 0; c < p; c++) v += G0[a * p + c] * x[c]; g[a] = v - h0[a] + rho * x[a]; Hs[a * p + a] += rho; }
      for (const e of act) { const hv = two - Dx[e]; for (let a = 0; a < p; a++) g[a] -= D[a * md + e] * hv; }
      for (let a = 0; a < p; a++) for (let c = a; c < p; c++) { let v = 0; for (const e of act) v += D[a * md + e] * D[c * md + e]; Hs[a * p + c] += v; if (c !== a) Hs[c * p + a] += v; }
      const dir = cholSolve(Hs, p, g); for (let a = 0; a < p; a++) dir[a] = -dir[a];
      let slope = 0; for (let a = 0; a < p; a++) slope += g[a] * dir[a];
      if (!(slope < 0)) break;
      // exact line search: f(x + t dir) is a convex piecewise quadratic in t; bisect its derivative
      let qa = 0; for (let a = 0; a < p; a++) { let v = 0; for (let c = 0; c < p; c++) v += G0[a * p + c] * dir[c]; Gd[a] = v; qa += dir[a] * v + rho * dir[a] * dir[a]; }
      for (let e = 0; e < md; e++) { let v = 0; for (let k = 0; k < p; k++) v += D[k * md + e] * dir[k]; Dd[e] = v; }
      const dfdt = t => {                         // half-derivative of f(x + t dir)
        let v = slope + qa * t;
        for (let e = 0; e < md; e++) { const hv = two - Dx[e] - t * Dd[e], h0v = two - Dx[e]; v += (h0v > 0 ? h0v * Dd[e] : 0) - (hv > 0 ? hv * Dd[e] : 0); }
        return v;
      };
      let hi = 1; while (dfdt(hi) < 0 && hi < 1e6) hi *= 2;
      let lo = 0;
      for (let k = 0; k < 60; k++) { const mid = 0.5 * (lo + hi); if (dfdt(mid) < 0) lo = mid; else hi = mid; }
      const t = 0.5 * (lo + hi);
      const xn = x.map((v, a) => v + t * dir[a]), fn = evalF(xn);
      if (!(fn <= f)) { evalF(x); break; }
      const dec = f - fn;
      x = xn; f = fn;
      if (dec <= 1e-15 * Math.max(f, 1e-300)) break;
    }
    Yhinge = smat(x, idx);
  }
  // baselines fitted to the same live measurements: c I (least squares), nonneg diagonal (NNLS)
  let Yid = new Float64Array(n * n), Ydg = new Float64Array(n * n), cId = 0, delta = new Float64Array(B);
  if (m > 0) {
    let a = 0, c = 0; for (let e = 0; e < m; e++) { a += y[e] * R.q[e]; c += R.q[e] * R.q[e]; }
    cId = a / c; for (let k = 0; k < n; k++) Yid[k * n + k] = cId;
    delta = nnls(R.dsq, m, B, y);
    const Cd = new Float64Array(B * B); for (let k = 0; k < B; k++) Cd[k * B + k] = delta[k];
    Ydg = toContrast(Cd, Q, B, n);
  }
  const Yr = psd(Yridge, n), Yh = psd(Yhinge, n), Yi = psd(Yid, n), Yd = psd(Ydg, n);
  return {
    n, p, m, md, nPairs: m + md, onesLeak, rank8, rank4, sv, rho, s, hingeIters,
    T1, T2, Yridge, Yhinge, Yr, Yh, Yi, Yd, cId, delta,
    errFixedPoint: fp, errSampling: relErr(T1, T2), errInverse: errInv, identFrac,
    gainMedian: gains.length ? gains[gains.length >> 1] : NaN,
    sc: {
      ridge: { Chat: score(Yr, T1, n), Sigma: score(Yr, T2, n) },
      hinge: { Chat: score(Yh, T1, n), Sigma: score(Yh, T2, n) },
      identity: { Chat: score(Yi, T1, n), Sigma: score(Yi, T2, n) },
      diag: { Chat: score(Yd, T1, n), Sigma: score(Yd, T2, n) },
    },
  };
}

/** Lawson–Hanson NNLS: min ||E x - y||, x >= 0; E row-major m x k (k small). */
export function nnls(E, m, k, y) {
  const EtE = new Float64Array(k * k), Ety = new Float64Array(k);
  for (let e = 0; e < m; e++) for (let a = 0; a < k; a++) { const v = E[e * k + a]; if (!v) continue; Ety[a] += v * y[e]; for (let c = 0; c < k; c++) EtE[a * k + c] += v * E[e * k + c]; }
  let sc = 0; for (let a = 0; a < k; a++) sc = Math.max(sc, EtE[a * k + a]);
  const tol = 1e-12 * Math.max(sc, 1e-300);
  const x = new Float64Array(k), P = new Array(k).fill(false);
  const grad = () => { const w = new Float64Array(k); for (let a = 0; a < k; a++) { let v = Ety[a]; for (let c = 0; c < k; c++) v -= EtE[a * k + c] * x[c]; w[a] = v; } return w; };
  const lsOn = set => {
    const id = []; for (let a = 0; a < k; a++) if (set[a]) id.push(a);
    const q = id.length, Mx = new Float64Array(q * q), r = new Float64Array(q);
    id.forEach((a, i) => { r[i] = Ety[a]; id.forEach((c, j) => { Mx[i * q + j] = EtE[a * k + c]; }); Mx[i * q + i] += 1e-14 * sc; });
    const z = cholSolve(Mx, q, r), out = new Float64Array(k); id.forEach((a, i) => { out[a] = z[i]; }); return out;
  };
  for (let outer = 0; outer < 3 * k; outer++) {
    const w = grad();
    let best = -1, bw = tol;
    for (let a = 0; a < k; a++) if (!P[a] && w[a] > bw) { bw = w[a]; best = a; }
    if (best < 0) break;
    P[best] = true;
    for (let inner = 0; inner < 3 * k; inner++) {
      const z = lsOn(P);
      let ok = true; for (let a = 0; a < k; a++) if (P[a] && z[a] <= 0) ok = false;
      if (ok) { x.set(z); break; }
      let al = Infinity; for (let a = 0; a < k; a++) if (P[a] && z[a] <= 0) al = Math.min(al, x[a] / (x[a] - z[a]));
      for (let a = 0; a < k; a++) x[a] += al * (z[a] - x[a]);
      for (let a = 0; a < k; a++) if (P[a] && x[a] <= 1e-15) { P[a] = false; x[a] = 0; }
    }
  }
  return x;
}
