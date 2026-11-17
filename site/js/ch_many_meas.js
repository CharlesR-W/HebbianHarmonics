// Chapter "many": a faster measurement-operator spectrum for the weight-space readout (pure module, no DOM).
//
// engine.measurementSpectrum builds the Gram matrix G = AᵀA of the readout's measurement operator on Sym(B) edge by
// edge, O(m p²) with p = B(B+1)/2, and diagonalises it with vectors (O(9p³)); that is fine for B ≲ 20 but takes
// seconds to minutes at B = 24..48. This module computes the SAME Gram matrix (same orthonormal basis of Sym(B):
// E_bb and (E_bc + E_cb)/√2) by grouping the equations by interior neuron:
//   every equation of neuron i reads  h_iᵀ C y  with y = h_j (I–I, coefficient γ) or y = e_b (I–B),
//   so its row is L_i y, where L_i : R^B → Sym(B), y ↦ sym(h_i yᵀ), and the rows of neuron i contribute
//   L_i M_i L_iᵀ with M_i = γ² Σ_j h_j h_jᵀ (I–I partners, each edge counted once) or Σ_b e_b e_bᵀ (I–B).
// L_i has two non-zeros per row, so one neuron costs O(p B + p²) instead of O(m_i p²). Eigenvalues come from a
// values-only Householder tridiagonalisation + implicit QL (Numerical Recipes tred2/tqli without vectors), about
// 7x cheaper than a full eigh. Checked against engine.measurementSpectrum (same Gram to 1e-12, same spectrum).

export function pairIndex(B) {
  const idx = new Int32Array(B * B), pb = [], pc = [];
  for (let b = 0, u = 0; b < B; b++) for (let c = b; c < B; c++, u++) { idx[b * B + c] = u; idx[c * B + b] = u; pb.push(b); pc.push(c); }
  return { idx, pb: Int32Array.from(pb), pc: Int32Array.from(pc), p: pb.length };
}

// Accumulate G += L_x M L_xᵀ for one neuron (x = h_i, M symmetric B×B). T = L_x M is p×B.
// Row u of L_x is the pair (b, c): for b = c one entry x_b at column b; for b < c, x_c/√2 at column b and x_b/√2
// at column c. av/dv hold those two coefficients (dv = 0 on the diagonal pairs).
function addNeuron(G, x, M, B, pi, T, av, dv) {
  const { pb, pc, p } = pi, r2 = Math.SQRT1_2;
  for (let u = 0; u < p; u++) { const b = pb[u], c = pc[u]; if (b === c) { av[u] = x[b]; dv[u] = 0; } else { av[u] = x[c] * r2; dv[u] = x[b] * r2; } }
  for (let u = 0; u < p; u++) {
    const mb = pb[u] * B, mc = pc[u] * B, a = av[u], d = dv[u], tu = u * B;
    for (let k = 0; k < B; k++) T[tu + k] = a * M[mb + k] + d * M[mc + k];
  }
  for (let u = 0; u < p; u++) {
    const tu = u * B, gu = u * p;
    for (let v = u; v < p; v++) G[gu + v] += T[tu + pb[v]] * av[v] + T[tu + pc[v]] * dv[v];
  }
}

// Gram matrices of the measurement families (upper triangles filled, then symmetrised). families: any of
// "ii", "ib" (each family alone) and "both" (one combined Gram, cheaper than summing the two).
// eq is engine.readoutEquations(...) output: {ii: [i0,j0,...], ib: [i0,b0,...], gamma}.
export function measurementGram(H, eq, N, B, { families = ["ii", "ib", "both"] } = {}) {
  const pi = pairIndex(B), p = pi.p, g2 = eq.gamma * eq.gamma;
  const T = new Float64Array(p * B), M = new Float64Array(B * B), x = new Float64Array(B), av = new Float64Array(p), dv = new Float64Array(p);
  const partII = Array.from({ length: N }, () => []), partIB = Array.from({ length: N }, () => []);
  for (let e = 0; e < eq.ii.length / 2; e++) partII[eq.ii[2 * e]].push(eq.ii[2 * e + 1]);   // each I–I edge once (i < j)
  for (let e = 0; e < eq.ib.length / 2; e++) partIB[eq.ib[2 * e]].push(eq.ib[2 * e + 1]);
  const build = (useII, useIB) => {
    const G = new Float64Array(p * p);
    for (let i = 0; i < N; i++) {
      const js = useII ? partII[i] : [], bs = useIB ? partIB[i] : [];
      if (!js.length && !bs.length) continue;
      M.fill(0);
      for (const j of js) for (let a = 0; a < B; a++) { const ha = H[j * B + a] * g2; if (ha === 0) continue; for (let c = 0; c < B; c++) M[a * B + c] += ha * H[j * B + c]; }
      for (const b of bs) M[b * B + b] += 1;
      for (let a = 0; a < B; a++) x[a] = H[i * B + a];
      addNeuron(G, x, M, B, pi, T, av, dv);
    }
    symm(G, p); return G;
  };
  const out = { p };
  if (families.includes("ii")) out.ii = build(true, false);
  if (families.includes("ib")) out.ib = build(false, true);
  if (families.includes("both")) out.both = build(true, true);
  return out;
}
function symm(G, p) { for (let u = 0; u < p; u++) for (let v = u + 1; v < p; v++) G[v * p + u] = G[u * p + v]; }

// Eigenvalues (descending) of a symmetric n×n matrix; A is overwritten. Householder tridiagonalisation without
// accumulating vectors, then implicit QL with Wilkinson shifts (tred2 / tqli, Numerical Recipes §11.2–11.3).
export function eigvalsSym(A, n) {
  const d = new Float64Array(n), e = new Float64Array(n);
  for (let i = n - 1; i > 0; i--) {
    const l = i - 1, ri = i * n; let h = 0;
    if (l > 0) {
      let scale = 0;
      for (let k = 0; k <= l; k++) scale += Math.abs(A[ri + k]);
      if (scale === 0) e[i] = A[ri + l];
      else {
        for (let k = 0; k <= l; k++) { A[ri + k] /= scale; h += A[ri + k] * A[ri + k]; }
        let f = A[ri + l];
        let g = f >= 0 ? -Math.sqrt(h) : Math.sqrt(h);
        e[i] = scale * g; h -= f * g; A[ri + l] = f - g; f = 0;
        for (let j = 0; j <= l; j++) {
          let gg = 0; const rj = j * n;
          for (let k = 0; k <= j; k++) gg += A[rj + k] * A[ri + k];
          for (let k = j + 1; k <= l; k++) gg += A[k * n + j] * A[ri + k];
          e[j] = gg / h; f += e[j] * A[ri + j];
        }
        const hh = f / (h + h);
        for (let j = 0; j <= l; j++) {
          const fj = A[ri + j], gj = e[j] - hh * fj; e[j] = gj; const rj = j * n;
          for (let k = 0; k <= j; k++) A[rj + k] -= fj * e[k] + gj * A[ri + k];
        }
      }
    } else e[i] = A[ri + l];
    d[i] = h;
  }
  for (let i = 0; i < n; i++) d[i] = A[i * n + i];
  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0;
  for (let l = 0; l < n; l++) {
    let iter = 0, m;
    for (;;) {
      for (m = l; m < n - 1; m++) { const dd = Math.abs(d[m]) + Math.abs(d[m + 1]); if (Math.abs(e[m]) <= 1e-15 * dd) break; }
      if (m === l || iter++ > 60) break;
      let g = (d[l + 1] - d[l]) / (2 * e[l]), r = Math.hypot(g, 1);
      g = d[m] - d[l] + e[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
      let s = 1, c = 1, p = 0, i, early = false;
      for (i = m - 1; i >= l; i--) {
        const f = s * e[i], b = c * e[i];
        e[i + 1] = r = Math.hypot(f, g);
        if (r === 0) { d[i + 1] -= p; e[m] = 0; early = true; break; }
        s = f / r; c = g / r; g = d[i + 1] - p; r = (d[i] - g) * s + 2 * c * b; p = s * r; d[i + 1] = g + p; g = c * r - b;
      }
      if (early) continue;
      d[l] -= p; e[l] = g; e[m] = 0;
    }
  }
  return Float64Array.from(d).sort((a, b) => b - a);
}

// Summary used on the page: eigenvalues normalised by the largest, and the count above `rtol`.
export function summarise(eig, rtol) {
  const mx = Math.max(eig[0] || 0, 0);
  const norm = Float64Array.from(eig, v => (mx > 0 ? Math.max(v, 0) / mx : 0));
  let rank = 0; for (const v of norm) if (v > rtol) rank++;
  return { rank, norm };
}
