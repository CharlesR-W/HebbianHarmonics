// Model side of the Perron-parts teaching figures (ch_perron_teach.js): presets for Σ, exact enumeration of the
// parts of A = Σ − θ11ᵀ (a port of hebb/big/common.py `enumerate_parts`), and average-linkage dendrograms on the
// correlation. Pure (no DOM), so it runs under node for the parity check against Python.
import { eigh, makeRng } from "./engine.js";

// ------------------------------------------------------------------ presets (B = 8, unit variances)
// Σ is a flat row-major Float64Array (B*B).
function hierarchy(B = 8, levels = [0.75, 0.45, 0.15]) {
  const S = new Float64Array(B * B);
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) {
    if (i === j) { S[i * B + j] = 1; continue; }
    // level of the lowest common ancestor in a binary tree over 0..B-1 (1 = siblings)
    let l = 0, a = i, b = j;
    while (a !== b) { a >>= 1; b >>= 1; l++; }
    S[i * B + j] = levels[l - 1];
  }
  return S;
}

// A fixed seeded symmetric perturbation of amplitude `amp` (uniform in [−amp, amp] off the diagonal), then the
// result projected to PSD (eigenvalues floored at 0.02) and rescaled to unit diagonal.
function perturbed(S0, B, amp = 0.06, seed = 7) {
  const rng = makeRng(seed), S = S0.slice();
  for (let i = 0; i < B; i++) for (let j = i + 1; j < B; j++) {
    const d = amp * (2 * rng.uniform() - 1);
    S[i * B + j] += d; S[j * B + i] += d;
  }
  const { values, vectors } = eigh(S, B);
  if (values[B - 1] < 0.02) {
    S.fill(0);
    for (let k = 0; k < B; k++) {
      const l = Math.max(values[k], 0.02);
      for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) S[i * B + j] += l * vectors[i * B + k] * vectors[j * B + k];
    }
  }
  const d = Array.from({ length: B }, (_, i) => Math.sqrt(S[i * B + i]));
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) S[i * B + j] = i === j ? 1 : S[i * B + j] / (d[i] * d[j]);
  return S;
}

function blocks(B = 8, k = 4, within = 0.5, across = 0.1) {
  const S = new Float64Array(B * B);
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++)
    S[i * B + j] = i === j ? 1 : (Math.floor(i / k) === Math.floor(j / k) ? within : across);
  return S;
}

function ring(B = 8, ell = 1.2) {
  const S = new Float64Array(B * B);
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) {
    const d = Math.min(Math.abs(i - j), B - Math.abs(i - j));
    S[i * B + j] = Math.exp(-(d * d) / (2 * ell * ell));
  }
  return S;
}

export const PRESETS = [
  { id: "hier", label: "hierarchy", note: "Binary tree of depth 3: correlation 0.75 between siblings, 0.45 within a group of four, 0.15 across the two halves." },
  { id: "hiern", label: "hierarchy + noise", note: "The hierarchy plus a fixed symmetric perturbation (amplitude 0.06, seed 7), rescaled to unit variances." },
  { id: "blocks", label: "two blocks", note: "Two blocks of four: correlation 0.5 within a block, 0.1 across." },
  { id: "ring", label: "ring", note: "Eight inputs on a ring: $\\Sigma_{bc} = \\exp(-d_{bc}^2/(2\\cdot 1.2^2))$, $d$ the distance around the ring." },
];
export function presetSigma(id) {
  const B = 8;
  if (id === "hiern") return { B, Sigma: perturbed(hierarchy(B), B) };
  if (id === "blocks") return { B, Sigma: blocks(B) };
  if (id === "ring") return { B, Sigma: ring(B) };
  return { B, Sigma: hierarchy(B) };
}

// Three inputs: g1, g2 correlated (ρ12), g3 independent, unit variances.
export function sigma3(r12) { return new Float64Array([1, r12, 0, r12, 1, 0, 0, 0, 1]); }

// ------------------------------------------------------------------ parts (port of enumerate_parts)
// Returns every part of A = Σ − θ11ᵀ: {S (sorted input indices), h (Float64Array(B), sums to 1, zero off S),
// c, margin, gap, stable}. Same algorithm, tolerances and output order as the Python. With `candidates: true`
// it also returns the subsets' positive top eigenvectors that fail (c ≤ 0 or an off-support violation), flagged
// {part: false, why}, for drawing faint.
export function enumerateParts(Sigma, B, theta, { tol = 1e-9, candidates = false } = {}) {
  const out = [];
  const idx = new Int32Array(B);
  for (let r = 1; r <= B; r++) {
    // combinations of r out of B in lexicographic order (itertools.combinations order)
    for (let i = 0; i < r; i++) idx[i] = i;
    for (;;) {
      const S = Array.from(idx.subarray(0, r));
      const M = new Float64Array(r * r);
      for (let a = 0; a < r; a++) for (let b = 0; b < r; b++) M[a * r + b] = Sigma[S[a] * B + S[b]] - theta;
      const { values, vectors } = eigh(M, r);           // descending
      // numpy's eigh is ascending: walk k = 0..r-1 in ascending order, so the output order matches
      for (let ka = 0; ka < r; ka++) {
        const k = r - 1 - ka;                            // index into the descending arrays
        let sum = 0;
        for (let a = 0; a < r; a++) sum += vectors[a * r + k];
        const sg = sum >= 0 ? 1 : -1;
        let vmin = Infinity;
        for (let a = 0; a < r; a++) vmin = Math.min(vmin, sg * vectors[a * r + k]);
        const ev = values[k], isTop = k === 0;
        if (vmin <= tol || ev <= tol) {
          if (candidates && isTop && vmin > tol) out.push(mk(S, vectors, k, sg, sum, r, ev, NaN, NaN, false, "c ≤ 0"));
          continue;
        }
        const h = new Float64Array(B);
        for (let a = 0; a < r; a++) h[S[a]] = sg * vectors[a * r + k] / (sg * sum);
        let margin = Infinity;
        if (r < B) {
          let mx = -Infinity;
          for (let b = 0; b < B; b++) {
            if (S.includes(b)) continue;
            let s = 0;
            for (const a of S) s += Sigma[b * B + a] * h[a];
            mx = Math.max(mx, s);
          }
          margin = theta - mx;
        }
        let gap;
        if (r > 1) {
          let other = -Infinity;
          for (let j = 0; j < r; j++) if (j !== k) other = Math.max(other, values[j]);
          gap = ev - other;
        } else gap = Infinity;
        if (margin < 0) {
          if (candidates && isTop) out.push({ S, h, c: ev, margin, gap, stable: false, part: false, why: "off-support violation" });
          continue;
        }
        out.push({ S, h, c: ev, margin, gap, stable: isTop && gap > tol && margin > tol, part: true });
      }
      // next combination
      let i = r - 1;
      while (i >= 0 && idx[i] === B - r + i) i--;
      if (i < 0) break;
      idx[i]++;
      for (let j = i + 1; j < r; j++) idx[j] = idx[j - 1] + 1;
    }
  }
  return out;
  function mk(S, vectors, k, sg, sum, r, ev, margin, gap, stable, why) {
    const h = new Float64Array(B);
    for (let a = 0; a < r; a++) h[S[a]] = sg * vectors[a * r + k] / (sg * sum);
    return { S, h, c: ev, margin, gap, stable, part: false, why };
  }
}
export const stableParts = (Sigma, B, theta) => enumerateParts(Sigma, B, theta).filter(p => p.stable);

// Stable parts on a θ grid: [{theta, parts: [{S, h, c, margin}]}].
export function partGrid(Sigma, B, thetas) {
  return thetas.map(theta => ({ theta, parts: stableParts(Sigma, B, theta).map(({ S, h, c, margin }) => ({ S, h, c, margin })) }));
}

// ------------------------------------------------------------------ dendrogram (average linkage on correlation)
// Returns {linkage: [[a, b, height, size]] (scipy layout: ids ≥ B are earlier merges, height = average
// correlation between the two merged groups, a similarity, so heights decrease up the tree), leafOrder}.
export function averageLinkage(Sigma, B) {
  const R = new Float64Array(B * B);
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) R[i * B + j] = Sigma[i * B + j] / Math.sqrt(Sigma[i * B + i] * Sigma[j * B + j]);
  let groups = Array.from({ length: B }, (_, i) => ({ id: i, members: [i] }));
  const linkage = [];
  let next = B;
  while (groups.length > 1) {
    let best = -Infinity, bi = 0, bj = 1;
    for (let i = 0; i < groups.length; i++) for (let j = i + 1; j < groups.length; j++) {
      let s = 0;
      for (const a of groups[i].members) for (const b of groups[j].members) s += R[a * B + b];
      s /= groups[i].members.length * groups[j].members.length;
      if (s > best + 1e-12) { best = s; bi = i; bj = j; }
    }
    const g = { id: next++, members: [...groups[bi].members, ...groups[bj].members] };
    linkage.push([groups[bi].id, groups[bj].id, best, g.members.length]);
    groups = groups.filter((_, k) => k !== bi && k !== bj).concat([g]);
  }
  // leaf order: depth-first, left child first
  const leafOrder = [];
  const walk = id => { if (id < B) leafOrder.push(id); else { const [a, b] = linkage[id - B]; walk(a); walk(b); } };
  walk(B + linkage.length - 1);
  return { linkage, leafOrder };
}

// Clusters of the dendrogram cut at θ: the groups whose internal merges are all above θ (sorted member lists).
export function cutClusters(linkage, B, theta) {
  const members = id => (id < B ? [id] : [...members(linkage[id - B][0]), ...members(linkage[id - B][1])]);
  const out = [];
  const walk = id => {
    if (id < B || linkage[id - B][2] > theta) out.push(members(id).sort((x, y) => x - y));
    else { walk(linkage[id - B][0]); walk(linkage[id - B][1]); }
  };
  walk(B + linkage.length - 1);
  return out;
}

export const keyOf = S => S.join(",");
