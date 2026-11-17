// Chapter "many": model side (no DOM), so the live figures and their tours can be checked headless in node.
// Two live systems, both simulated with onsite_engine.js:
//   * the population sim: N neurons, s random attachments each, B up to 24 inputs (planted blocks, hierarchy,
//     power law), linear or leaky-rectifying potentials; swept in b by hand or along a path (continuation);
//   * the line sim (Q-resolution 3b): 30 linear neurons whose attachments differ along ONE contrast mode of an
//     8-input covariance with a Hadamard eigenbasis, exactly as hebb/sweeps2/resolution.py builds them.
import * as O from "./onsite_engine.js";

export const RES_TOL = 1e-6;                    // equilibrium claims only below this projected residual (SPEC §1)
export const S_TRAIN = 256, S_EVAL = 1024;      // training bank (learning), evaluation bank (population geometry)
export const dtFor = alpha => Math.min(0.5, 1 / alpha);   // as hebb/sweeps2 (onsite_analysis.default_dt)

// ------------------------------------------------------------------ input covariances
export const SIGMAS = [
  { id: "blocks12", label: "planted blocks, B = 12 (as the sweep)", B: 12, family: "blocks",
    make: () => O.covBlocks({ sizes: [3, 3, 3, 3], within: 0.8, between: 0, variances: [2, 1, 0.5, 0.25] }).Sigma },
  { id: "blocks24", label: "planted blocks, B = 24", B: 24, family: "blocks",
    make: () => O.covBlocks({ sizes: [6, 6, 6, 6], within: 0.8, between: 0, variances: [2, 1, 0.5, 0.25] }).Sigma },
  { id: "hier24", label: "hierarchy 2 × 3 × 4, B = 24", B: 24, family: "hier",
    make: () => O.covHierarchical({ branching: [2, 3, 4], corr: [0, 0.4, 0.8], variance: 1 }).Sigma },
  { id: "power12", label: "power law, B = 12 (sweep spectrum)", B: 12, family: "power",
    make: () => O.covPowerLaw({ B: 12, alpha: 1, seed: 5 }) },
  { id: "power24", label: "power law, B = 24", B: 24, family: "power",
    make: () => O.covPowerLaw({ B: 24, alpha: 1, seed: 5 }) },
];
export const sigmaById = id => SIGMAS.find(s => s.id === id) || SIGMAS[0];

export const POTS = {
  linear: { klo: 0, khi: 0, sigmaT: 0, label: "linear" },
  rect: { klo: 3, khi: 0, sigmaT: 0.7, label: "leaky rectifier" },
};

export const LIVE_DEFAULTS = { sig: "blocks12", b: 0.1, a: 1, pot: "linear", n: 40, s: 3, seed: 1 };

// The population sim. Streams: network (attachments, thresholds), training bank, evaluation bank.
export function buildLive(p) {
  const sg = sigmaById(p.sig), B = sg.B, Sigma = sg.make(), pot = POTS[p.pot] || POTS.linear;
  const net = O.makeNet({ N: p.n, B, rng: O.makeRng(1000 + p.seed), s: Math.min(p.s, B), klo: pot.klo, khi: pot.khi, sigmaT: pot.sigmaT });
  const G = O.sampleInputs(Sigma, B, S_TRAIN, O.makeRng(2000 + p.seed));
  const Ge = O.sampleInputs(Sigma, B, S_EVAL, O.makeRng(3000 + p.seed));
  const sim = O.createSim({ net, G, S: S_TRAIN, b: p.b, alpha: p.a, dt: dtFor(p.a) });
  return { sim, Sigma, B, Ge, Se: S_EVAL };
}

// J(W) = mean energy over the training bank + sum_{i<j} (alpha/2 w^2 - b w)  (the objective the rule descends)
export function objectiveJ(sim) {
  const N = sim.net.N, W = sim.net.W;
  let r = 0;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const w = W[i * N + j]; if (w > 0) r += 0.5 * sim.alpha * w * w - sim.b * w; }
  return sim.energy() + r;
}

// Components with diameters (training bank): sizes ranked (largest first), member-to-prototype D (diamMax, diamMean)
// and span = max pairwise D inside the component. rank[k] = size rank of component k; order = neurons sorted by
// component rank (then by `key` inside a component, e.g. the first MDS coordinate).
export function components(sim, key = null) {
  const N = sim.net.N, { labels, K } = O.components(sim.net.W, N);
  const st = O.clusterStats(sim.U, sim.S, N, labels, K);
  const D2 = sim.D2;
  for (const c of st) {
    let sp = 0; const m = c.members;
    for (let p = 0; p < m.length; p++) for (let q = p + 1; q < m.length; q++) sp = Math.max(sp, D2[m[p] * N + m[q]]);
    c.span = Math.sqrt(Math.max(sp, 0));
  }
  const ids = st.map((_, k) => k).sort((a, b) => st[b].size - st[a].size || a - b);
  const rank = new Int32Array(K); ids.forEach((k, r) => { rank[k] = r; });
  const order = Array.from({ length: N }, (_, i) => i).sort((i, j) => rank[labels[i]] - rank[labels[j]] || (key ? key[i] - key[j] : i - j));
  const multi = st.filter(c => c.size > 1).length;
  return { labels, K, st, rank, ids, order, multi, largest: st.length ? st[ids[0]].size : 0 };
}

// Diameters and spans of the same components on another bank (U on that bank).
export function componentsOn(U, S, N, cs) {
  const st = O.clusterStats(U, S, N, cs.labels, cs.K);
  const D2 = O.dist2(U, S, N);
  st.forEach(c => {
    let sp = 0; const m = c.members;
    for (let p = 0; p < m.length; p++) for (let q = p + 1; q < m.length; q++) sp = Math.max(sp, D2[m[p] * N + m[q]]);
    c.span = Math.sqrt(Math.max(sp, 0));
  });
  return st;
}

// ------------------------------------------------------------------ tours of the population sim
// path: continuation (b values visited in order, each run to the fixed point, W carried over); otherwise a cold
// start (W = 0) at b.
const geo = (lo, hi, n) => Array.from({ length: n }, (_, k) => +(lo * Math.pow(hi / lo, k / (n - 1))).toPrecision(3));
const LOOP_UP = geo(0.005, 0.3, 10);
export const LIVE_TOURS = [
  { id: "singletons", label: "Singletons", p: { sig: "blocks12", pot: "linear", a: 1, b: 0.005 },
    note: "$b = 0.005$ from $W = 0$: almost no pair is within an edge's range $D^2 < 2b$, so almost every neuron is its own component." },
  { id: "chains", label: "Chains at percolation", p: { sig: "blocks12", pot: "linear", a: 1, b: 0.1 },
    note: "$b = 0.1$, $\\alpha = 1$ from $W = 0$: one large component beside small ones and singletons. Read the widest span$/\\sqrt{2b}$ tile and the red cells inside the large block of the matrix: members further apart than one edge's range." },
  { id: "blob", label: "One blob", p: { sig: "blocks12", pot: "linear", a: 1, b: 0.3 },
    note: "$b = 0.3$ from $W = 0$: one component holds all but at most a neuron or two, and it is tight: compare its span with the chains view." },
  { id: "down", label: "Down from a blob (α = 0.1)", p: { sig: "blocks12", pot: "linear", a: 0.1 },
    path: [0.3, 0.2, 0.13, 0.1, 0.065, 0.042, 0.03],
    note: "$\\alpha = 0.1$: start fused at $b = 0.3$, then lower $b$ step by step to $0.03$, carrying $W$ (continuation). The network stays in a few components. Then press reset ($W = 0$) at the same $b$, or take the next tour." },
  { id: "cold03", label: "Same b from W = 0", p: { sig: "blocks12", pot: "linear", a: 0.1, b: 0.03 },
    note: "$\\alpha = 0.1$, $b = 0.03$ from $W = 0$: dozens of components where the continuation from a blob kept a few. Same parameters, different history. Compare the $J$ tile with the one the previous tour ended on: at this $b$, near the low end of the hysteresis window, the fragmented state has the lower objective, so the fused state the continuation kept is a local minimum and not the better one. Higher in the window the order is usually reversed (panel g above)." },
  { id: "loop", label: "Loop up and down, B = 24", p: { sig: "blocks24", pot: "linear", a: 0.1 },
    path: [...LOOP_UP, ...LOOP_UP.slice(0, -1).reverse()],
    note: "Planted blocks on 24 inputs, $\\alpha = 0.1$: $b$ goes up from $0.005$ to $0.3$ and back, each step run to its fixed point. The dots in panel (b) trace the two branches." },
];
export const liveTourById = id => LIVE_TOURS.find(t => t.id === id) || null;

// ------------------------------------------------------------------ the line along one mode (resolution 3b)
export const LB = 8;
export const LAM8 = [1, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625];
// normalised Hadamard basis, column 0 uniform (hebb.onsite_analysis.contrast_basis): V[i*B + k] = component i of mode k
export const V8 = (() => {
  let H = [[1]];
  while (H.length < LB) { const n = H.length, M = []; for (let i = 0; i < 2 * n; i++) { M.push([]); for (let j = 0; j < 2 * n; j++) M[i].push(H[i % n][j % n] * (i >= n && j >= n ? -1 : 1)); } H = M; }
  const V = new Float64Array(LB * LB);
  for (let i = 0; i < LB; i++) for (let k = 0; k < LB; k++) V[i * LB + k] = H[k][i] / Math.sqrt(LB);
  return V;
})();
export const SIGMA8 = O.covFromSpectrum(LAM8, LB, V8);
export const XMAX = Math.sqrt(LB) / LB;           // |x| <= XMAX keeps 1/B + x v_a >= 0
export const LINE_N = 30;
export const LINE_DEFAULTS = { m: 1, b: 0.012, a: 0.01, seed: 3 };

export function secondMoment(G, S, B) {
  const C = new Float64Array(B * B);
  for (let s = 0; s < S; s++) for (let p = 0; p < B; p++) { const gp = G[s * B + p]; for (let q = 0; q < B; q++) C[p * B + q] += gp * G[s * B + q]; }
  for (let k = 0; k < B * B; k++) C[k] /= S;
  return C;
}
export const quadMode = (C, a) => { let r = 0; for (let p = 0; p < LB; p++) for (let q = 0; q < LB; q++) r += V8[p * LB + a] * C[p * LB + q] * V8[q * LB + a]; return r; };

// 30 linear neurons with attachments A_i = 1/B + x_i v_a, x_i ~ U(-0.95 XMAX, 0.95 XMAX), sorted.
export function buildLine(p) {
  const rng = O.makeRng(4000 + p.seed);
  const x = Array.from({ length: LINE_N }, () => (2 * rng.uniform() - 1) * 0.95 * XMAX).sort((u, v) => u - v);
  const a = p.m, A = new Float64Array(LINE_N * LB);
  for (let i = 0; i < LINE_N; i++) for (let c = 0; c < LB; c++) A[i * LB + c] = 1 / LB + x[i] * V8[c * LB + a];
  const net = O.makeNetFrom({ A, N: LINE_N, B: LB });
  const G = O.sampleInputs(SIGMA8, LB, S_TRAIN, O.makeRng(5000 + p.seed));
  const Ge = O.sampleInputs(SIGMA8, LB, 4096, O.makeRng(6000 + p.seed));
  const lh = quadMode(secondMoment(G, S_TRAIN, LB), a), lhE = quadMode(secondMoment(Ge, 4096, LB), a);
  const sim = O.createSim({ net, G, S: S_TRAIN, b: p.b, alpha: p.a, dt: dtFor(p.a) });
  return { sim, x, a, lh, lhE, lam: LAM8[a] };
}
// The uncoupled prediction (statement 2 applied to every pair): components of the graph whose edges have positive
// drift at W = 0, i.e. lh (x_i - x_j)^2 < 2b. On a line these are runs of consecutive gaps below sqrt(2b/lh).
export function predictedK(x, b, lh) { let K = 1; for (let i = 1; i < x.length; i++) if (lh * (x[i] - x[i - 1]) ** 2 >= 2 * b) K++; return K; }
// Current position along the mode: h_i = 1/B + y_i v_a exactly (the harmonic map keeps the line), y = H v_a.
export function linePositions(L) {
  const H = O.harmonic(L.sim.net), y = new Float64Array(LINE_N);
  for (let i = 0; i < LINE_N; i++) { let s = 0; for (let c = 0; c < LB; c++) s += H[i * LB + c] * V8[c * LB + L.a]; y[i] = s; }
  return y;
}
export function lineAnalysis(L) {
  const { sim, x, lh, lhE } = L, N = LINE_N, b = sim.b;
  const { labels, K } = O.components(sim.net.W, N);
  const y = linePositions(L), don = Math.sqrt(2 * b / lh);
  let bridged = 0, spanT = 0, spanE = 0;
  const comps = [];
  for (let k = 0; k < K; k++) {
    const m = []; for (let i = 0; i < N; i++) if (labels[i] === k) m.push(i);
    let gx = 0; for (let q = 1; q < m.length; q++) gx = Math.max(gx, x[m[q]] - x[m[q - 1]]);     // members are sorted in x
    let ylo = Infinity, yhi = -Infinity; for (const i of m) { ylo = Math.min(ylo, y[i]); yhi = Math.max(yhi, y[i]); }
    // span from the bank's D^2 (the identity D^2 = lh (y_i - y_j)^2 holds on the training bank; lhE on the other)
    let sp = 0; for (let p = 0; p < m.length; p++) for (let q = p + 1; q < m.length; q++) sp = Math.max(sp, sim.D2[m[p] * N + m[q]]);
    const spT = Math.sqrt(Math.max(sp, 0)), spE = Math.sqrt(lhE) * (yhi - ylo);
    bridged = Math.max(bridged, gx / don); spanT = Math.max(spanT, spT / Math.sqrt(2 * b)); spanE = Math.max(spanE, spE / Math.sqrt(2 * b));
    comps.push({ members: m, size: m.length, ylo, yhi, span: spT });
  }
  return { labels, K, Kpred: predictedK(x, b, lh), y, don, bridged, spanT, spanE, comps };
}

export const LINE_TOURS = [
  { id: "condense", label: "Condensation", p: { m: 1, a: 0.01, b: 0.012 },
    note: "Mode 1, $\\alpha = 0.01$, $b = 0.012$ from $W = 0$: the uncoupled chain is nearly or fully connected (statement 2 predicts one or two components), and the fixed point breaks the line into a few tight clusters." },
  { id: "weak", label: "Stiff weights (α = 0.1)", p: { m: 1, a: 0.1, b: 0.002 },
    note: "Mode 1, $\\alpha = 0.1$, small $b$: the edges stay weak, the responses barely move, and the count is close to statement (2)." },
  { id: "down1", label: "Down-continuation", p: { m: 1, a: 0.01 }, path: [0.05, 0.015, 0.006],
    note: "Mode 1, $\\alpha = 0.01$: start fused at $b = 0.05$ and lower $b$ to $0.006$, carrying $W$. Fewer components than (2) predicts; the bridged-gap tile shows a component holding neighbours further apart than $\\delta_{on}$." },
  { id: "cold1", label: "Same b from W = 0", p: { m: 1, a: 0.01, b: 0.006 },
    note: "Mode 1, $\\alpha = 0.01$, $b = 0.006$ from $W = 0$: more components than (2) predicts, at the $b$ where the continuation from a fused line ended with fewer." },
];
export const lineTourById = id => LINE_TOURS.find(t => t.id === id) || null;

// ------------------------------------------------------------------ resolution statement (1), frozen law
// Random frozen linear network on the 8-input Hadamard covariance (or a power law): compare D^2 on the training
// bank with sum_a lam_a (dh . v_a)^2 using the bank's own eigenpairs (identity) and Sigma_g's (sampling error).
export function frozenLaw({ sig = "contrast8", wrand = false, S = 256, seed = 0 }) {
  const B = LB, N = 30;
  const Sigma = sig === "power8" ? O.covPowerLaw({ B, alpha: 1, seed: 5 }) : SIGMA8;
  const net = O.makeNet({ N, B, rng: O.makeRng(7000 + seed), s: 3 });
  if (wrand) {
    const r = O.makeRng(7500 + seed);
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const w = r.uniform() * 0.5, keep = r.uniform() < 0.2; if (keep) { net.W[i * N + j] = w; net.W[j * N + i] = w; } }
  }
  const H = O.harmonic(net);
  const G = O.sampleInputs(Sigma, B, S, O.makeRng(8000 + seed)), Ge = O.sampleInputs(Sigma, B, 4096, O.makeRng(9000 + seed));
  const resp = (Gx, Sx) => { const U = new Float64Array(Sx * N); for (let s = 0; s < Sx; s++) for (let i = 0; i < N; i++) { let v = 0; for (let c = 0; c < B; c++) v += H[i * B + c] * Gx[s * B + c]; U[s * N + i] = v; } return U; };
  const D2t = O.dist2(resp(G, S), S, N), D2e = O.dist2(resp(Ge, 4096), 4096, N);
  const eS = O.eigh(Float64Array.from(Sigma), B), eC = O.eigh(secondMoment(G, S, B), B);
  const law = (e, dh) => { let r = 0; for (let k = 0; k < B; k++) { let p = 0; for (let c = 0; c < B; c++) p += dh[c] * e.vectors[c * B + k]; r += e.values[k] * p * p; } return r; };
  const pts = { meas: [], chat: [], sig: [], eval: [] };
  let mx = 0, mxE = 0, eC1 = 0, eS1 = 0, eE1 = 0, sumdh = 0;
  const dh = new Float64Array(B);
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    let s1 = 0; for (let c = 0; c < B; c++) { dh[c] = H[i * B + c] - H[j * B + c]; s1 += dh[c]; }
    sumdh = Math.max(sumdh, Math.abs(s1));
    const m = D2t[i * N + j], lc = law(eC, dh), ls = law(eS, dh);
    pts.meas.push(m); pts.chat.push(lc); pts.sig.push(ls); pts.eval.push(D2e[i * N + j]);
    mx = Math.max(mx, m); mxE = Math.max(mxE, D2e[i * N + j]);
  }
  for (let k = 0; k < pts.meas.length; k++) {
    eC1 = Math.max(eC1, Math.abs(pts.meas[k] - pts.chat[k])); eS1 = Math.max(eS1, Math.abs(pts.meas[k] - pts.sig[k])); eE1 = Math.max(eE1, Math.abs(pts.eval[k] - pts.sig[k]));
  }
  return { pts, errChat: eC1 / mx, errSig: eS1 / mx, errEval: eE1 / mxE, sumdh, S };
}
