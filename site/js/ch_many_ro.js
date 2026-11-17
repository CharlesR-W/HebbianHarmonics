// Chapter "many": readout helpers shared by the worker (ch_many_worker.js) and offline checks in node (pure module,
// no DOM). Scores in the three settings of SPEC §7.3, the parts-sum estimator of docs/findings.md F10, and the
// dense-vs-sparse comparison run to convergence.
import * as E from "./engine.js";
import { measurementGram, eigvalsSym, summarise } from "./ch_many_meas.js";

export const SETTINGS = ["known", "scale", "contrast"];

// Ck: estimate with theta known (scored under 'known' and 'scale'); Cp: theta-free estimate (scored under 'contrast')
export function scoreAll(Ck, Cp, S, B, k) {
  const out = {};
  for (const st of SETTINGS) {
    const m = E.compareCov(st === "contrast" ? Cp : Ck, S, B, { setting: st, k });
    out[st] = { cos: m.cosine, rel: m.relErr, bw: m.bw, sub: m.subspace, neg: m.negFrac, scale: m.scale };
  }
  return out;
}

// Dead neuron: every I–B weight at the floor (max_b w_ib ≤ 1.5 wmin), as in hebb/sweeps.
export function deadMask(Wib, Mib, N, B, wmin) {
  const dead = new Uint8Array(N);
  for (let i = 0; i < N; i++) { let mx = 0; for (let b = 0; b < B; b++) if (Mib[i * B + b] && Wib[i * B + b] > mx) mx = Wib[i * B + b]; dead[i] = mx <= 1.5 * wmin ? 1 : 0; }
  return dead;
}

// Cluster count, as in docs/findings.md and chapter three: dead neurons are split off (they all share the floor's
// embedding and carry no learned structure); a cluster counts when it has at least max(2, 2% N) live members;
// smaller live groups are stragglers; groups with no live member are not clusters.
export function liveClusters(labels, K, dead, N) {
  const live = new Int32Array(K);
  let nDead = 0;
  for (let i = 0; i < N; i++) { if (dead[i]) nDead++; else live[labels[i]]++; }
  const minPop = Math.max(2, Math.ceil(0.02 * N));
  let Klive = 0, kStrag = 0, nStrag = 0;
  for (let k = 0; k < K; k++) { if (live[k] >= minPop) Klive++; else if (live[k] > 0) { kStrag++; nStrag += live[k]; } }
  return { K: Klive, live, minPop, kStrag, nStrag, nDead };
}

// Parts-sum (findings F10): C = θ11ᵀ + Σ_k c_k h_k h_kᵀ/|h_k|², over clusters with at least one live member;
// h_k = member mean of h, c_k = (λ/η)·Σ_b w_ib of a member (mean over live members). Returns {C, K, parts:[{h, c, n}]}.
export function partsSum(H, Wib, Mib, N, B, P, labels, K) {
  const dead = deadMask(Wib, Mib, N, B, P.wmin);
  const h = new Float64Array(K * B), n = new Int32Array(K), nl = new Int32Array(K), cs = new Float64Array(K);
  for (let i = 0; i < N; i++) {
    const k = labels[i]; n[k]++;
    for (let b = 0; b < B; b++) h[k * B + b] += H[i * B + b];
    if (dead[i]) continue;
    nl[k]++; let s = 0; for (let b = 0; b < B; b++) s += Wib[i * B + b];
    cs[k] += s * P.lam / P.eta;
  }
  const C = new Float64Array(B * B).fill(P.theta), parts = [];
  for (let k = 0; k < K; k++) {
    if (!nl[k]) continue;
    const c = cs[k] / nl[k];
    let q = 0; for (let b = 0; b < B; b++) { h[k * B + b] /= n[k]; q += h[k * B + b] ** 2; }
    if (q <= 0) continue;
    for (let a = 0; a < B; a++) for (let b = 0; b < B; b++) C[a * B + b] += c * h[k * B + a] * h[k * B + b] / q;
    parts.push({ c, n: n[k] });
  }
  return { C, K: parts.length, parts };
}

// θ-unknown readout options. 'project' (θ absorbed by the 1-directions, ridge on PCP) for γ ∈ {0, 1}, as before;
// otherwise θ is fitted as a free unknown, except when no I–I equation is active (as at many converged states, and for
// the I–B-only row): then θ is not identifiable, 'fit' is singular (warm-started CG wanders: contrast cosines from −0.01
// to 0.96 along one run at γ = 0.5), and any fixed θ gives the same P Ĉ P, so θ = 0 is used. The caller projects with P.
export function thetaFreeOpts(Wii, Mii, N, gamma, wmin, use = "both") {
  if (gamma === 1 || gamma === 0) return { thetaMode: "project" };
  if (use === "ib") return { thetaMode: "known", theta: 0 };
  const thr = wmin * (1 + 1e-6);
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (Mii[i * N + j] && Wii[i * N + j] > thr) return { thetaMode: "fit" };
  return { thetaMode: "known", theta: 0 };
}

// Masked least-squares readout, theta known and theta unknown (see thetaFreeOpts).
export function readPair(H, Wii, Wib, Mii, Mib, N, B, P, opts = {}, warm = null, key = "") {
  const base = { theta: P.theta, lam: P.lam, eta: P.eta, gamma: P.gamma, wmin: P.wmin, maxIter: 300, ...opts };
  const rk = E.readoutLS(H, Wii, Wib, Mii, Mib, N, B, { ...base, thetaMode: "known", x0: warm ? warm.get(key + ":k") : undefined });
  const tf = thetaFreeOpts(Wii, Mii, N, P.gamma, P.wmin, opts.use), mode = tf.thetaMode === "known" ? "θ = 0 (not identifiable)" : tf.thetaMode;
  const rp = E.readoutLS(H, Wii, Wib, Mii, Mib, N, B, { ...base, ...tf, x0: warm ? warm.get(key + ":p") : undefined });
  if (warm) { warm.set(key + ":k", rk.C); warm.set(key + ":p", rp.C); }
  return { rk, rp, mode };
}

// One network run to convergence, then: K, dead, directions pinned down, masked LS and parts-sum scores, and the
// best masked-LS contrast cosine seen along training (the readout applied as if at equilibrium every `every` steps).
export function runAndScore({ Sigma, B, N, theta, gamma, wmin, seed, k, p, iiScale = 0.03, dt = 0.05, tol = 1e-7, maxSteps = 12000, every = 10, track = 600 }) {
  const P = { ...E.defaultParams(), theta, gamma, wmin, dt };
  const s = E.createState({ N, B, seed, p, iiScale });
  const warm = new Map();
  let steps = 0, peak = -Infinity, peakStep = 0;
  while (steps < maxSteps) {
    const n = steps < track ? every : 50;
    E.advance(s, Sigma, P, n); steps += n;
    if (steps <= track) {
      const H = E.harmonicFrom(s.Wii, s.Wib, N, B);
      const rp = E.readoutLS(H, s.Wii, s.Wib, s.Mii, s.Mib, N, B, { theta, lam: P.lam, eta: P.eta, gamma, wmin, maxIter: 200, ...thetaFreeOpts(s.Wii, s.Mii, N, gamma, wmin), x0: warm.get("t") });
      warm.set("t", rp.C);
      const c = E.compareCov(rp.C, Sigma, B, { setting: "contrast", k }).cosine;
      if (c > peak) { peak = c; peakStep = steps; }
    } else if (s.resid / P.eta < tol) break;
  }
  const r = E.stationarity(s, Sigma, P).r / P.eta;
  const H = E.harmonicFrom(s.Wii, s.Wib, N, B);
  const cl = E.clustersAuto(H, N, B);
  const lc = liveClusters(cl.labels, cl.K, deadMask(s.Wib, s.Mib, N, B, wmin), N), dead = lc.nDead;
  const pr = readPair(H, s.Wii, s.Wib, s.Mii, s.Mib, N, B, P, { maxIter: 2000 });
  const sc = scoreAll(pr.rk.C, pr.rp.C, Sigma, B, k);
  const ps = partsSum(H, s.Wib, s.Mib, N, B, P, cl.labels, cl.K);
  const sp = scoreAll(ps.C, E.project1(ps.C, B), Sigma, B, k);
  const eq = E.readoutEquations(s.Wii, s.Wib, s.Mii, s.Mib, N, B, { theta, lam: P.lam, eta: P.eta, gamma, wmin });
  const G = measurementGram(H, eq, N, B, { families: ["both"] });
  const dir = summarise(eigvalsSym(G.both, G.p), 1e-5).rank;
  return { p, wmin, K: lc.K, nStrag: lc.nStrag, dead, dir, pdim: G.p, steps, resid: r,
    cosK: sc.known.cos, cosC: sc.contrast.cos, bwC: sc.contrast.bw, sub: sc.contrast.sub,
    psC: sp.contrast.cos, psBW: sp.contrast.bw, peak, peakT: peakStep * dt };
}

export function identityRow(Sigma, B, k) {
  const I = new Float64Array(B * B); for (let b = 0; b < B; b++) I[b * B + b] = 1;
  const id = scoreAll(I, I, Sigma, B, k);
  return { p: "id", cosK: id.known.cos, cosC: id.contrast.cos, bwC: id.contrast.bw, sub: id.contrast.sub };
}
