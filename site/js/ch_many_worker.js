// Chapter "many": the weight-space readout, off the main thread (module worker; imports the shared engine).
// The page posts the CURRENT weights of the live simulation; this worker computes, from the weights alone
// (plus the model parameters where a setting says they are known), the readouts of SPEC §3/§7.3, their scores
// against the true Σ_g, the baselines, and the spectrum of the measurement operator. Nothing here feeds back
// into the simulation.
import * as E from "./engine.js";
import { measurementGram, eigvalsSym, summarise } from "./ch_many_meas.js";
import { scoreAll as scoreAllRaw, readPair as readPairRaw, partsSum, runAndScore, identityRow } from "./ch_many_ro.js";

let run = null;          // {id, N, B, Mii, Mib, Sigma, k, untrained: scores, base: scores}
const warm = new Map();  // warm starts per readout row
const scoreAll = scoreAllRaw;
const readPair = (H, Wii, Wib, P, opts, key) => readPairRaw(H, Wii, Wib, run.Mii, run.Mib, run.N, run.B, P, opts, warm, key);

function onRun(msg) {
  const { id, N, B, Mii, Mib, Sigma, k, W0ii, W0ib, P } = msg;
  run = { id, N, B, Mii, Mib, Sigma, k };
  warm.clear();
  // baselines (fixed per Sigma)
  const bl = E.baselines(Sigma, B, k);
  run.base = {
    identity: scoreAll(bl.identity, bl.identity, Sigma, B, k),
    diag: scoreAll(bl.diag, bl.diag, Sigma, B, k),
    rankK: scoreAll(bl.rankK, bl.rankK, Sigma, B, k),
  };
  // the untrained network, read out as if it were at equilibrium (a control: the assumption is violated)
  const H0 = E.harmonicFrom(W0ii, W0ib, N, B);
  const u = readPair(H0, W0ii, W0ib, P, {}, "untrained");
  run.untrained = scoreAll(u.rk.C, u.rp.C, Sigma, B, k);
  warm.clear();
}

function onJob(msg) {
  const { Wii, Wib, prevWii, prevWib, P, wantSpec, specFamilies, t, light } = msg;
  const { N, B, Mii, Mib, Sigma, k } = run;
  const t0 = performance.now();
  const H = E.harmonicFrom(Wii, Wib, N, B);
  const rows = {};
  const main = readPair(H, Wii, Wib, P, {}, "both");
  rows.both = scoreAll(main.rk.C, main.rp.C, Sigma, B, k);
  // light jobs (dense sampling early in a run): only the main readout, the parts-sum and the derivative-aware readout
  const ii = !light && P.gamma !== 0 ? readPair(H, Wii, Wib, P, { use: "ii" }, "ii") : null;
  rows.ii = ii ? scoreAll(ii.rk.C, ii.rp.C, Sigma, B, k) : light ? "skip" : null;
  const ib = light ? null : readPair(H, Wii, Wib, P, { use: "ib" }, "ib");
  rows.ib = ib ? scoreAll(ib.rk.C, ib.rp.C, Sigma, B, k) : "skip";
  // parts-sum sketch (findings F10): θ11ᵀ + Σ_k c_k h_k h_kᵀ/|h_k|², from the clusters of the current embeddings
  const cl = E.clustersAuto(H, N, B);
  const ps = partsSum(H, Wib, Mib, N, B, P, cl.labels, cl.K);
  rows.parts = scoreAll(ps.C, E.project1(ps.C, B), Sigma, B, k);
  // derivative-aware readout from the two Euler snapshots prev -> current
  let Ctr = null;
  if (prevWii) {
    const prev = { N, B, Wii: prevWii, Wib: prevWib, Mii, Mib }, next = { N, B, Wii, Wib, Mii, Mib };
    const tr = E.readoutTransient(prev, next, P, { maxIter: 400, x0: warm.get("tr") });
    warm.set("tr", tr.C); Ctr = tr.C;
    rows.transient = scoreAll(tr.C, E.project1(tr.C, B), Sigma, B, k);
  }
  // the note's naive formula C = H⁺ R H⁺ᵀ
  const nv = light ? null : E.readoutNaive(H, Wii, Mii, N, B, P);
  rows.naive = nv ? scoreAll(nv, E.project1(nv, B), Sigma, B, k) : light ? "skip" : null;
  rows.untrained = run.untrained;
  Object.assign(rows, run.base);
  // spectra of Sigma and of the readouts (for the overlay chart), contrast setting = P·P
  const eS = E.eigh(Sigma, B).values, eSp = E.eigh(E.project1(Sigma, B), B).values;
  const eK = E.eigh(main.rk.C, B).values, eP = E.eigh(main.rp.C, B).values;
  // measurement operator: rank / spectrum (expensive; only when asked)
  let spec = null;
  if (wantSpec) {
    const eq = E.readoutEquations(Wii, Wib, Mii, Mib, N, B, { theta: P.theta, lam: P.lam, eta: P.eta, gamma: P.gamma, wmin: P.wmin });
    const G = measurementGram(H, eq, N, B, { families: specFamilies });
    spec = { p: G.p, t, mII: eq.ii.length / 2, mIB: eq.ib.length / 2 };
    for (const f of specFamilies) {
      if (!G[f]) continue;
      const ev = eigvalsSym(G[f], G.p);
      const s5 = summarise(ev, 1e-5), s10 = summarise(ev, 1e-10);
      spec[f] = { rank: s5.rank, rankLoose: s10.rank, norm: s5.norm };
    }
  }
  return {
    t, rows, mode: main.mode,
    Ck: main.rk.C, Cp: main.rp.C, Ctr, Cps: ps.C, partsK: ps.K, robust: cl.robust,
    mII: main.rk.mII, mIB: main.rk.mIB, iters: main.rk.iters, relRes: main.rk.relResidual,
    eig: { sigma: eS, sigmaP: eSp, known: eK, proj: eP }, spec, ms: performance.now() - t0,
  };
}

// Dense vs sparse wiring (and dense with a lower floor), each run to convergence here (a separate worker instance;
// the page's live run is untouched). Rows are posted one at a time so the table fills as they finish.
function onCompare(msg) {
  const { Sigma, B, N, theta, gamma, wmin, seed, k, configs } = msg;
  self.postMessage({ kind: "compare", row: identityRow(Sigma, B, k) });
  for (const c of configs) {
    const r = runAndScore({ Sigma, B, N, theta, gamma, wmin: c.wmin ?? wmin, seed, k, p: c.p });
    self.postMessage({ kind: "compare", row: { ...r, key: c.key } });
  }
  self.postMessage({ kind: "compare", done: true });
}

self.onmessage = e => {
  const msg = e.data;
  try {
    if (msg.kind === "compare") { onCompare(msg); return; }
    if (msg.kind === "run") { onRun(msg); self.postMessage({ kind: "run", id: msg.id }); return; }
    if (msg.kind === "job") {
      if (!run || msg.runId !== run.id) { self.postMessage({ kind: "job", stale: true, seq: msg.seq }); return; }
      const res = onJob(msg);
      self.postMessage({ kind: "job", seq: msg.seq, runId: run.id, res });
    }
  } catch (err) {
    self.postMessage({ kind: msg.kind, seq: msg.seq, error: String(err && err.message || err) });
  }
};
