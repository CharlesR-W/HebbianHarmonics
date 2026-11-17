// Chapter "many": the weight-space readout, off the main thread (module worker; imports the shared engine).
// The page posts the CURRENT weights of the live simulation; this worker computes, from the weights alone
// (plus the model parameters where a setting says they are known), the readouts of SPEC §3/§7.3, their scores
// against the true Σ_g, the baselines, and the spectrum of the measurement operator. Nothing here feeds back
// into the simulation.
import * as E from "./engine.js";
import { measurementGram, eigvalsSym, summarise } from "./ch_many_meas.js";

let run = null;          // {id, N, B, Mii, Mib, Sigma, k, untrained: scores, base: scores}
const warm = new Map();  // warm starts per readout row

const settings = ["known", "scale", "contrast"];
function scoreAll(Ck, Cp, S, B, k) {
  // Ck: readout with theta known (scored under 'known' and 'scale'); Cp: theta-free readout (scored under 'contrast')
  const out = {};
  for (const st of settings) {
    const m = E.compareCov(st === "contrast" ? Cp : Ck, S, B, { setting: st, k });
    out[st] = { cos: m.cosine, rel: m.relErr, bw: m.bw, sub: m.subspace, neg: m.negFrac, scale: m.scale };
  }
  return out;
}

function readPair(H, Wii, Wib, P, opts, key) {
  const { N, B, Mii, Mib } = run;
  const base = { theta: P.theta, lam: P.lam, eta: P.eta, gamma: P.gamma, wmin: P.wmin, maxIter: 300, ...opts };
  const k1 = key + ":k", k2 = key + ":p";
  const rk = E.readoutLS(H, Wii, Wib, Mii, Mib, N, B, { ...base, thetaMode: "known", x0: warm.get(k1) });
  // unknown theta: 'project' is exact for gamma = 1 (engine author's recommendation); otherwise theta is fitted
  const mode = P.gamma === 1 ? "project" : "fit";
  const rp = E.readoutLS(H, Wii, Wib, Mii, Mib, N, B, { ...base, thetaMode: mode, x0: warm.get(k2) });
  warm.set(k1, rk.C); warm.set(k2, rp.C);
  return { rk, rp, mode };
}

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
  const { Wii, Wib, prevWii, prevWib, P, wantSpec, specFamilies, t } = msg;
  const { N, B, Mii, Mib, Sigma, k } = run;
  const t0 = performance.now();
  const H = E.harmonicFrom(Wii, Wib, N, B);
  const rows = {};
  const main = readPair(H, Wii, Wib, P, {}, "both");
  rows.both = scoreAll(main.rk.C, main.rp.C, Sigma, B, k);
  const ii = P.gamma !== 0 ? readPair(H, Wii, Wib, P, { use: "ii" }, "ii") : null;
  rows.ii = ii ? scoreAll(ii.rk.C, ii.rp.C, Sigma, B, k) : null;
  const ib = readPair(H, Wii, Wib, P, { use: "ib" }, "ib");
  rows.ib = scoreAll(ib.rk.C, ib.rp.C, Sigma, B, k);
  // derivative-aware readout from the two Euler snapshots prev -> current
  let Ctr = null;
  if (prevWii) {
    const prev = { N, B, Wii: prevWii, Wib: prevWib, Mii, Mib }, next = { N, B, Wii, Wib, Mii, Mib };
    const tr = E.readoutTransient(prev, next, P, { maxIter: 400, x0: warm.get("tr") });
    warm.set("tr", tr.C); Ctr = tr.C;
    rows.transient = scoreAll(tr.C, E.project1(tr.C, B), Sigma, B, k);
  }
  // the note's naive formula C = H⁺ R H⁺ᵀ
  const nv = E.readoutNaive(H, Wii, Mii, N, B, P);
  rows.naive = nv ? scoreAll(nv, E.project1(nv, B), Sigma, B, k) : null;
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
    Ck: main.rk.C, Cp: main.rp.C, Ctr,
    mII: main.rk.mII, mIB: main.rk.mIB, iters: main.rk.iters, relRes: main.rk.relResidual,
    eig: { sigma: eS, sigmaP: eSp, known: eK, proj: eP }, spec, ms: performance.now() - t0,
  };
}

// Dense vs sparse wiring, run to convergence here (a separate worker instance; the page's live run is untouched).
function onCompare(msg) {
  const { Sigma, B, N, theta, gamma, wmin, seed, k, ps } = msg;
  const P = { ...E.defaultParams(), theta, gamma, wmin, dt: 0.2 };
  const I = new Float64Array(B * B); for (let b = 0; b < B; b++) I[b * B + b] = 1;
  const id = scoreAll(I, I, Sigma, B, k);
  const identity = { p: "id", cosK: id.known.cos, cosC: id.contrast.cos, bwC: id.contrast.bw, sub: id.contrast.sub };
  return [identity, ...ps.map(p => {
    const s = E.createState({ N, B, seed, p });
    let steps = 0;
    while (steps < 8000) { E.advance(s, Sigma, P, 50); steps += 50; if (s.resid / P.eta < 1e-8) break; }
    const H = E.harmonicFrom(s.Wii, s.Wib, N, B);
    const cl = E.clustersAuto(H, N, B);
    run = { id: -1, N, B, Mii: s.Mii, Mib: s.Mib, Sigma, k }; warm.clear();
    const pr = readPair(H, s.Wii, s.Wib, P, { maxIter: 2000 }, "cmp");
    const sc = scoreAll(pr.rk.C, pr.rp.C, Sigma, B, k);
    const eq = E.readoutEquations(s.Wii, s.Wib, s.Mii, s.Mib, N, B, { theta, lam: P.lam, eta: P.eta, gamma, wmin });
    const G = measurementGram(H, eq, N, B, { families: ["both"] });
    const ev = eigvalsSym(G.both, G.p);
    return { p, K: cl.K, dir: summarise(ev, 1e-5).rank, pdim: G.p, cosK: sc.known.cos, cosC: sc.contrast.cos, bwC: sc.contrast.bw, bwK: sc.known.bw,
      sub: sc.contrast.sub, steps, resid: s.resid / P.eta };
  })];
}

self.onmessage = e => {
  const msg = e.data;
  try {
    if (msg.kind === "compare") { self.postMessage({ kind: "compare", res: onCompare(msg) }); return; }
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
