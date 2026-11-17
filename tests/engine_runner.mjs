// Node-side driver for tests/test_engine_parity.py.
// Usage: node engine_runner.mjs input.json  → prints JSON result to stdout.
// Input: {op, ...args}; matrices arrive as nested lists and leave as nested lists.
import { readFileSync } from 'node:fs';
import * as E from '../site/js/engine.js';

const inp = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const flat = a => Float64Array.from(a.flat(Infinity));
const mat = (x, r, c) => { const out = []; for (let i = 0; i < r; i++) out.push(Array.from(x.subarray(i * c, i * c + c))); return out; };
const loadState = s => E.loadState({ N: s.N, B: s.B, Wii: flat(s.Wii), Wib: flat(s.Wib), Mii: flat(s.Mii), Mib: flat(s.Mib) });
const stateOut = s => ({ Wii: mat(s.Wii, s.N, s.N), Wib: mat(s.Wib, s.N, s.B) });
const clean = r => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, ArrayBuffer.isView(v) ? Array.from(v) : v]));

const ops = {
  create_state({ N, B, seed, p, iiScale, ibScale }) {
    const s = E.createState({ N, B, seed, p, iiScale, ibScale });
    return { ...stateOut(s), Mii: mat(s.Mii, N, N), Mib: mat(s.Mib, N, B) };
  },
  gen({ name, args }) {
    const r = E[name](args);
    const S = r.Sigma || r;
    const B = Math.round(Math.sqrt(S.length));
    return { Sigma: mat(S, B, B), labels: r.labels ? Array.from(r.labels) : null };
  },
  run({ state, Sigma, P, nSteps }) {
    const s = loadState(state), S = flat(Sigma);
    E.advance(s, S, P, nSteps);
    const resid = s.resid;
    const Hstep = mat(s.H, s.N, s.B);
    const H = mat(E.harmonic(s), s.N, s.B);
    const stat = E.stationarity(s, S, P);
    return { ...stateOut(s), H, Hstep, resid, stat };
  },
  run_samples({ state, Gs, P, center }) {
    const s = loadState(state);
    for (const G of Gs) E.stepSamples(s, flat(G), G.length, P, center);
    return { ...stateOut(s), H: mat(E.harmonic(s), s.N, s.B) };
  },
  readout({ state, P, opts }) {
    const s = loadState(state);
    const H = E.harmonicFrom(s.Wii, s.Wib, s.N, s.B);
    const r = E.readoutLS(H, s.Wii, s.Wib, s.Mii, s.Mib, s.N, s.B, { theta: P.theta, lam: P.lam, eta: P.eta, gamma: P.gamma, wmin: P.wmin, ...opts });
    const naive = E.readoutNaive(H, s.Wii, s.Mii, s.N, s.B, { theta: P.theta, lam: P.lam, eta: P.eta, gamma: P.gamma });
    return { ...clean(r), C: mat(r.C, s.B, s.B), naive: naive && mat(naive, s.B, s.B) };
  },
  readout_H({ H, Wii, Wib, Mii, Mib, opts }) {
    const N = H.length, B = H[0].length;
    const r = E.readoutLS(flat(H), flat(Wii), flat(Wib), flat(Mii), flat(Mib), N, B, opts);
    return { ...clean(r), C: mat(r.C, B, B) };
  },
  transient({ prev, next, P, opts }) {
    const r = E.readoutTransient(loadState(prev), loadState(next), P, opts || {});
    return { ...clean(r), C: mat(r.C, prev.B, prev.B) };
  },
  spectrum({ state, P, opts }) {
    const s = loadState(state);
    const H = E.harmonicFrom(s.Wii, s.Wib, s.N, s.B);
    const eq = E.readoutEquations(s.Wii, s.Wib, s.Mii, s.Mib, s.N, s.B, { theta: P.theta, lam: P.lam, eta: P.eta, gamma: P.gamma, wmin: P.wmin, ...opts });
    const sp = E.measurementSpectrum(H, eq, s.N, s.B);
    const f = x => ({ rank: x.rank, cond: x.cond, eig: Array.from(x.eig) });
    return { ii: f(sp.ii), ib: f(sp.ib), both: f(sp.both), p: sp.p };
  },
  analysis({ H, Sigma, prevBasis, tol }) {
    const N = H.length, B = H[0].length, h = flat(H), S = flat(Sigma);
    const auto = E.clustersAuto(h, N, B);
    const fixed = E.clustersL1(h, N, B, tol);
    const leader = E.clustersLeader(h, N, B, tol);
    const st = E.clusterStats(h, N, B, auto.labels, auto.K, S);
    const pca = E.pca3(h, N, B, prevBasis ? { basis: flat(prevBasis) } : null);
    return {
      auto: { labels: Array.from(auto.labels), K: auto.K, tol: auto.tol, gap: auto.gap, robust: auto.robust },
      fixed: Array.from(fixed.labels), leader: Array.from(leader.labels),
      stats: { sizes: Array.from(st.sizes), parts: mat(st.parts, auto.K, B), energy: Array.from(st.energy), spread: Array.from(st.spread) },
      effRank: E.effectiveRank(h, N, B),
      pca: { coords: mat(pca.coords, N, 3), basis: mat(pca.basis, B, 3), variance: Array.from(pca.variance) },
      sim: mat(E.similarity(h, N, B, S), N, N),
      curve: E.clusterCountCurve(h, N, B, [0.001, 0.01, 0.1, 0.5]),
    };
  },
  compare({ Chat, Sigma, k }) {
    const B = Sigma.length, A = flat(Chat), S = flat(Sigma);
    const out = {};
    for (const setting of ['known', 'scale', 'contrast']) out[setting] = clean(E.compareCov(A, S, B, { setting, k }));
    const bl = E.baselines(S, B, k);
    out.baselines = Object.fromEntries(Object.entries(bl).map(([n, M]) => [n, mat(M, B, B)]));
    return out;
  },
  readout_relu({ H, Wii, Wib, Mii, Mib, opts }) {
    const N = H.length, B = H[0].length;
    const r = E.readoutRelu(flat(H), flat(Wii), flat(Wib), flat(Mii), flat(Mib), N, B, opts);
    return { ...clean(r), C: mat(r.C, B, B) };
  },
  dual({ vi, vj, c, k }) {
    return { dual: vi.map((_, q) => E.dualRelu(vi[q], vj[q], c[q])), inv: vi.map((_, q) => E.dualReluInv(vi[q], vj[q], k[q])) };
  },
  stochastic_relu({ state, Sigma, P, batch, n, seed }) {
    const s = loadState(state), sm = E.makeSampler(flat(Sigma), s.B, seed);
    let threw = false;
    try { E.stepStochastic(E.cloneState(s), sm, P, 1); } catch (e) { threw = true; }
    for (let t = 0; t < n; t++) E.stepStochastic(s, sm, P, batch);
    return { ...stateOut(s), threwBatch1: threw };
  },
  sampler({ Sigma, seed, n }) {
    const B = Sigma.length, sm = E.makeSampler(flat(Sigma), B, seed), G = new Float64Array(n * B);
    sm.draw(G, n);
    return { cov: mat(E.sampleCov(G, n, B), B, B) };
  },
};

process.stdout.write(JSON.stringify(ops[inp.op](inp)));
