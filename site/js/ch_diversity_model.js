// Chapter "diversity", model side: the live experiments (no DOM). Part A: the rank-one (scalar-drive) experiment,
// three arms with matched attachments, run to a fixed point and measured on an independent evaluation bank.
// Part B: the s ± c example and the trained block presets, compared against cluster prototypes and Σ_g modes.
// Everything uses ./onsite_engine.js; the browser RNG is not numpy's, so live numbers are comparable to the sweep
// JSONs (data/sweeps2/diversity.json, eigen.json) but not sample-for-sample identical.
import * as O from "./onsite_engine.js";

// ------------------------------------------------------------------ shared numerics
export const STOP_TOL = 1e-8;        // runs stop when the projected residual max|dw/dt| falls below this
export const CLAIM_TOL = 1e-6;       // equilibrium statements require r below this (SPEC §1)
export const MAX_STEPS = 20000;

// held-out relative MSE of y ≈ a + c x (scalar regressor): fit on [0, h), score on [h, S)
export function heldOutAffine(y, x, S, h = S >> 1) {
  let mx = 0, my = 0;
  for (let s = 0; s < h; s++) { mx += x[s]; my += y[s]; }
  mx /= h; my /= h;
  let sxx = 0, sxy = 0;
  for (let s = 0; s < h; s++) { const dx = x[s] - mx; sxx += dx * dx; sxy += dx * (y[s] - my); }
  const c = sxx > 0 ? sxy / sxx : 0, a = my - c * mx;
  let se = 0, mt = 0;
  for (let s = h; s < S; s++) mt += y[s];
  mt /= (S - h);
  let vt = 0;
  for (let s = h; s < S; s++) { const r = y[s] - a - c * x[s]; se += r * r; const d = y[s] - mt; vt += d * d; }
  return vt > 1e-14 * (S - h) ? se / vt : 0;
}
// in-sample least squares y ≈ a + c x over all S: {a, c, res (mean squared residual), varY}
export function fitAffine(y, x, S) {
  let mx = 0, my = 0;
  for (let s = 0; s < S; s++) { mx += x[s]; my += y[s]; }
  mx /= S; my /= S;
  let sxx = 0, sxy = 0, syy = 0;
  for (let s = 0; s < S; s++) { const dx = x[s] - mx, dy = y[s] - my; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
  const c = sxx > 0 ? sxy / sxx : 0;
  return { a: my - c * mx, c, res: Math.max(0, (syy - c * sxy) / S), varY: syy / S };
}
const col = (U, S, N, i, out = new Float64Array(S)) => { for (let s = 0; s < S; s++) out[s] = U[s * N + i]; return out; };

// centred, unit-norm columns (mean over the bank removed), as hebb/sweeps2/eigen.py unit_cols
function unitCols(X, S, n) {
  const Y = new Float64Array(S * n);
  for (let a = 0; a < n; a++) {
    let mu = 0; for (let s = 0; s < S; s++) mu += X[s * n + a]; mu /= S;
    let nn = 0; for (let s = 0; s < S; s++) { const v = X[s * n + a] - mu; Y[s * n + a] = v; nn += v * v; }
    nn = Math.sqrt(nn) || 1; for (let s = 0; s < S; s++) Y[s * n + a] /= nn;
  }
  return Y;
}
const colDot = (X, nx, a, Y, ny, b, S) => { let v = 0; for (let s = 0; s < S; s++) v += X[s * nx + a] * Y[s * ny + b]; return v; };

// a generic job runner: jobs advance a few ms per animation frame (pumped by the visible figures) and call their
// listeners once when finished. Cached by key, so figures that ask for the same run share it.
export function createRunner({ cap = 16 } = {}) {
  const cache = new Map();
  let lastStart = -1e9;
  const runner = {
    get(key, make) {
      let j = cache.get(key);
      if (!j) {
        j = make(); j.key = key; j.listeners = new Set(); j.prio = 0;
        cache.set(key, j);
        if (cache.size > cap) for (const [k, x] of cache) { if (cache.size <= cap) break; if (x.done && x.listeners.size === 0) cache.delete(k); }
      }
      return j;
    },
    // call cb(job) when the job is done (immediately if it already is)
    when(job, cb) { if (job.done) cb(job); else job.listeners.add(cb); return () => job.listeners.delete(cb); },
    pending() { return [...cache.values()].filter(j => !j.done && j.listeners.size > 0); },
    pump(budget = 14) {
      const now = performance.now();
      if (now - lastStart < 8) return;           // several figures on screen: pump once per frame
      lastStart = now;
      const t0 = now;
      const todo = runner.pending().sort((a, b) => b.prio - a.prio);
      for (const j of todo) {
        const left = budget - (performance.now() - t0);
        if (left <= 0.5) break;
        j.advance(left);
        if (j.done) { const ls = [...j.listeners]; j.listeners.clear(); ls.forEach(f => f(j)); }
      }
    },
  };
  return runner;
}

// a job wrapping a live sim run to stationarity, then analysed once
function simJob(sim, analyse) {
  const job = {
    sim, done: false, result: null, t0: performance.now(),
    advance(ms) {
      if (job.done) return;
      if (!(sim.residual < STOP_TOL) && sim.steps < MAX_STEPS) sim.stepFor(Math.max(1, ms), MAX_STEPS - sim.steps, STOP_TOL);
      if (sim.residual < STOP_TOL || sim.steps >= MAX_STEPS || !Number.isFinite(sim.residual)) {
        job.result = analyse(sim); job.done = true; job.ms = performance.now() - job.t0;
      }
    },
  };
  // the initial state may already be stationary (all pairs dead from W = 0)
  return job;
}

// ================================================================== Part A: scalar drive
export const NA = 40, BA = 12, SA = 3, S_TRAIN = 256, S_EVAL = 2048;
export const V = (() => { const v = []; for (let b = 0; b < BA; b++) v.push(0.5 + b / (BA - 1)); const m = Math.sqrt(v.reduce((a, x) => a + x * x, 0) / BA); return v.map(x => x / m); })();
export const ZGRID = Array.from({ length: 41 }, (_, k) => -2.5 + k * 0.125);
export const ARM_IDS = ["hetero", "homo", "linear"];
export const ARM_LABEL = { hetero: "heterogeneous", homo: "homogeneous", linear: "linear" };

const bankCache = new Map();
function scalarBank(S, seed) {
  const key = `${S}|${seed}`;
  if (bankCache.has(key)) return bankCache.get(key);
  const r = O.makeRng(seed), z = new Float64Array(S), G = new Float64Array(S * BA);
  for (let s = 0; s < S; s++) { z[s] = r.normal(); for (let b = 0; b < BA; b++) G[s * BA + b] = z[s] * V[b]; }
  const out = { z, G, S };
  bankCache.set(key, out);
  return out;
}
const GRID_G = (() => { const G = new Float64Array(ZGRID.length * BA); ZGRID.forEach((z, s) => { for (let b = 0; b < BA; b++) G[s * BA + b] = z * V[b]; }); return G; })();
// seed streams (browser RNG): attachments 7s+3, threshold draws 7s+4, training bank 7s+1, evaluation bank 7s+2
export const banksA = seed => ({ train: scalarBank(S_TRAIN, 7 * seed + 1), evalb: scalarBank(S_EVAL, 7 * seed + 2) });
function netA(spec) {
  const A = O.makeAttachments(NA, BA, SA, O.makeRng(7 * spec.seed + 3));
  const r = O.makeRng(7 * spec.seed + 4), t = new Float64Array(NA);
  const p = armPotential(spec);
  for (let i = 0; i < NA; i++) t[i] = p.st * r.normal();     // the same draws scaled by σ_t: arms are matched
  return O.makeNetFrom({ N: NA, B: BA, A, t, klo: p.klo, khi: p.khi });
}
// spec: {arm, klo, khi, st, b, alpha, seed}; the homogeneous arm keeps the potential and drops the thresholds
export function armPotential(spec) {
  if (spec.arm === "linear") return { klo: 0, khi: 0, st: 0 };
  if (spec.arm === "homo") return { klo: spec.klo, khi: spec.khi, st: 0 };
  return { klo: spec.klo, khi: spec.khi, st: spec.st };
}
export const keyA = s => `A|${s.arm}|${s.arm === "linear" ? "" : `${s.klo}|${s.khi}|${s.arm === "hetero" ? s.st : 0}`}|${s.b}|${s.alpha}|${s.seed}`;

export function jobA(runner, spec) {
  return runner.get(keyA(spec), () => {
    const net = netA(spec), { train } = banksA(spec.seed);
    const sim = O.createSim({ net, G: train.G, S: S_TRAIN, b: spec.b, alpha: spec.alpha, dt: Math.min(0.5, 1 / spec.alpha) });
    const j = simJob(sim, s => analyseA(s, spec));
    j.spec = spec;
    return j;
  });
}

// everything the figures need from one finished run
export function analyseA(sim, spec) {
  const net = sim.net, N = NA, b = spec.b;
  const { train, evalb } = banksA(spec.seed);
  const Se = evalb.S, St = train.S;
  const Ue = O.equilibrate(net, evalb.G, Se).U;
  const Ug = O.equilibrate(net, GRID_G, ZGRID.length).U;
  const { labels, K } = O.components(net.W, N);
  const cs = O.clusterStats(Ue, Se, N, labels, K);
  const yi = new Float64Array(Se), yt = new Float64Array(St);
  // per neuron: nonlinearity = held-out relative error of an affine fit from the input. With g = z v the input is
  // one scalar times a fixed vector, so an affine fit from g is an affine fit from z.
  const affErr = new Float64Array(N), varE = new Float64Array(N);
  for (let i = 0; i < N; i++) { col(Ue, Se, N, i, yi); affErr[i] = heldOutAffine(yi, evalb.z, Se); varE[i] = fitAffine(yi, evalb.z, Se).varY; }
  // training-bank prototypes (the fixed-point identities hold on the training bank)
  const comps = [];
  for (let k = 0; k < K; k++) {
    const c = cs[k], mem = [...c.members], m = mem.length;
    const comp = { k, members: mem, size: m, diamMax: c.diamMax, diamMean: c.diamMean, dEff: c.dEff };
    if (m >= 2) {
      // held-out prototype error per member (eval bank), and the affine map used for the "mapped" view (all of eval)
      comp.protoErr = mem.map(i => heldOutAffine(col(Ue, Se, N, i, yi), c.proto, Se));
      comp.map = mem.map(i => { const f = fitAffine(col(Ue, Se, N, i, yi), c.proto, Se); return { a: f.a, c: f.c }; });
      comp.protoErrMean = comp.protoErr.reduce((a, x) => a + x, 0) / m;
      comp.protoErrMax = Math.max(...comp.protoErr);
      // numerical rank of the centred member responses (singular values > 1e-3 of the largest)
      const C = new Float64Array(m * m), mu = mem.map(i => { let a = 0; for (let s = 0; s < Se; s++) a += Ue[s * N + i]; return a / Se; });
      for (let s = 0; s < Se; s++) for (let p = 0; p < m; p++) { const xp = Ue[s * N + mem[p]] - mu[p]; for (let q = p; q < m; q++) C[p * m + q] += xp * (Ue[s * N + mem[q]] - mu[q]); }
      for (let p = 0; p < m; p++) for (let q = p; q < m; q++) { C[p * m + q] /= Se; C[q * m + p] = C[p * m + q]; }
      const ev = O.eigh(C, m).values, top = Math.sqrt(Math.max(ev[0], 0));
      comp.rank = top > 0 ? ev.filter(x => Math.sqrt(Math.max(x, 0)) > 1e-3 * top).length : 0;
      // span (largest pairwise D on the training bank) and hop diameter of the live graph inside the component
      let span2 = 0;
      for (const i of mem) for (const j of mem) span2 = Math.max(span2, sim.D2[i * N + j]);
      comp.span = Math.sqrt(span2);
      let hops = 0;
      for (const i of mem) {
        const d = new Map([[i, 0]]), q = [i];
        while (q.length) { const x = q.shift(); for (const y of mem) if (!d.has(y) && net.W[x * N + y] > 1e-9) { d.set(y, d.get(x) + 1); q.push(y); } }
        for (const v of d.values()) hops = Math.max(hops, v);
      }
      comp.hops = hops;
      // training bank: squared distance of each member to the prototype, split into the part an affine map
      // (gain and offset) removes and the remaining shape part
      const pt = new Float64Array(St);
      for (let s = 0; s < St; s++) { let a = 0; for (const i of mem) a += sim.U[s * N + i]; pt[s] = a / m; }
      let d2 = 0, shape = 0;
      for (const i of mem) {
        col(sim.U, St, N, i, yt);
        let a = 0; for (let s = 0; s < St; s++) { const d = yt[s] - pt[s]; a += d * d; }
        d2 += a / St; shape += fitAffine(yt, pt, St).res;
      }
      comp.d2p = d2 / m; comp.shape = shape / m;
      // prototype on the z grid
      comp.protoGrid = ZGRID.map((_, s) => { let a = 0; for (const i of mem) a += Ug[s * N + i]; return a / m; });
    }
    comps.push(comp);
  }
  comps.sort((x, y) => y.size - x.size || x.k - y.k);
  const multi = comps.filter(c => c.size >= 2);
  const inMulti = multi.reduce((a, c) => a + c.size, 0);
  const wavg = key => inMulti ? multi.reduce((a, c) => a + c[key] * c.size, 0) / inMulti : NaN;
  // live edges: D²/2b on the training bank, and w
  const edges = [];
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (net.W[i * N + j] > 1e-9) edges.push({ i, j, r: sim.D2[i * N + j] / (2 * b), w: net.W[i * N + j] });
  let dev = 0; for (const e of edges) dev = Math.max(dev, Math.abs(spec.alpha * e.w - (b - b * e.r)));
  let slack = Infinity;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (!(net.W[i * N + j] > 1e-9)) slack = Math.min(slack, sim.D2[i * N + j] - 2 * b);
  return {
    spec, K, labels, comps, multi, inMulti,
    curves: Array.from({ length: N }, (_, i) => ZGRID.map((_, s) => Ug[s * N + i])),
    t: Float64Array.from(net.t), affErr, varE, Ue,
    affMean: affErr.reduce((a, x) => a + x, 0) / N,
    protoErr: wavg("protoErrMean"), dEff: wavg("dEff"), rank: wavg("rank"),
    protoErrMax: multi.length ? Math.max(...multi.map(c => c.protoErrMax)) : NaN,
    edges, fpDev: dev, deadSlack: slack,
    residual: sim.residual, eqResidual: sim.eqResidual, steps: sim.steps,
  };
}

// the hand-made output stage of the last figure: r_i = max(0, a_i u_i + c_i), with a_i = e^{0.3 ξ_i}/sd(u_i)
// and c_i = σ_off η_i (fixed draws). It is not part of the model and the rule never sees it.
export function outputStage(res, comp, sigOff, on = true, seed = 11) {
  const r = O.makeRng(seed), N = NA, Se = S_EVAL, m = comp.size;
  const gains = [], offs = [];
  for (let p = 0; p < m; p++) { gains.push(Math.exp(0.3 * r.normal())); offs.push(r.normal()); }
  const mem = comp.members;
  const sd = mem.map(i => Math.sqrt(Math.max(res.varE[i], 1e-30)));
  const f = (u, p) => on ? Math.max(0, gains[p] * u / sd[p] + sigOff * offs[p]) : u;
  // grid curves
  const curves = mem.map((i, p) => res.curves[i].map(u => f(u, p)));
  // eval-bank outputs and their prototype error
  const R = new Float64Array(Se * m);
  for (let s = 0; s < Se; s++) for (let p = 0; p < m; p++) R[s * m + p] = f(res.Ue[s * N + mem[p]], p);
  const proto = new Float64Array(Se);
  for (let s = 0; s < Se; s++) { let a = 0; for (let p = 0; p < m; p++) a += R[s * m + p]; proto[s] = a / m; }
  const y = new Float64Array(Se);
  const errs = []; let silent = 0;
  for (let p = 0; p < m; p++) {
    let nz = 0; for (let s = 0; s < Se; s++) { y[s] = R[s * m + p]; if (y[s] > 0) nz++; }
    if (nz === 0) { silent++; continue; }
    errs.push(heldOutAffine(y, proto, Se));
  }
  const protoGrid = ZGRID.map((_, s) => curves.reduce((a, c) => a + c[s], 0) / m);
  return { curves, protoGrid, offs: offs.map(x => sigOff * x), gains, protoErr: errs.length ? errs.reduce((a, x) => a + x, 0) / errs.length : NaN, protoErrMax: errs.length ? Math.max(...errs) : NaN, silent };
}

// ================================================================== Part B1: the s ± c example
// B = 2, g1 = s + c, g2 = s − c, var s = 1, var c = sc2; nA neurons attached to g1, nB to g2 (each with a leak
// share ε ~ U(0, mix) on the other input); linear neurons; b = sc2/2, α = 1 (as hebb/sweeps2/eigen.py part A).
export const SC_SEVAL = 4096;
export function computeSC({ sc2, nA, nB, mix = 0, seed = 0 }) {
  const N = nA + nB, B = 2;
  const rng = O.makeRng(100 + seed);
  const A = new Float64Array(N * 2);
  for (let i = 0; i < N; i++) { const e = rng.uniform() * mix; if (i < nA) { A[i * 2] = 1 - e; A[i * 2 + 1] = e; } else { A[i * 2 + 1] = 1 - e; A[i * 2] = e; } }
  const Sig = new Float64Array([1 + sc2, 1 - sc2, 1 - sc2, 1 + sc2]);
  // draw s and c directly, so the functions s and c are known exactly on each bank
  const draw = (S, sd) => { const r = O.makeRng(sd), G = new Float64Array(S * 2), sv = new Float64Array(S), cv = new Float64Array(S), sc = Math.sqrt(sc2);
    for (let q = 0; q < S; q++) { sv[q] = r.normal(); cv[q] = sc * r.normal(); G[q * 2] = sv[q] + cv[q]; G[q * 2 + 1] = sv[q] - cv[q]; } return { G, s: sv, c: cv, S }; };
  const tr = draw(256, 200 + seed), ev = draw(SC_SEVAL, 300 + seed);
  const net = O.makeNetFrom({ N, B, A });
  const b = sc2 / 2, alpha = 1;
  const run = O.run(net, tr.G, tr.S, b, alpha, { dt: 0.5, T: 20000, tol: 1e-10 });
  const { labels, K } = O.components(net.W, N);
  const Se = ev.S, Ue = O.equilibrate(net, ev.G, Se).U;
  const pm = O.populationModes(Ue, Se, N);
  // every linear response is x s + y c exactly (rows of H sum to 1, so x = 1); regress to get coordinates
  const coordOf = f => { let ss = 0, sc = 0, cc = 0, fs = 0, fc = 0;
    for (let q = 0; q < Se; q++) { const s = ev.s[q], c = ev.c[q], y = f(q); ss += s * s; sc += s * c; cc += c * c; fs += y * s; fc += y * c; }
    const det = ss * cc - sc * sc; return [(fs * cc - fc * sc) / det, (fc * ss - fs * sc) / det]; };
  const r = Math.min(2, pm.r);
  const psi = []; for (let a = 0; a < r; a++) psi.push(coordOf(q => pm.Psi[q * pm.r + a]));
  const H = O.harmonic(net);
  const neurons = []; for (let i = 0; i < N; i++) neurons.push([H[i * 2] + H[i * 2 + 1], H[i * 2] - H[i * 2 + 1]]);   // (x, y) in f = x s + y c
  // prototypes (means of the two clusters' responses), in (s, c) coordinates
  const proto = [0, 1].map(k => { const idx = k === 0 ? [...Array(nA).keys()] : [...Array(nB).keys()].map(x => x + nA); const p = [0, 0]; idx.forEach(i => { p[0] += neurons[i][0]; p[1] += neurons[i][1]; }); return [p[0] / idx.length, p[1] / idx.length]; });
  // geometry of the plane: <f, f'> = x x' + sc2 y y'
  const ip = (u, v) => u[0] * v[0] + sc2 * u[1] * v[1], nrm = u => Math.sqrt(ip(u, u)), cosf = (u, v) => ip(u, v) / (nrm(u) * nrm(v) || 1);
  const sDir = [1, 0], cDir = [0, 1];
  const nuA = ip(proto[0], proto[0]), nuB = ip(proto[1], proto[1]);
  const rho = cosf(proto[0], proto[1]);
  // tight-cluster prediction: eigenvalues of the cluster Gram (νn)^{1/2} PᵀP (νn)^{1/2}
  const gA = nuA * nA, gB = nuB * nB, off = Math.sqrt(gA * gB) * rho, tr2 = gA + gB, disc = Math.sqrt((gA - gB) ** 2 + 4 * off * off);
  const qGram = [(tr2 + disc) / 2, (tr2 - disc) / 2];
  const nun = [gA, gB].sort((x, y) => y - x);
  // exact population spectrum H Σ Hᵀ (no sampling)
  const HSH = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { let v = 0; for (let p = 0; p < 2; p++) for (let q = 0; q < 2; q++) v += H[i * 2 + p] * Sig[p * 2 + q] * H[j * 2 + q]; HSH[i * N + j] = v; }
  const qExact = O.eigh(HSH, N).values.slice(0, 2);
  // commutator of K = HᵀH with Σ (relative, Frobenius)
  const Km = [0, 0, 0, 0];
  for (let i = 0; i < N; i++) for (let p = 0; p < 2; p++) for (let q = 0; q < 2; q++) Km[p * 2 + q] += H[i * 2 + p] * H[i * 2 + q];
  const comm = commRel(Km, Sig, 2);
  const larger = nA >= nB ? 0 : 1;
  return {
    sc2, nA, nB, mix, N, K, labels, neurons, proto, psi, q: pm.q.slice(0, 2), qExact, qGram, nun,
    cosPsiS: psi.map(p => Math.abs(cosf(p, sDir))), cosPsiC: psi.map(p => Math.abs(cosf(p, cDir))),
    cosPsiProto: psi.map(p => [Math.abs(cosf(p, proto[0])), Math.abs(cosf(p, proto[1]))]),
    rho, larger, comm, degenerate: Math.abs(pm.q[0] - pm.q[1]) < 0.02 * pm.q[0] && Math.abs(qExact[0] - qExact[1]) < 1e-6 * qExact[0],
    residual: run.residual, eqResidual: run.eqResidual, steps: run.steps, b,
  };
}
function commRel(Kmat, Sig, B) {
  let cn = 0, kn = 0, sn = 0;
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) {
    let v = 0; for (let k = 0; k < B; k++) v += Kmat[i * B + k] * Sig[k * B + j] - Sig[i * B + k] * Kmat[k * B + j];
    cn += v * v; kn += Kmat[i * B + j] ** 2; sn += Sig[i * B + j] ** 2;
  }
  return Math.sqrt(cn) / (Math.sqrt(kn) * Math.sqrt(sn) || 1);
}

// ================================================================== Part B2: trained block presets
// N = 48, B = 12 in four blocks of three inputs (variances 2/1/0.5/0.25, within-block correlation 0.8, cross-block
// correlation 0 or 0.4); block attachments: 12 neurons per block, each attached to the three inputs of its block
// with Dirichlet(1) shares; α = 1 (as hebb/sweeps2/eigen.py part B).
export const NB2 = 48, BB2 = 12, NBLK = 4, BLK_VAR = [2, 1, 0.5, 0.25], SB_EVAL = 4096;
export function sigmaBlocks(rx) {
  const S = new Float64Array(BB2 * BB2), m = BB2 / NBLK;
  for (let i = 0; i < BB2; i++) for (let j = 0; j < BB2; j++) {
    const bi = Math.floor(i / m), bj = Math.floor(j / m);
    const r = i === j ? 1 : bi === bj ? 0.8 : rx;
    S[i * BB2 + j] = r * Math.sqrt(BLK_VAR[bi] * BLK_VAR[bj]);
  }
  return S;
}
function blockAttachments(seed) {
  const r = O.makeRng(500 + seed), A = new Float64Array(NB2 * BB2), m = BB2 / NBLK, per = NB2 / NBLK;
  for (let i = 0; i < NB2; i++) {
    const k = Math.floor(i / per), e = [0, 1, 2].map(() => r.exponential(1)), s = e[0] + e[1] + e[2];
    for (let q = 0; q < m; q++) A[i * BB2 + k * m + q] = e[q] / s;
  }
  return A;
}
export const keyB = s => `B|${s.rx}|${s.pot}|${s.b}|${s.seed}`;
export function jobB(runner, spec) {
  return runner.get(keyB(spec), () => {
    const Sig = sigmaBlocks(spec.rx);
    const A = blockAttachments(spec.seed);
    const rt = O.makeRng(600 + spec.seed), t = new Float64Array(NB2);
    const rect = spec.pot === "rect";
    for (let i = 0; i < NB2; i++) t[i] = rect ? 0.7 * rt.normal() : 0;
    const net = O.makeNetFrom({ N: NB2, B: BB2, A, t, klo: rect ? 3 : 0, khi: 0 });
    const G = O.sampleInputs(Sig, BB2, 256, O.makeRng(700 + spec.seed));
    const sim = O.createSim({ net, G, S: 256, b: spec.b, alpha: 1, dt: 0.5 });
    const j = simJob(sim, s => analyseB(s, spec, Sig));
    j.spec = spec;
    return j;
  });
}
function analyseB(sim, spec, Sig) {
  const net = sim.net, N = NB2, B = BB2, Se = SB_EVAL;
  const Ge = O.sampleInputs(Sig, B, Se, O.makeRng(800 + spec.seed));
  const Ue = O.equilibrate(net, Ge, Se).U;
  const { labels, K } = O.components(net.W, N);
  const cs = O.clusterStats(Ue, Se, N, labels, K);
  // clusters ordered by size (largest first), then by first member
  const order = cs.map((c, k) => k).sort((a, b) => cs[b].size - cs[a].size || a - b);
  const sizes = order.map(k => cs[k].size);
  const multi = order.filter(k => cs[k].size > 1);
  const Km = multi.length, r = Math.max(Km, 1);
  const P = new Float64Array(Se * K);
  order.forEach((k, kk) => { for (let s = 0; s < Se; s++) P[s * K + kk] = cs[k].proto[s]; });
  const Pu = unitCols(P, Se, K);
  // ν_k: mean member variance of cluster k
  const nu = order.map(k => { const mem = cs[k].members; let tot = 0; for (const i of mem) { let m = 0, q2 = 0; for (let s = 0; s < Se; s++) { const v = Ue[s * N + i]; m += v; q2 += v * v; } m /= Se; tot += q2 / Se - m * m; } return tot / mem.length; });
  const pm = O.populationModes(Ue, Se, N);
  const rr = Math.min(r, pm.r);
  const Psi = new Float64Array(Se * rr);
  for (let s = 0; s < Se; s++) for (let a = 0; a < rr; a++) Psi[s * rr + a] = pm.Psi[s * pm.r + a];
  const Psu = unitCols(Psi, Se, rr);
  const { values: lam, vectors: Vs } = O.eigh(Sig, B);
  const E = new Float64Array(Se * B);
  for (let s = 0; s < Se; s++) for (let a = 0; a < B; a++) { let v = 0; for (let q = 0; q < B; q++) v += Ge[s * B + q] * Vs[q * B + a]; E[s * B + a] = v; }
  const Eu = unitCols(E, Se, B);
  const nModes = Math.min(B, Math.max(6, rr + 2));
  const cosP = [], cosE = [], bestP = [], bestE = [];
  for (let a = 0; a < rr; a++) {
    const rp = [], re = [];
    for (let k = 0; k < K; k++) rp.push(colDot(Psu, rr, a, Pu, K, k, Se));
    for (let q = 0; q < B; q++) re.push(colDot(Psu, rr, a, Eu, B, q, Se));
    // fix the arbitrary sign of ψ_a: its best-matching prototype gets a positive cosine
    let bi = 0; rp.forEach((v, k) => { if (Math.abs(v) > Math.abs(rp[bi])) bi = k; });
    const sg = rp[bi] < 0 ? -1 : 1;
    cosP.push(rp.slice(0, Math.min(K, 8)).map(v => sg * v));
    // Σ-mode signs are arbitrary too: report |cos| there
    cosE.push(re.slice(0, nModes).map(v => Math.abs(v)));
    bestP.push(Math.max(...rp.map(Math.abs)));
    bestE.push(Math.max(...re.map(Math.abs)));
  }
  // prototype orthogonality among multi-member clusters
  let pc = 0, np = 0, pcMax = 0;
  for (let x = 0; x < Km; x++) for (let y = x + 1; y < Km; y++) { const v = Math.abs(colDot(Pu, K, x, Pu, K, y, Se)); pc += v; np++; pcMax = Math.max(pcMax, v); }
  // q vs ν n and vs the cluster Gram (identical members)
  const nun = nu.map((v, k) => v * sizes[k]).sort((x, y) => y - x).slice(0, rr);
  const Gc = new Float64Array(K * K);
  for (let x = 0; x < K; x++) for (let y = 0; y < K; y++) Gc[x * K + y] = Math.sqrt(nu[x] * sizes[x] * nu[y] * sizes[y]) * colDot(Pu, K, x, Pu, K, y, Se);
  const qGram = Array.from(O.eigh(Gc, K).values.slice(0, rr));
  const q = Array.from(pm.q.slice(0, rr));
  const qSum = q.reduce((a, x) => a + x, 0) || 1;
  const relErr = arr => arr.reduce((a, x, i) => a + Math.abs(q[i] - x), 0) / qSum;
  const out = {
    spec, K, Km, sizes, nu, rr, cosP, cosE, bestP, bestE, nModes, lam: Array.from(lam),
    protoCos: np ? pc / np : 0, protoCosMax: pcMax, q, nun, qGram, errNun: relErr(nun), errGram: relErr(qGram),
    diamMax: multi.length ? Math.max(...multi.map(k => cs[k].diamMax)) : 0,
    protoSd: multi.length ? multi.reduce((a, k) => { let m = 0, q2 = 0; const p = cs[k].proto; for (let s = 0; s < Se; s++) { m += p[s]; q2 += p[s] * p[s]; } m /= Se; return a + Math.sqrt(Math.max(0, q2 / Se - m * m)); }, 0) / multi.length : 0,
    residual: sim.residual, eqResidual: sim.eqResidual, steps: sim.steps,
  };
  if (spec.pot === "linear") {
    const H = O.harmonic(net), Kmat = new Float64Array(B * B);
    for (let i = 0; i < N; i++) for (let p = 0; p < B; p++) for (let q2 = 0; q2 < B; q2++) Kmat[p * B + q2] += H[i * B + p] * H[i * B + q2];
    out.comm = commRel(Kmat, Sig, B);
    // exact identity: H Σ Hᵀ and Σ^{1/2} K Σ^{1/2} share their nonzero spectrum
    const HS = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { let v = 0; for (let p = 0; p < B; p++) { let w = 0; for (let q2 = 0; q2 < B; q2++) w += Sig[p * B + q2] * H[j * B + q2]; v += H[i * B + p] * w; } HS[i * N + j] = v; }
    const e1 = O.eigh(HS, N).values.slice(0, B);
    const Sh = O.sqrtm(Sig, B), M = new Float64Array(B * B), T = new Float64Array(B * B);
    for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) { let v = 0; for (let k = 0; k < B; k++) v += Sh[i * B + k] * Kmat[k * B + j]; T[i * B + j] = v; }
    for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) { let v = 0; for (let k = 0; k < B; k++) v += T[i * B + k] * Sh[k * B + j]; M[i * B + j] = v; }
    for (let i = 0; i < B; i++) for (let j = i + 1; j < B; j++) { const v = (M[i * B + j] + M[j * B + i]) / 2; M[i * B + j] = v; M[j * B + i] = v; }
    const e2 = O.eigh(M, B).values;
    let d = 0; for (let k = 0; k < B; k++) d = Math.max(d, Math.abs(e1[k] - e2[k]));
    out.identityErr = d / e2[0];
    // sampling: evaluation-bank population spectrum against the analytic one
    let ds = 0; for (let k = 0; k < Math.min(B, pm.q.length); k++) ds = Math.max(ds, Math.abs(pm.q[k] - e2[k]));
    out.samplingErr = ds / e2[0];
  }
  return out;
}
