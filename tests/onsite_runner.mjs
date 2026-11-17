// Node-side driver for tests/test_onsite_parity.py.
// Usage: node onsite_runner.mjs input.json  → prints a JSON result to stdout.
// Input: {op, ...args}.  Matrices arrive and leave as nested lists; a net is
// {A (N×B), t, klo, khi (N), W (N×N)}; an input bank G is S×B.
import { readFileSync } from 'node:fs';
import * as O from '../site/js/onsite_engine.js';

const inp = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const flat = a => Float64Array.from(a.flat(Infinity));
const mat = (x, r, c) => { const out = []; for (let i = 0; i < r; i++) out.push(Array.from(x.subarray(i * c, i * c + c))); return out; };
const loadNet = n => O.makeNetFrom({ N: n.A.length, B: n.A[0].length, A: flat(n.A), t: n.t, klo: n.klo, khi: n.khi, W: n.W ? flat(n.W) : null });
const bank = G => ({ G: flat(G), S: G.length });

const ops = {
  rng({ seed, n }) {
    const r = O.makeRng(seed);
    return { uniform: Array.from({ length: n }, () => r.uniform()), normal: Array.from({ length: n }, () => r.normal()),
      laplace: Array.from({ length: n }, () => r.laplace()) };
  },
  make_net({ N, B, seed, s, deg, klo, khi, sigmaT, equal }) {
    const net = O.makeNet({ N, B, rng: O.makeRng(seed), s, deg, klo, khi, sigmaT, equal });
    return { A: mat(net.A, N, B), t: Array.from(net.t) };
  },
  sample({ Sigma, S, seed, kind }) {
    const B = Sigma.length;
    return { G: mat(O.sampleInputs(flat(Sigma), B, S, O.makeRng(seed), kind), S, B) };
  },
  gen({ name, args }) {
    const r = O[name](args);
    const S = r.Sigma || r;
    const B = Math.round(Math.sqrt(S.length));
    return { Sigma: mat(S, B, B) };
  },
  equilibrate({ net, G, U0, tol, exact }) {
    const n = loadNet(net), { G: g, S } = bank(G);
    const e = O.equilibrate(n, g, S, { U0: U0 ? flat(U0) : null, tol, exact });
    return { U: mat(e.U, S, n.N), residual: e.residual, iters: e.iters, factorizations: e.factorizations, updates: e.updates || 0, components: e.components };
  },
  analyze({ net, G, b, alpha, eps }) {
    const n = loadNet(net), { G: g, S } = bank(G), N = n.N;
    const e = O.equilibrate(n, g, S);
    const U = e.U;
    const { dW, D2 } = O.drift(n, g, S, b, alpha, U);
    const comp = O.components(n.W, N, eps);
    const st = O.clusterStats(U, S, N, comp.labels, comp.K);
    const pm = O.populationModes(U, S, N);
    const md = O.mds(D2, N, 3);
    return {
      U: mat(U, S, N), eqResidual: e.residual, D2: mat(D2, N, N), dW: mat(dW, N, N),
      residual: O.residual(n, g, S, b, alpha, U), energy: O.energy(n, U, g, S),
      L: mat(O.laplacian(n), N, N), H: mat(O.harmonic(n), N, n.B),
      labels: Array.from(comp.labels), K: comp.K,
      stats: st.map(c => ({ members: Array.from(c.members), size: c.size, proto: Array.from(c.proto), diam_max: c.diamMax, diam_mean: c.diamMean, d_eff: c.dEff })),
      modes: { q: Array.from(pm.q), V: mat(pm.V, N, N), Psi: mat(pm.Psi, S, pm.r), d_eff: pm.dEff, C: mat(pm.C, N, N) },
      mds: { X: mat(md.X, N, 3), values: Array.from(md.values), explained: md.explained, negMass: md.negMass },
    };
  },
  trajectory({ net, G, b, alpha, dt, nSteps, record }) {
    // Live loop: sim.step() one step at a time, snapshots of W at the requested steps.
    const n = loadNet(net), { G: g, S } = bank(G), N = n.N;
    const sim = O.createSim({ net: n, G: g, S, b, alpha, dt });
    const snaps = {}, eqRes = [];
    for (let k = 1; k <= nSteps; k++) {
      sim.step(1); eqRes.push(sim.eqResidual);
      if (record.includes(k)) snaps[k] = mat(n.W, N, N);
    }
    return { snaps, W: mat(n.W, N, N), residual: sim.residual, maxEqResidual: Math.max(...eqRes), U: mat(sim.U, S, N) };
  },
  run({ net, G, b, alpha, dt, T, tol, exact }) {
    const n = loadNet(net), { G: g, S } = bank(G), N = n.N;
    const r = O.run(n, g, S, b, alpha, { dt, T, tol, solver: { exact } });
    return { W: mat(n.W, N, N), steps: r.steps, residual: r.residual, eqResidual: r.eqResidual, U: mat(r.U, S, N) };
  },
  mds({ D2, dim, prev }) {
    const n = D2.length, r = O.mds(flat(D2), n, dim, prev ? flat(prev) : null);
    return { X: mat(r.X, n, dim), values: Array.from(r.values), explained: r.explained, negMass: r.negMass };
  },
};

process.stdout.write(JSON.stringify(ops[inp.op](inp)));
