// Benchmark of the live loop in site/js/onsite_engine.js.
// Usage: node tests/onsite_bench.mjs [--quick]
// For each configuration: build a net and a bank, then time sim.step() from W = 0 for a
// fixed number of steps (the transient, where clusters form) and again near the fixed
// point; prints steps/second, Cholesky factorisations per step and the worst equilibrium
// residual seen.  Also times the slow exact path for comparison at N = 40.
import * as O from '../site/js/onsite_engine.js';

const quick = process.argv.includes('--quick');
const now = () => performance.now();

function setup({ N, B, S = 256, klo, khi, sigmaT, s = 3, seed = 1, sigma = 'blocks' }) {
  const rng = O.makeRng(seed);
  const Sigma = sigma === 'blocks'
    ? O.covBlocks({ sizes: Array(4).fill(B / 4), within: 0.8, between: 0.0, variances: [2, 1, 0.5, 0.25] }).Sigma
    : O.covPowerLaw({ B, alpha: 1, seed });
  const net = O.makeNet({ N, B, rng, s, klo, khi, sigmaT });
  const G = O.sampleInputs(Sigma, B, S, rng);
  return { net, G, S };
}

function bench(cfg, { b, alpha = 1, dt = 0.2, steps = 60, label }) {
  const { net, G, S } = setup(cfg);
  const t0 = now();
  const sim = O.createSim({ net, G, S, b, alpha, dt });
  const tInit = now() - t0;
  let fact = 0, worst = 0;
  const t1 = now();
  for (let k = 0; k < steps; k++) { sim.step(1); fact += sim.lastSolve.factorizations; worst = Math.max(worst, sim.eqResidual); }
  const tTrans = now() - t1;
  // run on towards the fixed point (untimed), then time steps there
  for (let k = 0; k < (quick ? 100 : 400) && sim.lastChange > 1e-7; k++) sim.step(1);
  const comps = O.components(net.W, net.N).K;
  let fact2 = 0;
  const t2 = now();
  for (let k = 0; k < steps; k++) { sim.step(1); fact2 += sim.lastSolve.factorizations; worst = Math.max(worst, sim.eqResidual); }
  const tFix = now() - t2;
  const f = x => x.toFixed(1).padStart(7);
  console.log(`${label.padEnd(44)} init ${tInit.toFixed(0).padStart(4)} ms | transient ${f(1000 * steps / tTrans)} steps/s (${(fact / steps).toFixed(0).padStart(4)} chol/step) | near fixed pt ${f(1000 * steps / tFix)} steps/s (${(fact2 / steps).toFixed(0).padStart(4)} chol/step, K=${String(comps).padStart(2)}) | max eq resid ${worst.toExponential(1)}`);
}

console.log(`node ${process.version}; S = 256, B = 24, dt = 0.2, alpha = 1; ${quick ? 'quick' : 'full'} run`);
const steps = quick ? 15 : 60;
for (const N of [40, 80]) {
  for (const [name, nl] of [['linear', { klo: 0, khi: 0, sigmaT: 0 }],
    ['rectifying klo=3 khi=0.5 sigma_t=0.7', { klo: 3, khi: 0.5, sigmaT: 0.7 }],
    ['strong klo=10 khi=0 sigma_t=1', { klo: 10, khi: 0, sigmaT: 1 }]]) {
    for (const b of [0.05, 0.3, 2]) bench({ N, B: 24, ...nl }, { b, steps, label: `N=${N} ${name} b=${b}` });
  }
}

// exact (reference-port) path for comparison
{
  const { net, G, S } = setup({ N: 40, B: 24, klo: 3, khi: 0.5, sigmaT: 0.7 });
  const sim = O.createSim({ net, G, S, b: 0.3, alpha: 1, dt: 0.2, solver: { exact: true } });
  const n = quick ? 3 : 10, t0 = now();
  sim.step(n);
  console.log(`N=40 rectifying b=0.3, EXACT per-sample path: ${(1000 * n / (now() - t0)).toFixed(1)} steps/s`);
}
