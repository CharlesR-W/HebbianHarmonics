// Chapter "idea": the model side of the live figure (no DOM, so it can be checked under node).
// One small network: B = 3 clamped inputs, N = 8 interior neurons, every neuron attached to all three inputs with
// random shares (total conductance 1), two-slope on-site potentials, a training bank of S = 256 unit-variance
// Gaussian stimuli. Single-stimulus relaxation (the fast dynamics, integrated explicitly for display), exact
// equilibria (onsite_engine.js), the slow rule, and the envelope identity by finite differences.
import * as O from "./onsite_engine.js";

export const N = 8, B = 3, S = 256;
export const DEFAULTS = { g1: 1.2, g2: -0.9, g3: 0.4, b: 0.05, a: 1, pot: "rect", k: 3, st: 0.7, seed: 27 };
export const LIMITS = { g: [-2.5, 2.5], b: [0.005, 0.5], a: [0.05, 5] };
export const FD_STEP = 1e-4;            // central-difference step for the envelope check
export const STOP_TOL = 1e-10;          // the run pauses when max|dW|/dt falls below this
export const CLAIM_TOL = 1e-6;          // fixed point claimed when the plasticity residual is below this

export const TOURS = [
  { id: "relax", label: "One stimulus, relaxing",
    note: "$W = 0$, leaky rectifiers. The voltages start at $u = 0$ and flow downhill in $E$ (fast time $\\tau$) to the unique minimum; in (b) the selected neuron's dot moves onto its transfer curve.",
    params: { g1: 1.5, g2: -1.2, g3: 0.3, b: 0.05, a: 1, pot: "rect", k: 3, st: 0.7, seed: 27 }, relax: true },
  { id: "linear", label: "Linear neurons",
    note: "$V_i = 0$: every transfer curve in (b) is the straight line $u = I/d_i$, and each response is linear in the input, $F_i(g) = h_i \\cdot g$. At $W = 0$, $h_i$ is just the neuron's attachment shares.",
    params: { g1: 1.5, g2: -1.2, g3: 0.3, b: 0.05, a: 1, pot: "lin", k: 3, st: 0.7, seed: 27 }, relax: true },
  { id: "learn", label: "Learn to the fixed point",
    note: "Leaky rectifiers, $b = 0.05$, $\\alpha = 1$, from $W = 0$. When the run stops, every pair in (d) is on the line $w = (b - D^2/2)_+/\\alpha$; the envelope identity (e) holds at every step on the way.",
    params: { b: 0.05, a: 1, pot: "rect", k: 3, st: 0.7, seed: 27 }, run: true, speed: 0.25 },
  { id: "fuse", label: "Strong fusion",
    note: "$b = 0.3$, $\\alpha = 0.1$: every pair comes within range, the edges grow to several times the attachment conductance, and all eight neurons end in one component; the pair in (c) lies close to the diagonal.",
    params: { b: 0.3, a: 0.1, pot: "rect", k: 3, st: 0.7, seed: 27 }, run: true, speed: 16 },
];

const SIGMA = O.covDiag([1, 1, 1]);

/** Build the network and its training bank. Thresholds are t_i = st * z_i with z_i fixed by the seed. */
export function build(p) {
  const rng = O.makeRng(p.seed);
  const net = O.makeNet({ N, B, rng, s: 3, deg: 1, klo: 0, khi: 0, sigmaT: 1 });
  const z = Float64Array.from(net.t);
  const G = O.sampleInputs(SIGMA, B, S, O.makeRng(1000 + p.seed));
  const m = { net, z, G, p: { ...p } };
  setPotential(m, p);
  return m;
}

export function setPotential(m, p) {
  const k = p.pot === "lin" ? 0 : p.k;
  for (let i = 0; i < N; i++) { m.net.klo[i] = k; m.net.khi[i] = 0; m.net.t[i] = p.st * m.z[i]; }
  m.p.pot = p.pot; m.p.k = p.k; m.p.st = p.st;
}

export const dtOf = alpha => Math.min(0.5, 1 / alpha);

export function createSim(m, p) {
  return O.createSim({ net: m.net, G: m.G, S, b: p.b, alpha: p.a, dt: dtOf(p.a) });
}

// ------------------------------------------------------------ one neuron
export const Vof = (net, i, u) => { const x = u - net.t[i]; return 0.5 * (x < 0 ? net.klo[i] : net.khi[i]) * x * x; };
export const dVof = (net, i, u) => { const x = u - net.t[i]; return (x < 0 ? net.klo[i] : net.khi[i]) * x; };
export function degree(net, i) {
  let d = 0;
  for (let j = 0; j < N; j++) if (j !== i) d += net.W[i * N + j];
  for (let b = 0; b < B; b++) d += net.A[i * B + b];
  return d;
}
/** Input current into neuron i: sum_j w_ij u_j + sum_b a_ib g_b. */
export function current(net, u, g, i) {
  let I = 0;
  for (let j = 0; j < N; j++) if (j !== i) I += net.W[i * N + j] * u[j];
  for (let b = 0; b < B; b++) I += net.A[i * B + b] * g[b];
  return I;
}
/** dE/du_i = d_i u_i + V_i'(u_i) - I_i (the node-equation residual). */
export function gradE(net, u, g, out = new Float64Array(N)) {
  for (let i = 0; i < N; i++) out[i] = degree(net, i) * u[i] + dVof(net, i, u[i]) - current(net, u, g, i);
  return out;
}
export const maxAbs = v => { let m = 0; for (const x of v) m = Math.max(m, Math.abs(x)); return m; };

/** Exact equilibrium for one stimulus g (length B). */
export function equilibriumOne(net, g) {
  const e = O.equilibrate(net, Float64Array.from(g), 1);
  return Float64Array.from(e.U);
}

/** Advance the fast dynamics du/dtau = -dE/du by dtau (explicit Euler substeps, stable step). In place. */
export function relax(net, u, g, dtau) {
  let lmax = 0;
  for (let i = 0; i < N; i++) lmax = Math.max(lmax, 2 * degree(net, i) + Math.max(net.klo[i], net.khi[i]));
  const hmax = 0.5 / lmax, n = Math.max(1, Math.ceil(dtau / hmax)), h = dtau / n, gr = new Float64Array(N);
  for (let k = 0; k < n; k++) { gradE(net, u, g, gr); for (let i = 0; i < N; i++) u[i] -= h * gr[i]; }
  return u;
}

// ------------------------------------------------------------ pairs, the rule, the envelope identity
export const PAIRS = (() => { const P = []; for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) P.push([i, j]); return P; })();

/** Mean equilibrium energy over the bank, E*(W). */
export function meanEnergy(net, G) {
  const e = O.equilibrate(net, G, S);
  return O.energy(net, e.U, G, S);
}

/** Central differences of E*(W) in each w_ij (both entries of the symmetric W moved), against D_ij^2/2.
 *  Returns {fd (per pair), half (D^2/2 per pair), err: max |fd - D^2/2|}. W is restored exactly. */
export function envelopeCheck(net, G, D2, h = FD_STEP) {
  const W = net.W, fd = new Float64Array(PAIRS.length), half = new Float64Array(PAIRS.length);
  let err = 0;
  PAIRS.forEach(([i, j], q) => {
    const w0 = W[i * N + j];
    W[i * N + j] = W[j * N + i] = w0 + h; const ep = meanEnergy(net, G);
    W[i * N + j] = W[j * N + i] = w0 - h; const em = meanEnergy(net, G);
    W[i * N + j] = W[j * N + i] = w0;
    fd[q] = (ep - em) / (2 * h); half[q] = D2[i * N + j] / 2;
    err = Math.max(err, Math.abs(fd[q] - half[q]));
  });
  return { fd, half, err };
}

/** Fixed-point bookkeeping: largest |alpha w - (b - D^2/2)| over live edges, smallest D^2 - 2b over dead pairs. */
export function fixedPointErrors(net, D2, b, alpha) {
  let dev = 0, slack = Infinity, live = 0;
  for (const [i, j] of PAIRS) {
    const w = net.W[i * N + j], d2 = D2[i * N + j];
    if (w > 1e-9) { live++; dev = Math.max(dev, Math.abs(alpha * w - (b - d2 / 2))); } else slack = Math.min(slack, d2 - 2 * b);
  }
  return { dev, slack, live };
}

/** The pair with the smallest D^2 (a natural default selection). */
export function closestPair(D2) {
  let best = PAIRS[0], bd = Infinity;
  for (const [i, j] of PAIRS) if (D2[i * N + j] < bd) { bd = D2[i * N + j]; best = [i, j]; }
  return best;
}
