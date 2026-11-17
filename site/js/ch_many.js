// Chapter 2 "many": many boundary neurons, synthetic input statistics. A live simulation with the shared engine
// (mean-field by default, minibatch optional), four linked views of embeddings that live in a (B−1)-simplex, and a
// live weight-space readout of Σ_g computed in a worker (ch_many_worker.js) from the current weights alone.
import * as ui from "./ui.js";
import { heatmap, lineChart, barMultiples } from "./charts.js";
import { view3d } from "./view3d.js";
import { P as PAL, catColor } from "./palette.js";
import * as E from "./engine.js";

// ------------------------------------------------------------------------------------------------ constants
const BS = [8, 12, 16, 24, 32, 48];
const FAMS = [
  { value: "blocks", label: "planted blocks" },
  { value: "hier", label: "hierarchical blocks" },
  { value: "pow", label: "power-law spectrum, signed" },
  { value: "powpos", label: "power-law spectrum, correlations ≥ 0" },
  { value: "ring", label: "ring (stationary, Fourier eigenvectors)" },
  { value: "wish", label: "Wishart (random sample covariance)" },
];
const HIER = { 8: [2, 2, 2], 12: [2, 2, 3], 16: [2, 2, 4], 24: [2, 3, 4], 32: [2, 4, 4], 48: [3, 4, 4] };
const HIER_CORR = [0.05, 0.3, 0.6];
const DEF = { fam: "blocks", b: 24, theta: 0.3, p: 1, n: 150, gamma: 1, wmin: 1e-4, alpha: 1, dofr: 1, nb: 4, win: 0.5, btw: 0.1,
  uneq: false, cs: 1, seed: 1, mode: "mf", batch: 32, cont: false, ell: 1.5, jit: false };
const II_SCALE = 0.03;    // initial lateral weights ~ U(0, 0.03) (docs/findings.md default; ≤ 0.03 behaves like the floor)
const DT = 0.05;          // Euler step (docs/findings.md default; the readout's mid-training peak needs the finer path)
const JITTER = 1e-3;      // optional private multiplicative jitter per step, w ← w·exp(σξ)
const KEYS = Object.keys(DEF);
const EQ_TOL = 1e-6;      // the equilibrium readout is claimed only when r/η < EQ_TOL
const DONE_TOL = 1e-8;    // the run pauses ("converged") when r/η < DONE_TOL
const SEQ_TOL = 1e-6;
const EARLY_STEPS = 400, EARLY_EVERY = 10;   // readout at least every 10 steps over the first 400 (mean-field)     // a tour step counts as converged at r/η < SEQ_TOL (docs/findings.md uses 1e-5)
const SPEC_TOL = 1e-5;    // a direction of Sym(B) counts as pinned down when its AᵀA eigenvalue > SPEC_TOL × largest
const MAX_EDGES = 4000;
const sub = n => String(n).replace(/\d/g, d => "₀₁₂₃₄₅₆₇₈₉"[d]);
const gName = b => `g${sub(b + 1)}`;
const f3 = x => ui.fmtFixed(x, 3), f2 = x => ui.fmtFixed(x, 2);
const sci = x => (Number.isFinite(x) ? (x === 0 ? "0" : ui.fmt(x, 2)) : "n/a");

// Tours: exact states (every key), verified with the engine (see the chapter's report). `seq` steps θ through a list,
// each step run to convergence; `cont` keeps the weights between steps (otherwise each restarts from the same init).
const TOURS = [
  { id: "sweep", group: "c", label: "θ sweep on blocks", params: { fam: "blocks", b: 24, nb: 4, win: 0.5, btw: 0.1, theta: 0.05, p: 1, n: 150, seed: 1 }, seq: [0.05, 0.15, 0.3, 0.6],
    note: "Four planted blocks of six inputs (within 0.5, between 0.1; $\\gamma\\kappa_0 = 0.22$). Each step restarts from the same initial weights at a larger $\\theta$: $0.05 \\to$ one cluster on all 24 inputs; $0.15 \\to$ one cluster on ONE block (four block parts exist, the bound population takes one); $0.3 \\to$ one cluster per block; $0.6 \\to$ one cluster per input." },
  { id: "bound", group: "c", label: "Bound: four parts, one cluster", params: { fam: "blocks", b: 24, nb: 4, win: 0.5, btw: 0.1, theta: 0.15, p: 1, n: 150, seed: 1 },
    note: "The same blocks at $\\theta = 0.15$: above the between-block correlation, so each block is a stable part, but below $\\gamma\\kappa_0 = 0.22$, so every lateral synapse grows at first. The population moves as one blob and all 150 neurons land on a single block. <a href=\"#many?tour=unbound\">With $\\gamma = 0$</a> (no lateral plasticity) neurons choose independently: four clusters." },
  { id: "unbound", hidden: true, group: "c", label: "γ = 0 at θ = 0.15", params: { fam: "blocks", b: 24, nb: 4, win: 0.5, btw: 0.1, theta: 0.15, gamma: 0, p: 1, n: 150, seed: 1 },
    note: "The bound tour without lateral plasticity ($\\gamma = 0$): nothing gates the population, and every block gets a cluster." },
  { id: "signed", group: "c", label: "Signed Σ at θ = 0", params: { fam: "pow", b: 16, alpha: 1.5, cs: 2, theta: 0, p: 1, n: 150, seed: 1 },
    note: "Power-law spectrum with signed correlations and no pruning drift ($\\theta = 0$): synapses die only where covariance is negative. Two clusters on disjoint sets of inputs. <a href=\"#many?tour=signedpos\">Same spectrum, all correlations ≥ 0</a>: one cluster." },
  { id: "signedpos", hidden: true, group: "c", label: "Σ ≥ 0 at θ = 0", params: { fam: "powpos", b: 16, alpha: 1.5, cs: 2, theta: 0, p: 1, n: 150, seed: 1 },
    note: "The signed tour's $\\Sigma_g$ plus the smallest $c\\,\\mathbf{1}\\mathbf{1}^\\top$ that makes every correlation ≥ 0, still at $\\theta = 0$: nothing is pruned and the whole population forms one cluster." },
  { id: "takeall", group: "c", label: "One block takes everything", params: { fam: "blocks", b: 24, nb: 8, win: 0.6, btw: 0.05, theta: 0.1, p: 1, n: 150, seed: 1 },
    note: "Eight blocks of three (within 0.6, between 0.05), $\\theta = 0.1$: above the between-block correlation but below $\\kappa_0 \\approx 0.135$, the mean covariance of the initial neurons. The population moves as one and settles on a single block; seven blocks get no neurons." },
  { id: "tree", group: "c", label: "θ walks down a tree", params: { fam: "hier", b: 16, theta: 0.05, p: 1, n: 150, seed: 1 }, seq: [0.05, 0.15, 0.25, 0.35, 0.45, 0.65],
    note: "Hierarchical $\\Sigma_g$: 2 groups of 2 groups of 4 inputs, correlations 0.05 / 0.3 / 0.6 across / between sibling groups / within. Restarting at each $\\theta$: one cluster on all 16 → one on a top-level half → the two halves → one leaf group per half → all four leaf groups → one cluster per input (plus a few neurons that lose every synapse and float to the centroid)." },
  { id: "hyst", group: "c", label: "…without restarting", params: { fam: "hier", b: 16, theta: 0.05, p: 1, n: 150, seed: 1 }, seq: [0.05, 0.15, 0.25, 0.35, 0.45, 0.65], cont: true,
    note: "The same $\\theta$ steps, but keeping the weights between steps. The population never splits: one cluster narrows down a single branch of the tree, all 16 inputs → 8 → 4 → 2 → a single input." },
  { id: "ringfree", group: "c", label: "Ring: place fields", params: { fam: "ring", b: 16, ell: 1.5, theta: 0.4, gamma: 0, p: 1, n: 160, seed: 1 },
    note: "Ring covariance ($\\ell = 1.5$): its eigenvectors are Fourier modes, signed and spread round the ring. Without lateral plasticity ($\\gamma = 0$), $\\theta = 0.4$: 16 clusters, one bump of width 4 centred on each input. Localised place fields, not Fourier modes." },
  { id: "ringpack", group: "c", label: "Ring: packing", params: { fam: "ring", b: 16, ell: 1.5, theta: 0.4, gamma: 1, p: 1, n: 160, seed: 1 },
    note: "The same with $\\gamma = 1$: two clusters may coexist only if their cross-covariance $h_k^\\top\\Sigma_g h_l \\le \\theta/\\gamma$. Adjacent bumps overlap too much (0.645 > 0.4), so only a few non-overlapping bumps survive." },
  { id: "dense", group: "r", label: "Readout: dense wiring", params: { fam: "blocks", b: 24, nb: 4, win: 0.5, btw: 0.1, theta: 0.3, p: 1, n: 150, seed: 1 },
    note: "Default blocks, every potential synapse present ($p = 1$). Four clusters, every member a copy of its cluster's part: the weights pin down 24 of the 300 directions of $\\Sigma_g$." },
  { id: "sparse", group: "r", label: "Readout: sparse wiring", params: { fam: "blocks", b: 24, nb: 4, win: 0.5, btw: 0.1, theta: 0.3, p: 0.5, n: 150, seed: 1 },
    note: "The same, but each potential synapse exists with probability $p = 0.5$ (a fixed random mask). Still four clusters, but members differ, and the weights pin down far more of $\\Sigma_g$: contrast-subspace cosine ≈ 0.99." },
];

// ------------------------------------------------------------------------------------------------ input statistics
function blockSizes(B, nb, uneq, cs) {
  if (!uneq) { const q = Math.floor(B / nb), r = B % nb; return Array.from({ length: nb }, (_, k) => q + (k < r ? 1 : 0)); }
  const rng = E.makeRng(1000 + 97 * cs + nb);
  const w = Array.from({ length: nb }, () => Math.exp(0.7 * rng.normal()));
  const tot = w.reduce((a, b) => a + b, 0);
  const s = w.map(x => Math.max(2, Math.round(B * x / tot)));
  let d = B - s.reduce((a, b) => a + b, 0);
  while (d !== 0) { const k = d > 0 ? s.indexOf(Math.min(...s)) : s.indexOf(Math.max(...s)); s[k] += Math.sign(d); d -= Math.sign(d); }
  return s.sort((a, b) => b - a);
}
function makeSigma(st) {
  const B = st.b;
  let Sigma, labels = null, groups = null, desc = "";
  if (st.fam === "blocks") {
    const nb = Math.min(st.nb, Math.floor(B / 2));
    const sizes = blockSizes(B, nb, st.uneq, st.cs);
    let variances = 1;
    if (st.uneq) {
      const rng = E.makeRng(5000 + 31 * st.cs + nb);
      const v = sizes.map(() => Math.exp(0.45 * rng.normal()));
      const mean = v.reduce((a, x, k) => a + x * sizes[k], 0) / B;
      variances = v.map(x => x / mean);
    }
    ({ Sigma, labels } = E.covBlocks({ sizes, within: st.win, between: st.btw, variances }));
    groups = labels;
    desc = `${nb} planted blocks (sizes ${sizes.join(", ")}), correlation ${f2(st.win)} within, ${f2(st.btw)} between${st.uneq ? ", unequal variances" : ""}`;
  } else if (st.fam === "hier") {
    const br = HIER[B];
    ({ Sigma, labels } = E.covHierarchical({ branching: br, corr: HIER_CORR, variance: 1 }));
    const leafParent = br[br.length - 1];
    groups = Int32Array.from({ length: B }, (_, i) => Math.floor(i / leafParent));
    desc = `hierarchical blocks, branching ${br.join(" × ")}, correlations ${HIER_CORR.join(" / ")} (root / middle / siblings)`;
  } else if (st.fam === "pow" || st.fam === "powpos") {
    Sigma = E.covPowerLaw({ B, alpha: st.alpha, seed: st.cs, nonneg: st.fam === "powpos" });
    desc = `random orthogonal eigenbasis, eigenvalues ∝ k^−${f2(st.alpha)}${st.fam === "powpos" ? ", plus the smallest c·11ᵀ making all correlations ≥ 0" : ""}, mean variance 1`;
  } else if (st.fam === "ring") {
    Sigma = new Float64Array(B * B);
    for (let b = 0; b < B; b++) for (let c = 0; c < B; c++) { let d = Math.abs(b - c); d = Math.min(d, B - d); Sigma[b * B + c] = Math.exp(-d * d / (2 * st.ell * st.ell)); }
    labels = null; desc = `ring: (Σ_g)_bc = exp(−d(b,c)²/2ℓ²) with d the distance around a ring of ${B} inputs, ℓ = ${f2(st.ell)}`;
  } else {
    const dof = Math.max(1, Math.round(st.dofr * B));
    Sigma = E.covWishart({ B, dof, seed: st.cs });
    desc = `Wishart: sample covariance of ${dof} standard normal draws (rank ${Math.min(dof, B)}), mean variance 1`;
  }
  // display order of boundary neurons: planted order, or by angle in the plane of the top two eigenvectors
  const eg = E.eigh(Sigma, B);
  let order = Array.from({ length: B }, (_, i) => i);
  if (!labels && st.fam !== "ring") {
    const ang = order.map(b => Math.atan2(eg.vectors[b * B + 1], eg.vectors[b * B]));
    order.sort((a, c) => ang[a] - ang[c]);
  }
  let m1 = 0, md = 0;
  for (let k = 0; k < B * B; k++) m1 += Sigma[k];
  for (let b = 0; b < B; b++) md = Math.max(md, Sigma[b * B + b]);
  // k for the subspace score: the largest eigen-gap among the first few eigenvalues of PΣP (else 3)
  const ep = E.eigh(E.project1(Sigma, B), B).values;
  let k = 3, best = 1.5;
  for (let q = 1; q <= Math.min(6, B - 2); q++) { const r = ep[q - 1] / Math.max(ep[q], 1e-12); if (r > best) { best = r; k = q; } }
  return { Sigma, B, labels, groups, order, eig: eg, kappa0: m1 / (B * B), maxVar: md, k, desc };
}

// ------------------------------------------------------------------------------------------------ module
export async function init(panel, ctx) {
  // ---------- state from the URL (a tour id wins)
  let st = { ...DEF }, tour = null;
  const readState = params => {
    const id = params.get("tour"); const t = TOURS.find(x => x.id === id);
    if (t) return { st: { ...DEF, ...t.params }, tour: t };
    const o = { ...DEF };
    for (const k of KEYS) {
      if (!params.has(k)) continue;
      const v = params.get(k), d = DEF[k];
      o[k] = typeof d === "number" ? +v : typeof d === "boolean" ? (v === "1" || v === "true") : v;
    }
    if (!BS.includes(o.b)) o.b = DEF.b;
    o.n = Math.max(60, Math.min(250, Math.round(o.n)));
    if (!FAMS.some(f => f.value === o.fam)) o.fam = DEF.fam;
    return { st: o, tour: null };
  };
  ({ st, tour } = readState(ctx.params));
  const writeURL = () => {
    const o = {};
    for (const k of KEYS) o[k] = st[k] === DEF[k] ? null : st[k];
    ctx.setParams({ ...o, tour: null });
  };

  // ---------- simulation state
  let SG = makeSigma(st), Sigma = SG.Sigma, B = SG.B, N = st.n;
  const Pm = () => ({ ...E.defaultParams(), theta: st.theta, gamma: st.gamma, wmin: st.wmin, dt: DT });
  let PP = Pm();
  let jrng = null;
  let s = null, sampler = null, prevWii = null, prevWib = null, prevValid = false, runId = 0, version = 0;
  let pca = null, H = null, cl = null, colourOf = null, neuronColour = null, clusterColour = [], cstats = null;
  let tOff = 0, seq = null, seqIdx = 0, holdT = 0, seqSteps = 0, frameNo = 0, mbResid = NaN;
  const series = { t: [], K: [], er: [], r: [], rt: [], cosC: [], cosK: [], cosT: [], cosP: [], tr: [] };
  const seqMarks = [];
  let edgeBuf = null;

  const simTime = () => tOff + (s ? s.t * PP.dt : 0);
  const residual = () => (st.mode === "mf" ? (s ? s.resid / PP.eta : NaN) : mbResid / PP.eta);

  function newState() {
    N = st.n; PP = Pm();
    s = E.createState({ N, B, seed: st.seed, p: st.p, iiScale: II_SCALE });
    jrng = E.makeRng(7000 + st.seed);
    sampler = st.mode === "mb" ? E.makeSampler(Sigma, B, 1000 + st.seed) : null;
    prevWii = new Float64Array(N * N); prevWib = new Float64Array(N * B); prevValid = false;
    edgeBuf = { pairs: new Uint32Array(2 * (N * (N - 1) / 2 + N * B)), w: new Float32Array(N * (N - 1) / 2 + N * B) };
    H = E.harmonic(s).slice();
    pca = null; neuronColour = new Int8Array(N).fill(-1); mbResid = NaN; lastDispatchStep = 0;
    runId++; version++;
    postRun();
  }

  // ---------- figure 1: the simulation
  const live = ctx.slot("live");
  const L = ui.simLayout({ parent: ctx.slot("sim") });
  const v = view3d(L.view, { height: 480, zoom: 0.88, center: [0, 0, 0], hull: false, pointSize: 5, edgeAlpha: 0.5, hint: true,
    tooltip: (i, kind) => tipFor(i, kind) });
  let tick = 0, fps = 0, fpsT = performance.now(), fpsN = 0;
  const tp = ui.transport({
    parent: L.view, watch: live, speeds: [{ value: 1, label: "×1" }, { value: 4, label: "×4" }, { value: 16, label: "×16" }], speed: 1,
    autoplay: true, maxHistory: 240, timeFmt: t => `t = ${ui.fmt(t, 3)}`,
    onTick: speed => onTick(speed),
    onSeek: f => showFrame(f),
    onReset: () => { tOff = 0; seq = tour && tour.seq ? { list: tour.seq, cont: !!tour.cont } : null; seqIdx = 0; holdT = 0; seqSteps = 0;
      clearSeries(); newState(); analyse(); drawAll(true, true); return frameOf(); },
  });

  // side: tours, controls, stats, legend
  const visibleTours = TOURS.filter(t => !t.hidden);
  const chipsC = ui.chips({ parent: L.side, label: "Tours", items: visibleTours.filter(t => t.group === "c"), active: tour && tour.group === "c" ? tour.id : null, onPick: t => pickTour(t) });
  const famS = ui.select({ label: "input covariance $\\Sigma_g$", options: FAMS, value: st.fam, stack: true, onChange: x => change({ fam: x }) });
  const bS = ui.segmented({ label: "boundary neurons $B$", options: BS, value: st.b, stack: true, onChange: x => change({ b: x }) });
  let thetaS = null;
  const thetaHost = ui.h("div");
  const makeTheta = () => {
    // marks in priority order; a mark too close to an earlier one is dropped so labels never collide
    const gk = st.gamma * SG.kappa0;
    const marks = st.gamma === 1 ? [{ value: SG.kappa0, label: "γκ₀", title: "collective threshold γκ₀; κ₀ = 1ᵀΣ1/B²" }]
      : [{ value: gk, label: "γκ₀", title: "collective threshold γκ₀" }, { value: SG.kappa0, label: "κ₀", title: "κ₀ = 1ᵀΣ1/B², mean covariance of two initial neurons" }];
    if (st.fam === "blocks") marks.push({ value: st.btw, label: "between" }, { value: st.win, label: "within" });
    if (st.fam === "hier") HIER_CORR.forEach(c => marks.push({ value: c, label: f2(c) }));
    marks.push({ value: 0, label: "0" });
    const seen = [], mk = marks.filter(m => { if (m.value < -0.1 || m.value > 1 || seen.some(x => Math.abs(x - m.value) < 0.11)) return false; seen.push(m.value); return true; });
    const sl = ui.slider({ label: "pruning threshold $\\theta$", min: -0.1, max: 1, step: 0.01, value: st.theta, fmt: x => f2(x), marks: mk,
      help: `Synapses survive where $\\mathrm{Cov}(u_i,u_j) > \\theta$. κ₀ = 1ᵀΣ1/B² = ${f2(SG.kappa0)} is the mean covariance of two initial neurons; below γκ₀ = ${f2(st.gamma * SG.kappa0)} the population starts bound together.`,
      onInput: x => { if (st.cont) { st.theta = x; PP.theta = x; moved(); postRun(); } },
      onChange: x => change({ theta: x }) });
    thetaHost.replaceChildren(sl.el, thetaInfo); thetaS = sl;
    drawThetaInfo();
  };
  const thetaInfo = ui.h("div", { class: "u-help", style: { marginTop: "2px" } });
  function drawThetaInfo() {
    const gk = st.gamma * SG.kappa0, bound = st.theta < gk;
    thetaInfo.innerHTML = `$\\kappa_0 = \\mathbf 1^\\top\\Sigma_g\\mathbf 1/B^2 = ${f2(SG.kappa0)}$ · collective threshold $\\gamma\\kappa_0 = ${f2(gk)}$: `
      + (st.gamma === 0 ? "none ($\\gamma = 0$, neurons choose independently)" : bound ? "$\\theta$ is below it, the population starts bound together" : "$\\theta$ is above it, neurons choose independently")
      + ` · floor $\\omega = N w_{\\min} = ${ui.fmt(st.n * st.wmin, 2)}$`;
    ui.renderMath(thetaInfo);
  }
  makeTheta();
  const pS = ui.segmented({ label: "sparse wiring: connection probability $p$", options: [1, 0.7, 0.5, 0.3].map(x => ({ value: x, label: String(x) })), value: st.p, stack: true,
    help: "$p < 1$: a fixed random mask; each neuron can only ever touch a random subset of inputs and neurons.", onChange: x => change({ p: x }) });
  const nS = ui.slider({ label: "interior neurons $N$", min: 60, max: 250, step: 10, value: st.n, fmt: String, onChange: x => change({ n: x }) });
  const gS = ui.segmented({ label: "lateral gain $\\gamma$", options: [0, 0.5, 1, 2].map(x => ({ value: x, label: String(x) })), value: st.gamma, onChange: x => change({ gamma: x }) });
  const wS = ui.slider({ label: "floor $w_{\\min}\\lambda/\\eta$", min: 1e-6, max: 1e-2, log: true, value: st.wmin, fmt: x => `${ui.fmt(x, 2)} · ω = ${ui.fmt(x * st.n, 2)}`,
    help: "ω = N·w_min: the floor's total leak toward the population mean; keep ω ≲ 0.02 (docs/findings.md F5).", onChange: x => change({ wmin: x }) });
  const ellS = ui.slider({ label: "ring correlation length $\\ell$", min: 0.5, max: 4, step: 0.1, value: st.ell, fmt: f2, onChange: x => change({ ell: x }) });
  const jitT = ui.toggle({ label: "private jitter, $w \\leftarrow w\\,e^{\\sigma\\xi}$, $\\sigma = 10^{-3}$", value: st.jit,
    help: "Breaks exact symmetries: a tiny residual can be an unstable symmetric state (docs/findings.md F8).", onChange: x => change({ jit: x }) });
  const aS = ui.slider({ label: "spectrum exponent $\\alpha$", min: 0.25, max: 2.5, step: 0.05, value: st.alpha, fmt: f2, help: "eigenvalues $\\propto k^{-\\alpha}$", onChange: x => change({ alpha: x }) });
  const dS = ui.segmented({ label: "Wishart degrees of freedom / $B$", options: [0.25, 0.5, 1, 2].map(x => ({ value: x, label: String(x) })), value: st.dofr, onChange: x => change({ dofr: x }) });
  const nbS = ui.segmented({ label: "number of blocks", options: [2, 3, 4, 6, 8], value: st.nb, onChange: x => change({ nb: x }) });
  const wiS = ui.slider({ label: "correlation within a block", min: 0, max: 0.95, step: 0.05, value: st.win, fmt: f2, onChange: x => change({ win: x }) });
  const btS = ui.slider({ label: "correlation between blocks", min: -0.3, max: 0.5, step: 0.05, value: st.btw, fmt: f2, help: "negative: signed blocks (clamped to the nearest valid correlation matrix)", onChange: x => change({ btw: x }) });
  const uqT = ui.toggle({ label: "unequal block sizes and variances", value: st.uneq, onChange: x => change({ uneq: x }) });
  const csS = ui.select({ label: "$\\Sigma_g$ seed", options: [1, 2, 3, 4, 5], value: st.cs, onChange: x => change({ cs: x }) });
  const seedS = ui.select({ label: "initial-weight seed", options: [1, 2, 3, 4, 5], value: st.seed, onChange: x => change({ seed: x }) });
  const modeS = ui.segmented({ label: "plasticity sees", options: [{ value: "mf", label: "exact $\\Sigma_g$" }, { value: "mb", label: "minibatches" }], value: st.mode, onChange: x => change({ mode: x }) });
  const batchS = ui.segmented({ label: "minibatch size", options: [8, 32, 128], value: st.batch, onChange: x => change({ batch: x }) });
  const contT = ui.toggle({ label: "keep weights when $\\theta$ changes", value: st.cont, help: "Off: a new $\\theta$ restarts from the same initial weights. On: the run continues from the current weights (hysteresis).", onChange: x => change({ cont: x }) });
  const trailT = ui.toggle({ label: "fading trails", value: false, onChange: on => v.setTrails(on ? 10 : 0) });
  const famOnly = { blocks: [nbS, wiS, btS, uqT], pow: [aS, csS], powpos: [aS, csS], wish: [dS, csS], hier: [], ring: [ellS] };
  ui.controls({ parent: L.side, title: "Parameters", primary: [famS, bS, thetaHost, pS],
    advanced: [nS, gS, wS, nbS, wiS, btS, uqT, aS, dS, ellS, csS, seedS, modeS, batchS, contT, jitT, trailT], advancedLabel: "More controls" });
  const showFamControls = () => {
    for (const c of [aS, dS, nbS, wiS, btS, uqT, csS, ellS]) c.el.style.display = famOnly[st.fam].includes(c) ? "" : "none";
    batchS.el.style.display = st.mode === "mb" ? "" : "none";
  };
  showFamControls();
  const stats = ui.stats({ parent: L.side, items: [
    { key: "k", label: "clusters $K$", fmt: String },
    { key: "dead", label: "dead neurons", fmt: String },
    { key: "er", label: "effective rank of $H$", fmt: x => ui.fmt(x, 3) },
    { key: "r", label: "stationarity $r/\\eta$", fmt: sci },
    { key: "cos", label: "readout cosine, $\\theta$ unknown", fmt: f3 },
    { key: "ps", label: "parts-sum cosine, $\\theta$ unknown", fmt: f3 },
    { key: "dir", label: "directions pinned down", fmt: String },
    { key: "fps", label: "frame rate", unit: "fps", fmt: x => x.toFixed(0) }] });
  ui.legend({ parent: L.side, items: [
    { label: "interior neuron; colour = cluster", color: PAL.cat[0], shape: "dot" }, { label: "other / not clustered yet", color: PAL.other, shape: "dot" },
    { label: "boundary neuron $e_b$, projected", color: PAL.boundary, shape: "square" }, { label: "synapse, opacity $\\propto w/\\max w$", color: PAL.edge, shape: "line" }] });

  // below: Σ_g, similarity, parts, time series
  const box = (title, cls) => { const b = ui.h("div", { class: cls || "" }), t = ui.h("div", { class: "chart-title", html: title }); const wrap = ui.h("div", {}, t, b); return { wrap, b, t }; };
  const bxSigma = box("Input covariance $\\Sigma_g$ <span class=\"synthetic-tag\">synthetic Σ</span>");
  const bxSim = box("Voltage covariance $H\\Sigma_g H^\\top$, neurons sorted by cluster");
  const bxSpec = box("Eigenvalues of $\\Sigma_g$");
  L.below.append(bxSigma.wrap, bxSim.wrap, bxSpec.wrap);
  const bxParts = box("Cluster parts: mean embedding over boundary neurons");
  let partsLine = "eig";
  const partsSeg = ui.segmented({ label: "line", options: [{ value: "eig", label: "nearest eigenvector of $\\Sigma_g$" }, { value: "perron", label: "Perron vector of $(\\Sigma_g - \\theta\\mathbf{1}\\mathbf{1}^\\top)$ on the support" }],
    value: "eig", onChange: x => { partsLine = x; drawParts(); } });
  partsSeg.el.style.margin = "2px 0 8px";
  bxParts.t.after(partsSeg.el);
  bxParts.wrap.style.gridColumn = "1 / -1";
  const partsNote = ui.h("div", { class: "u-help" });
  bxParts.wrap.append(partsNote);
  const bxTs1 = box("Clusters and effective rank over time"), bxTs2 = box("Stationarity residual over time");
  L.below.append(bxParts.wrap, bxTs1.wrap, bxTs2.wrap);
  ui.renderMath(L.below);
  const sigHM = heatmap(bxSigma.b, { map: "div", range: [-1, 1], size: 300, zlabel: "$(\\Sigma_g)_{bc}$", xlabel: "input c", ylabel: "input b",
    tooltip: (i, j, val) => ui.tipRows(`${gName(SG.order[i])}, ${gName(SG.order[j])}`, [{ name: "covariance", value: f3(val) },
      { name: "correlation", value: f3(val / Math.sqrt(Sigma[SG.order[i] * B + SG.order[i]] * Sigma[SG.order[j] * B + SG.order[j]])) }]) });
  const simHM = heatmap(bxSim.b, { map: "div", range: [-1, 1], size: 300, zlabel: "$h_i^\\top\\Sigma_g h_j$", xlabel: "neuron j (sorted)", ylabel: "neuron i (sorted)" });
  const specC = lineChart(bxSpec.b, { series: [{ name: "Σ_g", y: [1], x: [1] }], ylog: true, height: 230, xlabel: "rank k", ylabel: "eigenvalue λ_k", legend: false });
  const partsC = barMultiples(bxParts.b, { labels: [], panels: [], yrange: [-0.75, 1], height: 92, panelWidth: 200, valueName: "cluster part $h/\\lVert h\\rVert_2$", overlayName: "nearest eigenvector of Σ_g" });
  const kC = lineChart(bxTs1.b, { series: [{ name: "clusters K", y: [], x: [] }, { name: "effective rank of H", y: [], x: [] }], height: 210, ylog: true, xlabel: "time t", ylabel: "count (log)", xname: "t" });
  const rC = lineChart(bxTs2.b, { series: [{ name: "r/η", y: [], x: [] }], ylog: true, height: 210, xlabel: "time t", ylabel: "r/η", legend: false, yfmt: y => (Math.round(Math.log10(y)) % 3 === 0 ? sci(y) : ""), refs: [{ y: EQ_TOL, label: "equilibrium tolerance" }] });

  const simCap = ui.h("figcaption");
  ctx.slot("sim").after(simCap);

  // ---------- figure 2: the readout (filled below)
  const RO = buildReadoutFigure(ctx.slot("readout"));

  // ------------------------------------------------------------------------------------------------ analysis
  function analyse() {
    pca = E.pca3(H, N, B, pca);
    const big = N * N * B > 1.6e6;
    if (!cl || !big || frameNo % 3 === 0) {
      cl = E.clustersAuto(H, N, B);
      cstats = E.clusterStats(H, N, B, cl.labels, cl.K, Sigma);
      assignColours();
    }
  }
  // colours: clusters keep their colour across frames (majority vote of the previous colours of their members);
  // new clusters get the lowest free colour in order of size (largest first); past 8, and before the cluster count is
  // robust, neurons are grey ("other").
  function assignColours() {
    const K = cl.K, sizes = cstats.sizes;
    clusterColour = new Array(K).fill(-1);
    if (!cl.robust) { colourOf = null; return; }
    const votes = [];
    for (let k = 0; k < K; k++) votes.push(new Map());
    for (let i = 0; i < N; i++) { const c = neuronColour[i]; if (c >= 0) { const m = votes[cl.labels[i]]; m.set(c, (m.get(c) || 0) + 1); } }
    const cand = [];
    votes.forEach((m, k) => m.forEach((n, c) => cand.push([n, k, c])));
    cand.sort((a, b) => b[0] - a[0] || sizes[b[1]] - sizes[a[1]]);
    const used = new Set();
    for (const [n, k, c] of cand) if (clusterColour[k] < 0 && !used.has(c) && n * 2 > Math.min(sizes[k], 2)) { clusterColour[k] = c; used.add(c); }
    const rest = Array.from({ length: K }, (_, k) => k).filter(k => clusterColour[k] < 0).sort((a, b) => sizes[b] - sizes[a]);
    for (const k of rest) { let c = 0; while (used.has(c)) c++; if (c < 8 && (sizes[k] > 1 || K <= 8)) { clusterColour[k] = c; used.add(c); } }
    for (let i = 0; i < N; i++) neuronColour[i] = clusterColour[cl.labels[i]];
    colourOf = Array.from({ length: N }, (_, i) => catColor(neuronColour[i]));
  }
  // cluster order for sorted views: coloured clusters by colour index, then grey ones by size
  function clusterOrder() {
    const K = cl.K, sizes = cstats.sizes;
    return Array.from({ length: K }, (_, k) => k).sort((a, b) => {
      const ca = clusterColour[a], cb = clusterColour[b];
      if (ca >= 0 && cb >= 0) return ca - cb; if (ca >= 0) return -1; if (cb >= 0) return 1; return sizes[b] - sizes[a];
    });
  }

  // ------------------------------------------------------------------------------------------------ drawing
  function anchorsOf() {
    const X = new Float64Array(B * B); for (let b = 0; b < B; b++) X[b * B + b] = 1;
    const Y = E.projectPCA(pca, X, B, B);
    const norms = Array.from({ length: B }, (_, b) => Math.hypot(Y[3 * b], Y[3 * b + 1], Y[3 * b + 2]));
    const R = Math.max(...norms) || 1;
    // corners that project to (nearly) the same point share one label, e.g. "g₁–g₆" or "g₂ +3"
    const lead = new Int32Array(B).fill(-1), members = [];
    for (let b = 0; b < B; b++) {
      if (lead[b] >= 0) continue;
      lead[b] = b; const grp = [b];
      for (let c = b + 1; c < B; c++) if (lead[c] < 0 && Math.hypot(Y[3 * b] - Y[3 * c], Y[3 * b + 1] - Y[3 * c + 1], Y[3 * b + 2] - Y[3 * c + 2]) < 0.03 * R) { lead[c] = b; grp.push(c); }
      members[b] = grp;
    }
    const leads = members.map((g, b) => (g ? b : -1)).filter(b => b >= 0);
    const ranked = leads.slice().sort((a, c) => norms[c] - norms[a]);
    // label the farthest groups first, skipping any whose position is within 0.12 R of one already labelled
    const labelled = new Set(), placed = [];
    for (const b of ranked) {
      if (labelled.size >= 12 || norms[b] < 0.12 * R) break;
      if (placed.some(c => Math.hypot(Y[3 * b] - Y[3 * c], Y[3 * b + 1] - Y[3 * c + 1], Y[3 * b + 2] - Y[3 * c + 2]) < 0.12 * R)) continue;
      labelled.add(b); placed.push(b);
    }
    const nameOf = g => { if (g.length === 1) return gName(g[0]); const run = g.every((x, k) => k === 0 || x === g[k - 1] + 1); return run ? `${gName(g[0])}–${gName(g[g.length - 1])}` : `${gName(g[0])} +${g.length - 1}`; };
    return Array.from({ length: B }, (_, b) => ({ xyz: [Y[3 * b], Y[3 * b + 1], Y[3 * b + 2]], label: members[b] && labelled.has(b) ? nameOf(members[b]) : "",
      color: PAL.boundary, size: B > 12 ? 2 : 2.6 }));
  }
  function buildEdges() {
    const { Wii, Wib, Mii, Mib } = s, thr = PP.wmin * 1.5;
    const pairs = edgeBuf.pairs, w = edgeBuf.w; let m = 0, wmax = 0;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const k = i * N + j; if (Mii[k] && Wii[k] > thr) { pairs[2 * m] = i; pairs[2 * m + 1] = j; w[m] = Wii[k]; if (Wii[k] > wmax) wmax = Wii[k]; m++; } }
    for (let i = 0; i < N; i++) for (let b = 0; b < B; b++) { const k = i * B + b; if (Mib[k] && Wib[k] > thr) { pairs[2 * m] = i; pairs[2 * m + 1] = N + b; w[m] = Wib[k]; if (Wib[k] > wmax) wmax = Wib[k]; m++; } }
    let cut = 0;
    if (m > MAX_EDGES) {
      const NB = 256, hist = new Int32Array(NB);
      for (let e = 0; e < m; e++) hist[Math.min(NB - 1, (w[e] / wmax * NB) | 0)]++;
      let acc = 0, bin = NB - 1; for (; bin > 0; bin--) { acc += hist[bin]; if (acc >= MAX_EDGES) break; }
      cut = bin / NB * wmax;
    }
    const outP = [], outA = [];
    for (let e = 0; e < m; e++) if (w[e] >= cut) { outP.push(pairs[2 * e], pairs[2 * e + 1]); outA.push(wmax > 0 ? w[e] / wmax : 0); if (outA.length >= MAX_EDGES) break; }
    return { pairs: Uint32Array.from(outP), alpha: Float32Array.from(outA), active: m };
  }
  let lastEdges = null;
  // the camera is refitted when the scene's extent has changed a lot (at most once a second), e.g. when the
  // corners stop being spread by a random initial PCA frame and the clusters settle
  let fitR = 0, fitT = 0;
  function maybeFit(xyz, an, force) {
    let R = 0;
    for (let i = 0; i < N; i++) R = Math.max(R, Math.hypot(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]));
    for (const a of an) R = Math.max(R, Math.hypot(...a.xyz));
    const now = performance.now();
    if (force || ((R > 1.5 * fitR || R < 0.6 * fitR) && now - fitT > 1000)) { v.fit(); fitR = R; fitT = now; }
  }
  function drawView(edges, force) {
    const xyz = new Float32Array(3 * N);
    for (let k = 0; k < 3 * N; k++) xyz[k] = pca.coords[k];
    const an = anchorsOf();
    v.setPoints(xyz, colourOf || PAL.other);
    v.setAnchors(an);
    maybeFit(xyz, an, force);
    if (edges) v.setEdges(edges.pairs, edges.alpha);
    let tot = 0; for (let i = 0; i < N; i++) for (let b = 0; b < B; b++) tot += (H[i * B + b] - pca.mean[b]) ** 2;
    const ex = tot > 0 ? (pca.variance[0] + pca.variance[1] + pca.variance[2]) * N / tot : 1;
    v.setCorner(`PCA of the ${N} embeddings · 3 axes hold ${ui.fmtPct(Math.min(1, ex))} of their variance`);
  }
  function drawSim() {
    const ordC = clusterOrder(), byC = ordC.map(() => []);
    const pos = new Int32Array(cl.K); ordC.forEach((k, q) => { pos[k] = q; });
    for (let i = 0; i < N; i++) byC[pos[cl.labels[i]]].push(i);
    const perm = byC.flat();
    const M = E.similarity(H, N, B, Sigma), S2 = new Float32Array(N * N);
    for (let a = 0; a < N; a++) for (let c = 0; c < N; c++) S2[a * N + c] = M[perm[a] * N + perm[c]];
    const groups = []; let o = 0;
    byC.forEach((list, q) => { const k = ordC[q], c = clusterColour[k]; if (list.length) groups.push({ start: o, end: o + list.length, color: catColor(c), label: c >= 0 ? `c${c + 1}` : "" }); o += list.length; });
    simHM.update({ values: S2, rows: N, cols: N, range: [-SG.maxVar, SG.maxVar], groups: cl.robust ? { rows: groups } : null,
      tooltip: (i, j, val) => ui.tipRows(`neurons ${perm[i]} and ${perm[j]}`, [{ name: "h_iᵀΣh_j", value: f3(val) },
        { name: "clusters", value: cl.robust ? `${clusterLabel(cl.labels[perm[i]])}, ${clusterLabel(cl.labels[perm[j]])}` : "not clustered yet" }]) });
  }
  const clusterLabel = k => (clusterColour[k] >= 0 ? `c${clusterColour[k] + 1}` : `other (${cstats.sizes[k]})`);
  function drawParts() {
    if (!cl.robust) { partsC.update({ labels: [], panels: [] }); partsNote.innerHTML = "Not clustered yet: the merge heights of the embeddings show no clear gap (ratio < 3)."; return; }
    const ordC = clusterOrder().filter(k => clusterColour[k] >= 0);
    const labels = SG.order.map(b => String(b + 1));
    const panels = ordC.map(k => {
      const part = Array.from(cstats.parts.subarray(k * B, k * B + B));
      const nrm = Math.hypot(...part) || 1, u = part.map(x => x / nrm);
      let bi = 0, bc = 0;
      for (let q = 0; q < B; q++) { let d = 0; for (let b = 0; b < B; b++) d += u[b] * SG.eig.vectors[b * B + q]; if (Math.abs(d) > Math.abs(bc)) { bc = d; bi = q; } }
      const sgn = bc < 0 ? -1 : 1, ev = SG.order.map(b => sgn * SG.eig.vectors[b * B + bi]);
      // the part's own Perron vector: top eigenvector of A_SS, A = Σ − θ11ᵀ, on the cluster's support S (F1)
      const S = []; for (let b = 0; b < B; b++) if (part[b] > 0.02) S.push(b);
      const m = S.length, A = new Float64Array(m * m);
      for (let a = 0; a < m; a++) for (let c = 0; c < m; c++) A[a * m + c] = Sigma[S[a] * B + S[c]] - st.theta;
      const ea = E.eigh(A, m), pv = new Float64Array(B);
      let ps = 0; for (let a = 0; a < m; a++) ps += ea.vectors[a * m];
      for (let a = 0; a < m; a++) pv[S[a]] = (ps < 0 ? -1 : 1) * ea.vectors[a * m];
      let num = 0; for (let b = 0; b < B; b++) num += u[b] * pv[b];
      // Perron vector of Σ_SS alone (no θ shift), and the stability margin max_{b∉S} (A h)_b, h summing to 1
      for (let a = 0; a < m; a++) for (let c = 0; c < m; c++) A[a * m + c] += st.theta;
      const es = E.eigh(A, m); let sgn2 = 0, numS = 0;
      for (let a = 0; a < m; a++) sgn2 += es.vectors[a * m];
      for (let a = 0; a < m; a++) numS += u[S[a]] * (sgn2 < 0 ? -1 : 1) * es.vectors[a * m];
      let off = -Infinity, hs = 0; for (let b = 0; b < B; b++) hs += part[b];
      const inS = new Uint8Array(B); S.forEach(b => { inS[b] = 1; });
      for (let b = 0; b < B; b++) if (!inS[b]) { let v0 = 0; for (let c = 0; c < B; c++) v0 += Sigma[b * B + c] * part[c] / hs; off = Math.max(off, v0 - st.theta); }
      const gapOK = m < 2 || ea.values[0] - ea.values[1] > 1e-6;
      const stab = !Number.isFinite(off) ? "stable (no inputs off S)" : off < 0 && gapOK ? `stable: max off S of $(Ah)_b$ = ${ui.fmt(off, 2)}` : `not a stable part: max off S of $(Ah)_b$ = ${ui.fmt(off, 2)}`;
      // c_k = (hᵀΣh − θ)/|h|² = a member's summed input conductance (λ = η = 1)
      let q = 0; for (let a = 0; a < B; a++) for (let c = 0; c < B; c++) q += part[a] * Sigma[a * B + c] * part[c];
      const ck = (q - st.theta) / (nrm * nrm);
      const showP = partsLine === "perron";
      return { title: `c${clusterColour[k] + 1} · ${cstats.sizes[k]} neurons · ${m} inputs · c = ${ui.fmt(ck, 3)}`, color: catColor(clusterColour[k]), values: SG.order.map(b => part[b] / nrm),
        overlay: showP ? SG.order.map(b => pv[b]) : ev,
        sub: `cos with eigenvector v${sub(bi + 1)} of Σ_g: <b>${f2(Math.abs(bc))}</b> · with Perron vector of $A_{SS}$: <b>${f3(num)}</b> (of $\\Sigma_{SS}$: ${f3(numS)}) · ${stab}`,
        h: part };
    });
    let lo = -0.25; panels.forEach(pn => pn.overlay.forEach(x => { if (x < lo) lo = x; }));
    partsC.update({ labels, panels, yrange: [Math.floor(lo * 4) / 4, 1], overlayName: partsLine === "perron" ? "Perron vector of (Σ_g − θ11ᵀ) on the cluster's support" : "nearest eigenvector of Σ_g" });
    const hidden = cl.K - ordC.length;
    // packing (F3): coexisting clusters need γ h_kᵀΣh_l ≤ θ (+ floor terms)
    let xc = -Infinity, pair = null;
    for (let a = 0; a < panels.length; a++) for (let c = a + 1; c < panels.length; c++) {
      const ha = panels[a].h, hc = panels[c].h; let q = 0;
      for (let x = 0; x < B; x++) { if (!ha[x]) continue; for (let y = 0; y < B; y++) q += ha[x] * Sigma[x * B + y] * hc[y]; }
      if (q > xc) { xc = q; pair = [a, c]; }
    }
    const pack = pair && st.gamma > 0 ? `Largest cross-covariance between two shown clusters: $h_k^\\top\\Sigma_g h_l = ${f3(xc)}$ (${panels[pair[0]].title.split(" ·")[0]}, ${panels[pair[1]].title.split(" ·")[0]}); packing bound $\\theta/\\gamma = ${f2(st.theta / st.gamma)}$. ` : "";
    partsNote.innerHTML = pack + (hidden > 0 ? `${hidden} further cluster${hidden > 1 ? "s" : ""} (grey in the views) not shown.` : "");
    ui.renderMath(partsNote); ui.renderMath(bxParts.b);
  }
  function drawSeries() {
    const refs = seqMarks.map(m => ({ x: m.t, label: m.label }));
    kC.update({ series: [{ name: "clusters K", x: series.t, y: series.K }, { name: "effective rank of H", x: series.t, y: series.er }], refs });
    rC.update({ series: [{ name: "r/η", x: series.t, y: series.r.map(x => Math.max(x, 1e-16)) }], refs: [{ y: EQ_TOL, label: "equilibrium tolerance" }, ...refs] });
    const t = simTime(); kC.setCursor(t); rC.setCursor(t);
    RO.drawTime(series, refs, t);
  }
  function drawStats() {
    const r = residual();
    stats.set("k", cl.K, cl.robust ? (cl.gap > 1000 ? "clear-cut (gap > 1000×)" : `gap ${ui.fmt(cl.gap, 2)}×`) : "not clustered yet");
    stats.set("er", Number.isFinite(effRank) ? effRank : E.effectiveRank(H, N, B), `of B = ${B}`);
    stats.set("r", r, Number.isFinite(r) ? (r < EQ_TOL ? "equilibrium: yes" : "equilibrium: not yet") : "");
    let nd = 0; const thr = 1.5 * PP.wmin;
    for (let i = 0; i < N; i++) { let m = 0; for (let b = 0; b < B; b++) if (s.Mib[i * B + b] && s.Wib[i * B + b] > m) m = s.Wib[i * B + b]; if (m <= thr) nd++; }
    stats.set("dead", nd, "every I–B weight at the floor");
  }
  function drawAll(heavy, force) {
    lastEdges = buildEdges();
    drawView(lastEdges, force);
    drawStats();
    if (heavy) { drawSim(); drawParts(); drawSeries(); }
  }
  function drawSigma() {
    const o = SG.order, S2 = new Float32Array(B * B);
    for (let a = 0; a < B; a++) for (let c = 0; c < B; c++) S2[a * B + c] = Sigma[o[a] * B + o[c]];
    const m = Math.max(...Array.from(Sigma, Math.abs));
    sigHM.update({ values: S2, rows: B, cols: B, range: [-m, m], rowLabels: o.map(b => gName(b)), colLabels: o.map(b => String(b + 1)), groups: groupsOf() });
    specC.update({ series: [{ name: "eigenvalue of Σ_g", x: Array.from({ length: B }, (_, k) => k + 1), y: Array.from(SG.eig.values, x => Math.max(x, 1e-6)), dots: true, color: PAL.fg }] });
    RO.setSigma();
    drawCaption();
  }
  function drawCaption() {
    drawThetaInfo();
    simCap.innerHTML = `<b>Plotted:</b> top: each circle is an interior neuron's harmonic embedding $h_i \\in \\mathbb{R}^{B}$ projected orthogonally onto the top three principal axes of the current set of embeddings (centred; the frame is rotated within that subspace to match the previous frame, so it never flips); colour = cluster (single linkage in $L_1$, cut at the largest gap of merge heights; ordered by size, largest blue; grey: past the eighth cluster, or before the count is robust). Squares: the corners $e_b$ of the simplex (the boundary neurons) under the same projection, labelled when far from the centre. Lines: active synapses (above $1.5\\,w_{\\min}$), opacity $w/\\max w$, strongest ${MAX_EDGES.toLocaleString("en-GB")}. Below, left to right: $\\Sigma_g$ (diverging, symmetric range), boundary neurons ${SG.labels ? "in planted order" : "sorted by angle in the plane of the top two eigenvectors"}; $h_i^\\top\\Sigma_g h_j$ for neurons sorted by cluster (bands: cluster colour), fixed range $\\pm\\max_b(\\Sigma_g)_{bb}$; eigenvalues of $\\Sigma_g$, log scale. Parts: for each coloured cluster, the member mean $h$ of $h_i$ scaled to unit length (bars, same input order as $\\Sigma_g$), and as ink dots either the eigenvector of $\\Sigma_g$ with the largest $|\\cos|$ to it (sign flipped to agree) or the Perron vector of $A_{SS}$, $A = \\Sigma_g - \\theta\\mathbf 1\\mathbf 1^\\top$, $S$ = the inputs with $h_b > 0.02$; title: $c = (h^\\top\\Sigma_g h - \\theta)/\\lVert h\\rVert^2$; sub-line: the cosines of the part with that eigenvector, with the Perron vector of $A_{SS}$ and with that of $\\Sigma_{SS}$, and $\\max_{b \\notin S}(Ah)_b$ for $h$ summing to 1 (stable part: negative, and $c$ the strict top eigenvalue of $A_{SS}$). Note under the panels: the largest $h_k^\\top\\Sigma_g h_l$ between two shown clusters. Time series: $K$, effective rank $\\lVert H\\rVert_F^4/\\lVert H^\\top H\\rVert_F^2$, and the projected residual $r/\\eta = \\max|dw/dt|/\\eta$ (floor edges with negative drift count 0); orange: the time shown; dashed: tour steps. <b>Values:</b> ${SG.desc}; $B = ${B}$, $N = ${st.n}$, $\\theta = ${f2(st.theta)}$, $\\gamma = ${st.gamma}$, $w_{\\min} = ${ui.fmt(st.wmin, 2)}$, $p = ${st.p}$, $\\lambda = \\eta = 1$, init seed ${st.seed}, initial $W_{IB} \\sim \\mathrm{Exp}(\\text{mean } 0.2)$, $W_{II} \\sim U(0, ${II_SCALE})$ clipped at $w_{\\min}$; Euler $dt = ${DT}$; ${st.mode === "mf" ? "mean-field (plasticity sees the exact $\\Sigma_g$)" : `minibatch: plasticity sees the covariance of ${st.batch} Gaussian samples per step`}${st.jit ? `; private jitter $\\sigma = ${JITTER}$` : ""}.`;
    ui.renderMath(simCap);
  }
  function groupsOf() {
    if (!SG.groups) return null;
    const o = SG.order, gs = []; let start = 0;
    for (let a = 1; a <= B; a++) if (a === B || SG.groups[o[a]] !== SG.groups[o[a - 1]]) { gs.push({ start, end: a, label: "" }); start = a; }
    return { rows: gs };
  }
  function tipFor(i, kind) {
    if (kind === "anchor") {
      let n = 0; for (let q = 0; q < N; q++) if (s.Mib[q * B + i] && s.Wib[q * B + i] > PP.wmin * 1.5) n++;
      return ui.tipRows(`boundary neuron ${gName(i)}`, [{ name: "variance (Σ_g)_bb", value: f3(Sigma[i * B + i]) }, { name: "interior neurons synapsing", value: String(n) }]);
    }
    if (!H || i >= N) return null;
    const h = Array.from(H.subarray(i * B, i * B + B)), top = h.map((x, b) => [x, b]).sort((a, c) => c[0] - a[0]).slice(0, 3);
    let q = 0; for (let a = 0; a < B; a++) for (let c = 0; c < B; c++) q += h[a] * Sigma[a * B + c] * h[c];
    return ui.tipRows(`interior neuron ${i}`, [
      { name: "cluster", value: cl && cl.robust ? clusterLabel(cl.labels[i]) : "not clustered yet", color: colourOf ? colourOf[i] : PAL.other },
      { name: "inputs with h > 0.02", value: String(h.filter(x => x > 0.02).length) },
      ...top.map(([x, b]) => ({ name: `h(${gName(b)})`, value: f3(x) })),
      { name: "voltage variance hᵀΣh", value: f3(q) }]);
  }

  // ------------------------------------------------------------------------------------------------ run loop
  function clearSeries() { for (const k in series) series[k] = []; seqMarks.length = 0; RO.clearTime(); }
  let effRank = NaN;
  function pushSeries() {
    effRank = E.effectiveRank(H, N, B);
    series.t.push(simTime()); series.K.push(cl.robust ? cl.K : NaN); series.er.push(effRank); series.r.push(residual());
    if (series.t.length > 1500) for (const k of ["t", "K", "er", "r"]) series[k] = series[k].filter((_, i) => i % 2 === 0);
  }
  const frameOf = () => ({ t: simTime(), H: Float32Array.from(H), edges: lastEdges ? { pairs: lastEdges.pairs.slice(0, 2 * 2500), alpha: lastEdges.alpha.slice(0, 2500) } : null });
  function stepOnce() {
    prevWii.set(s.Wii); prevWib.set(s.Wib); prevValid = true;
    if (st.mode === "mf") E.step(s, Sigma, PP); else E.stepStochastic(s, sampler, PP, st.batch);
    if (st.jit) {   // private multiplicative jitter on every potential synapse (symmetric for I–I), floor kept
      const { Wii, Wib, Mii, Mib } = s, fl = PP.wmin;
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const k = i * N + j; if (Mii[k]) { const w = Math.max(fl, Wii[k] * Math.exp(JITTER * jrng.normal())); Wii[k] = w; Wii[j * N + i] = w; } }
      for (let k = 0; k < N * B; k++) if (Mib[k]) Wib[k] = Math.max(fl, Wib[k] * Math.exp(JITTER * jrng.normal()));
    }
    version++; seqSteps++;
  }
  function onTick(speed) {
    const t0 = performance.now();
    const due = jobDue();
    let n = 0;
    // early in a run the readout is sampled at least every EARLY_EVERY steps (its mid-training peak is brief), so the
    // sim waits for the worker there rather than outrunning it
    const early = worker && st.mode === "mf" && !seq && s.t < EARLY_STEPS;
    if (!(early && busy && s.t - lastDispatchStep >= EARLY_EVERY))
      do { stepOnce(); n++; } while (n < speed && performance.now() - t0 < 14 && !(early && s.t - lastDispatchStep >= EARLY_EVERY));
    if (st.mode === "mb" && (frameNo % 8 === 0 || due)) { mbResid = E.stationarity(s, Sigma, PP).r; }
    H = s.H.slice();          // H of the weights the last step started from
    frameNo++; tick++;
    analyse();
    // linked views are staggered over frames so no single frame carries all of them
    const phase = tick % 8;
    pushSeries();
    drawAll(false);
    if (phase === 0) drawSim(); else if (phase === 3) drawParts(); else if (phase === 6) drawSeries();
    if (phase !== 6) { const t = simTime(); kC.setCursor(t); rC.setCursor(t); }
    if (due) dispatch();
    fpsN++; const now = performance.now(); if (now - fpsT > 1000) { fps = fpsN * 1000 / (now - fpsT); fpsN = 0; fpsT = now; stats.set("fps", fps); }
    const r = residual(), conv = st.mode === "mf" && r < DONE_TOL, seqConv = st.mode === "mf" && r < SEQ_TOL;
    const f = frameOf();
    if (seq) {
      if (seqConv || seqSteps > 6000) {
        if (!holdT) { holdT = now; drawSim(); drawParts(); drawSeries(); }
        else if (now - holdT > 1700) {
          seqIdx++; holdT = 0; seqSteps = 0;
          if (seqIdx >= seq.list.length) { seq = null; tp.setStatus(""); drawAll(true); dispatch(); return { ...f, done: true, doneLabel: "tour finished" }; }
          setThetaSeq(seq.list[seqIdx]);
        }
      }
      if (seq) tp.setStatus(`θ = ${f2(st.theta)} (${seqIdx + 1} of ${seq.list.length})`);
      return f;
    }
    if (conv) { drawAll(true); dispatch(); return { ...f, done: true, doneLabel: "converged" }; }
    return f;
  }
  function setThetaSeq(th) {
    st.theta = th; thetaS.set(th);
    seqMarks.push({ t: simTime(), label: `θ = ${f2(th)}` });
    if (seq.cont) { PP.theta = th; postRun(); }
    else { tOff = simTime(); newState(); analyse(); }
    drawCaption();
  }
  function showFrame(f) {
    H = Float64Array.from(f.H); pca = E.pca3(H, N, B, pca);
    cl = E.clustersAuto(H, N, B); cstats = E.clusterStats(H, N, B, cl.labels, cl.K, Sigma); assignColours();
    const xyz = Float32Array.from(pca.coords);
    v.setPoints(xyz, colourOf || PAL.other); v.setAnchors(anchorsOf());
    if (f.edges) v.setEdges(f.edges.pairs, f.edges.alpha);
    drawSim(); drawParts();
    kC.setCursor(f.t); rC.setCursor(f.t); RO.cursor(f.t);
    stats.set("k", cl.K, cl.robust ? "" : "not clustered yet"); stats.set("er", E.effectiveRank(H, N, B), `of B = ${B}`);
  }

  // ------------------------------------------------------------------------------------------------ worker
  let worker = null, busy = false, lastDispatch = 0, lastJobMs = 0, jobSeq = 0, lastResVersion = -1, specVersion = -1, jobCount = 0;
  try { worker = new Worker(new URL("./ch_many_worker.js", import.meta.url), { type: "module" }); }
  catch (err) { worker = null; }
  if (worker) worker.onmessage = e => onWorker(e.data);
  function postRun() {
    if (!worker || !s) return;
    const W0 = E.createState({ N, B, seed: st.seed, p: st.p, iiScale: II_SCALE });
    worker.postMessage({ kind: "run", id: runId, N, B, Mii: s.Mii, Mib: s.Mib, Sigma, k: SG.k, W0ii: W0.Wii, W0ib: W0.Wib, P: { ...PP } });
    busy = false; specVersion = -1; RO.pending(); stats.set("dir", null, "computing…");
  }
  // readout cadence: about twice a second, slower when a readout is expensive (it shares the CPU with the sim)
  let lastDispatchStep = 0;
  function jobDue() {
    if (!worker || busy) return false;
    if (st.mode === "mf" && !seq && s.t < EARLY_STEPS && s.t - lastDispatchStep >= EARLY_EVERY) return true;
    return performance.now() - lastDispatch > Math.max(450, (N * N * B > 1.6e6 ? 3 : 1.5) * lastJobMs);
  }
  function dispatch() {
    const havePrev = prevValid;
    if (!worker || busy || !s) return;
    const paused = !tp.playing || residual() < DONE_TOL;   // converged counts as paused (tour steps hold while converged)
    const wantSpec = specVersion !== version && (B <= 24 ? (paused || jobCount % 4 === 0) : paused);
    busy = true; lastDispatch = performance.now(); lastDispatchStep = s.t; jobCount++;
    const job = { kind: "job", seq: ++jobSeq, runId, version, t: simTime(), P: { ...PP }, Wii: s.Wii.slice(), Wib: s.Wib.slice(),
      wantSpec, specFamilies: B <= 24 ? ["both", "ii", "ib"] : ["both"], resid: residual() };
    if (havePrev) { job.prevWii = prevWii.slice(); job.prevWib = prevWib.slice(); }
    if (wantSpec) RO.specPending(B);
    worker.postMessage(job, [job.Wii.buffer, job.Wib.buffer, ...(havePrev ? [job.prevWii.buffer, job.prevWib.buffer] : [])]);
    job.versionAt = version; lastJob = job;
  }
  let lastJob = null;
  function onWorker(d) {
    if (d.kind === "run") return;
    busy = false;
    if (d.error) { RO.error(d.error); return; }
    if (d.stale || d.runId !== runId) { if (!tp.playing) dispatch(); return; }
    const res = d.res; lastJobMs = res.ms;
    lastResVersion = lastJob ? lastJob.versionAt : -1;
    if (res.spec) specVersion = lastResVersion;
    RO.show(res, lastJob ? lastJob.resid : NaN);
    const both = res.rows.both;
    series.rt.push(res.t); series.cosC.push(both.contrast.cos); series.cosK.push(both.known.cos); series.cosT.push(res.rows.transient ? res.rows.transient.known.cos : NaN); series.cosP.push(res.rows.parts.contrast.cos);
    stats.set("cos", both.contrast.cos, `masked least squares · BW ${f3(both.contrast.bw)}`);
    stats.set("ps", res.rows.parts.contrast.cos, `${res.partsK} part${res.partsK === 1 ? "" : "s"}${res.robust ? "" : " (not clustered yet)"} · BW ${f3(res.rows.parts.contrast.bw)}`);
    if (res.spec && res.spec.both) stats.set("dir", res.spec.both.rank, `of ${res.spec.p} in Sym(B), at t = ${ui.fmt(res.spec.t, 3)}`);
    RO.drawTime(series, seqMarks.map(m => ({ x: m.t, label: m.label })), simTime());
    // keep the readout in step with a paused run
    if (!tp.playing && (lastResVersion !== version || (specVersion !== version && ctx.visible))) dispatch();
  }

  // ------------------------------------------------------------------------------------------------ controls
  function moved() { chipsC.moved(); RO.chips.moved(); tour = null; seq = null; tp.setStatus(""); writeURL(); }
  function change(o) {
    Object.assign(st, o);
    moved();
    const sigmaKeys = ["fam", "b", "alpha", "dofr", "nb", "win", "btw", "uneq", "cs", "ell"];
    if (Object.keys(o).some(k => sigmaKeys.includes(k))) { SG = makeSigma(st); Sigma = SG.Sigma; B = SG.B; drawSigma(); makeTheta(); }
    if ("gamma" in o) makeTheta();
    if ("n" in o) wS.set(st.wmin);
    showFamControls();
    drawCaption();
    if (Object.keys(o).length === 1 && "theta" in o && st.cont) { PP.theta = st.theta; postRun(); if (!tp.playing) tp.play(); return; }
    if (Object.keys(o).length === 1 && "cont" in o) return;
    tp.reset(); if (!tp.playing) tp.play();
  }
  function syncControls() {
    famS.set(st.fam); bS.set(st.b); pS.set(st.p); nS.set(st.n); gS.set(st.gamma); wS.set(st.wmin); aS.set(st.alpha); dS.set(st.dofr); nbS.set(st.nb);
    wiS.set(st.win); btS.set(st.btw); uqT.set(st.uneq); csS.set(st.cs); seedS.set(st.seed); modeS.set(st.mode); batchS.set(st.batch); contT.set(st.cont);
    ellS.set(st.ell); jitT.set(st.jit);
    showFamControls();
  }
  function applyState(next, t) {
    st = next; tour = t;
    SG = makeSigma(st); Sigma = SG.Sigma; B = SG.B;
    syncControls(); makeTheta(); drawSigma();
    if (t) { (t.group === "c" ? chipsC : RO.chips).set(t.id); (t.group === "c" ? RO.chips : chipsC).clear(); } else { chipsC.clear(); RO.chips.clear(); }
    if (t) tp.setSpeed(t.group === "c" ? 16 : 1);   // structure tours run fast to convergence; readout tours at ×1 to show the transient
    tp.reset();
  }
  function pickTour(t) {
    applyState({ ...DEF, ...t.params }, t);
    const o = {}; for (const k of KEYS) o[k] = null;
    ctx.setParams({ ...o, tour: t.id });
    tp.play();
  }
  ctx.onParams(p => { const r = readState(p); applyState(r.st, r.tour); tp.play(); });
  ctx.onHide(() => v.setAutoRotate(false));

  // ------------------------------------------------------------------------------------------------ readout figure
  function buildReadoutFigure(host) {
    const top = ui.h("div", { class: "ro-top", style: { display: "flex", flexWrap: "wrap", gap: "12px 28px", alignItems: "flex-start", marginBottom: "10px", font: "13px/1.45 var(--sans)" } });
    const chips = ui.chips({ label: "Tours", items: visibleTours.filter(t => t.group === "r"), active: tour && tour.group === "r" ? tour.id : null, onPick: t => pickTour(t) });
    const setSeg = ui.segmented({ label: "show", options: [{ value: "known", label: "parameters known" }, { value: "contrast", label: "θ unknown: $P\\hat C P$ vs $P\\Sigma_g P$" }], value: "known", onChange: () => redraw() });
    const playBtn = ui.h("button", { type: "button", class: "u-chip", text: "play / pause the run above", style: { justifySelf: "start" }, onclick: () => tp.toggle() });
    const estSeg = ui.segmented({ label: "estimate", options: [{ value: "both", label: "masked least squares" }, { value: "parts", label: "parts-sum sketch" }], value: "both", onChange: () => redraw() });
    top.append(chips.el, ui.h("div", { style: { display: "grid", gap: "8px" } }, setSeg.el, estSeg.el, playBtn));
    const status = ui.h("div", { class: "callout", style: { margin: "6px 0 12px", padding: "8px 12px", fontSize: "14px" } });
    const grid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "16px 28px" } });
    const mk = title => { const b = ui.h("div"), w = ui.h("div", {}, ui.h("div", { class: "chart-title", html: title }), b); grid.append(w); return b; };
    const hS = mk("True $\\Sigma_g$ <span class=\"synthetic-tag\">synthetic Σ</span>"), hC = mk("Read out from the weights: $\\hat C$"), hD = mk("Difference $\\hat C - \\Sigma_g$");
    const tiles = ui.stats({ cols: window.innerWidth < 700 ? 2 : 4, items: [
      { key: "dir", label: "directions of $\\Sigma_g$ pinned down", fmt: String },
      { key: "eqs", label: "equations (active synapses)", fmt: ui.fmtInt },
      { key: "cos", label: "Frobenius cosine (this view)", fmt: f3 },
      { key: "bw", label: "Bures–Wasserstein (this view)", fmt: f3 }] });
    const tableHost = ui.h("div", { class: "table-scroll", style: { marginTop: "14px" } });
    const grid2 = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "16px 32px", marginTop: "14px" } });
    const mk2 = title => { const b = ui.h("div"), w = ui.h("div", {}, ui.h("div", { class: "chart-title", html: title }), b); grid2.append(w); return b; };
    const eigBox = mk2("Spectra: $\\Sigma_g$ against $\\hat C$"), measBox = mk2("What the weights can see: spectrum of the measurement operator"), timeBox = mk2("Readout quality over time");
    const measNote = ui.h("div", { class: "u-help" }); measBox.after(measNote);
    tiles.el.style.marginBottom = "14px";
    host.append(top, status, tiles.el, grid, tableHost, grid2);
    ui.renderMath(host);
    const common = { map: "div", size: 250, xlabel: "input c", ylabel: "input b" };
    const tip = name => (i, j, val) => ui.tipRows(`${gName(SG.order[i])}, ${gName(SG.order[j])}`, [{ name, value: f3(val) }]);
    const hmS = heatmap(hS, { ...common, zlabel: "$\\Sigma_g$", tooltip: tip("Σ_g") });
    const hmC = heatmap(hC, { ...common, zlabel: "$\\hat C$", tooltip: tip("Ĉ") });
    const hmD = heatmap(hD, { ...common, zlabel: "$\\hat C - \\Sigma_g$", tooltip: tip("Ĉ − Σ_g") });
    const eigC = lineChart(eigBox, { series: [{ name: "Σ_g", y: [1], x: [1] }], ylog: true, height: 230, xlabel: "rank k", ylabel: "eigenvalue" });
    const measC = lineChart(measBox, { series: [{ name: "I–I + I–B", y: [1], x: [1] }], ylog: true, xlog: true, yrange: [1e-16, 2], height: 230, xlabel: "rank among the B(B+1)/2 directions (log)", ylabel: "eigenvalue / largest",
      yfmt: y => (Math.round(Math.log10(y)) % 4 === 0 ? sci(y) : ""), refs: [{ y: SPEC_TOL, label: "pinned-down threshold" }] });
    const timeC = lineChart(timeBox, { series: [{ name: "θ unknown: cosine", y: [], x: [] }], height: 230, yrange: [-0.2, 1.02], xlabel: "time t", ylabel: "Frobenius cosine", xname: "t" });
    const timeNote = ui.h("div", { class: "u-help" }); timeBox.after(timeNote);
    let last = null, lastSpec = null;
    const cell = x => `<td class="num">${x == null || !Number.isFinite(x) ? "n/a" : f3(x)}</td>`;
    const ROWS = [
      ["both", "<b>Masked least squares</b> on the weights (I–I and I–B synapses)"],
      ["ii", "… I–I synapses only"],
      ["ib", "… I–B synapses only"],
      ["parts", "Parts-sum sketch $\\theta\\mathbf 1\\mathbf 1^\\top + \\sum_k c_k h_k h_k^\\top/\\lVert h_k\\rVert^2$"],
      ["transient", "Derivative-aware: weights and $dw/dt$ of the last step"],
      ["naive", "The note's formula $\\hat C = H^+RH^{+\\top}$"],
      ["untrained", "Untrained network, read as if at equilibrium (control)"],
      ["identity", "Baseline: identity"],
      ["diag", "Baseline: $\\mathrm{diag}(\\Sigma_g)$"],
      ["rankK", "Baseline: rank-$k$ truncation of $\\Sigma_g$ (oracle)"],
    ];
    function table(res, resid) {
      const r = res.rows;
      const head = `<thead><tr><th rowspan="2">estimate $\\hat C$</th><th colspan="2" class="num">parameters known</th><th colspan="2" class="num">unknown scale</th><th colspan="3" class="num">$\\theta$ unknown: contrast subspace</th></tr>
        <tr><th class="num">cosine</th><th class="num">BW</th><th class="num">rel. error</th><th class="num">BW</th><th class="num">cosine</th><th class="num">BW</th><th class="num">overlap, $k = ${SG.k}$</th></tr></thead>`;
      const body = ROWS.map(([key, label]) => {
        const x = r[key];
        if (!x) return `<tr><td>${label}</td><td colspan="7" class="num">${key === "transient" ? "needs two consecutive steps (play or step the run)" : "n/a (γ = 0)"}</td></tr>`;
        return `<tr${key === "both" ? ' style="background: var(--wash)"' : ""}><td>${label}</td>${cell(x.known.cos)}${cell(x.known.bw)}${cell(x.scale.rel)}${cell(x.scale.bw)}${cell(x.contrast.cos)}${cell(x.contrast.bw)}${cell(x.contrast.sub)}</tr>`;
      }).join("");
      tableHost.innerHTML = `<table class="data">${head}<tbody>${body}</tbody></table>`;
      ui.renderMath(tableHost);
    }
    let lastResid = NaN;
    function drawStatus() {
      if (!last) return;
      const res = last, resid = lastResid;
      const ok = resid < EQ_TOL;
      status.className = ok ? "callout key" : "callout caveat";
      status.innerHTML = ok
        ? `<b>Equilibrium assumption holds</b> at t = ${ui.fmt(res.t, 3)}: $r/\\eta = ${sci(resid)} < 10^{-6}$. Every active synapse satisfies its fixed-point equation, so the readout below uses only the weights${setSeg.get() === "known" ? " and the known $\\theta$, $\\lambda/\\eta$, $\\gamma$" : ", $\\lambda/\\eta$ and $\\gamma$ (not $\\theta$)"}.`
        : `<b>Equilibrium assumption violated</b> at t = ${ui.fmt(res.t, 3)}: $r/\\eta = ${sci(resid)} \\ge 10^{-6}$. The equilibrium readout below assumes $dw/dt = 0$ and is a control here; the derivative-aware row uses the observed $dw/dt$ and is exact away from equilibrium${st.mode === "mb" ? " (in minibatch mode, exact for the minibatch's covariance, not for $\\Sigma_g$)" : ""}.`;
      ui.renderMath(status);
    }
    function redraw() {
      drawStatus();
      const mode = setSeg.get(), o = SG.order;
      const S = mode === "contrast" ? E.project1(Sigma, B) : Sigma;
      const m = Math.max(1e-9, ...Array.from(S, Math.abs));
      const perm = A => { const X = new Float32Array(B * B); for (let a = 0; a < B; a++) for (let c = 0; c < B; c++) X[a * B + c] = A[o[a] * B + o[c]]; return X; };
      const groups = groupsOf(), lab = o.map(b => gName(b)), clab = o.map(b => String(b + 1));
      hmS.update({ values: perm(S), rows: B, cols: B, range: [-m, m], groups, rowLabels: lab, colLabels: clab, zlabel: mode === "contrast" ? "$P\\Sigma_g P$" : "$\\Sigma_g$" });
      if (!last) { hmC.update({ values: new Float32Array(B * B).fill(NaN), rows: B, cols: B, range: [-m, m], groups }); hmD.update({ values: new Float32Array(B * B).fill(NaN), rows: B, cols: B, range: [-m, m], groups }); return; }
      const est = estSeg.get(), x = last.rows[est][mode === "contrast" ? "contrast" : "known"];
      const C = est === "parts" ? (mode === "contrast" ? E.project1(last.Cps, B).map(v0 => v0 * x.scale) : last.Cps)
        : (mode === "contrast" ? last.Cp.map(v0 => v0 * x.scale) : last.Ck);
      const D = new Float64Array(B * B); for (let q = 0; q < B * B; q++) D[q] = C[q] - S[q];
      hmC.update({ values: perm(C), rows: B, cols: B, range: [-m, m], groups, rowLabels: lab, colLabels: clab, zlabel: mode === "contrast" ? "$s\\,P\\hat C P$ (best scale $s$)" : "$\\hat C$" });
      hmD.update({ values: perm(D), rows: B, cols: B, range: [-m, m], groups, rowLabels: lab, colLabels: clab, zlabel: "difference" });
      tiles.set("cos", x.cos, mode === "contrast" ? "θ unknown" : "parameters known");
      tiles.set("bw", x.bw, x.neg > 1e-6 ? `negative mass clipped: ${ui.fmtPct(x.neg, 0)}` : "");
      const ks = Array.from({ length: B }, (_, k) => k + 1);
      const es = mode === "contrast" ? last.eig.sigmaP : last.eig.sigma;
      const eh = est === "parts" ? E.eigh(mode === "contrast" ? E.project1(last.Cps, B) : last.Cps, B).values.map(v0 => v0 * (mode === "contrast" ? x.scale : 1))
        : mode === "contrast" ? last.eig.proj.map(v0 => v0 * x.scale) : last.eig.known;
      eigC.update({ series: [{ name: mode === "contrast" ? "PΣ_gP" : "Σ_g", x: ks, y: Array.from(es), color: PAL.fg, dots: true },
        { name: mode === "contrast" ? "s·PĈP" : "Ĉ", x: ks, y: Array.from(eh), color: PAL.cat[0], points: true }] });
    }
    function drawSpec() {
      const sp = lastSpec;
      if (!sp) return;
      const ser = [];
      const xs = n => Array.from({ length: n }, (_, k) => k + 1);
      if (sp.both) ser.push({ name: "I–I + I–B", x: xs(sp.p), y: Array.from(sp.both.norm, z => Math.max(z, 1e-17)), color: PAL.fg });
      if (sp.ii) ser.push({ name: "I–I only", x: xs(sp.p), y: Array.from(sp.ii.norm, z => Math.max(z, 1e-17)), color: PAL.cat[0], dash: true, width: 1.5 });
      if (sp.ib) ser.push({ name: "I–B only", x: xs(sp.p), y: Array.from(sp.ib.norm, z => Math.max(z, 1e-17)), color: PAL.cat[1], dash: true, width: 1.5 });
      measC.update({ series: ser });
      tiles.set("dir", sp.both.rank, `of ${sp.p} · ${sp.ii ? `I–I alone ${sp.ii.rank}, I–B alone ${sp.ib.rank} · ` : ""}at t = ${ui.fmt(sp.t, 3)}`);
      measNote.innerHTML = `Pinned down: eigenvalue of $A^\\top A$ above ${sci(SPEC_TOL)} of the largest. At a looser 10⁻¹⁰ the count is ${sp.both.rankLoose}. Equations: ${ui.fmtInt(sp.mII)} I–I, ${ui.fmtInt(sp.mIB)} I–B.`;
      ui.renderMath(measNote);
    }
    const api = {
      chips,
      setSigma() { last = null; lastSpec = null; redraw(); tiles.update({ dir: null, eqs: null, cos: null, bw: null }); measC.update({ series: [{ name: "I–I + I–B", y: [1], x: [1], color: PAL.fg }] }); tableHost.innerHTML = ""; status.textContent = "Waiting for the first readout…"; },
      pending() { /* a new run: keep the last picture until the first result arrives */ },
      specPending(b) { if (b >= 32) measNote.textContent = `Computing the measurement spectrum for B = ${b} (${b * (b + 1) / 2} directions; a few seconds)…`; },
      error(msg) { status.textContent = `Readout failed: ${msg}`; },
      show(res, resid) {
        last = res; lastResid = resid; if (res.spec) { lastSpec = res.spec; drawSpec(); }
        tiles.set("eqs", res.mII + res.mIB, `${ui.fmtInt(res.mII)} I–I, ${ui.fmtInt(res.mIB)} I–B`);
        table(res, resid); redraw();
      },
      drawTime(sr, refs, t) {
        timeC.update({ series: [
          { name: "least squares, θ unknown", x: sr.rt, y: sr.cosC, color: PAL.cat[0] },
          { name: "least squares, parameters known", x: sr.rt, y: sr.cosK, color: PAL.fg },
          { name: "parts-sum, θ unknown", x: sr.rt, y: sr.cosP, color: PAL.cat[1] },
          { name: "derivative-aware", x: sr.rt, y: sr.cosT, color: PAL.cat[2], dash: true }], refs });
        timeC.setCursor(t);
        let bi = -1; sr.cosC.forEach((c, i) => { if (Number.isFinite(c) && (bi < 0 || c > sr.cosC[bi])) bi = i; });
        const n = sr.cosC.length;
        timeNote.innerHTML = bi < 0 ? "" : `Least squares, θ unknown: best ${f3(sr.cosC[bi])} at t = ${ui.fmt(sr.rt[bi], 3)}; latest ${f3(sr.cosC[n - 1])} at t = ${ui.fmt(sr.rt[n - 1], 3)}.`;
      },
      cursor(t) { timeC.setCursor(t); },
      clearTime() { timeC.update({ series: [{ name: "least squares, θ unknown", y: [], x: [] }] }); timeNote.textContent = ""; },
    };
    return api;
  }

  // ------------------------------------------------------------------------------------------------ comparison (dense vs sparse)
  // Runs automatically once, when the table first scrolls into view, with the default setting; the button re-runs it
  // with whatever is selected above. Separate worker; the live run is untouched.
  const cmpHost = ctx.slot("compare");
  let cmpWorker = null;
  function runCompare(useCurrent) {
    const out = cmpHost.querySelector(".u-help"), note = panel.querySelector("[data-cmp-note]");
    if (typeof Worker === "undefined") { out.textContent = "Web workers are unavailable in this browser; the numbers shown were measured earlier."; return; }
    if (cmpWorker) cmpWorker.terminate();
    const cs = useCurrent ? st : { ...DEF }, sg = useCurrent ? SG : makeSigma(cs);
    panel.querySelectorAll("[data-cmp]").forEach(el => { el.style.opacity = "0.4"; });
    out.textContent = `Running three networks to convergence in a background thread (B = ${sg.B}, N = ${cs.n}, θ = ${f2(cs.theta)}); the rows fill in as they finish…`;
    const desc = `${sg.desc}; B = ${sg.B}, N = ${cs.n}, θ = ${f2(cs.theta)}, γ = ${cs.gamma}, w_min = ${ui.fmt(cs.wmin, 2)} (last row ${ui.fmt(cs.wmin / 100, 2)}), init seed ${cs.seed}, mean-field, dt = ${DT}, run until r/η < 10⁻⁷.`;
    cmpWorker = new Worker(new URL("./ch_many_worker.js", import.meta.url), { type: "module" });
    cmpWorker.onmessage = e => {
      const d = e.data; if (d.kind !== "compare") return;
      if (d.error) { out.textContent = `Comparison failed: ${d.error}`; return; }
      if (d.row) fillCompare(d.row);
      if (d.done) { cmpWorker.terminate(); cmpWorker = null; out.textContent = ""; if (note) note.textContent = `Computed in your browser just now: ${desc}`; }
    };
    cmpWorker.postMessage({ kind: "compare", Sigma: sg.Sigma, B: sg.B, N: cs.n, theta: cs.theta, gamma: cs.gamma, wmin: cs.wmin, seed: cs.seed, k: sg.k,
      configs: [{ key: "d", p: 1 }, { key: "s", p: 0.5 }, { key: "f", p: 1, wmin: cs.wmin / 100 }] });
  }
  if (cmpHost) {
    const btn = ui.h("button", { type: "button", class: "explore-link", text: "Re-run this comparison with the settings selected above" });
    cmpHost.append(btn, ui.h("div", { class: "u-help", style: { marginTop: "6px" } }));
    btn.addEventListener("click", () => runCompare(true));
    const io = new IntersectionObserver(es => { if (es.some(x => x.isIntersecting)) { io.disconnect(); runCompare(false); } }, { rootMargin: "200px" });
    io.observe(cmpHost);
  }
  function fillCompare(r) {
    const key = r.p === "id" ? "b" : r.key;
    const vals = r.p === "id" ? { cosK: f3(r.cosK), cosC: f3(r.cosC), bwC: f3(r.bwC), sub: f3(r.sub) }
      : { K: String(r.K), dead: String(r.dead), dir: `${r.dir} of ${r.pdim}`, cosK: f3(r.cosK), cosC: f3(r.cosC), bwC: f3(r.bwC), sub: f3(r.sub), ps: f3(r.psC),
        peak: `${f3(r.peak)} (t = ${ui.fmt(r.peakT, 2)})` };
    for (const [f, val] of Object.entries(vals)) {
      panel.querySelectorAll(`[data-cmp="${key}-${f}"]`).forEach(el => { el.textContent = val; el.style.opacity = ""; });
    }
  }

  // ------------------------------------------------------------------------------------------------ start
  drawSigma();
  tp.reset();
  if (tour) { (tour.group === "c" ? chipsC : RO.chips).set(tour.id); }
}
