// Part C of the primer (slides 8-11): readout (a computed toy recovery + rank against N from data/big), nonlinear neurons
// (hoped-for against measured), knobs, the end. Static SVG, redrawn on input.
const NS = "http://www.w3.org/2000/svg";
const FONT = "var(--font-sans, Inter, sans-serif)";

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function text(parent, x, y, s, opts = {}) {
  const t = el("text", {
    x, y, "text-anchor": opts.anchor || "middle",
    style: `font-family:${FONT};font-size:${opts.size || 15}px;fill:${opts.fill || "var(--fg)"};font-weight:${opts.weight || 500}`,
  }, parent);
  t.textContent = s;
  return t;
}
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }

// Deterministic PRNG so the figure is stable.
function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

// Diverging colour for value v in [-1, 1] using --c-div-0..8.
function divColor(v) {
  const i = Math.max(0, Math.min(8, Math.round((v + 1) * 4)));
  return `var(--c-div-${i})`;
}
function seqColor(v) {
  const i = Math.max(0, Math.min(8, Math.round(v * 8)));
  return `var(--c-seq-${i})`;
}

// A matrix as an SVG heatmap (rows x cols cells in a unit-square viewBox, so non-square matrices fill the slot).
function heatSvg(M, colorFn, aria) {
  const R = M.length, Cn = M[0].length;
  const svg = el("svg", { viewBox: "0 0 100 100", role: "img", "aria-label": aria, preserveAspectRatio: "none" });
  const cw = 100 / Cn, ch = 100 / R;
  for (let i = 0; i < R; i++) for (let j = 0; j < Cn; j++) {
    el("rect", { x: j * cw, y: i * ch, width: cw + 0.15, height: ch + 0.15, fill: colorFn(M[i][j]) }, svg);
  }
  el("rect", { x: 0, y: 0, width: 100, height: 100, fill: "none", stroke: "var(--line)", "vector-effect": "non-scaling-stroke" }, svg);
  return svg;
}

// Dense linear algebra for the toy below (small sizes only).
function solveLin(A, b) { // Gaussian elimination with partial pivoting; A is n x n
  const n = A.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r, i) => r[n] / r[i]);
}
function leastSquares(rows, y, m) { // Householder QR; returns {x | null, rank}
  const E = rows.length, A = rows.map((r) => r.slice()), b = y.slice(), d = [];
  for (let k = 0; k < m && k < E; k++) {
    let nrm = 0;
    for (let i = k; i < E; i++) nrm += A[i][k] ** 2;
    nrm = Math.sqrt(nrm);
    const al = A[k][k] > 0 ? -nrm : nrm, v = [];
    for (let i = k; i < E; i++) v.push(A[i][k]);
    v[0] -= al;
    const vv = v.reduce((s, t) => s + t * t, 0);
    if (vv > 0) {
      for (let j = k; j < m; j++) { let s = 0; for (let i = k; i < E; i++) s += v[i - k] * A[i][j]; s *= 2 / vv; for (let i = k; i < E; i++) A[i][j] -= s * v[i - k]; }
      let s = 0; for (let i = k; i < E; i++) s += v[i - k] * b[i]; s *= 2 / vv; for (let i = k; i < E; i++) b[i] -= s * v[i - k];
    }
    d.push(Math.abs(A[k][k]));
  }
  const mx = Math.max(...d), rank = d.filter((t) => t > 1e-9 * mx).length;
  if (rank < m) return { x: null, rank };
  const x = Array(m).fill(0);
  for (let k = m - 1; k >= 0; k--) { let s = b[k]; for (let j = k + 1; j < m; j++) s -= A[k][j] * x[j]; x[k] = s / A[k][k]; }
  return { x, rank };
}

// Slide 8, left: a real (toy) readout. N linear neurons, n inputs in three blocks, input covariance C.
//  1. Weights: the distance rule's fixed point, w_ij = (b - ½ d_ij)₊ / α with d_ij = (h_i - h_j)ᵀ C (h_i - h_j),
//     H = (L + diag(A1))⁻¹ A (L the Laplacian of W, A the attachments), by damped fixed-point iteration.
//  2. Readout, from W, A, b, α only: each live synapse gives (h_i - h_j)ᵀ X (h_i - h_j) = 2(b - α w_ij); solve by
//     least squares for X = PCP (zero row sums, n(n-1)/2 unknowns). Deterministic; exact up to round-off.
function readoutToy({ N = 18, n = 9, b = 0.08, alpha = 1, seed = 3 } = {}) {
  const r = rng(seed), blk = (i) => Math.floor(i / 3), strength = [1.0, 0.7, 0.45];
  const C = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) =>
    blk(i) === blk(j) ? strength[blk(i)] * (i === j ? 1 : 0.8) : 0.05));
  // attachments: every neuron to every input, three times more strongly to its own block
  const A = Array.from({ length: N }, (_, i) => Array.from({ length: n }, (_, k) => (blk(k) === Math.floor(3 * i / N) ? 1 : 0.3) * r()));
  const a1 = A.map((row) => row.reduce((s, v) => s + v, 0));
  const embed = (W) => {
    const M = W.map((row, i) => row.map((w, j) => (i === j ? row.reduce((s, v) => s + v, 0) + a1[i] : -w)));
    const cols = Array.from({ length: n }, (_, k) => solveLin(M, A.map((row) => row[k])));
    return Array.from({ length: N }, (_, i) => cols.map((c) => c[i]));
  };
  const dist = (H, i, j) => {
    let s = 0;
    for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) s += (H[i][a] - H[j][a]) * C[a][c] * (H[i][c] - H[j][c]);
    return s;
  };
  const W = Array.from({ length: N }, () => Array(N).fill(0));
  let H, step = 1;
  for (let t = 0; t < 3000 && step > 1e-13; t++) {
    H = embed(W); step = 0;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const w = 0.7 * W[i][j] + 0.3 * Math.max(0, b - dist(H, i, j) / 2) / alpha;
      step = Math.max(step, Math.abs(w - W[i][j])); W[i][j] = W[j][i] = w;
    }
  }
  H = embed(W);
  // readout: unknowns x_ac (a < c), X = sum x_ac (e_a - e_c)(e_a - e_c)ᵀ, so dᵀXd = sum x_ac (d_a - d_c)²
  const pairs = [];
  for (let a = 0; a < n; a++) for (let c = a + 1; c < n; c++) pairs.push([a, c]);
  const rows = [], rhs = [];
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    if (W[i][j] <= 1e-12) continue;
    const d = H[i].map((v, a) => v - H[j][a]);
    rows.push(pairs.map(([a, c]) => (d[a] - d[c]) ** 2)); rhs.push(2 * (b - alpha * W[i][j]));
  }
  const { x, rank } = leastSquares(rows, rhs, pairs.length);
  const rowM = C.map((row) => row.reduce((s, v) => s + v, 0) / n), all = rowM.reduce((s, v) => s + v, 0) / n;
  const truth = C.map((row, i) => row.map((v, j) => v - rowM[i] - rowM[j] + all));
  let rec = null;
  if (x) {
    rec = Array.from({ length: n }, () => Array(n).fill(0));
    pairs.forEach(([a, c], p) => { rec[a][c] -= x[p]; rec[c][a] -= x[p]; rec[a][a] += x[p]; rec[c][c] += x[p]; });
  }
  let err = 0, nrm = 0;
  if (rec) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { err += (rec[i][j] - truth[i][j]) ** 2; nrm += truth[i][j] ** 2; }
  return { W, H, truth, rec, rank, unknowns: pairs.length, live: rows.length, synapses: N * (N - 1) / 2, relErr: rec ? Math.sqrt(err / nrm) : null, step };
}

// The four panels as an HTML grid (labels stay legible on phones: 4 across on wide screens, 2 x 2 on narrow ones).
function drawReadout(host, stat) {
  const T = readoutToy();
  const wMax = Math.max(...T.W.flat()), hMax = Math.max(...T.H.flat()), cMax = Math.max(...T.truth.flat().map(Math.abs));
  const cell = (tag, M, colorFn, label, aria) => {
    const d = document.createElement("div"); d.className = "rd-cell";
    const t = document.createElement("div"); t.className = "rd-tag"; t.textContent = tag;
    const l = document.createElement("div"); l.className = "rd-lab"; l.innerHTML = label;
    d.append(t, heatSvg(M, colorFn, aria), l);
    return d;
  };
  const arrow = (s) => { const a = document.createElement("div"); a.className = "rd-arrow"; a.innerHTML = `<span>${s}</span>→`; return a; };
  const empty = T.truth.map((row) => row.map(() => 0));
  host.replaceChildren(
    cell("learned", T.W.map((row) => row.map((v) => v / wMax)), seqColor, "weights <i>W</i>", "weights W of the toy network at the rule's fixed point"),
    arrow("+ attachments"),
    cell("computed", T.H.map((row) => row.map((v) => v / hMax)), seqColor, "embeddings <i>H</i>", "embeddings H computed from W and the attachments"),
    arrow("+ rule, solve"),
    cell("inferred", (T.rec || empty).map((row) => row.map((v) => v / cMax)), divColor, "recovered <i>PCP</i>", "contrast covariance recovered from the live synapses"),
    arrow("vs"),
    cell("actual", T.truth.map((row) => row.map((v) => v / cMax)), divColor, "true <i>PCP</i>", "true contrast covariance"),
  );
  if (stat) {
    stat.textContent = T.rec
      ? `${T.live} of ${T.synapses} synapses live · rank ${T.rank}/${T.unknowns} · error ${T.relErr.toExponential(0)}`
      : `rank ${T.rank} of ${T.unknowns}: not identifiable`;
  }
}

// Slide 10: 30 neurons on a line, positions = 1-D "response". Group by gaps > threshold(b).
function makeKnobs(svg, bIn, aIn) {
  const r = rng(3);
  const centres = [0.08, 0.22, 0.31, 0.5, 0.58, 0.77, 0.9];
  const base = [];
  for (let i = 0; i < 30; i++) {
    const c = centres[i % centres.length];
    base.push({ c, off: (r() - 0.5) * 0.09 });
  }
  function draw() {
    clear(svg);
    const b = bIn.value / 100, alpha = aIn.value / 100;
    // tightness: low alpha -> points pulled hard toward their group mean.
    const pts = base.map((p) => p.c + p.off).sort((x, y) => x - y);
    const thr = 0.005 + b * b * 0.35; // larger b: fuse across larger gaps (fewer groups)
    const groups = [[pts[0]]];
    for (let i = 1; i < pts.length; i++) {
      if (pts[i] - pts[i - 1] <= thr) groups[groups.length - 1].push(pts[i]);
      else groups.push([pts[i]]);
    }
    const pull = 1 - (0.15 + 0.8 * alpha); // alpha small -> pull ~0.85 (tight)
    const X0 = 40, X1 = 600;
    el("line", { x1: X0, y1: 170, x2: X1, y2: 170, stroke: "var(--line)", "stroke-width": 1.5 }, svg);
    text(svg, X0, 236, "neuron response along one input direction (1-D toy) →", { anchor: "start", size: 14, fill: "var(--mute)" });
    groups.forEach((gp, k) => {
      const m = gp.reduce((a, c) => a + c, 0) / gp.length;
      const col = `var(--c-cat-${k % 8})`;
      const xs = gp.map((v) => X0 + (X1 - X0) * (v + (m - v) * pull));
      const lo = Math.min(...xs), hi = Math.max(...xs);
      el("rect", { x: lo - 12, y: 135, width: hi - lo + 24, height: 70 - 0, rx: 14, fill: col, "fill-opacity": 0.12, stroke: col, "stroke-opacity": 0.5 }, svg);
      gp.forEach((v, i) => {
        el("circle", { cx: xs[i], cy: 155 + (i % 3) * 13, r: 6.5, fill: col, stroke: "var(--bg, #fff)", "stroke-width": 1.2 }, svg);
      });
    });
    const tightLbl = alpha < 0.35 ? "tight, sticky" : alpha < 0.7 ? "medium" : "loose, little memory";
    text(svg, 320, 60, `${groups.length} group${groups.length === 1 ? "" : "s"}`, { size: 30, weight: 700 });
    text(svg, 320, 92, `resolution from b · ${tightLbl} from α`, { size: 15, fill: "var(--mute)" });
    text(svg, 320, 278, "small b: many small groups   ·   large b: few big groups", { size: 15, fill: "var(--mute)" });
  }
  bIn.addEventListener("input", draw);
  aIn.addEventListener("input", draw);
  draw();
}

// Slide 9: response curves as functions of one shared pattern: hoped-for diversity against the measured sketch.
function initFan(root, ctx) {
  const svg = root.querySelector(".tb-fan");
  if (!svg) return;
  const CAT = (k) => `var(--c-cat-${k % 8})`;
  const cap = root.querySelector(".tb-fancap");
  const bH = root.querySelector(".tb-hope"), bM = root.querySelector(".tb-meas");
  const x0 = 60, x1 = 616, y0 = 24, y1 = 340, zmin = -2, zmax = 2;
  const X = (z) => x0 + (z - zmin) / (zmax - zmin) * (x1 - x0), Yp = (v) => y1 - v * (y1 - y0);
  el("line", { x1: x0, y1, x2: x1, y2: y1, stroke: "var(--mute)", "stroke-width": 1.5 }, svg);
  el("line", { x1: x0, y1: y0, x2: x0, y2: y1, stroke: "var(--mute)", "stroke-width": 1.5 }, svg);
  text(svg, (x0 + x1) / 2, 380, "shared input pattern z", { size: 15 });
  const yl = text(svg, 22, (y0 + y1) / 2, "neuron response", { size: 15 });
  yl.setAttribute("transform", `rotate(-90 22 ${(y0 + y1) / 2})`);
  const sig = (u) => 1 / (1 + Math.exp(-u)), sp = (u) => Math.log1p(Math.exp(u));
  const K = 7;
  const hope = [
    (z) => sig(4 * (z + 1.2)), (z) => sig(2.5 * (z - 0.2)), (z) => sig(6 * (z - 1.1)),
    (z) => sp(2 * (z + 0.3)) / sp(4.6), (z) => 1 - sig(3 * (z - 0.6)), (z) => Math.exp(-((z + 0.5) ** 2) / 0.35),
    (z) => Math.max(0, z - 1),
  ];
  const proto = (z) => sp(1.6 * (z - 0.1)) / sp(1.6 * 1.9);
  const meas = Array.from({ length: K }, (_, k) => (z) => proto(z) * (1 + 0.025 * Math.sin(1.7 * k + 2.3 * z)));
  const paths = Array.from({ length: K }, (_, k) => el("path", { fill: "none", stroke: CAT(k), "stroke-width": 3, "stroke-linecap": "round" }, svg));
  let t = 0, target = 0, raf = 0;
  function draw() {
    paths.forEach((p, k) => {
      let d = "";
      for (let i = 0; i <= 120; i++) {
        const z = zmin + (zmax - zmin) * i / 120;
        const v = (1 - t) * hope[k](z) + t * meas[k](z);
        d += `${i ? "L" : "M"}${X(z).toFixed(1)},${Yp(Math.min(1.02, Math.max(0, v))).toFixed(1)}`;
      }
      p.setAttribute("d", d);
    });
    cap.textContent = target
      ? "Measured (a sketch of the result, not data): members of one group, each rescaled to its group's mean response; the shapes nearly coincide. In the precomputed runs the part a member does not share with its group is about 1–2% of its variance (compact groups, spread thresholds)."
      : "The hoped-for picture (illustration): neurons in one group respond to the same pattern, each with its own threshold and shape.";
  }
  function anim() {
    raf = 0;
    const dt = target - t;
    t = Math.abs(dt) < 0.02 || !ctx.visible ? target : t + dt * 0.12;
    draw();
    if (t !== target) raf = requestAnimationFrame(anim);
  }
  function set(m) {
    target = m; bH.setAttribute("aria-pressed", String(!m)); bM.setAttribute("aria-pressed", String(!!m));
    if (!raf) raf = requestAnimationFrame(anim);
  }
  bH.addEventListener("click", () => set(0));
  bM.addEventListener("click", () => set(1));
  draw();
}

// Slide 8, right: rank of the live-edge operator (of 66) against N, three b (data/big/readout_vs_N.json).
async function drawRank(svg) {
  let d;
  try {
    const r = await fetch("data/big/readout_vs_N.json");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    d = await r.json();
  } catch (e) {
    text(svg, 210, 150, `data not loaded (${e.message})`, { size: 14, fill: "var(--mute)" });
    return;
  }
  const dim = d.meta?.config?.contrast_dim || 66;
  const Ns = d.meta?.config?.Ns || [40, 80, 160, 320, 500], bs = d.meta?.config?.bs || [0.01, 0.036, 0.1];
  const L = 58, R = 392, T = 30, Bt = 240;
  const lx = (n) => L + (Math.log(n) - Math.log(Ns[0])) / (Math.log(Ns[Ns.length - 1]) - Math.log(Ns[0])) * (R - L);
  const ly = (v) => Bt - v / dim * (Bt - T);
  el("line", { x1: L, y1: Bt, x2: R, y2: Bt, stroke: "var(--mute)" }, svg);
  el("line", { x1: L, y1: T, x2: L, y2: Bt, stroke: "var(--mute)" }, svg);
  el("line", { x1: L, y1: ly(dim), x2: R, y2: ly(dim), stroke: "var(--mute)", "stroke-dasharray": "4 4" }, svg);
  text(svg, L + 4, ly(dim) - 6, `full rank (${dim})`, { anchor: "start", size: 14, fill: "var(--mute)" });
  [0, 33, 66].forEach((v) => text(svg, L - 6, ly(v) + 4, String(v), { anchor: "end", size: 14, fill: "var(--mute)" }));
  Ns.forEach((n) => text(svg, lx(n), Bt + 18, String(n), { size: 14, fill: "var(--mute)" }));
  text(svg, (L + R) / 2, Bt + 40, "neurons N (12 inputs)", { size: 15, fill: "var(--fg)" });
  const yl = text(svg, 16, (T + Bt) / 2, "rank", { size: 15 });
  yl.setAttribute("transform", `rotate(-90 16 ${(T + Bt) / 2})`);
  bs.forEach((b, k) => {
    const rows = Ns.map((n) => d.summary.find((s) => s.b === b && s.N === n)).filter(Boolean);
    const col = `var(--c-cat-${k})`;
    el("path", { d: rows.map((s, q) => `${q ? "L" : "M"}${lx(s.N).toFixed(1)},${ly(s.rank[0]).toFixed(1)}`).join(""), fill: "none", stroke: col, "stroke-width": 2.5 }, svg);
    rows.forEach((s) => el("circle", { cx: lx(s.N), cy: ly(s.rank[0]), r: 4, fill: col }, svg));
    const first = rows.find((s) => s.rank[0] >= dim - 1e-9);
    const leg = `b = ${b}` + (first ? `: full rank at N = ${first.N}, cos ${first.cos_Chat[0].toFixed(3)}` : "");
    text(svg, 14, Bt + 70 + 20 * k, leg, { anchor: "start", size: 14, fill: col, weight: 600 });   // legend below the axes
  });
}

export function init(root, ctx) {
  const q = (s) => root.querySelector(s);
  const ro = q("#primer-c-readout");
  if (ro) drawReadout(ro, q("#primer-c-readout-stat"));
  const ks = q("#primer-c-knobs-svg"), bIn = q("#primer-c-b"), aIn = q("#primer-c-alpha");
  if (ks && bIn && aIn) makeKnobs(ks, bIn, aIn);
  const rk = q("#primer-c-rank-svg");
  if (rk) drawRank(rk);
  initFan(root, ctx);
}
