// Chapter "rates", live figure: the linear rule at theta = rho* against the rectified rule (tau = 0) at
// theta = k~_0(rho*), on a synthetic ring covariance, from the same initial weights (as hebb/nonlin/synth.py).
import * as ui from "./ui.js";
import { heatmap } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as E from "./engine.js";

const B = 24, N = 96, ELL = 2, T_END = 4000, WMIN = 1e-3, DT = 0.05, HMAX = 0.4;

function ringSigma() {
  const S = new Float64Array(B * B);
  for (let a = 0; a < B; a++) for (let b = 0; b < B; b++) { let d = Math.abs(a - b); d = Math.min(d, B - d); S[a * B + b] = Math.exp(-d * d / (2 * ELL * ELL)); }
  return S;
}

/** live/dead analysis of one state: dead = every synapse <= 2 wmin; live clusters exclude dead-majority clusters */
function analyse(s, H) {
  const dead = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    let mx = 0;
    for (let b = 0; b < B; b++) mx = Math.max(mx, s.Wib[i * B + b]);
    for (let j = 0; j < N; j++) mx = Math.max(mx, s.Wii[i * N + j]);
    dead[i] = mx <= 2 * WMIN ? 1 : 0;
  }
  const cl = E.clustersAuto(H, N, B);
  if (!cl.robust && cl.K > N / 4) {   // no clear gap yet: still unclustered (early transient)
    const vals = new Float32Array(N * B); vals.set(H);
    return { K: null, nDead: dead.reduce((a, x) => a + x, 0), sizes: [], vals, groups: [], order: Array.from({ length: N }, (_, i) => i) };
  }
  const groups = new Map();
  for (let i = 0; i < N; i++) { const k = cl.labels[i]; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(i); }
  const live = [], deadIdx = [];
  for (const m of groups.values()) {
    const nd = m.reduce((a, i) => a + dead[i], 0);
    if (nd * 2 >= m.length) deadIdx.push(...m); else live.push(m);
  }
  live.sort((a, b) => b.length - a.length);
  const order = [].concat(...live, deadIdx);
  const rowsG = []; let start = 0;
  live.forEach((m, k) => { rowsG.push({ start, end: start + m.length, color: k < 8 ? catColor(k) : P.other, label: "", name: `c${k + 1}` }); start += m.length; });
  if (deadIdx.length) rowsG.push({ start, end: start + deadIdx.length, color: P.other, label: "", name: "dead" });
  const vals = new Float32Array(N * B);
  order.forEach((i, r) => { for (let b = 0; b < B; b++) vals[r * B + b] = H[i * B + b]; });
  return { K: live.length, nDead: dead.reduce((a, x) => a + x, 0), sizes: live.map(m => m.length), vals, groups: rowsG, order };
}

export function ringFigure(host, ctx) {
  const Sig = ringSigma(), k1 = E.reluCorr(1), SigR = Sig.map(x => x / k1);
  let rho = Math.min(0.8, Math.max(0.2, ctx.param("rho", 0.6))), seed = Math.max(0, Math.min(2, Math.round(ctx.param("seed", 0))));
  let sL, sR, PL, PR, cur = null;
  const fig = host.closest("figure");
  host.replaceChildren();
  const L = ui.simLayout({ parent: host, side: 300 });

  const pair = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "10px 28px" } });
  const mk = title => { const t = ui.h("div", { class: "chart-title", html: title }), sub = ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" } }), b = ui.h("div"); pair.append(ui.h("div", { style: { minWidth: 0 } }, t, sub, b)); return { b, sub }; };
  const boxL = mk("Linear rule: $\\mathrm{Cov}(u_i, u_j)$"), boxR = mk("Rectified, $\\tau = 0$: $c^2\\,\\mathrm{Cov}(r_i, r_j)$");
  L.view.append(pair);
  const hmOpts = side => ({ map: "seq", range: [0, HMAX], square: false, cellH: 3, size: 330, zlabel: "$h_{ib}$", xlabel: "input b (around the ring)", ylabel: "neuron (sorted)",
    tooltip: (r, b, v) => tip(side, r, b, v) });
  const hmL = heatmap(boxL.b, hmOpts("L")), hmR = heatmap(boxR.b, hmOpts("R"));
  ui.renderMath(pair);

  const tp = ui.transport({ parent: L.view, watch: fig, speeds: [10, 25, 50].map(v => ({ value: v, label: `×${v}` })), speed: 25, maxHistory: 200,
    timeFmt: t => `step ${ui.fmtInt(t)} / ${ui.fmtInt(T_END)}`,
    onTick: k => tick(k), onSeek: f => show(f), onReset: () => restart() });

  const rhoS = ui.slider({ label: "pairwise correlation threshold $\\rho^\\ast$", min: 0.2, max: 0.8, step: 0.05, value: rho, fmt: x => ui.fmtFixed(x, 2),
    marks: [0.3, 0.45, 0.6, 0.7].map(v => ({ value: v, label: String(v) })),
    help: "Linear rule at $\\theta = \\rho^\\ast$; rectified rule at $\\theta = \\tilde k_0(\\rho^\\ast)$ with $c^2 = 1/k_0(1)$. Both bind exactly the input pairs correlated above $\\rho^\\ast$. Marks: the values run in Python.",
    onInput: x => { rho = x; thetaLine(); }, onChange: x => { rho = x; ctx.setParams({ rho: x }); tp.reset(); } });
  const thLine = ui.h("div", { class: "u-help", style: { marginTop: "-2px" } });
  const seedS = ui.segmented({ label: "initial weights (seed)", options: [0, 1, 2].map(v => ({ value: v, label: String(v) })), value: seed,
    onChange: v => { seed = v; ctx.setParams({ seed: v }); tp.reset(); } });
  ui.controls({ parent: L.side, primary: [rhoS, thLine, seedS] });
  const stats = ui.stats({ parent: L.side, items: [
    { key: "kl", label: "live clusters, linear", fmt: String }, { key: "kr", label: "live clusters, rectified", fmt: String },
    { key: "dl", label: "dead neurons, linear", fmt: String }, { key: "dr", label: "dead neurons, rectified", fmt: String },
    { key: "res", label: "stationarity $r$ (max of both)", fmt: x => ui.fmt(x, 2), help: "Projected residual max |dw/dt| over synapses, floor synapses pushed down count 0." },
    { key: "t", label: "step", fmt: x => ui.fmtInt(x) },
  ] });
  ui.legend({ parent: L.side, items: [
    { label: "live cluster (band colour by size)", color: catColor(0), shape: "square" },
    { label: "dead neurons", color: P.other, shape: "square" },
  ] });
  ui.renderMath(L.side);

  function thetaLine() {
    const tr = E.reluCorr(rho) / k1;
    thLine.innerHTML = `θ: linear ${ui.fmtFixed(rho, 3)} · rectified ${ui.fmtFixed(tr, 3)}`;
  }

  function restart() {
    PL = { ...E.defaultParams(), theta: rho, wmin: WMIN, dt: DT, nonlin: "linear" };
    PR = { ...E.defaultParams(), theta: E.reluCorr(rho) / k1, wmin: WMIN, dt: DT, nonlin: "relu" };
    sL = E.createState({ N, B, seed, iiScale: 0, ibScale: 0.2 });
    sR = E.createState({ N, B, seed, iiScale: 0, ibScale: 0.2 });
    thetaLine();
    const f = frame(0);
    show(f); return f;
  }
  function frame(t) {
    const HL = E.harmonic(sL).slice(), HR = E.harmonic(sR).slice();
    const aL = analyse(sL, HL), aR = analyse(sR, HR);
    return { t, aL, aR, res: Math.max(sL.resid ?? NaN, sR.resid ?? NaN) };
  }
  function tick(k) {
    const n = Math.min(k, T_END - sL.t);
    if (n > 0) { E.advance(sL, Sig, PL, n); E.advance(sR, SigR, PR, n); }
    const f = frame(sL.t);
    if (sL.t >= T_END) {
      f.res = Math.max(E.stationarity(sL, Sig, PL).r, E.stationarity(sR, SigR, PR).r);
      f.done = true; f.doneLabel = "4,000 steps";
    }
    show(f); return f;
  }
  function show(f) {
    cur = f;
    hmL.update({ values: f.aL.vals, rows: N, cols: B, groups: { rows: f.aL.groups, cols: null } });
    hmR.update({ values: f.aR.vals, rows: N, cols: B, groups: { rows: f.aR.groups, cols: null } });
    const sub = a => (a.K == null ? "not clustered yet" : a.K ? `live cluster sizes ${a.sizes.slice(0, 12).join(" / ")}${a.sizes.length > 12 ? " …" : ""}` : "no live cluster");
    boxL.sub.textContent = `θ = ${ui.fmtFixed(PL.theta, 3)} · ${sub(f.aL)}`;
    boxR.sub.textContent = `θ = ${ui.fmtFixed(PR.theta, 3)} · ${sub(f.aR)}`;
    stats.update({ kl: f.aL.K ?? "–", kr: f.aR.K ?? "–", dl: f.aL.nDead, dr: f.aR.nDead, res: f.res, t: f.t });
  }
  function tip(side, r, b, v) {
    if (!cur) return null;
    const a = side === "L" ? cur.aL : cur.aR, g = a.groups.find(x => r >= x.start && r < x.end);
    return ui.tipRows(`${side === "L" ? "linear" : "rectified"} · neuron ${a.order[r]}`, [
      { name: "input b", value: String(b + 1) }, { name: "h_ib", value: ui.fmtFixed(v, 3) },
      { name: "cluster", value: g ? (g.name === "dead" ? "dead" : `${g.name} (${g.end - g.start} neurons)`) : "not clustered yet", color: g?.color, shape: "square" },
    ]);
  }

  tp.reset();
  ctx.onParams(p => {
    const r = parseFloat(p.get("rho")), s = parseInt(p.get("seed"), 10);
    let changed = false;
    if (Number.isFinite(r) && r !== rho) { rho = Math.min(0.8, Math.max(0.2, r)); rhoS.set(rho); changed = true; }
    if (Number.isFinite(s) && s !== seed && s >= 0 && s <= 2) { seed = s; seedS.set(s); changed = true; }
    if (changed) tp.reset();
  });
}
