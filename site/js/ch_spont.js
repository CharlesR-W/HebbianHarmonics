// Chapter "spont" (Spontaneous activity). Covariance-rule networks at their fixed point, inputs silent, independent
// noise in every neuron: C_sp = T K^-1 against the evoked covariance H Sigma H^T. Everything is read from
// data/sweeps2/spontaneous.json (hebb/sweeps2/spontaneous.py); the traces of the first figure are simulated here
// from the stored K, H and Sigma. Every number in the prose is filled from the JSON.
import * as ui from "./ui.js";
import { heatmap, lineChart, hidpi } from "./charts.js";
import { P, catColor } from "./palette.js";

const URL_DATA = "data/sweeps2/spontaneous.json";
const FAMILY_GROUPS = [
  { name: "blocks", test: f => f.startsWith("blocks") },
  { name: "hierarchy", test: f => f === "hier" },
  { name: "power-law spectrum", test: f => f.startsWith("power") },
  { name: "Wishart", test: f => f.startsWith("wishart") },
];
const GAMMA_FAMILIES = [["blocks", "three equal blocks"], ["blocks_uneven", "uneven blocks"], ["hier", "hierarchy"]];

export async function init(panel, ctx) {
  const r = await fetch(URL_DATA, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${URL_DATA}: HTTP ${r.status}`);
  const data = await r.json();
  const shows = data.showcases;
  let cur = shows.find(s => s.id === ctx.param("net", "blocks")) || shows[0];

  const pickers = [];
  const picker = parent => {
    const s = ui.select({ label: "inputs", parent, value: cur.id, options: shows.map(s => ({ value: s.id, label: s.label })),
      onChange: id => { cur = shows.find(s => s.id === id); pickers.forEach(p => p.set(id)); ctx.setParams({ net: id }); redraw(); } });
    pickers.push(s); return s;
  };
  const traces = tracesFigure(ctx.slot("traces"), picker);
  const corr = corrFigure(ctx.slot("corr"), picker);
  const redraw = () => { traces.show(cur); corr.show(cur); fillShow(panel, cur); };
  redraw();
  ctx.onParams(p => { const s = shows.find(s => s.id === p.get("net")); if (s && s !== cur) { cur = s; pickers.forEach(q => q.set(s.id)); redraw(); } });

  mFigure(ctx.slot("m"), data);
  gammaFigure(ctx.slot("gamma"), data);
  loudFigure(ctx.slot("loud"), data);
  fillSweep(panel, data);
  ctx.renderMath(panel);
}

// ---------------------------------------------------------------- helpers
function mulberry(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function gauss(rnd) { let u = 0; while (u === 0) u = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd()); }
function cholesky(A) {
  const n = A.length, L = A.map(() => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
  }
  return L;
}
const corrOf = C => C.map((row, i) => row.map((v, j) => v / Math.sqrt(C[i][i] * C[j][j])));
const groupsOf = labels => {
  const g = []; let start = 0;
  for (let i = 1; i <= labels.length; i++) if (i === labels.length || labels[i] !== labels[start]) { g.push({ start, end: i - 1, label: labels[start] }); start = i; }
  // colours by size, largest first; singletons grey
  const order = g.map((x, k) => k).sort((p, q) => (g[q].end - g[q].start) - (g[p].end - g[p].start));
  order.forEach((k, rank) => { g[k].color = g[k].end > g[k].start ? catColor(rank) : P.other; });
  return g;
};
const fill = (panel, key, html) => panel.querySelectorAll(`[data-fill="${key}"]`).forEach(el => { el.innerHTML = html; });
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;

// ---------------------------------------------------------------- figure 1: traces, driven and at rest
const STEPS = 2400, DT = 0.01, EVERY = 6, PER = 3, MAXCL = 4;
function simulate(show, seed) {
  const N = show.K.length, B = show.B, rnd = mulberry(seed);
  const n = STEPS / EVERY, ev = Array.from({ length: N }, () => new Float32Array(n)), sp = Array.from({ length: N }, () => new Float32Array(n));
  const L = cholesky(show.Sigma), z = new Float64Array(B), g = new Float64Array(B), u = new Float64Array(N), du = new Float64Array(N);
  for (let b = 0; b < B; b++) z[b] = gauss(rnd);
  for (let i = 0; i < N; i++) u[i] = gauss(rnd) * Math.sqrt(show.Csp[i][i]);       // start near stationarity
  const s = Math.sqrt(2 * DT);
  for (let t = 0; t < STEPS + 400; t++) {
    for (let b = 0; b < B; b++) z[b] += -z[b] * DT + s * gauss(rnd);                // unit OU, white across inputs
    for (let i = 0; i < N; i++) { let k = 0; const Ki = show.K[i]; for (let j = 0; j < N; j++) k += Ki[j] * u[j]; du[i] = -k * DT + s * gauss(rnd); }
    for (let i = 0; i < N; i++) u[i] += du[i];
    const tt = t - 400;
    if (tt >= 0 && tt % EVERY === 0) {
      for (let b = 0; b < B; b++) { let v = 0; for (let c = 0; c <= b; c++) v += L[b][c] * z[c]; g[b] = v; }
      const k = tt / EVERY;
      for (let i = 0; i < N; i++) { let v = 0; const Hi = show.H[i]; for (let b = 0; b < B; b++) v += Hi[b] * g[b]; ev[i][k] = v; sp[i][k] = u[i]; }
    }
  }
  return { ev, sp, n };
}
function tracesFigure(el, picker) {
  const head = ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "8px 18px", alignItems: "end", marginBottom: "8px" } });
  picker(head);
  let seed = 1, show = null;
  const again = ui.h("button", { type: "button", text: "new sample",
    style: { border: "1px solid var(--line)", background: "var(--surface)", borderRadius: "6px", padding: "4px 11px", fontSize: "12.5px", cursor: "pointer" }, onclick: () => { seed += 1; draw(); } });
  head.append(again);
  const mk = title => { const cv = ui.h("canvas", { style: { display: "block", width: "100%" } }); const box = ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", text: title }), cv); return { cv, box }; };
  const a = mk("inputs on (evoked)"), b = mk("inputs silent, private noise (spontaneous)");
  const pair = ui.h("div", { class: "space-pair" }, a.box, b.box);
  const key = ui.legend({ items: [] });
  el.replaceChildren(head, pair, key.el);
  let sim = null, simKey = "";
  function draw() {
    if (!show) return;
    const k = `${show.id}:${seed}`; if (k !== simKey) { sim = simulate(show, seed); simKey = k; }
    const groups = groupsOf(show.labels).filter(g => g.end > g.start).sort((p, q) => (q.end - q.start) - (p.end - p.start)).slice(0, MAXCL);
    key.set(groups.map((g, i) => ({ label: `cluster ${i + 1} (${g.end - g.start + 1} neurons)`, color: g.color, shape: "line" })));
    for (const [pn, tr, C] of [[a, sim.ev, show.Cev], [b, sim.sp, show.Csp]]) {
      const W = Math.floor(pn.box.clientWidth); if (!W) continue;
      const rowH = 64, Hh = groups.length * rowH + 26, g = hidpi(pn.cv, W, Hh);
      g.clearRect(0, 0, W, Hh);
      const sd = Math.sqrt(mean(groups.flatMap(gr => Array.from({ length: gr.end - gr.start + 1 }, (_, q) => C[gr.start + q][gr.start + q]))));
      const x0 = 6, x1 = W - 6, X = q => x0 + (x1 - x0) * q / (sim.n - 1), amp = rowH / 6.5;      // ±3.25 s.d. per row
      groups.forEach((gr, r) => {
        const y0 = r * rowH + rowH / 2 + 4;
        g.strokeStyle = P.line; g.lineWidth = 1; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y0); g.stroke();
        for (let q = 0; q < Math.min(PER, gr.end - gr.start + 1); q++) {
          const y = tr[gr.start + q];
          g.strokeStyle = gr.color; g.globalAlpha = 0.9 - 0.22 * q; g.lineWidth = 1.3; g.beginPath();
          for (let t = 0; t < sim.n; t++) { const yy = y0 - amp * y[t] / sd; if (t) g.lineTo(X(t), yy); else g.moveTo(X(t), yy); }
          g.stroke();
        }
        g.globalAlpha = 1;
      });
      g.fillStyle = P.mute; g.font = "11px Inter, system-ui, sans-serif"; g.textBaseline = "alphabetic";
      g.textAlign = "left"; g.fillText("time →", x0, Hh - 6);
      g.textAlign = "right"; g.fillText(`${ui.fmt(STEPS * DT, 2)} time constants`, x1, Hh - 6);
    }
  }
  new ResizeObserver(draw).observe(pair);
  return { show(s) { show = s; draw(); } };
}

// ---------------------------------------------------------------- figure 2: correlation heatmaps
function corrFigure(el, picker) {
  const head = ui.h("div", { style: { marginBottom: "8px" } }); picker(head);
  const mk = title => { const slot = ui.h("div"); return { slot, box: ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", text: title }), slot) }; };
  const a = mk("inputs on: correlation of evoked activity"), b = mk("inputs silent: correlation of spontaneous activity");
  el.replaceChildren(head, ui.h("div", { class: "space-pair" }, a.box, b.box));
  const opts = { map: "div", range: [-1, 1], size: 380, zlabel: "correlation", xlabel: "neuron j (sorted by cluster)", ylabel: "neuron i" };
  const tip = name => (i, j, v) => ui.tipRows(`neurons ${i + 1}, ${j + 1}`, [{ name, value: ui.fmtFixed(v, 3) }]);
  const ha = heatmap(a.slot, { ...opts, tooltip: tip("evoked correlation") }), hb = heatmap(b.slot, { ...opts, tooltip: tip("spontaneous correlation") });
  return { show(s) {
    const N = s.K.length, groups = { rows: groupsOf(s.labels).map(g => ({ start: g.start, end: g.end, color: g.color })) };
    ha.update({ values: corrOf(s.Cev), rows: N, cols: N, groups });
    hb.update({ values: corrOf(s.Csp), rows: N, cols: N, groups });
  } };
}

// ---------------------------------------------------------------- figure 3: r against the effective number of inputs
const clustersOf = runs => runs.flatMap(r => r.clusters.map(c => ({ ...c, family: r.family, theta: r.theta, gamma: r.gamma })));
function mFigure(el, data) {
  const cl = clustersOf(data.sweep);
  const mmax = Math.max(...cl.map(c => c.m_eff));
  const xs = Array.from({ length: 120 }, (_, i) => 1 + (Math.ceil(mmax) - 1) * i / 119);
  const series = [{ name: "1/(1 + m)", x: xs, y: xs.map(m => 1 / (1 + m)), color: P.fg, width: 1.5 },
    ...FAMILY_GROUPS.map((f, i) => { const c = cl.filter(c => f.test(c.family)); return { name: f.name, x: c.map(c => c.m_eff), y: c.map(c => c.r_sp), color: catColor(i), points: true, r: 3.2 }; })];
  el.replaceChildren();
  lineChart(el, { series, xrange: [1, Math.ceil(mmax)], yrange: [0, 0.55], height: 300, xlabel: "effective number of inputs the cluster listens to (m = 1/|h|²)",
    ylabel: "correlation between members at rest (r)", yfmt: v => ui.fmtFixed(v, 2) });
}

// ---------------------------------------------------------------- figure 4: r against gamma
function gammaFigure(el, data) {
  const gammas = [...new Set(data.gamma.map(r => r.gamma))].sort((p, q) => p - q);
  const series = [];
  GAMMA_FAMILIES.forEach(([fam, name], i) => {
    const at = (g, key) => mean(clustersOf(data.gamma.filter(r => r.family === fam && r.gamma === g)).map(c => c[key]));
    series.push({ name: `${name}: formula`, x: gammas, y: gammas.map(g => at(g, "r_pred_sigma")), color: catColor(i), width: 1.5 });
    series.push({ name: `${name}: measured`, x: gammas, y: gammas.map(g => at(g, "r_sp")), color: catColor(i), points: true, r: 3.6 });
  });
  el.replaceChildren();
  lineChart(el, { series, xlog: true, yrange: [0, 1], height: 280, xlabel: "lateral learning rate relative to input learning rate (γ)",
    ylabel: "correlation between members at rest (r)", yfmt: v => ui.fmtFixed(v, 1), refs: [{ x: 1, label: "γ = 1" }] });
}

// ---------------------------------------------------------------- figure 5: variance at rest
function loudFigure(el, data) {
  const cl = clustersOf(data.sweep);
  const xv = cl.map(c => c.n * c.a), lo = Math.min(...xv) * 0.8, hi = Math.max(...xv) * 1.25;
  const xs = Array.from({ length: 60 }, (_, i) => lo * Math.pow(hi / lo, i / 59));
  const series = [{ name: "T/(n a)", x: xs, y: xs.map(x => 1 / x), color: P.fg, width: 1.5 },
    ...FAMILY_GROUPS.map((f, i) => { const c = cl.filter(c => f.test(c.family)); return { name: f.name, x: c.map(c => c.n * c.a), y: c.map(c => c.sp_mean_var), color: catColor(i), points: true, r: 3.2 }; })];
  el.replaceChildren();
  lineChart(el, { series, xlog: true, ylog: true, height: 300, xlabel: "cluster size × input conductance per member (n·a)",
    ylabel: "variance of the cluster mean at rest (units of T)" });
}

// ---------------------------------------------------------------- numbers in the prose
function spearman(x, y) {
  const rank = v => { const o = v.map((_, i) => i).sort((p, q) => v[p] - v[q]), r = new Array(v.length); o.forEach((i, k) => { r[i] = k; }); return r; };
  const a = rank(x), b = rank(y), ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0; a.forEach((_, i) => { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; });
  return sab / Math.sqrt(saa * sbb);
}
function fillShow(panel, s) {
  const L = s.langevin;
  fill(panel, "s-lang-n", ui.fmtInt(L.reps * L.n_samples));
  fill(panel, "s-lang-dt", String(L.dt));
  fill(panel, "s-lang-err", ui.fmtPct(L.rel_err, 1));
  fill(panel, "s-ev-btw", s.summary.r_ev_between == null ? "n/a" : ui.fmtFixed(s.summary.r_ev_between, 2));
  fill(panel, "s-sp-btw", s.summary.r_sp_between == null ? "n/a" : ui.fmtFixed(s.summary.r_sp_between, 3));
}
function fillSweep(panel, data) {
  const runs = [...data.sweep, ...data.gamma, ...data.showcases.map(s => s.summary)], cl = clustersOf(data.sweep);
  fill(panel, "s-runs", ui.fmtInt(runs.length));
  fill(panel, "s-ident", `within ${ui.fmt(Math.max(...runs.map(r => r.identity_err)), 1)}`);
  fill(panel, "s-ncl", ui.fmtInt(cl.length));
  fill(panel, "s-nsweep", ui.fmtInt(data.sweep.length));
  fill(panel, "s-mgap", ui.fmtFixed(Math.max(...cl.map(c => Math.abs(c.r_sp - 1 / (1 + c.m_eff)))), 3));
  fill(panel, "s-spread", ui.fmt(Math.max(...cl.map(c => c.spread)), 1));
  fill(panel, "s-rank", ui.fmtFixed(spearman(cl.map(c => c.ev_var), cl.map(c => c.sp_var)), 2));
}
