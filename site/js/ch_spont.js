// Chapter "spont" (Spontaneous activity). Covariance-rule networks at their fixed point, inputs silent, independent
// noise in every neuron: C_sp = T K^-1 against the evoked covariance H Sigma H^T. Everything is read from
// data/sweeps2/spontaneous.json (hebb/sweeps2/spontaneous.py); the traces of the first figure are simulated here
// from the stored K, H and Sigma. Every number in the prose is filled from the JSON.
import * as ui from "./ui.js";
import { heatmap, hidpi, svgEl, scale, niceTicks, logTicks } from "./charts.js";
import { P, catColor, colorOf } from "./palette.js";

const URL_DATA = "data/sweeps2/spontaneous.json";
const FAMILY_GROUPS = [
  { name: "blocks", test: f => f.startsWith("blocks") },
  { name: "hierarchy", test: f => f === "hier" },
  { name: "power-law spectrum", test: f => f.startsWith("power") },
  { name: "Wishart", test: f => f.startsWith("wishart") },
];
const GAMMA_FAMILIES = [["blocks", "three equal blocks"], ["blocks_uneven", "uneven blocks"], ["hier", "three-level hierarchy"]];

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
  fillTwins(panel, shows.find(s => s.id === "blocks_high"));
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
const STEPS = 2400, DT = 0.01, EVERY = 6, PER = 2, MAXCL = 4;
function simulate(show, seed) {
  const N = show.K.length, B = show.B, rnd = mulberry(seed);
  const n = STEPS / EVERY, ev = Array.from({ length: N }, () => new Float32Array(n)), sp = Array.from({ length: N }, () => new Float32Array(n));
  const L = cholesky(show.Sigma), z = new Float64Array(B), g = new Float64Array(B), u = new Float64Array(N), du = new Float64Array(N);
  for (let b = 0; b < B; b++) z[b] = gauss(rnd);
  for (let i = 0; i < N; i++) u[i] = gauss(rnd) * Math.sqrt(show.Csp[i][i]);       // start near stationarity
  const s = Math.sqrt(2 * DT);
  // Euler substeps for the network: h * (largest eigenvalue of K) <= 0.04 (Gershgorin: eig <= 2 max K_ii), so the
  // discretisation inflates no variance by more than ~2%
  const kmax = 2 * Math.max(...show.K.map((row, i) => row[i])), sub = Math.max(1, Math.ceil(DT * kmax / 0.04)), hs = DT / sub, ss = Math.sqrt(2 * hs);
  for (let t = 0; t < STEPS + 400; t++) {
    for (let b = 0; b < B; b++) z[b] += -z[b] * DT + s * gauss(rnd);                // unit OU, white across inputs
    for (let q = 0; q < sub; q++) {
      for (let i = 0; i < N; i++) { let k = 0; const Ki = show.K[i]; for (let j = 0; j < N; j++) k += Ki[j] * u[j]; du[i] = -k * hs + ss * gauss(rnd); }
      for (let i = 0; i < N; i++) u[i] += du[i];
    }
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
  const a = mk("inputs on (driven)"), b = mk("inputs silent, private noise in each neuron (at rest)");
  const pair = ui.h("div", { class: "space-pair" }, a.box, b.box);
  const key = ui.legend({ items: [] });
  el.replaceChildren(head, pair, key.el);
  let sim = null, simKey = "", rows = [];
  const ROWH = 66;
  // exact correlation between two members of a cluster (all members are copies, so any pair will do)
  const rOf = (C, i, j) => C[i][j] / Math.sqrt(C[i][i] * C[j][j]);
  function draw() {
    if (!show) return;
    const k = `${show.id}:${seed}`; if (k !== simKey) { sim = simulate(show, seed); simKey = k; }
    const groups = groupsOf(show.labels).filter(g => g.end > g.start).sort((p, q) => (q.end - q.start) - (p.end - p.start)).slice(0, MAXCL);
    rows = groups;
    key.set(groups.map((g, i) => ({ label: `cluster ${i + 1} (${g.end - g.start + 1} neurons)`, color: g.color, shape: "line" })));
    for (const [pn, tr, C] of [[a, sim.ev, show.Cev], [b, sim.sp, show.Csp]]) {
      const W = Math.floor(pn.box.clientWidth); if (!W) continue;
      const rowH = ROWH, Hh = groups.length * rowH + 22, g = hidpi(pn.cv, W, Hh);
      g.clearRect(0, 0, W, Hh);
      // right of the traces: member 2 against member 1 at every sampled time (a cloud), then the correlation r
      const sq = rowH - 14, rw = 56, x0 = 4, x1 = W - sq - rw - 14, sx0 = x1 + 10;
      const X = q => x0 + (x1 - x0) * q / (sim.n - 1), amp = rowH / 6.5;      // ±3.25 s.d. per row
      g.font = "11.5px Inter, system-ui, sans-serif"; g.textBaseline = "middle";
      groups.forEach((gr, r) => {
        const y0 = r * rowH + rowH / 2 + 2, sd = Math.sqrt(C[gr.start][gr.start]);      // each cluster in units of its own s.d.
        g.strokeStyle = P.line; g.lineWidth = 1; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y0); g.stroke();
        for (let q = PER - 1; q >= 0; q--) {
          const y = tr[gr.start + q];
          g.strokeStyle = gr.color; g.globalAlpha = q ? 0.5 : 0.95; g.lineWidth = 1.3; g.beginPath();
          for (let t = 0; t < sim.n; t++) { const yy = y0 - amp * y[t] / sd; if (t) g.lineTo(X(t), yy); else g.moveTo(X(t), yy); }
          g.stroke();
        }
        g.globalAlpha = 1;
        const sy0 = y0 - sq / 2, half = sq / 2, ext = 3.5;
        g.strokeStyle = P.line; g.strokeRect(sx0 + 0.5, sy0 + 0.5, sq, sq);
        g.fillStyle = gr.color; g.globalAlpha = 0.4;
        const u = tr[gr.start], v = tr[gr.start + 1];
        for (let t = 0; t < sim.n; t++) {
          const px = sx0 + half + half * Math.max(-1, Math.min(1, u[t] / sd / ext)), py = sy0 + half - half * Math.max(-1, Math.min(1, v[t] / sd / ext));
          g.fillRect(px - 1, py - 1, 2, 2);
        }
        g.globalAlpha = 1;
        g.fillStyle = P.fg; g.textAlign = "left";
        g.fillText(`r = ${ui.fmtFixed(rOf(C, gr.start, gr.start + 1), 2)}`, sx0 + sq + 8, y0);
      });
      g.fillStyle = P.mute; g.font = "11px Inter, system-ui, sans-serif"; g.textBaseline = "alphabetic";
      g.textAlign = "left"; g.fillText("time →", x0, Hh - 5);
      g.textAlign = "right"; g.fillText(`${ui.fmt(STEPS * DT, 2)} time constants`, x1, Hh - 5);
      g.textAlign = "left"; g.fillText("neuron 2 vs 1", sx0, Hh - 5);
    }
  }
  new ResizeObserver(draw).observe(pair);
  for (const [pn, which, name] of [[a, "Cev", "inputs on"], [b, "Csp", "at rest"]]) {
    pn.cv.addEventListener("pointermove", ev => {
      const gr = rows[Math.floor((ev.clientY - pn.cv.getBoundingClientRect().top - 2) / ROWH)];
      if (!gr || !show) return ui.hideTip();
      ui.showTip(ev.clientX, ev.clientY, ui.tipRows(`cluster ${rows.indexOf(gr) + 1}, ${gr.end - gr.start + 1} neurons`,
        [{ name: `correlation between two members, ${name}`, value: ui.fmtFixed(rOf(show[which], gr.start, gr.start + 1), 3), color: gr.color }]));
    });
    pn.cv.addEventListener("pointerleave", () => ui.hideTip());
  }
  return { show(s) { show = s; draw(); } };
}

// ---------------------------------------------------------------- figure 2: correlation heatmaps
function corrFigure(el, picker) {
  const head = ui.h("div", { style: { marginBottom: "8px" } }); picker(head);
  // under each map: the mean correlation within and between clusters, each with a swatch of its colour in the map
  const mk = title => { const slot = ui.h("div"), sum = ui.h("div", { class: "spont-cmean", style: { font: "12.5px/1.5 var(--sans)", color: "var(--ink-2)", margin: "2px 0 8px" } });
    return { slot, sum, box: ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", text: title }), sum, slot) }; };
  const a = mk("inputs on (driven)"), b = mk("inputs silent, private noise (at rest)");
  el.replaceChildren(head, ui.h("div", { class: "space-pair" }, a.box, b.box));
  const opts = { map: "div", range: [-1, 1], size: 380, zlabel: "correlation", xlabel: "neuron (sorted by cluster)", ylabel: "neuron" };
  const tip = name => (i, j, v) => ui.tipRows(`neurons ${i + 1} and ${j + 1}`, [{ name, value: ui.fmtFixed(v, 3) }]);
  const ha = heatmap(a.slot, { ...opts, tooltip: tip("correlation, inputs on") }), hb = heatmap(b.slot, { ...opts, tooltip: tip("correlation at rest") });
  const sw = v => `<span class="swatch" style="--sw: ${colorOf("div", v, -1, 1)}; outline: 1px solid var(--line)"></span>`;
  const nw = html => `<span style="white-space: nowrap">${html}</span>`;
  const meanLine = (within, between, d) => `mean correlation: ${nw(`within a cluster ${sw(within)}<b>${ui.fmtFixed(within, 2)}</b>`)}`
    + ` · ${nw(`between clusters ${between == null ? "<b>n/a</b>" : `${sw(between)}<b>${ui.fmtFixed(between, d)}</b>`}`)}`;
  return { show(s) {
    const N = s.K.length, groups = { rows: groupsOf(s.labels).map(g => ({ start: g.start, end: g.end, color: g.color })) };
    ha.update({ values: corrOf(s.Cev), rows: N, cols: N, groups });
    hb.update({ values: corrOf(s.Csp), rows: N, cols: N, groups });
    a.sum.innerHTML = meanLine(s.summary.r_ev_within, s.summary.r_ev_between, 2);
    b.sum.innerHTML = meanLine(s.summary.r_sp_within, s.summary.r_sp_between, 3);
  } };
}

// ---------------------------------------------------------------- a small SVG scatter-over-curves chart (chapter-local)
// The shared lineChart puts log ticks only at decades, draws a curve under the dots it predicts and answers hover with
// the nearest x. Here: ticks where asked, the curve drawn over the dots (so both stay visible where they coincide),
// and a tooltip for the nearest dot that also counts the dots hidden under it.
// o: {height, xlog?, ylog?, xrange, yrange, xticks?, yticks?, xfmt?, yfmt?, xlabel, ylabel, xshort?, yshort?, refs?: [{x | y, label, left?}],
//     points: [{x, y, color, d}], r?, alpha?, curves: [{x: [], y: [], color, width?}], tip(d, hidden), legend: items}
function xyPlot(el, o) {
  el.classList.add("chart");
  const host = ui.h("div"), lg = ui.legend({ items: o.legend || [] });
  lg.el.classList.add("chart-legend");
  el.replaceChildren(host, lg.el);
  const tx = (p, x, y, s, a = {}) => { const t = svgEl("text", { x, y, ...a }, p); t.textContent = s; return t; };
  const cid = "spont-clip-" + Math.random().toString(36).slice(2);
  let lastW = 0;
  function draw() {
    const W = Math.floor(el.clientWidth); if (!W || W === lastW) return; lastW = W;
    host.replaceChildren();
    const Hh = o.height, m = { l: 56, r: 14, t: 10, b: 40 }, R = o.r || 3.2;
    const X = scale(o.xlog ? "log" : "linear", o.xrange[0], o.xrange[1], m.l, W - m.r);
    const Y = scale(o.ylog ? "log" : "linear", o.yrange[0], o.yrange[1], Hh - m.b, m.t);
    const xt = o.xticks || (o.xlog ? logTicks(...o.xrange) : niceTicks(o.xrange[0], o.xrange[1], Math.max(3, Math.round(W / 90))));
    const yt = o.yticks || (o.ylog ? logTicks(...o.yrange) : niceTicks(o.yrange[0], o.yrange[1], Math.max(3, Math.round(Hh / 55))));
    const xf = o.xfmt || (v => ui.fmt(v)), yf = o.yfmt || (v => ui.fmt(v));
    const svg = svgEl("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": o.ylabel }, host);
    const grid = svgEl("g", { class: "grid" }, svg), ax = svgEl("g", { class: "axis" }, svg);
    yt.forEach(v => { svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid); tx(svg, m.l - 6, Y(v) + 3.5, yf(v), { "text-anchor": "end" }); });
    if (o.xlog) xt.forEach(v => svgEl("line", { x1: X(v), x2: X(v), y1: m.t, y2: Hh - m.b }, grid));
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Hh - m.b + .5, y2: Hh - m.b + .5 }, ax);
    xt.forEach(v => { svgEl("line", { x1: X(v), x2: X(v), y1: Hh - m.b, y2: Hh - m.b + 4 }, ax); tx(svg, X(v), Hh - m.b + 15, xf(v), { "text-anchor": "middle" }); });
    // axis titles: a shorter wording when the long one does not fit (phones)
    const fit = (t, short, room) => { if (short && t.getComputedTextLength() > room) t.textContent = short; };
    fit(tx(svg, (m.l + W - m.r) / 2, Hh - 4, o.xlabel, { "text-anchor": "middle", class: "ttl" }), o.xshort, W - m.l - 4);
    fit(tx(svg, 0, 0, o.ylabel, { "text-anchor": "middle", class: "ttl", transform: `translate(12,${(m.t + Hh - m.b) / 2}) rotate(-90)` }), o.yshort, Hh - m.b);
    const halo = { style: "paint-order: stroke; stroke: var(--surface); stroke-width: 3px; stroke-linejoin: round" };   // legible where a line crosses
    for (const r of o.refs || []) {
      if (r.y != null) { svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y), stroke: P.mute2, "stroke-dasharray": "3 3" }, svg); if (r.left) tx(svg, m.l + 4, Y(r.y) - 5, r.label, halo); else tx(svg, W - m.r - 2, Y(r.y) - 5, r.label, { "text-anchor": "end", ...halo }); }
      else { svgEl("line", { x1: X(r.x), x2: X(r.x), y1: m.t, y2: Hh - m.b, stroke: P.mute2, "stroke-dasharray": "3 3" }, svg); tx(svg, X(r.x) + 5, m.t + 10, r.label, halo); }
    }
    svgEl("rect", { x: m.l, y: m.t - R, width: W - m.l - m.r, height: Hh - m.t - m.b + 2 * R }, svgEl("clipPath", { id: cid }, svgEl("defs", {}, svg)));
    const plot = svgEl("g", { "clip-path": `url(#${cid})` }, svg);
    const pts = o.points.map(p => ({ ...p, px: X(p.x), py: Y(p.y) }));
    pts.forEach(p => svgEl("circle", { cx: p.px, cy: p.py, r: R, fill: p.color, "fill-opacity": o.alpha ?? .75, stroke: P.surface, "stroke-width": .75 }, plot));
    for (const c of o.curves || []) {
      let d = ""; c.x.forEach((xv, i) => { if (Number.isFinite(c.y[i])) d += (d ? "L" : "M") + X(xv).toFixed(1) + "," + Y(c.y[i]).toFixed(1); });
      svgEl("path", { d, fill: "none", stroke: c.color, "stroke-width": c.width || 1.5, "stroke-linejoin": "round", "stroke-linecap": "round" }, plot);
    }
    const ring = svgEl("circle", { r: R + 3, fill: "none", stroke: P.fg, "stroke-width": 1.5, opacity: 0 }, svg);
    const hit = svgEl("rect", { x: m.l, y: m.t - R, width: W - m.l - m.r, height: Hh - m.t - m.b + 2 * R, fill: "transparent" }, svg);
    const off = () => { ring.setAttribute("opacity", 0); ui.hideTip(); };
    hit.addEventListener("pointermove", ev => {
      const bb = svg.getBoundingClientRect(), x = ev.clientX - bb.left, y = ev.clientY - bb.top;
      let best = null, bd = 22 * 22;
      for (const p of pts) { const dd = (p.px - x) ** 2 + (p.py - y) ** 2; if (dd < bd) { bd = dd; best = p; } }
      if (!best) return off();
      const hidden = pts.filter(p => p !== best && Math.abs(p.px - best.px) < 1.5 && Math.abs(p.py - best.py) < 1.5).length;
      ring.setAttribute("cx", best.px); ring.setAttribute("cy", best.py); ring.setAttribute("opacity", 1);
      ui.showTip(ev.clientX, ev.clientY, o.tip(best.d, hidden));
    });
    hit.addEventListener("pointerleave", off);
    lg.el.style.marginLeft = m.l + "px";
  }
  new ResizeObserver(draw).observe(el);
}
const famName = f => FAMILY_GROUPS.find(g => g.test(f))?.name ?? f;
const famColor = f => catColor(Math.max(0, FAMILY_GROUPS.findIndex(g => g.test(f))));
const famLegend = () => FAMILY_GROUPS.map((f, i) => ({ label: f.name, color: catColor(i), shape: "dot" }));
// tooltip for one cluster of the sweep: what it is, the two coordinates and the formula
const clusterTip = (rows, note) => (c, hidden) => ui.tipRows(`${famName(c.family)} inputs, θ = ${c.theta}, ${c.n} neurons`,
  rows.map(([name, f]) => ({ name, value: ui.fmt(f(c), 3) })), hidden ? `${hidden} more cluster${hidden > 1 ? "s" : ""} on the same spot` : note);

// ---------------------------------------------------------------- figure 3: r against the effective number of inputs
const clustersOf = runs => runs.flatMap(r => r.clusters.map(c => ({ ...c, family: r.family, theta: r.theta, gamma: r.gamma })));
function mFigure(el, data) {
  const cl = clustersOf(data.sweep), mmax = Math.ceil(Math.max(...cl.map(c => c.m_eff)));
  const xs = Array.from({ length: 160 }, (_, i) => 1 + (mmax - 1) * i / 159);
  xyPlot(el, { height: 300, xrange: [1, mmax], yrange: [0, 0.55], xticks: [1, 2, 4, 6, 8, 10, 12].filter(v => v <= mmax), yticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    yfmt: v => ui.fmtFixed(v, 1), xlabel: "effective number of inputs the cluster listens to (m = 1/|h|²)", xshort: "effective number of inputs (m = 1/|h|²)",
    ylabel: "correlation between members at rest (r)", yshort: "correlation at rest (r)",
    refs: [{ y: 0.5, label: "one input: r = ½" }],
    points: cl.map(c => ({ x: c.m_eff, y: c.r_sp, color: famColor(c.family), d: c })),
    curves: [{ x: xs, y: xs.map(m => 1 / (1 + m)), color: P.fg, width: 1.4 }],
    legend: [...famLegend(), { label: "$1/(1+m)$", color: P.fg, shape: "line" }],
    tip: clusterTip([["inputs listened to (m)", c => c.m_eff], ["correlation at rest", c => c.r_sp], ["1/(1 + m)", c => 1 / (1 + c.m_eff)]]) });
}

// ---------------------------------------------------------------- figure 4: r against gamma
function gammaFigure(el, data) {
  const gammas = [...new Set(data.gamma.map(r => r.gamma))].sort((p, q) => p - q), points = [], curves = [];
  GAMMA_FAMILIES.forEach(([fam, name], i) => {
    const cls = g => clustersOf(data.gamma.filter(r => r.family === fam && r.gamma === g)), at = (g, key) => mean(cls(g).map(c => c[key]));
    curves.push({ x: gammas, y: gammas.map(g => at(g, "r_pred_sigma")), color: catColor(i), width: 1.6 });
    gammas.forEach(g => points.push({ x: g, y: at(g, "r_sp"), color: catColor(i), d: { name, g, r: at(g, "r_sp"), f: at(g, "r_pred_sigma"), k: cls(g).length } }));
  });
  xyPlot(el, { height: 290, xlog: true, xrange: [0.5, 12], xticks: [0.6, 1, 2, 3, 5, 10], yrange: [0, 1], yticks: [0, 0.2, 0.4, 0.6, 0.8, 1],
    yfmt: v => ui.fmtFixed(v, 1), r: 4, alpha: 1,
    xlabel: "lateral learning rate relative to input learning rate (γ)", xshort: "lateral ÷ input learning rate (γ)",
    ylabel: "correlation between members at rest (r)", yshort: "correlation at rest (r)",
    refs: [{ x: 1, label: "equal rates" }, { y: 0.5, label: "ceiling when γ = 1", left: true }], points, curves,
    legend: [...GAMMA_FAMILIES.map(([, name], i) => ({ label: name, color: catColor(i), shape: "square" })),
      { label: "measured (dots)", color: P.fg, shape: "dot" }, { label: "formula $w/(a+w)$ (lines)", color: P.fg, shape: "line" }],
    tip: d => ui.tipRows(`${d.name}, γ = ${d.g}`, [{ name: "measured correlation", value: ui.fmtFixed(d.r, 3) }, { name: "formula", value: ui.fmtFixed(d.f, 3) }],
      `mean over ${d.k} clusters from three random starts`) });
}

// ---------------------------------------------------------------- figure 5: variance at rest
function loudFigure(el, data) {
  const cl = clustersOf(data.sweep), xs = [0.1, 300];
  xyPlot(el, { height: 300, xlog: true, ylog: true, xrange: [0.1, 300], yrange: [0.003, 10],
    xlabel: "cluster size × input conductance per member (n·a)", xshort: "size × input conductance (n·a)",
    ylabel: "variance of the cluster mean at rest (units of T)", yshort: "variance of the mean at rest (T)",
    points: cl.map(c => ({ x: c.n * c.a, y: c.sp_mean_var, color: famColor(c.family), d: c })),
    curves: [{ x: xs, y: xs.map(x => 1 / x), color: P.fg, width: 1.4 }],
    legend: [...famLegend(), { label: "$T/(na)$", color: P.fg, shape: "line" }],
    tip: clusterTip([["size n", c => c.n], ["input conductance a", c => c.a], ["variance of the mean at rest", c => c.sp_mean_var], ["T/(n a)", c => 1 / (c.n * c.a)]]) });
}

// ---------------------------------------------------------------- numbers in the prose
function spearman(x, y) {
  const rank = v => {                                   // average ranks for ties
    const o = v.map((_, i) => i).sort((p, q) => v[p] - v[q]), r = new Array(v.length);
    for (let k = 0; k < o.length;) { let e = k; while (e + 1 < o.length && Math.abs(v[o[e + 1]] - v[o[k]]) < 1e-9) e++; for (let q = k; q <= e; q++) r[o[q]] = (k + e) / 2; k = e + 1; }
    return r;
  };
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
  const runs = [...data.sweep, ...data.sweep_floor, ...data.gamma, ...data.showcases.map(s => s.summary)], cl = clustersOf(data.sweep);
  const gaps = c => c.map(c => Math.abs(c.r_sp - 1 / (1 + c.m_eff))).sort((p, q) => p - q), median = v => (v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2);
  const g = gaps(cl), gf = gaps(clustersOf(data.sweep_floor));
  fill(panel, "s-runs", ui.fmtInt(runs.length));
  fill(panel, "s-ident", `within ${ui.fmt(Math.max(...runs.map(r => r.identity_err)), 1)}`);
  fill(panel, "s-ncl", ui.fmtInt(cl.length));
  fill(panel, "s-nsweep", ui.fmtInt(data.sweep.length));
  fill(panel, "s-mgap-med", ui.fmtFixed(median(g), 4));
  fill(panel, "s-mfrac", ui.fmtPct(g.filter(x => x < 0.02).length / g.length));
  fill(panel, "s-mgap", ui.fmtFixed(g[g.length - 1], 3));
  fill(panel, "s-fgap-med", ui.fmtFixed(median(gf), 3));
  fill(panel, "s-fgap-max", ui.fmtFixed(gf[gf.length - 1], 3));
  fill(panel, "s-spread", ui.fmt(Math.max(...cl.map(c => c.spread)), 1));
  fill(panel, "s-rank", ui.fmtFixed(spearman(cl.map(c => c.ev_var), cl.map(c => c.sp_var)), 2));
  const within = data.sweep.filter(r => r.clusters.length >= 3).map(r => spearman(r.clusters.map(c => c.ev_var), r.clusters.map(c => c.sp_var)))
    .filter(Number.isFinite).sort((p, q) => p - q);
  fill(panel, "s-rank-n", ui.fmtInt(within.length));
  fill(panel, "s-rank-neg", ui.fmtInt(within.filter(x => x < 0).length));
  fill(panel, "s-rank-med", ui.fmtFixed(median(within), 2));
}
// neurons in different clusters with the same mixture (evoked correlation > 0.999): the unwired copies
function fillTwins(panel, s) {
  const Re = corrOf(s.Cev), Rs = corrOf(s.Csp), N = s.labels.length, who = new Set();
  let ev = 1, sp = 0;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (s.labels[i] !== s.labels[j] && Re[i][j] > 0.999) {
    who.add(i); who.add(j); ev = Math.min(ev, Re[i][j]); sp = Math.max(sp, Rs[i][j]);
  }
  fill(panel, "s-twin-n", ["no", "one", "two", "three", "four", "five", "six"][who.size] ?? String(who.size));
  fill(panel, "s-twin-ev", ui.fmtFixed(ev, 3));
  fill(panel, "s-twin-sp", ui.fmtFixed(sp, 3));
}
