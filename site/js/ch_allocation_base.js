// Chapter "allocation", section (a): the base model with block attachments. A live figure (four block panels drawn
// as exact isometric triangles, block 1's variance and neuron count as sliders) and the precomputed sweep views.
import * as ui from "./ui.js";
import { lineChart, scatter, hidpi } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as M from "./ch_allocation_model.js";
import { netview } from "./netview.js";

const SUBD = "₀₁₂₃₄₅₆₇₈₉";
export const sub = n => String(n).split("").map(d => SUBD[+d] ?? d).join("");

export const DEF_A = { v1: 1, n1: 12, b: 0.02, seed: 0 };
const KEYS_A = { v1: "av", n1: "an", b: "ab", seed: "as" };
export const TOURS_A = [
  { id: "quiet", label: "Quiet block", note: "$v_1 = 0.25$, $b = 0.02$: block 1's triangle is still wider than the edge range, but its 12 neurons start close enough that nearly every pair is in range (drag the time bar to $t = 0$); the learned edges pull them in to about one edge range, and they end in one cluster.", p: { v1: 0.25, n1: 12, b: 0.02, seed: 0 } },
  { id: "loud", label: "Loud block fragments", note: "$v_1 = 8$, $b = 0.02$: the same 12 neurons, spread over a triangle many edge ranges wide, split into small clusters and singletons.", p: { v1: 8, n1: 12, b: 0.02, seed: 0 } },
  { id: "loud-coarse", label: "Loud block, larger $b$", note: "$v_1 = 8$, $b = 0.05$: a longer edge range merges some of the fragments; still 12 neurons.", p: { v1: 8, n1: 12, b: 0.05, seed: 0 } },
  { id: "more", label: "More neurons attached", note: "$v_1 = 1$, $n_1 = 24$, $b = 0.02$: twice the neurons on block 1, and its cluster doubles.", p: { v1: 1, n1: 24, b: 0.02, seed: 0 } },
];

function clampA(p) {
  return {
    v1: Math.min(8, Math.max(0.25, +p.v1 || DEF_A.v1)),
    n1: Math.min(24, Math.max(4, Math.round(+p.n1 || DEF_A.n1))),
    b: Math.min(0.2, Math.max(0.002, +p.b || DEF_A.b)),
    seed: Math.min(20, Math.max(0, Math.round(+p.seed || 0))),
  };
}

export function baseFigure(ctx, slot) {
  const fig = slot.closest("figure");
  const readURL = () => {
    const t = TOURS_A.find(x => x.id === ctx.param("ta", ""));
    if (t) return { p: { ...t.p }, tour: t };
    const p = {}; for (const k in KEYS_A) p[k] = ctx.param(KEYS_A[k], DEF_A[k]);
    return { p: clampA(p), tour: null };
  };
  let { p, tour } = readURL();
  let base = null, cur = null, acc = 0;

  const L = ui.simLayout({ parent: slot });
  // ---------------------------------------------------------------- panels
  // input space: one simplex per block (synapses opt-in); below, the whole network in neuron space; hover links them
  let synOn = false, hi = null, nv = null, nvN = -1;
  const head = ui.h("div", { class: "panel-head", style: { justifyContent: "space-between", gap: "4px 14px", marginBottom: "6px" } },
    ui.spaceBadge("input", "input space · one simplex per block"));
  ui.synapseToggle({ parent: head, label: "show synapses", onChange: on => { synOn = on; if (cur) drawPanels(cur); } });
  const grid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "10px 18px" } });
  const nvHead = ui.panelHead("The whole network: all four blocks", null);
  const nvSub = ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" }, html: "The same neurons and colours at random positions, each labelled with its block; inputs (squares) at random too. Lines: live lateral synapses, width ∝ $w_{ij}$. Hover a neuron here or in a triangle to ring it in both; hover an input to ring the neurons attached to it." });
  const nvHost = ui.h("div");
  L.view.append(head, grid, ui.h("div", { style: { marginTop: "14px" } }, nvHead, nvSub, nvHost));
  ui.renderMath(nvSub);
  const panels = [0, 1, 2, 3].map(k => {
    const cv = ui.h("canvas", { style: { display: "block", width: "100%", cursor: "crosshair" } });
    const head = ui.h("div", { class: "chart-title", style: { marginBottom: "2px" } });
    const capt = ui.h("div", { class: "u-help", style: { minHeight: "2.6em", marginTop: "2px" } });
    const box = ui.h("div", { style: { minWidth: "0" } }, head, cv, capt);
    grid.append(box);
    const pn = { k, cv, head, capt, box, pts: [], W: 0 };
    cv.addEventListener("pointermove", ev => hoverPanel(pn, ev));
    cv.addEventListener("pointerleave", () => { ui.hideTip(); setHi(null); });
    return pn;
  });
  new ResizeObserver(() => { if (cur) drawPanels(cur); }).observe(grid);

  const tp = ui.transport({
    parent: L.view, watch: fig, autoplay: true, maxHistory: 300,
    speeds: [{ value: 0.25, label: "×¼" }, { value: 1, label: "×1" }, { value: 8, label: "×8" }], speed: 0.25,
    timeFmt: t => `t = ${ui.fmtFixed(t, 1)}`,
    onTick: s => tick(s), onSeek: f => { cur = f; show(f); }, onReset: () => restart(),
  });

  // ---------------------------------------------------------------- side: tours, controls, stats, legend
  const tours = ui.chips({ parent: L.side, label: "Tours", items: TOURS_A, active: tour ? tour.id : null, onPick: t => applyTour(t) });
  const vS = ui.slider({ label: "variance of block 1, $v_1$", min: 0.25, max: 8, log: true, sig: 2, value: p.v1, fmt: x => ui.fmt(x, 2),
    marks: [{ value: 1, label: "1 (others)" }], onChange: x => { p.v1 = x; changed(); } });
  const nS = ui.slider({ label: "neurons attached to block 1, $n_1$", min: 4, max: 24, step: 1, value: p.n1, fmt: x => String(x),
    marks: [{ value: 12, label: "12 (others)" }], onChange: x => { p.n1 = x; changed(); } });
  const bS = ui.slider({ label: "fusion scale $b$", min: 0.002, max: 0.2, log: true, sig: 2, value: p.b, fmt: x => ui.fmt(x, 2),
    help: "An edge lives iff $D_{ij}^2 &lt; 2b$, i.e. within the range $\\sqrt{2b}$.", onChange: x => { p.b = x; changed(); } });
  const sS = ui.slider({ label: "seed (attachments and bank)", min: 0, max: 20, step: 1, value: p.seed, fmt: x => String(x), onChange: x => { p.seed = x; changed(); } });
  ui.controls({ parent: L.side, title: "Block 1 and the fusion scale", primary: [vS, nS, bS, sS] });
  const stats = ui.stats({ parent: L.side, items: [
    { key: "k", label: "block 1: components $K_1$", fmt: x => String(x) },
    { key: "n", label: "block 1: neurons", fmt: x => String(x) },
    { key: "d", label: "block 1: largest diameter", fmt: x => ui.fmt(x, 2) },
    { key: "r", label: "stationarity $r$", fmt: x => ui.fmt(x, 2) },
  ] });
  ui.legend({ parent: L.side, items: [
    { label: "neuron; colour: its component in the block (largest first)", color: catColor(0), shape: "dot" },
    { label: "singleton (no live edge)", color: P.other, shape: "dot" },
    { label: "input $g_b$ (clamped)", color: P.boundary, shape: "square" },
    { label: "live synapse ($w_{ij}$; triangles: with show synapses on)", color: P.edge || P.fg, shape: "line" },
    { label: "edge range $\\sqrt{2b}$, same scale", color: P.accent, shape: "line" },
  ] });

  // ---------------------------------------------------------------- simulation
  function setHi(list, from) {
    const key = list == null ? "" : [].concat(list).join(","), old = hi == null ? "" : hi.join(",");
    if (key === old) return;
    hi = list == null ? null : [].concat(list);
    if (nv && from !== "nv") nv.setHighlight(hi);
    if (cur) drawPanels(cur);
  }
  const attachedTo = b => { const A = base.sim.net.A, out = []; for (let i = 0; i < base.N; i++) if (A[i * M.B + b] > 0) out.push(i); return out; };
  function ensureNet() {
    if (nv && nvN === base.N) return;
    nv?.destroy(); nvN = base.N;
    nv = netview(nvHost, { N: base.N, B: M.B, seed: 5, aspect: 2.2, maxHeight: 380, minHeight: 300, radius: nvHost.clientWidth < 600 ? 7 : 10, edgeWidth: 2.8,
      labels: Array.from({ length: base.N }, (_, i) => String(base.home[i] + 1)),
      tooltip: (i, kind) => (kind === "neuron" ? tipNeuron(i) : ui.tipRows(`input g${sub(i + 1)}, block ${M.blockOf(i) + 1}`, [{ name: "neurons attached", value: String(attachedTo(i).length) }], "ringed in both views")),
      onHover: (i, kind) => {
        const list = i == null ? null : kind === "neuron" ? [i] : attachedTo(i);
        nv.setHighlight(list); setHi(list, "nv");
      } });
  }
  function drawNet(f) {
    ensureNet();
    const cols = new Array(f.N).fill(P.interior);
    for (let k = 0; k < 4; k++) compColour(f, k).forEach((v, i) => { cols[i] = v.col; });
    const pr = [], ww = [];
    for (let i = 0; i < f.N; i++) for (let j = i + 1; j < f.N; j++) { const w = f.W[i * f.N + j]; if (w > 1e-9) { pr.push(i, j); ww.push(w); } }
    nv.setColors(cols); nv.setEdges(pr, ww);
    nv.setCorner(`${f.N} neurons · ${M.B} inputs · ${pr.length / 2} live synapses`);
  }
  function restart() {
    base = M.makeBase(p); acc = 0; hi = null;
    const f = frameOf(); cur = f; show(f); return f;
  }
  function frameOf() {
    const a = M.analyseBase(base);
    const N = base.N, W = new Float32Array(base.sim.net.W);
    let wmax = 0; for (let q = 0; q < W.length; q++) wmax = Math.max(wmax, W[q]);
    const done = a.lastChange < M.RUN_TOL || a.steps >= 40000;
    return { t: a.steps * 0.5, a, W, N, wmax, home: base.home, vars: base.vars, b: base.b, done, doneLabel: a.lastChange < M.RUN_TOL ? "fixed point" : "stopped at 40,000 steps" };
  }
  function tick(speed) {
    acc += speed; let n = Math.floor(acc); acc -= n;
    if (n === 0 && tp.playing) return null;
    n = Math.max(1, n);
    const t0 = performance.now();
    for (let q = 0; q < n; q++) { base.sim.step(1); if (base.sim.lastChange < M.RUN_TOL || performance.now() - t0 > 20) break; }
    const f = frameOf(); cur = f; show(f);
    return f;
  }

  // ---------------------------------------------------------------- drawing
  function show(f) { drawPanels(f); drawNet(f); showStats(f); }
  function compColour(f, k) {
    // per neuron in block k: colour by component rank (multi-member only), singletons grey
    const map = new Map(); let r = 0;
    for (const c of f.a.blocks[k].comps) { const col = c.size > 1 ? catColor(r++) : P.other; for (const i of c.members) map.set(i, { col, c }); }
    return map;
  }
  function drawPanels(f) {
    const sideOf = v => Math.sqrt(2 * v * (1 - M.RHO));
    const sMax = Math.max(...f.vars.map(sideOf));
    for (const pn of panels) {
      const W = Math.floor(pn.box.clientWidth); if (!W) continue;
      const spMax = 0.62 * W, Hh = Math.round(spMax * 0.866 + 74), g = hidpi(pn.cv, W, Hh);
      pn.W = W;
      const k = pn.k, v = f.vars[k], s = sideOf(v), ppu = spMax / sMax, sp = s * ppu;
      const R = sp / Math.sqrt(3), cx = W / 2, cy = 20 + spMax * 0.866 - R / 2;   // bases aligned
      const V = [[cx - sp / 2, cy + R / 2], [cx + sp / 2, cy + R / 2], [cx, cy - R]];
      g.clearRect(0, 0, W, Hh);
      // triangle
      g.strokeStyle = P.line; g.lineWidth = 1; g.beginPath(); g.moveTo(...V[0]); g.lineTo(...V[1]); g.lineTo(...V[2]); g.closePath(); g.stroke();
      // neurons of this block
      const blk = f.a.blocks[k], H = f.a.H, B = M.B, cols = compColour(f, k);
      const pos = new Map();
      for (const i of blk.members) {
        let x = 0, y = 0, tot = 0;
        for (let a = 0; a < M.PER; a++) { const h = H[i * B + k * M.PER + a]; x += h * V[a][0]; y += h * V[a][1]; tot += h; }
        pos.set(i, [x / tot, y / tot]);
      }
      // live edges (only with "show synapses" on: this panel is input space)
      if (synOn) for (const i of blk.members) for (const j of blk.members) {
        if (j <= i) continue; const w = f.W[i * f.N + j]; if (!(w > 1e-9)) continue;
        const a1 = pos.get(i), a2 = pos.get(j);
        g.strokeStyle = `rgba(40,40,44,${Math.max(0.08, 0.7 * w / (f.wmax || 1)).toFixed(3)})`; g.lineWidth = 1;
        g.beginPath(); g.moveTo(...a1); g.lineTo(...a2); g.stroke();
      }
      pn.pts = [];
      for (const i of blk.members) {
        const [x, y] = pos.get(i), c = cols.get(i);
        g.beginPath(); g.arc(x, y, 4.2, 0, 2 * Math.PI); g.fillStyle = c ? c.col : P.interior; g.fill();
        g.lineWidth = 1; g.strokeStyle = P.surface; g.stroke();
        pn.pts.push({ i, x, y, c: c && c.c });
      }
      if (hi) for (const q of pn.pts) if (hi.includes(q.i)) { g.beginPath(); g.arc(q.x, q.y, 7.5, 0, 2 * Math.PI); g.lineWidth = 2.2; g.strokeStyle = P.accent; g.stroke(); }
      // inputs: squares at the corners, labelled g_b (global index), pushed outward
      g.font = "600 11.5px Inter, system-ui, sans-serif";
      V.forEach(([x, y], a) => {
        g.fillStyle = P.boundary; g.fillRect(x - 4.5, y - 4.5, 9, 9); g.lineWidth = 1.3; g.strokeStyle = P.fg; g.strokeRect(x - 4.5, y - 4.5, 9, 9);
        const dx = x - cx, dy = y - cy, dn = Math.hypot(dx, dy) || 1, lx = x + 13 * dx / dn, ly = y + 13 * dy / dn;
        g.fillStyle = P.fg; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(`g${sub(k * M.PER + a + 1)}`, lx, ly);
      });
      // edge-range ruler
      const rl = Math.sqrt(2 * f.b) * ppu, rx = 10, ry = Hh - 9, rlen = Math.min(rl, W - 20);
      g.strokeStyle = P.accent; g.lineWidth = 2; g.beginPath(); g.moveTo(rx, ry); g.lineTo(rx + rlen, ry); g.stroke();
      g.lineWidth = 1; g.beginPath(); g.moveTo(rx, ry - 4); g.lineTo(rx, ry + 4); g.moveTo(rx + rlen, ry - 4); g.lineTo(rx + rlen, ry + 4); g.stroke();
      g.fillStyle = P.mute; g.font = "11px Inter, system-ui, sans-serif"; g.textAlign = "left"; g.textBaseline = "alphabetic";
      g.fillText(`√(2b) = ${ui.fmt(Math.sqrt(2 * f.b), 2)}${rl > W - 20 ? " (clipped)" : ""}`, rx, ry - 7);
      // header and caption
      pn.head.innerHTML = `block ${k + 1} · $v_${k + 1} = ${ui.fmt(v, 2)}$`;
      const sizes = blk.comps.map(c => c.size), dmax = Math.max(0, ...blk.comps.map(c => c.diamMax));
      pn.capt.innerHTML = `${blk.members.length} neurons → <b>${blk.K}</b> component${blk.K === 1 ? "" : "s"}<br>sizes ${sizes.length > 9 ? sizes.slice(0, 9).join("·") + "…" : sizes.join("·")} · diameter ${ui.fmt(dmax, 2)}`;
      ui.renderMath(pn.head);
    }
  }
  function hoverPanel(pn, ev) {
    if (!cur || !pn.pts.length) return;
    const r = pn.cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = null, bd = 100;
    for (const q of pn.pts) { const d = (q.x - x) ** 2 + (q.y - y) ** 2; if (d < bd) { bd = d; best = q; } }
    if (!best) { ui.hideTip(); setHi(null); return; }
    setHi(best.i);
    ui.showTip(ev.clientX, ev.clientY, tipNeuron(best.i));
  }
  function tipNeuron(i) {
    const H = cur.a.H, k = base.home[i], blk = cur.a.blocks[k], c = (blk.comps.find(q => q.members.includes(i)) || null);
    const rows = [0, 1, 2].map(a => ({ name: `share of g${sub(k * M.PER + a + 1)}, hᵢ`, value: ui.fmtFixed(H[i * M.B + k * M.PER + a], 3), color: P.boundary, shape: "square" }));
    if (c) {
      rows.push({ name: "component size", value: String(c.size) });
      rows.push({ name: "component diameter (max D to prototype)", value: ui.fmt(c.diamMax, 3) });
      rows.push({ name: "component span (max pairwise D)", value: `${ui.fmt(c.span, 3)} = ${ui.fmt(c.span / Math.sqrt(2 * cur.b), 2)} √(2b)` });
    }
    return ui.tipRows(`neuron ${i}, block ${k + 1}`, rows);
  }
  function showStats(f) {
    const b1 = f.a.blocks[0], sizes = b1.comps.map(c => c.size);
    stats.set("k", b1.K, `sizes ${sizes.length > 10 ? sizes.slice(0, 10).join(" · ") + " …" : sizes.join(" · ")}`);
    const tot = sizes.reduce((x, y) => x + y, 0);
    stats.set("n", tot, `attached: ${b1.members.length}; components spanning two blocks: ${f.a.mixed}`);
    const c = b1.comps.reduce((m, q) => (q.diamMax > (m ? m.diamMax : -1) ? q : m), null);
    stats.set("d", c ? c.diamMax : 0, c ? `max D to prototype (Σ_g); its span is ${ui.fmt(c.span / Math.sqrt(2 * f.b), 2)} √(2b)` : "");
    stats.set("r", f.a.residual, `${f.a.residual < M.RES_TOL ? "below 10⁻⁶: fixed point" : "still moving"}; neuronal ${ui.fmt(f.a.eqResidual, 1)}`);
  }

  // ---------------------------------------------------------------- state plumbing
  function syncControls() { vS.set(p.v1); nS.set(p.n1); bS.set(p.b); sS.set(p.seed); }
  function writeURL() {
    const o = { ta: tour ? tour.id : null };
    for (const k in KEYS_A) o[KEYS_A[k]] = tour || p[k] === DEF_A[k] ? null : p[k];
    ctx.setParams(o);
  }
  function changed() { if (tour) { tour = null; tours.moved(); } writeURL(); tp.reset(); tp.play(); }
  function applyTour(t) { tour = t; p = { ...t.p }; syncControls(); writeURL(); tp.reset(); tp.play(); }
  ctx.onParams(params => {
    if (![...params.keys()].some(k => k === "ta" || Object.values(KEYS_A).includes(k))) return;
    const r = readURL();
    if (r.tour) { tours.set(r.tour.id); applyTour(r.tour); return; }
    tour = null; tours.clear(); p = r.p; syncControls(); tp.reset(); tp.play();
  });

  tp.record(restart());
  return { tp };
}

// ---------------------------------------------------------------- precomputed: sweeps of block 1
const meanSd = xs => { const m = xs.reduce((a, b) => a + b, 0) / xs.length; return [m, Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length)]; };
export function groupPlanted(data, kind) {
  const rows = data.a.planted.filter(r => r.kind === kind), bs = [...new Set(rows.map(r => r.b))].sort((a, b) => a - b);
  const vals = [...new Set(rows.map(r => r.val))].sort((a, b) => a - b);
  return { bs, vals, at: (b, v) => rows.filter(r => r.b === b && r.val === v), rows };
}
export { meanSd };

export function sweepCharts(data, elVar, elCount) {
  const gv = groupPlanted(data, "variance");
  const seriesV = gv.bs.map((b, s) => {
    const ms = gv.vals.map(v => meanSd(gv.at(b, v).map(r => r.K_block0)));
    return { name: `b = ${b}`, x: gv.vals, y: ms.map(m => m[0]), band: [ms.map(m => m[0] - m[1]), ms.map(m => m[0] + m[1])], dots: true, color: catColor(s) };
  });
  const n0 = gv.at(gv.bs[0], gv.vals[0])[0].n_block0;
  seriesV.push({ name: `n₁ = ${n0} neurons attached`, x: gv.vals, y: gv.vals.map(() => n0), dash: true, color: P.mute, width: 1.5 });
  lineChart(elVar, { series: seriesV, xlog: true, yrange: [0, n0 + 1], xlabel: "variance of block 1, v₁", ylabel: "components K₁", xname: "v₁", height: 230 });

  const gc = groupPlanted(data, "count");
  const seriesC = gc.bs.map((b, s) => {
    const ms = gc.vals.map(v => meanSd(gc.at(b, v).map(r => r.mean_size_block0)));
    return { name: `b = ${b}`, x: gc.vals, y: ms.map(m => m[0]), band: [ms.map(m => m[0] - m[1]), ms.map(m => m[0] + m[1])], dots: true, color: catColor(s) };
  });
  seriesC.push({ name: "size = n₁", x: gc.vals, y: gc.vals, dash: true, color: P.mute, width: 1.5 });
  lineChart(elCount, { series: seriesC, xlabel: "neurons attached to block 1, n₁", ylabel: "mean component size", xname: "n₁", height: 230, yrange: [0, Math.max(...gc.vals) + 1] });
}

// Spearman rank correlation (average ranks for ties)
export function spearman(x, y) {
  const rank = v => { const idx = v.map((_, i) => i).sort((a, b) => v[a] - v[b]), r = new Array(v.length); let i = 0;
    while (i < idx.length) { let j = i; while (j + 1 < idx.length && v[idx[j + 1]] === v[idx[i]]) j++; for (let q = i; q <= j; q++) r[idx[q]] = (i + j) / 2; i = j + 1; } return r; };
  const rx = rank(x), ry = rank(y), n = x.length, mx = rx.reduce((a, b) => a + b, 0) / n, my = ry.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < n; i++) { sxy += (rx[i] - mx) * (ry[i] - my); sxx += (rx[i] - mx) ** 2; syy += (ry[i] - my) ** 2; }
  return sxy / Math.sqrt(sxx * syy);
}
export function randomStats(data) {
  const bs = [...new Set(data.a.random.map(r => r.b))].sort((a, b) => a - b);
  return bs.map(b => {
    const pts = data.a.random.filter(r => r.b === b).flatMap(r => r.pts.map(q => ({ size: q[0], pv: q[1], dmax: q[2], seed: r.seed })));
    const multi = pts.filter(q => q.size > 1);
    return { b, pts, multi, rho: spearman(multi.map(q => q.size), multi.map(q => q.pv)), n: multi.length, conv: data.a.random.filter(r => r.b === b).every(r => r.converged) };
  });
}

export function randomFigure(data, el) {
  const rs = randomStats(data);
  let sel = rs.length - 1;
  const note = ui.h("div", { class: "u-help", style: { margin: "4px 0 6px" } });
  const seg = ui.segmented({ label: "fusion scale", options: rs.map((r, i) => ({ value: i, label: `b = ${r.b}` })), value: sel, onChange: i => { sel = i; draw(); } });
  const box = ui.h("div");
  el.replaceChildren(seg.el, note, box);
  const ymax = Math.max(...rs.flatMap(r => r.pts.map(q => q.size)));
  const xmax = Math.max(...rs.flatMap(r => r.pts.map(q => q.pv)));
  const sc = scatter(box, { x: [], y: [], height: 280, xlabel: "prototype variance (evaluation bank)", ylabel: "component size", xrange: [0, xmax * 1.05], yrange: [0, ymax + 1] });
  function draw() {
    const r = rs[sel];
    sc.update({ x: r.pts.map(q => q.pv), y: r.pts.map(q => q.size), color: r.pts.map(q => (q.size > 1 ? P.ink2 : P.other)), size: r.pts.map(q => (q.size > 1 ? 4.5 : 3)),
      tooltip: i => ui.tipRows(`component, seed ${r.pts[i].seed}`, [{ name: "size", value: String(r.pts[i].size) }, { name: "prototype variance", value: ui.fmt(r.pts[i].pv, 3) }, { name: "diameter (max D to prototype)", value: ui.fmt(r.pts[i].dmax, 3) }]) });
    note.innerHTML = `Spearman $\\rho$(size, prototype variance) over the ${r.n} components with at least two members: <b>${ui.fmt(r.rho, 2)}</b>. ${r.pts.length} components in all, ${r.conv ? "all runs converged" : "not all runs converged"}.`;
    ui.renderMath(note);
  }
  draw();
}
