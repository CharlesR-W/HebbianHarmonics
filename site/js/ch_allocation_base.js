// Chapter "allocation", section 1: fixed (block) attachments. A live figure (four block triangles drawn as exact
// isometric maps at one common scale; block 1's variance, its neuron count and b as sliders) and the precomputed
// variance sweep.
import * as ui from "./ui.js";
import { lineChart, hidpi } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as M from "./ch_allocation_model.js";

const SUBD = "₀₁₂₃₄₅₆₇₈₉";
export const sub = n => String(n).split("").map(d => SUBD[+d] ?? d).join("");

export const DEF_A = { v1: 8, n1: 12, b: 0.02, seed: 0 };
const KEYS_A = { v1: "av", n1: "an", b: "ab" };

function clampA(p) {
  return {
    v1: Math.min(8, Math.max(0.25, +p.v1 || DEF_A.v1)),
    n1: Math.min(24, Math.max(4, Math.round(+p.n1 || DEF_A.n1))),
    b: Math.min(0.2, Math.max(0.002, +p.b || DEF_A.b)),
    seed: 0,
  };
}

// The live figure: four block triangles in input space at one common scale, three sliders (v1, n1, b).
export function baseFigure(ctx, slot) {
  const fig = slot.closest("figure");
  const readURL = () => { const p = {}; for (const k in KEYS_A) p[k] = ctx.param(KEYS_A[k], DEF_A[k]); return clampA(p); };
  let p = readURL();
  let base = null, cur = null, acc = 0, hi = null;

  const L = ui.simLayout({ parent: slot });
  const head = ui.h("div", { class: "panel-head", style: { marginBottom: "6px" } }, ui.spaceBadge("input", "input space · one scale"));
  const grid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "10px 16px" } });
  L.view.append(head, grid);
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
    speeds: [{ value: 1, label: "×1" }, { value: 8, label: "×8" }, { value: 40, label: "×40" }], speed: 8,
    timeFmt: t => `t = ${ui.fmtFixed(t, 1)}`,
    onTick: s => tick(s), onSeek: f => { cur = f; show(f); }, onReset: () => restart(),
  });

  // ---------------------------------------------------------------- side: three sliders, two tiles, legend
  const vS = ui.slider({ label: "variance of block 1, $v_1$", min: 0.25, max: 8, log: true, sig: 2, value: p.v1, fmt: x => ui.fmt(x, 2),
    marks: [{ value: 1, label: "1 (others)" }], onChange: x => { p.v1 = x; changed(); } });
  const nS = ui.slider({ label: "neurons attached to block 1, $n_1$", min: 4, max: 24, step: 1, value: p.n1, fmt: x => String(x),
    marks: [{ value: 12, label: "12 (others)" }], onChange: x => { p.n1 = x; changed(); } });
  const bS = ui.slider({ label: "fusion scale $b$", min: 0.002, max: 0.2, log: true, sig: 2, value: p.b, fmt: x => ui.fmt(x, 2),
    help: "A synapse survives only within the range $\\sqrt{2b}$ (the bar under each triangle).", onChange: x => { p.b = x; changed(); } });
  ui.controls({ parent: L.side, title: "Block 1 and the fusion scale", primary: [vS, nS, bS] });
  const stats = ui.stats({ parent: L.side, items: [
    { key: "n", label: "block 1: neurons", fmt: x => String(x) },
    { key: "k", label: "block 1: components", fmt: x => String(x) },
  ] });
  ui.legend({ parent: L.side, items: [
    { label: "neuron; colour: its component within the block", color: catColor(0), shape: "dot" },
    { label: "neuron with no live synapse", color: P.other, shape: "dot" },
    { label: "input $g_b$", color: P.boundary, shape: "square" },
    { label: "edge range $\\sqrt{2b}$, same scale", color: P.accent, shape: "line" },
  ] });

  // ---------------------------------------------------------------- simulation
  function setHi(i) { if (i === hi) return; hi = i; if (cur) drawPanels(cur); }
  function restart() { base = M.makeBase(p); acc = 0; hi = null; const f = frameOf(); cur = f; show(f); return f; }
  function frameOf() {
    const a = M.analyseBase(base);
    const done = a.lastChange < M.RUN_TOL || a.steps >= 40000;
    return { t: a.steps * 0.5, a, N: base.N, vars: base.vars, b: base.b, done, doneLabel: a.lastChange < M.RUN_TOL ? "fixed point" : "stopped at 40,000 steps" };
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
  function show(f) { drawPanels(f); showStats(f); }
  function compColour(f, k) {
    // per neuron in block k: colour by component rank (multi-member only), neurons without a live synapse grey
    const map = new Map(); let r = 0;
    for (const c of f.a.blocks[k].comps) { const col = c.size > 1 ? catColor(r++) : P.other; for (const i of c.members) map.set(i, { col, c }); }
    return map;
  }
  function drawPanels(f) {
    const sideOf = v => Math.sqrt(2 * v * (1 - M.RHO));
    const sMax = Math.max(...f.vars.map(sideOf));
    for (const pn of panels) {
      const W = Math.floor(pn.box.clientWidth); if (!W) continue;
      const spMax = 0.78 * W, Hh = Math.round(spMax * 0.866 + 64), g = hidpi(pn.cv, W, Hh);
      pn.W = W;
      const k = pn.k, v = f.vars[k], s = sideOf(v), ppu = spMax / sMax, sp = s * ppu;
      const R = sp / Math.sqrt(3), cx = W / 2, cy = 20 + spMax * 0.866 - R / 2;   // bases aligned
      const V = [[cx - sp / 2, cy + R / 2], [cx + sp / 2, cy + R / 2], [cx, cy - R]];
      g.clearRect(0, 0, W, Hh);
      g.strokeStyle = P.line; g.lineWidth = 1; g.beginPath(); g.moveTo(...V[0]); g.lineTo(...V[1]); g.lineTo(...V[2]); g.closePath(); g.stroke();
      const blk = f.a.blocks[k], H = f.a.H, B = M.B, cols = compColour(f, k);
      pn.pts = [];
      for (const i of blk.members) {
        let x = 0, y = 0, tot = 0;
        for (let a = 0; a < M.PER; a++) { const h = H[i * B + k * M.PER + a]; x += h * V[a][0]; y += h * V[a][1]; tot += h; }
        x /= tot; y /= tot;
        const c = cols.get(i);
        g.beginPath(); g.arc(x, y, 4.2, 0, 2 * Math.PI); g.fillStyle = c ? c.col : P.interior; g.fill();
        g.lineWidth = 1; g.strokeStyle = P.surface; g.stroke();
        pn.pts.push({ i, x, y, c: c && c.c });
      }
      if (hi != null) for (const q of pn.pts) if (q.i === hi) { g.beginPath(); g.arc(q.x, q.y, 7.5, 0, 2 * Math.PI); g.lineWidth = 2.2; g.strokeStyle = P.accent; g.stroke(); }
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
      g.fillText(`edge range √(2b)${rl > W - 20 ? " (clipped)" : ""}`, rx, ry - 7);
      pn.head.innerHTML = `block ${k + 1} · variance ${ui.fmt(v, 2)}`;
      pn.capt.innerHTML = `${blk.members.length} neurons → <b>${blk.K}</b> component${blk.K === 1 ? "" : "s"}`;
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
    const rows = [0, 1, 2].map(a => ({ name: `share of g${sub(k * M.PER + a + 1)}`, value: ui.fmtFixed(H[i * M.B + k * M.PER + a], 3), color: P.boundary, shape: "square" }));
    if (c) rows.push({ name: "component size", value: String(c.size) });
    return ui.tipRows(`neuron ${i}, block ${k + 1}`, rows);
  }
  function showStats(f) {
    const b1 = f.a.blocks[0], sizes = b1.comps.map(c => c.size), tot = sizes.reduce((x, y) => x + y, 0);
    stats.set("n", tot, f.a.mixed ? `${f.a.mixed} component(s) span two blocks` : "no component spans two blocks");
    stats.set("k", b1.K, `sizes ${sizes.length > 10 ? sizes.slice(0, 10).join(" · ") + " …" : sizes.join(" · ")}`);
  }

  // ---------------------------------------------------------------- state plumbing
  function syncControls() { vS.set(p.v1); nS.set(p.n1); bS.set(p.b); }
  function writeURL() { const o = {}; for (const k in KEYS_A) o[KEYS_A[k]] = p[k] === DEF_A[k] ? null : p[k]; ctx.setParams(o); }
  function changed() { writeURL(); tp.reset(); tp.play(); }
  ctx.onParams(params => {
    if (![...params.keys()].some(k => Object.values(KEYS_A).includes(k))) return;
    p = readURL(); syncControls(); tp.reset(); tp.play();
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

export function sweepCharts(data, elVar) {
  const gv = groupPlanted(data, "variance");
  const seriesV = gv.bs.map((b, s) => {
    const ms = gv.vals.map(v => meanSd(gv.at(b, v).map(r => r.K_block0)));
    return { name: `components, b = ${b}`, x: gv.vals, y: ms.map(m => m[0]), band: [ms.map(m => m[0] - m[1]), ms.map(m => m[0] + m[1])], dots: true, color: catColor(s) };
  });
  const n0 = gv.at(gv.bs[0], gv.vals[0])[0].n_block0;
  const nObs = gv.vals.map(v => meanSd(gv.rows.filter(r => r.val === v).map(r => r.n_block0))[0]);
  seriesV.push({ name: `neurons on block 1 (${n0} attached)`, x: gv.vals, y: nObs, dash: true, color: P.fg, width: 1.5 });
  lineChart(elVar, { series: seriesV, xlog: true, yrange: [0, n0 + 1], xlabel: "variance of block 1 (others: 1)", ylabel: "count", xname: "variance", height: 250 });
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
