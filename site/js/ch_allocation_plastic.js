// Chapter "allocation", section 2: learning attachments under the energy rule. A live figure (the rows of A as
// stacked columns, mean block shares over time) and the sweep as one stacked bar per setting.
import * as ui from "./ui.js";
import { lineChart, hidpi } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as O from "./onsite_engine.js";
import * as M from "./ch_allocation_model.js";

export const DEF_B = { alphaA: 1, lateral: false };
const LAT_B = 0.1;                                    // lateral fusion scale when lateral learning is on
const BLOCK_COL = [0, 1, 2, 3].map(k => catColor(k));
const vlab = v => String(v);

function clampB(p) { return { alphaA: Math.min(30, Math.max(0.03, +p.alphaA || 1)), lateral: !!p.lateral }; }

// The live figure: (a) each neuron's attachment row as a stacked column, (b) the mean share of each block over
// time. Two controls: the ridge alpha_A and lateral learning (distance rule, b = 0.1). Linear neurons, seed 0.
export function plasticFigure(ctx, slot) {
  const fig = slot.closest("figure");
  const readURL = () => clampB({ alphaA: ctx.param("ba", DEF_B.alphaA), lateral: ctx.param("bl", DEF_B.lateral) });
  let p = readURL();
  let st = null, cur = null, hist = null, tickN = 0;
  const lamMax = O.eigh(M.sigmaBlocks(M.VARS_B), M.B).values[0];

  const L = ui.simLayout({ parent: slot });
  const tp = ui.transport({
    parent: L.view, watch: fig, autoplay: true, maxHistory: 300,
    speeds: [{ value: 1, label: "×1" }, { value: 5, label: "×5" }, { value: 25, label: "×25" }], speed: 5,
    timeFmt: t => `t = ${ui.fmtFixed(t, 1)}`,
    onTick: s => tick(s), onSeek: f => { cur = f; show(f, true); }, onReset: () => restart(),
  });
  const ttl = (t, s) => ui.h("div", {}, ui.h("div", { class: "chart-title", html: t }), ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" }, html: s }));
  const colsHead = ttl("(a) Each neuron's attachments", "one column per neuron; each coloured segment is one attachment $a_{ib}$, coloured by the block of input $b$");
  const colsCv = ui.h("canvas", { style: { display: "block", width: "100%" } });
  const shareHead = ttl("(b) Mean share of each block over time", "average over neurons of the attachments to that block's three inputs");
  const shareBox = ui.h("div");
  L.view.append(ui.h("div", { style: { marginTop: "12px" } }, colsHead, colsCv), ui.h("div", { style: { marginTop: "14px" } }, shareHead, shareBox));
  ui.renderMath(L.view);
  const shareChart = lineChart(shareBox, { series: [0, 1, 2, 3].map(k => ({ name: `block ${k + 1} (variance ${vlab(M.VARS_B[k])})`, x: [0], y: [0.25], color: BLOCK_COL[k] })), yrange: [0, 1], xlabel: "time t", ylabel: "mean share", xname: "t", height: 190 });
  new ResizeObserver(() => { if (cur) drawCols(cur); }).observe(colsCv.parentElement);
  colsCv.addEventListener("pointermove", ev => hoverCols(ev));
  colsCv.addEventListener("pointerleave", () => { ui.hideTip(); hiCol(null); });

  // ---------------------------------------------------------------- side
  const aS = ui.slider({ label: "attachment ridge $\\alpha_A$", min: 0.03, max: 30, log: true, sig: 2, value: p.alphaA, fmt: x => ui.fmt(x, 2),
    marks: [{ value: lamMax, label: "λmax(Σ)" }], help: "Above $\\lambda_{\\max}(\\Sigma_g)$ (the mark) every row converges to the same spread row.", onChange: x => { p.alphaA = x; changed(); } });
  const latT = ui.toggle({ label: `lateral learning too (distance rule, $b = ${LAT_B}$)`, value: p.lateral, onChange: x => { p.lateral = x; changed(); } });
  ui.controls({ parent: L.side, title: "Learning attachments", primary: [aS, latT] });
  const stats = ui.stats({ parent: L.side, items: [
    { key: "maj", label: "majority on block 1 · 2 · 3 · 4", fmt: x => x },
    { key: "q", label: "mean share, quietest block", fmt: x => ui.fmtFixed(x, 2) },
  ] });
  ui.legend({ parent: L.side, items: M.VARS_B.map((v, k) => ({ label: `block ${k + 1}, variance ${v} (inputs g${subs(3 * k + 1)}–g${subs(3 * k + 3)})`, color: BLOCK_COL[k], shape: "square" })) });

  // ---------------------------------------------------------------- simulation
  function restart() {
    st = M.makePlastic({ alphaA: p.alphaA, lateral: p.lateral, b: LAT_B, seed: 0 }); tickN = 0;
    hist = { t: [0], s: [[], [], [], []] };
    const sm = M.plasticSummary(st); sm.share.forEach((x, k) => hist.s[k].push(x));
    st.res = NaN;
    tp.setSpeed(p.alphaA < 0.3 ? 25 : 5);              // a weak ridge settles slowly
    const f = frameOf(); cur = f; show(f, true); return f;
  }
  function frameOf() {
    const sm = M.plasticSummary(st);
    const conv = st.res < 1e-8, out = st.steps >= 6000;
    return { t: st.steps * st.dt, A: new Float64Array(st.net.A), sm, res: st.res, steps: st.steps,
      done: conv || out, doneLabel: conv ? "converged" : "stopped at 6000 steps" };
  }
  function tick(speed) {
    const t0 = performance.now();
    for (let q = 0; q < speed; q++) { st.step(); if (st.res < 1e-8 || st.steps >= 6000 || performance.now() - t0 > 18) break; }
    const f = frameOf(); cur = f; tickN++;
    hist.t.push(f.t); f.sm.share.forEach((x, k) => hist.s[k].push(x));
    show(f, f.done || tickN % 6 === 0);
    return f;
  }
  function show(f, heavy) {
    drawCols(f); showStats(f);
    if (heavy) shareChart.update({ series: [0, 1, 2, 3].map(k => ({ name: `block ${k + 1} (variance ${vlab(M.VARS_B[k])})`, x: hist.t, y: hist.s[k], color: BLOCK_COL[k] })), xrange: [0, Math.max(1, hist.t[hist.t.length - 1])] });
    shareChart.setCursor(f.t);
  }
  let colGeom = null, colHi = null;
  function drawCols(f) {
    const W = Math.floor(colsCv.parentElement.clientWidth); if (!W) return;
    const Hh = 200, g = hidpi(colsCv, W, Hh), N = st.N, B = M.B, ml = 34, mb = 18, mt = 4;
    g.clearRect(0, 0, W, Hh);
    const order = [...Array(N).keys()].sort((i, j) => f.sm.home[i] - f.sm.home[j] || f.sm.mass[j][f.sm.home[j]] - f.sm.mass[i][f.sm.home[i]] || i - j);
    const cw = (W - ml - 4) / N, ph = Hh - mb - mt;
    g.font = "11px Inter, system-ui, sans-serif"; g.fillStyle = P.mute; g.textAlign = "right";
    [0, 0.5, 1].forEach(y => { const yy = mt + ph * (1 - y); g.fillText(String(y), ml - 6, yy + 3.5); g.strokeStyle = P.line; g.beginPath(); g.moveTo(ml, Math.round(yy) + .5); g.lineTo(W - 4, Math.round(yy) + .5); g.stroke(); });
    order.forEach((i, c) => {
      let y = mt + ph;
      for (let b = 0; b < B; b++) {
        const a = f.A[i * B + b]; if (a <= 0) continue;
        const hgt = a * ph, k = M.blockOf(b);
        g.fillStyle = BLOCK_COL[k]; g.fillRect(ml + c * cw + 0.5, y - hgt, Math.max(1, cw - 1), hgt);
        y -= hgt;
        if (hgt > 2) { g.fillStyle = "rgba(255,255,255,0.9)"; g.fillRect(ml + c * cw + 0.5, y, Math.max(1, cw - 1), 1); }
      }
      if (i === colHi) { g.strokeStyle = P.accent; g.lineWidth = 2; g.strokeRect(ml + c * cw - 0.5, mt - 1, cw + 1, ph + 2); }
    });
    g.fillStyle = P.ink2; g.font = "560 11.5px Inter, system-ui, sans-serif"; g.textAlign = "center";
    g.fillText("neurons, sorted by the block holding most of their attachments", (ml + W) / 2, Hh - 3);
    g.save(); g.translate(11, mt + ph / 2); g.rotate(-Math.PI / 2); g.fillText("attachment", 0, 0); g.restore();
    colGeom = { ml, cw, order, f };
  }
  function hoverCols(ev) {
    if (!colGeom) return;
    const r = colsCv.getBoundingClientRect(), c = Math.floor((ev.clientX - r.left - colGeom.ml) / colGeom.cw);
    if (c < 0 || c >= colGeom.order.length) { ui.hideTip(); hiCol(null); return; }
    const i = colGeom.order[c];
    hiCol(i);
    ui.showTip(ev.clientX, ev.clientY, tipRow(i));
  }
  function hiCol(i) { if (i === colHi) return; colHi = i; if (cur) drawCols(cur); }
  function tipRow(i) {
    const f = colGeom ? colGeom.f : cur, m = f.sm.mass[i];
    const rows = m.map((x, k) => ({ name: `block ${k + 1} (variance ${vlab(M.VARS_B[k])}) share`, value: ui.fmtFixed(x, 3), color: BLOCK_COL[k], shape: "square" }));
    let sup = 0; for (let b = 0; b < M.B; b++) if (f.A[i * M.B + b] > 1e-6) sup++;
    rows.push({ name: "inputs with a_ib > 10⁻⁶", value: String(sup) });
    return ui.tipRows(`neuron ${i}`, rows);
  }
  function showStats(f) {
    stats.set("maj", f.sm.maj.join(" · "), `of ${st.N} neurons, more than half the attachments on that block`);
    stats.set("q", f.sm.share[3], `block 4, variance ${vlab(M.VARS_B[3])}`);
  }

  // ---------------------------------------------------------------- state plumbing
  function syncControls() { aS.set(p.alphaA); latT.set(p.lateral); }
  function writeURL() { ctx.setParams({ ba: p.alphaA === DEF_B.alphaA ? null : p.alphaA, bl: p.lateral ? true : null }); }
  function changed() { writeURL(); tp.reset(); tp.play(); }
  ctx.onParams(params => {
    if (![...params.keys()].some(k => k === "ba" || k === "bl")) return;
    p = readURL(); syncControls(); tp.reset(); tp.play();
  });
  tp.record(restart());
  return { tp };
}
const subs = n => String(n).split("").map(d => "₀₁₂₃₄₅₆₇₈₉"[+d]).join("");

// ---------------------------------------------------------------- the sweep, as one stacked bar per setting
// Linear runs only (the sweep's rectifier rows are not shown). Each bar: the 40 neurons split by the block holding
// more than half of their attachments (grey: no majority), mean over seeds.
export function plasticBars(data, el) {
  const N = data.meta.config.b.N;
  const rows0 = data.b.filter(r => !r.tag.startsWith("rect"));
  const tags = [...new Set(rows0.map(r => r.tag))];
  const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
  const rows = tags.map(tag => {
    const rs = rows0.filter(r => r.tag === tag), r0 = rs[0];
    const maj = [0, 1, 2, 3].map(k => mean(rs.map(r => r.alloc_majority[k])));
    const none = N - maj.reduce((s, x) => s + x, 0);
    return { tag, aA: r0.alphaA, lat: r0.lateral, b: r0.b, maj, none, quiet: mean(rs.map(r => r.mass_per_block[3])), conv: rs.filter(r => r.converged).length, n: rs.length };
  }).sort((a, b) => (a.lat - b.lat) || (a.lat ? a.b - b.b : a.aA - b.aA));
  const label = r => (r.lat ? `ridge ${r.aA}, lateral learning, b = ${r.b}` : `ridge ${r.aA}, no lateral learning`) + (r.conv < r.n ? `<br><span style="color:var(--mute)">${r.conv} of ${r.n} converged</span>` : "");
  const bar = r => {
    const seg = (n, col, name) => (n > 0.05 ? `<span title="${name}: ${ui.fmtFixed(n, 1)} of ${N} neurons" style="flex:${n} 0 0;background:${col};height:100%;display:flex;align-items:center;justify-content:center;color:#fff;font:600 11px Inter,system-ui,sans-serif;overflow:hidden;white-space:nowrap">${n >= 3 ? ui.fmtFixed(n, 1) : ""}</span>` : "");
    return `<div style="display:flex;height:22px;border-radius:3px;overflow:hidden;background:var(--wash)">${r.maj.map((n, k) => seg(n, BLOCK_COL[k], `block ${k + 1} (variance ${M.VARS_B[k]})`)).join("")}${seg(r.none, P.other, "no majority")}</div>`;
  };
  const lines = rows.map(r => `<div style="display:grid;grid-template-columns:minmax(150px,240px) 1fr;gap:4px 12px;align-items:center;margin:5px 0"><div style="font:12.5px/1.3 Inter,system-ui,sans-serif;color:var(--ink-2)">${label(r)}</div>${bar(r)}</div>`).join("");
  const key = [...M.VARS_B.map((v, k) => ({ label: `majority on block ${k + 1} (variance ${v})`, color: BLOCK_COL[k], shape: "square" })), { label: "no majority", color: P.other, shape: "square" }];
  el.innerHTML = `<div>${lines}</div><div style="display:grid;grid-template-columns:minmax(150px,240px) 1fr;gap:12px"><span></span><div style="display:flex;justify-content:space-between;font:11px Inter,system-ui,sans-serif;color:var(--mute)"><span>0</span><span>${N} neurons</span></div></div>`;
  ui.legend({ parent: el, items: key });
  return rows;
}
