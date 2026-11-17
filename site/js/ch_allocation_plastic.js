// Chapter "allocation", section (b): plastic attachments under the energy rule. A live figure (the rows of A as
// stacked columns, mean block shares over time, and the per-block row costs of eq. 4) and the sweep's table.
import * as ui from "./ui.js";
import { lineChart, hidpi } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as O from "./onsite_engine.js";
import * as M from "./ch_allocation_model.js";

export const DEF_B = { alphaA: 1, lateral: false, b: 0.1, rect: false, seed: 0 };
const KEYS_B = { alphaA: "ba", lateral: "bl", b: "bb", rect: "bp", seed: "bs" };
export const TOURS_B = [
  { id: "drain", label: "Drain to the quiet block", note: "$\\alpha_A = 1$, linear, no lateral learning: the case eq. (3)–(4) describe exactly.", p: { alphaA: 1, lateral: false, b: 0.1, rect: false, seed: 0 } },
  { id: "lateral-rect", label: "With fusion and the rectifier", note: "$\\alpha_A = 1$, lateral rule on ($b = 0.1$, $\\alpha = 1$), rectifying potential: outside the derivation, same destination.", p: { alphaA: 1, lateral: true, b: 0.1, rect: true, seed: 0 } },
  { id: "strong", label: "Strong ridge", note: "$\\alpha_A = 10 &gt; \\lambda_{\\max}(\\Sigma_g)$: each neuron's objective is convex, so every row converges to the same spread row.", p: { alphaA: 10, lateral: false, b: 0.1, rect: false, seed: 0 } },
  { id: "weak", label: "Weak ridge", note: "$\\alpha_A = 0.1 &lt; v_k(1-\\rho)$ on the louder blocks: pure attachments are local minima there; slow to settle.", p: { alphaA: 0.1, lateral: false, b: 0.1, rect: false, seed: 0 }, speed: 25 },
];
const BLOCK_COL = [0, 1, 2, 3].map(k => catColor(k));
const vlab = v => String(v);

function clampB(p) {
  return { alphaA: Math.min(30, Math.max(0.03, +p.alphaA || 1)), lateral: !!p.lateral, b: Math.min(0.5, Math.max(0.005, +p.b || 0.1)), rect: !!p.rect, seed: Math.min(20, Math.max(0, Math.round(+p.seed || 0))) };
}

export function plasticFigure(ctx, slot) {
  const fig = slot.closest("figure");
  const readURL = () => {
    const t = TOURS_B.find(x => x.id === ctx.param("tb", ""));
    if (t) return { p: { ...t.p }, tour: t };
    return { p: clampB({ alphaA: ctx.param("ba", DEF_B.alphaA), lateral: ctx.param("bl", DEF_B.lateral), b: ctx.param("bb", DEF_B.b), rect: ctx.param("bp", "lin") === "rect", seed: ctx.param("bs", 0) }), tour: null };
  };
  let { p, tour } = readURL();
  let st = null, cur = null, hist = null, tickN = 0;
  const lamMax = O.eigh(M.sigmaBlocks(M.VARS_B), M.B).values[0];

  const L = ui.simLayout({ parent: slot });
  const tp = ui.transport({
    parent: L.view, watch: fig, autoplay: false, maxHistory: 300,
    speeds: [{ value: 1, label: "×1" }, { value: 5, label: "×5" }, { value: 25, label: "×25" }], speed: 5,
    timeFmt: t => `t = ${ui.fmtFixed(t, 1)}`,
    onTick: s => tick(s), onSeek: f => { cur = f; show(f, true); }, onReset: () => restart(),
  });
  const ttl = (t, s) => ui.h("div", {}, ui.h("div", { class: "chart-title", html: t }), ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" }, html: s }));
  const colsHead = ttl("(a) Each neuron's attachment row", "one column per neuron, stacked by input, coloured by block");
  const colsCv = ui.h("canvas", { style: { display: "block", width: "100%" } });
  const shareHead = ttl("(b) Mean share of each block", "mean over neurons of $\\sum_{b \\in k} a_{ib}$");
  const shareBox = ui.h("div");
  L.view.append(ui.h("div", { style: { marginTop: "12px" } }, colsHead, colsCv), ui.h("div", { style: { marginTop: "14px" } }, shareHead, shareBox));
  ui.renderMath(L.view);
  const shareChart = lineChart(shareBox, { series: [0, 1, 2, 3].map(k => ({ name: `block ${k + 1} (variance ${vlab(M.VARS_B[k])})`, x: [0], y: [0.25], color: BLOCK_COL[k] })), yrange: [0, 1], xlabel: "time t", ylabel: "mean share", xname: "t", height: 190 });
  new ResizeObserver(() => { if (cur) drawCols(cur); }).observe(colsCv.parentElement);
  colsCv.addEventListener("pointermove", ev => hoverCols(ev));
  colsCv.addEventListener("pointerleave", () => ui.hideTip());

  // ---------------------------------------------------------------- side
  const tours = ui.chips({ parent: L.side, label: "Tours", items: TOURS_B, active: tour ? tour.id : null, onPick: t => applyTour(t) });
  const aS = ui.slider({ label: "attachment ridge $\\alpha_A$", min: 0.03, max: 30, log: true, sig: 2, value: p.alphaA, fmt: x => ui.fmt(x, 2),
    marks: [{ value: lamMax, label: "λmax(Σ)" }], help: "Above $\\lambda_{\\max}(\\Sigma_g)$ each neuron's objective at $W = 0$ is convex.", onChange: x => { p.alphaA = x; changed(); } });
  const latT = ui.toggle({ label: "lateral learning (distance rule, $\\alpha = 1$)", value: p.lateral, onChange: x => { p.lateral = x; bS.setDisabled(!x); changed(); } });
  const potS = ui.segmented({ label: "potential", options: [{ value: false, label: "linear" }, { value: true, label: "rectifying" }], value: p.rect, onChange: x => { p.rect = x; changed(); } });
  const sS = ui.slider({ label: "seed (rows, thresholds, bank)", min: 0, max: 20, step: 1, value: p.seed, fmt: x => String(x), onChange: x => { p.seed = x; changed(); } });
  const bS = ui.slider({ label: "lateral fusion scale $b$", min: 0.005, max: 0.5, log: true, sig: 2, value: p.b, fmt: x => ui.fmt(x, 2), onChange: x => { p.b = x; changed(); } });
  bS.setDisabled(!p.lateral);
  ui.controls({ parent: L.side, title: "Plasticity of the attachments", primary: [aS, latT, potS, sS], advanced: [bS] });
  const stats = ui.stats({ parent: L.side, items: [
    { key: "maj", label: "majority on block 1 · 2 · 3 · 4", fmt: x => x },
    { key: "q", label: "mean share, quiet block", fmt: x => ui.fmtFixed(x, 3) },
    { key: "sup", label: "mean support", fmt: x => ui.fmtFixed(x, 1) },
    { key: "r", label: "stationarity $r$", fmt: x => ui.fmt(x, 2) },
  ] });
  ui.legend({ parent: L.side, items: M.VARS_B.map((v, k) => ({ label: `block ${k + 1}, variance ${v} (inputs g${subs(3 * k + 1)}–g${subs(3 * k + 3)})`, color: BLOCK_COL[k], shape: "square" })) });

  // ---------------------------------------------------------------- the per-block row costs (eq. 4)
  const mech = ui.h("div", { style: { marginTop: "6px" } });
  L.below.append(ui.h("div", { style: { gridColumn: "1 / -1", minWidth: "0" } }, ui.h("div", { class: "chart-title", html: "Best row inside each block at $W = 0$, linear neurons (eq. 4)" }), mech));
  function drawMech() {
    const rows = M.blockRowCosts(M.VARS_B, p.alphaA), best = Math.min(...rows.map(r => r.best));
    const td = (x, cls = "num") => `<td class="${cls}">${x}</td>`;
    const body = rows.map((r, k) => `<tr${r.best === best ? ' style="font-weight:600"' : ""}><td><span class="swatch" style="--sw:${BLOCK_COL[k]}"></span>${k + 1}</td>${td(r.v)}${td(ui.fmtFixed(r.v * (1 - M.RHO), 3))}${td(ui.fmtFixed(r.pure, 3))}${td(ui.fmtFixed(r.even, 3))}${td(r.form, "")}${td(ui.fmtFixed(r.best, 3))}</tr>`).join("");
    mech.innerHTML = `<div class="table-scroll" style="margin:.6em 0"><table class="data"><thead><tr><th>block</th><th class="num">$v_k$</th><th class="num">$v_k(1-\\rho)$</th><th class="num">pure</th><th class="num">even</th><th>best</th><th class="num">$J_i$</th></tr></thead><tbody>${body}</tbody></table></div>
      <p class="u-help" style="margin-top:6px">Pure: one input, cost $\\alpha_A/2$. Even: the block's three inputs equally, cost $v_k(1-\\rho)/3 + \\alpha_A/6$. Best: the cheaper of the two, with its cost $J_i$. Bold: the cheapest block. $\\lambda_{\\max}(\\Sigma_g) = ${ui.fmt(lamMax, 3)}$; at $\\alpha_A = ${ui.fmt(p.alphaA, 2)}$ the per-neuron objective is ${p.alphaA > lamMax ? "<b>strictly convex</b>: one minimiser, shared by every neuron" : "<b>not guaranteed convex</b>: different neurons can stop in different local minima"}. ${p.lateral || p.rect ? "With lateral learning or the rectifier on, the live run is outside this calculation." : ""}</p>`;
    ui.renderMath(mech);
  }

  // ---------------------------------------------------------------- simulation
  function restart() {
    st = M.makePlastic(p); tickN = 0;
    hist = { t: [0], s: [[], [], [], []] };
    const sm = M.plasticSummary(st); sm.share.forEach((x, k) => hist.s[k].push(x));
    st.res = NaN;
    const f = frameOf(); cur = f; show(f, true); drawMech(); return f;
  }
  function frameOf() {
    const sm = M.plasticSummary(st);
    const conv = st.res < 1e-8, out = st.steps >= 6000;
    return { t: st.steps * st.dt, A: new Float64Array(st.net.A), sm, res: st.res, eqRes: st.eqRes, steps: st.steps, K: p.lateral ? O.components(st.net.W, st.N).K : null,
      done: conv || out, doneLabel: conv ? "converged: r < 10⁻⁸" : "stopped at 6000 steps" };
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
    if (heavy) { shareChart.update({ series: [0, 1, 2, 3].map(k => ({ name: `block ${k + 1} (variance ${vlab(M.VARS_B[k])})`, x: hist.t, y: hist.s[k], color: BLOCK_COL[k] })), xrange: [0, Math.max(1, hist.t[hist.t.length - 1])] }); }
    shareChart.setCursor(f.t);
  }
  let colGeom = null;
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
    });
    g.fillStyle = P.ink2; g.font = "560 11.5px Inter, system-ui, sans-serif"; g.textAlign = "center";
    g.fillText("neurons (sorted by the block holding most of the row)", (ml + W) / 2, Hh - 3);
    g.save(); g.translate(11, mt + ph / 2); g.rotate(-Math.PI / 2); g.fillText("attachment a_ib", 0, 0); g.restore();
    colGeom = { ml, cw, order, f };
  }
  function hoverCols(ev) {
    if (!colGeom) return;
    const r = colsCv.getBoundingClientRect(), c = Math.floor((ev.clientX - r.left - colGeom.ml) / colGeom.cw);
    if (c < 0 || c >= colGeom.order.length) { ui.hideTip(); return; }
    const i = colGeom.order[c], f = colGeom.f, m = f.sm.mass[i];
    const rows = m.map((x, k) => ({ name: `block ${k + 1} (variance ${vlab(M.VARS_B[k])}) share`, value: ui.fmtFixed(x, 3), color: BLOCK_COL[k], shape: "square" }));
    let sup = 0; for (let b = 0; b < M.B; b++) if (f.A[i * M.B + b] > 1e-6) sup++;
    rows.push({ name: "inputs with a_ib > 10⁻⁶", value: String(sup) });
    rows.push({ name: "its objective Jᵢ at W = 0 (Σ_g)", value: ui.fmt(M.rowObjective(f.A.subarray(i * M.B, (i + 1) * M.B), st.Sigma, p.alphaA), 3) });
    ui.showTip(ev.clientX, ev.clientY, ui.tipRows(`neuron ${i}`, rows, p.lateral || p.rect ? "Jᵢ here ignores the lateral edges and the potential." : null));
  }
  function showStats(f) {
    stats.set("maj", f.sm.maj.join(" · "), `of ${st.N} neurons, more than half the row on that block`);
    stats.set("q", f.sm.share[3], `block 4, variance ${vlab(M.VARS_B[3])}`);
    stats.set("sup", f.sm.support, "inputs per neuron with a_ib > 10⁻⁶" + (f.K != null ? ` · lateral components: ${f.K}` : ""));
    stats.set("r", Number.isFinite(f.res) ? f.res : NaN, Number.isFinite(f.res) ? `${f.res < M.RES_TOL ? "below 10⁻⁶: equilibrium" : "still moving"}; neuronal ${ui.fmt(f.eqRes, 1)}; step ${f.steps}` : "not started");
  }

  // ---------------------------------------------------------------- state plumbing
  function syncControls() { aS.set(p.alphaA); latT.set(p.lateral); potS.set(p.rect); sS.set(p.seed); bS.set(p.b); bS.setDisabled(!p.lateral); }
  function writeURL() {
    if (tour) { ctx.setParams({ tb: tour.id, ba: null, bl: null, bb: null, bp: null, bs: null }); return; }
    ctx.setParams({ tb: null, ba: p.alphaA === DEF_B.alphaA ? null : p.alphaA, bl: p.lateral ? true : null, bb: p.b === DEF_B.b ? null : p.b, bp: p.rect ? "rect" : null, bs: p.seed ? p.seed : null });
  }
  function changed() { if (tour) { tour = null; tours.moved(); } writeURL(); tp.reset(); tp.play(); }
  function applyTour(t) { tour = t; p = { ...t.p }; syncControls(); writeURL(); if (t.speed) tp.setSpeed(t.speed); tp.reset(); tp.play(); }
  ctx.onParams(params => {
    if (![...params.keys()].some(k => k === "tb" || Object.values(KEYS_B).includes(k))) return;
    const r = readURL();
    if (r.tour) { tours.set(r.tour.id); applyTour(r.tour); return; }
    tour = null; tours.clear(); p = r.p; syncControls(); tp.reset(); tp.play();
  });
  tp.record(restart());
  // a tour in the URL on first load starts the run (the figure otherwise waits for play)
  if (tour) tp.play();
  return { tp };
}
const subs = n => String(n).split("").map(d => "₀₁₂₃₄₅₆₇₈₉"[+d]).join("");

// ---------------------------------------------------------------- the sweep's table
export function plasticTable(data, el) {
  const tags = [...new Set(data.b.map(r => r.tag))];
  const desc = tag => {
    const r = data.b.find(x => x.tag === tag);
    return { aA: r.alphaA, lat: r.lateral ? `on, b = ${r.b}` : "off", pot: tag.startsWith("rect") ? "rectifying" : "linear" };
  };
  const order = tags.slice().sort((a, b) => {
    const A = desc(a), Bq = desc(b);
    return (A.pot === "rectifying") - (Bq.pot === "rectifying") || (A.lat !== "off") - (Bq.lat !== "off") || A.aA - Bq.aA || a.localeCompare(b);
  });
  const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
  const rows = order.map(tag => {
    const rs = data.b.filter(r => r.tag === tag), d = desc(tag);
    const maj = [0, 1, 2, 3].map(k => mean(rs.map(r => r.alloc_majority[k])));
    const majRange = [0, 1, 2, 3].map(k => [Math.min(...rs.map(r => r.alloc_majority[k])), Math.max(...rs.map(r => r.alloc_majority[k]))]);
    return { tag, ...d, maj, majRange, quiet: mean(rs.map(r => r.mass_per_block[3])), sup: mean(rs.map(r => r.support)), conv: rs.filter(r => r.converged).length, n: rs.length };
  });
  const head = `<tr><th class="num">$\\alpha_A$</th><th>lateral</th><th>potential</th>${M.VARS_B.map((v, k) => `<th class="num"><span class="swatch" style="--sw:${BLOCK_COL[k]}"></span> var ${v}</th>`).join("")}<th class="num">quiet-block share</th><th class="num">support</th><th class="num">converged</th></tr>`;
  const body = rows.map(r => `<tr><td class="num">${r.aA}</td><td>${r.lat}</td><td>${r.pot}</td>${r.maj.map((m, k) => `<td class="num" title="range over seeds ${r.majRange[k][0]}–${r.majRange[k][1]}">${ui.fmtFixed(m, 1)}</td>`).join("")}<td class="num">${ui.fmtFixed(r.quiet, 3)}</td><td class="num">${ui.fmtFixed(r.sup, 1)}</td><td class="num">${r.conv}/${r.n}</td></tr>`).join("");
  el.innerHTML = `<div class="u-help" style="margin-bottom:6px">Columns "var …": neurons (of ${data.meta.config.b.N}) with more than half their row on that block, mean over seeds (hover for the range).</div><div class="table-scroll"><table class="data"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
  ui.renderMath(el);
  return rows;
}
