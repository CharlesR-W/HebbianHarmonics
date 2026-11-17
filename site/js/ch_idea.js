// Chapter "idea": the landing chapter. Two small live figures:
//  (1) a hand-made resistor network whose conductances can be dragged; the selected neuron's harmonic embedding
//      h_i (row of H = L_II^{-1} W_IB, computed with the engine) is shown as a bar chart;
//  (2) "embeddings -> covariance -> weight change" for the synapse selected in (1).
import * as ui from "./ui.js";
import { barChart, svgEl } from "./charts.js";
import { P, boundaryColor, mixBary, colorOf } from "./palette.js";
import * as E from "./engine.js";

const B = 3, SUB = "₁₂₃";
const INPUTS = [{ x: 40, y: 48 }, { x: 40, y: 150 }, { x: 40, y: 252 }];
const NEURONS = [
  { id: "A", x: 168, y: 56 }, { id: "B", x: 292, y: 104 }, { id: "C", x: 238, y: 222 },
  { id: "D", x: 138, y: 168 }, { id: "E", x: 360, y: 250 },
];
const N = NEURONS.length;
// synapses: [kind, a, b, w0]; "ib" = interior a to input b, "ii" = interior a to interior b
const EDGES = [
  ["ib", 0, 0, 2.5], ["ib", 1, 0, 0.5], ["ib", 3, 1, 2.0], ["ib", 2, 1, 0.4], ["ib", 4, 2, 2.0], ["ib", 2, 2, 0.7],
  ["ii", 0, 1, 1.5], ["ii", 0, 3, 1.0], ["ii", 1, 4, 1.2], ["ii", 3, 2, 0.8], ["ii", 2, 4, 0.6], ["ii", 1, 2, 0.5],
];
const W_LO = 0.05, W_HI = 5;
const SIGMA = [1, 0.6, 0, 0.6, 1, 0, 0, 0, 1];   // unit variances, g1-g2 correlated 0.6, g3 independent
const WMIN = 1e-4;
const clampW = w => Math.min(W_HI, Math.max(W_LO, w));
const f2 = x => ui.fmtFixed(x, 2);
const gName = b => `g${SUB[b]}`;

export async function init(panel, ctx) {
  const bcol = [0, 1, 2].map(b => boundaryColor(b, B));
  // ------------------------------------------------------------ state (URL: nrn, syn, theta, w)
  const w = EDGES.map(e => e[3]);
  const wUrl = ctx.param("w", "");
  if (wUrl) wUrl.split("_").forEach((s, k) => { const x = parseFloat(s); if (k < w.length && Number.isFinite(x)) w[k] = clampW(x); });
  let sel = Math.max(0, Math.min(N - 1, "ABCDE".indexOf(ctx.param("nrn", "B")))); if (sel < 0) sel = 1;
  let syn = Math.max(0, Math.min(EDGES.length - 1, ctx.param("syn", 6)));
  let theta = ctx.param("theta", 0.4);
  let H = null;

  function compute() {
    const Wii = new Float64Array(N * N), Wib = new Float64Array(N * B);
    EDGES.forEach(([kind, a, b], k) => {
      if (kind === "ib") Wib[a * B + b] = w[k];
      else { Wii[a * N + b] = w[k]; Wii[b * N + a] = w[k]; }
    });
    H = E.harmonicFrom(Wii, Wib, N, B);
  }
  const hRow = i => [H[i * B], H[i * B + 1], H[i * B + 2]];

  // ------------------------------------------------------------ figure 1: network + bars
  const host = ctx.slot("network"); host.replaceChildren();
  const grid = ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "14px 36px", alignItems: "center", justifyContent: "center" } });
  host.append(grid);
  const svg = svgEl("svg", { viewBox: "0 0 400 290", role: "img", "aria-label": "A resistor network: three clamped inputs on the left, five interior neurons joined by synapses whose line width shows conductance.", style: "flex:1 1 360px;max-width:540px;min-width:0;height:auto;display:block;font-family:var(--sans);touch-action:pan-y" });
  grid.append(svg);
  const side = ui.h("div", { style: { flex: "1 1 260px", maxWidth: "360px", minWidth: "0" } });
  grid.append(side);

  const gEdges = svgEl("g", {}, svg), gNodes = svgEl("g", {}, svg), gLabels = svgEl("g", {}, svg);
  const endXY = (kind, a, b) => [NEURONS[a], kind === "ib" ? INPUTS[b] : NEURONS[b]];
  const edgeEls = EDGES.map(([kind, a, b], k) => {
    const [p, q] = endXY(kind, a, b);
    const g = svgEl("g", { style: "cursor:ns-resize" }, gEdges);
    const halo = svgEl("line", { x1: p.x, y1: p.y, x2: q.x, y2: q.y, stroke: P.line, "stroke-linecap": "round", opacity: 0 }, g);
    const line = svgEl("line", { x1: p.x, y1: p.y, x2: q.x, y2: q.y, stroke: P.edge, "stroke-linecap": "round", opacity: 0.72 }, g);
    const hit = svgEl("line", { x1: p.x, y1: p.y, x2: q.x, y2: q.y, stroke: "transparent", "stroke-width": 16, "pointer-events": "stroke", style: "touch-action:none" }, g);
    attachDrag(hit, k);
    hit.addEventListener("pointermove", ev => { if (!drag) ui.showTip(ev.clientX, ev.clientY, edgeTip(k)); });
    hit.addEventListener("pointerleave", () => ui.hideTip());
    return { halo, line, mid: [(p.x + q.x) / 2, (p.y + q.y) / 2] };
  });
  const wLabel = svgEl("text", { "font-size": 11.5, fill: P.ink2, "text-anchor": "middle", "pointer-events": "none", "paint-order": "stroke", stroke: P.surface, "stroke-width": 4 }, gLabels);

  INPUTS.forEach((p, b) => {
    svgEl("rect", { x: p.x - 10, y: p.y - 10, width: 20, height: 20, fill: bcol[b], stroke: P.edge, "stroke-width": 1.5 }, gNodes);
    const t = svgEl("text", { x: p.x - 16, y: p.y + 5, "font-size": 14, "font-weight": 600, fill: P.fg, "text-anchor": "end" }, gNodes);
    t.textContent = gName(b);
  });
  const nodeEls = NEURONS.map((n, i) => {
    const g = svgEl("g", { style: "cursor:pointer", tabindex: 0, role: "button", "aria-label": `neuron ${n.id}` }, gNodes);
    const ring = svgEl("circle", { cx: n.x, cy: n.y, r: 17, fill: "none", stroke: P.fg, "stroke-width": 2, opacity: 0 }, g);
    const c = svgEl("circle", { cx: n.x, cy: n.y, r: 12.5, fill: P.interior, stroke: P.surface, "stroke-width": 2 }, g);
    const t = svgEl("text", { x: n.x, y: n.y + 4.3, "font-size": 12, "font-weight": 650, fill: "#fff", "text-anchor": "middle", "pointer-events": "none" }, g);
    t.textContent = n.id;
    const pick = () => { sel = i; redraw(); writeUrl(); };
    g.addEventListener("click", pick);
    g.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(); } });
    g.addEventListener("pointermove", ev => ui.showTip(ev.clientX, ev.clientY, nodeTip(i)));
    g.addEventListener("pointerleave", () => ui.hideTip());
    return { ring, c };
  });
  const note = svgEl("text", { x: 200, y: 285, "font-size": 11, fill: P.mute, "text-anchor": "middle" }, svg);
  note.textContent = "line width ∝ √conductance · drag a synapse to change it";

  const barTitle = ui.h("div", { class: "chart-title" });
  const barBox = ui.h("div");
  const hText = ui.h("div", { class: "u-help", style: { margin: "2px 0 14px", fontSize: "12.5px" } });
  side.append(barTitle, barBox, hText);
  const bars = barChart(barBox, { labels: [0, 1, 2].map(gName), values: [0, 0, 0], colors: bcol, yrange: [0, 1], height: 150,
    ylabel: "share of voltage", yfmt: x => ui.fmtFixed(x, 1), tooltip: i => ui.tipRows(`input ${gName(i)}`, [{ name: `share of ${NEURONS[sel].id}'s voltage`, value: f2(H[sel * B + i]), color: bcol[i], shape: "square" }]) });
  const synSlider = ui.slider({ label: "", min: W_LO, max: W_HI, log: true, sig: 2, value: w[syn], fmt: x => ui.fmt(x, 2),
    onInput: x => { w[syn] = x; redraw(); }, onChange: () => writeUrl() });
  const resetBtn = ui.h("button", { type: "button", class: "explore-link", style: { cursor: "pointer", marginTop: "10px", fontSize: "13px" },
    onclick: () => { EDGES.forEach((e, k) => { w[k] = e[3]; }); redraw(); writeUrl(); } }, "reset conductances");
  side.append(synSlider.el, resetBtn);

  // ------------------------------------------------------------ figure 2: embeddings -> covariance -> weight change
  const dHost = ctx.slot("drift"); dHost.replaceChildren();
  const thetaS = ui.slider({ label: "pruning threshold $\\theta$", min: 0, max: 1, step: 0.01, value: theta, fmt: x => ui.fmtFixed(x, 2),
    marks: [{ value: 0, label: "0" }, { value: 0.6, label: "ρ₁₂" }, { value: 1, label: "1" }],
    onInput: x => { theta = x; drawDrift(); }, onChange: () => writeUrl() });
  const thetaWrap = ui.h("div", { style: { maxWidth: "340px", marginBottom: "12px" } }, thetaS.el);
  const chain = ui.h("div", { style: { display: "flex", flexWrap: "wrap", alignItems: "center", gap: "10px 14px", font: "14px/1.4 var(--sans)" } });
  const verdict = ui.h("div", { style: { marginTop: "12px", font: "14.5px/1.5 var(--sans)", color: P.fg } });
  dHost.append(thetaWrap, chain, verdict);

  // ------------------------------------------------------------ drawing
  function redraw() {
    compute();
    const wmax = Math.max(...w);
    EDGES.forEach((e, k) => {
      const el = edgeEls[k], width = 1 + 3.2 * Math.sqrt(w[k] / wmax) * Math.sqrt(wmax / 2.5);
      el.line.setAttribute("stroke-width", width.toFixed(2));
      el.halo.setAttribute("stroke-width", (width + 9).toFixed(2));
      el.halo.setAttribute("opacity", k === syn ? 1 : 0);
    });
    const [mx, my] = edgeEls[syn].mid;
    wLabel.setAttribute("x", mx); wLabel.setAttribute("y", my - 9);
    wLabel.textContent = `w = ${ui.fmt(w[syn], 2)}`;
    NEURONS.forEach((n, i) => {
      nodeEls[i].c.setAttribute("fill", mixBary(hRow(i), bcol));
      nodeEls[i].ring.setAttribute("opacity", i === sel ? 1 : 0);
    });
    const h = hRow(sel), id = NEURONS[sel].id;
    barTitle.innerHTML = `Harmonic embedding of neuron ${id}, $h_{${id}}$`;
    ui.renderMath(barTitle);
    bars.update({ values: h });
    hText.innerHTML = `$h_{${id}} = (${h.map(f2).join(",\\ ")})$, sums to 1. Its voltage is $u_{${id}} = h_{${id}}^\\top g$.`;
    ui.renderMath(hText);
    synSlider.set(w[syn]);
    synSlider.el.querySelector(".u-name").innerHTML = `conductance of synapse ${edgeName(syn)}`;
    drawDrift();
  }

  function edgeName(k) {
    const [kind, a, b] = EDGES[k];
    return `${NEURONS[a].id}–${kind === "ib" ? gName(b) : NEURONS[b].id}`;
  }
  function edgeTip(k) {
    return ui.tipRows(`synapse ${edgeName(k)}`, [
      { name: "conductance w", value: ui.fmt(w[k], 3) },
      { name: "starting value", value: ui.fmt(EDGES[k][3], 3) },
    ], "drag up to strengthen, down to weaken");
  }
  function nodeTip(i) {
    const h = hRow(i);
    return ui.tipRows(`neuron ${NEURONS[i].id}`, h.map((x, b) => ({ name: `h to ${gName(b)}`, value: f2(x), color: bcol[b], shape: "square" })), "click to show as bars");
  }

  // drag a synapse vertically: up = stronger (log scale)
  let drag = null;
  function attachDrag(el, k) {
    el.addEventListener("pointerdown", ev => {
      ev.preventDefault(); ui.hideTip();
      drag = { k, y: ev.clientY, w: w[k], id: ev.pointerId };
      el.setPointerCapture(ev.pointerId);
      syn = k; redraw();
    });
    el.addEventListener("pointermove", ev => {
      if (!drag || drag.id !== ev.pointerId) return;
      w[k] = clampW(+(drag.w * Math.exp((drag.y - ev.clientY) / 45)).toPrecision(3));
      redraw();
    });
    const end = ev => { if (drag && drag.id === ev.pointerId) { drag = null; writeUrl(); } };
    el.addEventListener("pointerup", end); el.addEventListener("pointercancel", end);
  }

  // small SVG glyphs for figure 2
  function miniBars(h, base, sub) {
    const W = 74, Hh = 70, s = svgEl("svg", { width: W, height: Hh + 18, viewBox: `0 0 ${W} ${Hh + 18}` });
    svgEl("line", { x1: 2, x2: W - 2, y1: Hh, y2: Hh, stroke: P.mute2, "stroke-width": 1 }, s);
    h.forEach((x, b) => {
      const bh = Math.max(0.5, x * (Hh - 14)), bx = 6 + b * 22;
      svgEl("rect", { x: bx, y: Hh - bh, width: 18, height: bh, fill: bcol[b], rx: 1.5 }, s);
      const t = svgEl("text", { x: bx + 9, y: Hh - bh - 3, "font-size": 10, fill: P.ink2, "text-anchor": "middle" }, s); t.textContent = x < 0.005 ? "0" : f2(x).replace(/^0/, "");
    });
    const t = svgEl("text", { x: W / 2, y: Hh + 14, "font-size": 13, "font-style": "italic", fill: P.fg, "text-anchor": "middle" }, s);
    t.append(base); const ts = svgEl("tspan", { "font-size": 9.5, dy: 3, "font-style": "normal" }, t); ts.textContent = sub;
    return s;
  }
  function miniMatrix() {
    const c = 26, s = svgEl("svg", { width: 3 * c + 4, height: 3 * c + 22, viewBox: `0 0 ${3 * c + 4} ${3 * c + 22}` });
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      const v = SIGMA[i * 3 + j];
      svgEl("rect", { x: 2 + j * c, y: 2 + i * c, width: c - 1, height: c - 1, fill: colorOf("div", v, -1, 1) }, s);
      const t = svgEl("text", { x: 2 + j * c + c / 2, y: 2 + i * c + c / 2 + 4, "font-size": 10.5, fill: Math.abs(v) > 0.7 ? "#fff" : P.fg, "text-anchor": "middle" }, s);
      t.textContent = v === 0 ? "0" : String(v).replace(/^0/, "");
    }
    const t = svgEl("text", { x: (3 * c + 4) / 2, y: 3 * c + 18, "font-size": 13, "font-style": "italic", fill: P.fg, "text-anchor": "middle" }, s);
    t.append("Σ"); const ts = svgEl("tspan", { "font-size": 9.5, dy: 3 }, t); ts.textContent = "g";
    return s;
  }
  const op = (txt, big) => ui.h("span", { style: { fontSize: big ? "22px" : "15px", color: P.mute, lineHeight: "1" }, html: txt });

  function drawDrift() {
    const [kind, a, b] = EDGES[syn];
    const hi = hRow(a), hj = kind === "ib" ? [0, 1, 2].map(c => (c === b ? 1 : 0)) : hRow(b);
    const li = NEURONS[a].id, lj = kind === "ib" ? gName(b) : NEURONS[b].id;
    let cov = 0;
    for (let x = 0; x < 3; x++) for (let y = 0; y < 3; y++) cov += hi[x] * SIGMA[x * 3 + y] * hj[y];
    const gamma = 1, thr = kind === "ib" ? theta : theta / gamma;
    const target = Math.max(WMIN, gamma * cov - theta);
    const grows = gamma * cov - theta > WMIN;
    const rhsTex = kind === "ib" ? `h_{${li}}^\\top\\Sigma_g e_{${b + 1}}` : `h_{${li}}^\\top\\Sigma_g h_{${lj}}`;
    chain.replaceChildren(
      miniBars(hi, "h", li), op("ᵀ"),
      miniMatrix(), kind === "ib" ? miniBars(hj, "e", String(b + 1)) : miniBars(hj, "h", lj),
      op("=", true),
      ui.h("div", { style: { textAlign: "center" } },
        ui.h("div", { style: { font: "600 22px/1.2 var(--sans)", color: P.fg }, text: ui.fmtFixed(cov, 3) }),
        ui.h("div", { class: "u-help", html: `$${rhsTex}$` })),
      op(grows ? "&gt;" : "≤", true),
      ui.h("div", { style: { textAlign: "center" } },
        ui.h("div", { style: { font: "600 22px/1.2 var(--sans)", color: P.fg }, text: ui.fmtFixed(thr, 2) }),
        ui.h("div", { class: "u-help", html: kind === "ib" ? "threshold $\\theta$" : "threshold $\\theta/\\gamma$" })),
    );
    verdict.innerHTML = grows
      ? `<b>Synapse ${edgeName(syn)} grows</b> towards $w^\\ast = \\tfrac{\\eta}{\\lambda}(${kind === "ib" ? "" : "\\gamma "}h^\\top\\Sigma_g h - \\theta) = ${ui.fmtFixed(target, 3)}$: the two ends covary by more than the threshold.`
      : `<b>Synapse ${edgeName(syn)} is pruned</b> to the floor $w_{\\min}$: the two ends do not covary by more than the threshold.`;
    ui.renderMath(chain); ui.renderMath(verdict);
  }

  function writeUrl() {
    const edited = w.some((x, k) => Math.abs(x - EDGES[k][3]) > 1e-9);
    ctx.setParams({ nrn: NEURONS[sel].id, syn, theta, w: edited ? w.map(x => +x.toPrecision(3)).join("_") : null });
  }

  ctx.onParams(p => {
    const n = "ABCDE".indexOf(p.get("nrn") || "B"); if (n >= 0) sel = n;
    const s = parseInt(p.get("syn"), 10); if (s >= 0 && s < EDGES.length) syn = s;
    const th = parseFloat(p.get("theta")); if (Number.isFinite(th)) { theta = th; thetaS.set(th); }
    const ws = p.get("w");
    EDGES.forEach((e, k) => { w[k] = e[3]; });
    if (ws) ws.split("_").forEach((x, k) => { const v = parseFloat(x); if (k < w.length && Number.isFinite(v)) w[k] = clampW(v); });
    redraw();
  });

  redraw();
}
