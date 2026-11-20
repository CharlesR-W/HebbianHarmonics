// Chapter "basics": six figures, linear neurons only.
//  1 figNet     the resistor network alive: inputs jump, neurons relax (neuron space) + voltage traces
//  2 figEmbed   each neuron is a mixture: pies in the wiring diagram and the triangle, linked hover
//  3 figPull    one synapse pulls its two neurons together along a straight line in the triangle
//  4 figCov     covariance of two neurons = h_i' Sigma h_j: draggable mixtures + voltage scatter
//  5 figLearn   the covariance rule (engine.js, the perron chapter's parameters) on 60 neurons: triangle and
//               lateral weight matrix (sorted by final group) over training, plus a six-frame film strip
//  6 figEnergy  the distance rule: J(W) falls monotonically; mixtures before / after
// Model code in ch_basics_model.js, drawing helpers in ch_basics_draw.js.
import * as ui from "./ui.js";
import { svgEl, scatter, heatmap } from "./charts.js";
import { P, colorOf, mixBary } from "./palette.js";
import * as E from "./engine.js";
import * as M from "./ch_basics_model.js";
import * as D from "./ch_basics_draw.js";

const B = 3, SUB = D.SUBS;
const INC = [0, 1, 2].map(b => P.cat[b]);
const bary = h => mixBary(h, INC);
const pct = x => `${Math.round(100 * x)}%`;
const f2 = x => ui.fmtFixed(x, 2);

export async function init(panel, ctx) {
  figNet(ctx);
  figEmbed(ctx);
  figPull(ctx);
  figCov(ctx);
  const learn = learnShared(ctx);                  // run state (URL keys theta, s, seed) and the run itself
  figLearn(ctx, learn, { slot: "learn", details: "learn-details" });
  learn.set({});
  figEnergy(ctx);
}

// a mixture written out: "0.50 g1 + 0.20 g2 + 0.30 g3" (HTML, inputs coloured)
const mixHtml = h => h.map((v, b) => `${f2(v)}&thinsp;<b style="color:${INC[b]}">g${SUB[b]}</b>`).join(" + ");

// ============================================================ 1. the network, alive
function figNet(ctx) {
  const slot = ctx.slot("net");
  const two = ui.h("div", { class: "bx-two" }); slot.append(two);
  const left = D.panel(two, "neuron", "neuron space");
  const right = ui.h("div", { class: "bx-panel" }); two.append(right);
  const svg = D.svgRoot(left, ...D.NET_VB, "Resistor network: three input squares and ten neuron circles coloured by voltage");
  const L = D.netLayout(), VR = 2.5, vcol = v => colorOf("div", v, -VR, VR);
  D.drawEdges(svgEl("g", {}, svg), L, { opacity: 0.32 });
  const sq = D.drawInputs(svg, L, { fill: () => vcol(0) });
  let sel = ctx.param("nn", 2) - 1; if (!(sel >= 0 && sel < M.NS)) sel = 1;
  const circ = [], rings = [];
  L.int.forEach(([x, y], i) => {
    const g = svgEl("g", { style: "cursor:pointer" }, svg);
    rings.push(svgEl("circle", { cx: x, cy: y, r: 22, fill: "none", stroke: P.fg, "stroke-width": 2.5, opacity: 0 }, g));
    circ.push(svgEl("circle", { cx: x, cy: y, r: 16, fill: vcol(0), stroke: P.fg, "stroke-width": 1.2 }, g));
    D.txt(g, x, y + 5, String(i + 1), { "text-anchor": "middle", "font-size": 14, "font-weight": 600, style: "font-family:var(--sans);pointer-events:none" });
    svgEl("circle", { cx: x, cy: y, r: 24, fill: "transparent" }, g);
    g.addEventListener("pointerenter", () => select(i));
    g.addEventListener("click", () => { select(i); ctx.setParams({ nn: i + 1 }); });
  });
  ui.colorbar({ parent: left, map: "div", range: [-VR, VR], label: "voltage", ticks: [-2, -1, 0, 1, 2], width: 260 });

  const trace = D.traceChart(right, {
    span: 10, yr: [-VR, VR], height: 330,
    series: [...INC.map(c => ({ color: c, width: 2, opacity: 0.85 })), { color: P.fg, width: 3.4 }],
  });
  const leg = ui.legend({ parent: right, items: [] });
  const btn = ui.h("button", { type: "button", class: "bx-btn", text: "Pause" });
  right.append(ui.h("div", { class: "bx-row" }, btn));

  function select(i) {
    sel = i;
    rings.forEach((r, k) => r.setAttribute("opacity", k === i ? 1 : 0));
    leg.set([...[0, 1, 2].map(b => ({ label: `input g${SUB[b]}`, color: INC[b], shape: "line" })), { label: `<b>neuron ${i + 1}</b>`, color: P.fg, shape: "line" }]);
    trace.clear(); for (const [t, g, u] of hist) trace.push(t, [g[0], g[1], g[2], u[i]]); trace.redraw();
  }

  // dynamics: inputs jump every HOLD seconds; tau du/dt = -(L u - A g); slowest mode decays with e-folding 0.3 s
  const HOLD = 2.5, rate = 1 / (0.3 * M.slowestRate()), relax = M.makeRelaxer(rate);
  const rnd = M.rng(7), draw = () => [0, 1, 2].map(() => Math.max(-2.4, Math.min(2.4, rnd.normal())));
  let t = 0, nextJump = 0, g = [0, 0, 0];
  const u = new Float64Array(M.NS), hist = [];
  function advance(dt) {
    if (t >= nextJump) { g = draw(); nextJump += HOLD; }
    relax(u, g, dt); t += dt;
    hist.push([t, g.slice(), Float64Array.from(u)]);
    while (hist.length > 2 && hist[1][0] < t - 10) hist.shift();
    trace.push(t, [g[0], g[1], g[2], u[sel]]);
  }
  function paint() {
    g.forEach((v, b) => sq[b].setAttribute("fill", vcol(v)));
    u.forEach((v, i) => circ[i].setAttribute("fill", vcol(v)));
    trace.redraw();
  }
  for (let k = 0; k < 600; k++) advance(1 / 60);       // ten seconds of history, so the figure is never empty
  select(sel); paint();
  ctx.onParams(q => { const k = Math.round(+q.get("nn")) - 1; if (k >= 0 && k < M.NS && k !== sel) select(k); });
  let playing = !ui.reducedMotion();
  const setBtn = () => { btn.textContent = playing ? "Pause" : "Play"; btn.setAttribute("aria-pressed", String(playing)); };
  btn.addEventListener("click", () => { playing = !playing; setBtn(); });
  setBtn();
  ui.animate(slot, dt => { if (!playing || !ctx.visible) return; advance(dt); paint(); });
}

// ============================================================ 2. neuron space and input space
function figEmbed(ctx) {
  const slot = ctx.slot("embed");
  const H = M.smallH(), hOf = i => M.rowOf(H, i);
  const pair = ui.h("div", { class: "space-pair" }); slot.append(pair);
  const left = D.panel(pair, "neuron", "neuron space"), right = D.panel(pair, "input", "input space");
  const sN = D.svgRoot(left, ...D.NET_VB, "The network, each neuron drawn as a pie of the three input colours");
  const sT = D.svgRoot(right, ...D.NET_VB, "The same neurons as points in the triangle of inputs");
  const L = D.netLayout();
  D.drawEdges(svgEl("g", {}, sN), L, { opacity: 0.28 });
  D.drawInputs(sN, L);
  const T = D.triangle(sT, ...D.NET_VB, { centre: true, top: 20 });
  const ringN = svgEl("circle", { r: 24, fill: "none", stroke: P.fg, "stroke-width": 2.5, opacity: 0 }, sN);
  const ringT = svgEl("circle", { r: 18, fill: "none", stroke: P.fg, "stroke-width": 2.5, opacity: 0 }, sT);
  const read = ui.h("div", { class: "bx-readout" }); slot.append(read);
  const nodes = svgEl("g", {}, sN), dots = svgEl("g", {}, sT);
  function select(i) {
    const [x, y] = L.int[i], [tx, ty] = T.at(hOf(i));
    ringN.setAttribute("cx", x); ringN.setAttribute("cy", y); ringN.setAttribute("opacity", 1);
    ringT.setAttribute("cx", tx); ringT.setAttribute("cy", ty); ringT.setAttribute("opacity", 1);
    read.innerHTML = `<b>Neuron ${i + 1}</b> always sits at u<sub>${i + 1}</sub> = ${mixHtml(hOf(i))}`;
  }
  const bind = (g, i) => { g.addEventListener("pointerenter", () => select(i)); g.addEventListener("click", () => select(i)); };
  L.int.forEach(([x, y], i) => {
    const g = svgEl("g", { style: "cursor:pointer" }, nodes);
    D.pie(g, x, y, 17, hOf(i));
    D.txt(g, x, y - 22, String(i + 1), { "text-anchor": "middle", "font-size": 13, fill: P.ink2 });
    svgEl("circle", { cx: x, cy: y, r: 22, fill: "transparent" }, g);
    bind(g, i);
  });
  for (let i = 0; i < M.NS; i++) {
    const [x, y] = T.at(hOf(i)), g = svgEl("g", { style: "cursor:pointer" }, dots);
    D.pie(g, x, y, 12, hOf(i));
    D.txt(g, x + 15, y - 7, String(i + 1), { "font-size": 13, fill: P.ink2 });
    svgEl("circle", { cx: x, cy: y, r: 15, fill: "transparent" }, g);
    bind(g, i);
  }
  select(1);
}

// ============================================================ 3. one synapse pulls two neurons together
function figPull(ctx) {
  const slot = ctx.slot("pull");
  let pi = ctx.param("pi", 7) - 1, pj = ctx.param("pj", 8) - 1, w = Math.max(0, Math.min(10, ctx.param("pw", 2)));
  if (!(pi >= 0 && pi < M.NS && pj >= 0 && pj < M.NS && pi !== pj)) { pi = 6; pj = 7; }
  const H0 = M.smallH();                                   // the network as it is (ghosts)
  const pairEl = ui.h("div", { class: "space-pair" }); slot.append(pairEl);
  const left = D.panel(pairEl, "neuron", "neuron space"), right = D.panel(pairEl, "input", "input space");
  const sN = D.svgRoot(left, ...D.NET_VB, "The network with one synapse highlighted"), sT = D.svgRoot(right, ...D.NET_VB, "The triangle, with the paths of the two chosen neurons");
  const L = D.netLayout();
  const gEdges = svgEl("g", {}, sN), gSyn = svgEl("g", {}, sN);
  D.drawInputs(sN, L);
  const gNodes = svgEl("g", {}, sN);
  const T = D.triangle(sT, ...D.NET_VB, { top: 20 });
  const gGhost = svgEl("g", {}, sT), gPath = svgEl("g", {}, sT), gDots = svgEl("g", {}, sT);
  const ctl = ui.h("div", { class: "bx-controls" }); slot.append(ctl);
  const pairLab = ui.h("div", { class: "bx-readout" });
  const sl = ui.slider({ label: "conductance $w$ of their synapse", min: 0, max: 10, step: 0.05, value: w, fmt: x => ui.fmtFixed(x, 2),
    onInput: x => { w = x; draw(); ctx.setParams({ pw: x }); } });
  ctl.append(sl.el, pairLab);
  const pick = k => {
    if (k === pi || k === pj) return;
    pi = pj; pj = k; draw(); ctx.setParams({ pi: pi + 1, pj: pj + 1 });
  };

  function draw() {
    const Hw = M.smallH({ i: pi, j: pj, w }), Hz = M.smallH({ i: pi, j: pj, w: 0 }), Hinf = M.smallH({ i: pi, j: pj, w: 1e7 });
    const R = M.effRes(pi, pj, 0), shrink = 1 / (1 + w * R);
    const hv = (Hm, i) => M.rowOf(Hm, i);
    // neuron space: the other synapses, then the chosen one in accent (dashed when cut)
    gEdges.replaceChildren(); gSyn.replaceChildren(); gNodes.replaceChildren();
    M.EDGES.forEach(([a, b, c]) => {
      if ((a === pi && b === pj) || (a === pj && b === pi)) return;
      const [x1, y1] = D.posOf(L, a), [x2, y2] = D.posOf(L, b);
      svgEl("line", { x1, y1, x2, y2, stroke: P.edge, "stroke-width": Math.min(9, 4 * c), "stroke-linecap": "round", opacity: 0.28 }, gEdges);
    });
    const [x1, y1] = L.int[pi], [x2, y2] = L.int[pj];
    svgEl("line", { x1, y1, x2, y2, stroke: P.accent, "stroke-width": w > 0 ? Math.min(16, 1.5 + 4 * w) : 2, "stroke-linecap": "round",
      "stroke-dasharray": w > 0 ? null : "6 6", opacity: w > 0 ? 0.85 : 0.9 }, gSyn);
    L.int.forEach(([x, y], i) => {
      const g = svgEl("g", { style: "cursor:pointer" }, gNodes), on = i === pi || i === pj;
      if (on) svgEl("circle", { cx: x, cy: y, r: 23, fill: "none", stroke: P.accent, "stroke-width": 3 }, g);
      D.pie(g, x, y, 17, hv(Hw, i));
      D.txt(g, x, y - 23 - (on ? 4 : 0), String(i + 1), { "text-anchor": "middle", "font-size": 13, fill: on ? P.fg : P.ink2, "font-weight": on ? 700 : 400 });
      svgEl("circle", { cx: x, cy: y, r: 22, fill: "transparent" }, g);
      g.addEventListener("click", () => pick(i));
    });
    // input space: ghosts of the original network, the two straight paths, the meeting point, current pies
    gGhost.replaceChildren(); gPath.replaceChildren(); gDots.replaceChildren();
    for (let i = 0; i < M.NS; i++) { const [x, y] = T.at(hv(H0, i)); svgEl("circle", { cx: x, cy: y, r: 5, fill: "none", stroke: P.mute2, "stroke-width": 1.2 }, gGhost); }
    const [mx, my] = T.at(hv(Hinf, pi));
    for (const i of [pi, pj]) {
      const [ax, ay] = T.at(hv(Hz, i));
      svgEl("line", { x1: ax, y1: ay, x2: mx, y2: my, stroke: P.accent, "stroke-width": 2, "stroke-dasharray": "5 5" }, gPath);
      svgEl("circle", { cx: ax, cy: ay, r: 3.5, fill: P.accent }, gPath);
    }
    svgEl("path", { d: `M${mx - 6},${my - 6}L${mx + 6},${my + 6}M${mx - 6},${my + 6}L${mx + 6},${my - 6}`, stroke: P.accent, "stroke-width": 2.5 }, gPath);
    for (let i = 0; i < M.NS; i++) {
      const on = i === pi || i === pj, [x, y] = T.at(hv(Hw, i)), g = svgEl("g", { style: "cursor:pointer" }, gDots);
      if (on) svgEl("circle", { cx: x, cy: y, r: 17, fill: "none", stroke: P.accent, "stroke-width": 3 }, g);
      D.pie(g, x, y, on ? 12 : 8, hv(Hw, i), { sw: on ? 1.2 : 0.8 });
      D.txt(g, x + (on ? 19 : 11), y - (on ? 9 : 6), String(i + 1), { "font-size": on ? 14 : 12, fill: on ? P.fg : P.mute, "font-weight": on ? 700 : 400 });
      svgEl("circle", { cx: x, cy: y, r: 14, fill: "transparent" }, g);
      g.addEventListener("click", () => pick(i));
    }
    pairLab.innerHTML = `Neurons <b>${pi + 1}</b> and <b>${pj + 1}</b>${M.lateralW(pi, pj) > 0 ? ` (wired with $w = ${M.lateralW(pi, pj)}$ in the original network)` : " (not wired in the original network)"}: ` +
      (w > 0 ? `the gap between their mixtures is <b>${pct(shrink)}</b> of what it is with the synapse cut.` : "synapse cut; the dots on the dashed lines mark this state.");
    ui.renderMath(pairLab);
  }
  draw();
  ctx.onParams(q => {
    const a = Math.round(+(q.get("pi") ?? pi + 1)) - 1, b = Math.round(+(q.get("pj") ?? pj + 1)) - 1, x = q.get("pw");
    let ch = false;
    if (a >= 0 && a < M.NS && b >= 0 && b < M.NS && a !== b && (a !== pi || b !== pj)) { pi = a; pj = b; ch = true; }
    if (x != null && Number.isFinite(+x)) { const v = Math.max(0, Math.min(10, +x)); if (v !== w) { w = v; sl.set(v); ch = true; } }
    if (ch) draw();
  });
}

// ============================================================ 4. firing together = overlap
const COV_PRESETS = [
  { value: "ind", label: "independent", S: M.sigma3(0, 0, 0) },
  { value: "pos", label: "g₁, g₂ correlated", S: M.sigma3(0.8, 0, 0) },
  { value: "neg", label: "g₁, g₂ anticorrelated", S: M.sigma3(-0.8, 0, 0) },
];
function figCov(ctx) {
  const slot = ctx.slot("cov"), NSAMP = 2000;
  let cs = ctx.param("cs", "ind"); if (!COV_PRESETS.some(p => p.value === cs)) cs = "ind";
  const hs = [[0.75, 0.15, 0.10], [0.35, 0.50, 0.15]], names = ["i", "j"];
  const two = ui.h("div", { class: "bx-two cov" }); slot.append(two);
  const left = D.panel(two, "input", "input space"), right = ui.h("div", { class: "bx-panel" }); two.append(right);
  const VW = 520, VH = 420;
  const svg = D.svgRoot(left, VW, VH, "Two draggable neurons in the triangle of inputs");
  const T = D.triangle(svg, VW, VH, { top: 26 });
  const gN = svgEl("g", {}, svg);
  const scBox = ui.h("div"); right.append(scBox);
  const read = ui.h("div", { class: "bx-readout" }); right.append(read);
  const seg = ui.segmented({ label: "inputs", options: COV_PRESETS.map(p => ({ value: p.value, label: p.label })), value: cs,
    onChange: v => { cs = v; ctx.setParams({ cs: v }); update(); } });
  slot.append(ui.h("div", { class: "bx-controls" }, seg.el));
  const Z = M.sampleInputs(M.sigma3(), NSAMP, 11);                 // standard normal z; g = L z for each preset
  const sc = scatter(scBox, { x: [], y: [], color: P.ink2, size: 1.8, xrange: [-3.2, 3.2], yrange: [-3.2, 3.2], equal: true, height: 360,
    xlabel: "voltage of neuron i", ylabel: "voltage of neuron j" });
  let G = null;
  function samples() {
    const S = COV_PRESETS.find(p => p.value === cs).S, Lc = M.chol3(S);
    G = new Float64Array(3 * NSAMP);
    for (let s = 0; s < NSAMP; s++) for (let a = 0; a < 3; a++) { let v = 0; for (let k = 0; k <= a; k++) v += Lc[a * 3 + k] * Z[3 * s + k]; G[3 * s + a] = v; }
    return S;
  }
  let S = samples();
  function drawNeurons() {
    gN.replaceChildren();
    hs.forEach((h, k) => {
      const [x, y] = T.at(h), g = svgEl("g", { style: "cursor:grab" }, gN);
      svgEl("circle", { cx: x, cy: y, r: 24, fill: "transparent" }, g);
      D.pie(g, x, y, 17, h, { sw: 1.5 });
      D.txt(g, x + 22, y - 14, `neuron ${names[k]}`, { "font-size": 15, "font-weight": 700 });
      g.addEventListener("pointerdown", ev => startDrag(ev, k));
    });
  }
  let drag = null;
  function startDrag(ev, k) { drag = k; svg.setPointerCapture(ev.pointerId); ev.preventDefault(); }
  svg.addEventListener("pointermove", ev => { if (drag == null) return; const p = D.svgPoint(svg, ev); hs[drag] = T.inv(p.x, p.y); update(); });
  const end = () => { drag = null; };
  svg.addEventListener("pointerup", end); svg.addEventListener("pointercancel", end);
  function update() {
    if (S !== COV_PRESETS.find(p => p.value === cs).S) S = samples();
    drawNeurons();
    const x = new Float64Array(NSAMP), y = new Float64Array(NSAMP);
    let mx = 0, my = 0;
    for (let s = 0; s < NSAMP; s++) {
      let a = 0, c = 0; for (let b = 0; b < 3; b++) { a += hs[0][b] * G[3 * s + b]; c += hs[1][b] * G[3 * s + b]; }
      x[s] = a; y[s] = c; mx += a; my += c;
    }
    mx /= NSAMP; my /= NSAMP;
    let cxy = 0; for (let s = 0; s < NSAMP; s++) cxy += (x[s] - mx) * (y[s] - my);
    cxy /= NSAMP - 1;
    let pred = 0; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) pred += hs[0][a] * S[a * 3 + b] * hs[1][b];
    sc.update({ x: Array.from(x), y: Array.from(y) });
    read.innerHTML = `<div>h<sub>i</sub> = (${hs[0].map(f2).join(", ")}), h<sub>j</sub> = (${hs[1].map(f2).join(", ")})</div>` +
      `<div>covariance measured from the ${ui.fmtInt(NSAMP)} dots: <b>${ui.fmtFixed(cxy, 3)}</b> · h<sub>i</sub><sup>⊤</sup>Σ<sub>g</sub>h<sub>j</sub>: <b>${ui.fmtFixed(pred, 3)}</b></div>`;
  }
  update();
  ctx.onParams(q => { const v = q.get("cs"); if (v && v !== cs && COV_PRESETS.some(x => x.value === v)) { cs = v; seg.set(v); update(); } });
}

// ============================================================ 5. Hebb's rule at work (covariance rule, engine.js)
// Parameters and names as the perron chapter (ch_perron_model.js DEFAULTS): lambda = eta = 1, gamma = 1, w_min = 1e-4,
// lateral weights start at 0 (the floor after one step), input weights ~ Exp(0.2). N = 60 here.
const LEARN = { N: 60, lam: 1, eta: 1, gamma: 1, wmin: 1e-4, ib: 0.2, dt: 0.02, tMax: 40, resTol: 1e-6 };
const LEARN_PRESETS = [
  { value: "indep", label: "independent", S: () => E.cov3({ variances: [1, 1, 1], corr: [0, 0, 0] }) },
  { value: "pair", label: "g₁, g₂ correlated (0.6)", S: () => E.cov3({ variances: [1, 1, 1], corr: [0.6, 0, 0] }) },
];
const FILM_T = [0, 0.2, 0.3, 0.4, 0.6, null];      // null: the end of the run
function runLearn(p) {
  const { N } = LEARN, Sigma = LEARN_PRESETS.find(x => x.value === p.s).S();
  const s = E.createState({ N, B, seed: p.seed, iiScale: 0, ibScale: LEARN.ib });
  const P = { ...E.defaultParams(), theta: p.theta, lam: LEARN.lam, eta: LEARN.eta, gamma: LEARN.gamma, wmin: LEARN.wmin, dt: LEARN.dt };
  const frames = [], nTri = N * (N - 1) / 2;
  const snap = () => {
    const H = Float32Array.from(E.harmonicFrom(s.Wii, s.Wib, N, B));
    const W = new Float32Array(nTri + N * B); let k = 0;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) W[k++] = s.Wii[i * N + j];
    for (let q = 0; q < N * B; q++) W[k++] = s.Wib[q];
    frames.push({ t: s.t * LEARN.dt, H, W });
  };
  snap();
  const steps = Math.round(LEARN.tMax / LEARN.dt);
  let k = 0, resid = NaN;
  for (k = 1; k <= steps; k++) {
    E.step(s, Sigma, P);
    const t = k * LEARN.dt;
    const every = t <= 2 ? 1 : t <= 6 ? 5 : 25;
    if (k % every === 0) snap();
    if (k > 2 && s.resid / LEARN.eta < LEARN.resTol) { resid = E.stationarity(s, Sigma, P).r / LEARN.eta; if (frames[frames.length - 1].t < t) snap(); break; }
  }
  if (!Number.isFinite(resid)) { resid = E.stationarity(s, Sigma, P).r / LEARN.eta; snap(); }
  const last = frames[frames.length - 1];
  for (const f of frames) f.cl = E.clustersAuto(Float64Array.from(f.H), N, B);
  return { frames, resid, converged: resid < LEARN.resTol * 10, tEnd: last.t, Sigma };
}
function groupsOf(f, N) {
  if (!f.cl.robust) return null;
  const g = new Map();
  for (let i = 0; i < N; i++) { const l = f.cl.labels[i]; if (!g.has(l)) g.set(l, { n: 0, h: [0, 0, 0] }); const c = g.get(l); c.n++; for (let b = 0; b < B; b++) c.h[b] += f.H[i * B + b]; }
  return [...g.values()].map(c => ({ n: c.n, h: c.h.map(v => v / c.n) })).sort((a, b) => b.n - a.n);
}
// dot radius: once the neurons are grouped, members coincide, so each dot grows with its group's size
function radiiOf(f, N, r0) {
  const r = new Float32Array(N).fill(r0);
  if (!f.cl.robust) return r;
  const cnt = new Map(); for (let i = 0; i < N; i++) cnt.set(f.cl.labels[i], (cnt.get(f.cl.labels[i]) || 0) + 1);
  for (let i = 0; i < N; i++) { const n = cnt.get(f.cl.labels[i]); if (n > 1) r[i] = r0 * (0.8 + 2.2 * Math.sqrt(n / N)); }
  return r;
}

function learnShared(ctx) {
  const p = { theta: Math.max(0, Math.min(0.8, ctx.param("theta", 0.4))), s: ctx.param("s", "indep"), seed: Math.max(0, Math.round(ctx.param("seed", 0))) };
  if (!LEARN_PRESETS.some(x => x.value === p.s)) p.s = "indep";
  const subs = [], N = LEARN.N;
  const L = { p, run: null, subscribe: fn => subs.push(fn) };
  L.set = patch => {
    Object.assign(p, patch);
    ctx.setParams({ theta: p.theta, s: p.s, seed: p.seed });
    const run = runLearn(p);
    let wII = 0; const nTri = N * (N - 1) / 2; for (const f of run.frames) for (let k = 0; k < nTri; k++) if (f.W[k] > wII) wII = f.W[k];
    run.wmaxII = wII || 1;
    L.run = run;
    subs.forEach(fn => fn(run));
  };
  ctx.onParams(q => {
    const th = q.get("theta"), s = q.get("s"), sd = q.get("seed"), patch = {};
    if (th != null && Number.isFinite(+th) && +th !== p.theta) patch.theta = Math.max(0, Math.min(0.8, +th));
    if (s && s !== p.s && LEARN_PRESETS.some(x => x.value === s)) patch.s = s;
    if (sd != null && Number.isFinite(+sd) && +sd !== p.seed) patch.seed = Math.max(0, Math.round(+sd));
    if (Object.keys(patch).length) L.set(patch);
  });
  return L;
}

function figLearn(ctx, shared, opts) {
  const slot = ctx.slot(opts.slot), N = LEARN.N, p = shared.p;

  // controls
  const ctl = ui.h("div", { class: "bx-controls" }); slot.append(ctl);
  const thS = ui.slider({ label: "pruning threshold $\\theta$", min: 0, max: 0.8, step: 0.01, value: p.theta, fmt: x => ui.fmtFixed(x, 2),
    onChange: x => shared.set({ theta: x }) });
  const sSeg = ui.segmented({ label: "inputs", options: LEARN_PRESETS.map(x => ({ value: x.value, label: x.label })), value: p.s, onChange: v => shared.set({ s: v }) });
  const seedBtn = ui.h("button", { type: "button", class: "bx-btn", text: "new random start" });
  seedBtn.addEventListener("click", () => shared.set({ seed: (p.seed + 1) % 100 }));
  ctl.append(thS.el, sSeg.el, ui.h("div", { class: "u-field" }, seedBtn));

  // film strip (six frames, fixed axes)
  const strip = ui.h("div", { class: "bx-strip" }); slot.append(strip);
  // player
  const playBtn = ui.h("button", { type: "button", class: "u-play", "aria-label": "play" });
  const scrub = ui.h("input", { type: "range", class: "u-scrub", min: 0, max: 1, step: 1, value: 0, "aria-label": "training time" });
  const tLab = ui.h("span", { class: "bx-time" });
  slot.append(ui.h("div", { class: "bx-player" }, playBtn, scrub, tLab));
  const status = ui.h("div", { class: "bx-readout" }); slot.append(status);
  // the two pictures
  const pairEl = ui.h("div", { class: "space-pair" }); slot.append(pairEl);
  const left = D.panel(pairEl, "input", "input space"), right = ui.h("div", { class: "bx-panel" }); pairEl.append(right);
  const VW = 520, VH = 440, svg = D.svgRoot(left, VW, VH, "Neurons as points in the triangle of inputs");
  let order = null;
  right.append(ui.h("div", { class: "chart-title", text: "Synapses between neurons (rows and columns: neurons sorted by their final group)" }));
  const box = ui.h("div"); right.append(box);
  const hm = heatmap(box, { values: new Float32Array(N * N), rows: N, cols: N, map: "ink", range: [0, 1], size: 420, zlabel: "synapse weight $w_{ij}$",
      xlabel: "neuron j", ylabel: "neuron i", tooltip: (i, j, v) => `neurons ${order[i] + 1} and ${order[j] + 1}: w = ${ui.fmt(v, 3)}` });
  const T = D.triangle(svg, VW, VH, { top: 26 });
  const gDots = svgEl("g", {}, svg), gLab = svgEl("g", {}, svg);
  const details = ctx.slot(opts.details);

  let run = null, cur = null, idx = 0, playing = false, acc = 0;
  const hOf = (f, i) => [f.H[i * B], f.H[i * B + 1], f.H[i * B + 2]];
  function show(k) {
    idx = k; cur = run.frames[k]; scrub.value = k;
    tLab.textContent = `t = ${ui.fmtFixed(cur.t, 2)}`;
    gDots.replaceChildren(); gLab.replaceChildren();
    const rad = radiiOf(cur, N, 6);
    for (let i = 0; i < N; i++) {
      const h = hOf(cur, i), [x, y] = T.at(h);
      svgEl("circle", { cx: x, cy: y, r: rad[i], fill: bary(h), stroke: P.fg, "stroke-width": 0.8 }, gDots);
    }
    const gr = groupsOf(cur, N);
    if (gr && gr.length <= 8) {                      // group sizes, written on the side towards the centre
      const [gx, gy] = T.at([1 / 3, 1 / 3, 1 / 3]);
      gr.forEach(c => {
        const [x, y] = T.at(c.h), dx = gx - x, dy = gy - y, n = Math.hypot(dx, dy) || 1, off = 6 * (0.8 + 2.2 * Math.sqrt(c.n / N)) + 16;
        D.txt(gLab, x + dx / n * off, y + dy / n * off + 5, `${c.n}`, { "font-size": 15, "font-weight": 700, fill: P.fg, "text-anchor": "middle" });
      });
    }
    {
      const V = new Float32Array(N * N), at = new Int32Array(N); for (let a = 0; a < N; a++) at[order[a]] = a;
      let k = 0;
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const w = cur.W[k++]; V[at[i] * N + at[j]] = w; V[at[j] * N + at[i]] = w; }
      hm.update({ values: V, range: [0, run.wmaxII] });
    }
    status.innerHTML = gr ? `<b>${gr.length} group${gr.length > 1 ? "s" : ""}</b> of ${gr.map(c => c.n).join(", ")} neurons` : "not yet grouped: the neurons are still spread out";
    strip.querySelectorAll(".bx-frame").forEach(el => el.classList.toggle("on", +el.dataset.k === k));
  }
  function buildStrip() {
    strip.replaceChildren();
    const fr = run.frames;
    FILM_T.forEach(tt => {
      let k = fr.length - 1;
      if (tt != null) { k = 0; for (let q = 0; q < fr.length; q++) if (Math.abs(fr[q].t - tt) < Math.abs(fr[k].t - tt)) k = q; }
      const f = fr[k], box = ui.h("button", { type: "button", class: "bx-frame", "data-k": k, "aria-label": `jump to t = ${ui.fmtFixed(f.t, 1)}` });
      const s = D.svgRoot(box, 220, 200, `the triangle at t = ${ui.fmtFixed(f.t, 1)}`);
      const Tt = D.triangle(s, 220, 200, { sq: 12, labels: false, top: 0 });
      const rad = radiiOf(f, N, 3.4);
      for (let i = 0; i < N; i++) { const h = hOf(f, i), [x, y] = Tt.at(h); svgEl("circle", { cx: x, cy: y, r: rad[i], fill: bary(h), stroke: P.fg, "stroke-width": 0.5 }, s); }
      box.append(ui.h("div", { class: "bx-frame-t", text: tt == null ? `t = ${ui.fmtFixed(f.t, 1)} (end)` : `t = ${+f.t.toFixed(2)}` }));
      box.addEventListener("click", () => { pause(); show(k); });
      strip.append(box);
    });
  }
  function onRun(r) {
    pause();
    run = r;
    thS.set(p.theta); sSeg.set(p.s);
    {
      const lastF = run.frames[run.frames.length - 1];
      order = Array.from(E.orderByCluster(lastF.cl.labels, N));
      const gs = [], at = order.map(i => lastF.cl.labels[i]);
      for (let a = 0; a < N;) { let b = a; while (b < N && at[b] === at[a]) b++; const hm_ = [0, 0, 0]; for (let q = a; q < b; q++) for (let c = 0; c < B; c++) hm_[c] += lastF.H[order[q] * B + c] / (b - a); gs.push({ start: a, end: b, color: bary(hm_) }); a = b; }
      hm.update({ groups: { rows: gs } });
    }
    scrub.max = run.frames.length - 1;
    buildStrip();
    show(run.frames.length - 1);
    const last = run.frames[run.frames.length - 1], gr = groupsOf(last, N);
    details.innerHTML = `<p>Covariance rule (<a href="#basics/basics-eq-rule">5</a>) with $\\rho(w) = -\\eta\\theta - \\lambda w$, the same form and parameter names as <a href="#perron">Perron clusters</a>: $N = ${N}$ neurons, three inputs with unit variances (${LEARN_PRESETS.find(x => x.value === p.s).label}), $\\theta = ${ui.fmtFixed(p.theta, 2)}$, $\\lambda = \\eta = 1$, floor $w_{\\min} = 10^{-4}$, every synapse plastic (lateral and onto the inputs, lateral gain $\\gamma = 1$). Start: synapses between neurons at zero, synapses onto the inputs drawn from an exponential distribution with mean 0.2 (seed ${p.seed}). The rule uses the exact covariance $\\Sigma_g$ (no sampling), Euler steps of $dt = ${LEARN.dt}$, engine.js. The run ${run.converged ? "settled" : "was stopped"} at $t = ${ui.fmtFixed(run.tEnd, 1)}$ with largest remaining weight drift ${ui.fmt(run.resid, 2)}. Groups: single linkage on the mixtures, cut at the largest multiplicative gap between merge heights (and only when that gap is clear); ${gr ? `${gr.length} at the end` : "no clear grouping at the end"}. The matrix shows the weights of the synapses between neurons on one scale for the whole run; neurons are sorted by their group at the end.</p>`;
    ui.renderMath(details);
  }
  // player
  const ICON_PLAY = '<svg width="14" height="14" viewBox="0 0 14 14"><path d="M3 1.5v11l9-5.5z" fill="currentColor"/></svg>';
  const ICON_PAUSE = '<svg width="14" height="14" viewBox="0 0 14 14"><path d="M3 1.5h3v11H3zM8 1.5h3v11H8z" fill="currentColor"/></svg>';
  function pause() { playing = false; playBtn.innerHTML = ICON_PLAY; playBtn.setAttribute("aria-label", "play"); }
  function play() { if (idx >= run.frames.length - 1) show(0); playing = true; playBtn.innerHTML = ICON_PAUSE; playBtn.setAttribute("aria-label", "pause"); }
  playBtn.addEventListener("click", () => (playing ? pause() : play()));
  scrub.addEventListener("input", () => { pause(); show(+scrub.value); });
  ui.animate(slot, dt => {
    if (!playing || !ctx.visible) return;
    acc += dt * 30;                                   // 30 frames per second
    if (acc < 1) return;
    const k = Math.min(run.frames.length - 1, idx + Math.floor(acc)); acc -= Math.floor(acc);
    show(k);
    if (k >= run.frames.length - 1) pause();
  });
  pause();
  shared.subscribe(onRun);
}

// ============================================================ 6. the distance rule descends J
const DIST = { N: 30, seed: 3, b: 0.05, alpha: 0.05, dt: 0.5, T: 240, plotT: 240 };
function figEnergy(ctx) {
  const slot = ctx.slot("energy"), { N } = DIST, S = M.sigma3();
  const A = M.dirichletAttach(N, DIST.seed);
  const fr = M.runDistance({ N, A, S, b: DIST.b, alpha: DIST.alpha, dt: DIST.dt, T: DIST.T, every: 1 });
  const first = fr[0], last = fr[fr.length - 1];
  let rise = -Infinity; for (let k = 1; k < fr.length; k++) rise = Math.max(rise, fr[k].J - fr[k - 1].J);
  // plasticity residual at the end (projected)
  const { D2 } = M.meanEnergy(last.W, A, last.H, N, S);
  let resid = 0;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const w = last.W[i * N + j], d = DIST.b - DIST.alpha * w - D2[i * N + j] / 2; if (w > 0 || d > 0) resid = Math.max(resid, Math.abs(d)); }

  const two = ui.h("div", { class: "bx-two energy" }); slot.append(two);
  const left = D.panel(two, "input", "input space"), right = ui.h("div", { class: "bx-panel" }); two.append(right);
  const VW = 440, VH = 380, svg = D.svgRoot(left, VW, VH, "Mixtures before and after learning under the distance rule");
  const T = D.triangle(svg, VW, VH, { top: 26 });
  const hOf = (f, i) => [f.H[i * B], f.H[i * B + 1], f.H[i * B + 2]];
  for (let i = 0; i < N; i++) { const [x0, y0] = T.at(hOf(first, i)), [x1, y1] = T.at(hOf(last, i)); svgEl("line", { x1: x0, y1: y0, x2: x1, y2: y1, stroke: P.mute2, "stroke-width": 1.2 }, svg); }
  for (let i = 0; i < N; i++) { const [x, y] = T.at(hOf(first, i)); svgEl("circle", { cx: x, cy: y, r: 4.5, fill: P.surface, stroke: P.mute, "stroke-width": 1.3 }, svg); }
  for (let i = 0; i < N; i++) { const h = hOf(last, i), [x, y] = T.at(h); svgEl("circle", { cx: x, cy: y, r: 5.5, fill: bary(h), stroke: P.fg, "stroke-width": 0.8 }, svg); }
  ui.legend({ parent: left, items: [{ label: "before learning", color: P.mute, shape: "ring" }, { label: `after (t = ${DIST.T})`, color: P.ink2, shape: "dot" }] });

  // envelope check at the final weights: central differences of E* in every live w_ij against D_ij^2 / 2
  let envErr = 0, nEnv = 0;
  { const W = Float64Array.from(last.W), hstep = 1e-5, Es = W_ => M.meanEnergy(W_, A, M.harmonic(W_, A, N), N, S).E;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const w0 = W[i * N + j]; if (w0 <= hstep) continue;
      W[i * N + j] = W[j * N + i] = w0 + hstep; const ep = Es(W);
      W[i * N + j] = W[j * N + i] = w0 - hstep; const em = Es(W);
      W[i * N + j] = W[j * N + i] = w0;
      envErr = Math.max(envErr, Math.abs((ep - em) / (2 * hstep) - D2[i * N + j] / 2)); nEnv++;
    } }
  const shown = fr.filter(f => f.t <= DIST.plotT);
  const ts = shown.map(f => f.t), lo = Math.min(...fr.map(f => Math.min(f.cost, f.J, f.E))), hi = Math.max(...fr.map(f => Math.max(f.E, f.J)));
  const padY = 0.08 * (hi - lo);
  D.linePlot(right, {
    height: 340, xr: [0, DIST.plotT], yr: [lo - padY, hi + 2 * padY], xlabel: "training time t", ylabel: "objective and its terms",
    series: [
      { x: ts, y: shown.map(f => f.E), color: P.cat[3], width: 2, dash: "6 4", label: "average half-dissipation E*", labelDy: -9 },
      { x: ts, y: shown.map(f => f.cost), color: P.cat[4], width: 2, dash: "2 4", label: "synapse cost", labelDy: -9 },
      { x: ts, y: shown.map(f => f.J), color: P.fg, width: 3.2, label: "J = E* + cost", labelDy: -9 },
    ],
  });
  ctx.slot("energy-details").innerHTML = `<p>Distance rule (<a href="#basics/basics-eq-dist">6</a>) on $N = ${N}$ linear neurons. Each neuron is attached to all three inputs with fixed conductances drawn as Dirichlet(1) shares of a total of 1 (seed ${DIST.seed}); the synapses between neurons start at zero and learn with $b = ${DIST.b}$, $\\alpha = ${DIST.alpha}$. Inputs independent with unit variance; expectations are exact ($\\mathbb E[(u_i - u_j)^2] = (h_i - h_j)^\\top\\Sigma_g(h_i - h_j)$, no sampling), so $E^*$ and $J$ are computed in closed form at every step. Projected Euler steps of $dt = ${DIST.dt}$ up to $t = ${DIST.T}$, where the largest remaining drift $|dw_{ij}/dt|$ is ${ui.fmt(resid, 2)}. ${rise <= 0 ? `$J$ fell at every step (the smallest decrease was ${ui.fmt(-rise, 2)}).` : `The largest step-to-step increase of $J$ was ${ui.fmt(rise, 2)}, that is, rounding.`} $J$ went from ${ui.fmtFixed(first.J, 3)} to ${ui.fmtFixed(last.J, 3)}; ${last.live} of ${N * (N - 1) / 2} possible synapses are alive at the end. The envelope step, $\\partial E^*/\\partial w_{ij} = \\tfrac12\\mathbb E[(u_i - u_j)^2]$, checked by central differences of $E^*$ at the final weights over all ${nEnv} live synapses: largest error ${ui.fmt(envErr, 2)}.</p>`;
  ui.renderMath(ctx.slot("energy-details"));
}
