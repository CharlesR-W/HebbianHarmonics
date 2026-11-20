// Chapter "perron": three-input figures drawn in the triangle of mixtures (input space).
//   [data-slot="grid"]  "Neurons end on the predicted spots": small multiples, 3 covariances x 4 thresholds, each a
//                       finished run of the rule (computed live, deterministic) over the predicted stable parts.
//   [data-slot="live"]  the same for one run you steer (theta, rho12), animated: the alternative to the grid.
// Shared triangle helpers are exported for ch_perron_land.js.
import * as ui from "./ui.js";
import { P, boundaryColor, mixBary } from "./palette.js";
import * as M from "./ch_perron_model.js";
import * as T from "./ch_perron_teach_model.js";

const NS = "http://www.w3.org/2000/svg";
export const FONT = "Inter, system-ui, sans-serif";
export const BCOL = [0, 1, 2].map(b => boundaryColor(b, 3));
export const GL = ["g₁", "g₂", "g₃"];
export const fx = (x, d = 2) => ui.fmtFixed(x, d);
export const hstr = h => `(${Array.from(h).slice(0, 3).map(x => fx(x)).join(", ")})`;

export function svg(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
export function stext(parent, x, y, s, attrs = {}) {
  const t = svg("text", { x, y, "font-size": 11, fill: P.ink2, "font-family": FONT, ...attrs }, parent);
  t.textContent = s;
  return t;
}

// Triangle of mixtures in a W x H box: g1 at the top, g2 bottom left, g3 bottom right.
export function triGeom(W, H, pad = 22) {
  const side = Math.min(W - 2 * pad, (H - 2 * pad) / (Math.sqrt(3) / 2));
  const hgt = side * Math.sqrt(3) / 2, cx = W / 2, top = (H - hgt) / 2;
  const V = [[cx, top], [cx - side / 2, top + hgt], [cx + side / 2, top + hgt]];
  const at = h => { const s = h[0] + h[1] + h[2] || 1; return [(h[0] * V[0][0] + h[1] * V[1][0] + h[2] * V[2][0]) / s, (h[0] * V[0][1] + h[1] * V[1][1] + h[2] * V[2][1]) / s]; };
  // barycentric coordinates of a point (may be negative outside the triangle)
  const bary = (x, y) => {
    const [[x0, y0], [x1, y1], [x2, y2]] = V, d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    const a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / d, b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / d;
    return [a, b, 1 - a - b];
  };
  return { W, H, V, at, bary, side };
}

// Outline and the three inputs (squares with an ink outline just outside their corner, labelled).
export function drawFrame(el, G, { sq = 9, font = 12, gap = RING_R + 3 } = {}) {
  svg("path", { d: `M${G.V[0]}L${G.V[1]}L${G.V[2]}Z`, fill: "none", stroke: P.mute2, "stroke-width": 1.1 }, el);
  const cen = [(G.V[0][0] + G.V[1][0] + G.V[2][0]) / 3, (G.V[0][1] + G.V[1][1] + G.V[2][1]) / 3];
  G.V.forEach(([x, y], b) => {
    const dx = x - cen[0], dy = y - cen[1], n = Math.hypot(dx, dy), off = sq / 2 + gap, qx = x + off * dx / n, qy = y + off * dy / n;
    svg("rect", { x: qx - sq / 2, y: qy - sq / 2, width: sq, height: sq, rx: 1.5, fill: BCOL[b], stroke: P.fg, "stroke-width": 1 }, el);
    if (b === 0) stext(el, qx + sq / 2 + 4, qy + font / 3, GL[b], { "font-size": font, fill: P.fg, "font-weight": 600 });
    else stext(el, qx, qy + sq / 2 + font, GL[b], { "text-anchor": "middle", "font-size": font, fill: P.fg, "font-weight": 600 });
  });
}

export const RING_R = 10;
export function drawRing(el, G, h, r = RING_R) {
  const [x, y] = G.at(h);
  return svg("circle", { cx: x, cy: y, r, fill: "none", stroke: P.fg, "stroke-width": 1.5 }, el);
}
export const groupRadius = (n, N) => 2.6 + 5.4 * Math.sqrt(n / N);
export function drawGroup(el, G, g, N) {
  const [x, y] = G.at(g.h);
  return svg("circle", { cx: x, cy: y, r: groupRadius(g.n, N), fill: mixBary(g.h, BCOL), stroke: P.surface, "stroke-width": 1.2 }, el);
}

// Nearest stable part to a mixture (L1 distance).
export function nearestPart(parts, h) {
  let best = null, bd = Infinity;
  for (const p of parts) { const d = Math.abs(p.h[0] - h[0]) + Math.abs(p.h[1] - h[1]) + Math.abs(p.h[2] - h[2]); if (d < bd) { bd = d; best = p; } }
  return { part: best, d: bd };
}

export function keyItems() {
  return [
    { label: "predicted resting place (stable part)", color: P.fg, shape: "ring" },
    { label: "a group of neurons with nearly identical mixtures at the end of the run; larger = more neurons", color: mixBary([0.5, 0.5, 0], BCOL), shape: "dot" },
  ];
}

// ======================================================================= the grid
export const GRID_THETAS = [0.15, 0.35, 0.55, 0.75];
export const GRID_ROWS = ["pair", "tilt", "tri"];
export function gridParams(sid, theta) { return { ...M.DEFAULTS, theta, Sigma: M.sigmaOf(M.SIGMAS[sid]) }; }

export function mountGrid(slot, ctx) {
  const grid = ui.h("div", { class: "pg-grid" });
  const status = ui.h("div", { class: "pc-read" });
  const key = ui.legend({ items: keyItems() });
  key.el.classList.add("chart-legend");
  slot.replaceChildren(grid, key.el, status);
  grid.append(ui.h("div", { class: "pg-corner" }));
  for (const th of GRID_THETAS) grid.append(ui.h("div", { class: "pg-col", text: `θ = ${fx(th)}` }));
  const ROWTXT = {
    pair: ["g₁, g₂ correlated", "correlation 0.6; g₃ independent"],
    tilt: ["… and g₂ louder", "as above, but g₂ has variance 1.6"],
    tri: ["all three correlated", "ρ₁₂ = 0.6, ρ₂₃ = 0.45, ρ₁₃ = 0.3"],
  };
  const cells = [];
  for (const sid of GRID_ROWS) {
    grid.append(ui.h("div", { class: "pg-row" }, ui.h("b", { text: ROWTXT[sid][0] }), ROWTXT[sid][1]));
    for (const th of GRID_THETAS) {
      const s = svg("svg", { viewBox: "0 0 200 198", role: "img", "aria-label": `run at theta ${th}` });
      const G = triGeom(200, 176, 30);
      const p = gridParams(sid, th);
      const parts = T.stableParts(p.Sigma, 3, th);
      drawFrame(s, G, { sq: 8, font: 12 });
      stext(s, 4, 14, `θ = ${fx(th)}`, { class: "pg-cellth", "font-size": 13, fill: P.fg, "font-weight": 600 });   // shown on phones only
      const gR = svg("g", {}, s), gD = svg("g", {}, s);
      for (const q of parts) drawRing(gR, G, q.h);
      grid.append(s);
      const cell = { sid, th, s, G, gD, parts, p, res: null };
      cells.push(cell);
      s.addEventListener("pointermove", e => ui.showTip(e.clientX, e.clientY, tip(cell)));
      s.addEventListener("pointerleave", () => ui.hideTip());
    }
  }
  function tip(c) {
    const rows = c.parts.map(q => ({ name: `predicted ${hstr(q.h)}`, value: "" }));
    if (c.res) for (const g of c.res.final.groups) {
      const { d } = nearestPart(c.parts, g.h);
      rows.push({ name: `${g.n} neurons at ${hstr(g.h)}`, value: `${fx(d, 3)} from the nearest ring`, color: mixBary(g.h, BCOL) });
    }
    return ui.tipRows(`${ROWTXT[c.sid][0]}, θ = ${fx(c.th)}`, rows, c.res ? `distance: sum of |differences| of the mixtures. ${c.res.final.resid < M.RES_TOL ? "Converged" : "Stopped at the step limit"} at t = ${fx(c.res.final.t, 1)} (largest weight change ${ui.fmt(c.res.final.resid, 2)}).` : "running…");
  }
  let done = 0;
  status.textContent = `Running the rule: 0 of ${cells.length} runs done…`;
  cells.forEach(c => M.run(c.p, { record: 10 }).then(res => {
    c.res = res;
    for (const g of res.final.groups) drawGroup(c.gD, c.G, g, c.p.n);
    done++;
    if (done < cells.length) { status.textContent = `Running the rule: ${done} of ${cells.length} runs done…`; return; }
    let nG = 0, dmax = 0, empty = 0, nParts = 0, dead = 0;
    for (const x of cells) {
      const gs = x.res.final.groups; nG += gs.length; nParts += x.parts.length; dead += x.res.final.nDead;
      for (const g of gs) dmax = Math.max(dmax, nearestPart(x.parts, g.h).d);
      empty += x.parts.filter(q => !gs.some(g => nearestPart([q], g.h).d < 0.05)).length;
    }
    const conv = cells.filter(x => x.res.final.resid < M.RES_TOL).length;
    status.innerHTML = `${conv === cells.length ? `All ${cells.length} runs converged` : `${conv} of ${cells.length} runs converged`}; ${nG} groups. The largest distance from a group to its nearest ring is ${ui.fmt(dmax, 3)} (sum of absolute differences of the mixtures); ${empty} of the ${nParts} rings ${empty === 1 ? "is" : "are"} empty${dead ? `; ${dead} neurons died` : ""}.` +
      ` <span data-g1>With γ = 1: computing…</span>`;
    // the same 12 runs at γ = 1, for the comparison quoted in the caption
    Promise.all(cells.map(x => M.run({ ...x.p, gamma: 1 }).then(r => ({ x, r })))).then(rs => {
      let e1 = 0, d1 = 0;
      for (const { x, r } of rs) {
        const gs = r.final.groups;
        for (const g of gs) d1 = Math.max(d1, nearestPart(x.parts, g.h).d);
        e1 += x.parts.filter(q => !gs.some(g => nearestPart([q], g.h).d < 0.05)).length;
      }
      const el = status.querySelector("[data-g1]");
      if (el) el.textContent = `With γ = 1 the same runs leave ${e1} of the ${nParts} rings empty (largest distance ${ui.fmt(d1, 3)}).`;
    });
  }));
  return { cells };
}

// ======================================================================= the live version
// mountLive(slot, ctx, {r13, r23, before, onRun}): one steerable run on Σ with correlations (ρ, r13, r23); `before`
// (an element) is placed first in the layout (the phase map); onRun(θ, ρ) is called at every restart. Returns {set(θ, ρ)}.
export function mountLive(slot, ctx, { r13 = 0, r23 = 0, before = null, onRun = null } = {}) {
  let theta = Math.min(0.75, Math.max(0.05, ctx.param("lt", 0.45))), rho = Math.min(0.9, Math.max(-0.5, ctx.param("lr", 0.6)));
  const ctl = ui.h("div", { class: "pt-ctl" });
  const sT = ui.slider({ label: "threshold $\\theta$", min: 0.05, max: 0.75, step: 0.01, value: theta, fmt: x => fx(x),
    onInput: x => { theta = x; ctx.setParams({ lt: x }); restart(); } });
  const sR = ui.slider({ label: "correlation of $g_1$ and $g_2$", min: -0.5, max: 0.9, step: 0.01, value: rho, fmt: x => fx(x),
    onInput: x => { rho = x; ctx.setParams({ lr: x }); restart(); } });
  const again = ui.h("button", { type: "button", class: "pc-btn", text: "run again from a new random start" });
  let seed = 0;
  again.addEventListener("click", () => { seed++; restart(); });
  ctl.append(sT.el, sR.el);
  const s = svg("svg", { viewBox: "0 0 440 400", role: "img", "aria-label": "neurons moving in the triangle of mixtures" });
  const G = triGeom(440, 370, 36);
  drawFrame(s, G, { sq: 12, font: 15, gap: 15 });
  const gR = svg("g", {}, s), gN = svg("g", {}, s), gG = svg("g", {}, s);
  const side = ui.h("div");
  const read = ui.h("div", { class: "pc-read" });
  const key = ui.legend({ items: [keyItems()[0], { label: "a neuron (colour: its mixture of the inputs)", color: mixBary([0.4, 0.3, 0.3], BCOL), shape: "dot" }] });
  side.append(read, ui.h("div", { style: { marginTop: "10px" } }, again), key.el);
  slot.append(ctl, ui.h("div", { class: before ? "ph-wrap" : "pl-wrap" }, ...(before ? [before] : []), s, side));

  const N = M.DEFAULTS.n;
  const dots = Array.from({ length: N }, () => svg("circle", { r: 3.2, stroke: P.surface, "stroke-width": 0.8 }, gN));
  let sim = null, parts = [], finished = false, pending = 0, tie = null;
  const TIE_NOTES = {
    removed: "θ equals a covariance here, a tie of the prediction: the borderline resting places fail the strict test and are not drawn as rings, and the run may not converge.",
    at: "θ equals a covariance here, a tie of the prediction; the run may converge slowly.",
    near: "θ is close to a covariance (a near tie): convergence is slow, and groups may sit a little off the rings.",
  };
  const marginalNote = () => tie ? `<br>${TIE_NOTES[tie]}` : "";
  function restart() {
    const Sigma = M.sigmaOf({ v: [1, 1, 1], r: [rho, r13, r23] });
    onRun?.(theta, rho);
    parts = T.stableParts(Sigma, 3, theta);
    // a tie in the prediction: some candidate's outside margin or eigenvalue gap is (nearly) zero
    const cands = T.enumerateParts(Sigma, 3, theta, { candidates: true });
    // only candidates that are (nearly) the top vector on their support can tie: an outside margin near zero, or a
    // top-eigenvalue gap near zero, with the other condition not clearly failing
    const within = (q, e) => (Number.isFinite(q.margin) && Math.abs(q.margin) < e && (!q.part || q.S.length === 1 || q.gap > -e)) ||
      (q.part && q.S.length > 1 && Number.isFinite(q.gap) && Math.abs(q.gap) < e && !(q.margin < -e));
    tie = null;
    if (cands.some(q => within(q, 1e-6))) tie = cands.some(q => !q.stable && within(q, 1e-6)) ? "removed" : "at";
    else if (cands.some(q => within(q, 0.02))) tie = "near";
    gR.replaceChildren(); for (const q of parts) drawRing(gR, G, q.h, 12);
    sim = M.createSim({ ...M.DEFAULTS, seed, theta, Sigma });
    sim.step(); finished = false; pending = 0; gG.replaceChildren(); gN.style.display = "";
    draw();
  }
  function draw() {
    const H = sim.s.H;
    for (let i = 0; i < N; i++) {
      const h = [H[3 * i], H[3 * i + 1], H[3 * i + 2]], [x, y] = G.at(h);
      dots[i].setAttribute("cx", x.toFixed(1)); dots[i].setAttribute("cy", y.toFixed(1)); dots[i].setAttribute("fill", mixBary(h, BCOL));
    }
    const t = sim.steps * M.DT;
    if (!finished) { read.innerHTML = `Running: time ${fx(t, 1)}. Predicted: ${parts.length} ring${parts.length === 1 ? "" : "s"}.${marginalNote()}`; return; }
    const a = M.analyse(sim);
    if (!(a.resid < M.RES_TOL)) {
      read.innerHTML = `<b>Stopped at the step limit</b> (time ${fx(t, 1)}) without converging: the largest weight change is still ${ui.fmt(a.resid, 2)} per unit time (target 10⁻⁷). Predicted: ${parts.length} ring${parts.length === 1 ? "" : "s"}.${marginalNote()}`;
      return;
    }
    // settled: draw each group once, sized by its number of neurons, with the count
    gN.style.display = "none"; gG.replaceChildren();
    for (const g of a.groups) {
      const [x, y] = G.at(g.h), r = Math.min(9.5, 1.3 * groupRadius(g.n, N));
      svg("circle", { cx: x, cy: y, r, fill: mixBary(g.h, BCOL), stroke: P.surface, "stroke-width": 1.5 }, gG);
      stext(gG, x + 16, y - 10, String(g.n), { "font-size": 13, fill: P.fg, "font-weight": 600 });
    }
    const lines = a.groups.map(g => `${g.n} neurons at ${hstr(g.h)}, ${fx(nearestPart(parts, g.h).d, 3)} from the nearest ring`);
    read.innerHTML = `<b>Converged</b> at time ${fx(t, 1)} (largest weight change ${ui.fmt(a.resid, 2)}): ${a.groups.length} group${a.groups.length === 1 ? "" : "s"} for ${parts.length} predicted ring${parts.length === 1 ? "" : "s"}.<br>` + lines.join("<br>") + (a.nDead ? `<br>${a.nDead} neurons died.` : "") + marginalNote();
  }
  restart();
  const reduced = ui.reducedMotion();
  ui.animate(slot, () => {
    if (finished) return;
    const t0 = performance.now();
    // a few steps per frame early on (so the motion is visible), more later
    const k = reduced ? 400 : sim.steps < 160 ? 1 : sim.steps < 400 ? 4 : 16;
    for (let j = 0; j < k && !sim.done() && performance.now() - t0 < 12; j++) sim.step();
    if (sim.done()) finished = true;
    draw();
  });
  ctx.onParams?.(p => {
    const a = p.get("lt"), b = p.get("lr");
    let ch = false;
    if (a != null && Number.isFinite(+a) && +a !== theta) { theta = +a; sT.set(theta); ch = true; }
    if (b != null && Number.isFinite(+b) && +b !== rho) { rho = +b; sR.set(rho); ch = true; }
    if (ch) restart();
  });
  return { set(t, r) { theta = Math.min(0.75, Math.max(0.05, t)); rho = Math.min(0.9, Math.max(-0.5, r)); sT.set(theta); sR.set(rho); ctx.setParams({ lt: theta, lr: rho }); restart(); } };
}
