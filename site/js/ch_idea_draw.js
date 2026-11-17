// Chapter "idea": drawing helpers for the live figure. A small SVG plot whose layers are redrawn cheaply on every
// frame (curves, dots, reference lines, clickable points), and the network diagram (three clamped inputs at the
// corners of a triangle, eight interior neurons at their attachment shares, attachments and plastic synapses).
import * as ui from "./ui.js";
import { svgEl, scale, niceTicks } from "./charts.js";
import { P, colorOf, catColor } from "./palette.js";

const txt = (parent, x, y, s, attrs = {}) => { const t = svgEl("text", { x, y, ...attrs }, parent); t.textContent = s; return t; };

/** miniPlot(el, {height, xlabel, ylabel}) -> {draw(spec)}.
 *  spec: {xr: [lo, hi], yr: [lo, hi], equal?, layers: [
 *    {type: "line", x, y, color, width?, dash?, opacity?},
 *    {type: "dots", x, y, color (css | css[]), r?, stroke?, ring?: bool, tip?(k) -> html, click?(k)},
 *    {type: "vline" | "hline", v, color?, dash?, label?}], note?: string (a line of text above the plot)} */
export function miniPlot(el, o = {}) {
  el.classList.add("chart");
  const noteEl = ui.h("div", { class: "u-help", style: { minHeight: "1.4em", margin: "0 0 2px", fontVariantNumeric: "tabular-nums" } });
  const host = ui.h("div"); el.replaceChildren(noteEl, host);
  const H = o.height || 200;
  let last = null, Wd = 0;
  const ro = new ResizeObserver(() => { const w = el.clientWidth; if (w && Math.abs(w - Wd) > 1 && last) draw(last); });
  ro.observe(el);
  function draw(spec) {
    last = spec;
    const W = Math.max(200, el.clientWidth || 300); Wd = W;
    const m = { l: o.ylabel ? 60 : 44, r: 10, t: 10, b: o.xlabel ? 36 : 20 };
    let [x0, x1] = spec.xr, [y0, y1] = spec.yr;
    if (spec.equal) {                         // same data length per pixel on both axes (pair scatter)
      const pw = W - m.l - m.r, ph = H - m.t - m.b, sx = (x1 - x0) / pw, sy = (y1 - y0) / ph;
      if (sx > sy) { const c = (y0 + y1) / 2, hh = sx * ph / 2; y0 = c - hh; y1 = c + hh; } else { const c = (x0 + x1) / 2, hw = sy * pw / 2; x0 = c - hw; x1 = c + hw; }
    }
    const X = scale("linear", x0, x1, m.l, W - m.r), Y = scale("linear", y0, y1, H - m.b, m.t);
    const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.ylabel || "plot" });
    const g = svgEl("g", { class: "grid" }, svg);
    const yt = niceTicks(y0, y1, Math.max(3, Math.round(H / 55))), xt = niceTicks(x0, x1, Math.max(3, Math.round(W / 85)));
    yt.forEach(v => svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g));
    const ax = svgEl("g", { class: "axis" }, svg);
    svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b + .5, y2: H - m.b + .5 }, ax);
    svgEl("line", { x1: m.l - .5, x2: m.l - .5, y1: m.t, y2: H - m.b }, ax);
    yt.forEach(v => txt(svg, m.l - 6, Y(v) + 3.5, ui.fmt(v, 2), { "text-anchor": "end" }));
    xt.forEach(v => { svgEl("line", { x1: X(v), x2: X(v), y1: H - m.b, y2: H - m.b + 4 }, ax); txt(svg, X(v), H - m.b + 15, ui.fmt(v, 2), { "text-anchor": "middle" }); });
    if (o.xlabel) txt(svg, (m.l + W - m.r) / 2, H - 4, o.xlabel, { "text-anchor": "middle", class: "ttl" });
    if (o.ylabel) txt(svg, 0, 0, o.ylabel, { "text-anchor": "middle", class: "ttl", transform: `translate(11,${(m.t + H - m.b) / 2}) rotate(-90)` });
    const cid = (o._cid = o._cid || "ic" + Math.random().toString(36).slice(2));
    const clip = svgEl("clipPath", { id: cid }, svgEl("defs", {}, svg));
    svgEl("rect", { x: m.l, y: m.t - 3, width: W - m.l - m.r, height: H - m.t - m.b + 6 }, clip);
    const plot = svgEl("g", { "clip-path": `url(#${cid})` }, svg);
    const hits = [];
    for (const L of spec.layers) {
      if (L.type === "line") {
        let d = "", pen = false;
        L.x.forEach((xv, k) => { const yv = L.y[k]; if (!Number.isFinite(xv) || !Number.isFinite(yv)) { pen = false; return; } d += (pen ? "L" : "M") + X(xv).toFixed(1) + "," + Y(yv).toFixed(1); pen = true; });
        svgEl("path", { d, fill: "none", stroke: L.color || P.fg, "stroke-width": L.width || 1.6, "stroke-dasharray": L.dash || null, opacity: L.opacity ?? 1, "stroke-linejoin": "round" }, plot);
      } else if (L.type === "vline" || L.type === "hline") {
        const at = L.type === "vline" ? { x1: X(L.v), x2: X(L.v), y1: m.t, y2: H - m.b } : { x1: m.l, x2: W - m.r, y1: Y(L.v), y2: Y(L.v) };
        svgEl("line", { ...at, stroke: L.color || P.mute2, "stroke-dasharray": L.dash || "3 3", "stroke-width": 1 }, plot);
        if (L.label) {
          if (L.type === "vline") txt(svg, X(L.v) + 4, m.t + 10, L.label, { fill: L.color || P.mute });
          else txt(svg, W - m.r - 3, Y(L.v) - 4, L.label, { "text-anchor": "end", fill: L.color || P.mute });
        }
      } else if (L.type === "dots") {
        L.x.forEach((xv, k) => {
          const yv = L.y[k]; if (!Number.isFinite(xv) || !Number.isFinite(yv)) return;
          const col = Array.isArray(L.color) ? L.color[k] : L.color || P.fg;
          const cx = X(xv), cy = Y(yv);
          svgEl("circle", L.ring ? { cx, cy, r: L.r || 5, fill: "none", stroke: col, "stroke-width": L.sw || 1.6 }
            : { cx, cy, r: L.r || 3, fill: col, stroke: L.stroke || P.surface, "stroke-width": L.stroke ? 1.5 : 0.8, opacity: L.opacity ?? 1 }, plot);
          if (L.tip || L.click) hits.push({ cx, cy, L, k });
        });
      }
    }
    noteEl.textContent = spec.note || "\u00a0";
    if (hits.length) {
      const hit = svgEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
      const near = ev => { const r = svg.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top; let best = null, bd = 100; for (const q of hits) { const d = (q.cx - px) ** 2 + (q.cy - py) ** 2; if (d < bd) { bd = d; best = q; } } return best; };
      hit.addEventListener("pointermove", ev => { const q = near(ev); if (q && q.L.tip) ui.showTip(ev.clientX, ev.clientY, q.L.tip(q.k)); else ui.hideTip(); hit.style.cursor = q && q.L.click ? "pointer" : "default"; });
      hit.addEventListener("pointerleave", () => ui.hideTip());
      hit.addEventListener("click", ev => { const q = near(ev); if (q && q.L.click) q.L.click(q.k); });
    }
    host.replaceChildren(svg);
  }
  return { draw, destroy: () => ro.disconnect() };
}

/** The network diagram. opts: {A (N x B), N, B, onPick(i), onDrag(b, value), tip(i) -> html, range}.
 *  update({u, g, W, sel: [i, j]}) redraws voltages, synapses and selection. */
export function networkView(el, o) {
  const { N, B, A } = o, VW = 520, VH = 440, R = 16, SQ = 17;
  const corners = [[260, 40], [56, 384], [464, 384]];
  const pos = [];
  for (let i = 0; i < N; i++) { let x = 0, y = 0; for (let b = 0; b < B; b++) { x += A[i * B + b] * corners[b][0]; y += A[i * B + b] * corners[b][1]; } pos.push([x, y]); }
  const svg = svgEl("svg", { viewBox: `0 0 ${VW} ${VH}`, width: "100%", role: "img", "aria-label": "network of three clamped inputs and eight interior neurons", style: "display:block;max-height:470px;touch-action:none;user-select:none" });
  el.append(svg);
  svgEl("polygon", { points: corners.map(c => c.join(",")).join(" "), fill: "none", stroke: P.line, "stroke-width": 1, "stroke-dasharray": "2 4" }, svg);
  const gA = svgEl("g", {}, svg), gW = svgEl("g", {}, svg), gN = svgEl("g", {}, svg), gB = svgEl("g", {}, svg);
  // attachments (fixed)
  for (let i = 0; i < N; i++) for (let b = 0; b < B; b++) {
    const a = A[i * B + b]; if (a < 1e-6) continue;
    svgEl("line", { x1: pos[i][0], y1: pos[i][1], x2: corners[b][0], y2: corners[b][1], stroke: catColor(b), "stroke-width": (0.4 + 3.2 * a).toFixed(2), opacity: 0.32 }, gA);
  }
  // neurons
  const circ = [], lab = [], selRing = [];
  for (let i = 0; i < N; i++) {
    const gi = svgEl("g", { style: "cursor:pointer" }, gN);
    selRing.push(svgEl("circle", { cx: pos[i][0], cy: pos[i][1], r: R + 4.5, fill: "none", stroke: P.accent, "stroke-width": 2.5, opacity: 0 }, gi));
    circ.push(svgEl("circle", { cx: pos[i][0], cy: pos[i][1], r: R, fill: P.interior, stroke: P.fg, "stroke-width": 1 }, gi));
    lab.push(txt(gi, pos[i][0], pos[i][1] + 4, String(i + 1), { "text-anchor": "middle", style: "font:600 13.5px var(--sans);pointer-events:none" }));
    gi.addEventListener("click", () => o.onPick?.(i));
    gi.addEventListener("pointermove", ev => { const h = o.tip?.(i); if (h) ui.showTip(ev.clientX, ev.clientY, h); });
    gi.addEventListener("pointerleave", () => ui.hideTip());
  }
  // inputs (draggable squares)
  const sq = [], val = [];
  for (let b = 0; b < B; b++) {
    const [cx, cy] = corners[b], gb = svgEl("g", { style: "cursor:ns-resize" }, gB);
    sq.push(svgEl("rect", { x: cx - SQ, y: cy - SQ, width: 2 * SQ, height: 2 * SQ, rx: 2, fill: P.surface, stroke: catColor(b), "stroke-width": 3 }, gb));
    const lx = b === 0 ? cx + SQ + 8 : b === 1 ? cx - SQ : cx + SQ, ly = b === 0 ? cy + 6 : cy + SQ + 24;
    val.push(txt(gb, lx, ly, "", { "text-anchor": b === 0 ? "start" : b === 1 ? "start" : "end", style: "font:600 16px var(--sans)", fill: P.fg }));
    let drag = null;
    gb.addEventListener("pointerdown", ev => { drag = { y: ev.clientY, v: o.getG(b) }; gb.setPointerCapture(ev.pointerId); ev.preventDefault(); });
    gb.addEventListener("pointermove", ev => {
      if (drag) { const sc = VH / svg.getBoundingClientRect().height; o.onDrag?.(b, drag.v - (ev.clientY - drag.y) * sc / 60); return; }
      ui.showTip(ev.clientX, ev.clientY, ui.tipRows(`input g${"₁₂₃"[b]} (clamped)`, [{ name: "voltage", value: ui.fmt(o.getG(b), 3) }], "Drag up or down to change it."));
    });
    const end = () => { if (drag) { drag = null; o.onDragEnd?.(); } };
    gb.addEventListener("pointerup", end); gb.addEventListener("pointercancel", end);
    gb.addEventListener("pointerleave", () => ui.hideTip());
  }
  const [lo, hi] = o.range;
  function update({ u, g, W, sel }) {
    for (let b = 0; b < B; b++) { sq[b].setAttribute("fill", colorOf("div", g[b], lo, hi)); val[b].textContent = `g${"₁₂₃"[b]} = ${ui.fmtFixed(g[b], 2)}`; }
    for (let i = 0; i < N; i++) {
      circ[i].setAttribute("fill", colorOf("div", u[i], lo, hi));
      lab[i].setAttribute("fill", Math.abs(u[i]) > 0.62 * hi ? "#fff" : P.fg);
      const s = sel[0] === i ? 1 : sel[1] === i ? 2 : 0;
      selRing[i].setAttribute("opacity", s ? 1 : 0);
      selRing[i].setAttribute("stroke-dasharray", s === 2 ? "4 3" : "none");
    }
    gW.replaceChildren();
    let wmax = 0; for (let q = 0; q < N * N; q++) wmax = Math.max(wmax, W[q]);
    if (wmax > 1e-9) for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const w = W[i * N + j]; if (w <= 1e-9) continue;
      const f = w / wmax;
      svgEl("line", { x1: pos[i][0], y1: pos[i][1], x2: pos[j][0], y2: pos[j][1], stroke: P.edge, "stroke-width": (1 + 3.5 * f).toFixed(2), opacity: (0.15 + 0.8 * f).toFixed(3), "stroke-linecap": "round" }, gW);
    }
  }
  return { update, svg };
}
