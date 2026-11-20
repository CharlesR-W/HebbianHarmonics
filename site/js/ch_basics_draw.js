// Chapter "basics": drawing helpers. SVG pictures of the small network (neuron space, seeded random layout, the same
// picture as the talk), the triangle of mixtures (input space), pies, a scrolling voltage trace, and a small line plot.
import * as ui from "./ui.js";
import { svgEl, scale, niceTicks } from "./charts.js";
import { P } from "./palette.js";
import { randomLayout } from "./netview.js";
import { EDGES, NS } from "./ch_basics_model.js";

export const SUBS = ["₁", "₂", "₃"];
export const inCol = b => P.cat[b];
export const txt = (parent, x, y, s, attrs = {}) => {
  const t = svgEl("text", { x, y, "font-size": 14, fill: P.fg, style: "font-family:var(--sans)", ...attrs }, parent);
  t.textContent = s; return t;
};

/** An SVG that scales with its container (viewBox w x h). */
export function svgRoot(parent, w, h, aria, extraStyle = "") {
  const s = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", role: "img", "aria-label": aria,
    style: `display:block;max-height:${Math.round(h * 1.15)}px;touch-action:manipulation;user-select:none;${extraStyle}` });
  parent.append(s); return s;
}
/** A panel with a space badge in its top-left corner. */
export function panel(parent, kind, text) {
  const p = ui.h("div", { class: "bx-panel" }, ui.h("span", { class: "space-badge bx-badge", "data-space": kind, text }));
  parent.append(p); return p;
}

// ------------------------------------------------------------ the small network, neuron space
// Same layout as the talk's embedding slide (viewBox 560 x 400, box and seed below), so the reader sees one network.
export const NET_VB = [560, 400];
const NET_BOX = { x0: 40, x1: 520, y0: 60, y1: 376 }, NET_SEED = 3;
export function netLayout() {
  const w = NET_BOX.x1 - NET_BOX.x0, h = NET_BOX.y1 - NET_BOX.y0;
  const r = randomLayout(3 + NS, { seed: NET_SEED, aspect: w / h });
  const at = k => [NET_BOX.x0 + r[2 * k] * w, NET_BOX.y0 + r[2 * k + 1] * h];
  return { bnd: [0, 1, 2].map(at), int: Array.from({ length: NS }, (_, i) => at(3 + i)) };
}
export const posOf = (L, k) => (typeof k === "string" ? L.bnd[+k[1]] : L.int[k]);

/** Synapses as ink lines, width rising with conductance. Returns the line elements in EDGES order. */
export function drawEdges(g, L, { opacity = 0.42, widthPerW = 4 } = {}) {
  return EDGES.map(([a, b, w]) => {
    const [x1, y1] = posOf(L, a), [x2, y2] = posOf(L, b);
    return svgEl("line", { x1, y1, x2, y2, stroke: P.edge, "stroke-width": Math.min(9, widthPerW * w), "stroke-linecap": "round", opacity }, g);
  });
}
/** Input squares (filled with `fill` or their hue, outlined in their hue) with labels g1..g3 outside the network. */
export function drawInputs(g, L, { fill = null, size = 34 } = {}) {
  return L.bnd.map(([x, y], b) => {
    const r = svgEl("rect", { x: x - size / 2, y: y - size / 2, width: size, height: size, rx: 5, fill: fill ? fill(b) : inCol(b), stroke: inCol(b), "stroke-width": 3.5 }, g);
    const right = x < NET_VB[0] / 2;
    txt(g, x + (right ? 1 : -1) * (size / 2 + 7), y + 6, `g${SUBS[b]}`, { "text-anchor": right ? "start" : "end", "font-size": 17, "font-weight": 600, fill: P.ink2 });
    return r;
  });
}

export function pie(parent, cx, cy, r, h, { stroke = P.fg, sw = 1 } = {}) {
  let a0 = -Math.PI / 2;
  const g = svgEl("g", {}, parent);
  h.forEach((f, b) => {
    if (f < 1e-4) return;
    if (f > 0.9999) { svgEl("circle", { cx, cy, r, fill: inCol(b) }, g); return; }
    const a1 = a0 + 2 * Math.PI * f, large = a1 - a0 > Math.PI ? 1 : 0;
    svgEl("path", { d: `M${cx},${cy} L${cx + r * Math.cos(a0)},${cy + r * Math.sin(a0)} A${r},${r} 0 ${large} 1 ${cx + r * Math.cos(a1)},${cy + r * Math.sin(a1)} Z`, fill: inCol(b) }, g);
    a0 = a1;
  });
  svgEl("circle", { cx, cy, r, fill: "none", stroke, "stroke-width": sw }, g);
  return g;
}

// ------------------------------------------------------------ the triangle, input space
/** Draw the triangle of mixtures in a w x h viewBox. Corners: g1 top, g2 bottom left, g3 bottom right. Each input
 *  square sits just outside the corner it defines (so neurons on a corner stay visible), labelled beside it.
 *  Returns {at(h) -> [x, y], inv(x, y) -> h (clamped into the triangle), C (corners)}. */
export function triangle(g, w, h, { labels = true, squares = true, centre = false, sq = 20, font = 16, top = 0, margin = null } = {}) {
  // room around the triangle: the squares (outside the corners) and the labels beside them
  const mx = margin ?? (squares ? sq * 1.3 + (labels ? font * 1.6 + 8 : 4) : 8), my = squares ? sq * 1.4 + 2 : 8;
  const side = Math.min(w - 2 * mx, (h - top - 2 * my) * 2 / Math.sqrt(3));
  const th = side * Math.sqrt(3) / 2, cx = w / 2, y0 = top + (h - top - th) / 2;
  const C = [[cx, y0], [cx - side / 2, y0 + th], [cx + side / 2, y0 + th]];
  svgEl("path", { d: `M${C[0]}L${C[1]}L${C[2]}Z`, fill: P.surface, stroke: P.line, "stroke-width": 1.5 }, g);
  const at = hv => [hv[0] * C[0][0] + hv[1] * C[1][0] + hv[2] * C[2][0], hv[0] * C[0][1] + hv[1] * C[1][1] + hv[2] * C[2][1]];
  if (centre) {
    const [mx, my] = at([1 / 3, 1 / 3, 1 / 3]);
    svgEl("path", { d: `M${mx - 5},${my}H${mx + 5}M${mx},${my - 5}V${my + 5}`, stroke: P.mute2, "stroke-width": 1.2 }, g);
  }
  if (squares) {
    const [gx, gy] = at([1 / 3, 1 / 3, 1 / 3]);
    C.forEach(([x, y], b) => {
      const dx = x - gx, dy = y - gy, n = Math.hypot(dx, dy), off = sq * 0.95;
      const sx = x + dx / n * off, sy = y + dy / n * off;
      svgEl("rect", { x: sx - sq / 2, y: sy - sq / 2, width: sq, height: sq, rx: 3.5, fill: inCol(b), stroke: inCol(b) }, g);
      if (labels) txt(g, b === 1 ? sx - sq / 2 - 5 : sx + sq / 2 + 5, sy + font * 0.36, `g${SUBS[b]}`, { "text-anchor": b === 1 ? "end" : "start", "font-size": font, "font-weight": 600, fill: P.ink2 });
    });
  }
  // inverse map (barycentric coordinates), clamped to the triangle
  const inv = (x, y) => {
    const [[x1, y1], [x2, y2], [x3, y3]] = C, det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    let l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det, l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det, l3 = 1 - l1 - l2;
    let v = [l1, l2, l3].map(t => Math.max(0, t)); const s = v[0] + v[1] + v[2]; v = v.map(t => t / s);
    return v;
  };
  return { at, inv, C, side };
}

/** Pointer position in an SVG's viewBox coordinates. */
export function svgPoint(svg, ev) {
  const p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
  return p.matrixTransform(svg.getScreenCTM().inverse());
}

// ------------------------------------------------------------ a scrolling trace (figure 1)
/** traceChart(el, {series: [{name, color, width?}], yr, span (s), height}) -> {push(t, values), redraw()} */
export function traceChart(el, o) {
  const W = 420, H = o.height || 300, m = { l: 40, r: 14, t: 12, b: 34 };
  const svg = svgRoot(el, W, H, "voltages over time");
  const X = scale("linear", 0, o.span, m.l, W - m.r), Y = scale("linear", o.yr[0], o.yr[1], H - m.b, m.t);
  const grid = svgEl("g", { class: "grid" }, svg);
  niceTicks(o.yr[0], o.yr[1], 5).forEach(v => {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: v === 0 ? P.mute2 : P.line, "stroke-width": 1 }, grid);
    txt(svg, m.l - 6, Y(v) + 4, ui.fmt(v, 2), { "text-anchor": "end", "font-size": 12, fill: P.mute });
  });
  txt(svg, (m.l + W - m.r) / 2, H - 8, `time (the last ${o.span} seconds; newest on the right)`, { "text-anchor": "middle", "font-size": 12.5, fill: P.mute });
  txt(svg, 0, 0, "voltage", { "text-anchor": "middle", "font-size": 12.5, fill: P.mute, transform: `translate(12,${(m.t + H - m.b) / 2}) rotate(-90)` });
  const paths = o.series.map(s => svgEl("path", { fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round", opacity: s.opacity ?? 1 }, svg));
  let buf = [];
  function push(t, vals) {
    buf.push([t, vals]);
    while (buf.length > 2 && buf[1][0] < t - o.span) buf.shift();
  }
  function redraw() {
    if (!buf.length) return;
    const tEnd = buf[buf.length - 1][0], t0 = tEnd - o.span;
    paths.forEach((p, k) => {
      let d = "";
      for (const [t, v] of buf) { if (t < t0) continue; d += (d ? "L" : "M") + X(t - t0).toFixed(1) + "," + Y(v[k]).toFixed(1); }
      p.setAttribute("d", d);
    });
  }
  return { push, redraw, clear: () => { buf = []; }, setSeries(k, attrs) { for (const a in attrs) paths[k].setAttribute(a, attrs[a]); } };
}

// ------------------------------------------------------------ a small line plot (figure 6)
/** linePlot(el, {height, xlabel, ylabel, series: [{x, y, color, width?, dash?}], xr, yr}) */
export function linePlot(el, o) {
  const W = 440, H = o.height || 300, m = { l: 52, r: 14, t: 14, b: 40 };
  const svg = svgRoot(el, W, H, o.ylabel || "plot");
  const X = scale("linear", o.xr[0], o.xr[1], m.l, W - m.r), Y = scale("linear", o.yr[0], o.yr[1], H - m.b, m.t);
  niceTicks(o.yr[0], o.yr[1], 5).forEach(v => {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: v === 0 ? P.mute2 : P.line }, svg);
    txt(svg, m.l - 6, Y(v) + 4, ui.fmt(v, 2), { "text-anchor": "end", "font-size": 12, fill: P.mute });
  });
  niceTicks(o.xr[0], o.xr[1], 5).forEach(v => {
    svgEl("line", { x1: X(v), x2: X(v), y1: H - m.b, y2: H - m.b + 4, stroke: P.mute2 }, svg);
    txt(svg, X(v), H - m.b + 17, ui.fmt(v, 2), { "text-anchor": "middle", "font-size": 12, fill: P.mute });
  });
  svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b + 0.5, y2: H - m.b + 0.5, stroke: P.mute2 }, svg);
  if (o.xlabel) txt(svg, (m.l + W - m.r) / 2, H - 6, o.xlabel, { "text-anchor": "middle", "font-size": 12.5, fill: P.mute });
  if (o.ylabel) txt(svg, 0, 0, o.ylabel, { "text-anchor": "middle", "font-size": 12.5, fill: P.mute, transform: `translate(12,${(m.t + H - m.b) / 2}) rotate(-90)` });
  for (const s of o.series) {
    let d = ""; s.x.forEach((xv, k) => { d += (k ? "L" : "M") + X(xv).toFixed(1) + "," + Y(s.y[k]).toFixed(1); });
    svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-dasharray": s.dash || null, "stroke-linejoin": "round" }, svg);
    if (s.label) { const k = s.labelAt ?? s.x.length - 1; txt(svg, X(s.x[k]) - 4, Y(s.y[k]) + (s.labelDy ?? -8), s.label, { "text-anchor": "end", "font-size": 13, "font-weight": 600, fill: s.color }); }
  }
  return { svg, X, Y };
}
