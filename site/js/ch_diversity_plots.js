// Chapter "diversity": three small SVG plots that charts.js does not provide.
//   curvesPanel  a bundle of member curves u_i(z) with a thick prototype; hover picks the nearest curve
//   planePlot    the (s, c) plane of linear response functions for the s ± c example (equal axes)
//   logBars      grouped bars on a log axis with a dashed reference line
import { svgEl, scale, niceTicks, logTicks } from "./charts.js";
import { h, fmt, showTip, hideTip, tipRows, legend } from "./ui.js";
import { P } from "./palette.js";

const txt = (parent, x, y, s, attrs = {}) => { const t = svgEl("text", { x, y, ...attrs }, parent); t.textContent = s; return t; };
function sized(el, draw) {
  let w = 0;
  const ro = new ResizeObserver(() => { const cw = Math.floor(el.clientWidth); if (cw > 0 && cw !== w) { w = cw; draw(cw); } });
  ro.observe(el);
  return { redraw() { const cw = Math.floor(el.clientWidth); if (cw > 0) { w = cw; draw(cw); } } };
}

// ---------------------------------------------------------------- curves
// update({x, curves: number[][], colors: css[], proto: number[] | null, yrange, title?, note?, tooltip?(i)})
export function curvesPanel(el, { height = 230, xlabel = "z", ylabel = "" } = {}) {
  el.classList.add("chart");
  const head = h("div", { class: "chart-title" }), host = h("div"), note = h("div", { class: "u-help", style: { minHeight: "1.3em" } });
  el.replaceChildren(head, host, note);
  let st = { curves: [], colors: [], proto: null, x: [], yrange: [-1, 1] }, hov = null;
  function draw(W) {
    host.replaceChildren();
    const Hh = height, m = { l: ylabel ? 46 : 36, r: 8, t: 8, b: 34 };
    const svg = svgEl("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": st.title || "response curves" }, host);
    const xs = st.x, [y0, y1] = st.yrange;
    if (!xs.length) return;
    const X = scale("linear", xs[0], xs[xs.length - 1], m.l, W - m.r), Y = scale("linear", y0, y1, Hh - m.b, m.t);
    const g = svgEl("g", { class: "grid" }, svg);
    const yt = niceTicks(y0, y1, 4), xt = niceTicks(xs[0], xs[xs.length - 1], Math.max(3, Math.round(W / 70)));
    yt.forEach(v => { svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); txt(svg, m.l - 5, Y(v) + 3.5, fmt(v), { "text-anchor": "end" }); });
    const ax = svgEl("g", { class: "axis" }, svg);
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Hh - m.b + .5, y2: Hh - m.b + .5 }, ax);
    xt.forEach(v => { svgEl("line", { x1: X(v), x2: X(v), y1: Hh - m.b, y2: Hh - m.b + 4 }, ax); txt(svg, X(v), Hh - m.b + 15, fmt(v), { "text-anchor": "middle" }); });
    if (y0 < 0 && y1 > 0) svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: "zero" }, svg);
    txt(svg, (m.l + W - m.r) / 2, Hh - 4, xlabel, { "text-anchor": "middle", class: "ttl" });
    if (ylabel) txt(svg, 0, 0, ylabel, { "text-anchor": "middle", class: "ttl", transform: `translate(11,${(m.t + Hh - m.b) / 2}) rotate(-90)` });
    const path = ys => ys.map((v, k) => (k ? "L" : "M") + X(xs[k]).toFixed(1) + "," + Y(Math.max(y0 - (y1 - y0), Math.min(y1 + (y1 - y0), v))).toFixed(1)).join("");
    const cid = st._cid || (st._cid = "dvc" + Math.random().toString(36).slice(2));
    const clip = svgEl("clipPath", { id: cid }, svgEl("defs", {}, svg));
    svgEl("rect", { x: m.l, y: m.t - 2, width: W - m.l - m.r, height: Hh - m.t - m.b + 4 }, clip);
    const plot = svgEl("g", { "clip-path": `url(#${cid})` }, svg);
    st.curves.forEach((ys, i) => svgEl("path", { d: path(ys), fill: "none", stroke: st.colors[i] || P.mute, "stroke-width": i === hov ? 2.2 : 1.1, opacity: hov == null || i === hov ? .9 : .3 }, plot));
    if (st.proto) svgEl("path", { d: path(st.proto), fill: "none", stroke: P.fg, "stroke-width": 2.6, opacity: .85, "stroke-dasharray": hov == null ? null : "5 3" }, plot);
    if (!st.curves.length && st.empty) txt(svg, (m.l + W - m.r) / 2, (m.t + Hh - m.b) / 2, st.empty, { "text-anchor": "middle", fill: P.mute });
    const hit = svgEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: Hh - m.t - m.b, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", ev => {
      const r = svg.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
      const xv = X.inv(px); let k = 0; xs.forEach((x, q) => { if (Math.abs(x - xv) < Math.abs(xs[k] - xv)) k = q; });
      let best = null, bd = 14;
      st.curves.forEach((ys, i) => { const d = Math.abs(Y(ys[k]) - py); if (d < bd) { bd = d; best = i; } });
      if (best !== hov) { hov = best; draw(W); }
      if (best == null) { hideTip(); return; }
      showTip(ev.clientX, ev.clientY, st.tooltip ? st.tooltip(best, k) : tipRows(`curve ${best}`, []));
    });
    hit.addEventListener("pointerleave", () => { hideTip(); if (hov != null) { hov = null; draw(W); } });
  }
  const sz = sized(host, draw);
  return {
    update(d) {
      st = { ...st, ...d }; hov = null;
      head.innerHTML = st.title || ""; note.innerHTML = st.note || "";
      sz.redraw();
    },
  };
}

// ---------------------------------------------------------------- the (s, c) plane
// Coordinates: a linear response f = x s + y c is drawn at (x, σ_c y), so the drawing is isometric to the
// function-space geometry <f, f'> = x x' + σ_c² y y'. update({sc2, neurons: [[x, y]], groups: int[], proto: [[x,y]]×2,
// psi: [[x, y]]×2, q: [q1, q2], N, colors, larger})
export function planePlot(el, { height = 360 } = {}) {
  el.classList.add("chart");
  const host = h("div"); el.replaceChildren(host);
  let st = null;
  function draw(W) {
    host.replaceChildren();
    if (!st) return;
    const Hh = Math.min(height, Math.max(260, W * 0.9)), m = 14;
    const svg = svgEl("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "plane of response functions" }, host);
    const sc = Math.sqrt(st.sc2), R = 1.62;
    const side = Math.min(W - 2 * m, Hh - 2 * m), cx = W / 2, cy = Hh / 2, k = side / (2 * R);
    const X = x => cx + k * x, Y = y => cy - k * y;
    const P2 = ([x, y]) => [x, sc * y];               // isometric coordinates
    // grid circles: unit norm
    svgEl("circle", { cx, cy, r: k, fill: "none", stroke: P.line, "stroke-dasharray": "2 3" }, svg);
    txt(svg, X(Math.SQRT1_2) + 4, Y(Math.SQRT1_2) - 4, "‖f‖ = 1", { fill: P.mute, "font-size": 10.5 });
    // axes: s and c (Σ_g's modes)
    svgEl("line", { x1: X(-R), x2: X(R), y1: cy, y2: cy, stroke: P.mute2 }, svg);
    svgEl("line", { x1: cx, x2: cx, y1: Y(-R), y2: Y(R), stroke: P.mute2 }, svg);
    txt(svg, X(R) - 2, cy + 15, "s (Σ mode)", { "text-anchor": "end", fill: P.ink2, "font-size": 11 });
    txt(svg, cx - 6, Y(R) + 10, "c (Σ mode)", { "text-anchor": "end", fill: P.ink2, "font-size": 11 });
    // population ellipse: semi-axes sqrt(q_a / N) along psi_a (drawn from the exact operator)
    if (st.ell) {
      const [a1, a2] = st.ell.axes, ang = st.ell.angle;
      svgEl("ellipse", { cx, cy, rx: k * a1, ry: k * a2, transform: `rotate(${-ang * 180 / Math.PI} ${cx} ${cy})`, fill: P.fg, "fill-opacity": .05, stroke: P.mute, "stroke-opacity": .6, "stroke-width": 1.2 }, svg);
    }
    // psi lines (sign-free: full lines through the origin)
    (st.psi || []).forEach((p, a) => {
      const v = P2(p), n = Math.hypot(v[0], v[1]) || 1, u = [v[0] / n, v[1] / n], L = R * 0.98;
      svgEl("line", { x1: X(-L * u[0]), y1: Y(-L * u[1]), x2: X(L * u[0]), y2: Y(L * u[1]), stroke: P.fg, "stroke-width": a === 0 ? 2 : 1.3, "stroke-dasharray": a === 0 ? null : "6 4" }, svg);
      const sgn = u[0] < -1e-9 || (Math.abs(u[0]) <= 1e-9 && u[1] < 0) ? -1 : 1, d = [sgn * u[0], sgn * u[1]], at = a === 0 ? 0.42 : 0.72;
      // offset to the upper-left of the line's direction, in pixels
      const nx = -d[1], ny = d[0];
      txt(svg, X(at * R * d[0]) + 13 * nx, Y(at * R * d[1]) - 13 * ny + 4, a === 0 ? "ψ₁" : "ψ₂", { fill: P.fg, "font-weight": 600, "font-size": 13, "text-anchor": "middle" });
    });
    // prototypes: rays from the origin
    (st.proto || []).forEach((p, kk) => {
      const v = P2(p);
      svgEl("line", { x1: cx, y1: cy, x2: X(v[0]), y2: Y(v[1]), stroke: st.colors[kk], "stroke-width": 1.6 }, svg);
    });
    // neurons
    const counts = new Map();
    st.neurons.forEach((p, i) => { const v = P2(p), key = `${v[0].toFixed(3)},${v[1].toFixed(3)},${st.groups[i]}`; counts.set(key, (counts.get(key) || 0) + 1); });
    st.neurons.forEach((p, i) => { const v = P2(p); svgEl("circle", { cx: X(v[0]), cy: Y(v[1]), r: 4.2, fill: st.colors[st.groups[i]], stroke: P.surface, "stroke-width": 1 }, svg); });
    (st.proto || []).forEach((p, kk) => {
      const v = P2(p);
      const lab = kk === 0 ? `p_A, n_A = ${st.n[0]}` : `p_B, n_B = ${st.n[1]}`, right = X(v[0]) + 10 + 78 <= W;
      txt(svg, right ? X(v[0]) + 9 : X(v[0]) + 4, Y(v[1]) + (kk === 0 ? (right ? 4 : -10) : (right ? 4 : 18)), lab, { fill: st.colors[kk], "font-weight": 600, "font-size": 11.5, "text-anchor": right ? "start" : "end" });
    });
  }
  const sz = sized(host, draw);
  return { update(d) { st = d; sz.redraw(); } };
}

// ---------------------------------------------------------------- grouped bars on a log axis
// update({labels, series: [{name, values, color}], ref: {y, label}, yrange: [lo, hi], ylabel, tooltip(i, s)})
export function logBars(el, { height = 210 } = {}) {
  el.classList.add("chart");
  const host = h("div"); el.replaceChildren(host);
  let st = null;
  function draw(W) {
    host.replaceChildren();
    if (!st || !st.labels.length) { if (st && st.empty) host.append(h("div", { class: "u-help", style: { padding: "40px 0", textAlign: "center" } }, st.empty)); return; }
    const Hh = height, m = { l: st.ylabel ? 50 : 40, r: 8, t: 8, b: 34 };
    const svg = svgEl("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}` }, host);
    const [lo, hi] = st.yrange, Y = scale("log", lo, hi, Hh - m.b, m.t);
    const g = svgEl("g", { class: "grid" }, svg);
    logTicks(lo, hi).forEach(v => { svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); txt(svg, m.l - 5, Y(v) + 3.5, fmt(v), { "text-anchor": "end" }); });
    const L = st.labels.length, ns = st.series.length, span = W - m.l - m.r, bw = Math.min(span / L, 130), x0 = m.l + (span - bw * L) / 2, gap = Math.min(16, bw * .25), sw = (bw - gap) / ns;
    st.series.forEach((s, si) => s.values.forEach((v, i) => {
      if (!(v > 0)) return;
      const x = x0 + i * bw + gap / 2 + si * sw, y = Y(Math.min(hi, Math.max(lo, v)));
      const r = svgEl("rect", { x: x.toFixed(1), y: y.toFixed(1), width: Math.max(1, sw - 1).toFixed(1), height: Math.max(.5, Hh - m.b - y).toFixed(1), fill: s.color, rx: 1.5 }, svg);
      r.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, st.tooltip ? st.tooltip(i, si) : tipRows(st.labels[i], [{ name: s.name, value: fmt(v), color: s.color, shape: "square" }])));
      r.addEventListener("pointerleave", hideTip);
    }));
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Hh - m.b + .5, y2: Hh - m.b + .5, stroke: P.line }, svg);
    if (st.ref) { svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(st.ref.y), y2: Y(st.ref.y), stroke: P.fg, "stroke-dasharray": "4 3" }, svg); txt(svg, W - m.r - 2, Y(st.ref.y) - 4, st.ref.label, { "text-anchor": "end", fill: P.fg }); }
    const every = Math.max(1, Math.ceil(L / Math.max(1, Math.floor((W - m.l - m.r) / 34))));
    st.labels.forEach((s, i) => { if (i % every === 0) txt(svg, x0 + (i + .5) * bw, Hh - m.b + 13, s, { "text-anchor": "middle" }); });
    if (st.xlabel) txt(svg, (m.l + W - m.r) / 2, Hh - 4, st.xlabel, { "text-anchor": "middle", class: "ttl" });
    if (st.ylabel) txt(svg, 0, 0, st.ylabel, { "text-anchor": "middle", class: "ttl", transform: `translate(11,${(m.t + Hh - m.b) / 2}) rotate(-90)` });
    const lg = legend({ items: st.series.map(s => ({ label: s.name, color: s.color, shape: "square" })) });
    lg.el.classList.add("chart-legend"); lg.el.style.marginLeft = m.l + "px"; host.append(lg.el);
  }
  const sz = sized(host, draw);
  return { update(d) { st = d; sz.redraw(); } };
}
