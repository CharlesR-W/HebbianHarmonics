// Chapter "fusion": small purpose-built charts the shared charts.js does not provide.
//   graphView   points (response functions) with live edges, input squares, an edge-range scale bar (canvas)
//   wdScatter   w_ij against D^2_ij with the fixed-point line (canvas)
//   svgPlot     axes + a render callback, for the two-neuron drift and bifurcation panels (SVG)
// They follow the look of charts.js (fonts, grid, axis colours) and redraw on resize.
import { svgEl, scale, niceTicks, logTicks, hidpi } from "./charts.js";
import { P, rgbToCss } from "./palette.js";
import * as ui from "./ui.js";

const FONT = "11px Inter, system-ui, sans-serif", FONT_B = "560 11.5px Inter, system-ui, sans-serif";

function onWidth(el, fn) {
  let w = 0;
  const ro = new ResizeObserver(() => { const cw = Math.floor(el.clientWidth); if (cw > 0 && cw !== w) { w = cw; fn(cw); } });
  ro.observe(el);
  return () => Math.floor(el.clientWidth);
}

// ---------------------------------------------------------------- graph view
export function graphView(el, { height = 460, tooltip, onHover } = {}) {
  el.classList.add("chart");
  const wrap = ui.h("div", { style: { position: "relative" } });
  const cv = ui.h("canvas", { style: { touchAction: "pan-y", cursor: "default" } });
  const corner = ui.h("div", { class: "u-help", style: { position: "absolute", left: "8px", top: "6px", maxWidth: "62%", pointerEvents: "none", background: "rgba(255,255,255,.82)", padding: "2px 5px", borderRadius: "3px" } });
  wrap.append(cv, corner); el.replaceChildren(wrap);
  let st = null, W = 0, box = null, proj = null, hl = [], hlPair = null, hov = -1;
  const Hh = () => Math.max(300, Math.min(height, Math.round(W * 0.92)));

  function frameBox(pts) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, r = Math.max(x1 - x0, y1 - y0, 1e-6) / 2;
    return { cx, cy, r };
  }
  function draw() {
    W = Math.floor(el.clientWidth); if (!W || !st) return;
    const H = Hh(), g = hidpi(cv, W, H);
    g.clearRect(0, 0, W, H);
    const N = st.xy.length / 2, pts = [];
    for (let i = 0; i < N; i++) pts.push([st.xy[2 * i], st.xy[2 * i + 1]]);
    for (const a of st.anchors) pts.push(a);
    // framing: refit when the content leaves the box or shrinks to under half of it (no jitter in between)
    const fb = frameBox(pts);
    const outside = box && (Math.hypot(fb.cx - box.cx, fb.cy - box.cy) + fb.r > box.r || fb.r < 0.5 * box.r);
    if (!box || st.refit || box.mode !== st.mode || outside) {
      box = { ...fb, r: fb.r * 1.08, mode: st.mode }; st.refit = false;
    }
    const pad = { l: 34, r: 34, t: Math.max(38, corner.hidden ? 0 : corner.offsetHeight + 46), b: 40 };
    const s = Math.min((W - pad.l - pad.r) / (2 * box.r), (H - pad.t - pad.b) / (2 * box.r));
    const ox = (pad.l + W - pad.r) / 2, oy = (pad.t + H - pad.b) / 2;
    proj = (x, y) => [ox + (x - box.cx) * s, oy - (y - box.cy) * s];
    // input triangle
    const A = st.anchors.map(a => proj(a[0], a[1]));
    g.strokeStyle = "rgba(141,140,134,0.55)"; g.lineWidth = 1; g.setLineDash(st.mode === "mds" ? [4, 4] : []);
    g.beginPath(); A.forEach(([x, y], k) => (k ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.stroke(); g.setLineDash([]);
    // edges
    const P2 = []; for (let i = 0; i < N; i++) P2.push(proj(st.xy[2 * i], st.xy[2 * i + 1]));
    g.lineCap = "round";
    for (const e of st.edges) {
      g.strokeStyle = rgbToCss([22, 22, 26], Math.min(1, 0.08 + 0.72 * e.a)); g.lineWidth = 1.1;
      g.beginPath(); g.moveTo(P2[e.i][0], P2[e.i][1]); g.lineTo(P2[e.j][0], P2[e.j][1]); g.stroke();
    }
    if (hlPair) {
      g.strokeStyle = P.accent; g.lineWidth = 2;
      g.beginPath(); g.moveTo(P2[hlPair[0]][0], P2[hlPair[0]][1]); g.lineTo(P2[hlPair[1]][0], P2[hlPair[1]][1]); g.stroke();
    }
    // points: larger first is irrelevant here; singletons (grey) under coloured ones
    const order = [...Array(N).keys()].sort((a, b) => (st.colors[a] === P.other) - (st.colors[b] === P.other)).reverse();
    for (const i of order) {
      const [x, y] = P2[i], r = st.sizes ? st.sizes[i] : 4.5;
      g.beginPath(); g.arc(x, y, r + 1.2, 0, 2 * Math.PI); g.fillStyle = P.surface; g.fill();
      g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fillStyle = st.colors[i]; g.fill();
    }
    for (const i of hl) if (i >= 0 && i < N) { const [x, y] = P2[i]; g.beginPath(); g.arc(x, y, 8, 0, 2 * Math.PI); g.strokeStyle = P.fg; g.lineWidth = 1.5; g.stroke(); }
    // inputs: squares with an ink outline, labels pushed away from the centroid
    const cx = A.reduce((s, a) => s + a[0], 0) / A.length, cy = A.reduce((s, a) => s + a[1], 0) / A.length;
    g.font = "600 12.5px Inter, system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
    A.forEach(([x, y], b) => {
      g.fillStyle = P.surface; g.fillRect(x - 6.5, y - 6.5, 13, 13);
      g.fillStyle = P.boundary; g.fillRect(x - 5, y - 5, 10, 10);
      const dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy) || 1;
      g.fillStyle = P.fg; g.fillText(st.anchorLabels[b], x + dx / d * 17, y + dy / d * 17);
    });
    g.textBaseline = "alphabetic";
    // scale bar: the edge range sqrt(2b) in D units (only when the drawing is isometric in D)
    if (st.scaleBar) {
      const L = st.scaleBar.len * s, x0 = W - pad.r - Math.min(L, W * 0.6), y0 = H - 14;
      g.strokeStyle = P.fg; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x0 + Math.min(L, W * 0.6), y0);
      g.moveTo(x0, y0 - 4); g.lineTo(x0, y0 + 4); g.moveTo(x0 + Math.min(L, W * 0.6), y0 - 4); g.lineTo(x0 + Math.min(L, W * 0.6), y0 + 4); g.stroke();
      g.font = FONT; g.fillStyle = P.ink2; g.textAlign = "right"; g.fillText(st.scaleBar.label + (L > W * 0.6 ? " (clipped)" : ""), x0 - 6, y0 + 4);
    }
    st.P2 = P2; st.A = A;
  }
  function pick(ev) {
    if (!st || !st.P2) return null;
    const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = null, bd = 12 * 12;
    st.P2.forEach(([px, py], i) => { const d = (px - x) ** 2 + (py - y) ** 2; if (d < bd) { bd = d; best = { kind: "point", index: i }; } });
    st.A.forEach(([px, py], b) => { const d = (px - x) ** 2 + (py - y) ** 2; if (d < bd) { bd = d; best = { kind: "anchor", index: b }; } });
    return best;
  }
  cv.addEventListener("pointermove", ev => {
    const p = pick(ev);
    if (!p) { ui.hideTip(); if (hov !== -1) { hov = -1; onHover?.(null); } return; }
    if (tooltip) { const c = tooltip(p.index, p.kind); if (c) ui.showTip(ev.clientX, ev.clientY, c); }
    const id = p.kind === "point" ? p.index : -1;
    if (id !== hov) { hov = id; onHover?.(id >= 0 ? id : null); }
  });
  cv.addEventListener("pointerleave", () => { ui.hideTip(); if (hov !== -1) { hov = -1; onHover?.(null); } });
  onWidth(el, () => draw());
  return {
    el,
    // {xy (2N), anchors [[x,y]], anchorLabels, colors, sizes?, edges [{i,j,a}], mode, scaleBar?, refit?}
    set(s) { st = s; draw(); },
    setCorner(html) { corner.innerHTML = html; corner.hidden = !html; ui.renderMath(corner); },
    highlight(list, pair = null) { hl = list || []; hlPair = pair; draw(); },
    refit() { box = null; },
  };
}


// ---------------------------------------------------------------- w against D^2 with the fixed-point line
export function wdScatter(el, { height = 250, tooltip, onHover } = {}) {
  el.classList.add("chart");
  const cv = ui.h("canvas", { style: { touchAction: "pan-y" } });
  el.replaceChildren(cv);
  let st = null, pts = [], hov = null;
  function draw() {
    const W = Math.floor(el.clientWidth); if (!W || !st) return;
    const H = height, g = hidpi(cv, W, H), m = { l: 50, r: 12, t: 12, b: 38 };
    g.clearRect(0, 0, W, H);
    const N = st.N, b = st.b, al = st.alpha, x1 = 4 * b, y1 = 1.12 * Math.max(b / al, st.wmax || 0);
    const X = scale("linear", 0, x1, m.l, W - m.r), Y = scale("linear", 0, y1, H - m.b, m.t);
    g.font = FONT; g.lineWidth = 1;
    const xt = niceTicks(0, x1, Math.max(3, Math.round(W / 90))), yt = niceTicks(0, y1, Math.max(3, Math.round(H / 55)));
    g.strokeStyle = "#efeee9"; g.fillStyle = P.mute;
    g.textAlign = "right"; yt.forEach(v => { g.beginPath(); g.moveTo(m.l, Math.round(Y(v)) + .5); g.lineTo(W - m.r, Math.round(Y(v)) + .5); g.stroke(); g.fillText(ui.fmt(v), m.l - 6, Y(v) + 3.5); });
    g.textAlign = "center"; xt.forEach(v => g.fillText(ui.fmt(v), X(v), H - m.b + 14));
    g.strokeStyle = "#c9c8c1"; g.beginPath(); g.moveTo(m.l, H - m.b + .5); g.lineTo(W - m.r, H - m.b + .5); g.stroke();
    // 2b reference and theory line
    g.setLineDash([3, 3]); g.strokeStyle = P.mute2; g.beginPath(); g.moveTo(X(2 * b), m.t); g.lineTo(X(2 * b), H - m.b); g.stroke(); g.setLineDash([]);
    g.fillStyle = P.mute; g.textAlign = "left"; g.fillText("D² = 2b", X(2 * b) + 4, m.t + 10);
    g.strokeStyle = P.fg; g.lineWidth = 1.4; g.beginPath(); g.moveTo(X(0), Y(b / al)); g.lineTo(X(2 * b), Y(0)); g.lineTo(X(x1), Y(0)); g.stroke();
    // pairs
    pts = []; let off = 0;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const d = st.D2[i * N + j], w = st.W[i * N + j];
      if (d > x1) { off++; continue; }
      pts.push([X(d), Y(Math.min(w, y1)), i, j, w > 1e-9]);
    }
    for (const [x, y, i, j, live] of pts) if (!live) { g.fillStyle = P.other; g.fillRect(x - 1, y - 4, 2, 5); }
    for (const [x, y, i, j, live] of pts) if (live) {
      g.beginPath(); g.arc(x, y, 3.2, 0, 2 * Math.PI); g.fillStyle = st.colorOf(i); g.fill(); g.strokeStyle = P.surface; g.lineWidth = 0.8; g.stroke();
    }
    if (hov) { g.beginPath(); g.arc(hov[0], hov[1], 6, 0, 2 * Math.PI); g.strokeStyle = P.fg; g.lineWidth = 1.5; g.stroke(); }
    g.font = FONT_B; g.fillStyle = P.ink2; g.textAlign = "center";
    g.fillText("D²ᵢⱼ (training bank)", (m.l + W - m.r) / 2, H - 6);
    g.save(); g.translate(12, (m.t + H - m.b) / 2); g.rotate(-Math.PI / 2); g.fillText("wᵢⱼ", 0, 0); g.restore();
    g.font = FONT; g.fillStyle = P.mute; g.textAlign = "right";
    if (off) g.fillText(W < 420 ? `${off} pairs off scale →` : `${off} pair${off > 1 ? "s" : ""} beyond D² = 4b not shown →`, W - m.r, m.t + 26);
  }
  cv.addEventListener("pointermove", ev => {
    const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = null, bd = 10 * 10;
    for (const p of pts) { const d = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (d < bd) { bd = d; best = p; } }
    if (!best) { ui.hideTip(); if (hov) { hov = null; draw(); onHover?.(null); } return; }
    if (tooltip) ui.showTip(ev.clientX, ev.clientY, tooltip(best[2], best[3]));
    if (hov !== best) { hov = best; draw(); onHover?.([best[2], best[3]]); }
  });
  cv.addEventListener("pointerleave", () => { ui.hideTip(); if (hov) { hov = null; draw(); onHover?.(null); } });
  onWidth(el, () => draw());
  return { el, set(s) { st = s; hov = null; draw(); } };
}

// ---------------------------------------------------------------- generic SVG panel
// svgPlot(el, {height, xlabel, ylabel, xlog?, ylog?, render(api)}) ; api = {svg, X, Y, W, H, m, plot (clipped g)}
// update(opts) merges opts (xrange, yrange, render, ...) and redraws.
export function svgPlot(el, o) {
  el.classList.add("chart");
  const host = ui.h("div"); el.replaceChildren(host);
  let st = { height: 230, ...o }, cid = "fp" + Math.random().toString(36).slice(2);
  function draw() {
    const W = Math.floor(el.clientWidth); if (!W) return;
    host.replaceChildren();
    const H = st.height, m = { l: st.ylabel ? 54 : 40, r: 12, t: 10, b: st.xlabel ? 38 : 22 };
    const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": st.aria || st.ylabel || "chart" }, host);
    const [x0, x1] = st.xrange, [y0, y1] = st.yrange;
    const X = scale(st.xlog ? "log" : "linear", x0, x1, m.l, W - m.r), Y = scale(st.ylog ? "log" : "linear", y0, y1, H - m.b, m.t);
    const gg = svgEl("g", { class: "grid" }, svg), ax = svgEl("g", { class: "axis" }, svg);
    const yt = st.ylog ? logTicks(y0, y1) : niceTicks(y0, y1, Math.max(3, Math.round(H / 55)));
    const xt = st.xlog ? logTicks(x0, x1) : niceTicks(x0, x1, Math.max(3, Math.round(W / 85)));
    const T = (x, y, s, a = {}) => { const t = svgEl("text", { x, y, ...a }, svg); t.textContent = s; return t; };
    yt.forEach(v => { svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, gg); T(m.l - 6, Y(v) + 3.5, (st.yfmt || ui.fmt)(v), { "text-anchor": "end" }); });
    xt.forEach(v => { svgEl("line", { x1: X(v), x2: X(v), y1: H - m.b, y2: H - m.b + 4 }, ax); T(X(v), H - m.b + 15, (st.xfmt || ui.fmt)(v), { "text-anchor": "middle" }); });
    svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b + .5, y2: H - m.b + .5 }, ax);
    if (!st.ylog && y0 < 0 && y1 > 0) svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: "zero" }, svg);
    if (st.xlabel) T((m.l + W - m.r) / 2, H - 4, st.xlabel, { "text-anchor": "middle", class: "ttl" });
    if (st.ylabel) T(0, 0, st.ylabel, { "text-anchor": "middle", class: "ttl", transform: `translate(12,${(m.t + H - m.b) / 2}) rotate(-90)` });
    const clip = svgEl("clipPath", { id: cid }, svgEl("defs", {}, svg));
    svgEl("rect", { x: m.l - 8, y: m.t - 6, width: W - m.l - m.r + 16, height: H - m.t - m.b + 12 }, clip);
    const plot = svgEl("g", { "clip-path": `url(#${cid})` }, svg);
    st.render?.({ svg, plot, X, Y, W, H, m, T });
  }
  onWidth(el, draw);
  draw();
  return { el, update(d) { st = { ...st, ...d }; draw(); } };
}
export { svgEl };
