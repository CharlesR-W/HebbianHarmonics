// Hand-written chart primitives (canvas for big data, SVG for axes-heavy small data). No dependencies.
// Every chart: const c = kind(containerEl, opts); c.update(dataOrOpts); c.destroy().
// Charts fill their container's width, redraw on resize (and when a hidden tab becomes visible), are HiDPI-sharp,
// and show a tooltip on hover. API and examples: DESIGN.md "charts.js".
import { P, lut, cmap, rgbToCss, catColor } from "./palette.js";
import { h, fmt, showTip, hideTip, tipRows, colorbar, legend, renderMath } from "./ui.js";

const NS = "http://www.w3.org/2000/svg";
export function svgEl(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
const txt = (parent, x, y, s, attrs = {}) => { const t = svgEl("text", { x, y, ...attrs }, parent); t.textContent = s; return t; };

// ---------------------------------------------------------------- scales and ticks
export function scale(kind, d0, d1, r0, r1) {
  let f, inv;
  if (kind === "log") {
    const a = Math.log10(d0), b = Math.log10(d1);
    f = v => r0 + (Math.log10(v) - a) / ((b - a) || 1) * (r1 - r0);
    inv = p => Math.pow(10, a + (p - r0) / (r1 - r0) * (b - a));
  } else {
    f = v => r0 + (v - d0) / ((d1 - d0) || 1) * (r1 - r0);
    inv = p => d0 + (p - r0) / ((r1 - r0) || 1) * (d1 - d0);
  }
  f.inv = inv; f.kind = kind; f.domain = [d0, d1];
  return f;
}
export function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n + .5) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toFixed(10));
  return out;
}
export function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); Math.pow(10, e) <= hi * 1.0001; e++) if (Math.pow(10, e) >= lo * 0.9999) out.push(Math.pow(10, e));
  if (out.length < 2) { const m = [1, 2, 5]; out.length = 0; for (let e = Math.floor(Math.log10(lo)); Math.pow(10, e) <= hi; e++) for (const k of m) { const v = k * Math.pow(10, e); if (v >= lo * .9999 && v <= hi * 1.0001) out.push(v); } }
  return out;
}
const finite = a => a.filter(Number.isFinite);
const extent = (arr, log) => { let lo = Infinity, hi = -Infinity; for (const v of arr) { if (!Number.isFinite(v) || (log && v <= 0)) continue; if (v < lo) lo = v; if (v > hi) hi = v; } return lo <= hi ? [lo, hi] : [log ? 1 : 0, log ? 10 : 1]; };
const pad = ([lo, hi], f = .05) => { const d = (hi - lo) || Math.abs(hi) || 1; return [lo - d * f, hi + d * f]; };

// ---------------------------------------------------------------- HiDPI canvas + resize plumbing
export function hidpi(canvas, w, hgt) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(hgt * dpr);
  canvas.style.width = w + "px"; canvas.style.height = hgt + "px";
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}
// calls draw(width) now (if visible) and whenever the container's width changes; returns {redraw, disconnect}
function sized(el, draw) {
  let w = 0, queued = false;
  const run = () => { queued = false; const cw = Math.floor(el.clientWidth); if (cw > 0) { w = cw; draw(cw); } };
  const ro = new ResizeObserver(() => { const cw = Math.floor(el.clientWidth); if (cw > 0 && cw !== w && !queued) { queued = true; requestAnimationFrame(run); } });
  ro.observe(el);
  run();
  return { redraw: () => { const cw = Math.floor(el.clientWidth); if (cw > 0) { w = cw; draw(cw); } }, disconnect: () => ro.disconnect() };
}
let measureCtx = null;
const textWidth = (s, font = "10.5px Inter, system-ui, sans-serif") => {
  measureCtx = measureCtx || document.createElement("canvas").getContext("2d"); measureCtx.font = font; return measureCtx.measureText(String(s)).width;
};
const mapOf = m => (typeof m === "function" ? m : cmap(m || "seq"));

// ---------------------------------------------------------------- heatmap
// heatmap(el, {values, rows, cols, map: "seq"|"ink"|"div"|fn(t)->[r,g,b], range: [lo, hi], size?, square?: true,
//              groups?: {rows: [{start, end, label?, color?}], cols?: same (defaults to rows when square)},
//              rowLabels?, colLabels?, xlabel?, ylabel?, zlabel?, colorbar?: true, tooltip?(i, j, v), onHover?, onClick?})
// values: Float32Array (row-major, rows*cols) or array of rows. NaN cells are drawn in the wash colour.
// range is FIXED by default when given (so frames are comparable); without it: [-m, m] for "div", [min, max] else.
// update({values, range?, groups?, ...}) re-renders; update(values) is shorthand. Fast to ~1500x1500.
export function heatmap(el, o) {
  el.classList.add("chart");
  const wrap = h("div", { class: "hm-wrap" }); const cv = h("canvas"); wrap.append(cv); el.replaceChildren(wrap);
  let st = { colorbar: true, square: true, ...o }, cb = null, geom = null, hover = null;
  const off = document.createElement("canvas");
  const lab = h("div", { class: "hm-labels" }); wrap.append(lab);
  if (st.colorbar) { cb = colorbar({ map: typeof st.map === "string" ? st.map : "seq", label: st.zlabel, range: [0, 1] }); el.append(cb.el); }

  const get = (i, j) => { const v = st.values; return v ? (ArrayBuffer.isView(v) ? v[i * st.cols + j] : v[i][j]) : NaN; };
  function normalise() {
    const v = st.values;
    if (v && !ArrayBuffer.isView(v)) { st.rows = v.length; st.cols = v[0]?.length || 0; }
    if (!st.range) {
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < st.rows; i++) for (let j = 0; j < st.cols; j++) { const x = get(i, j); if (Number.isFinite(x)) { if (x < lo) lo = x; if (x > hi) hi = x; } }
      if (!(lo <= hi)) { lo = 0; hi = 1; }
      st._range = st.map === "div" ? [-Math.max(Math.abs(lo), Math.abs(hi)) || 1, Math.max(Math.abs(lo), Math.abs(hi)) || 1] : [lo, hi === lo ? lo + 1 : hi];
    } else st._range = st.range;
  }
  function paintOff() {
    const R = st.rows, C = st.cols; if (!R || !C) return;
    off.width = C; off.height = R;
    const ctx = off.getContext("2d"), img = ctx.createImageData(C, R), d = img.data;
    const table = typeof st.map === "function" ? null : lut(st.map || "seq");
    const f = table ? null : st.map;
    const [lo, hi] = st._range, k = 255 / ((hi - lo) || 1);
    const nan = [242, 241, 236];
    for (let i = 0, p = 0; i < R; i++) for (let j = 0; j < C; j++, p += 4) {
      const x = get(i, j);
      if (!Number.isFinite(x)) { d[p] = nan[0]; d[p + 1] = nan[1]; d[p + 2] = nan[2]; d[p + 3] = 255; continue; }
      if (table) { const q = Math.max(0, Math.min(255, Math.round((x - lo) * k))) * 4; d[p] = table[q]; d[p + 1] = table[q + 1]; d[p + 2] = table[q + 2]; d[p + 3] = 255; }
      else { const c = f((x - lo) / ((hi - lo) || 1)); d[p] = c[0]; d[p + 1] = c[1]; d[p + 2] = c[2]; d[p + 3] = 255; }
    }
    ctx.putImageData(img, 0, 0);
  }
  function draw(W) {
    const R = st.rows, C = st.cols; if (!R || !C) return;
    const gr = st.groups?.rows, gc = st.groups?.cols || (st.square && R === C ? gr : null);
    const band = gr || gc ? 6 : 0;
    const showRL = st.rowLabels, showCL = st.colLabels;
    // provisional cell size decides which labels fit (cell labels need >= 9 px rows / 11 px columns)
    const guessS = Math.min(st.size || 520, W - 60), fitR0 = guessS / Math.max(R, st.square ? C : 1) >= 9, fitC0 = guessS / C >= 11;
    const rowTexts = showRL && fitR0 ? st.rowLabels : gr && !fitR0 ? gr.map(g => g.label || "") : [];
    const hasColText = (showCL && fitC0) || (gc && !fitC0 && gc.some(g => g.label));
    const lw = rowTexts.length ? Math.ceil(Math.max(...rowTexts.map(textWidth))) + 6 : 0;
    const ml = (st.ylabel ? 18 : 0) + lw + band + 2;
    const mb = (st.xlabel ? 18 : 0) + band + (hasColText ? 15 : 0) + 2;
    const maxS = Math.min(st.size || 520, W - ml - 4);
    let cw = maxS / C, chh = st.square ? cw : (st.cellH || cw);
    if (st.square && R > C) { chh = cw = maxS / R; }
    const mw = cw * C, mh = chh * R;
    const Wt = Math.ceil(ml + mw + 2), Ht = Math.ceil(mh + mb + 2);
    geom = { ml, mt: 1, cw, chh, mw, mh };
    const ctx = hidpi(cv, Wt, Ht);
    ctx.imageSmoothingEnabled = cw < 1 || chh < 1;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, ml, 1, mw, mh);
    // group separators (between groups) and side bands with group colours
    ctx.strokeStyle = "rgba(255,255,255,.95)"; ctx.lineWidth = 1.5;
    const seps = (gs, horiz) => gs && gs.forEach(g => { if (g.start > 0) { ctx.beginPath(); if (horiz) { const y = 1 + g.start * chh; ctx.moveTo(ml, y); ctx.lineTo(ml + mw, y); } else { const x = ml + g.start * cw; ctx.moveTo(x, 1); ctx.lineTo(x, 1 + mh); } ctx.stroke(); } });
    seps(gr, true); seps(gc, false);
    if (gr) gr.forEach((g, k) => { ctx.fillStyle = g.color || (k % 2 ? P.mute2 : P.ink2); ctx.fillRect(ml - band, 1 + g.start * chh + .5, 3.5, (g.end - g.start) * chh - 1); });
    if (gc) gc.forEach((g, k) => { ctx.fillStyle = g.color || (k % 2 ? P.mute2 : P.ink2); ctx.fillRect(ml + g.start * cw + .5, 1 + mh + band - 3.5, (g.end - g.start) * cw - 1, 3.5); });
    if (hover) { ctx.strokeStyle = P.fg; ctx.lineWidth = 1.25; ctx.strokeRect(ml + hover[1] * cw, 1 + hover[0] * chh, Math.max(cw, 2), Math.max(chh, 2)); }
    // labels (HTML so they stay crisp and selectable): only those that fit
    lab.replaceChildren();
    const fitR = fitR0, fitC = fitC0;
    if (showRL && fitR) st.rowLabels.forEach((s, i) => lab.append(h("span", { style: { right: `${Wt - ml + band + 4}px`, top: `${1 + (i + .5) * chh - 5}px` } }, s)));
    if (showCL && fitC) st.colLabels.forEach((s, j) => lab.append(h("span", { style: { left: `${ml + (j + .5) * cw}px`, top: `${mh + band + 5}px`, transform: "translateX(-50%)" } }, s)));
    if (gr && !fitR) gr.forEach(g => g.label && lab.append(h("span", { style: { right: `${Wt - ml + band + 4}px`, top: `${1 + (g.start + g.end) / 2 * chh - 5}px` } }, g.label)));
    if (gc && !fitC) gc.forEach(g => g.label && lab.append(h("span", { style: { left: `${ml + (g.start + g.end) / 2 * cw}px`, top: `${mh + band + 5}px`, transform: "translateX(-50%)" } }, g.label)));
    if (st.xlabel) lab.append(h("span", { class: "hm-axis", style: { left: `${ml + mw / 2}px`, top: `${Ht - 13}px`, transform: "translateX(-50%)" } }, st.xlabel));
    if (st.ylabel) lab.append(h("span", { class: "hm-axis", style: { left: "7px", top: `${1 + mh / 2}px`, transform: "translate(-50%, -50%) rotate(-90deg)" } }, st.ylabel));
    wrap.style.width = Wt + "px"; wrap.style.height = Ht + "px";
    if (cb) { cb.set({ range: st._range, label: st.zlabel, map: typeof st.map === "string" ? st.map : "seq", ticks: st.ticks }); cb.el.style.marginLeft = ml + "px"; cb.el.style.width = Math.min(220, mw) + "px"; }
  }
  const pick = ev => {
    if (!geom) return null;
    const r = cv.getBoundingClientRect(), x = ev.clientX - r.left - geom.ml, y = ev.clientY - r.top - geom.mt;
    const j = Math.floor(x / geom.cw), i = Math.floor(y / geom.chh);
    return i >= 0 && j >= 0 && i < st.rows && j < st.cols ? [i, j] : null;
  };
  cv.addEventListener("pointermove", ev => {
    const p = pick(ev);
    if (!p) { if (hover) { hover = null; sz.redraw(); } hideTip(); st.onHover?.(null); return; }
    const [i, j] = p, v = get(i, j);
    if (!hover || hover[0] !== i || hover[1] !== j) { hover = p; sz.redraw(); st.onHover?.(i, j, v); }
    const content = st.tooltip ? st.tooltip(i, j, v) : tipRows(null, [
      { name: st.ylabel || "row", value: st.rowLabels ? st.rowLabels[i] : i },
      { name: st.xlabel || "col", value: st.colLabels ? st.colLabels[j] : j },
      { name: (st.zlabel || "value").replace(/\$/g, ""), value: fmt(v) }]);
    showTip(ev.clientX, ev.clientY, content);
  });
  cv.addEventListener("pointerleave", () => { hover = null; hideTip(); sz.redraw(); st.onHover?.(null); });
  cv.addEventListener("click", ev => { const p = pick(ev); if (p && st.onClick) st.onClick(p[0], p[1], get(p[0], p[1])); });
  normalise(); paintOff();
  const sz = sized(el, draw);
  return {
    el,
    update(d) {
      if (ArrayBuffer.isView(d) || Array.isArray(d)) d = { values: d };
      st = { ...st, ...d }; normalise(); paintOff(); sz.redraw();
    },
    get range() { return st._range; },
    destroy() { sz.disconnect(); el.replaceChildren(); },
  };
}

// ---------------------------------------------------------------- line chart (SVG)
// lineChart(el, {x?, series: [{name, y, x?, color?, dash?, width?, band?: [lo[], hi[]], dots?, points?, r?}], xlog?, ylog?,
//                xrange?, yrange?, xlabel?, ylabel?, height?: 220, xfmt?, yfmt?, refs?: [{x|y, label?}],
//                cursor?: x, legend?: auto, xname?})
// `points: true` draws a series as dots only (a scatter over a curve); `dots: true` adds dots to a line.
// Series colours default to the categorical order. update({...}) merges and redraws; setCursor(x) moves the
// vertical "now" marker (for linking to a transport) without a full redraw.
export function lineChart(el, o) {
  el.classList.add("chart");
  const host = h("div"); el.replaceChildren(host);
  let st = { height: 220, ...o }, svg = null, X = null, Y = null, cursorLine = null, lg = null;
  const xsOf = s => s.x || st.x || s.y.map((_, i) => i);
  function draw(W) {
    host.replaceChildren();
    const Hh = st.height, series = st.series.map((s, i) => ({ color: catColor(i), ...s }));
    const m = { l: st.ylabel ? 56 : 42, r: 12, t: 8, b: st.xlabel ? 38 : 22 };
    svg = svgEl("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": st.ylabel || "line chart" }, host);
    const allX = series.flatMap(xsOf), allY = series.flatMap(s => [...s.y, ...(s.band ? [...s.band[0], ...s.band[1]] : [])]);
    const nd = ([a, b], log) => (a < b ? [a, b] : log ? [a / 10, b * 10] : [a - 1, b + 1]);   // a single point still gets an axis
    const xr = st.xrange || nd(extent(allX, st.xlog), st.xlog), yr = st.yrange || (st.ylog ? nd(extent(allY, true), true) : pad(extent(allY)));
    X = scale(st.xlog ? "log" : "linear", xr[0], xr[1], m.l, W - m.r);
    Y = scale(st.ylog ? "log" : "linear", yr[0], yr[1], Hh - m.b, m.t);
    const yfmt = st.yfmt || (v => fmt(v)), xfmt = st.xfmt || (v => fmt(v));
    const yt = st.ylog ? logTicks(yr[0], yr[1]) : niceTicks(yr[0], yr[1], Math.max(3, Math.round(Hh / 55)));
    const xt = st.xlog ? logTicks(xr[0], xr[1]) : niceTicks(xr[0], xr[1], Math.max(3, Math.round(W / 90)));
    const g = svgEl("g", { class: "grid" }, svg);
    yt.forEach(v => svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g));
    const ax = svgEl("g", { class: "axis" }, svg);
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Hh - m.b + .5, y2: Hh - m.b + .5 }, ax);
    yt.forEach(v => txt(svg, m.l - 6, Y(v) + 3.5, yfmt(v), { "text-anchor": "end" }));
    xt.forEach(v => { svgEl("line", { x1: X(v), x2: X(v), y1: Hh - m.b, y2: Hh - m.b + 4 }, ax); txt(svg, X(v), Hh - m.b + 15, xfmt(v), { "text-anchor": "middle" }); });
    if (!st.ylog && yr[0] < 0 && yr[1] > 0) svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: "zero" }, svg);
    if (st.xlabel) txt(svg, (m.l + W - m.r) / 2, Hh - 4, st.xlabel, { "text-anchor": "middle", class: "ttl" });
    if (st.ylabel) txt(svg, 0, 0, st.ylabel, { "text-anchor": "middle", class: "ttl", transform: `translate(12,${(m.t + Hh - m.b) / 2}) rotate(-90)` });
    for (const r of st.refs || []) {
      if (r.y != null) { svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y), stroke: P.mute2, "stroke-dasharray": "3 3" }, svg); if (r.label) txt(svg, W - m.r - 2, Y(r.y) - 4, r.label, { "text-anchor": "end" }); }
      if (r.x != null) { svgEl("line", { x1: X(r.x), x2: X(r.x), y1: m.t, y2: Hh - m.b, stroke: P.mute2, "stroke-dasharray": "3 3" }, svg); if (r.label) txt(svg, X(r.x) + 4, m.t + 10, r.label); }
    }
    const ok = (xv, yv) => Number.isFinite(xv) && Number.isFinite(yv) && (!st.xlog || xv > 0) && (!st.ylog || yv > 0);
    const clip = svgEl("clipPath", { id: (st._cid = st._cid || "c" + Math.random().toString(36).slice(2)) }, svgEl("defs", {}, svg));
    svgEl("rect", { x: m.l, y: m.t - 2, width: W - m.l - m.r, height: Hh - m.t - m.b + 4 }, clip);
    const plot = svgEl("g", { "clip-path": `url(#${st._cid})` }, svg);
    const groups = series.map(s => {
      const gg = svgEl("g", {}, plot), xs = xsOf(s);
      if (s.band) {
        let d = "";
        xs.forEach((xv, i) => { if (ok(xv, s.band[1][i])) d += (d ? "L" : "M") + X(xv).toFixed(1) + "," + Y(s.band[1][i]).toFixed(1); });
        for (let i = xs.length - 1; i >= 0; i--) if (ok(xs[i], s.band[0][i])) d += "L" + X(xs[i]).toFixed(1) + "," + Y(s.band[0][i]).toFixed(1);
        if (d) svgEl("path", { d: d + "Z", fill: s.color, opacity: .13 }, gg);
      }
      let d = "", pen = false;
      xs.forEach((xv, i) => { const yv = s.y[i]; if (!ok(xv, yv)) { pen = false; return; } d += (pen ? "L" : "M") + X(xv).toFixed(1) + "," + Y(yv).toFixed(1); pen = true; });
      if (!s.points) svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash ? "5 4" : null }, gg);
      if (s.dots || s.points) xs.forEach((xv, i) => { if (ok(xv, s.y[i])) svgEl("circle", { cx: X(xv), cy: Y(s.y[i]), r: s.r || 3, fill: s.color, stroke: P.surface, "stroke-width": 1, opacity: s.points ? .85 : 1 }, gg); });
      return gg;
    });
    cursorLine = svgEl("line", { y1: m.t, y2: Hh - m.b, stroke: P.accent, "stroke-width": 1.25, opacity: 0 }, svg);
    if (st.cursor != null) api.setCursor(st.cursor);
    // hover: crosshair + every series' value at the nearest x
    const cross = svgEl("line", { y1: m.t, y2: Hh - m.b, stroke: P.fg, "stroke-width": 1, opacity: 0 }, svg);
    const dots = series.map(s => svgEl("circle", { r: 4, fill: s.color, stroke: P.surface, "stroke-width": 2, opacity: 0 }, svg));
    const hit = svgEl("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: Hh, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", ev => {
      const r = svg.getBoundingClientRect(), xv = X.inv(ev.clientX - r.left);
      cross.setAttribute("x1", X(xv)); cross.setAttribute("x2", X(xv)); cross.setAttribute("opacity", .25);
      const rows = series.map((s, si) => {
        const xs = xsOf(s); let bi = -1, bd = Infinity;
        xs.forEach((x, i) => { if (!ok(x, s.y[i])) return; const dd = Math.abs(X(x) - X(xv)); if (dd < bd) { bd = dd; bi = i; } });
        if (bi < 0) { dots[si].setAttribute("opacity", 0); return null; }
        dots[si].setAttribute("cx", X(xs[bi])); dots[si].setAttribute("cy", Y(s.y[bi])); dots[si].setAttribute("opacity", 1);
        return { name: s.name || "", value: yfmt(s.y[bi]), color: s.color, x: xs[bi] };
      }).filter(Boolean);
      if (rows.length) showTip(ev.clientX, ev.clientY, tipRows(`${st.xname || st.xlabel || "x"} = ${xfmt(rows[0].x)}`, rows));
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); dots.forEach(d => d.setAttribute("opacity", 0)); hideTip(); });
    const showLegend = st.legend ?? series.length > 1;
    if (showLegend) {
      lg = legend({ items: series.map((s, i) => ({ label: s.name, color: s.color, shape: s.points ? "dot" : s.dash ? "dash" : "line",
        onHover: on => groups.forEach((gg, j) => gg.setAttribute("opacity", on && j !== i ? .15 : 1)) })) });
      lg.el.classList.add("chart-legend"); lg.el.style.marginLeft = m.l + "px"; host.append(lg.el);
    }
    st._m = m; st._H = Hh;
  }
  const api = {
    el,
    update(d) { st = { ...st, ...d }; sz.redraw(); },
    setCursor(x) {
      st.cursor = x; if (!cursorLine || !X) return;
      if (x == null || !Number.isFinite(X(x))) { cursorLine.setAttribute("opacity", 0); return; }
      cursorLine.setAttribute("x1", X(x)); cursorLine.setAttribute("x2", X(x)); cursorLine.setAttribute("opacity", .8);
    },
    destroy() { sz.disconnect(); el.replaceChildren(); },
  };
  const sz = sized(el, draw);
  return api;
}

// ---------------------------------------------------------------- scatter (canvas)
// scatter(el, {x, y, color?: css[] | css, category?: int[] (palette P.cat, -1 = other), value?: number[], map?, range?,
//              size?: px | px[], shape?: ("dot"|"square")[], anchors?: [{x, y, label, color}], xrange?, yrange?, equal?,
//              xlabel?, ylabel?, height?: 320, tooltip?(i), onHover?(i|null), onSelect?(i|null)})
// Points have a 1 px surface ring so overlaps stay legible. highlight(i) rings a point (null clears).
export function scatter(el, o) {
  el.classList.add("chart");
  const cv = h("canvas"); el.replaceChildren(cv);
  let st = { height: 320, size: 3.5, ...o }, proj = null, hl = null, hov = null;
  const colorAt = i => {
    if (st.color) return typeof st.color === "string" ? st.color : st.color[i];
    if (st.category) return catColor(st.category[i]);
    if (st.value) { const [lo, hi] = st.range || extent(st.value); return rgbToCss(mapOf(st.map)((st.value[i] - lo) / ((hi - lo) || 1))); }
    return P.interior;
  };
  function draw(W) {
    const Hh = st.height, m = { l: st.ylabel ? 50 : 38, r: 10, t: 8, b: st.xlabel ? 36 : 22 };
    const ctx = hidpi(cv, W, Hh);
    const n = st.x.length, ax = [...st.x, ...(st.anchors || []).map(a => a.x)], ay = [...st.y, ...(st.anchors || []).map(a => a.y)];
    let xr = st.xrange || pad(extent(ax)), yr = st.yrange || pad(extent(ay));
    let pw = W - m.l - m.r, ph = Hh - m.t - m.b;
    if (st.equal) { // same data units per pixel on both axes
      const sx = (xr[1] - xr[0]) / pw, sy = (yr[1] - yr[0]) / ph, s = Math.max(sx, sy);
      const cx = (xr[0] + xr[1]) / 2, cy = (yr[0] + yr[1]) / 2; xr = [cx - s * pw / 2, cx + s * pw / 2]; yr = [cy - s * ph / 2, cy + s * ph / 2];
    }
    const X = scale("linear", xr[0], xr[1], m.l, W - m.r), Y = scale("linear", yr[0], yr[1], Hh - m.b, m.t);
    // axes
    ctx.font = "11px Inter, system-ui, sans-serif"; ctx.fillStyle = P.mute; ctx.strokeStyle = P.line; ctx.lineWidth = 1;
    const xt = niceTicks(xr[0], xr[1], Math.max(3, Math.round(W / 90))), yt = niceTicks(yr[0], yr[1], Math.max(3, Math.round(Hh / 60)));
    ctx.textAlign = "center"; xt.forEach(v => { ctx.beginPath(); ctx.moveTo(Math.round(X(v)) + .5, m.t); ctx.lineTo(Math.round(X(v)) + .5, Hh - m.b); ctx.stroke(); ctx.fillText(fmt(v), X(v), Hh - m.b + 14); });
    ctx.textAlign = "right"; yt.forEach(v => { ctx.beginPath(); ctx.moveTo(m.l, Math.round(Y(v)) + .5); ctx.lineTo(W - m.r, Math.round(Y(v)) + .5); ctx.stroke(); ctx.fillText(fmt(v), m.l - 6, Y(v) + 3.5); });
    ctx.fillStyle = P.ink2; ctx.font = "560 11.5px Inter, system-ui, sans-serif";
    if (st.xlabel) { ctx.textAlign = "center"; ctx.fillText(st.xlabel, (m.l + W - m.r) / 2, Hh - 4); }
    if (st.ylabel) { ctx.save(); ctx.translate(12, (m.t + Hh - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText(st.ylabel, 0, 0); ctx.restore(); }
    // points
    proj = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) {
      const px = X(st.x[i]), py = Y(st.y[i]); proj[2 * i] = px; proj[2 * i + 1] = py;
      if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
      const r = Array.isArray(st.size) || ArrayBuffer.isView(st.size) ? st.size[i] : st.size, sq = st.shape && st.shape[i] === "square";
      ctx.beginPath(); if (sq) ctx.rect(px - r, py - r, 2 * r, 2 * r); else ctx.arc(px, py, r, 0, 2 * Math.PI);
      ctx.fillStyle = colorAt(i); ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = P.surface; ctx.stroke();
    }
    for (const a of st.anchors || []) {
      const px = X(a.x), py = Y(a.y);
      ctx.fillStyle = a.color || P.boundary; ctx.fillRect(px - 5, py - 5, 10, 10); ctx.lineWidth = 1.5; ctx.strokeStyle = P.fg; ctx.strokeRect(px - 5, py - 5, 10, 10);
      if (a.label) { ctx.fillStyle = P.fg; ctx.font = "600 12px Inter, system-ui, sans-serif"; ctx.textAlign = "left"; ctx.fillText(a.label, px + 8, py - 6); }
    }
    for (const [i, col] of [[hl, P.fg], [hov, P.fg]]) {
      if (i == null || !proj) continue;
      ctx.beginPath(); ctx.arc(proj[2 * i], proj[2 * i + 1], (Array.isArray(st.size) ? st.size[i] : st.size) + 3, 0, 2 * Math.PI); ctx.lineWidth = 1.5; ctx.strokeStyle = col; ctx.stroke();
    }
  }
  const nearest = ev => {
    if (!proj) return null;
    const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let bi = null, bd = 100; // within 10 px
    for (let i = 0; i < proj.length / 2; i++) { const dx = proj[2 * i] - x, dy = proj[2 * i + 1] - y, d = dx * dx + dy * dy; if (d < bd) { bd = d; bi = i; } }
    return bi;
  };
  cv.addEventListener("pointermove", ev => {
    const i = nearest(ev);
    if (i !== hov) { hov = i; sz.redraw(); st.onHover?.(i); }
    if (i == null) { hideTip(); return; }
    showTip(ev.clientX, ev.clientY, st.tooltip ? st.tooltip(i) : tipRows(`point ${i}`, [{ name: st.xlabel || "x", value: fmt(st.x[i]) }, { name: st.ylabel || "y", value: fmt(st.y[i]) }]));
  });
  cv.addEventListener("pointerleave", () => { hov = null; hideTip(); sz.redraw(); st.onHover?.(null); });
  cv.addEventListener("click", ev => { const i = nearest(ev); st.onSelect?.(i); });
  const sz = sized(el, draw);
  return { el, update(d) { st = { ...st, ...d }; sz.redraw(); }, highlight(i) { hl = i; sz.redraw(); }, destroy() { sz.disconnect(); el.replaceChildren(); } };
}

// ---------------------------------------------------------------- bar chart (SVG)
// barChart(el, {labels, values?: number[], series?: [{name, values, color?}], colors?: css[] (per bar, single
//               series), color?, overlay?: {values, name, color?}, yrange?, xlabel?, ylabel?, height?: 180,
//               yfmt?, tooltip?(i, seriesIndex), labelEvery?})
// Signed values get a zero line. `overlay` draws a second quantity as ink dots joined by a thin line (e.g. an
// eigenvector over the same boundary neurons as the bars). update({...}) merges and redraws.
export function barChart(el, o) {
  el.classList.add("chart");
  const host = h("div"); el.replaceChildren(host);
  let st = { height: 180, ...o };
  function draw(W) {
    host.replaceChildren();
    const Hh = st.height, L = st.labels.length;
    const series = st.series || [{ name: st.name || "", values: st.values, color: st.color || P.ink2 }];
    const all = [...series.flatMap(s => s.values), ...(st.overlay ? st.overlay.values : []), 0];
    const yr = st.yrange || pad(extent(all), .06);
    const m = { l: st.ylabel ? 50 : 38, r: 6, t: 6, b: (st.xlabel ? 34 : 18) + (st.compact ? -4 : 0) };
    const svg = svgEl("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}` }, host);
    const Y = scale("linear", yr[0], yr[1], Hh - m.b, m.t), bw = (W - m.l - m.r) / L, yfmt = st.yfmt || (v => fmt(v));
    const g = svgEl("g", { class: "grid" }, svg);
    const yt = niceTicks(yr[0], yr[1], Math.max(2, Math.round(Hh / 50)));
    yt.forEach(v => { svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); txt(svg, m.l - 5, Y(v) + 3.5, yfmt(v), { "text-anchor": "end" }); });
    const y0 = Y(Math.max(yr[0], Math.min(yr[1], 0)));
    const ns = series.length, gap = Math.min(2, bw * .15), sw = (bw - gap) / ns;
    series.forEach((s, si) => s.values.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      const x = m.l + i * bw + gap / 2 + si * sw, y = Y(v), hgt = Math.abs(y - y0);
      const col = (st.colors && ns === 1 ? st.colors[i] : s.color) || catColor(si);
      const r = svgEl("rect", { x: x.toFixed(2), y: Math.min(y, y0).toFixed(2), width: Math.max(.8, sw - (ns > 1 ? .5 : 0)).toFixed(2), height: Math.max(.5, hgt).toFixed(2), fill: col, rx: Math.min(1.5, sw / 4) }, svg);
      r.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, st.tooltip ? st.tooltip(i, si) :
        tipRows(st.labels[i], [{ name: s.name || st.ylabel || "value", value: yfmt(v), color: col, shape: "square" }, ...(st.overlay ? [{ name: st.overlay.name || "overlay", value: yfmt(st.overlay.values[i]), color: st.overlay.color || P.fg }] : [])])));
      r.addEventListener("pointerleave", hideTip);
    }));
    svgEl("line", { x1: m.l, x2: W - m.r, y1: y0, y2: y0, class: "zero" }, svg);
    if (st.overlay) {
      const oc = st.overlay.color || P.fg; let d = "";
      st.overlay.values.forEach((v, i) => { if (Number.isFinite(v)) d += (d ? "L" : "M") + (m.l + (i + .5) * bw).toFixed(1) + "," + Y(v).toFixed(1); });
      svgEl("path", { d, fill: "none", stroke: oc, "stroke-width": 1, opacity: .7 }, svg);
      st.overlay.values.forEach((v, i) => { if (Number.isFinite(v)) svgEl("circle", { cx: m.l + (i + .5) * bw, cy: Y(v), r: Math.min(2.6, bw / 3 + .8), fill: oc, stroke: P.surface, "stroke-width": 1, "pointer-events": "none" }, svg); });
    }
    const every = st.labelEvery || Math.max(1, Math.ceil(L / Math.max(1, Math.floor((W - m.l - m.r) / 22))));
    st.labels.forEach((s, i) => { if (i % every === 0) txt(svg, m.l + (i + .5) * bw, Hh - m.b + 12, s, { "text-anchor": "middle" }); });
    if (st.xlabel) txt(svg, (m.l + W - m.r) / 2, Hh - 4, st.xlabel, { "text-anchor": "middle", class: "ttl" });
    if (st.ylabel) txt(svg, 0, 0, st.ylabel, { "text-anchor": "middle", class: "ttl", transform: `translate(11,${(m.t + Hh - m.b) / 2}) rotate(-90)` });
    if (st.legend ?? (ns > 1 || st.overlay)) {
      const items = st.series ? series.map((s, i) => ({ label: s.name, color: s.color || catColor(i), shape: "square" })) : [{ label: st.name || st.ylabel || "value", color: st.color || P.ink2, shape: "square" }];
      if (st.overlay) items.push({ label: st.overlay.name || "overlay", color: st.overlay.color || P.fg, shape: "dot" });
      const lg = legend({ items }); lg.el.classList.add("chart-legend"); lg.el.style.marginLeft = m.l + "px"; host.append(lg.el);
    }
  }
  const sz = sized(el, draw);
  return { el, update(d) { st = { ...st, ...d }; sz.redraw(); }, destroy() { sz.disconnect(); el.replaceChildren(); } };
}

// barMultiples(el, {labels, panels: [{title, values, color?, sub?, overlay?: number[]}], yrange (shared; required
//                   for honest comparison), overlayName?, valueName?, height?: 96, panelWidth?: 180})
// Small multiples of bars over the same categories (e.g. one panel per cluster, bars over boundary neurons).
export function barMultiples(el, o) {
  el.classList.add("chart");
  let st = { height: 96, panelWidth: 180, ...o }, charts = [];
  function build() {
    charts.forEach(c => c.destroy()); charts = [];
    const grid = h("div", { class: "bars-multi", style: { "--panel-w": st.panelWidth + "px" } });
    el.replaceChildren(grid);
    st.panels.forEach((p, k) => {
      const box = h("div");
      box.append(h("div", { class: "bm-title" }, h("i", { class: "u-mk dot", style: { "--sw": p.color || catColor(k) } }), h("span", { html: p.title || `panel ${k + 1}` })));
      if (p.sub) box.append(h("div", { class: "bm-sub", html: p.sub }));
      const c = h("div"); box.append(c); grid.append(box);
      charts.push(barChart(c, { labels: st.labels, values: p.values, color: p.color || catColor(k), yrange: st.yrange, height: st.height, compact: true, legend: false, name: st.valueName,
        overlay: p.overlay ? { values: p.overlay, name: st.overlayName } : null, labelEvery: st.labelEvery }));
    });
    if (st.legend !== false && st.panels.some(p => p.overlay)) {
      const lg = legend({ items: [{ label: st.valueName || "value", color: P.ink2, shape: "square" }, { label: st.overlayName || "overlay", color: P.fg, shape: "dot" }] });
      lg.el.classList.add("chart-legend"); el.append(lg.el);
    }
  }
  build();
  return { el, update(d) { st = { ...st, ...d }; build(); }, destroy() { charts.forEach(c => c.destroy()); el.replaceChildren(); } };
}

// ---------------------------------------------------------------- image grid
// imageGrid(el, {arrays?: Float32Array[] (each w*h, row-major), w, h, map?: "div"|"seq"|"ink", range?: [lo,hi] |
//                "tile" (each tile its own symmetric/min-max range) — default: one shared range over all tiles,
//                sprite?: {src, tileW, tileH, cols}, indices?: int[] (sprite tiles to show),
//                labels?: string[], colors?: css[] (a swatch line under each tile), tile?: 44 (min px),
//                rows?: [{label, count}] (labelled rows: tiles in order, `count` per row, fixed tile size),
//                tooltip?(i), onHover?(i|null), onClick?(i)})
// Tiles from Float32 arrays are rendered through the colour map; sprite tiles are copied from the PNG.
export function imageGrid(el, o) {
  el.classList.add("chart");
  let st = { tile: 44, ...o }, sel = null, spriteImg = null, spriteSrc = null;
  const grid = h("div", { class: "img-grid" }); el.replaceChildren(grid);
  async function loadSprite() {
    if (!st.sprite || spriteSrc === st.sprite.src) return spriteImg;
    spriteSrc = st.sprite.src; spriteImg = new Image(); spriteImg.src = st.sprite.src;
    await spriteImg.decode(); return spriteImg;
  }
  async function build() {
    grid.style.setProperty("--tile", st.tile + "px");
    grid.classList.toggle("ig-rows", !!st.rows);
    const n = st.arrays ? st.arrays.length : (st.indices || []).length;
    let range = st.range, table = lut(st.map || (st.arrays ? "div" : "seq"));
    if (st.arrays && !range) { // shared range over all tiles
      let m = 0, lo = Infinity, hi = -Infinity;
      st.arrays.forEach(a => { for (const v of a) if (Number.isFinite(v)) { m = Math.max(m, Math.abs(v)); lo = Math.min(lo, v); hi = Math.max(hi, v); } });
      range = (st.map || "div") === "div" ? [-m || -1, m || 1] : [lo, hi];
    }
    const img = st.sprite ? await loadSprite() : null;
    grid.replaceChildren();
    // labelled rows: tiles are taken in order, `count` per row, each row headed by its label
    let rowHost = grid, rowLeft = 0, ri = -1;
    for (let i = 0; i < n; i++) {
      if (st.rows && rowLeft === 0 && ri + 1 < st.rows.length) {
        ri += 1; rowLeft = st.rows[ri].count;
        rowHost = h("div", { class: "ig-row-tiles" });
        grid.append(h("div", { class: "ig-row" }, h("div", { class: "ig-row-label", html: st.rows[ri].label || "" }), rowHost));
      }
      const tile = h("div", { class: "ig-tile" }), cv = h("canvas"); tile.append(cv);
      if (st.arrays) {
        const a = st.arrays[i], W = st.w, H = st.h; cv.width = W; cv.height = H;
        let [lo, hi] = range === "tile" ? (() => { let m = 0, l = Infinity, u = -Infinity; for (const v of a) { m = Math.max(m, Math.abs(v)); l = Math.min(l, v); u = Math.max(u, v); } return (st.map || "div") === "div" ? [-m || -1, m || 1] : [l, u]; })() : range;
        const ctx = cv.getContext("2d"), im = ctx.createImageData(W, H), d = im.data, k = 255 / ((hi - lo) || 1);
        for (let p = 0; p < W * H; p++) { const q = Math.max(0, Math.min(255, Math.round((a[p] - lo) * k))) * 4; d[4 * p] = table[q]; d[4 * p + 1] = table[q + 1]; d[4 * p + 2] = table[q + 2]; d[4 * p + 3] = 255; }
        ctx.putImageData(im, 0, 0);
      } else if (img) {
        const { tileW, tileH, cols } = st.sprite, idx = st.indices[i];
        cv.width = tileW; cv.height = tileH;
        cv.getContext("2d").drawImage(img, (idx % cols) * tileW, Math.floor(idx / cols) * tileH, tileW, tileH, 0, 0, tileW, tileH);
      }
      if (st.labels?.[i] != null) tile.append(h("span", { class: "ig-lab" }, st.labels[i]));
      if (st.colors?.[i]) tile.append(h("i", { class: "ig-sw", style: { "--sw": st.colors[i] } }));
      tile.classList.toggle("sel", sel === i);
      tile.addEventListener("pointermove", ev => { st.onHover?.(i); if (st.tooltip) showTip(ev.clientX, ev.clientY, st.tooltip(i)); });
      tile.addEventListener("pointerleave", () => { st.onHover?.(null); hideTip(); });
      tile.addEventListener("click", () => st.onClick?.(i));
      rowHost.append(tile); if (st.rows) rowLeft -= 1;
    }
    if (st.rows) renderMath(grid);
    st._range = range;
  }
  const ready = build();
  return {
    el, ready,
    update(d) { st = { ...st, ...d }; return build(); },
    select(i) { sel = i; grid.querySelectorAll(".ig-tile").forEach((t, k) => t.classList.toggle("sel", k === i)); },
    get range() { return st._range; },
    destroy() { el.replaceChildren(); },
  };
}

// ---------------------------------------------------------------- filmstrip: one heatmap over snapshots
// filmstrip(el, {frames: [values...] (each as for heatmap), frameLabels: string[], sliderLabel?: "checkpoint",
//                range: [lo, hi] (FIXED across frames; computed over all frames if omitted), thumbs?: true,
//                fps?: 4, value?: index, onFrame?(i), ...any heatmap option})
// A slider (with play) steps through the frames; a row of thumbnails (click to jump) shows the whole run at once.
export function filmstrip(el, o) {
  el.classList.add("chart");
  let st = { thumbs: true, fps: 4, sliderLabel: "snapshot", ...o }, k = o.value ?? o.frames.length - 1, timer = 0;
  if (!st.range) {
    let lo = Infinity, hi = -Infinity;
    for (const f of st.frames) for (const v of (ArrayBuffer.isView(f) ? f : f.flat())) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    st.range = st.map === "div" ? [-Math.max(Math.abs(lo), Math.abs(hi)), Math.max(Math.abs(lo), Math.abs(hi))] : [lo, hi];
  }
  const play = h("button", { type: "button", class: "u-tbtn", "aria-label": "Play through snapshots", html: PLAY });
  const inp = h("input", { type: "range", min: 0, max: st.frames.length - 1, step: 1, value: k, class: "fs-slider", "aria-label": st.sliderLabel });
  const lab = h("span", { class: "fs-label" });
  const bar = h("div", { class: "fs-bar" }, play, h("span", { class: "u-name" }, st.sliderLabel), inp, lab);
  const hmHost = h("div"), thumbs = h("div", { class: "fs-thumbs" });
  el.replaceChildren(bar, hmHost, ...(st.thumbs ? [thumbs] : []));
  const { frames, frameLabels, thumbs: _t, fps, value, onFrame, sliderLabel, ...hmOpts } = st;
  const hm = heatmap(hmHost, { ...hmOpts, values: frames[k] });
  const table = lut(typeof st.map === "string" ? st.map : "seq"), [lo, hi] = st.range;
  if (st.thumbs) frames.forEach((f, i) => {
    const R = ArrayBuffer.isView(f) ? st.rows : f.length, C = ArrayBuffer.isView(f) ? st.cols : f[0].length;
    const cv = h("canvas", { width: C, height: R, title: frameLabels?.[i] ?? String(i) }), ctx = cv.getContext("2d"), im = ctx.createImageData(C, R);
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const v = ArrayBuffer.isView(f) ? f[r * C + c] : f[r][c], q = Math.max(0, Math.min(255, Math.round((v - lo) / ((hi - lo) || 1) * 255))) * 4, p = 4 * (r * C + c);
      im.data[p] = table[q]; im.data[p + 1] = table[q + 1]; im.data[p + 2] = table[q + 2]; im.data[p + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    const b = h("button", { type: "button", class: "fs-thumb" }, cv, h("span", {}, frameLabels?.[i] ?? String(i)));
    b.addEventListener("click", () => { stop(); go(i); });
    thumbs.append(b);
  });
  const go = i => {
    k = i; inp.value = i; lab.textContent = frameLabels?.[i] ?? String(i);
    hm.update({ values: frames[i] });
    thumbs.querySelectorAll(".fs-thumb").forEach((t, j) => t.classList.toggle("on", j === i));
    onFrame?.(i);
  };
  const stop = () => { clearInterval(timer); timer = 0; play.innerHTML = PLAY; };
  play.addEventListener("click", () => {
    if (timer) return stop();
    if (k >= frames.length - 1) go(0);
    play.innerHTML = PAUSE;
    timer = setInterval(() => { if (k >= frames.length - 1) stop(); else go(k + 1); }, 1000 / fps);
  });
  inp.addEventListener("input", () => { stop(); go(+inp.value); });
  go(k);
  return { el, heatmap: hm, go, get index() { return k; }, destroy() { stop(); hm.destroy(); el.replaceChildren(); } };
}
const PLAY = '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="M4.5 2.6v10.8L13 8z" fill="currentColor"/></svg>';
const PAUSE = '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><rect x="3.5" y="2.8" width="3.2" height="10.4" rx=".8" fill="currentColor"/><rect x="9.3" y="2.8" width="3.2" height="10.4" rx=".8" fill="currentColor"/></svg>';
