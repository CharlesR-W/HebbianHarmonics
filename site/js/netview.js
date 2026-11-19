// Neuron-space network diagram (DESIGN.md "netview.js"). Nodes are neurons, edges are synapses. Node positions are
// seeded-random (uniform in the panel with a minimum spacing) so that the physical layout is visibly unrelated to
// what a neuron computes; never position neurons by their embedding here (that is an input-space panel).
// Canvas 2D, HiDPI, redraws on resize and when a hidden tab becomes visible, draws only when something changed.
//
//   const nv = netview(el, { N, B, seed, tooltip, onHover, onSelect });
//   nv.setColors(colors); nv.setEdges(pairs, weights); nv.setHighlight(i);
//
// Indices: neurons 0..N-1; in edge lists an index >= N is input (index - N), as in view3d.setEdges.
import { P, boundaryColor } from "./palette.js";
import { h, showTip, hideTip, spaceBadge } from "./ui.js";
export { spaceBadge };
import { hidpi } from "./charts.js";

// ---------------------------------------------------------------- seeded random layout
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** randomLayout(n, {seed = 1, aspect = 1.5, spacing = 0.62}) -> Float32Array(2n) of positions in [0, 1]^2.
 *  Dart throwing in a box of the given aspect (width / height): each point is uniform in the box and at least
 *  `spacing * sqrt(area / n)` from every earlier point (the bound relaxes by 10% for a point only after 80 misses).
 *  Deterministic in (n, seed, aspect, spacing). Use it directly for SVG figures that must share the convention. */
export function randomLayout(n, { seed = 1, aspect = 1.5, spacing = 0.62 } = {}) {
  const rnd = mulberry32((seed * 2654435761) ^ (n * 40503) ^ 0x5bd1e995);
  const out = new Float32Array(2 * n), A = aspect;
  let dmin = spacing * Math.sqrt(A / Math.max(1, n));          // in box units (box = A x 1)
  const cell = Math.max(dmin, 1e-3), gx = Math.ceil(A / cell), gy = Math.ceil(1 / cell);
  const grid = new Map(), key = (cx, cy) => cx * 100003 + cy;
  for (let k = 0; k < n; k++) {
    let x = 0, y = 0, d = dmin, tries = 0;
    for (;;) {
      x = rnd() * A; y = rnd();
      const cx = Math.floor(x / cell), cy = Math.floor(y / cell), r = Math.ceil(d / cell);
      let ok = true;
      for (let i = cx - r; i <= cx + r && ok; i++) for (let j = cy - r; j <= cy + r && ok; j++) {
        const list = grid.get(key(i, j)); if (!list) continue;
        for (const q of list) { const dx = out[2 * q] * A - x, dy = out[2 * q + 1] - y; if (dx * dx + dy * dy < d * d) { ok = false; break; } }
      }
      if (ok) break;
      if (++tries % 80 === 0) d *= 0.9;
    }
    out[2 * k] = x / A; out[2 * k + 1] = y;
    const kk = key(Math.min(gx, Math.floor(x / cell)), Math.min(gy, Math.floor(y / cell)));
    (grid.get(kk) || grid.set(kk, []).get(kk)).push(k);
  }
  return out;
}

const SUBS = "₀₁₂₃₄₅₆₇₈₉";
const sub = n => String(n).split("").map(c => SUBS[+c]).join("");

/** netview(el, opts) -> {setColors, setInputColors, setSizes, setLabels, setEdges, setHighlight, setSelected,
 *                        positions, render, el, destroy}
 *  opts: N, B = 0, seed = 1, aspect = 1.5 (layout box; the panel is drawn at this aspect up to maxHeight),
 *        maxHeight = 460, minHeight = 220, height (fixed px; overrides aspect), radius (px; default from spacing),
 *        inputLabels (default g₁ g₂ …), inputColors (default boundaryColor(b, B)), colors (default P.interior),
 *        labels (strings drawn inside neurons when the radius allows), edgeWidth = 3.2 (px at w = wmax),
 *        denseAt = 150 (above this many edges lines thin and fade as sqrt(denseAt / E)),
 *        edgeCap (default 0.45 x panel width in px; above this many synapses only each node's strongest k are drawn,
 *        k = max(1, round(edgeCap / nodes)); the ringed, hovered or tapped node shows all its synapses; 0: draw all),
 *        edgeColor = P.edge, badge = "neuron space" (false: none), corner (HTML note, top right),
 *        tooltip(i, kind) -> HTML | Node (kind "neuron" | "input"), onHover(i | null, kind), onSelect(i | null, kind),
 *        glyph(ctx2d, i, x, y, r) -> true if it drew neuron i itself (e.g. a pie), ariaLabel. */
export function netview(el, o = {}) {
  const N = o.N | 0, B = o.B | 0, M = N + B, aspect = o.aspect || 1.5;
  const layout = randomLayout(M, { seed: o.seed ?? 1, aspect, spacing: o.spacing ?? 0.62 });
  // inputs take the first B slots of the layout: their places do not depend on N
  const posIn = layout.subarray(0, 2 * B), posN = layout.subarray(2 * B);
  const cv = h("canvas", { role: "img", "aria-label": o.ariaLabel || `network of ${N} neurons${B ? ` and ${B} inputs` : ""}, drawn at random positions` });
  const wrap = h("div", { class: "netview chart", style: { position: "relative" } }, cv);
  const thinNote = h("div", { class: "u-help nv-thin", hidden: true });
  wrap.append(thinNote);
  if (o.badge !== false) wrap.append(h("span", { class: "space-badge nv-badge", "data-space": "neuron", text: o.badge || "neuron space" }));
  const corner = h("div", { class: "nv-corner u-help" }); if (o.corner) { corner.innerHTML = o.corner; wrap.append(corner); }
  el.append(wrap);

  const st = {
    colors: o.colors || P.interior, inColors: o.inputColors || Array.from({ length: B }, (_, b) => boundaryColor(b, B)),
    sizes: 1, labels: o.labels || null, inLabels: o.inputLabels || Array.from({ length: B }, (_, b) => `g${sub(b + 1)}`),
    pairs: null, w: null, wmax: 1, order: null, keep: null, keepW: 0, hi: new Set(), hiKind: "neuron", sel: null, selKind: "neuron", hover: null,
  };
  let W = 0, Hh = 0, px = new Float32Array(2 * M), r0 = 4, queued = false, ctx2 = null;
  const pad = () => ({ l: 14 + r0, r: 14 + r0, t: (o.badge !== false ? 30 : 12) + r0, b: 12 + r0 + (B ? 12 : 0) });

  function geometry() {
    const cw = Math.floor(wrap.clientWidth); if (cw <= 0) return false;
    W = cw; Hh = o.height || Math.round(Math.max(o.minHeight || 220, Math.min(o.maxHeight || 460, W / aspect)));
    const spacingPx = Math.sqrt((W * Hh) / Math.max(1, M)) * (o.spacing ?? 0.62);
    r0 = o.radius || Math.max(2.4, Math.min(15, spacingPx * 0.3));
    const p = pad();
    for (let k = 0; k < M; k++) { px[2 * k] = p.l + layout[2 * k] * (W - p.l - p.r); px[2 * k + 1] = p.t + layout[2 * k + 1] * (Hh - p.t - p.b); }
    ctx2 = hidpi(cv, W, Hh);
    return true;
  }
  const nx = i => px[2 * (B + i)], ny = i => px[2 * (B + i) + 1], ix = b => px[2 * b], iy = b => px[2 * b + 1];
  const pt = q => (q >= N ? [ix(q - N), iy(q - N)] : [nx(q), ny(q)]);
  const colOf = i => (Array.isArray(st.colors) ? st.colors[i] : st.colors) || P.interior;
  const sizeOf = i => r0 * (Array.isArray(st.sizes) || ArrayBuffer.isView(st.sizes) ? st.sizes[i] ?? 1 : st.sizes);

  function draw() {
    queued = false;
    if (!ctx2 || Math.floor(wrap.clientWidth) !== W) { if (!geometry()) return; }
    const c = ctx2; c.clearRect(0, 0, W, Hh);
    const shown = new Set();      // nodes whose synapses are all drawn (edge-list indices: inputs are N + b)
    for (const q of st.hi) shown.add(st.hiKind === "input" ? N + q : q);
    for (const s of [st.hover, st.sel != null ? { i: st.sel, kind: st.selKind } : null]) if (s) shown.add(s.kind === "input" ? N + s.i : s.i);
    const touched = q => shown.has(q);
    if (st.order && st.keepW !== W) thin();
    // synapses: weakest first so strong ones sit on top; highlighted neurons' synapses drawn last, darker
    if (st.pairs && st.order) {
      c.lineCap = "round"; c.strokeStyle = o.edgeColor || P.edge;
      // dense pictures (a few hundred synapses or more) get thinner, fainter lines so that clusters stay legible
      const dens = Math.min(st.keep ? 0.55 : 1, Math.sqrt((o.denseAt || 150) / (st.keepN || st.order.length))), aF = Math.max(0.22, dens), wF = Math.max(0.3, dens);
      const ew = o.edgeWidth || 3.2, late = [];
      for (const e of st.order) {
        const a = st.pairs[2 * e], b = st.pairs[2 * e + 1], f = Math.min(1, st.w[e] / st.wmax);
        if (shown.size && (touched(a) || touched(b))) { late.push(e); continue; }
        if (st.keep && !st.keep[e]) continue;
        const [x1, y1] = pt(a), [x2, y2] = pt(b);
        c.globalAlpha = (0.12 + 0.64 * f) * aF * (shown.size ? 0.45 : 1); c.lineWidth = 0.4 + (ew - 0.4) * f * wF;
        c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
      }
      for (const e of late) {
        const a = st.pairs[2 * e], b = st.pairs[2 * e + 1], f = Math.min(1, st.w[e] / st.wmax);
        const [x1, y1] = pt(a), [x2, y2] = pt(b);
        c.globalAlpha = 0.3 + 0.6 * f; c.lineWidth = 0.8 + (ew - 0.4) * f * Math.max(0.5, wF);
        c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
      }
      c.globalAlpha = 1;
    }
    // neurons
    const showLab = st.labels && r0 >= 8.5;
    for (let i = 0; i < N; i++) {
      const x = nx(i), y = ny(i), r = sizeOf(i);
      if (o.glyph && o.glyph(c, i, x, y, r)) continue;
      c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.fillStyle = colOf(i); c.fill();
      c.lineWidth = r > 5 ? 1 : 0.6; c.strokeStyle = r > 5 ? P.fg : P.surface; c.stroke();
      if (showLab) { c.fillStyle = o.labelColor ? o.labelColor(i) : P.fg; c.font = `600 ${Math.round(Math.min(13.5, r * 0.85))}px Inter, system-ui, sans-serif`; c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(st.labels[i], x, y + 0.5); }
    }
    // inputs: squares with an ink outline, labelled
    const s = Math.max(4.5, r0 * 1.15 + 1.5);
    for (let b = 0; b < B; b++) {
      const x = ix(b), y = iy(b), fill = Array.isArray(st.inColors) ? st.inColors[b] : st.inColors;
      c.fillStyle = fill; c.fillRect(x - s, y - s, 2 * s, 2 * s);
      c.lineWidth = 1.4; c.strokeStyle = o.inputStroke ? o.inputStroke(b) : P.boundary; c.strokeRect(x - s, y - s, 2 * s, 2 * s);
      if (st.inLabels[b]) {
        c.font = `600 ${s > 9 ? 14 : 11.5}px Inter, system-ui, sans-serif`; c.fillStyle = P.fg; c.textBaseline = "middle";
        const right = x < W - 60; c.textAlign = right ? "left" : "right";
        const lx = x + (right ? s + 4 : -s - 4);
        c.lineWidth = 3.5; c.strokeStyle = P.surface; c.lineJoin = "round"; c.strokeText(st.inLabels[b], lx, y); c.fillText(st.inLabels[b], lx, y);
      }
    }
    // highlight rings (external or hover) and the selection
    const ring = (q, kind, dash) => {
      const [x, y] = kind === "input" ? [ix(q), iy(q)] : [nx(q), ny(q)], r = (kind === "input" ? s * 1.25 : sizeOf(q)) + 3.5;
      c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.lineWidth = 2.2; c.strokeStyle = P.accent; c.setLineDash(dash ? [4, 3] : []); c.stroke(); c.setLineDash([]);
    };
    for (const q of st.hi) ring(q, st.hiKind, false);
    if (st.sel != null && !(st.hi.has(st.sel) && st.hiKind === st.selKind)) ring(st.sel, st.selKind, true);
  }
  // many synapses: keep only each node's strongest k (union over both ends), k from the cap; recomputed per width
  function thin() {
    st.keepW = W; st.keep = null; st.keepN = 0;
    const E = st.order.length, cap = o.edgeCap ?? Math.round(0.45 * W);
    if (cap > 0 && E > cap) {
      const k = Math.max(1, Math.round(cap / M)), cnt = new Uint16Array(M), keep = new Uint8Array(st.w.length);
      for (let q = E - 1; q >= 0; q--) {
        const e = st.order[q], a = st.pairs[2 * e], b = st.pairs[2 * e + 1], ka = a >= N ? a - N : B + a, kb = b >= N ? b - N : B + b;
        const ta = cnt[ka]++ < k, tb = cnt[kb]++ < k;
        if (ta || tb) { keep[e] = 1; st.keepN++; }
      }
      st.keep = keep;
      thinNote.textContent = `${st.keepN} of ${E} synapses drawn, each node's strongest${k > 1 ? " " + k : ""}. Hover or tap a neuron for all of its own.`;
    }
    thinNote.hidden = !st.keep;
  }
  const render = () => { if (!queued) { queued = true; requestAnimationFrame(draw); } };

  // ---- interaction
  function pick(ev) {
    const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = null, bd = Infinity;
    for (let b = 0; b < B; b++) { const d = (ix(b) - x) ** 2 + (iy(b) - y) ** 2; if (d < bd) { bd = d; best = { i: b, kind: "input" }; } }
    for (let i = 0; i < N; i++) { const d = (nx(i) - x) ** 2 + (ny(i) - y) ** 2; if (d < bd) { bd = d; best = { i, kind: "neuron" }; } }
    const lim = Math.max(10, r0 * 1.6 + 4);
    return best && bd <= lim * lim ? best : null;
  }
  const same = (a, b) => (a && b ? a.i === b.i && a.kind === b.kind : a === b);
  cv.addEventListener("pointermove", ev => {
    const q = pick(ev);
    if (!same(q, st.hover)) { st.hover = q; o.onHover?.(q ? q.i : null, q ? q.kind : null); render(); }
    cv.style.cursor = q && o.onSelect ? "pointer" : "default";
    const tip = q && o.tooltip ? o.tooltip(q.i, q.kind) : null;
    if (tip) showTip(ev.clientX, ev.clientY, tip); else hideTip();
  });
  cv.addEventListener("pointerleave", () => { hideTip(); if (st.hover) { st.hover = null; o.onHover?.(null, null); render(); } });
  cv.addEventListener("click", ev => {
    const q = pick(ev);
    if (o.onSelect) { o.onSelect(q ? q.i : null, q ? q.kind : null); return; }
    // no handler: a tap toggles a local selection (dashed ring, all its synapses drawn)
    const again = q && st.sel === q.i && st.selKind === q.kind;
    st.sel = q && !again ? q.i : null; st.selKind = q ? q.kind : "neuron"; render();
  });

  const ro = new ResizeObserver(() => { const cw = Math.floor(wrap.clientWidth); if (cw > 0 && cw !== W) { ctx2 = null; render(); } });
  ro.observe(wrap);
  geometry(); render();

  return {
    el: wrap, canvas: cv,
    /** neuron fill colours: one CSS colour or an array of N */
    setColors(c) { st.colors = c; render(); },
    /** input fill colours: one CSS colour or an array of B */
    setInputColors(c) { st.inColors = c; render(); },
    /** radius multipliers (1 = default): a number or an array of N */
    setSizes(s) { st.sizes = s; render(); },
    setLabels(l) { st.labels = l; render(); },
    setInputLabels(l) { st.inLabels = l; render(); },
    /** pairs: flat [i, j, ...] (Uint32Array or array; index >= N is input index - N); w: conductances (any scale,
     *  entries <= 0 skipped). Width and opacity grow with w / wmax; wmax defaults to max(w) (pass it to keep a
     *  fixed scale across frames). setEdges(null) clears. */
    setEdges(pairs, w, { wmax } = {}) {
      if (!pairs || !w || !w.length) { st.pairs = null; st.order = null; st.keep = null; thinNote.hidden = true; render(); return; }
      let m = 0; const idx = [];
      for (let e = 0; e < w.length; e++) if (w[e] > 0) { idx.push(e); if (w[e] > m) m = w[e]; }
      idx.sort((a, b) => w[a] - w[b]);
      st.pairs = pairs; st.w = w; st.wmax = wmax || m || 1; st.order = idx; st.keepW = -1; render();
    },
    /** ring neuron(s) i (or input(s) with kind "input") and darken their synapses; null clears. Does not call
     *  onHover / onSelect, so it is safe to drive from a linked panel. */
    setHighlight(i, kind = "neuron") {
      st.hi = new Set(i == null ? [] : Array.isArray(i) ? i : [i]); st.hiKind = kind; render();
    },
    /** a persistent dashed ring (e.g. the clicked neuron); null clears */
    setSelected(i, kind = "neuron") { st.sel = i; st.selKind = kind; render(); },
    setCorner(html) { corner.innerHTML = html || ""; if (!corner.parentNode) wrap.append(corner); },
    /** layout in [0, 1]^2: {neurons: Float32Array(2N), inputs: Float32Array(2B)} */
    positions() { return { neurons: posN.slice(), inputs: posIn.slice() }; },
    render,
    destroy() { ro.disconnect(); wrap.remove(); },
  };
}
