// Film player for precomputed big runs (site/data/big/, format: docs/PLAN_curation.md "Film data format" and
// site/data/big/README.md). Lazy-loads <id>.json + <id>.bin, plays / scrubs on the frame axis (frames are spaced
// in log time), and draws the same state twice, side by side with linked highlighting:
//   left:  input space (space "simplex": rows of H in the triangle of inputs) or response-function space
//          (space "pca": the first two principal axes, fixed by the final frame; optional 3-D with the third axis);
//   right: neuron space (netview.js; random layout seeded by the film's network_layout_seed; the film's display
//          edges, plus the strongest input synapses when the file has them).
// Colours: the neuron's cluster at the end of the run (final.labels, re-ranked by size so the largest is cat-0).
//
//   const fp = filmPlayer(el, { filter: e => e.family === "A", id: "A_cov_ind_th0.4" });
//   fp.load("B_cov_ring16_th0.5"); fp.go(10); fp.play(); fp.pause();
import * as ui from "./ui.js";
import { P, catColor, boundaryColor } from "./palette.js";
import { hidpi } from "./charts.js";
import { netview } from "./netview.js";
import { view3d, simplexVertices } from "./view3d.js";

const DTYPES = { uint8: Uint8Array, uint16: Uint16Array, int16: Int16Array, uint32: Uint32Array, int32: Int32Array, float32: Float32Array };
const SUBS = "₀₁₂₃₄₅₆₇₈₉", sub = n => String(n).split("").map(c => SUBS[+c]).join("");
const indexCache = new Map(), filmCache = new Map();

/** loadIndex(base = "data/big/") -> Promise<[{id, title, family, rule, nonlin, B, N, file, bytes}]> */
export function loadIndex(base = "data/big/") {
  if (!indexCache.has(base)) indexCache.set(base, fetch(base + "index.json", { cache: "no-cache" }).then(r => { if (!r.ok) throw new Error(`${base}index.json: HTTP ${r.status}`); return r.json(); }));
  return indexCache.get(base);
}

/** loadFilm(id, base = "data/big/") -> Promise<film>: the JSON metadata plus decoded accessors:
 *  film.F, film.coord(f) -> Float32Array(N*D) (decoded with lo/hi), film.H(f) -> Float32Array(N*B) | null,
 *  film.edges(f) -> {pairs: Uint32Array(2E), w: Float32Array(E)}, film.inW(f) -> Float32Array(N*B) | null. */
export function loadFilm(id, base = "data/big/") {
  const key = base + id;
  if (!filmCache.has(key)) filmCache.set(key, (async () => {
    const r = await fetch(`${base}${id}.json`, { cache: "no-cache" });
    if (!r.ok) throw new Error(`${id}.json: HTTP ${r.status}`);
    const meta = await r.json();
    const rb = await fetch(base + (meta.bin || `${id}.bin`));
    if (!rb.ok) throw new Error(`${meta.bin || id + ".bin"}: HTTP ${rb.status}`);
    return decodeFilm(meta, await rb.arrayBuffer());
  })());
  return filmCache.get(key);
}

function view(buf, a) {
  const T = DTYPES[a.dtype]; if (!T) throw new Error(`unknown dtype ${a.dtype}`);
  const n = a.shape.reduce((x, y) => x * y, 1);
  if (a.offset % T.BYTES_PER_ELEMENT === 0) return new T(buf, a.offset, n);
  return new T(buf.slice(a.offset, a.offset + n * T.BYTES_PER_ELEMENT));       // unaligned: copy
}
export function decodeFilm(meta, buf) {
  const A = meta.arrays, F = meta.frames.length, N = meta.N, B = meta.B;
  const C = view(buf, A.coords), D = A.coords.shape[2], lo = A.coords.lo, hi = A.coords.hi;
  const Hq = A.H ? view(buf, A.H) : null;
  const E = A.edges ? view(buf, A.edges) : null, Ew = A.edge_w ? view(buf, A.edge_w) : null, Emax = A.edges ? A.edges.shape[1] : 0;
  const count = A.edges ? (A.edges.count || Array(F).fill(Emax)) : null;
  const Iw = A.in_w ? view(buf, A.in_w) : null;
  const Lb = A.labels ? view(buf, A.labels) : null;      // optional per-frame cluster labels (int16 [F, N])
  const wq = (v, wmax) => wmax * (v / 255) * (v / 255);
  return {
    ...meta, F, D,
    coord(f) { const out = new Float32Array(N * D), o = f * N * D; for (let k = 0; k < N * D; k++) { const d = k % D; out[k] = lo[d] + (hi[d] - lo[d]) * C[o + k] / 65535; } return out; },
    H(f) { if (!Hq) return null; const out = new Float32Array(N * B), o = f * N * B; for (let k = 0; k < N * B; k++) out[k] = Hq[o + k] / 255; return out; },
    edges(f) {
      if (!E) return { pairs: new Uint32Array(0), w: new Float32Array(0) };
      const n = Math.min(count[f] ?? Emax, Emax), pairs = new Uint32Array(2 * n), w = new Float32Array(n);
      for (let e = 0; e < n; e++) { pairs[2 * e] = E[2 * (f * Emax + e)]; pairs[2 * e + 1] = E[2 * (f * Emax + e) + 1]; w[e] = Ew ? wq(Ew[f * Emax + e], A.edge_w.wmax) : 1; }
      return { pairs, w };
    },
    labels(f) { return Lb ? Lb.subarray(f * N, (f + 1) * N) : null; },
    inW(f) { if (!Iw) return null; const out = new Float32Array(N * B), o = f * N * B; for (let k = 0; k < N * B; k++) out[k] = wq(Iw[o + k], A.in_w.wmax); return out; },
  };
}

// ---------------------------------------------------------------- 2-D input / function-space panel (canvas)
function planePanel(el, o) {
  const cv = ui.h("canvas", { role: "img", "aria-label": o.aria || "neurons in input space" });
  const badge = ui.spaceBadge(o.kind, o.badgeText); badge.classList.add("nv-badge");
  const note = ui.h("div", { class: "nv-corner u-help" });
  const wrap = ui.h("div", { class: "chart", style: { position: "relative" } }, cv, badge, note);
  el.append(wrap);
  let W = 0, Hh = 0, c = null, pts = null, cols = P.interior, hi = null, anchors = [], xr = [0, 1], yr = [0, 1], sx = null, sy = null, queued = false, hover = null;
  const r0 = () => Math.max(2.2, Math.min(5, 90 / Math.sqrt(Math.max(1, pts ? pts.length / 2 : 1))));
  function geom() {
    W = Math.floor(wrap.clientWidth); if (W <= 0) return false;
    note.style.display = W < 460 ? "none" : "";          // on phones the corner note would run into the badge
    Hh = Math.round(Math.max(260, Math.min(o.maxHeight || 440, W / (o.aspect || 1.1))));
    c = hidpi(cv, W, Hh);
    const pad = { l: 26, r: 26, t: 34, b: o.axisLabels ? 34 : 22 };
    const pw = W - pad.l - pad.r, ph = Hh - pad.t - pad.b;
    // equal units on both axes, centred
    const s = Math.min(pw / ((xr[1] - xr[0]) || 1), ph / ((yr[1] - yr[0]) || 1));
    const cx = pad.l + pw / 2, cy = pad.t + ph / 2, mx = (xr[0] + xr[1]) / 2, my = (yr[0] + yr[1]) / 2;
    sx = x => cx + (x - mx) * s; sy = y => cy - (y - my) * s;
    return true;
  }
  function draw() {
    queued = false;
    if (!c || Math.floor(wrap.clientWidth) !== W) { if (!geom()) return; }
    c.clearRect(0, 0, W, Hh);
    if (o.guides) o.guides(c, sx, sy);
    if (o.axisLabels) {
      c.font = "11px Inter, system-ui, sans-serif"; c.fillStyle = P.mute; c.textAlign = "center"; c.textBaseline = "alphabetic";
      c.fillText(o.axisLabels[0], W / 2, Hh - 6);
      c.save(); c.translate(11, Hh / 2); c.rotate(-Math.PI / 2); c.fillText(o.axisLabels[1], 0, 0); c.restore();
    }
    if (pts) {
      const n = pts.length / 2, r = r0();
      for (let i = 0; i < n; i++) {
        c.beginPath(); c.arc(sx(pts[2 * i]), sy(pts[2 * i + 1]), r, 0, 2 * Math.PI);
        c.fillStyle = Array.isArray(cols) ? cols[i] : cols; c.globalAlpha = 0.85; c.fill(); c.globalAlpha = 1;
      }
      if (o.countLabels) {
        // neurons that coincide on screen (a cluster of clones) get a count, so a collapsed cluster stays legible
        const bk = new Map();
        for (let i = 0; i < n; i++) { const x = sx(pts[2 * i]), y = sy(pts[2 * i + 1]), k = Math.round(x / 3) + "," + Math.round(y / 3); const e = bk.get(k); if (e) e.n++; else bk.set(k, { x, y, n: 1 }); }
        c.font = "600 11.5px Inter, system-ui, sans-serif"; c.fillStyle = P.fg; c.textBaseline = "middle";
        const placed = [];                               // label boxes already drawn: step down past any overlap
        for (const e of [...bk.values()].sort((p, q) => q.n - p.n)) {
          if (e.n < Math.max(5, 0.04 * n)) continue;
          const dx = W / 2 - e.x, dy = Hh / 2 - e.y, dl = Math.hypot(dx, dy) || 1, ox = e.x + (dx / dl) * (r + 14);
          let oy = e.y + (dy / dl) * (r + 14);
          const text = `${e.n} neurons`, tw = c.measureText(text).width, x0 = dx >= 0 ? ox - 4 : ox + 4 - tw;
          const hit = y => placed.some(b => x0 < b.x1 && x0 + tw > b.x0 && Math.abs(y - b.y) < 13);
          for (let k = 0; k < 4 && hit(oy); k++) oy += dy >= 0 ? -13 : 13;
          placed.push({ x0, x1: x0 + tw, y: oy });
          c.textAlign = dx >= 0 ? "left" : "right"; c.fillText(text, ox + (dx >= 0 ? -4 : 4), oy);
        }
      }
      if (hi != null && hi < n) { c.beginPath(); c.arc(sx(pts[2 * hi]), sy(pts[2 * hi + 1]), r + 4, 0, 2 * Math.PI); c.lineWidth = 2.2; c.strokeStyle = P.accent; c.stroke(); }
    }
    for (const a of anchors) {
      const x = sx(a.x), y = sy(a.y), s = 6;
      if (a.from) { c.beginPath(); c.moveTo(sx(a.from[0]), sy(a.from[1])); c.lineTo(x, y); c.strokeStyle = a.color; c.lineWidth = 1.2; c.stroke(); }
      c.fillStyle = a.color; c.fillRect(x - s, y - s, 2 * s, 2 * s); c.lineWidth = 1.4; c.strokeStyle = P.boundary; c.strokeRect(x - s, y - s, 2 * s, 2 * s);
      c.font = "600 12px Inter, system-ui, sans-serif"; c.fillStyle = P.fg; c.textBaseline = "middle"; c.textAlign = a.align || "left";
      c.fillText(a.label, x + (a.align === "right" ? -s - 4 : a.align === "center" ? 0 : s + 4), y + (a.dy || 0));
    }
  }
  const render = () => { if (!queued) { queued = true; requestAnimationFrame(draw); } };
  cv.addEventListener("pointermove", ev => {
    if (!pts) return;
    const rc = cv.getBoundingClientRect(), x = ev.clientX - rc.left, y = ev.clientY - rc.top;
    let best = null, bd = 144;
    for (let i = 0; i < pts.length / 2; i++) { const d = (sx(pts[2 * i]) - x) ** 2 + (sy(pts[2 * i + 1]) - y) ** 2; if (d < bd) { bd = d; best = i; } }
    if (best !== hover) { hover = best; o.onHover?.(best); }
    const tip = best != null && o.tooltip ? o.tooltip(best) : null;
    if (tip) ui.showTip(ev.clientX, ev.clientY, tip); else ui.hideTip();
  });
  cv.addEventListener("pointerleave", () => { ui.hideTip(); if (hover != null) { hover = null; o.onHover?.(null); } });
  const ro = new ResizeObserver(() => { if (Math.floor(wrap.clientWidth) !== W) { c = null; render(); } });
  ro.observe(wrap);
  return {
    el: wrap,
    setRange(x, y) { xr = x; yr = y; c = null; render(); },
    setPoints(p, colors) { pts = p; if (colors) cols = colors; render(); },
    setAnchors(a) { anchors = a; render(); },
    highlight(i) { hi = i; render(); },
    setNote(html) { note.innerHTML = html || ""; ui.renderMath(note); },
    render, destroy() { ro.disconnect(); wrap.remove(); },
  };
}

/** filmPlayer(el, opts) -> {load(id) -> Promise, go(f), play(), pause(), get frame, get film, destroy()}
 *  opts: base = "data/big/", id (film to show first; default: the first in the index that passes `filter`),
 *        filter(entry) -> bool (which films the picker offers), picker = true (a select over the index),
 *        fps = 6 (frames per second while playing), inputEdges = 600 (strongest input synapses drawn in neuron
 *        space, 0 for none), height = 420 (max panel height, px), onFrame(f, film),
 *        colors(film, clusterColors) -> css[] | null (override the per-neuron colours, e.g. barycentric for B = 3),
 *        countLabels = false (label neurons that coincide on screen, a cluster of clones, with their number),
 *        frameLabels = false (colour by each frame's own cluster labels when the film has a `labels` array,
 *          0 = largest; else by final.labels), frameLabel(frame, f, film) -> string (the transport label; default
 *          "t = …"), readout(frame, f, film) -> html (the line under the transport), spacing (sentence on how the
 *          frames are spaced; default "Frames are spaced in log time."). */
export function filmPlayer(el, o = {}) {
  const base = o.base || "data/big/", fps = o.fps || 6;
  const root = ui.h("div", { class: "filmplayer" });
  el.replaceChildren(root);
  const head = ui.h("div", { class: "fp-head" }), pairEl = ui.h("div", { class: "space-pair" }), left = ui.h("div"), right = ui.h("div");
  pairEl.append(left, right);
  const tr = ui.h("div", { class: "u-transport fp-transport" });
  const playBtn = ui.h("button", { type: "button", class: "u-play", "aria-label": "Play", html: '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="currentColor"/></svg>' });
  const range = ui.h("input", { type: "range", min: 0, max: 0, step: 1, value: 0, "aria-label": "frame", style: { flex: "1 1 180px", minWidth: "120px" } });
  const tLab = ui.h("span", { class: "u-time", style: { fontFamily: "var(--mono)", minWidth: "8.5em" } });
  tr.append(playBtn, range, tLab);
  const readout = ui.h("div", { class: "fp-readout u-help" }), note = ui.h("div", { class: "fp-note u-help" });
  root.append(head, pairEl, tr, readout, note);

  let spaceKind = "input", spaceText = "input space", v3host = null, film = null, f = 0, playing = false, acc = 0, nv = null, pl = null, v3 = null, colors = null, labels = null, sizeRank = null, use3d = false, token = 0;
  const loop = ui.animate(root, dt => {
    if (!playing || !film) return;
    acc += dt * fps;
    if (acc >= 1) { const n = Math.floor(acc); acc -= n; if (f + n >= film.F - 1) { go(film.F - 1); pause(); } else go(f + n); }
  });
  function play() { if (!film) return; if (f >= film.F - 1) go(0); playing = true; acc = 0; root.classList.add("playing"); tr.classList.add("playing"); playBtn.setAttribute("aria-label", "Pause"); playBtn.innerHTML = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor"/></svg>'; }
  function pause() { playing = false; tr.classList.remove("playing"); playBtn.setAttribute("aria-label", "Play"); playBtn.innerHTML = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="currentColor"/></svg>'; }
  playBtn.addEventListener("click", () => (playing ? pause() : play()));
  range.addEventListener("input", () => { pause(); go(+range.value); });

  // ---- picker
  let picker = null;
  async function buildPicker() {
    let idx = [];
    try { idx = await loadIndex(base); } catch (e) { head.replaceChildren(ui.h("p", { class: "u-help", text: `No precomputed runs yet (${e.message}).` })); return []; }
    idx = idx.filter(e => !o.filter || o.filter(e));
    if (o.picker !== false && idx.length > 1) {
      picker = ui.select({ label: "run", options: idx.map(e => ({ value: e.id, label: e.title || e.id })), value: o.id || idx[0].id, onChange: id => load(id) });
      head.replaceChildren(picker.el);
    }
    return idx;
  }

  function clusterColours(fl) {
    const lab = fl.final?.labels || [];
    const sizes = new Map(); lab.forEach(l => { if (l >= 0) sizes.set(l, (sizes.get(l) || 0) + 1); });
    const order = [...sizes.keys()].sort((a, b) => sizes.get(b) - sizes.get(a));
    sizeRank = new Map(order.map((l, k) => [l, k]));
    labels = lab; colors = Array.from({ length: fl.N }, (_, i) => (lab[i] >= 0 ? catColor(sizeRank.get(lab[i])) : P.other));
    if (o.colors) { const c = o.colors(fl, colors); if (c) colors = c; }       // chapter override, e.g. barycentric colour
  }
  // per-frame colours (opts.frameLabels): label l at frame f is already ranked by size (0 = largest)
  let perFrame = false;
  function frameColours(fl, k) {
    const lab = fl.labels?.(k); if (!lab) return false;
    labels = lab; sizeRank = null;
    colors = Array.from(lab, l => (l >= 0 ? catColor(l) : P.other));
    return true;
  }
  function clusterOf(i) {
    const l = labels?.[i]; if (l == null || l < 0) return null;
    if (perFrame) { const fr = film.frames[f]; return { rank: l, info: { size: fr.sizes?.[l], support: fr.supports?.[l] } }; }
    return { rank: sizeRank.get(l), info: film.final?.clusters?.[l] };
  }
  function tip(i, kind) {
    if (!film) return null;
    if (kind === "input") return ui.tipRows(`input g${sub(i + 1)}`, [], "Clamped to the stimulus.");
    const c = clusterOf(i), rows = [];
    if (c) {
      rows.push({ name: perFrame ? "cluster (this frame)" : "cluster (at the end)", value: `c${c.rank + 1}${c.info?.size ? `, ${c.info.size} neurons` : ""}`, color: colors[i] });
      if (c.info?.support) rows.push({ name: "support (inputs)", value: c.info.support.map(b => `g${sub(b + 1)}`).join(" ") });
      if (c.info?.perron_cos != null) rows.push({ name: "cos(centroid, Perron vector)", value: ui.fmt(c.info.perron_cos, 4) });
      if (c.info?.best_eig_cos != null) rows.push({ name: "best eigenvector cos", value: ui.fmt(c.info.best_eig_cos, 3) });
    } else rows.push({ name: "cluster", value: "none", color: P.other });
    const Hf = film.H(f);
    if (Hf && film.B <= 4) rows.push({ name: "h (this frame)", value: Array.from(Hf.subarray(i * film.B, (i + 1) * film.B), x => ui.fmtFixed(x, 2)).join(", ") });
    return ui.tipRows(`neuron ${i}`, rows);
  }
  const link = i => { nv?.setHighlight(i); pl?.highlight(i); v3?.highlight(i); };

  function buildViews(fl) {
    left.replaceChildren(); right.replaceChildren(); nv = pl = v3 = null; use3d = false;
    const B = fl.B, N = fl.N, isSimplex = fl.space === "simplex" && fl.D === 3;
    // linear neurons: a PCA of x_i = Sigma^{1/2} h_i is still input space (a stated projection, DESIGN 9a);
    // nonlinear neurons: response-function space
    const kind = fl.nonlin === "linear" ? "input" : /function/i.test(fl.space_label || "") || fl.space === "pca" ? "function" : "input";
    const badgeText = kind === "function" ? "response-function space" : "input space";
    spaceKind = kind; spaceText = badgeText;
    if (isSimplex) {
      const V = [[0.5, 1], [0, 0], [1, 0]].map(([x, y]) => [x, y * Math.sqrt(3) / 2]);
      pl = planePanel(left, { kind, badgeText, countLabels: o.countLabels, aspect: 1.1, maxHeight: o.height || 420, tooltip: i => tip(i, "neuron"), onHover: link,
        guides: (c, sx, sy) => { c.beginPath(); V.forEach(([x, y], k) => (k ? c.lineTo(sx(x), sy(y)) : c.moveTo(sx(x), sy(y)))); c.closePath(); c.strokeStyle = P.line; c.lineWidth = 1; c.stroke(); } });
      pl.setRange([-0.12, 1.12], [-0.1, Math.sqrt(3) / 2 + 0.1]);
      // input squares sit just outside their corners (a cluster at a corner would hide under them), with a leader
      const cx = 0.5, cy = Math.sqrt(3) / 6, out = ([x, y]) => [cx + 1.13 * (x - cx), cy + 1.13 * (y - cy)];
      pl.setAnchors(V.map((v, b) => { const [x, y] = out(v); return { x, y, from: v, label: `g${sub(b + 1)}`, color: boundaryColor(b, 3), align: b === 1 ? "right" : "left" }; }));
      pl.setNote("rows $h_i$ of $H$: barycentric position in the triangle of inputs");
      pl._toXY = cd => { const out = new Float32Array(2 * N); for (let i = 0; i < N; i++) { let x = 0, y = 0; for (let b = 0; b < 3; b++) { x += cd[3 * i + b] * V[b][0]; y += cd[3 * i + b] * V[b][1]; } out[2 * i] = x; out[2 * i + 1] = y; } return out; };
    } else {
      const ev = fl.pca?.explained_var, tot = fl.pca?.total_var;
      const pct = k => (ev && tot ? ` (${Math.round(100 * ev[k] / tot)}% of variance)` : "");
      pl = planePanel(left, { kind, badgeText, countLabels: o.countLabels, aspect: 1.1, maxHeight: o.height || 420, tooltip: i => tip(i, "neuron"), onHover: link, axisLabels: [`PC1${pct(0)}`, `PC2${pct(1)}`] });
      const lo = fl.arrays.coords.lo, hi = fl.arrays.coords.hi;
      pl.setRange([lo[0], hi[0]], [lo[1], hi[1]]);
      const fixedBy = /pooled/.test(fl.pca?.features || "") ? "fixed by all frames pooled" : "fixed by the final frame";
      pl.setNote(kind === "input" ? `principal axes of $\\Sigma^{1/2} h_i$ (distances approximate RMS response distances; exact before projection), ${fixedBy}` : `principal axes of the responses, ${fixedBy}`);
      pl._toXY = cd => { const D = fl.D, out = new Float32Array(2 * N); for (let i = 0; i < N; i++) { out[2 * i] = cd[D * i]; out[2 * i + 1] = cd[D * i + 1]; } return out; };
      // optional third axis, when it carries variance
      if (fl.D >= 3 && ev && tot && ev[2] / tot > 0.01) {
        const t = ui.synapseToggle({ label: "3-D (third axis)", onChange: on => { use3d = on; show3d(); } });
        left.append(ui.h("div", { style: { display: "flex", justifyContent: "flex-end" } }, t.el));
      }
    }
    nv = netview(right, { N, B, seed: fl.network_layout_seed ?? 1, aspect: 1.1, maxHeight: o.height || 420, minHeight: 260, edgeWidth: 2.4,
      tooltip: (i, k) => tip(i, k), onHover: (i, k) => { if (k !== "input") link(i); else link(null); } });
    nv.setColors(colors);
  }
  function show3d() {
    if (!use3d) { if (v3) { v3.destroy?.(); v3host?.remove(); v3 = null; v3host = null; } pl.el.hidden = false; go(f); return; }
    pl.el.hidden = true;
    const host = v3host = ui.h("div"); left.prepend(host);
    v3 = view3d(host, { height: Math.min(o.height || 420, 420), badge: { kind: spaceKind, text: spaceText }, hull: false, pointSize: 3,
      tooltip: (i, k) => (k === "point" ? tip(i, "neuron") : null) });
    v3.onHover(i => { nv?.setHighlight(i); });
    go(f);
  }

  function go(k) {
    if (!film) return;
    f = Math.max(0, Math.min(film.F - 1, k | 0)); range.value = f;
    const fr = film.frames[f], cd = film.coord(f);
    if (perFrame) { frameColours(film, f); nv.setColors(colors); }
    pl.setPoints(pl._toXY(cd), colors);
    if (v3) {
      const N = film.N, D = film.D, lo = film.arrays.coords.lo, hi = film.arrays.coords.hi, xyz = new Float32Array(3 * N);
      const sc = 2 / Math.max(...[0, 1, 2].map(d => (hi[d] - lo[d]) || 1));
      for (let i = 0; i < N; i++) for (let d = 0; d < 3; d++) xyz[3 * i + d] = (cd[D * i + d] - (lo[d] + hi[d]) / 2) * sc;
      v3.setPoints(xyz, colors);
    }
    // neuron space: the file's display edges (lateral), plus the strongest input synapses
    const { pairs, w } = film.edges(f), iw = (o.inputEdges ?? 600) > 0 ? film.inW(f) : null;
    if (iw) {
      const N = film.N, B = film.B, cap = o.inputEdges ?? 600, idx = [];
      let m = 0; for (const x of iw) m = Math.max(m, x);
      for (let q = 0; q < N * B; q++) if (iw[q] > 0.05 * m) idx.push(q);
      idx.sort((a, b) => iw[b] - iw[a]); idx.length = Math.min(idx.length, cap);
      // input synapses share the lateral scale only by rank: draw them at their own w / max w, mapped into the lateral range
      const wl = w.length ? Math.max(...w) : 1;
      const P2 = new Uint32Array(pairs.length + 2 * idx.length), W2 = new Float32Array(w.length + idx.length);
      P2.set(pairs); W2.set(w);
      idx.forEach((q, e) => { P2[pairs.length + 2 * e] = Math.floor(q / B); P2[pairs.length + 2 * e + 1] = N + (q % B); W2[w.length + e] = wl * iw[q] / m; });
      nv.setEdges(P2, W2);
    } else nv.setEdges(pairs, w);
    tLab.textContent = o.frameLabel ? o.frameLabel(fr, f, film) : `t = ${ui.fmt(fr.t, 3)}`;
    readout.innerHTML = o.readout ? o.readout(fr, f, film) : `frame ${f + 1} of ${film.F} · t = ${ui.fmt(fr.t, 3)} · residual ${ui.fmt(fr.residual, 2)} · live edges ${ui.fmtInt(fr.n_live)} · components ${ui.fmtInt(fr.n_components)}`;
    o.onFrame?.(f, film);
  }

  async function load(id) {
    const my = ++token; pause();
    readout.textContent = "loading…";
    let fl;
    try { fl = await loadFilm(id, base); } catch (e) { if (my === token) readout.textContent = `Could not load ${id}: ${e.message}`; return; }
    if (my !== token) return;
    film = fl; clusterColours(fl);
    perFrame = !!o.frameLabels && frameColours(fl, fl.F - 1);
    buildViews(fl);
    range.max = String(fl.F - 1);
    const cl = fl.final?.clusters || [];
    note.innerHTML = `${fl.title || fl.id}. Panels: ${spaceText}${fl.space === "pca" ? " (principal axes)" : fl.space === "simplex" ? " (harmonic embedding)" : ""}, and neuron space (layout seed ${fl.network_layout_seed}). ` +
      `${o.spacing ?? "Frames are spaced in log time."} ${o.colors ? "Colour: see the caption" : perFrame ? "Colour: cluster at the frame shown (largest first)" : `Colour: cluster at the end of the run (${cl.length} cluster${cl.length === 1 ? "" : "s"}, largest first)`}. ` +
      `<span class="mute">Run ${fl.id}, B = ${fl.B}, N = ${fl.N}, seed ${fl.seed}; ${fl.script || ""}.</span>`;
    if (picker && picker.get() !== id) picker.set(id);
    go(fl.F - 1);
  }

  (async () => { const idx = await buildPicker(); const first = o.id || idx[0]?.id; if (first) load(first); else readout.textContent = "No precomputed runs match."; })();
  return {
    load, go, play, pause,
    get frame() { return f; }, get film() { return film; },
    destroy() { loop.stop(); root.remove(); },
  };
}
