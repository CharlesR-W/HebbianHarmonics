// 3D point / line viewer on a Canvas 2D (hand-written, no dependencies). Built for the centrepiece figure:
// interior neurons as points (circles), boundary neurons as labelled anchors (squares), synapses as segments
// with per-edge opacity, optional fading trails. Orbit by drag, zoom by wheel (or pinch / the buttons), gentle
// auto-rotate. Draws only when something changed or while rotating, and only while on screen.
// API and examples: DESIGN.md "view3d.js".
import { P } from "./palette.js";
import { h, animate, showTip, hideTip, tipRows, reducedMotion, spaceBadge } from "./ui.js";
import { hidpi } from "./charts.js";

// ---- simplex helpers
// Vertices of the regular simplex for B = 2 (segment), 3 (triangle in the z = 0 plane) or 4 (tetrahedron),
// centred on the origin with circumradius 1. Returns an array of [x, y, z].
export function simplexVertices(B) {
  if (B === 2) return [[-1, 0, 0], [1, 0, 0]];
  if (B === 3) return [90, 210, 330].map(a => [Math.cos(a * Math.PI / 180), Math.sin(a * Math.PI / 180), 0]);
  if (B === 4) { const s = 1 / Math.sqrt(3); return [[s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]]; }
  throw new Error(`simplexVertices: B = ${B} has no 3D simplex; project the embeddings instead`);
}
// Barycentric -> 3D: H is row-major N x B (rows are harmonic embeddings, each sums to 1); returns Float32Array(3N).
// Optional `lift` (Float32Array N) is added to z, e.g. to give the flat B = 3 triangle a third dimension.
export function baryToXYZ(H, B, verts = simplexVertices(B), out = null, lift = null) {
  const N = H.length / B; out = out || new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    let x = 0, y = 0, z = 0;
    for (let b = 0; b < B; b++) { const w = H[i * B + b]; x += w * verts[b][0]; y += w * verts[b][1]; z += w * verts[b][2]; }
    out[3 * i] = x; out[3 * i + 1] = y; out[3 * i + 2] = z + (lift ? lift[i] : 0);
  }
  return out;
}

const ICON = {
  rotate: '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><ellipse cx="8" cy="8" rx="6.2" ry="2.6" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M12.6 3.6l1.6 1.9-2.4.6" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/><circle cx="8" cy="8" r="1.4" fill="currentColor"/></svg>',
  home: '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path d="M8 2.5L14 13H2z" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/><circle cx="8" cy="9.4" r="1.3" fill="currentColor"/></svg>',
};

// view3d(el, {height?: 460, yaw?: -0.5, pitch?: 0.95, zoom?: 1, autoRotate?: false, rotateSpeed?: 0.18 (rad/s),
//             center?: [x,y,z], radius?: r, hull?: true, pointSize?: 4, edgeWidth?: 1, edgeAlpha?: 0.6,
//             edgeColor?: P.edge, trailColor?: P.fg, depthFade?: 0.35, hint?: true, tooltip?(i, kind) -> html|Node,
//             badge?: "input" | "function" | {kind, text}  (a space badge, top left; the corner note moves below it)})
export function view3d(el, o = {}) {
  const opt = { height: 460, yaw: -0.5, pitch: 0.95, zoom: 1, autoRotate: false, rotateSpeed: 0.18, hull: true, pointSize: 4,
    edgeWidth: 1, edgeAlpha: 0.6, edgeColor: P.edge, trailColor: P.fg, depthFade: 0.35, hint: true, ...o };
  const cv = h("canvas", { tabindex: "0", "aria-label": "3D view: drag to orbit, scroll to zoom" });
  const rotBtn = h("button", { type: "button", title: "Auto-rotate", "aria-label": "Auto-rotate", "aria-pressed": String(!!opt.autoRotate && !reducedMotion()), html: ICON.rotate });
  const homeBtn = h("button", { type: "button", title: "Reset the view", "aria-label": "Reset the view", html: ICON.home });
  const corner = h("div", { class: "v3-corner" });
  const badge = opt.badge ? (typeof opt.badge === "string" ? spaceBadge(opt.badge) : spaceBadge(opt.badge.kind, opt.badge.text)) : null;
  if (badge) badge.classList.add("v3-badge");
  const box = h("div", { class: "v3" + (badge ? " has-badge" : ""), style: { height: opt.height + "px" } }, cv, badge, corner, h("div", { class: "v3-btns" }, rotBtn, homeBtn),
    opt.hint ? h("div", { class: "v3-hint" }, h("span", { class: "mouse" }, "drag to orbit · scroll to zoom"), h("span", { class: "touch" }, "drag sideways to orbit · pinch to zoom")) : null);
  el.replaceChildren(box);

  // ---- scene state
  let N = 0, pos = new Float32Array(0), cols = P.interior, sizes = null;
  let anchors = [], guides = [];
  let pairs = new Uint32Array(0), alphas = new Float32Array(0), edgeGroup = null, edgePalette = null;
  let trailN = 0, trail = [];
  let center = opt.center || null, radius = opt.radius || null;
  const cam = { yaw: opt.yaw, pitch: opt.pitch, zoom: opt.zoom };
  let rotating = !!opt.autoRotate && !reducedMotion(), dirty = true, W = 0, H = opt.height, ctx = null;
  let hoverCb = null, cameraCb = null, hovered = null, highlighted = null;
  // projected screen coordinates (x, y, scale, depth) for points then anchors
  let sx = new Float32Array(0), sy = new Float32Array(0), sd = new Float32Array(0), ss = new Float32Array(0), order = new Uint32Array(0);

  function fit() {
    const all = [];
    for (let i = 0; i < N; i++) all.push([pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]);
    anchors.forEach(a => all.push(a.xyz));
    if (!all.length) return;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    all.forEach(p => p.forEach((v, k) => { if (Number.isFinite(v)) { lo[k] = Math.min(lo[k], v); hi[k] = Math.max(hi[k], v); } }));
    center = opt.center || lo.map((v, k) => (v + hi[k]) / 2);
    radius = opt.radius || Math.max(1e-6, ...all.map(p => Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2])));
  }
  function project() {
    const M = N + anchors.length;
    if (sx.length < M) { sx = new Float32Array(M); sy = new Float32Array(M); sd = new Float32Array(M); ss = new Float32Array(M); }
    const cy = Math.cos(cam.yaw), sny = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const D = 4.2, f = Math.max(40, Math.min(W, H) / 2 - 34) * cam.zoom / radius, ox = W / 2, oy = H / 2 + (opt.hint ? -6 : 0);
    const P3 = (x, y, z, k) => {
      x -= center[0]; y -= center[1]; z -= center[2];
      const x1 = cy * x - sny * y, y1 = sny * x + cy * y;
      const up = cp * z + sp * y1, dep = -sp * z + cp * y1;            // dep > 0: away from the camera
      const s = D / (D + dep / radius);
      sx[k] = ox + x1 * f * s; sy[k] = oy - up * f * s; sd[k] = dep / radius; ss[k] = s;
    };
    for (let i = 0; i < N; i++) P3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2], i);
    anchors.forEach((a, j) => P3(a.xyz[0], a.xyz[1], a.xyz[2], N + j));
    return P3;
  }
  const colorOf = i => (typeof cols === "string" ? cols : cols[i] || P.interior);
  const sizeOf = i => (sizes == null ? opt.pointSize : typeof sizes === "number" ? sizes : sizes[i]);

  function draw() {
    dirty = false;
    if (!W || !center) return;
    ctx = ctx || cv.getContext("2d");
    const dpr = cv.width / W;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const P3 = project();
    ctx.lineCap = "round";
    // hull (simplex edges between anchors) and guides
    if (opt.hull && anchors.length > 1) {
      ctx.strokeStyle = "#c9c8c1"; ctx.lineWidth = 1; ctx.beginPath();
      for (let a = 0; a < anchors.length; a++) for (let b = a + 1; b < anchors.length; b++) { ctx.moveTo(sx[N + a], sy[N + a]); ctx.lineTo(sx[N + b], sy[N + b]); }
      ctx.stroke();
    }
    if (guides.length) {
      const M = N + anchors.length;
      if (sx.length < M + 2) { const grow = a => { const b = new Float32Array(M + 2); b.set(a); return b; }; sx = grow(sx); sy = grow(sy); sd = grow(sd); ss = grow(ss); }
      for (const g of guides) {
        P3(g.a[0], g.a[1], g.a[2], M); P3(g.b[0], g.b[1], g.b[2], M + 1);   // two scratch slots past the anchors
        ctx.strokeStyle = g.color || P.mute2; ctx.lineWidth = 1; ctx.setLineDash(g.dash ? [3, 3] : []);
        ctx.beginPath(); ctx.moveTo(sx[M], sy[M]); ctx.lineTo(sx[M + 1], sy[M + 1]); ctx.stroke(); ctx.setLineDash([]);
        if (g.label) { ctx.fillStyle = P.mute; ctx.font = "11px Inter, system-ui, sans-serif"; ctx.textAlign = "left"; ctx.fillText(g.label, sx[M + 1] + 5, sy[M + 1] + 4); }
      }
    }
    // trails: segments from each point's current position back through the stored frames (newest last in
    // `trail`), one stroke per age so older segments are fainter
    if (trailN && trail.length) {
      const K = trail.length, px = new Float32Array(N * (K + 1)), py = new Float32Array(N * (K + 1));
      for (let i = 0; i < N; i++) { px[i] = sx[i]; py[i] = sy[i]; }
      for (let a = 1; a <= K; a++) {
        const fr = trail[K - a];
        for (let i = 0; i < N; i++) { P3(fr[3 * i], fr[3 * i + 1], fr[3 * i + 2], i); px[a * N + i] = sx[i]; py[a * N + i] = sy[i]; }
      }
      project();   // restore the current projection
      ctx.lineWidth = 1; ctx.strokeStyle = opt.trailColor;
      for (let a = 0; a < K; a++) {
        ctx.globalAlpha = 0.34 * (1 - a / K);
        ctx.beginPath();
        for (let i = 0; i < N; i++) { const p = a * N + i, q = p + N; ctx.moveTo(px[p], py[p]); ctx.lineTo(px[q], py[q]); }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    // edges: bucketed by (colour group, opacity level) so ~5,000 segments cost a few dozen strokes
    const M = alphas.length;
    if (M) {
      const LEVELS = 24, groups = edgePalette ? edgePalette.length : 1;
      const buckets = Array.from({ length: groups * LEVELS }, () => []);
      for (let e = 0; e < M; e++) {
        const a = alphas[e]; if (!(a > 0.004)) continue;
        const lv = Math.min(LEVELS - 1, Math.floor(a * LEVELS)), g = edgeGroup ? edgeGroup[e] : 0;
        buckets[g * LEVELS + lv].push(e);
      }
      ctx.lineWidth = opt.edgeWidth;
      for (let b = 0; b < buckets.length; b++) {
        const list = buckets[b]; if (!list.length) continue;
        const lv = b % LEVELS, g = (b / LEVELS) | 0;
        ctx.strokeStyle = edgePalette ? edgePalette[g] : opt.edgeColor;
        ctx.globalAlpha = opt.edgeAlpha * (lv + 0.5) / LEVELS;
        ctx.beginPath();
        for (const e of list) { const i = pairs[2 * e], j = pairs[2 * e + 1]; ctx.moveTo(sx[i], sy[i]); ctx.lineTo(sx[j], sy[j]); }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    // interior points, far to near, with a surface ring and a mild depth fade
    if (order.length < N) order = new Uint32Array(N);
    for (let i = 0; i < N; i++) order[i] = i;
    const ord = order.subarray(0, N).sort((a, b) => sd[b] - sd[a]);
    ctx.lineWidth = 1; ctx.strokeStyle = P.surface;
    for (const i of ord) {
      if (!Number.isFinite(sx[i])) continue;
      const r = sizeOf(i) * ss[i];
      ctx.globalAlpha = 1 - opt.depthFade * Math.max(0, Math.min(1, (sd[i] + 1) / 2));
      ctx.beginPath(); ctx.arc(sx[i], sy[i], r, 0, 6.2832); ctx.fillStyle = colorOf(i); ctx.fill(); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // anchors (boundary neurons): squares with an ink outline and a label pushed away from the centre
    const ccx = W / 2, ccy = H / 2;
    anchors.forEach((a, j) => {
      const k = N + j, s = (a.size || 6) * Math.sqrt(ss[k]);
      ctx.fillStyle = a.color || P.boundary; ctx.fillRect(sx[k] - s, sy[k] - s, 2 * s, 2 * s);
      ctx.lineWidth = 1.5; ctx.strokeStyle = P.fg; ctx.strokeRect(sx[k] - s, sy[k] - s, 2 * s, 2 * s);
      if (a.label) {
        let dx = sx[k] - ccx, dy = sy[k] - ccy; const L = Math.hypot(dx, dy) || 1; dx /= L; dy /= L;
        ctx.font = "600 13px Inter, system-ui, sans-serif"; ctx.fillStyle = P.fg;
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        const tw = ctx.measureText(a.label).width, ax = dx > 0.35 ? 0 : dx < -0.35 ? -tw : -tw / 2;   // left / right / centred
        const lx = Math.max(4, Math.min(W - 4 - tw, sx[k] + dx * (s + 9) + ax)), ly = Math.max(9, Math.min(H - 9, sy[k] + dy * (s + 10)));
        ctx.fillText(a.label, lx, ly);
        ctx.textBaseline = "alphabetic";
      }
    });
    // hover / highlight rings
    const ring = (k, w) => { if (k == null || k >= N + anchors.length) return; const r = (k < N ? sizeOf(k) * ss[k] : (anchors[k - N].size || 6) * 1.4) + 3.5; ctx.beginPath(); ctx.arc(sx[k], sy[k], r, 0, 6.2832); ctx.lineWidth = w; ctx.strokeStyle = P.fg; ctx.stroke(); };
    (Array.isArray(highlighted) ? highlighted : highlighted != null ? [highlighted] : []).forEach(k => ring(k, 1.5));
    ring(hovered, 2);
  }

  function resize() {
    const w = Math.floor(box.clientWidth); if (!w) return;
    const hh = Math.round(Math.min(opt.height, Math.max(280, w * 0.92)));     // shorter on phones
    if (w === W && hh === H && ctx) return;
    box.style.height = hh + "px";
    W = w; H = hh; hidpi(cv, W, H); ctx = cv.getContext("2d"); touch();
  }
  // mark for redraw; when the loop is not running (off screen) paint once anyway, so the view is never blank
  let pending = false;
  function touch() {
    dirty = true;
    if (!loop || loop.running || pending) return;
    pending = true; requestAnimationFrame(() => { pending = false; if (dirty) draw(); });
  }
  let loop = null;
  const ro = new ResizeObserver(resize); ro.observe(box);
  loop = animate(box, dt => {
    if (rotating) { cam.yaw += opt.rotateSpeed * dt; dirty = true; }
    if (dirty) draw();
  });

  // ---- interaction: orbit, zoom, hover, pinch
  let drag = null; const touches = new Map(); let pinch0 = null;
  cv.addEventListener("pointerdown", ev => {
    touches.set(ev.pointerId, [ev.clientX, ev.clientY]);
    if (touches.size === 2) { const [a, b] = [...touches.values()]; pinch0 = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), z: cam.zoom }; drag = null; return; }
    drag = { x: ev.clientX, y: ev.clientY, yaw: cam.yaw, pitch: cam.pitch, moved: false };
    cv.setPointerCapture(ev.pointerId); cv.classList.add("dragging"); hideTip();
  });
  cv.addEventListener("pointermove", ev => {
    if (touches.has(ev.pointerId)) touches.set(ev.pointerId, [ev.clientX, ev.clientY]);
    if (pinch0 && touches.size === 2) { const [a, b] = [...touches.values()]; cam.zoom = Math.max(.3, Math.min(6, pinch0.z * Math.hypot(a[0] - b[0], a[1] - b[1]) / pinch0.d)); dirty = true; return; }
    if (drag) {
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      cam.yaw = drag.yaw + dx * 0.008; cam.pitch = Math.max(-0.1, Math.min(Math.PI / 2, drag.pitch + dy * 0.008));
      dirty = true; cameraCb?.({ ...cam }); return;
    }
    // hover: nearest projected point or anchor within 10 px
    const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = null, bd = 100;
    for (let k = 0; k < N + anchors.length; k++) { const d = (sx[k] - x) ** 2 + (sy[k] - y) ** 2; if (d < bd) { bd = d; best = k; } }
    if (best !== hovered) { hovered = best; dirty = true; hoverCb?.(best == null ? null : best < N ? best : null, best == null ? null : best < N ? { kind: "point", index: best } : { kind: "anchor", index: best - N }); }
    if (best == null) { hideTip(); return; }
    const kind = best < N ? "point" : "anchor", idx = best < N ? best : best - N;
    const content = opt.tooltip ? opt.tooltip(idx, kind) : tipRows(kind === "point" ? `interior neuron ${idx}` : `boundary neuron ${anchors[idx].label || idx}`, [], null);
    if (content) showTip(ev.clientX, ev.clientY, content);
  });
  const end = ev => {
    touches.delete(ev.pointerId); if (touches.size < 2) pinch0 = null;
    if (drag) { cv.classList.remove("dragging"); drag = null; }
  };
  cv.addEventListener("pointerup", end); cv.addEventListener("pointercancel", end);
  cv.addEventListener("pointerleave", () => { if (hovered != null) { hovered = null; dirty = true; hoverCb?.(null, null); } hideTip(); });
  cv.addEventListener("wheel", ev => { ev.preventDefault(); cam.zoom = Math.max(.3, Math.min(6, cam.zoom * Math.exp(-ev.deltaY * 0.0015))); dirty = true; cameraCb?.({ ...cam }); }, { passive: false });
  cv.addEventListener("keydown", ev => {
    const step = { ArrowLeft: [-.1, 0], ArrowRight: [.1, 0], ArrowUp: [0, -.1], ArrowDown: [0, .1] }[ev.key];
    if (step) { ev.preventDefault(); cam.yaw += step[0]; cam.pitch = Math.max(-0.1, Math.min(Math.PI / 2, cam.pitch + step[1])); dirty = true; }
    if (ev.key === "+" || ev.key === "=") { cam.zoom *= 1.15; dirty = true; }
    if (ev.key === "-") { cam.zoom /= 1.15; dirty = true; }
  });
  rotBtn.addEventListener("click", () => api.setAutoRotate(!rotating));
  homeBtn.addEventListener("click", () => { cam.yaw = opt.yaw; cam.pitch = opt.pitch; cam.zoom = opt.zoom; dirty = true; cameraCb?.({ ...cam }); });

  const api = {
    el: box,
    // xyz: Float32Array(3N). colors: one CSS colour or an array of N. sizes: radius px (number or array of N).
    // Call every frame if positions change; with trails on, each call also pushes the previous frame to the trail.
    setPoints(xyz, colors, sz) {
      const newN = xyz.length / 3;
      if (trailN && newN === N && pos.length) { trail.push(Float32Array.from(pos)); if (trail.length > trailN) trail.shift(); }
      if (newN !== N) trail = [];
      N = newN; pos = xyz;
      if (colors !== undefined) cols = colors;
      if (sz !== undefined) sizes = sz;
      if (!center) fit();
      touch();
    },
    setColors(colors) { cols = colors; touch(); },
    // pairs: Uint32Array(2M) of indices; an index >= N (the number of points) means anchor (index - N).
    // alpha: Float32Array(M) in [0, 1] (multiplied by opts.edgeAlpha). group/palette: optional per-edge colour.
    setEdges(p, a, { group = null, palette = null } = {}) { pairs = p; alphas = a; edgeGroup = group; edgePalette = palette; touch(); },
    // [{xyz: [x,y,z], label, color, size?}]  boundary neurons
    setAnchors(list) { anchors = list || []; if (!center || !opt.center) fit(); touch(); },
    // [{a: [x,y,z], b: [x,y,z], label?, dash?, color?}]  reference segments (axes, a "height" scale, ...)
    setGuides(list) { guides = list || []; touch(); },
    setTrails(n) { trailN = Math.max(0, n | 0); trail = []; touch(); },
    clearTrails() { trail = []; touch(); },
    highlight(k) { highlighted = k; touch(); },
    onHover(cb) { hoverCb = cb; },
    onCamera(cb) { cameraCb = cb; },
    setCorner(html) { corner.innerHTML = html || ""; },
    setAutoRotate(on) { rotating = !!on; rotBtn.setAttribute("aria-pressed", String(rotating)); touch(); },
    getCamera: () => ({ ...cam }),
    setCamera(c) { Object.assign(cam, c); touch(); },
    fit() { center = null; radius = null; fit(); touch(); },
    render() { touch(); },
    destroy() { loop.stop(); ro.disconnect(); el.replaceChildren(); },
  };
  resize();
  return api;
}
