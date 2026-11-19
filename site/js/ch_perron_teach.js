// Teaching figures for the Perron-parts chapter (#perron). mountTeach(ctx) fills whichever of these slots exist:
//   [data-slot="fig-a"]      "Thresholding the covariance": Σ and A = Σ − θ11ᵀ, the dendrogram of Σ cut at θ,
//                            the stable parts outlined on A's diagonal (B = 8 presets).
//   [data-slot="fig-parts3"] "Where clusters can sit, three inputs": every candidate part in the triangle.
//   [data-slot="fig-hier"]   "The part diagram": stable parts against θ (drawPartDiagram), Perron vectors at the cursor.
// drawPartDiagram(el, opts) is exported for precomputed data (e.g. 16 inputs plus an overlay of populated parts).
// The maths (presets, exact part enumeration ported from hebb/big/common.py, dendrograms) is in ch_perron_teach_model.js.
import * as ui from "./ui.js";
import { hidpi, niceTicks } from "./charts.js";
import { P, colorOf, cmap, rgbToCss, catColor } from "./palette.js";
import { eigh } from "./engine.js";
import * as M from "./ch_perron_teach_model.js";

const NS = "http://www.w3.org/2000/svg";
const FONT = "Inter, system-ui, sans-serif";
const SUBD = "₀₁₂₃₄₅₆₇₈₉";
export const inputLabel = b => "g" + String(b + 1).split("").map(d => SUBD[d]).join("");
const fx = (x, d = 2) => ui.fmtFixed(x, d);

function svg(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function stext(parent, x, y, s, attrs = {}) {
  const t = svg("text", { x, y, "font-size": 11, fill: P.ink2, "font-family": FONT, ...attrs }, parent);
  t.textContent = s;
  return t;
}

// "{g₁–g₄}" for runs of three or more consecutive inputs, else listed: "{g₁, g₂, g₇, g₈}"
export function supportName(S) {
  const s = [...S].sort((a, b) => a - b), parts = [];
  for (let i = 0; i < s.length;) {
    let j = i;
    while (j + 1 < s.length && s[j + 1] === s[j] + 1) j++;
    if (j - i >= 2) parts.push(`${inputLabel(s[i])}–${inputLabel(s[j])}`);
    else for (let k = i; k <= j; k++) parts.push(inputLabel(s[k]));
    i = j + 1;
  }
  return `{${parts.join(", ")}}`;
}
// contiguous runs of row positions [[r0, r1], ...] of a support, in display (leaf) order
function rowRuns(S, pos) {
  const r = S.map(b => pos[b]).sort((a, b) => a - b), out = [];
  for (let i = 0; i < r.length;) { let j = i; while (j + 1 < r.length && r[j + 1] === r[j] + 1) j++; out.push([r[i], r[j]]); i = j + 1; }
  return out;
}

let styled = false;
function injectStyle() {
  if (styled) return; styled = true;
  document.head.append(ui.h("style", { text: `
.pt-ctl { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr)); gap: 10px 28px; align-items: end; margin: 4px 0 12px; }
.pt-ctl .seg { flex-wrap: wrap; }
.pt-ctl .seg button { font-family: var(--sans); }
.pt-key3 { display: flex; flex-wrap: wrap; gap: 6px 18px; font: 11.5px/1.3 var(--sans); color: var(--ink-2); }
.pt-key3 span { display: inline-flex; align-items: center; gap: 6px; }
.pt-panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 250px), 1fr)); gap: 14px 22px; align-items: start; }
.pt-panels > * { min-width: 0; }
.pt-panels svg, .pt-tri svg { width: 100%; height: auto; display: block; }
.pt-read { font: 12.5px/1.5 var(--sans); color: var(--ink-2); margin: 10px 0 2px; }
.pt-read b { color: var(--fg); font-weight: 600; }
.pt-foot { display: flex; flex-wrap: wrap; gap: 8px 26px; align-items: center; margin-top: 8px; }
.pt-tri { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 14px 28px; align-items: start; }
@media (max-width: 700px) { .pt-tri { grid-template-columns: minmax(0, 1fr); } }
.pt-tab { width: 100%; border-collapse: collapse; font: 12px/1.35 var(--sans); margin-top: 8px; }
.pt-tab th { text-align: left; font-weight: 600; color: var(--mute); border-bottom: 1px solid var(--line); padding: 3px 4px; }
.pt-tab td { padding: 3px 4px; border-bottom: 1px solid var(--line-2, var(--line)); color: var(--ink-2); white-space: nowrap; }
.pt-tab td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pt-tab tr.faint td { color: var(--mute-2); }
.pt-tab tr.stable td { color: var(--fg); }
.pt-pd { position: relative; touch-action: pan-y; }
.pt-pd canvas { display: block; cursor: crosshair; }
.pt-pd-key { display: flex; flex-wrap: wrap; gap: 6px 20px; align-items: center; font: 11.5px/1.3 var(--sans); color: var(--ink-2); margin-top: 6px; }
.pt-pd-key .pt-grad { display: inline-block; width: 110px; height: 10px; border-radius: 2px; vertical-align: middle; margin: 0 6px; }
.pt-pd-key i { display: inline-block; vertical-align: middle; margin-right: 6px; }
` }));
}

// ======================================================================= mountTeach
// mountTeach(ctx): ctx is the chapter context (ctx.slot, ctx.param, ctx.setParams, ctx.renderMath). Mounts each of
// the three figures whose slot is present; returns {a, parts3, hier} (each null or the figure's {setTheta, ...}).
// URL keys: fig A `pa` (preset), `ta` (θ); fig parts3 `t3`, `r3`; fig hier `ph`, `th`.
export function mountTeach(ctx) {
  injectStyle();
  const out = { a: null, parts3: null, hier: null };
  const sa = ctx.slot("fig-a"), s3 = ctx.slot("fig-parts3"), sh = ctx.slot("fig-hier");
  if (sa) out.a = mountA(sa, ctx);
  if (s3) out.parts3 = mountParts3(s3, ctx);
  if (sh) out.hier = mountHier(sh, ctx);
  return out;
}

const PRESET_IDS = M.PRESETS.map(p => p.id);
const presetCache = {};
function preset(id) {
  if (!presetCache[id]) {
    const { B, Sigma } = M.presetSigma(id);
    const { linkage, leafOrder } = M.averageLinkage(Sigma, B);
    const pos = new Array(B); leafOrder.forEach((b, r) => (pos[b] = r));
    presetCache[id] = { id, B, Sigma, linkage, leafOrder, pos, grid: null };
  }
  return presetCache[id];
}
function presetControl(ctx, value, onChange) {
  const note = ui.h("div", { class: "u-help" });
  const setNote = id => { note.innerHTML = M.PRESETS.find(p => p.id === id)?.note || ""; ctx.renderMath(note); };
  const seg = ui.segmented({ label: "inputs $\\Sigma$", options: M.PRESETS.map(p => ({ value: p.id, label: p.label })), value,
    onChange: v => { setNote(v); onChange(v); } });
  seg.el.append(note); setNote(value);
  return { el: seg.el, set: v => { seg.set(v); setNote(v); } };
}
// stable parts sorted by their first row in leaf order, with a colour each
function coloured(parts, pos) {
  return parts.map(p => ({ ...p, first: Math.min(...p.S.map(b => pos[b])) }))
    .sort((a, b) => a.first - b.first || a.S.length - b.S.length)
    .map((p, k) => ({ ...p, k, color: catColor(k), key: M.keyOf(p.S) }));
}

// ======================================================================= F1: thresholding the covariance
function mountA(slot, ctx) {
  let pid = ctx.param("pa", "hier"); if (!PRESET_IDS.includes(pid)) pid = "hier";
  let theta = Math.min(1, Math.max(0, ctx.param("ta", 0.3)));
  const ctl = ui.h("div", { class: "pt-ctl" });
  const pc = presetControl(ctx, pid, v => { pid = v; ctx.setParams({ pa: v }); draw(); });
  const sl = ui.slider({ label: "threshold $\\theta$", min: 0, max: 1, step: 0.01, value: theta, fmt: x => fx(x),
    onInput: x => { theta = x; ctx.setParams({ ta: x }); draw(); } });
  ctl.append(pc.el, sl.el);
  const pS = ui.h("div"), pA = ui.h("div"), pD = ui.h("div");
  pS.append(ui.panelHead("$\\Sigma$, the input covariance"));
  pA.append(ui.panelHead("$A = \\Sigma - \\theta\\,\\mathbf 1\\mathbf 1^\\top$, stable parts outlined"));
  pD.append(ui.panelHead("dendrogram of $\\Sigma$, cut at $\\theta$"));
  const svgS = svg("svg", { viewBox: "0 0 300 300", role: "img", "aria-label": "heatmap of Sigma" }, pS);
  const svgA = svg("svg", { viewBox: "0 0 300 300", role: "img", "aria-label": "heatmap of A" }, pA);
  const svgD = svg("svg", { viewBox: "0 0 300 300", role: "img", "aria-label": "dendrogram" }, pD);
  const read = ui.h("div", { class: "pt-read" });
  const foot = ui.h("div", { class: "pt-foot" });
  ui.colorbar({ parent: foot, map: "div", range: [-1, 1], label: "$\\Sigma_{bc}$ and $A_{bc}$", width: 220, ticks: [-1, -0.5, 0, 0.5, 1] });
  const leg = ui.legend({ parent: foot, items: [{ label: "θ", color: P.accent, shape: "dash" }] });
  slot.append(ctl, ui.h("div", { class: "pt-panels" }, pS, pA, pD), read, foot);

  const L0 = 30, T0 = 30, SZ = 264;           // matrix area: left/top label margin, side
  function matrix(el, pr, M_, parts) {
    el.replaceChildren();
    const B = pr.B, cs = SZ / B;
    const g = svg("g", {}, el);
    for (let r = 0; r < B; r++) for (let c = 0; c < B; c++) {
      const i = pr.leafOrder[r], j = pr.leafOrder[c], v = M_[i * B + j];
      svg("rect", { x: L0 + c * cs, y: T0 + r * cs, width: cs + 0.4, height: cs + 0.4, fill: colorOf("div", v, -1, 1), "data-i": i, "data-j": j }, g);
    }
    // thin white grid
    for (let k = 1; k < B; k++) {
      svg("line", { x1: L0 + k * cs, x2: L0 + k * cs, y1: T0, y2: T0 + SZ, stroke: P.surface || "#fff", "stroke-width": 0.8, "pointer-events": "none" }, el);
      svg("line", { y1: T0 + k * cs, y2: T0 + k * cs, x1: L0, x2: L0 + SZ, stroke: P.surface || "#fff", "stroke-width": 0.8, "pointer-events": "none" }, el);
    }
    for (let r = 0; r < B; r++) {
      const lab = inputLabel(pr.leafOrder[r]);
      stext(el, L0 - 5, T0 + (r + 0.5) * cs + 4, lab, { "text-anchor": "end" });
      stext(el, L0 + (r + 0.5) * cs, T0 - 7, lab, { "text-anchor": "middle" });
    }
    for (const p of parts || []) {
      const runs = rowRuns(p.S, pr.pos);
      for (const [a0, a1] of runs) for (const [b0, b1] of runs)
        svg("rect", { x: L0 + b0 * cs + 1.5, y: T0 + a0 * cs + 1.5, width: (b1 - b0 + 1) * cs - 3, height: (a1 - a0 + 1) * cs - 3,
          fill: "none", stroke: p.color, "stroke-width": 3, rx: 2, "pointer-events": "none" }, el);
    }
  }
  function tipCell(e, pr, A) {
    const t = e.target; if (!t.dataset || t.dataset.i == null) { ui.hideTip(); return; }
    const i = +t.dataset.i, j = +t.dataset.j, B = pr.B, s = pr.Sigma[i * B + j], a = A[i * B + j];
    ui.showTip(e.clientX, e.clientY, ui.tipRows(`${inputLabel(i)}, ${inputLabel(j)}`,
      [{ name: "Σ", value: ui.fmt(s) }, { name: `A = Σ − θ (θ = ${fx(theta)})`, value: ui.fmt(a) }],
      i !== j ? (a < 0 ? "negative: below threshold" : "positive: above threshold") : null));
  }
  let curA = null, curPr = null;
  for (const el of [svgS, svgA]) {
    el.addEventListener("mousemove", e => curPr && tipCell(e, curPr, curA));
    el.addEventListener("mouseleave", () => ui.hideTip());
  }

  function dendro(el, pr, parts) {
    el.replaceChildren();
    const B = pr.B, cs = SZ / B, lk = pr.linkage;
    const yOf = v => T0 + 8 + (1 - v) * (SZ - 18);       // correlation 1 at the top, 0 at the bottom
    const xs = new Array(B + lk.length), ys = new Array(B + lk.length);
    for (let b = 0; b < B; b++) { xs[b] = L0 + (pr.pos[b] + 0.5) * cs; ys[b] = pr.Sigma[b * B + b] / pr.Sigma[b * B + b]; }
    lk.forEach(([a, b, hgt], k) => { xs[B + k] = (xs[a] + xs[b]) / 2; ys[B + k] = hgt; });
    // axis
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      svg("line", { x1: L0 - 3, x2: L0 + SZ, y1: yOf(v), y2: yOf(v), stroke: P.line, "stroke-width": v === 0 ? 1 : 0.5 }, el);
      stext(el, L0 - 5, yOf(v) + 4, v === 0 ? "0" : String(v), { "text-anchor": "end", "font-size": 10, fill: P.mute });
    }
    stext(el, L0 + SZ, yOf(0) + 14, "merge height (mean correlation)", { "text-anchor": "end", "font-size": 10, fill: P.mute });
    const partOf = new Map(parts.map(p => [p.key, p]));
    const mem = id => (id < B ? [id] : [...mem(lk[id - B][0]), ...mem(lk[id - B][1])]);
    const seg = (x1, y1, x2, y2, col, w) => svg("line", { x1, y1, x2, y2, stroke: col, "stroke-width": w, "stroke-linecap": "round" }, el);
    const faint = P.mute2 || P.line;
    // vertical piece from parent level yp (lower correlation) up to node level yn, split at θ
    function vert(x, yp, yn, col) {
      if (col && yn > theta) {
        const lo = Math.max(yp, theta);
        if (yp < theta) seg(x, yOf(yp), x, yOf(theta), faint, 1.3);
        seg(x, yOf(lo), x, yOf(yn), col, 2.4);
      } else seg(x, yOf(yp), x, yOf(yn), faint, 1.3);
    }
    function walk(id, yp, col) {
      const yn = ys[id];
      if (!col && yn > theta) {
        const S = mem(id).sort((a, b) => a - b), p = partOf.get(M.keyOf(S));
        col = p ? p.color : P.ink2;
      }
      vert(xs[id], yp, yn, col);
      if (id >= B) {
        const [a, b] = lk[id - B];
        seg(xs[a], yOf(yn), xs[b], yOf(yn), col && yn > theta ? col : faint, col && yn > theta ? 2.4 : 1.3);
        walk(a, yn, col); walk(b, yn, col);
      }
    }
    walk(B + lk.length - 1, 0, null);
    for (let b = 0; b < B; b++) stext(el, xs[b], T0 - 7, inputLabel(b), { "text-anchor": "middle" });
    // θ line
    svg("line", { x1: L0 - 3, x2: L0 + SZ, y1: yOf(theta), y2: yOf(theta), stroke: P.accent, "stroke-width": 1.6, "stroke-dasharray": "5 3" }, el);
    stext(el, L0 + SZ, yOf(theta) - 5, `θ = ${fx(theta)}`, { "text-anchor": "end", fill: P.accent, "font-weight": 600 });
  }

  function draw() {
    const pr = preset(pid), B = pr.B;
    const A = pr.Sigma.map(v => v - theta);
    const parts = coloured(M.stableParts(pr.Sigma, B, theta), pr.pos);
    curA = A; curPr = pr;
    matrix(svgS, pr, pr.Sigma, null);
    matrix(svgA, pr, A, parts);
    dendro(svgD, pr, parts);
    const cut = M.cutClusters(pr.linkage, B, theta).filter(S => S.every(b => pr.Sigma[b * B + b] > theta));
    const cutKeys = new Set(cut.map(M.keyOf)), partKeys = new Set(parts.map(p => p.key));
    const same = cut.length === parts.length && parts.every(p => cutKeys.has(p.key));
    const list = parts.length ? parts.map(p => `<span class="swatch" style="--sw:${p.color}"></span>${supportName(p.S)} <span style="color:var(--mute)">(c = ${ui.fmt(p.c)})</span>`).join(", ") : "none";
    read.innerHTML = `<b>θ = ${fx(theta)}</b>: ${parts.length} stable part${parts.length === 1 ? "" : "s"}: ${list}. Dendrogram cut at θ: ${cut.length} cluster${cut.length === 1 ? "" : "s"}, ${same ? "the same supports" :
      `<b>not</b> the same supports (clusters ${cut.filter(S => !partKeys.has(M.keyOf(S))).map(supportName).join(", ") || "none"} are not stable parts)`}.`;
  }
  draw();
  ctx.onParams?.(p => {
    const np = p.get("pa"), nt = p.get("ta");
    if (np && PRESET_IDS.includes(np)) { pid = np; pc.set(np); }
    if (nt != null && Number.isFinite(+nt)) { theta = +nt; sl.set(theta); }
    draw();
  });
  return { setTheta(t) { theta = t; sl.set(t); draw(); }, setPreset(id) { pid = id; pc.set(id); draw(); } };
}

// ======================================================================= F2: three inputs
const TRI = [[230, 40], [58, 340], [402, 340]];
const at = h => [h[0] * TRI[0][0] + h[1] * TRI[1][0] + h[2] * TRI[2][0], h[0] * TRI[0][1] + h[1] * TRI[1][1] + h[2] * TRI[2][1]];
function starPath(cx, cy, r) {
  let d = "";
  for (let k = 0; k < 10; k++) { const a = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r * 0.45 : r; d += `${k ? "L" : "M"}${(cx + rr * Math.cos(a)).toFixed(1)},${(cy + rr * Math.sin(a)).toFixed(1)}`; }
  return d + "Z";
}
const radiusOf = c => 4 + 10 * Math.sqrt(Math.max(c, 0) / 2);     // fixed scale: c = 2 gives 14 px

function mountParts3(slot, ctx) {
  let theta = Math.min(1.2, Math.max(0, ctx.param("t3", 0.5))), rho = Math.min(0.95, Math.max(-0.9, ctx.param("r3", 0.6)));
  const ctl = ui.h("div", { class: "pt-ctl" });
  const sT = ui.slider({ label: "threshold $\\theta$", min: 0, max: 1.2, step: 0.01, value: theta, fmt: x => fx(x),
    marks: [{ value: 0, label: "0" }, { value: 1, label: "1" }], onInput: x => { theta = x; ctx.setParams({ t3: x }); draw(); } });
  const sR = ui.slider({ label: "correlation $\\rho_{12}$ of $g_1, g_2$", min: -0.9, max: 0.95, step: 0.01, value: rho, fmt: x => fx(x),
    marks: [{ value: 0, label: "0" }], onInput: x => { rho = x; ctx.setParams({ r3: x }); draw(); } });
  ctl.append(sT.el, sR.el);
  const left = ui.h("div"), right = ui.h("div");
  const head = ui.h("div", { class: "chart-title", style: { marginBottom: "0" } });
  left.append(head);
  const s = svg("svg", { viewBox: "0 0 460 400", role: "img", "aria-label": "triangle of three inputs with the parts" }, left);
  const note = ui.h("div", { class: "pt-read" });
  const tab = ui.h("div");
  right.append(ui.panelHead("candidates at this $\\theta$ (one per support)"), tab, note);
  const foot = ui.h("div", { class: "pt-foot" });
  const key3 = ui.h("div", { class: "pt-key3" });
  const glyph = draw => { const g = svg("svg", { width: 18, height: 18, viewBox: "0 0 18 18" }); draw(g); return g; };
  key3.append(
    ui.h("span", {}, glyph(g => svg("circle", { cx: 9, cy: 9, r: 6, fill: catColor(3) }, g)), "stable part (larger: bigger c)"),
    ui.h("span", {}, glyph(g => svg("circle", { cx: 9, cy: 9, r: 6, fill: "none", stroke: P.ink2, "stroke-width": 1.4, "stroke-dasharray": "3 2" }, g)), "part, not stable"),
    ui.h("span", {}, glyph(g => svg("path", { d: "M5,5L13,13M5,13L13,5", stroke: P.mute2 || P.mute, "stroke-width": 1.6 }, g)), "candidate that fails (c ≤ 0 or off-support)"),
    ui.h("span", {}, glyph(g => svg("path", { d: starPath(9, 9.5, 8), fill: "none", stroke: P.fg, "stroke-width": 1.4 }, g)), "top eigenvector of Σ"));
  foot.append(key3);
  slot.append(ctl, ui.h("div", { class: "pt-tri" }, left, right), foot);

  // static frame
  const gFace = svg("g", {}, s);
  svg("path", { d: `M${TRI[0]}L${TRI[1]}L${TRI[2]}Z`, fill: "none", stroke: P.line, "stroke-width": 1.5 }, s);
  const gMk = svg("g", {}, s);
  // inputs: squares just outside the corner they define, with a short leader in their colour
  const cen = [(TRI[0][0] + TRI[1][0] + TRI[2][0]) / 3, (TRI[0][1] + TRI[1][1] + TRI[2][1]) / 3];
  TRI.forEach(([x, y], b) => {
    const dx = x - cen[0], dy = y - cen[1], n = Math.hypot(dx, dy), qx = x + 26 * dx / n, qy = y + 26 * dy / n;
    svg("line", { x1: x, y1: y, x2: qx, y2: qy, stroke: catColor(b), "stroke-width": 1.5 }, s);
    svg("rect", { x: qx - 8, y: qy - 8, width: 16, height: 16, rx: 2, fill: catColor(b), stroke: P.fg, "stroke-width": 1.2 }, s);
    stext(s, b === 0 ? qx + 14 : qx, b === 0 ? qy + 5 : qy + 27, inputLabel(b), { "text-anchor": b === 0 ? "start" : "middle", "font-size": 16, fill: P.fg, "font-weight": 600 });
  });

  const statusOf = p => (p.stable ? "stable" : p.part ? "part, not stable" : p.why);
  function draw() {
    const Sig = M.sigma3(rho);
    const all = M.enumerateParts(Sig, 3, theta, { candidates: true });
    const stable = all.filter(p => p.stable).sort((a, b) => b.c - a.c).map((p, k) => ({ ...p, color: catColor(3 + k) }));
    const others = all.filter(p => !p.stable);
    gFace.replaceChildren(); gMk.replaceChildren();
    // supports of stable parts: shaded corner / edge / face
    for (const p of stable) {
      const S = p.S;
      if (S.length === 2) svg("line", { x1: TRI[S[0]][0], y1: TRI[S[0]][1], x2: TRI[S[1]][0], y2: TRI[S[1]][1], stroke: p.color, "stroke-width": 9, "stroke-opacity": 0.3, "stroke-linecap": "round" }, gFace);
      else if (S.length === 1) svg("circle", { cx: TRI[S[0]][0], cy: TRI[S[0]][1], r: 24, fill: p.color, "fill-opacity": 0.2 }, gFace);
      else svg("path", { d: `M${TRI[0]}L${TRI[1]}L${TRI[2]}Z`, fill: p.color, "fill-opacity": 0.08 }, gFace);
    }
    const hover = (el, p) => {
      el.addEventListener("mouseenter", e => ui.showTip(e.clientX, e.clientY, tipPart(p)));
      el.addEventListener("mousemove", e => ui.showTip(e.clientX, e.clientY, tipPart(p)));
      el.addEventListener("mouseleave", () => ui.hideTip());
    };
    // failing candidates and unstable parts (faint), then the top eigenvector, then stable parts on top
    for (const p of others) {
      const [x, y] = at(p.h);
      const g = svg("g", { style: "cursor:default" }, gMk);
      if (p.part) svg("circle", { cx: x, cy: y, r: radiusOf(p.c), fill: "none", stroke: P.ink2, "stroke-width": 1.4, "stroke-dasharray": "3 2" }, g);
      else {
        const r = 4.5, c = P.mute2 || P.mute;
        svg("path", { d: `M${x - r},${y - r}L${x + r},${y + r}M${x - r},${y + r}L${x + r},${y - r}`, stroke: c, "stroke-width": 1.6 }, g);
      }
      svg("circle", { cx: x, cy: y, r: 9, fill: "transparent" }, g);
      hover(g, p);
    }
    for (const p of stable) {
      const [x, y] = at(p.h);
      const c = svg("circle", { cx: x, cy: y, r: radiusOf(p.c), fill: p.color, stroke: P.surface || "#fff", "stroke-width": 2 }, gMk);
      hover(c, p);
    }
    // top eigenvector of Σ
    const { values, vectors } = eigh(Sig, 3);
    const v = [0, 1, 2].map(i => vectors[i * 3]), sg = v[0] + v[1] + v[2] >= 0 ? 1 : -1, w = v.map(x => sg * x);
    let eigNote;
    if (values[0] - values[1] < 1e-9) eigNote = `The top eigenvalue of $\\Sigma$ is repeated (${ui.fmt(values[0])}), so its top eigenvector is not unique; no star.`;
    else if (Math.min(...w) < -1e-9) eigNote = `The top eigenvector of $\\Sigma$, $(${w.map(x => fx(x)).join(", ")})$, has mixed signs, so it is not a point of the triangle; no star.`;
    else {
      const sum = w[0] + w[1] + w[2], hh = w.map(x => Math.max(0, x) / sum), [x, y] = at(hh);
      svg("path", { d: starPath(x, y, 14), fill: "none", stroke: P.fg, "stroke-width": 1.8, "pointer-events": "none" }, gMk);
      eigNote = `Star: the top eigenvector of $\\Sigma$ (eigenvalue ${ui.fmt(values[0])}), normalised to sum 1: $(${hh.map(x => fx(x)).join(", ")})$.`;
    }
    head.textContent = `θ = ${fx(theta)}, ρ₁₂ = ${fx(rho)}: ${stable.length} stable part${stable.length === 1 ? "" : "s"}`;
    // table: stable first, then other parts, then failing candidates
    const rows = [...stable, ...others.filter(p => p.part), ...others.filter(p => !p.part)];
    const t = ui.h("table", { class: "pt-tab" },
      ui.h("thead", {}, ui.h("tr", {}, ...["support", "h", "c", "margin", "status"].map((x, i) => ui.h("th", { text: x, style: i && i < 4 ? { textAlign: "right" } : null })))),
      ui.h("tbody", {}, ...rows.map(p => ui.h("tr", { class: p.stable ? "stable" : p.part ? "" : "faint" },
        ui.h("td", {}, p.stable ? ui.h("span", { class: "swatch round", style: { "--sw": p.color } }) : null, supportName(p.S)),
        ui.h("td", { class: "num", text: `(${Array.from(p.h).map(x => fx(x)).join(", ")})` }),
        ui.h("td", { class: "num", text: ui.fmt(p.c) }),
        ui.h("td", { class: "num", text: Number.isFinite(p.margin) ? ui.fmt(p.margin) : (p.S.length === 3 ? "none off S" : "n/a") }),
        ui.h("td", { text: statusOf(p) })))));
    tab.replaceChildren(ui.h("div", { class: "table-scroll" }, t));
    note.innerHTML = eigNote;
    ctx.renderMath(note);
  }
  function tipPart(p) {
    return ui.tipRows(`support ${supportName(p.S)}`, [
      { name: "h", value: `(${Array.from(p.h).map(x => fx(x)).join(", ")})` },
      { name: "c (eigenvalue of A_SS)", value: ui.fmt(p.c) },
      { name: "margin θ − max_{b∉S} (Σh)_b", value: Number.isFinite(p.margin) ? ui.fmt(p.margin) : "n/a" },
      { name: "status", value: statusOf(p) }]);
  }
  draw();
  ctx.onParams?.(p => {
    const a = p.get("t3"), b = p.get("r3");
    if (a != null && Number.isFinite(+a)) { theta = +a; sT.set(theta); }
    if (b != null && Number.isFinite(+b)) { rho = +b; sR.set(rho); }
    draw();
  });
  return { setTheta(t) { theta = t; sT.set(t); draw(); }, setRho(r) { rho = r; sR.set(r); draw(); } };
}

// ======================================================================= F3: the part diagram (live)
const GRID = Array.from({ length: 201 }, (_, i) => i / 200);
function mountHier(slot, ctx) {
  let pid = ctx.param("ph", "hier"); if (!PRESET_IDS.includes(pid)) pid = "hier";
  let theta = Math.min(1, Math.max(0, ctx.param("th", 0.3)));
  const ctl = ui.h("div", { class: "pt-ctl" });
  const pc = presetControl(ctx, pid, v => { pid = v; ctx.setParams({ ph: v }); load(); });
  const sl = ui.slider({ label: "threshold $\\theta$ (or drag on the diagram)", min: 0, max: 1, step: 0.005, value: theta, fmt: x => fx(x, 3),
    onInput: x => setTheta(x) });
  ctl.append(pc.el, sl.el);
  const box = ui.h("div");
  const read = ui.h("div", { class: "pt-read" });
  slot.append(ctl, box, read);
  let pd = null;
  function setTheta(x, fromDiagram) {
    theta = x; if (fromDiagram) sl.set(x);
    ctx.setParams({ th: x });
    pd.setCursor(x); readout();
  }
  function load() {
    const pr = preset(pid);
    if (!pr.grid) pr.grid = M.partGrid(pr.Sigma, pr.B, GRID);
    const opts = { thetas: GRID, parts: pr.grid.map(g => g.parts), leafOrder: pr.leafOrder, linkage: pr.linkage,
      selfHeights: Array.from({ length: pr.B }, (_, b) => pr.Sigma[b * pr.B + b]), cursor: theta, onCursor: x => setTheta(x, true) };
    if (!pd) pd = drawPartDiagram(box, opts); else pd.update({ ...opts, overlay: null });
    readout();
  }
  function readout() {
    const pr = preset(pid), parts = M.stableParts(pr.Sigma, pr.B, theta);
    read.innerHTML = `<b>θ = ${fx(theta, 3)}</b>: ${parts.length} stable part${parts.length === 1 ? "" : "s"}${parts.length ? ": " + parts.map(p => `${supportName(p.S)} (c = ${ui.fmt(p.c)})`).join(", ") : ""}.`;
  }
  load();
  ctx.onParams?.(p => {
    const np = p.get("ph"), nt = p.get("th");
    if (np && PRESET_IDS.includes(np) && np !== pid) { pid = np; pc.set(np); load(); }
    if (nt != null && Number.isFinite(+nt)) { theta = +nt; sl.set(theta); pd.setCursor(theta); readout(); }
  });
  return { setTheta: x => { sl.set(x); setTheta(x); }, diagram: () => pd };
}

// ======================================================================= drawPartDiagram
// drawPartDiagram(el, opts) -> {el, update(partialOpts), setCursor(theta), destroy()}
// opts:
//   thetas    number[] (ascending; e.g. 200 values in [0, 1])
//   parts     per θ, the stable parts: Array<Array<{S | support: int[], h?: number[] (length B), c?: number, margin?: number}>>
//   leafOrder int[] (row r shows input leafOrder[r]; default 0..B−1)
//   labels    string[] by input index (default g₁, g₂, …)
//   B         number of inputs (default: labels/leafOrder length, else 1 + the largest index seen)
//   linkage   optional dendrogram, scipy layout [[a, b, height, size]], ids ≥ B are earlier merges, height a
//             correlation (similarity: the merge's average correlation). Drawn at the left, rotated: x is the merge
//             height on the same 0 to 1 scale as θ, leaves at the right.
//   selfHeights optional leaf heights for the dendrogram (Σ_bb / correlation 1; default 1)
//   overlay   optional populated parts, either [{theta, parts: [{support, n}]}] or an array aligned with `thetas`
//             (per θ a list of {support, n, stable?}). Drawn as an ink bar over the support rows with a dot whose
//             size rises with n; filled when `stable` (default: the support is a stable part at the nearest θ of the
//             grid), hollow when not.
//   cursor    θ of the cursor (accent line); onCursor(θ) called on click / drag (θ snapped to the grid)
//   vectors   true (default): panel of the Perron vectors h_S of the parts at the cursor (needs h; falls back
//             to the uniform vector on S)
//   crange    [0, cmax] colour range of c (default 0 to the largest c in `parts`)
//   overlayLabel  legend text for the overlay (default "populated part (simulation)")
export function drawPartDiagram(el, opts0) {
  injectStyle();
  let o = { vectors: true, ...opts0 };
  const wrap = ui.h("div", { class: "pt-pd" });
  const cv = ui.h("canvas");
  const key = ui.h("div", { class: "pt-pd-key" });
  wrap.append(cv, key);
  el.append(wrap);
  let D = null, Lay = null, width = 0, dragging = false;
  const cSeq = cmap("seq");
  const fillOf = (c, a = 1) => rgbToCss(cSeq(0.14 + 0.5 * Math.max(0, Math.min(1, c / D.cmax))), a);
  const lineOf = c => rgbToCss(cSeq(0.55 + 0.4 * Math.max(0, Math.min(1, c / D.cmax))));

  function prep() {
    const th = o.thetas;
    const norm = p => {
      const S = [...(p.S || p.support)].sort((a, b) => a - b);
      return { S, key: M.keyOf(S), h: p.h ? Array.from(p.h) : null, c: p.c, margin: p.margin };
    };
    const parts = o.parts.map(list => (list || []).map(norm));
    let B = o.B || (o.labels && o.labels.length) || (o.leafOrder && o.leafOrder.length) || 0;
    if (!B) for (const l of parts) for (const p of l) for (const b of p.S) B = Math.max(B, b + 1);
    const leafOrder = o.leafOrder || Array.from({ length: B }, (_, i) => i);
    const pos = new Array(B); leafOrder.forEach((b, r) => (pos[b] = r));
    const labels = o.labels || Array.from({ length: B }, (_, b) => inputLabel(b));
    let cmax = o.crange ? o.crange[1] : 0;
    if (!o.crange) for (const l of parts) for (const p of l) if (Number.isFinite(p.c)) cmax = Math.max(cmax, p.c);
    cmax = cmax || 1;
    // runs of the same support over consecutive grid indices
    const runs = [], open = new Map();
    parts.forEach((list, i) => {
      const seen = new Set();
      for (const p of list) {
        seen.add(p.key);
        const r = open.get(p.key);
        if (r && r.i1 === i - 1) r.i1 = i; else { const nr = { key: p.key, S: p.S, i0: i, i1: i, rr: rowRuns(p.S, pos) }; runs.push(nr); open.set(p.key, nr); }
      }
    });
    // overlay
    let overlay = null, nmax = 1;
    if (o.overlay) {
      const src = o.overlay.length && Array.isArray(o.overlay[0]) || (o.overlay.length === th.length && o.overlay.every(x => Array.isArray(x)))
        ? o.overlay.map((list, i) => ({ theta: th[i], parts: list })) : o.overlay;
      overlay = src.map(e => {
        const gi = nearest(th, e.theta), keys = new Set(parts[gi].map(p => p.key));
        return { theta: e.theta, items: (e.parts || []).map(q => { const S = [...q.support || q.S].sort((a, b) => a - b); nmax = Math.max(nmax, q.n || 1);
          return { S, n: q.n || 1, rr: rowRuns(S, pos), stable: q.stable ?? keys.has(M.keyOf(S)), dx: 0 }; }) };
      });
      // items of one θ whose row spans overlap go side by side (lanes 7 px apart)
      for (const e of overlay) {
        const lanes = [];
        const span = q => [q.rr[0][0], q.rr[q.rr.length - 1][1]];
        const its = [...e.items].sort((a, b) => (span(b)[1] - span(b)[0]) - (span(a)[1] - span(a)[0]));
        for (const q of its) {
          const [a, b] = span(q);
          let l = 0;
          while (lanes[l] && lanes[l].some(([c, d]) => !(b < c || a > d))) l++;
          (lanes[l] = lanes[l] || []).push([a, b]); q.lane = l;
        }
        const nl = lanes.length;
        for (const q of e.items) q.dx = (q.lane - (nl - 1) / 2) * 7;
      }
    }
    D = { th, parts, B, leafOrder, pos, labels, cmax, runs, overlay, nmax };
    renderKey();
  }
  function renderKey() {
    const stops = Array.from({ length: 9 }, (_, i) => `${fillOf(D.cmax * i / 8)} ${(i / 8 * 100).toFixed(1)}%`).join(", ");
    const items = [ui.h("span", {}, "stable part, coloured by c:", ui.h("span", { class: "pt-grad", style: { background: `linear-gradient(to right, ${stops})` } }), `0 … ${ui.fmt(D.cmax)}`)];
    if (D.overlay) {
      items.push(ui.h("span", {}, ui.h("i", { style: { width: "10px", height: "10px", borderRadius: "50%", background: P.fg } }), o.overlayLabel || "populated part (simulation), area ∝ n"));
      items.push(ui.h("span", {}, ui.h("i", { style: { width: "8px", height: "8px", borderRadius: "50%", border: `1.6px solid ${P.fg}`, background: P.surface || "#fff" } }), "populated, not a stable part there"));
    }
    items.push(ui.h("span", {}, ui.h("i", { style: { width: "14px", height: "0", borderTop: `2px solid ${P.accent}` } }), "cursor θ"));
    key.replaceChildren(...items);
  }
  const nearest = (arr, x) => { let bi = 0, bd = Infinity; arr.forEach((v, i) => { const d = Math.abs(v - x); if (d < bd) { bd = d; bi = i; } }); return bi; };

  function layout(W) {
    const B = D.B, narrow = W < 640;
    const rowH = B <= 8 ? (narrow ? 26 : 30) : B <= 16 ? (narrow ? 18 : 22) : Math.max(8, 360 / B);
    const top = 26, plotH = rowH * B, axisH = 38;
    const dendW = o.linkage ? (narrow ? Math.min(78, W * 0.2) : Math.min(150, W * 0.15)) : 0;
    const labW = B > 9 ? 30 : 24;
    const cur = D.parts[nearest(D.th, o.cursor ?? D.th[0])] || [];
    const nv = Math.max(1, cur.length);
    let vecW = 0, vecBelow = false;
    if (o.vectors) {
      if (!narrow) vecW = Math.min(320, Math.max(200, nv * 40));
      else vecBelow = true;
    }
    const x0 = dendW + labW + (dendW ? 8 : 0), x1 = W - (vecW ? vecW + 18 : 6);
    const vecTop = top + plotH + axisH + 30;
    const H = vecBelow ? vecTop + plotH + 34 : top + plotH + axisH + (o.vectors ? 6 : 0);
    return { W, H, B, rowH, top, plotH, axisH, dendW, labW, x0, x1, vecW, vecBelow, vecTop, narrow };
  }
  const yRow = r => Lay.top + r * Lay.rowH;
  const xTh = t => { const a = D.th[0], b = D.th[D.th.length - 1]; return Lay.x0 + (t - a) / ((b - a) || 1) * (Lay.x1 - Lay.x0); };
  const thX = x => { const a = D.th[0], b = D.th[D.th.length - 1]; return a + (x - Lay.x0) / ((Lay.x1 - Lay.x0) || 1) * (b - a); };

  function render() {
    width = Math.floor(wrap.clientWidth); if (width <= 0 || !D) return;
    Lay = layout(width);
    const g = hidpi(cv, width, Lay.H), { B, rowH, top, plotH, x0, x1 } = Lay;
    g.clearRect(0, 0, width, Lay.H);
    const th = D.th, n = th.length;
    const dx = (x1 - x0) / Math.max(1, n - 1);
    // background rows
    g.fillStyle = P.wash; g.fillRect(x0, top, x1 - x0, plotH);
    g.strokeStyle = P.surface || "#fff"; g.lineWidth = 1;
    for (let r = 1; r < B; r++) { g.beginPath(); g.moveTo(x0, yRow(r) + 0.5); g.lineTo(x1, yRow(r) + 0.5); g.stroke(); }
    // fills: per grid column, per part, per row run (translucent so that overlapping parts stack)
    const inset = Math.min(3, rowH * 0.12);
    const anyOverlap = D.parts.some(l => { const cnt = new Array(B).fill(0); for (const p of l) for (const b of p.S) if (++cnt[b] > 1) return true; return false; });
    // column edges at the midpoints between grid values, rounded so neighbouring columns abut without overlap
    const edge = i => Math.round(i <= 0 ? x0 : i >= n ? x1 : xTh((th[i - 1] + th[i]) / 2));
    D.parts.forEach((list, i) => {
      const xa = edge(i), xb = edge(i + 1);
      for (const p of list) {
        g.fillStyle = fillOf(p.c ?? D.cmax * 0.5, anyOverlap ? 0.55 : 1);
        for (const [r0, r1] of rowRuns(p.S, D.pos)) g.fillRect(xa, yRow(r0) + inset, xb - xa, (r1 - r0 + 1) * rowH - 2 * inset);
      }
    });
    // outlines of each run (a part that persists over consecutive θ)
    g.lineWidth = 1;
    for (const r of D.runs) {
      const xa = edge(r.i0), xb = edge(r.i1 + 1);
      const mid = D.parts[Math.round((r.i0 + r.i1) / 2)].find(p => p.key === r.key);
      g.strokeStyle = lineOf(mid?.c ?? D.cmax * 0.5);
      for (const [r0, r1] of r.rr) g.strokeRect(xa + 0.5, yRow(r0) + inset + 0.5, xb - xa - 1, (r1 - r0 + 1) * rowH - 2 * inset - 1);
      if (r.rr.length > 1) {   // a support that is not contiguous in leaf order: join its row runs
        g.setLineDash([2, 2]); g.beginPath();
        const xm = xa + 3; g.moveTo(xm, yRow(r.rr[0][1] + 1) - inset); g.lineTo(xm, yRow(r.rr[r.rr.length - 1][0]) + inset); g.stroke(); g.setLineDash([]);
      }
    }
    const ya = top + plotH;
    // cursor (under the overlay)
    if (o.cursor != null) {
      const x = xTh(o.cursor);
      g.strokeStyle = P.accent; g.lineWidth = 2; g.beginPath(); g.moveTo(x, top - 4); g.lineTo(x, ya + 2); g.stroke();
    }
    // overlay
    if (D.overlay) {
      // dense overlays on narrow screens: shrink dots and halos to the spacing between overlay columns
      let sp = Infinity;
      const ox = D.overlay.map(e => xTh(e.theta)).sort((p, q) => p - q);
      for (let i = 1; i < ox.length; i++) if (ox[i] - ox[i - 1] > 0.5) sp = Math.min(sp, ox[i] - ox[i - 1]);
      const shrink = Math.min(1, Math.max(0.3, (sp / 2 - 0.5) / 7.5)), halo = Math.min(5, Math.max(2.5, sp - 1));
      for (const e of D.overlay) {
        const xc = xTh(e.theta);
        if (xc < x0 - 1 || xc > x1 + 1) continue;
        for (const q of e.items) {
          const x = xc + q.dx * shrink;
          const ya = yRow(q.rr[0][0]) + inset + 3, yb = yRow(q.rr[q.rr.length - 1][1] + 1) - inset - 3;
          g.lineCap = "round";
          g.strokeStyle = "rgba(255,255,255,0.9)"; g.lineWidth = halo; g.beginPath(); g.moveTo(x, ya); g.lineTo(x, yb); g.stroke();
          g.strokeStyle = P.fg; g.lineWidth = Math.min(2, halo - 1);
          for (const [r0, r1] of q.rr) { g.beginPath(); g.moveTo(x, yRow(r0) + inset + 3); g.lineTo(x, yRow(r1 + 1) - inset - 3); g.stroke(); }
          const rad = shrink * (2.5 + 5 * Math.sqrt(q.n / D.nmax)), ym = (ya + yb) / 2;
          g.beginPath(); g.arc(x, ym, rad + Math.min(1.5, shrink * 2), 0, 2 * Math.PI); g.fillStyle = "rgba(255,255,255,0.95)"; g.fill();
          g.beginPath(); g.arc(x, ym, rad, 0, 2 * Math.PI);
          if (q.stable) { g.fillStyle = P.fg; g.fill(); } else { g.fillStyle = P.surface || "#fff"; g.fill(); g.lineWidth = Math.min(1.6, rad * 0.6); g.strokeStyle = P.fg; g.stroke(); }
        }
      }
    }
    // row labels
    g.font = `${B > 12 ? 10.5 : 11.5}px ${FONT}`; g.fillStyle = P.ink2; g.textAlign = "right"; g.textBaseline = "middle";
    for (let r = 0; r < B; r++) if (rowH >= 9 || r % 2 === 0) g.fillText(D.labels[D.leafOrder[r]], x0 - 5, yRow(r) + rowH / 2);
    // θ axis
    g.strokeStyle = P.line; g.beginPath(); g.moveTo(x0, ya + 0.5); g.lineTo(x1, ya + 0.5); g.stroke();
    g.fillStyle = P.mute; g.font = `10.5px ${FONT}`; g.textAlign = "center"; g.textBaseline = "top";
    for (const t of niceTicks(th[0], th[n - 1], Lay.narrow ? 4 : 6)) {
      const x = xTh(t); g.beginPath(); g.moveTo(x, ya); g.lineTo(x, ya + 4); g.stroke(); g.fillText(ui.fmt(t), x, ya + 6);
    }
    g.fillStyle = P.ink2; g.font = `560 11.5px ${FONT}`; g.fillText("pruning threshold θ", (x0 + x1) / 2, ya + 21);
    g.textAlign = "left"; g.textBaseline = "alphabetic"; g.font = `600 12px ${FONT}`; g.fillStyle = P.fg;
    g.fillText("stable parts against θ (rows: inputs)", x0, top - 9);
    if (o.linkage) drawDendro(g);
    if (o.vectors) drawVectors(g);
  }

  function drawDendro(g) {
    const { B, top, plotH, rowH, dendW } = Lay, lk = o.linkage;
    const a = 6, b = dendW - 4;                                     // x range of the dendrogram: height 0 → a, 1 → b
    const xH = v => a + Math.max(0, Math.min(1, v)) * (b - a);
    const ys = new Array(B + lk.length), hs = new Array(B + lk.length);
    for (let i = 0; i < B; i++) { ys[i] = top + (D.pos[i] + 0.5) * rowH; hs[i] = o.selfHeights ? o.selfHeights[i] : 1; }
    lk.forEach(([p, q, hgt], k) => { ys[B + k] = (ys[p] + ys[q]) / 2; hs[B + k] = hgt; });
    g.strokeStyle = P.ink2; g.lineWidth = 1.4; g.lineCap = "round";
    g.beginPath();
    lk.forEach(([p, q, hgt], k) => {
      const x = xH(hgt);
      g.moveTo(x, ys[p]); g.lineTo(x, ys[q]);                     // the merge (vertical, joining the two children)
      g.moveTo(x, ys[p]); g.lineTo(xH(hs[p]), ys[p]);             // child p back to its own level
      g.moveTo(x, ys[q]); g.lineTo(xH(hs[q]), ys[q]);
    });
    const root = B + lk.length - 1; g.moveTo(xH(0), ys[root]); g.lineTo(xH(hs[root]), ys[root]);
    g.stroke();
    // axis 0 … 1 under it
    const ya = top + plotH;
    g.strokeStyle = P.line; g.lineWidth = 1; g.beginPath(); g.moveTo(a, ya + 0.5); g.lineTo(b, ya + 0.5); g.stroke();
    g.fillStyle = P.mute; g.font = `10.5px ${FONT}`; g.textAlign = "center"; g.textBaseline = "top";
    for (const t of [0, 0.5, 1]) { const x = xH(t); g.beginPath(); g.moveTo(x, ya); g.lineTo(x, ya + 4); g.stroke(); g.fillText(String(t), x, ya + 6); }
    g.fillStyle = P.ink2; g.font = `560 11px ${FONT}`; g.fillText(Lay.narrow ? "merge ρ" : "merge correlation", (a + b) / 2, ya + 21);
    g.textAlign = "left"; g.textBaseline = "alphabetic"; g.font = `600 12px ${FONT}`; g.fillStyle = P.fg;
    g.fillText("dendrogram", a, top - 9);
    if (o.cursor != null) {
      const x = xH(o.cursor); g.strokeStyle = P.accent; g.lineWidth = 1.5; g.setLineDash([4, 3]);
      g.beginPath(); g.moveTo(x, top - 2); g.lineTo(x, ya); g.stroke(); g.setLineDash([]);
    }
  }

  function drawVectors(g) {
    const { B, rowH, plotH, x1, vecW, vecBelow, narrow } = Lay;
    const gi = nearest(D.th, o.cursor ?? D.th[0]);
    const parts = [...(D.parts[gi] || [])].sort((p, q) => Math.min(...p.S.map(b => D.pos[b])) - Math.min(...q.S.map(b => D.pos[b])));
    const top = vecBelow ? Lay.vecTop : Lay.top;
    let xa, xb;
    if (vecBelow) { xa = Lay.labW + 4; xb = Lay.W - 6; } else { xa = x1 + 18; xb = Lay.W - 4; }
    g.textAlign = "left"; g.textBaseline = "alphabetic"; g.font = `600 12px ${FONT}`; g.fillStyle = P.fg;
    g.fillText(`Perron vectors h_S, θ = ${ui.fmtFixed(D.th[gi], 3)}`, xa, top - 9);
    if (vecBelow) {   // own row labels
      g.font = `${B > 12 ? 10.5 : 11.5}px ${FONT}`; g.fillStyle = P.ink2; g.textAlign = "right"; g.textBaseline = "middle";
      for (let r = 0; r < B; r++) if (rowH >= 9 || r % 2 === 0) g.fillText(D.labels[D.leafOrder[r]], xa - 6, top + (r + 0.5) * rowH);
    }
    g.fillStyle = P.wash; g.fillRect(xa, top, xb - xa, plotH);
    if (!parts.length) {
      g.fillStyle = P.mute; g.font = `11.5px ${FONT}`; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText("no stable part", (xa + xb) / 2, top + plotH / 2); return;
    }
    const k = parts.length, gap = 4, cw = Math.min(70, (xb - xa - gap * (k - 1)) / k);
    let hmax = 0;
    const hs = parts.map(p => { const h = p.h || Array.from({ length: B }, (_, b) => (p.S.includes(b) ? 1 / p.S.length : 0)); for (const b of p.S) hmax = Math.max(hmax, h[b]); return h; });
    parts.forEach((p, j) => {
      const cx = xa + j * (cw + gap);
      g.strokeStyle = P.line; g.lineWidth = 1; g.beginPath(); g.moveTo(cx + 0.5, top); g.lineTo(cx + 0.5, top + plotH); g.stroke();
      g.fillStyle = fillOf(p.c ?? D.cmax * 0.5);
      for (const b of p.S) {
        const r = D.pos[b], w = (cw - 4) * hs[j][b] / (hmax || 1);
        g.fillRect(cx + 1, top + r * rowH + 2, w, rowH - 4);
        g.strokeStyle = lineOf(p.c ?? D.cmax * 0.5); g.strokeRect(cx + 1.5, top + r * rowH + 2.5, Math.max(0, w - 1), rowH - 5);
      }
      if (Number.isFinite(p.c) && cw >= 26) {
        g.fillStyle = P.ink2; g.font = `10px ${FONT}`; g.textAlign = "left"; g.textBaseline = "top";
        g.fillText(cw >= 44 ? `c=${ui.fmt(p.c, 2)}` : ui.fmt(p.c, 2), cx + 2, top + plotH + 4);
      }
    });
    g.fillStyle = P.mute; g.font = `10.5px ${FONT}`; g.textAlign = "right"; g.textBaseline = "top";
    g.fillText(`bar length: h_b (max ${ui.fmt(hmax, 2)})`, xb, top + plotH + (narrow ? 18 : 18));
  }

  // ---------------------------------------------------------------- interaction
  const inPlot = (x, y) => Lay && x >= Lay.x0 - 4 && x <= Lay.x1 + 4 && y >= Lay.top - 6 && y <= Lay.top + Lay.plotH + 6;
  const local = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  function pick(x) { const t = thX(Math.max(Lay.x0, Math.min(Lay.x1, x))); return D.th[nearest(D.th, t)]; }
  cv.addEventListener("pointerdown", e => {
    const [x, y] = local(e); if (!inPlot(x, y)) return;
    dragging = true; cv.setPointerCapture(e.pointerId);
    const t = pick(x); o.cursor = t; render(); o.onCursor?.(t);
  });
  cv.addEventListener("pointermove", e => {
    const [x, y] = local(e);
    if (dragging) { const t = pick(x); if (t !== o.cursor) { o.cursor = t; render(); o.onCursor?.(t); } }
    if (inPlot(x, y)) tip(e, x, y); else ui.hideTip();
  });
  const stop = () => { dragging = false; };
  cv.addEventListener("pointerup", stop); cv.addEventListener("pointercancel", stop);
  cv.addEventListener("pointerleave", () => ui.hideTip());
  function tip(e, x, y) {
    const gi = nearest(D.th, thX(x)), r = Math.floor((y - Lay.top) / Lay.rowH);
    if (r < 0 || r >= D.B) { ui.hideTip(); return; }
    const b = D.leafOrder[r], here = D.parts[gi].filter(p => p.S.includes(b));
    const rows = here.map(p => ({ name: supportName(p.S), value: Number.isFinite(p.c) ? `c = ${ui.fmt(p.c)}${Number.isFinite(p.margin) ? `, margin ${ui.fmt(p.margin)}` : ""}` : "", color: fillOf(p.c ?? D.cmax * 0.5), shape: "square" }));
    let note = here.length ? null : "no stable part contains this input";
    if (D.overlay) {
      const oe = D.overlay[nearest(D.overlay.map(e2 => e2.theta), thX(x))];
      if (oe && Math.abs(xTh(oe.theta) - x) < 8) {
        const qs = oe.items.filter(q => q.S.includes(b));
        for (const q of qs) rows.push({ name: `populated ${supportName(q.S)}`, value: `n = ${q.n}${q.stable ? "" : " (not stable)"}`, color: P.fg, shape: "dot" });
      }
    }
    ui.showTip(e.clientX, e.clientY, ui.tipRows(`θ = ${ui.fmtFixed(D.th[gi], 3)}, input ${D.labels[b]}`, rows, note));
  }

  const ro = new ResizeObserver(() => { const w = Math.floor(wrap.clientWidth); if (w > 0 && w !== width) requestAnimationFrame(render); });
  ro.observe(wrap);
  prep(); render();
  return {
    el: wrap,
    update(p) { o = { ...o, ...p }; prep(); render(); },
    setCursor(t) { o.cursor = t; render(); },
    destroy() { ro.disconnect(); wrap.remove(); },
  };
}
