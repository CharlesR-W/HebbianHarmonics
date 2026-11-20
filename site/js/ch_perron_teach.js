// Chapter "perron", figure "θ cuts the correlation tree" ([data-slot="tree"]): A = Σ − θ11ᵀ for 8 inputs with the
// stable parts outlined, beside the average-linkage dendrogram of the correlations cut at θ, branches above the cut
// coloured like the stable part with the same inputs. The maths (presets, exact part enumeration ported from
// hebb/big/common.py, dendrograms) is in ch_perron_teach_model.js.
import * as ui from "./ui.js";
import { P, colorOf, catColor } from "./palette.js";
import * as M from "./ch_perron_teach_model.js";

const NS = "http://www.w3.org/2000/svg";
const NOPART_KEY = "ink2";          // a branch above the cut that is not a stable part: drawn in ink, dashed
let NOPART = null;
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

const PRESETS = M.PRESETS.filter(p => p.id !== "blocks");
const PRESET_LABEL = { hier: "exact hierarchy", hiern: "comparison: + noise", ring: "comparison: ring" };
const FRAMES = [0.05, 0.3, 0.45, 0.6, 0.85];      // the filmstrip: between and at the merge levels 0.15 / 0.45 / 0.75
const PRESET_IDS = PRESETS.map(p => p.id);
const presetCache = {};
function preset(id) {
  if (!presetCache[id]) {
    const { B, Sigma } = M.presetSigma(id);
    const { linkage, leafOrder } = M.averageLinkage(Sigma, B);
    const pos = new Array(B); leafOrder.forEach((b, r) => (pos[b] = r));
    presetCache[id] = { id, B, Sigma, linkage, leafOrder, pos };
  }
  return presetCache[id];
}
// stable parts sorted by their first row in leaf order, with a colour each
function coloured(parts, pos) {
  return parts.map(p => ({ ...p, first: Math.min(...p.S.map(b => pos[b])) }))
    .sort((a, b) => a.first - b.first || a.S.length - b.S.length)
    .map((p, k) => ({ ...p, k, color: catColor(k), key: M.keyOf(p.S) }));
}

export function mountTree(slot, ctx) {
  NOPART = P[NOPART_KEY];
  let pid = ctx.param("pa", "hier"); if (!PRESET_IDS.includes(pid)) pid = "hier";
  let theta = Math.min(0.99, Math.max(0, ctx.param("ta", 0.3)));
  const ctl = ui.h("div", { class: "pt-ctl" });
  const sl = ui.slider({ label: "threshold $\\theta$", min: 0, max: 0.99, step: 0.01, value: theta, fmt: x => fx(x),
    onInput: x => { theta = x; ctx.setParams({ ta: x }); draw(); } });
  const note = ui.h("div", { class: "u-help" });
  const setNote = id => { note.innerHTML = PRESETS.find(p => p.id === id)?.note || ""; ctx.renderMath(note); };
  const seg = ui.segmented({ label: "inputs", options: PRESETS.map(p => ({ value: p.id, label: PRESET_LABEL[p.id] })), value: pid,
    onChange: v => { pid = v; setNote(v); ctx.setParams({ pa: v }); draw(); } });
  seg.el.append(note); setNote(pid);
  ctl.append(sl.el, seg.el);
  const pA = ui.h("div"), pD = ui.h("div");
  pA.append(ui.h("div", { class: "pc-panel-title", html: "$A = \\Sigma - \\theta$: which pairs covary above $\\theta$" }));
  pD.append(ui.h("div", { class: "pc-panel-title", text: "The correlation tree, cut at θ" }));
  const svgA = svg("svg", { viewBox: "0 0 300 300", role: "img", "aria-label": "heatmap of Sigma minus theta" }, pA);
  const svgD = svg("svg", { viewBox: "0 0 300 300", role: "img", "aria-label": "dendrogram of the correlations" }, pD);
  const read = ui.h("div", { class: "pc-read" });
  const foot = ui.h("div", { class: "pc-foot" });
  ui.colorbar({ parent: foot, map: "div", range: [-1, 1], label: "$A_{bc} = \\Sigma_{bc} - \\theta$", width: 220, ticks: [-1, -0.5, 0, 0.5, 1] });
  ui.legend({ parent: foot, items: [{ label: "stable part (outline / branch, one colour each)", color: catColor(0), shape: "ring" },
    { label: "branch with no matching stable part", color: NOPART, shape: "dash" }, { label: "θ", color: P.accent, shape: "dash" }] });
  // filmstrip: the same matrix at five fixed thresholds; click a frame to move the slider there
  const strip = ui.h("div", { class: "pt-strip" });
  const frames = FRAMES.map(t => {
    const box = ui.h("button", { type: "button", class: "pt-frame", title: `show θ = ${fx(t)}` });
    const head = ui.h("div", { class: "pt-frame-head" });
    const sv = svg("svg", { viewBox: "0 0 300 300", role: "img", "aria-label": `Sigma minus theta at theta ${t}` });
    box.append(head, sv);
    box.addEventListener("click", () => { theta = t; sl.set(t); ctx.setParams({ ta: t }); draw(); });
    strip.append(box);
    return { t, head, sv };
  });
  slot.append(ctl, strip, ui.h("div", { class: "pt-panels" }, pA, pD), foot, read);
  ui.renderMath(slot);

  const L0 = 30, T0 = 30, SZ = 264;           // matrix area: left/top label margin, side
  function matrix(el, pr, A, parts, labels = true) {
    el.replaceChildren();
    const B = pr.B, cs = SZ / B;
    const g = svg("g", {}, el);
    for (let r = 0; r < B; r++) for (let c = 0; c < B; c++) {
      const i = pr.leafOrder[r], j = pr.leafOrder[c];
      svg("rect", { x: L0 + c * cs, y: T0 + r * cs, width: cs + 0.4, height: cs + 0.4, fill: colorOf("div", A[i * B + j], -1, 1), "data-i": i, "data-j": j }, g);
    }
    for (let k = 1; k < B; k++) {
      svg("line", { x1: L0 + k * cs, x2: L0 + k * cs, y1: T0, y2: T0 + SZ, stroke: P.surface, "stroke-width": 0.8, "pointer-events": "none" }, el);
      svg("line", { y1: T0 + k * cs, y2: T0 + k * cs, x1: L0, x2: L0 + SZ, stroke: P.surface, "stroke-width": 0.8, "pointer-events": "none" }, el);
    }
    for (let r = 0; r < B && labels; r++) {
      const lab = inputLabel(pr.leafOrder[r]);
      stext(el, L0 - 5, T0 + (r + 0.5) * cs + 4, lab, { "text-anchor": "end" });
      stext(el, L0 + (r + 0.5) * cs, T0 - 7, lab, { "text-anchor": "middle" });
    }
    for (const p of parts) {
      const runs = rowRuns(p.S, pr.pos);
      for (const [a0, a1] of runs) for (const [b0, b1] of runs)
        svg("rect", { x: L0 + b0 * cs + 1.5, y: T0 + a0 * cs + 1.5, width: (b1 - b0 + 1) * cs - 3, height: (a1 - a0 + 1) * cs - 3,
          fill: "none", stroke: p.color, "stroke-width": labels ? 3 : 9, rx: 2, "pointer-events": "none" }, el);
    }
  }
  let curA = null, curPr = null;
  svgA.addEventListener("mousemove", e => {
    const t = e.target; if (!curPr || !t.dataset || t.dataset.i == null) { ui.hideTip(); return; }
    const i = +t.dataset.i, j = +t.dataset.j, B = curPr.B;
    ui.showTip(e.clientX, e.clientY, ui.tipRows(`${inputLabel(i)}, ${inputLabel(j)}`,
      [{ name: "covariance Σ", value: ui.fmt(curPr.Sigma[i * B + j]) }, { name: `A = Σ − θ`, value: ui.fmt(curA[i * B + j]) }]));
  });
  svgA.addEventListener("mouseleave", () => ui.hideTip());

  function dendro(el, pr, parts) {
    el.replaceChildren();
    const B = pr.B, cs = SZ / B, lk = pr.linkage;
    const yOf = v => T0 + 8 + (1 - v) * (SZ - 18);       // correlation 1 at the top, 0 at the bottom
    const xs = new Array(B + lk.length), ys = new Array(B + lk.length);
    for (let b = 0; b < B; b++) { xs[b] = L0 + (pr.pos[b] + 0.5) * cs; ys[b] = 1; }
    lk.forEach(([a, b, hgt], k) => { xs[B + k] = (xs[a] + xs[b]) / 2; ys[B + k] = hgt; });
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      svg("line", { x1: L0 - 3, x2: L0 + SZ, y1: yOf(v), y2: yOf(v), stroke: P.line, "stroke-width": v === 0 ? 1 : 0.5 }, el);
      stext(el, L0 - 5, yOf(v) + 4, v === 0 ? "0" : String(v), { "text-anchor": "end", "font-size": 10, fill: P.mute });
    }
    stext(el, L0 + SZ, yOf(0) + 14, "height: mean correlation of the merged groups", { "text-anchor": "end", "font-size": 10, fill: P.mute });
    const partOf = new Map(parts.map(p => [p.key, p]));
    const mem = id => (id < B ? [id] : [...mem(lk[id - B][0]), ...mem(lk[id - B][1])]);
    const seg = (x1, y1, x2, y2, col, w) => svg("line", { x1, y1, x2, y2, stroke: col, "stroke-width": w, "stroke-linecap": "round" }, el);
    const faint = P.mute2;
    function vert(x, yp, yn, col) {
      if (col && yn > theta) {
        const lo = Math.max(yp, theta);
        if (yp < theta) seg(x, yOf(yp), x, yOf(theta), faint, 1.3);
        const l = seg(x, yOf(lo), x, yOf(yn), col, 2.6);
        if (col === NOPART) l.setAttribute("stroke-dasharray", "4 3");
      } else seg(x, yOf(yp), x, yOf(yn), faint, 1.3);
    }
    function walk(id, yp, col) {
      const yn = ys[id];
      if (!col && yn > theta) {
        const S = mem(id).sort((a, b) => a - b), p = partOf.get(M.keyOf(S));
        col = p ? p.color : NOPART;
      }
      vert(xs[id], yp, yn, col);
      if (id >= B) {
        const [a, b] = lk[id - B], on = col && yn > theta;
        const l = seg(xs[a], yOf(yn), xs[b], yOf(yn), on ? col : faint, on ? 2.6 : 1.3);
        if (on && col === NOPART) l.setAttribute("stroke-dasharray", "4 3");
        walk(a, yn, col); walk(b, yn, col);
      }
    }
    walk(B + lk.length - 1, 0, null);
    for (let b = 0; b < B; b++) stext(el, xs[b], T0 - 7, inputLabel(b), { "text-anchor": "middle" });
    svg("line", { x1: L0 - 3, x2: L0 + SZ, y1: yOf(theta), y2: yOf(theta), stroke: P.accent, "stroke-width": 1.6, "stroke-dasharray": "5 3" }, el);
    stext(el, L0 + SZ, yOf(theta) - 5, `θ = ${fx(theta)}`, { "text-anchor": "end", fill: P.accent, "font-weight": 600 });
  }

  function drawFrames(pr) {
    for (const f of frames) {
      const parts = coloured(M.stableParts(pr.Sigma, pr.B, f.t), pr.pos);
      matrix(f.sv, pr, pr.Sigma.map(v => v - f.t), parts, false);
      const lvl = pr.id === "hier" && [0.15, 0.45, 0.75].some(l => Math.abs(l - f.t) < 1e-9);
      f.head.textContent = `θ = ${fx(f.t)}: ${parts.length} part${parts.length === 1 ? "" : "s"}${lvl ? " (a merge level)" : ""}`;
    }
  }
  function draw() {
    const pr = preset(pid), B = pr.B;
    drawFrames(pr);
    frames.forEach(f => f.head.parentNode.classList.toggle("on", Math.abs(f.t - theta) < 1e-9));
    const A = pr.Sigma.map(v => v - theta);
    const parts = coloured(M.stableParts(pr.Sigma, B, theta), pr.pos);
    curA = A; curPr = pr;
    matrix(svgA, pr, A, parts);
    dendro(svgD, pr, parts);
    const cut = M.cutClusters(pr.linkage, B, theta).filter(S => S.every(b => pr.Sigma[b * B + b] > theta));
    const partKeys = new Set(parts.map(p => p.key)), cutKeys = new Set(cut.map(M.keyOf));
    const same = cut.length === parts.length && parts.every(p => cutKeys.has(p.key));
    const n = parts.length;
    read.innerHTML = `<b>θ = ${fx(theta)}</b>: ${n} stable part${n === 1 ? "" : "s"}${n ? ` (${parts.map(p => `<span class="swatch round" style="--sw:${p.color}"></span>${supportName(p.S)}`).join(", ")})` : ""}; ` +
      (same ? `exactly the ${cut.length} branch${cut.length === 1 ? "" : "es"} above the cut.` :
        `the branches above the cut are ${cut.map(S => supportName(S) + (partKeys.has(M.keyOf(S)) ? "" : " (not a stable part)")).join(", ") || "none"}.`);
  }
  draw();
  ctx.onParams?.(p => {
    const np = p.get("pa"), nt = p.get("ta");
    if (np && PRESET_IDS.includes(np)) { pid = np; seg.set(np); setNote(np); }
    if (nt != null && Number.isFinite(+nt)) { theta = +nt; sl.set(theta); }
    draw();
  });
}
