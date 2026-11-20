// Chapter "perron", figure "groups are parts, not eigenvectors" ([data-slot="big"]), from the precomputed big runs
// (site/data/big/cov_all.json, hebb/big/cov_runs.py; findings_v2 B1). Left, for one run: each group's mixture (its
// centroid, scaled to unit length) as bars with the predicted part as ink dots, and the top eigenvectors of Σ as signed
// bars on the same scale. Right: every group of every B = 12–16 run, cosine with its predicted part against cosine
// with the closest eigenvector of Σ (both read from the file; degenerate eigenspaces grouped, as in cov_runs.py).
import * as ui from "./ui.js";
import { svgEl } from "./charts.js";
import { P, catColor, colorOf } from "./palette.js";
import { eigh } from "./engine.js";

const SUBS = "₀₁₂₃₄₅₆₇₈₉", sub = n => String(n).split("").map(c => SUBS[+c]).join("");
const gl = b => `g${sub(b + 1)}`;
const OMEGA = 0.009;
const DEFAULT_RUN = "ring16_0.5";
const NAME = { blocks12: "3 blocks of 4 inputs", hier16: "hierarchy, 16 inputs", ring16: "ring, 16 inputs" };
const isMajor = (c, N) => c.size >= Math.max(2, 0.02 * N);
const runKey = r => `${r.name}_${r.theta}`;
const Bof = r => r.config.Sigma.length;
const unit = v => { const n = Math.hypot(...v) || 1; return v.map(x => x / n); };
const N_EIG = 4;

export async function mountBig(slot, ctx) {
  const data = await fetch("data/big/cov_all.json").then(r => { if (!r.ok) throw new Error(`cov_all.json: HTTP ${r.status}`); return r.json(); });
  const all = data.runs.filter(r => Bof(r) >= 12).sort((a, b) => a.name.localeCompare(b.name) || a.theta - b.theta || b.omega - a.omega);
  const pickable = all.filter(r => r.omega === OMEGA && r.seed === 0);
  let cur = pickable.find(r => runKey(r) === ctx.param("pr", DEFAULT_RUN)) || pickable[0];

  slot.replaceChildren();
  const sel = ui.select({ label: "run", options: pickable.map(r => ({ value: runKey(r), label: `${NAME[r.name]}, θ = ${r.theta}` })), value: runKey(cur),
    onChange: k => { cur = pickable.find(r => runKey(r) === k) || cur; ctx.setParams({ pr: k === DEFAULT_RUN ? null : k }); draw(); } });
  sel.el.style.maxWidth = "380px";
  const left = ui.h("div", { class: "pb-rows" }), right = ui.h("div");
  const sumLine = ui.h("div", { class: "pc-panel-sub" });
  left.append(sumLine);
  const rowsBox = ui.h("div"); left.append(rowsBox);
  right.append(ui.h("div", { class: "pc-panel-title", text: `Every group of ≥ ${Math.max(2, Math.ceil(0.02 * all[0].N))} neurons, all ${all.length} runs` }));
  const scBox = ui.h("div"); right.append(scBox);
  const read = ui.h("div", { class: "pc-read" });
  slot.append(ui.h("div", { class: "pt-ctl" }, sel.el), ui.h("div", { class: "pb-wrap" }, left, right), read);

  // scatter data: every major group of every run (both floors)
  const pts = [];
  for (const r of all) r.clusters.filter(c => isMajor(c, r.N)).forEach((c, k) => pts.push({ r, c, k }));
  const strip = stripPlot(scBox, pts);
  const minPart = Math.min(...pts.map(p => p.c.nearest_stable_part_cos)), eigLo = Math.min(...pts.map(p => p.c.best_eig_cos)), eigHi = Math.max(...pts.map(p => p.c.best_eig_cos));
  read.innerHTML = `Over ${all.length} runs and ${pts.length} groups of at least ${Math.max(2, Math.ceil(0.02 * all[0].N))} neurons, the cosine with the nearest stable part is at least ${ui.fmtFixed(Math.floor(minPart * 1e4) / 1e4, 4)}; with the closest eigenvector of Σ it lies between ${ui.fmtFixed(eigLo, 2)} and ${ui.fmtFixed(eigHi, 2)}.`;

  function draw() {
    const r = cur, B = Bof(r), Sigma = Float64Array.from(r.config.Sigma.flat());
    const groups = r.clusters.filter(c => isMajor(c, r.N));
    const parts = r.metrics.stable_parts || [];
    const gRows = groups.map((c, k) => {
      const part = parts.find(s => s.S.join(",") === (c.nearest_stable_part_support || c.support).join(","));
      return { k, c, bars: unit(c.centroid), dots: part ? unit(part.h) : null };
    });
    const { values, vectors } = eigh(Sigma, B);
    const eRows = Array.from({ length: N_EIG }, (_, q) => {
      let v = Array.from({ length: B }, (_, i) => vectors[i * B + q]);
      const s = v.reduce((a, x) => a + x, 0) || v[0];
      if (s < 0) v = v.map(x => -x);
      return { q, v, lam: values[q] };
    });
    let m = 0;
    for (const g of gRows) for (const x of [...g.bars, ...(g.dots || [])]) m = Math.max(m, Math.abs(x));
    for (const e of eRows) for (const x of e.v) m = Math.max(m, Math.abs(x));
    m = Math.ceil(m * 10) / 10;
    const em = Math.max(...eRows.flatMap(e => e.v.map(Math.abs)));      // colour range of the eigenvector bars
    sumLine.innerHTML = `${NAME[r.name]}, θ = ${r.theta}, ${r.N} neurons: ${groups.length} group${groups.length === 1 ? "" : "s"} of at least ${Math.max(2, Math.ceil(0.02 * r.N))} neurons (${r.metrics.n_stable_parts} stable parts exist).`;
    rowsBox.replaceChildren(
      ui.h("div", { class: "pc-panel-title", text: "Mixtures of the groups (bars) and their predicted parts (dots)" }),
      ...gRows.map(g => barRow({ B, m, signed: false, values: g.bars, dots: g.dots, color: catColor(g.k),
        label: `${g.c.size} neurons`, tip: b => ui.tipRows(`${gl(b)}, group of ${g.c.size}`, [
          { name: "group mixture (unit length)", value: ui.fmtFixed(g.bars[b], 3), color: catColor(g.k), shape: "square" },
          { name: "predicted part (unit length)", value: g.dots ? ui.fmtFixed(g.dots[b], 3) : "n/a", color: P.fg }]) })),
      ui.h("div", { class: "pc-panel-title", style: { marginTop: "14px" }, text: `The top ${N_EIG} eigenvectors of Σ` }),
      ...eRows.map(e => barRow({ B, m, cm: em, signed: true, values: e.v, label: `eigenvalue ${ui.fmtFixed(e.lam, 2)}`,
        tip: b => ui.tipRows(`${gl(b)}, eigenvector ${e.q + 1}`, [{ name: "entry (unit length)", value: ui.fmtFixed(e.v[b], 3) }]) })),
      axisRow(B));
    // highlight this run's groups in the scatter with their colours
    strip.highlight(r);
  }
  draw();
  ctx.onParams?.(p => { const k = p.get("pr"); const r = k && pickable.find(x => runKey(x) === k); if (r && r !== cur) { cur = r; sel.set(k); draw(); } });
}

// one row of bars over the inputs: label at the left, bars (colour, or diverging colours when signed), optional dots
const LW = 104;
function barRow(o) {
  const box = ui.h("div", { class: "chart" });
  const render = () => {
    const W = Math.max(260, box.clientWidth || 500), unitPx = 34;
    const H = Math.round(o.signed ? 2 * o.m * unitPx + 6 : o.m * unitPx + 6), y0 = o.signed ? H / 2 : H - 3;
    const bw = (W - LW - 4) / o.B, X = i => LW + (i + 0.5) * bw, Y = v => y0 - v * unitPx;
    const s = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img" });
    const t = svgEl("text", { x: LW - 8, y: y0 - (o.signed ? -4 : 6), "text-anchor": "end" }, s); t.textContent = o.label;
    svgEl("line", { x1: LW, x2: W - 4, y1: y0, y2: y0, stroke: P.mute2, "stroke-width": 1 }, s);
    o.values.forEach((v, i) => svgEl("rect", { x: X(i) - bw * 0.38, width: bw * 0.76, y: Math.min(Y(v), y0), height: Math.abs(Y(v) - y0),
      fill: o.signed ? colorOf("div", v, -o.cm, o.cm) : o.color, rx: 1 }, s));
    if (o.dots) o.dots.forEach((v, i) => { if (v > 1e-6) svgEl("circle", { cx: X(i), cy: Y(v), r: 3, fill: P.fg, stroke: P.surface, "stroke-width": 1 }, s); });
    const hit = svgEl("rect", { x: LW, y: 0, width: W - LW - 4, height: H, fill: "transparent" }, s);
    hit.addEventListener("pointermove", ev => { const rr = s.getBoundingClientRect(), i = Math.max(0, Math.min(o.B - 1, Math.floor((ev.clientX - rr.left - LW) / bw))); ui.showTip(ev.clientX, ev.clientY, o.tip(i)); });
    hit.addEventListener("pointerleave", () => ui.hideTip());
    box.replaceChildren(s);
  };
  new ResizeObserver(() => requestAnimationFrame(render)).observe(box);
  return box;
}
function axisRow(B) {
  const box = ui.h("div", { class: "chart" });
  const render = () => {
    const W = Math.max(260, box.clientWidth || 500), bw = (W - LW - 4) / B, s = svgEl("svg", { width: W, height: 30, viewBox: `0 0 ${W} 30` });
    const every = bw < 22 ? 2 : 1;
    for (let b = 0; b < B; b += every) { const t = svgEl("text", { x: LW + (b + 0.5) * bw, y: 12, "text-anchor": "middle" }, s); t.textContent = gl(b); }
    const t = svgEl("text", { x: LW + (W - LW) / 2, y: 27, "text-anchor": "middle" }, s); t.textContent = "input";
    box.replaceChildren(s);
  };
  new ResizeObserver(() => requestAnimationFrame(render)).observe(box);
  return box;
}

// Two tracks on a shared 0–1 axis: each group's cosine with its predicted part (top) and with the closest eigenvector
// of Σ (bottom), one dot per group, spread vertically within a track so dots do not hide each other. The groups of
// the selected run are drawn in their colours, the rest grey.
function stripPlot(box, pts) {
  let sel = null;
  const tip = p => ui.tipRows(`${NAME[p.r.name]}, θ = ${p.r.theta}${p.r.omega === OMEGA ? "" : " (lower floor)"}`, [
    { name: "group", value: `${p.c.size} neurons on ${p.c.support.length} input${p.c.support.length === 1 ? "" : "s"}` },
    { name: "cosine with nearest stable part", value: ui.fmtFixed(p.c.nearest_stable_part_cos, 5) },
    { name: "cosine with closest eigenvector", value: ui.fmtFixed(p.c.best_eig_cos, 3) }]);
  const jit = pts.map((_, i) => ((i * 0.6180339887) % 1) - 0.5);       // fixed, evenly spread offsets
  const render = () => {
    const W = Math.max(240, box.clientWidth || 300), H = 200, l = 14, r = 14, X = v => l + v * (W - l - r);
    const tracks = [{ y: 50, name: "cosine with the nearest stable part", get: p => p.c.nearest_stable_part_cos }, { y: 138, name: "best cosine with an eigenspace of Σ", get: p => p.c.best_eig_cos }];
    const s = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "cosines of every group with its part and with the closest eigenvector" });
    for (const tr of tracks) {
      const t = svgEl("text", { x: l, y: tr.y - 30, "font-weight": 600 }, s); t.textContent = tr.name;
      svgEl("line", { x1: X(0), x2: X(1), y1: tr.y + 22, y2: tr.y + 22, stroke: P.line }, s);
      for (const v of [0, 0.25, 0.5, 0.75, 1]) {
        svgEl("line", { x1: X(v), x2: X(v), y1: tr.y - 20, y2: tr.y + 25, stroke: P.line, "stroke-width": 0.6 }, s);
        const tt = svgEl("text", { x: X(v), y: tr.y + 37, "text-anchor": "middle" }, s); tt.textContent = ui.fmt(v);
      }
    }
    const order = pts.map((p, i) => i).sort((a, b) => (pts[a].r === sel) - (pts[b].r === sel));
    for (const tr of tracks) for (const i of order) {
      const p = pts[i], on = p.r === sel;
      const c = svgEl("circle", { cx: X(tr.get(p)), cy: tr.y + jit[i] * 30, r: on ? 5 : 3.4, fill: on ? catColor(p.k) : P.other, stroke: P.surface, "stroke-width": 1 }, s);
      c.addEventListener("pointermove", ev => ui.showTip(ev.clientX, ev.clientY, tip(p)));
      c.addEventListener("pointerleave", () => ui.hideTip());
    }
    const t = svgEl("text", { x: (X(0) + X(1)) / 2, y: H - 4, "text-anchor": "middle" }, s); t.textContent = "cosine (1 = same direction)";
    box.replaceChildren(s);
  };
  box.classList.add("chart");
  new ResizeObserver(() => requestAnimationFrame(render)).observe(box);
  return { highlight(r) { sel = r; render(); } };
}
