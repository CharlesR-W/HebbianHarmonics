// Chapter "perron", figure "which arrangement of groups, where" ([data-slot="phase"]): the plane of threshold θ
// against the correlation ρ of g1 and g2 (g3 correlates 0.3 with both, unit variances), coloured by the predicted
// arrangement of groups (the set of stable parts, exact enumeration, precomputed by hebb/perron/phase3.mjs as
// θ-intervals per ρ-row), with an icon per region, and the 80 precomputed runs (same file) as dots marked by whether
// they ended in that arrangement. Beside it, one live run at the point you pick (mountLive in ch_perron_tri.js).
import * as ui from "./ui.js";
import { hidpi, niceTicks } from "./charts.js";
import { P, catColor, hexToRgb, rgbToCss } from "./palette.js";
import * as M from "./ch_perron_model.js";
import * as T from "./ch_perron_teach_model.js";
import { mountLive, triGeom, BCOL, FONT, fx, hstr, nearestPart } from "./ch_perron_tri.js";

const R13 = 0.3, R23 = 0.3;
// the four arrangements that fill the plane (ties between them are lines); colours are region identities
export const CLASSES = {
  "012": { name: "one group on all three inputs", color: catColor(6) },
  "01|2": { name: "a g₁g₂ group and a g₃ group", color: catColor(4) },
  "02|12": { name: "a g₁g₃ group and a g₂g₃ group", color: catColor(3) },
  "0|1|2": { name: "one group per input", color: catColor(5) },
};
const keyOfParts = parts => parts.map(p => p.S.join("")).sort().join("|");
const sigmaAt = rho => M.sigmaOf({ v: [1, 1, 1], r: [rho, R13, R23] });
// an opaque tint (the colour mixed with white), so that abutting rows do not darken where they overlap
const tint = (hex, a) => rgbToCss(hexToRgb(hex).map(v => v * a + 255 * (1 - a)));

// How a run ended, against the prediction at its (θ, ρ): every group within 0.1 (sum of absolute differences) of a
// stable part and every part holding a group ("as predicted"); groups on parts but some parts empty ("partial");
// or some group off every part ("off").
export function outcome(run) {
  const parts = T.stableParts(sigmaAt(run.rho), 3, run.theta);
  const near = run.groups.map(g => nearestPart(parts, g.h));
  if (near.some(x => !(x.d < 0.1))) return { kind: "off", parts };
  const filled = new Set(near.map(x => x.part));
  return { kind: filled.size === parts.length ? "ok" : "partial", parts };
}

export async function mountPhase(slot, ctx) {
  const data = await fetch("data/perron/phase3.json").then(r => { if (!r.ok) throw new Error(`phase3.json: HTTP ${r.status}`); return r.json(); });
  const map = data.map, runs = data.runs.map(r => ({ ...r, out: outcome(r) }));
  slot.replaceChildren();
  const mapBox = ui.h("div", { class: "ph-map" });
  const cv = ui.h("canvas", { style: { display: "block", width: "100%", cursor: "crosshair", touchAction: "pan-y" } });
  mapBox.append(ui.h("div", { class: "pc-panel-title", text: "Predicted arrangement, and how 80 runs ended" }), cv);
  let cur = { theta: 0.45, rho: 0.6 }, ready = false;
  const live = mountLive(slot, ctx, { r13: R13, r23: R23, before: mapBox, onRun: (t, r) => { cur = { theta: t, rho: r }; if (ready) render(); } });

  // legend: regions (with their names) and the three kinds of run dot
  const foot = ui.h("div", { class: "pc-foot" });
  ui.legend({ parent: foot, items: Object.values(CLASSES).map(c => ({ label: c.name, color: tint(c.color, 0.45), shape: "square" })) });
  ui.legend({ parent: foot, items: [
    { label: "run ended in the predicted arrangement", color: P.fg, shape: "dot" },
    { label: "groups on predicted parts, but some parts empty", color: P.fg, shape: "ring" },
    { label: "the run shown in the triangle", color: P.accent, shape: "ring" }] });
  const read = ui.h("div", { class: "pc-read" });
  slot.append(foot, read);
  const nOk = runs.filter(r => r.out.kind === "ok").length, nPart = runs.filter(r => r.out.kind === "partial").length, nOff = runs.length - nOk - nPart;
  read.textContent = `Of the ${runs.length} runs, ${nOk} ended in the predicted arrangement and ${nPart} put every group on a predicted part but left ${nPart === 1 ? "a part" : "parts"} empty` +
    `${nOff ? `; ${nOff} had a group off every predicted part` : ""}. All ${runs.filter(r => r.converged).length === runs.length ? runs.length : runs.filter(r => r.converged).length} converged.`;

  // region icons: for each class, the midpoint of its segment on the median row where it occurs (computed)
  const icons = Object.keys(CLASSES).map(key => {
    const rows = map.segments.map((r, j) => {
      const k = r.seg.findIndex(s => s[1] === key); if (k < 0) return null;
      // the longest interval of this class in the row
      let best = null;
      r.seg.forEach((s, i) => { if (s[1] !== key) return; const end = i + 1 < r.seg.length ? r.seg[i + 1][0] : map.theta[1]; if (!best || end - s[0] > best.len) best = { a: s[0], len: end - s[0] }; });
      return { j, rho: r.rho, theta: best.a + best.len / 2, len: best.len };
    }).filter(Boolean);
    const wide = rows.filter(r => r.len >= 0.6 * Math.max(...rows.map(x => x.len)));
    const m = wide[Math.floor(wide.length / 2)];
    return { key, theta: m.theta, rho: m.rho, parts: T.stableParts(sigmaAt(m.rho), 3, m.theta) };
  });

  let W = 0, Lay = null;
  function render() {
    W = Math.floor(mapBox.clientWidth); if (W <= 0) return;
    const H = Math.round(Math.min(460, W * 0.95)), m = { l: 48, r: 8, t: 6, b: 40 };
    const X = t => m.l + (t - map.theta[0]) / (map.theta[1] - map.theta[0]) * (W - m.l - m.r);
    const Y = r => H - m.b - (r - map.rho[0]) / (map.rho[1] - map.rho[0]) * (H - m.t - m.b);
    Lay = { m, X, Y, W, H };
    const g = hidpi(cv, W, H); g.clearRect(0, 0, W, H);
    const nr = map.segments.length, rh = (H - m.t - m.b) / (nr - 1);
    // regions
    map.segments.forEach((r, j) => {
      const y = Y(r.rho) - rh / 2;
      r.seg.forEach((s, i) => {
        const end = i + 1 < r.seg.length ? r.seg[i + 1][0] : map.theta[1], c = CLASSES[s[1]];
        g.fillStyle = c ? tint(c.color, 0.32) : P.surface;
        const xa = X(s[0]), xb = X(end);
        g.fillRect(Math.floor(xa), Math.floor(y), Math.ceil(xb - xa) + 1, Math.ceil(rh) + 1);
      });
    });
    // boundaries: the breaks between two of the four classes, joined row to row
    g.strokeStyle = P.ink2; g.lineWidth = 1.3;
    const pts = map.segments.map(r => r.seg.slice(1).map((s, i) => ({ x: X(s[0]), y: Y(r.rho), pair: r.seg[i][1] + ">" + s[1] })).filter(b => CLASSES[b.pair.split(">")[0]] || CLASSES[b.pair.split(">")[1]]));
    for (let j = 1; j < pts.length; j++) for (const b of pts[j]) {
      let best = null, bd = 6;
      for (const a of pts[j - 1]) { const d = Math.abs(a.x - b.x); if (d < bd) { bd = d; best = a; } }
      if (best) { g.beginPath(); g.moveTo(best.x, best.y); g.lineTo(b.x, b.y); g.stroke(); }
    }
    // axes
    g.strokeStyle = P.line; g.lineWidth = 1; g.fillStyle = P.mute; g.font = `10.5px ${FONT}`;
    g.textAlign = "center"; g.textBaseline = "top";
    for (const t of niceTicks(map.theta[0], map.theta[1], 5)) { g.beginPath(); g.moveTo(X(t), H - m.b); g.lineTo(X(t), H - m.b + 4); g.stroke(); g.fillText(ui.fmt(t), X(t), H - m.b + 6); }
    g.textAlign = "right"; g.textBaseline = "middle";
    for (const r of niceTicks(map.rho[0], map.rho[1], 6)) { g.beginPath(); g.moveTo(m.l - 4, Y(r)); g.lineTo(m.l, Y(r)); g.stroke(); g.fillText(ui.fmt(r), m.l - 6, Y(r)); }
    g.strokeRect(m.l + 0.5, m.t + 0.5, W - m.l - m.r - 1, H - m.t - m.b - 1);
    g.fillStyle = P.ink2; g.font = `560 11.5px ${FONT}`; g.textAlign = "center"; g.textBaseline = "alphabetic";
    g.fillText("pruning threshold θ", (m.l + W - m.r) / 2, H - 6);
    g.save(); g.translate(13, (m.t + H - m.b) / 2); g.rotate(-Math.PI / 2); g.fillText("correlation of g₁ and g₂", 0, 0); g.restore();
    // icons: a small triangle with the predicted groups (ink dots), on a white card
    const s = W < 420 ? 34 : 44;
    for (const ic of icons) {
      const cx = X(ic.theta), cy = Y(ic.rho);
      const G = triGeom(s, s * 0.95, 5), ox = cx - s / 2, oy = cy - s * 0.95 / 2;
      g.fillStyle = "rgba(255,255,255,0.85)"; g.strokeStyle = CLASSES[ic.key].color; g.lineWidth = 1.5;
      g.beginPath(); g.roundRect ? g.roundRect(ox - 2, oy - 2, s + 4, s * 0.95 + 4, 4) : g.rect(ox - 2, oy - 2, s + 4, s * 0.95 + 4); g.fill(); g.stroke();
      g.strokeStyle = P.mute2; g.lineWidth = 1; g.beginPath(); G.V.forEach(([x, y], k) => g[k ? "lineTo" : "moveTo"](ox + x, oy + y)); g.closePath(); g.stroke();
      G.V.forEach(([x, y], b) => { g.fillStyle = BCOL[b]; g.fillRect(ox + x - 2.5, oy + y - 2.5, 5, 5); });
      for (const p of ic.parts) { const [x, y] = G.at(p.h); g.beginPath(); g.arc(ox + x, oy + y, 3.6, 0, 2 * Math.PI); g.fillStyle = P.fg; g.fill(); }
    }
    // runs
    for (const r of runs) {
      const x = X(r.theta), y = Y(r.rho);
      g.beginPath(); g.arc(x, y, 4.2, 0, 2 * Math.PI);
      if (r.out.kind === "ok") { g.fillStyle = P.fg; g.fill(); g.lineWidth = 1.2; g.strokeStyle = P.surface; g.stroke(); }
      else if (r.out.kind === "partial") { g.fillStyle = P.surface; g.fill(); g.lineWidth = 1.8; g.strokeStyle = P.fg; g.stroke(); }
      else { g.strokeStyle = P.accent; g.lineWidth = 2; g.beginPath(); g.moveTo(x - 4, y - 4); g.lineTo(x + 4, y + 4); g.moveTo(x - 4, y + 4); g.lineTo(x + 4, y - 4); g.stroke(); }
    }
    // the live run
    const cx = X(cur.theta), cy = Y(cur.rho);
    g.beginPath(); g.arc(cx, cy, 8, 0, 2 * Math.PI); g.lineWidth = 2.2; g.strokeStyle = P.accent; g.stroke();
  }
  ready = true;
  new ResizeObserver(() => { const w = Math.floor(mapBox.clientWidth); if (w > 0 && w !== W) requestAnimationFrame(render); }).observe(mapBox);
  render();

  const at = e => {
    if (!Lay) return null;
    const rc = cv.getBoundingClientRect(), x = e.clientX - rc.left, y = e.clientY - rc.top, { m, W: w, H: h } = Lay;
    if (x < m.l || x > w - m.r || y < m.t || y > h - m.b) return null;
    return { x, y, theta: map.theta[0] + (x - m.l) / (w - m.l - m.r) * (map.theta[1] - map.theta[0]), rho: map.rho[0] + (h - m.b - y) / (h - m.t - m.b) * (map.rho[1] - map.rho[0]) };
  };
  let drag = false;
  cv.addEventListener("pointerdown", e => { const p = at(e); if (!p) return; drag = true; live.set(+p.theta.toFixed(2), +p.rho.toFixed(2)); });
  cv.addEventListener("pointerup", () => { drag = false; });
  cv.addEventListener("pointermove", e => {
    const p = at(e); if (!p) { ui.hideTip(); return; }
    if (drag && e.buttons) live.set(+p.theta.toFixed(2), +p.rho.toFixed(2));
    const parts = T.stableParts(sigmaAt(p.rho), 3, p.theta), key = keyOfParts(parts);
    const rows = [{ name: "predicted", value: CLASSES[key]?.name || (parts.length ? parts.map(q => hstr(q.h)).join(" ") : "no strict part (a tie)") }];
    const near = runs.find(r => Math.hypot(Lay.X(r.theta) - p.x, Lay.Y(r.rho) - p.y) < 8);
    if (near) rows.push({ name: `run at θ = ${fx(near.theta)}, ρ = ${fx(near.rho)}`, value: near.groups.map(gr => `${gr.n} at ${hstr(gr.h)}`).join("; ") });
    ui.showTip(e.clientX, e.clientY, ui.tipRows(`θ = ${fx(p.theta)}, ρ = ${fx(p.rho)}`, rows, "click to run the network here"));
  });
  cv.addEventListener("pointerleave", () => { drag = false; ui.hideTip(); });
}
