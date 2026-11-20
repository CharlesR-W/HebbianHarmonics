// Chapter "readout": small figure-specific drawings that the shared charts do not cover.
//   lawPlot    canvas scatter of every neuron pair (D^2 across, w up) with the fixed-point law drawn over it
//   matPanel   a titled heat-map panel (thin wrapper over charts.heatmap)
import * as ui from "./ui.js";
import { heatmap, hidpi, niceTicks } from "./charts.js";
import { P, catColor } from "./palette.js";

const FONT = "11px Inter, system-ui, sans-serif";

// keep a canvas the width of its container; draw(ctx, W, H) on every resize
function sizedCanvas(el, height, draw) {
  const cv = ui.h("canvas", { style: { display: "block" } }); el.replaceChildren(cv);
  let w = 0;
  const run = () => { const cw = Math.floor(el.clientWidth); if (cw > 0) { w = cw; draw(hidpi(cv, cw, height), cw, height); } };
  new ResizeObserver(() => { if (Math.floor(el.clientWidth) !== w) requestAnimationFrame(run); }).observe(el);
  return { cv, redraw: run };
}

/** Scatter of all pairs: x = D^2 (measured), y = w, live vs pruned, with w = (b - D^2/2)_+/alpha drawn over it. */
export function lawPlot(el, { height = 340 } = {}) {
  let st = null, proj = null, hov = null;
  const box = ui.h("div", { class: "chart" }); el.replaceChildren(box);
  const plot = ui.h("div"), leg = ui.legend({ items: [
    { label: "live synapse", color: catColor(0), shape: "dot" },
    { label: "pruned pair (w = 0)", color: P.other, shape: "dot" },
    { label: "fixed point of the rule, w = (b − D²/2)₊ / α", color: P.fg, shape: "line" },
  ] });
  leg.el.classList.add("chart-legend");
  box.append(plot, leg.el);
  const c = sizedCanvas(plot, height, draw);
  function draw(ctx, W, Hh) {
    if (!st) return;
    const { D2, w, live, b, alpha } = st;
    const m = { l: 58, r: 14, t: 12, b: 40 };
    const xr = [0, 4 * b], yr = [-0.04 * b / alpha, 1.08 * b / alpha];
    const X = v => m.l + (v - xr[0]) / (xr[1] - xr[0]) * (W - m.l - m.r), Y = v => Hh - m.b - (v - yr[0]) / (yr[1] - yr[0]) * (Hh - m.t - m.b);
    ctx.font = FONT; ctx.lineWidth = 1;
    // grid and ticks
    ctx.strokeStyle = P.line; ctx.fillStyle = P.mute;
    ctx.textAlign = "center";
    niceTicks(xr[0], xr[1], 6).forEach(v => { ctx.beginPath(); ctx.moveTo(Math.round(X(v)) + .5, m.t); ctx.lineTo(Math.round(X(v)) + .5, Hh - m.b); ctx.stroke(); ctx.fillText(ui.fmt(v, 2), X(v), Hh - m.b + 14); });
    ctx.textAlign = "right";
    niceTicks(0, yr[1], 4).forEach(v => { ctx.beginPath(); ctx.moveTo(m.l, Math.round(Y(v)) + .5); ctx.lineTo(W - m.r, Math.round(Y(v)) + .5); ctx.stroke(); ctx.fillText(ui.fmt(v, 2), m.l - 6, Y(v) + 3.5); });
    ctx.fillStyle = P.ink2; ctx.font = "560 11.5px Inter, system-ui, sans-serif"; ctx.textAlign = "center";
    ctx.fillText(W < 560 ? "response difference D²ᵢⱼ (measured)" : "how differently the two neurons respond, D²ᵢⱼ (measured)", (m.l + W - m.r) / 2, Hh - 6);
    ctx.save(); ctx.translate(13, (m.t + Hh - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("weight wᵢⱼ", 0, 0); ctx.restore();
    // threshold D^2 = 2b
    ctx.setLineDash([4, 4]); ctx.strokeStyle = P.mute2; ctx.beginPath(); ctx.moveTo(X(2 * b), m.t); ctx.lineTo(X(2 * b), Hh - m.b); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = P.mute; ctx.font = FONT; ctx.textAlign = "left"; ctx.fillText("D² = 2b", X(2 * b) + 5, m.t + 11);
    // points: pruned first, then live
    proj = new Float32Array(2 * D2.length);
    let off = 0;
    for (const pass of [false, true]) {
      ctx.fillStyle = pass ? catColor(0) : P.other;
      for (let k = 0; k < D2.length; k++) {
        if (live[k] !== (pass ? 1 : 0)) continue;
        if (D2[k] > xr[1]) { if (!pass) off++; proj[2 * k] = NaN; continue; }
        const px = X(D2[k]), py = Y(w[k]); proj[2 * k] = px; proj[2 * k + 1] = py;
        ctx.beginPath(); ctx.arc(px, py, pass ? 3.2 : 2.6, 0, 2 * Math.PI); ctx.fill();
      }
    }
    // the law, over the points
    ctx.strokeStyle = P.fg; ctx.lineWidth = 1.3; ctx.globalAlpha = .85;
    ctx.beginPath(); ctx.moveTo(X(0), Y(b / alpha)); ctx.lineTo(X(2 * b), Y(0)); ctx.lineTo(X(xr[1]), Y(0)); ctx.stroke(); ctx.globalAlpha = 1;
    if (off) {
      // narrow screens: the note would cross the line near the axis, so put it in the empty top-right corner
      ctx.fillStyle = P.mute; ctx.textAlign = "right";
      if (W < 560) ctx.fillText(`+ ${ui.fmtInt(off)} more pruned →`, W - m.r - 2, m.t + 30);
      else ctx.fillText(`+ ${ui.fmtInt(off)} more pruned pairs further right →`, W - m.r - 2, Y(0) - 10);
    }
    if (hov != null && Number.isFinite(proj[2 * hov])) { ctx.strokeStyle = P.fg; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(proj[2 * hov], proj[2 * hov + 1], 6, 0, 2 * Math.PI); ctx.stroke(); }
  }
  c.cv.addEventListener("pointermove", ev => {
    if (!proj || !st) return;
    const r = c.cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let bi = null, bd = 64;
    for (let k = 0; k < st.D2.length; k++) { const dx = proj[2 * k] - x, dy = proj[2 * k + 1] - y, d = dx * dx + dy * dy; if (d < bd) { bd = d; bi = k; } }
    if (bi !== hov) { hov = bi; c.redraw(); }
    if (bi == null) { ui.hideTip(); return; }
    const pred = Math.max(st.b - st.D2[bi] / 2, 0) / st.alpha;
    ui.showTip(ev.clientX, ev.clientY, ui.tipRows(`neurons ${st.pi[bi] + 1} and ${st.pj[bi] + 1}`, [
      { name: "D² (measured)", value: ui.fmt(st.D2[bi], 4) },
      { name: "weight w", value: ui.fmt(st.w[bi], 4) },
      { name: "rule's fixed point", value: ui.fmt(pred, 4) },
    ], st.live[bi] ? "live synapse" : "pruned pair"));
  });
  c.cv.addEventListener("pointerleave", () => { hov = null; ui.hideTip(); c.redraw(); });
  return { update(s) { st = s; c.redraw(); } };
}

/** A heat-map panel with a small kicker and a title above it; returns the heatmap. */
export function matPanel(parent, { kicker, title, cls = "", ...opts }) {
  const body = ui.h("div");
  const head = ui.h("div", { class: "ro-head" }, kicker ? ui.h("span", { class: "ro-kicker", text: kicker }) : null, ui.h("span", { class: "ro-title", html: title || "" }));
  const p = ui.h("div", { class: "ro-panel " + cls }, head, body);
  parent.append(p); ui.renderMath(head);
  const hm = heatmap(body, opts);
  hm.setTitle = html => { head.querySelector(".ro-title").innerHTML = html; ui.renderMath(head); };
  hm.panel = p;
  return hm;
}
