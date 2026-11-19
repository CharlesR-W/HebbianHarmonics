// Chapter "nonlinear" (nonlinear neurons; findings B2, B3, G12, G23). Figures:
//   transfer  one neuron's potential and transfer curve, two thresholds (miniPlot from the idea chapter's drawing helpers)
//   films     linear vs rectifier films, matched protocol, and the three-arm table (ch_nonlinear_big.js)
//   nl        the distance readout of a rectifying network, live, and the N = 40 sweep table (ch_nonlinear_readout.js,
//             moved from the readout chapter)
//   readout-big, div   distance readout and diversity on the big runs (ch_nonlinear_big.js)
import * as ui from "./ui.js";
import { P, catColor } from "./palette.js";
import { miniPlot } from "./ch_idea_draw.js";
import { nlFigure, nlTable } from "./ch_nonlinear_readout.js";
import { initBig } from "./ch_nonlinear_big.js";

export async function init(panel, ctx) {
  transferFigure(ctx);
  const nl = ctx.slot("nl") ? nlFigure(ctx) : null;
  ctx.onParams(() => { nl?.fromURL(); });
  const sweepP = fetch("data/sweeps2/readout.json").then(r => { if (!r.ok) throw new Error(`readout.json: HTTP ${r.status}`); return r.json(); })
    .then(data => nlTable(ctx, data));
  await Promise.all([initBig(ctx), sweepP]);
}

// ---------------------------------------------------------------- (1) the transfer curve
const TDEF = { klo: 3, khi: 0, t1: -0.5, t2: 0.5 };
function transferFigure(ctx) {
  const slot = ctx.slot("transfer"); if (!slot) return;
  slot.replaceChildren();
  const d = 1;
  const p = { klo: ctx.param("klo", TDEF.klo), khi: ctx.param("khi", TDEF.khi), t1: ctx.param("t1", TDEF.t1), t2: ctx.param("t2", TDEF.t2) };
  const write = () => ctx.setParams(Object.fromEntries(Object.entries(p).map(([k, v]) => [k, v === TDEF[k] ? null : v])));
  const sl = (key, label, min, max, step) => ui.slider({ label, min, max, step, value: p[key], fmt: x => ui.fmtFixed(x, 2), onInput: x => { p[key] = x; draw(); write(); } });
  const controls = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(200px, 100%), 1fr))", gap: "6px 24px", marginBottom: "10px" } },
    sl("klo", "below threshold $k_{lo}$", 0, 10, 0.1).el, sl("khi", "above threshold $k_{hi}$", 0, 3, 0.1).el,
    sl("t1", "threshold $t_1$", -2, 2, 0.05).el, sl("t2", "threshold $t_2$", -2, 2, 0.05).el);
  const note = ui.h("p", { class: "u-help", style: { margin: "0 0 6px" } });
  const grid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(300px, 100%), 1fr))", gap: "12px 28px" } });
  const boxA = ui.h("div"), boxB = ui.h("div");
  const head = (t) => { const e = ui.h("div", { class: "chart-title", html: t }); ui.renderMath(e); return e; };
  grid.append(ui.h("div", {}, head("(a) On-site potential $V(u)$"), boxA), ui.h("div", {}, head("(b) Transfer curve: voltage $u$ against input current $I$"), boxB));
  slot.append(controls, note, grid); ui.renderMath(controls);
  const pa = miniPlot(boxA, { height: 230, xlabel: "voltage u", ylabel: "V(u)" });
  const pb = miniPlot(boxB, { height: 230, xlabel: "input current I", ylabel: "voltage u" });
  const cols = [catColor(0), catColor(1)];
  const V = (u, t) => { const x = u - t; return 0.5 * (x < 0 ? p.klo : p.khi) * x * x; };
  const uOf = (I, t) => (I >= d * t ? (I + p.khi * t) / (d + p.khi) : (I + p.klo * t) / (d + p.klo));
  // E over I ~ N(0,1) by Gauss-Hermite-like midpoint quadrature on [-6, 6]
  const expect = f => { let s = 0, w = 0; for (let k = 0; k <= 1200; k++) { const I = -6 + 12 * k / 1200, q = Math.exp(-I * I / 2); s += q * f(I); w += q; } return s / w; };
  function draw() {
    const us = Array.from({ length: 241 }, (_, k) => -3 + 6 * k / 240);
    const Vmax = Math.max(1, ...us.map(u => Math.max(V(u, p.t1), V(u, p.t2))));
    pa.draw({ xr: [-3, 3], yr: [0, Math.min(Vmax, 12)], layers: [
      { type: "line", x: us, y: us.map(u => V(u, p.t1)), color: cols[0], width: 2 },
      { type: "line", x: us, y: us.map(u => V(u, p.t2)), color: cols[1], width: 2 },
      { type: "vline", v: p.t1, color: cols[0], label: "t₁" }, { type: "vline", v: p.t2, color: cols[1], label: "t₂" },
    ] });
    const Is = Array.from({ length: 241 }, (_, k) => -3 + 6 * k / 240);
    pb.draw({ xr: [-3, 3], yr: [-3, 3], layers: [
      { type: "line", x: Is, y: Is.map(I => I / d), color: P.mute, dash: "4 3", width: 1.2 },
      { type: "line", x: Is, y: Is.map(I => uOf(I, p.t1)), color: cols[0], width: 2 },
      { type: "line", x: Is, y: Is.map(I => uOf(I, p.t2)), color: cols[1], width: 2 },
      { type: "hline", v: p.t1, color: cols[0] }, { type: "hline", v: p.t2, color: cols[1] },
    ] });
    const D2 = expect(I => (uOf(I, p.t1) - uOf(I, p.t2)) ** 2), m1 = expect(I => uOf(I, p.t1) ** 2);
    note.innerHTML = `Same current $I \\sim \\mathcal N(0, 1)$ into both: $D^2 = \\mathbb E[(u_1 - u_2)^2] = ${ui.fmt(D2, 3)}$, against $\\mathbb E[u_1^2] = ${ui.fmt(m1, 3)}$.` +
      (p.t1 === p.t2 ? " Equal thresholds: the two neurons are one function." : p.klo === p.khi ? " $k_{lo} = k_{hi}$: both are linear, shifted, and differ by a constant." : "");
    ui.renderMath(note);
  }
  draw();
}
