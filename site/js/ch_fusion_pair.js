// Chapter "fusion": the two-neuron calibration figure. Analytic drift and fixed points (ch_fusion_model.js),
// the bifurcation diagram with the bistable window, and a live check with the engine itself.
import * as ui from "./ui.js";
import { P } from "./palette.js";
import { svgPlot, svgEl } from "./ch_fusion_draw.js";
import * as M from "./ch_fusion_model.js";

export const PAIR_DEFAULTS = { b: 0.4, alpha: 0.2, D02: 2 };
export const PAIR_TOURS = [
  { id: "bistable", label: "Bistable", params: { D02: 2, b: 0.4, alpha: 0.2 },
    note: "$D_0^2 = 2$ (two independent unit-variance inputs), $b = 0.4$, $\\alpha = 0.2$: inside the window, so $w = 0$ and the fused state both attract." },
  { id: "stiff", label: "Stiff, no window", params: { D02: 2, b: 1.5, alpha: 5 },
    note: "$\\alpha = 5 &gt; 2D_0^2$: no saddle-node; above $b = D_0^2/2$ the edge grows continuously from $w = 0$ and there is one fixed point for every $b$." },
];

export function initPair(slot, ctx, { loop } = {}) {
  const tourId = ctx.param("ptour", "");
  let tour = PAIR_TOURS.find(t => t.id === tourId) || null;
  let p = tour ? { ...tour.params } : { b: ctx.param("pb", PAIR_DEFAULTS.b), alpha: ctx.param("pa", PAIR_DEFAULTS.alpha), D02: ctx.param("pd", PAIR_DEFAULTS.D02) };
  clamp(p);

  const chips = ui.chips({ parent: slot, label: "Presets", items: PAIR_TOURS, active: tour ? tour.id : null, onPick: t => { tour = t; p = { ...t.params }; sync(); update(true); } });
  const grid3 = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "6px 24px", margin: "10px 0 6px" } });
  slot.append(grid3);
  const moved = () => { if (tour) { tour = null; chips.moved(); } update(true); };
  const sb = ui.slider({ parent: grid3, label: "fusion scale $b$", min: 0.01, max: 3, log: true, sig: 3, value: p.b, fmt: x => ui.fmt(x, 3), onInput: x => { p.b = x; moved(); } });
  const sa = ui.slider({ parent: grid3, label: "stiffness $\\alpha$", min: 0.01, max: 10, log: true, sig: 2, value: p.alpha, fmt: x => ui.fmt(x, 2), onInput: x => { p.alpha = x; moved(); } });
  const sd = ui.slider({ parent: grid3, label: "uncoupled distance $D_0^2$", min: 0.25, max: 4, log: true, sig: 2, value: p.D02, fmt: x => ui.fmt(x, 2), onInput: x => { p.D02 = x; moved(); } });
  const sync = () => { sb.set(p.b); sa.set(p.alpha); sd.set(p.D02); };

  const row = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 300px), 1fr))", gap: "14px 28px", marginTop: "8px" } });
  slot.append(row);
  const panel = (title) => { const b = ui.h("div"); row.append(ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", html: title }), b)); return b; };
  const boxA = panel("Drift $dw/dt$ against $w$"), boxB = panel("Fixed points $w^\\ast$ against $b$");
  ui.renderMath(row);
  const legB = ui.legend({ parent: row, items: [] });
  legB.el.style.gridColumn = "1 / -1";
  const readout = ui.h("div", { class: "u-help", style: { marginTop: "8px", fontSize: "12.5px", lineHeight: "1.55" } });
  slot.append(readout);

  const chartA = svgPlot(boxA, { height: 240, xlabel: "conductance w", ylabel: "dw/dt", xrange: [0, 1], yrange: [-1, 1] });
  const chartB = svgPlot(boxB, { height: 240, xlabel: "fusion scale b", ylabel: "w*", xrange: [0, 1], yrange: [0, 1] });

  let engine = { token: 0, res: null };
  let timer = 0;

  function update(urlToo) {
    const fps = M.pairFixedPoints(p), win = M.pairWindow(p);
    drawA(fps); drawB(fps, win);
    describe(fps, win);
    if (urlToo) ctx.setParams(tour ? { ptour: tour.id, pb: null, pa: null, pd: null } : { ptour: null, pb: p.b, pa: p.alpha, pd: p.D02 });
    clearTimeout(timer); engine.token++; engine.res = null; renderEngine();
    timer = setTimeout(runEngine, 120);
  }

  function drawA(fps) {
    const wTop = Math.max(...fps.map(f => f.w), p.b / p.alpha, 0.2);
    const wmax = 1.2 * wTop, n = 300, xs = [], ys = [];
    for (let k = 0; k <= n; k++) { const w = wmax * k / n; xs.push(w); ys.push(M.pairDrift(w, p)); }
    let lo = Math.min(0, ...ys), hi = Math.max(0, ...ys); const pad = 0.08 * ((hi - lo) || 1); lo -= pad; hi += pad;
    chartA.update({ xrange: [0, wmax], yrange: [lo, hi], render: ({ plot, svg, X, Y, m, W, H }) => {
      let d = ""; xs.forEach((x, k) => { d += (k ? "L" : "M") + X(x).toFixed(1) + "," + Y(ys[k]).toFixed(1); });
      svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: "#b5b4ad" }, plot);
      svgEl("path", { d, fill: "none", stroke: P.fg, "stroke-width": 2 }, plot);
      // flow arrows between consecutive fixed points
      const cuts = [0, ...fps.map(f => f.w).filter(w => w > 0), wmax];
      for (let k = 0; k + 1 < cuts.length; k++) {
        const a = cuts[k], b = cuts[k + 1]; if (X(b) - X(a) < 22) continue;
        const mid = (a + b) / 2, dir = Math.sign(M.pairDrift(mid, p)); if (!dir) continue;
        const x = X(mid), y = Y(0);
        svgEl("path", { d: `M${x - 6 * dir},${y - 5}L${x + 6 * dir},${y}L${x - 6 * dir},${y + 5}Z`, fill: P.mute2 }, plot);
      }
      for (const f of fps) {
        const c = svgEl("circle", { cx: X(f.w), cy: Y(0), r: 5.5, fill: f.stable ? P.fg : P.surface, stroke: P.fg, "stroke-width": 1.8 }, plot);
        svgEl("title", {}, c).textContent = `${f.stable ? "stable" : "unstable"} fixed point w = ${ui.fmt(f.w, 4)}`;
      }
      if (engine.res) engine.res.forEach((r, k) => {
        if (!r.done) return;
        const x = X(r.w), y = Y(0) - 13 - 9 * k;
        const g = svgEl("path", { d: `M${x},${y - 5}L${x + 5},${y}L${x},${y + 5}L${x - 5},${y}Z`, fill: k ? P.cat[1] : P.cat[0], stroke: P.surface, "stroke-width": 1 }, plot);
        svgEl("title", {}, g).textContent = `engine from w = ${ui.fmt(r.w0, 3)}: ends at w = ${ui.fmt(r.w, 6)}`;
      });
    } });
  }

  function drawB(fps, win) {
    const c = p.D02 / 2, bmax = Math.max(1.6 * c, 1.25 * p.b, win.nonempty ? 1.3 * win.bsn : 0);
    const br = M.pairBranches(p, bmax);
    const wmaxB = Math.max(...br.stable.filter(([b]) => b <= bmax).map(([, w]) => w), ...fps.map(f => f.w), 0.1) * 1.1;
    const showLoop = loop && Math.abs(p.alpha - loop.alpha) < 1e-9 && Math.abs(p.D02 - loop.D02) < 1e-9;
    chartB.update({ xrange: [0, bmax], yrange: [0, wmaxB], render: ({ plot, X, Y, m, H }) => {
      if (win.nonempty) svgEl("rect", { x: X(win.bsn), y: m.t, width: Math.max(0, X(win.bon) - X(win.bsn)), height: H - m.b - m.t, fill: P.cat[0], opacity: 0.1 }, plot);
      svgEl("line", { x1: X(0), x2: X(Math.min(c, bmax)), y1: Y(0) - 1, y2: Y(0) - 1, stroke: P.fg, "stroke-width": 2.2 }, plot);
      const path = pts => pts.map(([b, w], k) => (k ? "L" : "M") + X(b).toFixed(1) + "," + Y(w).toFixed(1)).join("");
      if (br.unstable.length) svgEl("path", { d: path(br.unstable), fill: "none", stroke: P.fg, "stroke-width": 1.4, "stroke-dasharray": "4 3" }, plot);
      svgEl("path", { d: path(br.stable), fill: "none", stroke: P.fg, "stroke-width": 2.2 }, plot);
      if (showLoop) {
        loop.b.forEach((b, k) => {
          svgEl("circle", { cx: X(b), cy: Y(loop.w_up[k]), r: 3, fill: P.cat[2], opacity: 0.9 }, plot);
          svgEl("circle", { cx: X(b), cy: Y(loop.w_down[k]), r: 2.2, fill: P.cat[0], opacity: 0.9 }, plot);
        });
      }
      svgEl("line", { x1: X(p.b), x2: X(p.b), y1: m.t, y2: H - m.b, stroke: P.accent, "stroke-width": 1.25, opacity: 0.85 }, plot);
      for (const f of fps) svgEl("circle", { cx: X(p.b), cy: Y(f.w), r: 4.5, fill: f.stable ? P.fg : P.surface, stroke: P.fg, "stroke-width": 1.6 }, plot);
    } });
    const items = [
      { label: "stable", color: P.fg, shape: "line" }, { label: "unstable", color: P.fg, shape: "dash" },
      { label: "bistable window $b_{sn} &lt; b &lt; D_0^2/2$", color: "rgba(42,120,214,0.25)", shape: "band" },
      { label: "current $b$", color: P.accent, shape: "line" },
    ];
    if (showLoop) items.push({ label: "sweep, b up", color: P.cat[2], shape: "dot" }, { label: "sweep, b down", color: P.cat[0], shape: "dot" });
    legB.set(items);
  }

  function describe(fps, win) {
    const st = fps.filter(f => f.stable), un = fps.filter(f => !f.stable);
    const list = a => a.map(f => `w = ${ui.fmt(f.w, 4)}`).join(", ");
    const w = win.nonempty
      ? `Window: $b_{sn} = ${ui.fmt(win.bsn, 4)}$ to $D_0^2/2 = ${ui.fmt(win.bon, 4)}$; the current $b$ is ${p.b > win.bsn && p.b < win.bon ? "<b>inside</b>" : "outside"} it.`
      : `No bistable window: $D_0^2 = ${ui.fmt(p.D02, 3)} \\le \\alpha/2 = ${ui.fmt(p.alpha / 2, 3)}$.`;
    readout.dataset.analytic = `${w} Analytic fixed points: stable ${st.length ? list(st) : "none"}${un.length ? `; unstable ${list(un)}` : ""}.`;
    renderEngine();
  }
  function renderEngine() {
    let e = "";
    if (!engine.res) e = "Engine check: running…";
    else {
      const fps = M.pairFixedPoints(p).filter(f => f.stable);
      e = "Engine check: " + engine.res.map(r => {
        const near = fps.reduce((b, f) => Math.min(b, Math.abs(f.w - r.w)), Infinity);
        return r.done ? `from $w = ${ui.fmt(r.w0, 3)}$ it ends at $w = ${ui.fmt(r.w, 6)}$ (distance to the nearest analytic stable point ${ui.fmt(near, 2)}, residual ${ui.fmt(r.r, 2)}, ${ui.fmtInt(r.steps)} steps)` : `from $w = ${ui.fmt(r.w0, 3)}$: not settled after ${ui.fmtInt(r.steps)} steps`;
      }).join("; ") + ".";
    }
    readout.innerHTML = `${readout.dataset.analytic || ""}<br>${e}`;
    ui.renderMath(readout);
  }

  // the engine runs two pair simulations in slices of ~12 ms (cancelled when the parameters change)
  function runEngine() {
    const token = ++engine.token, q = { ...p };
    const sims = [0, q.b / q.alpha].map(w0 => ({ w0, sim: M.makePairSim(q, w0), done: false }));
    const slice = () => {
      if (token !== engine.token) return;
      let all = true;
      for (const s of sims) if (!s.done) { s.done = M.advancePair(s.sim, 6, 1e-10, 300000); all = all && s.done; }
      if (!all) { setTimeout(slice, 0); return; }
      engine.res = sims.map(s => ({ w0: s.w0, w: s.sim.net.W[1], r: s.sim.residual, steps: s.sim.steps, done: s.sim.lastChange < 1e-10 }));
      drawA(M.pairFixedPoints(p)); renderEngine();
    };
    slice();
  }

  update(false);
  return {
    apply(params) {
      const t = PAIR_TOURS.find(x => x.id === params.get("ptour"));
      if (t) { tour = t; chips.set(t.id); p = { ...t.params }; }
      else if (params.has("pb") || params.has("pa") || params.has("pd")) { tour = null; chips.clear(); p = { b: +params.get("pb") || p.b, alpha: +params.get("pa") || p.alpha, D02: +params.get("pd") || p.D02 }; clamp(p); }
      else return;
      sync(); update(false);
    },
  };
}

function clamp(p) {
  p.b = Math.min(3, Math.max(0.01, p.b)); p.alpha = Math.min(10, Math.max(0.01, p.alpha)); p.D02 = Math.min(4, Math.max(0.25, p.D02));
}
