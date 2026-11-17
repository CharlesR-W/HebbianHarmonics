// Chapter "mnist" (Real data): the Hebbian resistor network trained on kernel-lifted MNIST. Everything is
// precomputed (data/mnist, format in its README.md; method in docs/mnist.md). One selected run drives the
// training explorer, the parts, the graph, the allocation and the readout figures.
import * as ui from "./ui.js";
import { lineChart } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as D from "./ch_mnist_data.js";
import { runFigure } from "./ch_mnist_run.js";
import { partsFigure, graphFigure, allocFigure } from "./ch_mnist_parts.js";
import { readoutFigure, samplesFigure } from "./ch_mnist_ro.js";

const DEFAULT = { lift: "pixels", init: "generic", theta: 0.75 };

export async function init(panel, ctx) {
  const index = await D.json("index.json");
  const S = D.summarise(index);
  D.fillNumbers(ctx.article, S);

  // ---------------------------------------------------------------- the selected run (shared state)
  const featured = index.runs.filter(r => r.featured);
  const state = { run: null, listeners: [] };
  const find = (lift, init, theta) => {
    let c = featured.filter(r => r.lift === lift);
    if (!c.length) c = featured.filter(r => r.lift === DEFAULT.lift);
    const ci = c.filter(r => D.initKey(r) === init);
    if (ci.length) c = ci;
    return c.reduce((a, b) => (Math.abs(b.theta - theta) < Math.abs(a.theta - theta) ? b : a));
  };
  state.find = find;
  state.featured = featured;
  state.select = (run, { push = false, scroll = false } = {}) => {
    if (!run || run === state.run) return;
    state.run = run;
    ctx.setParams({ lift: run.lift, init: D.initKey(run), theta: run.theta }, { push });
    state.listeners.forEach(f => f(run));
    if (scroll) document.getElementById("fig-mnist-run")?.scrollIntoView({ behavior: ui.reducedMotion() ? "auto" : "smooth", block: "start" });
  };
  state.onRun = f => state.listeners.push(f);
  state.run = find(ctx.param("lift", DEFAULT.lift), ctx.param("init", DEFAULT.init), ctx.param("theta", DEFAULT.theta));

  filmImage(ctx.slot("film"), index);
  sweepFigure(ctx.slot("sweep"), index, state, ctx);
  runFigure(ctx.slot("run"), ctx.slot("run-cap"), index, state, ctx);
  partsFigure(ctx.slot("parts"), ctx.slot("parts-cap"), index, state, ctx);
  graphFigure(ctx.slot("graph"), ctx.slot("graph-cap"), index, state, ctx);
  allocFigure(ctx.slot("alloc"), ctx.slot("alloc-cap"), index, state, ctx);
  readoutFigure(ctx.slot("readout"), ctx.slot("readout-cap"), index, state, ctx);
  samplesFigure(ctx.slot("samples"), ctx.slot("samples-cap"), index, state, ctx);

  ctx.onParams(p => {
    const run = find(p.get("lift") || state.run.lift, p.get("init") || D.initKey(state.run), p.has("theta") ? +p.get("theta") : state.run.theta);
    if (run !== state.run) { state.run = run; state.listeners.forEach(f => f(run)); }
  });
}

// ------------------------------------------------------------------ the exported filmstrip of the headline run
function filmImage(host, index) {
  const img = ui.h("img", { src: D.BASE + index.filmstrip.png, alt: "H Sigma_g H^T of the headline run at six training steps, from flat to a block diagonal of small clusters plus a large dead pool", loading: "lazy",
    style: { display: "block", width: "100%", minWidth: "680px", height: "auto" } });
  host.replaceChildren(ui.h("div", { style: { overflowX: "auto" } }, img));
}

// ------------------------------------------------------------------ the theta sweep over all 114 runs
const FAMILIES = [{ value: "pix", label: "pixel lift" }, { value: "rff", label: "RFF lifts" }];
const famOf = r => (r.lift === "pixels" ? "pix" : "rff");

function sweepGroups(fam) {
  const base = r => r.mode === "sgd" && r.seed === 0 && D.omega(r) === 0.5;
  const line = (name, color, dash, f) => ({ name, color, dash, f: r => base(r) && f(r), line: true });
  const pts = (name, color, f) => ({ name, color, f, line: false });
  if (fam === "pix") return [
    line("N = 2048, heavy-tailed", catColor(0), false, r => r.N === 2048 && D.initKey(r) === "heavy"),
    line("N = 2048, generic", catColor(0), true, r => r.N === 2048 && D.initKey(r) === "generic"),
    line("N = 512, heavy-tailed", catColor(1), false, r => r.N === 512 && D.initKey(r) === "heavy"),
    line("N = 512, generic", catColor(1), true, r => r.N === 512 && D.initKey(r) === "generic"),
    pts("mean-field (heavy, both N)", catColor(2), r => r.mode === "mf"),
    pts("seed 1 (N = 2048)", catColor(3), r => r.seed === 1),
    pts("ω = 0.05 (N = 2048, heavy)", catColor(4), r => D.omega(r) === 0.05),
    pts("ω = 5 (N = 2048, heavy)", catColor(5), r => D.omega(r) === 5),
  ];
  const L = (lift, N) => r => r.lift === lift && r.N === N;
  return [
    line("D = 1024, N = 2048, heavy", catColor(0), false, r => L("rff1024", 2048)(r) && D.initKey(r) === "heavy"),
    line("D = 1024, N = 2048, generic", catColor(0), true, r => L("rff1024", 2048)(r) && D.initKey(r) === "generic"),
    line("D = 256, N = 2048, heavy", catColor(1), false, r => L("rff256", 2048)(r) && D.initKey(r) === "heavy"),
    line("D = 256, N = 2048, generic", catColor(1), true, r => L("rff256", 2048)(r) && D.initKey(r) === "generic"),
    line("D = 256, N = 512, heavy", catColor(2), false, r => L("rff256", 512)(r) && D.initKey(r) === "heavy"),
    line("D = 256, N = 512, generic", catColor(2), true, r => L("rff256", 512)(r) && D.initKey(r) === "generic"),
    line("D = 64, N = 512, heavy", catColor(3), false, r => L("rff64", 512)(r) && D.initKey(r) === "heavy"),
    line("D = 64, N = 512, generic", catColor(3), true, r => L("rff64", 512)(r) && D.initKey(r) === "generic"),
    pts("bandwidth ×0.5 (D = 256)", catColor(4), r => r.lift === "rff256bw0.5"),
    pts("bandwidth ×2 (D = 256)", catColor(5), r => r.lift === "rff256bw2"),
    pts("mean-field (D = 256)", catColor(6), r => r.mode === "mf"),
    pts("seed 1 (D = 1024)", catColor(7), r => r.seed === 1),
  ];
}

const METRICS = [
  { key: "live", label: "live clusters", y: r => r.final.n_live_clusters, fmt: v => String(Math.round(v)) },
  { key: "dead", label: "dead fraction", y: r => r.final.n_dead / r.N, fmt: v => ui.fmtPct(v, 0), yrange: [0, 1] },
  { key: "ro", label: "readout: contrast-subspace cosine", y: r => (r.final.n_live_clusters > 0 ? r.readout.ls_both.contrast_cos : NaN), fmt: v => ui.fmtFixed(v, 2), yrange: [0, 1] },
];

function sweepFigure(host, index, state, ctx) {
  let fam = ["pix", "rff"].includes(ctx.param("sw", "")) ? ctx.param("sw", "") : famOf(state.run);
  host.replaceChildren();
  const seg = ui.segmented({ label: "lift", options: FAMILIES, value: fam, onChange: v => { fam = v; ctx.setParams({ sw: v }); draw(); } });
  const bar = ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "8px 24px", alignItems: "center", marginBottom: "6px" } }, seg.el);
  const grid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "6px 18px" } });
  const lg = ui.legend({ items: [] });
  lg.el.classList.add("chart-legend");
  host.append(bar, grid, lg.el);
  const boxes = METRICS.map(m => { const b = ui.h("div", { style: { minWidth: 0 } }); grid.append(ui.h("div", { style: { minWidth: 0 } }, ui.h("div", { class: "chart-title", text: m.label }), b)); return b; });
  let charts = [], featPts = [];

  function draw() {
    const runs = index.runs.filter(r => famOf(r) === fam);
    const groups = sweepGroups(fam);
    const xr = fam === "pix" ? [0.25, 1.15] : [0.03, 0.32];
    featPts = runs.filter(r => r.featured);
    const maxLive = Math.max(...runs.map(r => r.final.n_live_clusters));
    charts.forEach(c => c.destroy());
    charts = METRICS.map((m, mi) => {
      const series = [{ name: "full export (click to load)", x: featPts.map(r => r.theta), y: featPts.map(m.y), points: true, r: 6.5, color: P.fg }];
      for (const g of groups) {
        const rs = runs.filter(g.f).sort((a, b) => a.theta - b.theta);
        if (!rs.length) continue;
        series.push(g.line ? { name: g.name, x: rs.map(r => r.theta), y: rs.map(m.y), color: g.color, dash: g.dash, dots: true, r: 2.5, width: 1.8 }
          : { name: g.name, x: rs.map(r => r.theta), y: rs.map(m.y), color: g.color, points: true, r: 3.5 });
      }
      const refs = [];
      if (m.key === "ro" && fam === "pix") {
        const b = index.runs.find(r => r.lift === "pixels").baselines;
        refs.push({ y: b.diag.contrast_cos, label: `diag Σ ${ui.fmtFixed(b.diag.contrast_cos, 2)}` }, { y: b.identity.contrast_cos, label: `identity ${ui.fmtFixed(b.identity.contrast_cos, 2)}` });
      }
      const c = lineChart(boxes[mi], { series, xrange: xr, yrange: m.yrange || [0, maxLive + 1], height: 210, xlabel: "θ", xname: "θ", legend: false,
        yfmt: m.fmt, xfmt: v => ui.fmt(v, 2), refs });
      return c;
    });
    lg.set([{ label: "full export: click to load", color: P.fg, shape: "dot" },
      ...groups.map(g => ({ label: g.name, color: g.color, shape: g.line ? (g.dash ? "dash" : "line") : "dot" }))]);
    cursor();
  }
  function cursor() { charts.forEach(c => c.setCursor(famOf(state.run) === fam ? state.run.theta : null)); }

  // click a featured point: find it in screen space (lineChart margins: l = 42, r = 12, t = 8, b = 38 with an x label)
  boxes.forEach((b, mi) => b.addEventListener("click", ev => {
    const svg = b.querySelector("svg"); if (!svg) return;
    const rc = svg.getBoundingClientRect(), W = rc.width, Hh = rc.height, m = { l: 42, r: 12, t: 8, b: 38 };
    const xr = fam === "pix" ? [0.25, 1.15] : [0.03, 0.32];
    const yr = METRICS[mi].yrange || [0, Math.max(...index.runs.filter(r => famOf(r) === fam).map(r => r.final.n_live_clusters)) + 1];
    const X = v => m.l + (v - xr[0]) / (xr[1] - xr[0]) * (W - m.l - m.r), Y = v => Hh - m.b - (v - yr[0]) / (yr[1] - yr[0]) * (Hh - m.t - m.b);
    let best = null, bd = 16 * 16;
    for (const r of featPts) {
      const y = METRICS[mi].y(r); if (!Number.isFinite(y)) continue;
      const dx = X(r.theta) - (ev.clientX - rc.left), dy = Y(y) - (ev.clientY - rc.top), d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = r; }
    }
    if (best) state.select(best, { scroll: true });
  }));
  boxes.forEach(b => { b.style.cursor = "pointer"; });
  draw();
  state.onRun(run => { if (famOf(run) !== fam) { fam = famOf(run); seg.set(fam); draw(); } else cursor(); });
  ctx.onParams(p => { const v = p.get("sw"); if (v && v !== fam && ["pix", "rff"].includes(v)) { fam = v; seg.set(v); draw(); } });
}
