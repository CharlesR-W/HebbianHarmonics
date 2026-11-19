// Chapter "nonlinear": the precomputed big runs of the distance rule (site/data/big/, hebb/big/dist_runs.py and
// dist_export.py; findings B2, B3). Matched protocol: linear, rectifier and homogeneous-rectifier arms share seed,
// attachments, W = 0, banks and b.
//   films        one setting picker; the linear film and the rectifier film side by side (filmplayer.js, no pickers)
//   films-table  the three arms at that b, mean ± sd over seeds (runs[]), plus the matched-density b (matched[])
//   readout-big  the distance readout of the rectifier arms at every regime b (runs[].distance_readout)
//   div          per-neuron nonlinearity and within-component diversity against b (matched[sigma].grid)
// Numbers in the prose (data-nl="…") are read from the same files.
import * as ui from "./ui.js";
import { lineChart } from "./charts.js";
import { catColor, P } from "./palette.js";
import { filmPlayer } from "./filmplayer.js";

const SETTINGS = [
  { id: "a-frag", fam: "A", sigma: "ind3", regime: "fragmented", alpha: 1, lin: "A_dist_ind3_lin_fragmented", rect: "A_dist_ind3_rect_fragmented", label: "B = 3, fragmented (b = 0.0003)" },
  { id: "a-perc", fam: "A", sigma: "ind3", regime: "percolation", alpha: 1, lin: "A_dist_ind3_lin_percolation", rect: "A_dist_ind3_rect_percolation", label: "B = 3, percolation (b = 0.0017)" },
  { id: "a-fused", fam: "A", sigma: "ind3", regime: "fused", alpha: 1, lin: "A_dist_ind3_lin_fused", rect: "A_dist_ind3_rect_fused", label: "B = 3, fused (b = 0.01)" },
  { id: "a-perc01", fam: "A", sigma: "ind3", regime: "percolation", alpha: 0.1, lin: "A_dist_ind3_lin_percolation_a0.1", rect: "A_dist_ind3_rect_percolation_a0.1", label: "B = 3, percolation, α = 0.1" },
  { id: "b-blocks", fam: "B", sigma: "blocks", regime: "percolation", alpha: 1, lin: "B_dist_blocks_lin_percolation", rect: "B_dist_blocks_rect_percolation", label: "B = 12 blocks, percolation (b = 0.018)" },
  { id: "b-power", fam: "B", sigma: "power", regime: "percolation", alpha: 1, lin: "B_dist_power_lin_percolation", rect: "B_dist_power_rect_percolation", label: "B = 12 power law, percolation (b = 0.045)" },
];
const ARMS = [["linear", "linear"], ["rect", "rectifier, σt = 0.7"], ["rect_homog", "rectifier, σt = 0 (control)"]];
const SHORT = { ind3: "B = 3", blocks: "blocks", power: "power law" };
const SIG = { ind3: "B = 3 independent", blocks: "B = 12 blocks", power: "B = 12 power law" };
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const sd = a => { const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))); };
const pct = (x, d = 1) => `${ui.fmtFixed(100 * x, d)}%`;

export async function initBig(ctx) {
  const get = f => fetch(`data/big/${f}`).then(r => { if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`); return r.json(); });
  const [A, Bd] = await Promise.all([get("dist_A.json"), get("dist_B.json")]);
  const fam = { A, B: Bd };
  const regimeB = (s) => fam[s.fam].meta.regimes[s.sigma][s.regime];
  films(ctx, fam, regimeB);
  readoutTable(ctx, fam);
  divFigure(ctx, fam);
  numbers(ctx, fam);
}

// ---------------------------------------------------------------- films and the three-arm table
function films(ctx, fam, regimeB) {
  const slot = ctx.slot("films"); if (!slot) return;
  slot.replaceChildren();
  let cur = SETTINGS.find(s => s.id === ctx.param("set", "a-perc")) || SETTINGS[1];
  const sel = ui.select({ label: "setting", options: SETTINGS.map(s => ({ value: s.id, label: s.label })), value: cur.id,
    onChange: id => { cur = SETTINGS.find(s => s.id === id); show(); ctx.setParams({ set: id === "a-perc" ? null : id }); } });
  sel.el.style.maxWidth = "420px";
  const colL = ui.h("div", { style: { minWidth: "0" } }), colR = ui.h("div", { style: { minWidth: "0" } });
  const hL = ui.h("div", { class: "chart-title", style: { margin: "10px 0 6px" }, text: "Linear neurons" });
  const hR = ui.h("div", { class: "chart-title", style: { margin: "10px 0 6px" }, text: "Rectifier (k_lo = 3, σt = 0.7)" });
  const bodyL = ui.h("div"), bodyR = ui.h("div");
  colL.append(hL, bodyL); colR.append(hR, bodyR);
  const two = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(460px, 100%), 1fr))", gap: "10px 32px", alignItems: "start" } }, colL, colR);
  const table = ui.h("div", { style: { marginTop: "14px" } });
  slot.append(sel.el, two, table);
  const opts = { picker: false, countLabels: true, inputEdges: 300, height: 330 };
  const pL = filmPlayer(bodyL, { ...opts, id: cur.lin });
  const pR = filmPlayer(bodyR, { ...opts, id: cur.rect });
  function show() {
    pL.load(cur.lin); pR.load(cur.rect);
    const b = cur.alpha === 1 ? regimeB(cur) : regimeB(cur);
    const F = fam[cur.fam];
    const rows = ARMS.map(([arm, lab]) => {
      const rs = F.runs.filter(r => r.sigma === cur.sigma && r.arm === arm && Math.abs(r.b - b) < 1e-9 && r.alpha === cur.alpha);
      if (!rs.length) return null;
      const ms = (f, fm) => { const v = rs.map(f); return rs.length > 1 ? `${fm(mean(v))} ± ${fm(sd(v))}` : fm(v[0]); };
      return ui.h("tr", {},
        ui.h("td", { text: lab }), ui.h("td", { class: "num", text: String(rs.length) }),
        ui.h("td", { class: "num", text: ms(r => r.K, x => ui.fmtFixed(x, 0)) }),
        ui.h("td", { class: "num", text: ms(r => r.largest_frac, x => ui.fmtFixed(x, 2)) }),
        ui.h("td", { class: "num", text: ms(r => r.edge_density, x => ui.fmt(x, 2)) }),
        ui.h("td", { class: "num", text: ms(r => r.span_largest_over_range, x => ui.fmtFixed(x, 1)) }),
        ui.h("td", { class: "num", text: ms(r => r.span_over_range, x => ui.fmtFixed(x, 1)) }),
        ui.h("td", { class: "num", text: (() => { const v = rs.map(r => r.diversity.proto_err_w_compact).filter(x => x != null); return v.length ? (v.length > 1 ? `${ui.fmt(mean(v), 2)} ± ${ui.fmt(sd(v), 2)}` : ui.fmt(v[0], 2)) + (v.length < rs.length ? ` (${v.length} seed${v.length > 1 ? "s" : ""})` : "") : "–"; })() }),
        ui.h("td", { class: "num", text: ms(r => r.diversity.affine_err_mean, x => (x < 1e-6 ? "0" : ui.fmtFixed(x, 3))) }));
    }).filter(Boolean);
    // matched density, for this run itself (seed 0): the linear b with the same edge density, log-interpolated on the
    // linear arm's grid at this run's density (dist_export.py, matched[sigma].regime_match)
    let match = "";
    const m = cur.alpha === 1 && F.matched?.[cur.sigma]?.regime_match?.find(x => x.arm === "rect" && x.regime === cur.regime);
    if (m && m.linear_b_same_density) match = `Matched by edge density instead: this rectifier run (seed 0) has density ${ui.fmt(m.edge_density, 2)}; the linear arm reaches it at $b \\approx ${ui.fmt(m.linear_b_same_density, 2)}$ (log-interpolated on its grid), so here the rectifier needs ${m.ratio >= 1 ? `${ui.fmtFixed(m.ratio, 1)} times the linear arm's $b$` : `a smaller $b$ (${ui.fmtFixed(m.ratio, 2)} times)`}.`;
    else if (m) match = `Matched by edge density: this rectifier run's density (${ui.fmt(m.edge_density, 2)}, seed 0) is outside the linear arm's grid.`;
    const th = (t, title) => ui.h("th", { text: t, title });
    table.replaceChildren(
      ui.h("div", { class: "table-scroll" }, ui.h("table", { class: "data" },
        ui.h("thead", {}, ui.h("tr", {}, th("arm, same b = " + ui.fmt(b, 2) + (cur.alpha !== 1 ? `, α = ${cur.alpha}` : "")), th("seeds"), th("K"), th("largest / N"), th("edge density"),
          th("span, largest comp.", "largest pairwise D in the largest component, in edge ranges √(2b); above 2: a chain"),
          th("span, max", "largest pairwise D in any component, in edge ranges √(2b)"),
          th("diversity, compact comps.", "held-out relative error of member ≈ a + c·prototype, size-weighted over components of 2 or more spanning at most 2 edge ranges; – if there are none"),
          th("per-neuron nonlinearity", "held-out relative error of an affine fit of each response from the input"))),
        ui.h("tbody", {}, ...rows))),
      ui.h("p", { class: "u-help", html: match }));
    ui.renderMath(table);
  }
  show();
}

// ---------------------------------------------------------------- distance readout, big runs
function readoutTable(ctx, fam) {
  const el = ctx.slot("readout-big"); if (!el) return;
  const rows = [];
  for (const F of [fam.A, fam.B]) for (const [sigma, rg] of Object.entries(F.meta.regimes)) for (const [regime, b] of Object.entries(rg)) {
    const rs = F.runs.filter(r => r.sigma === sigma && r.arm === "rect" && Math.abs(r.b - b) < 1e-9 && r.alpha === 1);
    if (!rs.length) continue;
    const dr = rs.map(r => r.distance_readout);
    const mOr = (k, f) => { const v = dr.map(x => x[k]).filter(x => x != null && Number.isFinite(x)); return v.length ? f(mean(v)) : "–"; };
    const rngOr = (k, f) => { const v = dr.map(x => x[k]).filter(x => x != null && Number.isFinite(x)); if (!v.length) return "–"; const lo = Math.min(...v), hi = Math.max(...v); return f(lo) === f(hi) ? f(lo) : `${f(lo)}–${f(hi)}`; };
    rows.push(ui.h("tr", {},
      ui.h("td", { text: `${SHORT[sigma]}, ${regime}` }), ui.h("td", { class: "num", text: ui.fmt(b, 2) }),
      ui.h("td", { class: "num", text: mOr("frac_exact", x => pct(x, x < 0.01 ? 2 : 1)) }),
      ui.h("td", { class: "num", text: mOr("frac_upper_bounded", x => pct(x, 1)) }),
      ui.h("td", { class: "num", text: mOr("frac_lower_only", x => pct(x, 1)) }),
      ui.h("td", { class: "num", text: ui.fmt(Math.max(...dr.map(x => x.edge_abs_err_train_max)), 2) }),
      ui.h("td", { class: "num", text: mOr("edge_rel_err_eval_median", x => pct(x, 1)) }),
      ui.h("td", { class: "num", text: (() => { const v = dr.map(x => x.dead_lower_bound_holds_eval).filter(x => x != null); return v.length ? pct(Math.min(...v), 2) : "–"; })() }),
      ui.h("td", { class: "num", text: rngOr("sp_over_eval_median", x => ui.fmtFixed(x, 2)) }),
      ui.h("td", { class: "num", text: rngOr("mds_completed_vs_measured_eval_disparity", x => ui.fmtFixed(x, 3)) })));
  }
  const th = (t, title) => ui.h("th", { text: t, title });
  el.replaceChildren(ui.h("div", { class: "table-scroll" }, ui.h("table", { class: "data" },
    ui.h("thead", {}, ui.h("tr", {}, th("inputs, regime", "regime of the linear arm at this b"), th("b"), th("exact", "pairs whose D the weights give exactly (live edges), as a fraction of all pairs"),
      th("two-sided", "same component, no edge: √(2b) ≤ D ≤ shortest path"), th("lower only", "different components: D ≥ √(2b)"),
      th("train error", "live edges: max |√(2(b − αw)) − D| on the training bank, worst seed"), th("held-out error", "live edges: median relative error against the held-out D"),
      th("bound holds", "fraction of cross-component pairs with held-out D ≥ √(2b), worst seed"), th("path / D", "median ratio of shortest path to held-out D inside components; range over seeds"),
      th("MDS disparity", "range over seeds. Procrustes disparity of 3-D classical MDS of the completed distances vs of the measured held-out distances, largest component; – where there is no component to compare"))),
    ui.h("tbody", {}, ...rows))),
    ui.h("p", { class: "u-help", html: "Columns (hover a header for its definition): fraction of all pairs known exactly (live edges), bounded on both sides, bounded below only; error of the live entries against the training bank (absolute) and the held-out bank (relative, median); share of cross-component pairs obeying the lower bound; shortest path over held-out $D$; MDS disparity of completed against measured distances. Rectifier arm ($k_{lo} = 3$, $\\sigma_t = 0.7$), $\\alpha = 1$, mean over seeds unless stated (family A: 3 seeds, $N = 300$; family B: 2 seeds, $N = 400$). Held-out: the evaluation bank of 4096 inputs. From <code>dist_A.json</code>, <code>dist_B.json</code> (<code>runs[].distance_readout</code>)." }));
  ui.renderMath(el);
}

// ---------------------------------------------------------------- diversity against b
function divFigure(ctx, fam) {
  const slot = ctx.slot("div"); if (!slot) return;
  slot.replaceChildren();
  let sigma = ctx.param("dsig", "blocks");
  if (!["ind3", "blocks", "power"].includes(sigma)) sigma = "blocks";
  const seg = ui.segmented({ label: "inputs", options: [{ value: "ind3", label: "B = 3" }, { value: "blocks", label: "B = 12 blocks" }, { value: "power", label: "B = 12 power law" }], value: sigma,
    onChange: v => { sigma = v; draw(); ctx.setParams({ dsig: v === "blocks" ? null : v }); } });
  const grid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(380px, 100%), 1fr))", gap: "18px 28px", marginTop: "10px" } });
  const box = (t, sub) => { const b = ui.h("div"); const e = ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", html: t }), ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" }, html: sub }), b); grid.append(e); ui.renderMath(e); return b; };
  slot.append(seg.el, grid);
  const bA = box("(a) Within-component diversity", "relative error of member ≈ a + c·prototype, compact components only");
  const bB = box("(b) Per-neuron nonlinearity", "relative error of an affine fit from the input");
  const colors = { linear: catColor(0), rect: catColor(1), rect_homog: catColor(2) };
  const names = { linear: "linear", rect: "rectifier σt = 0.7", rect_homog: "rectifier σt = 0" };
  let cA = null, cB = null;
  function draw() {
    const F = sigma === "ind3" ? fam.A : fam.B, G = F.matched[sigma].grid;
    const x = G.linear.map(g => g.b);
    const sA = Object.keys(names).map(arm => ({ name: names[arm], color: colors[arm], dots: true, x: G[arm].map(g => g.b),
      y: G[arm].map(g => (g.proto_err_w_compact != null ? Math.max(g.proto_err_w_compact, 1e-6) : NaN)) }));
    const sB = Object.keys(names).map(arm => ({ name: names[arm], color: colors[arm], dots: true, x: G[arm].map(g => g.b), y: G[arm].map(g => Math.max(g.affine_err_mean, 1e-4)) }));
    const common = { xlog: true, height: 230, xlabel: "fusion scale b", xname: "b" };
    if (!cA) { cA = lineChart(bA, { ...common, series: sA, ylog: true, yrange: [1e-6, 1], ylabel: "relative error" }); cB = lineChart(bB, { ...common, series: sB, yrange: [0, 0.14], ylabel: "relative error" }); }
    else { cA.update({ series: sA }); cB.update({ series: sB }); }
    void x;
  }
  draw();
}

// ---------------------------------------------------------------- numbers in the prose
function numbers(ctx, fam) {
  const rect = [fam.A, fam.B].flatMap(F => F.runs.filter(r => r.arm === "rect" && r.alpha === 1 && r.distance_readout));
  const ev = rect.map(r => r.distance_readout.edge_rel_err_eval_median).filter(Number.isFinite);
  const lb = rect.map(r => r.distance_readout.dead_lower_bound_holds_eval).filter(x => x != null && Number.isFinite(x));
  const at = (F, sigma, regime, arm) => F.runs.filter(r => r.sigma === sigma && r.arm === arm && r.alpha === 1 && Math.abs(r.b - F.meta.regimes[sigma][regime]) < 1e-9);
  const mds = (sigma, regime) => mean(at(fam.B, sigma, regime, "rect").map(r => r.distance_readout.mds_completed_vs_measured_eval_disparity).filter(x => x != null));
  // diversity in the compact regimes of family B (fragmented and fused), seed means per setting
  // compact components only (span <= 2 edge ranges: diversity.proto_err_w_compact)
  const divRange = arm => { const v = []; for (const s of ["blocks", "power"]) for (const g of ["fragmented", "fused"]) v.push(mean(at(fam.B, s, g, arm).map(r => r.diversity.proto_err_w_compact))); return [Math.min(...v), Math.max(...v)]; };
  const fragA = Math.max(...["linear", "rect", "rect_homog"].flatMap(arm => at(fam.A, "ind3", "fragmented", arm).map(r => r.diversity.proto_err_w_compact)).filter(x => x != null));
  const [rlo, rhi] = divRange("rect"), [llo, lhi] = divRange("linear"), [hlo, hhi] = divRange("rect_homog");
  const aff = ["blocks", "power"].map(s => mean(at(fam.B, s, "fused", "rect").map(r => r.diversity.affine_err_mean)));
  // bounds rounded outward: upper bounds up, lower bounds down
  const up1 = x => { const e = Math.floor(Math.log10(x)); return ui.fmt(Math.ceil(x / 10 ** e - 1e-9) * 10 ** e, 1); };
  const pctDn = (x, d) => `${ui.fmtFixed(Math.floor(x * 100 * 10 ** d) / 10 ** d, d)}%`;
  const pctUp = (x, d) => `${ui.fmtFixed(Math.ceil(x * 100 * 10 ** d) / 10 ** d, d)}%`;
  const v = {
    nrect: String(rect.length),
    errtr: up1(Math.max(...rect.map(r => r.distance_readout.edge_abs_err_train_max))),
    evlo: pctDn(Math.min(...ev), 1), evhi: pctUp(Math.max(...ev), 1),
    lbmin: pctDn(Math.min(...lb), 2),
    divfragA: pctUp(fragA, 2),
    mdsfused: ui.fmtFixed(mds("blocks", "fused"), 3),
    mdsperc: `${ui.fmtFixed(mds("blocks", "percolation"), 2)} on the blocks, ${ui.fmtFixed(mds("power", "percolation"), 2)} on the power law`,
    divrlo: pct(rlo, 1), divrhi: pct(rhi, 1), divllo: pct(llo, 2), divlhi: pct(lhi, 2), divhlo: pct(hlo, 2), divhhi: pct(hhi, 1),
    afffused: `${ui.fmtFixed(aff[0], 3)} (blocks) and ${ui.fmtFixed(aff[1], 3)} (power law)`,
  };
  ctx.panel.querySelectorAll("[data-nl]").forEach(el => { const k = el.dataset.nl; if (v[k] != null) el.textContent = v[k]; });
}
