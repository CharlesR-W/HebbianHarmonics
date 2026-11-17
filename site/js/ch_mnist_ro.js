// Chapter "mnist": the weight-space readout (C_hat against Sigma_g, spectra, the score table) and the
// moment-based sample synthesis sheets.
import * as ui from "./ui.js";
import { heatmap, lineChart } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as D from "./ch_mnist_data.js";

const VARIANTS = [
  { value: "ls_both", label: "least squares", long: "masked least squares, both edge families" },
  { value: "ls_ib", label: "input synapses only", long: "least squares, input synapses only" },
  { value: "ls_ii", label: "lateral only", long: "least squares, lateral synapses only" },
  { value: "naive_ib", label: "pseudo-inverse", long: "the note's pseudo-inverse, input analogue sym(H⁺R_IB)" },
  { value: "init_ls_both", label: "untrained net", long: "least squares on the untrained network (equilibrium assumption violated)" },
];
const ROWS = [
  { group: "from the weights (θ and λ/η supplied)" },
  { key: "ls_both", src: "readout", label: "masked least squares, both families" },
  { key: "ls_ib", src: "readout", label: "least squares, input synapses only" },
  { key: "ls_ii", src: "readout", label: "least squares, lateral synapses only" },
  { key: "naive_ii", src: "readout", label: "the note's pseudo-inverse $H^+RH^{+\\top}$" },
  { key: "naive_ib", src: "readout", label: "pseudo-inverse, input analogue" },
  { key: "instant_ls_both", src: "readout", label: "least squares on the final instantaneous weights" },
  { group: "controls and references (need $\\Sigma_g$, or violate equilibrium)" },
  { key: "init_ls_both", src: "readout", label: "least squares on the untrained network" },
  { key: "identity", src: "baselines", label: "identity" },
  { key: "diag", src: "baselines", label: "diag $\\Sigma_g$" },
  { key: "rank_k_truncation", src: "baselines", label: "rank-$n_{\\text{clusters}}$ truncation of $\\Sigma_g$ (dead pool counted)" },
];
const COLS = [
  { key: "known_relfro", label: "rel. error", better: "lo", grp: 0 }, { key: "known_bw", label: "BW", better: "lo", grp: 0 },
  { key: "scale_cos", label: "cosine", better: "hi", grp: 1 }, { key: "scale_bw", label: "BW", better: "lo", grp: 1 },
  { key: "contrast_cos", label: "cosine", better: "hi", grp: 2 }, { key: "contrast_bw", label: "BW", better: "lo", grp: 2 },
  { key: "overlap_k10", label: "overlap top 10", better: "hi", grp: 2 }, { key: "overlap_kclust", label: "overlap top $n_{\\text{cl}}$", better: "hi", grp: 2 },
];

function permute(a, B, order) {
  const out = new Float32Array(B * B);
  for (let i = 0; i < B; i++) { const oi = order[i] * B; for (let j = 0; j < B; j++) out[i * B + j] = a[oi + order[j]]; }
  return out;
}

export function readoutFigure(host, capHost, index, state, ctx) {
  let token = 0, variant = VARIANTS.some(v => v.value === ctx.param("ro", "")) ? ctx.param("ro", "") : "ls_both";
  host.replaceChildren();
  const seg = ui.segmented({ label: "estimator", options: VARIANTS.map(v => ({ value: v.value, label: v.label, title: v.long })), value: variant,
    onChange: v => { variant = v; ctx.setParams({ ro: v }); load(state.run); } });
  const top = ui.h("div", { style: { marginBottom: "10px" } }, seg.el);
  const maps = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "12px 24px", alignItems: "start" } });
  const table = ui.h("div", { style: { marginTop: "16px" } }), foot = ui.h("div", { class: "u-help", style: { marginTop: "8px", fontSize: "12.5px", maxWidth: "860px" } });
  host.append(top, maps, table, foot);

  async function load(r) {
    const my = ++token;
    const L = index.lifts[r.lift], F = r.files;
    const chatFile = F["chat_" + variant];
    const [d, sig, chat] = await Promise.all([D.runData(r), L.files.sigma ? D.f32(L.files.sigma.file) : null, chatFile ? D.f32(chatFile.file) : null]);
    if (my !== token) return;
    const B = r.B, ro = d.readout, order = ro.order, ns = ro.n_supported;
    maps.replaceChildren();
    const head = html => ui.h("div", { class: "chart-title", html });
    if (sig && chat) {
      const groups = { rows: [{ start: 0, end: ns, color: P.fg, label: "parts" }, { start: ns, end: B, color: P.mute2, label: "rest" }] };
      const tip = name => (i, j, v) => ui.tipRows(name, [{ name: `inputs ${order[i]}, ${order[j]}`, value: ui.fmtFixed(v, 3) }]);
      const a = ui.h("div", { style: { minWidth: 0 } }), b = ui.h("div", { style: { minWidth: 0 } });
      maps.append(ui.h("div", { style: { minWidth: 0 } }, head("true $\\Sigma_g$"), a), ui.h("div", { style: { minWidth: 0 } }, head(`readout $\\hat C$: ${VARIANTS.find(v => v.value === variant).long}`), b));
      const common = { rows: B, cols: B, map: "div", range: [-1, 1], size: 230, groups, xlabel: "input (parts' inputs first)", ylabel: "input" };
      heatmap(a, { ...common, values: permute(sig, B, order), zlabel: "$\\Sigma_g$ (saturates beyond ±1)", tooltip: tip("Σg") });
      heatmap(b, { ...common, values: permute(chat, B, order), zlabel: "$\\hat C$ (same scale)", tooltip: tip("Ĉ") });
    } else {
      maps.append(ui.h("div", { class: "u-help", style: { alignSelf: "center", maxWidth: "420px", fontSize: "13px" }, html: `With $B = ${B}$ the $B \\times B$ matrices were not exported (only for $B \\le 256$), nor for this estimator on this run; the spectra and the scores below are complete.` }));
    }
    const sp = ro.spectra, spBox = ui.h("div", { style: { minWidth: 0 } });
    maps.append(ui.h("div", { style: { minWidth: 0 } }, head("spectra of $P\\hat CP$ and $P\\Sigma_gP$"), spBox));
    const norm = arr => { const m = arr[0] || 1; return arr.map(v => (v > 0 ? v / m : NaN)); };
    const xs = sp.sigma_contrast.map((_, i) => i + 1), est = sp[variant] || sp.ls_both;
    lineChart(spBox, { x: xs, xlog: true, ylog: true, height: 250, xlabel: "eigenvalue index", ylabel: "λᵢ / λ₁", xname: "index", yfmt: v => ui.fmt(v, 1), xfmt: v => ui.fmt(v, 2),
      yrange: [1e-5, 1.5], series: [{ name: "PΣgP", y: norm(sp.sigma_contrast), color: P.fg }, { name: `PĈP (${VARIANTS.find(v => v.value === variant).label})`, y: norm(est), color: catColor(0), dots: true, r: 2 }],
      refs: [{ x: r.measurement_rank.both.rank }] });
    ui.renderMath(maps);
    scoreTable(r);
    const s = r.support, m = r.measurement_rank, inf = r.readout_info.ls_both;
    foot.innerHTML = `<b>Where the weights measure.</b> On the ${s.n_supported} inputs covered by live parts, least squares reproduces $\\Sigma_g$ with cosine ${D.f2(s.ls_both?.cos_on_support)}; median part-energy error ${ui.fmtPct(s.ls_both?.part_energy_relerr_median, 1)}; ${ui.fmtPct(s.ls_both?.frac_C_mass_on_support_block, 0)} of $\\hat C$'s Frobenius mass sits on that block, which holds ${ui.fmtPct(s.sigma_contrast_mass_on_support_block, 0)} of the mass of $P\\Sigma_gP$. <b>How much they measure.</b> ${ui.fmtInt(inf.n_ii)} active lateral and ${ui.fmtInt(inf.n_ib)} active input equations; rank of the cluster-level measurement operator ${m.both.rank} (lateral ${m.ii.rank}, input ${m.ib.rank}) on the ${ui.fmtInt(m.dim_sym)}-dimensional space of symmetric $B \\times B$ matrices; condition number ${ui.fmt(m.both.cond, 2)}.`;
    ui.renderMath(foot);
    D.caption(capHost, `$\\Sigma_g$ and the readout $\\hat C$ (fitted with the true $\\theta$ and $\\lambda/\\eta$) in one order and on one colour range $[-1, 1]$; the spectra of $P\\hat CP$ and $P\\Sigma_gP$; one reconstruction scored three ways against $\\Sigma_g$, with the baselines scored identically.`,
      `Heatmaps: absolute units; inputs covered by live parts first, grouped by owning part (band "parts"), the rest ordered by the leading eigenvector (band "rest"); diverging map, saturating beyond ±1. Spectra: eigenvalues of $P\\hat CP$ and $P\\Sigma_gP$ ($P = I - \\mathbf 1\\mathbf 1^\\top/B$), each divided by its largest, log–log; non-positive eigenvalues not drawn; dashed: the rank of the measurement operator. Table, full score: relative Frobenius error $\\lVert\\hat C - \\Sigma_g\\rVert_F/\\lVert\\Sigma_g\\rVert_F$ and $\\mathrm{BW}^2/\\mathrm{tr}\\,\\Sigma_g$; scale-invariant: Frobenius cosine and BW after the best rescaling; contrast-subspace: the same on $P\\hat CP$ against $P\\Sigma_gP$, and the overlap of the top-$k$ eigenspaces (mean $\\cos^2$ of principal angles; 1 = identical). Negative eigenvalues clipped before BW. Bold: the estimator shown above; shaded: the best value per column among the estimators from the weights.`,
      `run <code>${r.name}</code>; <code>index.json</code>, <code>runs/${r.name}/readout.json</code>, <code>chat_*.f32</code>, <code>lifts/${r.lift}/sigma.f32</code>.`);
  }

  function scoreTable(r) {
    const val = row => r[row.src][row.key];
    const best = COLS.map(c => { const vs = ROWS.filter(x => x.src === "readout" && x.key !== "init_ls_both").map(x => val(x)?.[c.key]).filter(Number.isFinite); return c.better === "hi" ? Math.max(...vs) : Math.min(...vs); });
    const grpHead = `<tr><th></th><th colspan="2" style="text-align:center">full</th><th colspan="2" style="text-align:center">scale-invariant</th><th colspan="4" style="text-align:center">contrast-subspace</th></tr>`;
    const colHead = `<tr><th>estimator</th>${COLS.map(c => `<th class="num">${c.label} ${c.better === "hi" ? "↑" : "↓"}</th>`).join("")}</tr>`;
    const body = ROWS.map(row => {
      if (row.group) return `<tr><td colspan="${COLS.length + 1}" style="font:600 11.5px/1.3 var(--sans);color:var(--mute);padding:8px 8px 2px">${row.group}</td></tr>`;
      const v = val(row), sel = row.key === variant;
      return `<tr${sel ? ' style="font-weight:620"' : ""}><td style="padding:3px 8px">${row.label}</td>${COLS.map((c, ci) => {
        const x = v?.[c.key], hl = row.src === "readout" && row.key !== "init_ls_both" && Number.isFinite(x) && D.f2(x) === D.f2(best[ci]);
        return `<td class="num" style="padding:3px 8px${hl ? ";background:var(--wash)" : ""}">${D.f2(x)}</td>`;
      }).join("")}</tr>`;
    }).join("");
    table.innerHTML = `<div class="table-scroll" style="margin:0"><table class="data" style="font-size:13px;min-width:720px"><thead>${grpHead}${colHead}</thead><tbody>${body}</tbody></table></div>`;
    ui.renderMath(table);
  }

  state.onRun(load);
  load(state.run);
  ctx.onParams(p => { const v = p.get("ro"); if (v && v !== variant && VARIANTS.some(x => x.value === v)) { variant = v; seg.set(v); load(state.run); } });
}

// ------------------------------------------------------------------ sample sheets
const pretty = t => t.replace(/</g, "&lt;").replace(/C_hat/g, "Ĉ").replace(/Sigma_g/g, "Σg").replace(/mu_g/g, "μg").replace(/lambda\/eta/g, "λ/η")
  .replace(/\btheta\b/g, "θ").replace(/\bphi\b/g, "φ").replace(/\^2/g, "²").replace(/\|\|/g, "‖").replace(/x0/g, "x₀");
export function samplesFigure(host, capHost, index, state, ctx) {
  const sheets = index.samples, byName = Object.fromEntries(index.runs.map(r => [r.name, r]));
  const label = s => { const r = byName[s.run]; return r.lift === "pixels" ? `pixels, θ = ${r.theta} (${r.final.n_live_clusters} parts)` : `RFF D = ${r.B}, θ = ${r.theta} (${r.final.n_live_clusters} parts)`; };
  const forRun = r => sheets.findIndex(s => s.run === r.name);
  let si = ctx.param("sheet", -1);
  if (!(si >= 0 && si < sheets.length)) si = forRun(state.run) >= 0 ? forRun(state.run) : Math.max(0, sheets.findIndex(s => s.run === index.headline));
  const seg = ui.select({ label: "sheet", options: sheets.map((s, i) => ({ value: i, label: label(s) })), value: si, onChange: v => { si = v; ctx.setParams({ sheet: v }); draw(); } });
  const body = ui.h("div"), gen = ui.h("div", { class: "u-help", style: { marginTop: "10px", fontSize: "12.5px", maxWidth: "860px" } });
  host.replaceChildren(ui.h("div", { style: { marginBottom: "10px" } }, seg.el), body, gen);
  let token = 0;
  async function draw() {
    const my = ++token, s = sheets[si], img = await D.image(s.png);
    if (my !== token) return;
    const grid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "minmax(120px, 190px) minmax(0, 1fr)", gap: "8px 14px", alignItems: "center" } });
    s.rows.forEach((row, ri) => {
      const tiles = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(16, minmax(0, 30px))", gap: "2px" } });
      for (let c = 0; c < s.layout.ncol; c++) tiles.append(D.spriteCanvas(img, s.layout, ri * s.layout.ncol + c));
      const bits = [];
      if (row.mmd2_to_heldout != null) bits.push(`MMD² ${ui.fmt(row.mmd2_to_heldout, 3)}`);
      if (row.psd_negative_mass_clipped > 1e-6) bits.push(`clipped ${ui.fmtPct(row.psd_negative_mass_clipped, 0)}`);
      if (row.preimage_rel_residual_median != null) bits.push(`pre-image residual ${ui.fmt(row.preimage_rel_residual_median, 2)}`);
      if (row.rel_residual_median != null) bits.push(`residual ${ui.fmt(row.rel_residual_median, 2)}`);
      const tr = row.target_trace_over_true_trace;
      if (tr != null && tr < 0.99) bits.push(`variance ${ui.fmtPct(tr, 0)} of true`);
      if (tr != null && tr > 1.01) bits.push(`variance ${ui.fmt(tr, 2)}× true`);
      grid.append(ui.h("div", { style: { font: "560 12px/1.3 var(--sans)", color: P.ink2, textAlign: "right" }, title: row.detail || "", html: `${row.label}${bits.length ? `<br><span style="font-weight:400;color:var(--mute)">${bits.join(" · ")}</span>` : ""}` }),
        ui.h("div", { style: { minWidth: 0, overflowX: "auto" } }, tiles));
    });
    body.replaceChildren(grid);
    const g = s.generator;
    const items = Object.entries(g).filter(([k]) => !["n_metric", "seed", "kind"].includes(k)).map(([k, v]) => `<b>${k.replace(/_/g, " ")}:</b> ${pretty(String(v))}`);
    gen.innerHTML = `Generator (<code>${g.kind}</code>), as exported: ${items.join("; ")}. Noise floor (real training digits against held-out): MMD² ${ui.fmt(s.mmd.reference_real_train_vs_heldout, 2)}.`;
    D.caption(capHost, `16 samples per row, 14×14, grey scale clipped to $[0, 1]$, enlarged without smoothing; every generator row uses the same random draws. Per row: squared MMD to ${g.n_metric} held-out test digits.`,
      `Rows as exported: real held-out digits${s.rows.some(r => /round trip/.test(r.label)) ? "; the same digits mapped through $\\phi$ and back by the pre-image optimiser" : ""}; the true-moment generator (the exact $\\Sigma_g$: a same-generator reference, not a universal ceiling); the weight-reconstructed generator ($\\hat C$ from masked least squares, parameters known); baselines diag $\\Sigma_g$ and the readout of the untrained network. MMD: unbiased squared MMD with a Gaussian kernel on 14×14 pixel vectors, median bandwidth. Also per row: the share of negative eigen-mass clipped${s.rows.some(r => r.preimage_rel_residual_median != null) ? ", the median relative pre-image residual $\\lVert\\phi(x^\\ast) - g\\rVert/\\lVert g\\rVert$" : ""}, and the generator's total variance relative to $\\mathrm{tr}\\,\\Sigma_g$ when it differs by more than 1%.`,
      `run <code>${s.run}</code>; seed ${g.seed}; <code>data/mnist/${s.png}</code>.`);
  }
  draw();
  state.onRun(r => { const i = forRun(r); if (i >= 0 && i !== si) { si = i; seg.set(i); draw(); } });
}
