// Chapter "readout": the readout against N at fixed B (data/big/readout_vs_N.json, hebb/big/readout_vs_n.py,
// findings B4). Four panels against N: rank of the live-edge operator, its condition number, the recovery cosine,
// and the three errors kept apart (fixed point, inverse problem, sampling). Lines: mean over seeds (the
// condition number: geometric mean); bands: min to max over seeds. Every number in the prose slots comes from the file.
import * as ui from "./ui.js";
import { lineChart } from "./charts.js";
import { catColor } from "./palette.js";

export async function nFigure(ctx) {
  const slot = ctx.slot("vsn"); if (!slot) return;
  const data = await fetch("data/big/readout_vs_N.json").then(r => { if (!r.ok) throw new Error(`readout_vs_N.json: HTTP ${r.status}`); return r.json(); });
  const cfg = data.meta.config, Ns = cfg.Ns, bs = cfg.bs, dim = cfg.contrast_dim;
  const runs = data.runs;
  const at = (b, N) => runs.filter(r => r.b === b && r.N === N);
  const stat = (b, get, how = "mean") => {
    const m = [], lo = [], hi = [];
    for (const N of Ns) {
      const v = at(b, N).map(get).filter(Number.isFinite);
      lo.push(Math.min(...v)); hi.push(Math.max(...v));
      m.push(how === "geo" ? 10 ** (v.reduce((a, x) => a + Math.log10(x), 0) / v.length) : how === "max" ? Math.max(...v) : v.reduce((a, x) => a + x, 0) / v.length);
    }
    return { m, lo, hi };
  };
  const bl = b => `b = ${b}`;
  slot.replaceChildren();
  const grid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(400px, 100%), 1fr))", gap: "18px 28px" } });
  slot.append(grid);
  const box = (title, sub) => {
    const body = ui.h("div");
    const el = ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", html: title }), ui.h("div", { class: "u-help", style: { margin: "-2px 0 6px" }, html: sub }), body);
    grid.append(el); ui.renderMath(el); return body;
  };
  const common = { x: Ns, xlog: true, height: 220, xlabel: "interior neurons N", xname: "N" };

  const rank = bs.map((b, k) => { const s = stat(b, r => r.rank_1e8); return { name: bl(b), y: s.m, band: [s.lo, s.hi], color: catColor(k), dots: true }; });
  lineChart(box("(a) Rank of the live-edge operator", `of ${dim}; dashed: full rank`), { ...common, series: rank, yrange: [0, dim + 4], ylabel: "rank", refs: [{ y: dim, label: `${dim}` }] });

  const cond = bs.map((b, k) => { const s = stat(b, r => r.condition, "geo"); return { name: bl(b), y: s.m, band: [s.lo, s.hi], color: catColor(k), dots: true }; });
  lineChart(box("(b) Condition number of the live-edge operator", "log scale; on the singular values above $10^{-8}\\sigma_{\\max}$"), { ...common, series: cond, ylog: true, ylabel: "σmax / σmin" });

  const cos = [
    ...bs.map((b, k) => { const s = stat(b, r => r.ridge_psd_cos_Chat); return { name: `${bl(b)}, vs P Ĉ P`, y: s.m, band: [s.lo, s.hi], color: catColor(k), dots: true }; }),
    ...bs.map((b, k) => ({ name: `${bl(b)}, vs P Σ P`, y: stat(b, r => r.ridge_psd_cos_Sigma).m, color: catColor(k), dash: "4 3", width: 1.2 })),
  ];
  lineChart(box("(c) Recovery: Frobenius cosine", "solid: with $P\\hat C P$, the exact target; dashed: with $P\\Sigma_g P$"), { ...common, series: cos, yrange: [0, 1.02], ylabel: "cosine" });

  const samp = runs.reduce((a, r) => a + r.err_sampling, 0) / runs.length;
  const errs = [
    ...bs.map((b, k) => { const s = stat(b, r => Math.max(r.err_inverse_identifiable, 1e-16), "max"); return { name: `${bl(b)}, inverse problem`, y: s.m, color: catColor(k), dots: true }; }),
    ...bs.map((b, k) => ({ name: `${bl(b)}, fixed point`, y: stat(b, r => r.err_fixed_point, "max").m, color: catColor(k), dash: "2 3", width: 1.2 })),
  ];
  lineChart(box("(d) Three errors, kept apart", "log scale; worst seed; dashed line: sampling (mean over runs)"), { ...common, series: errs, ylog: true, ylabel: "error", yrange: [1e-10, 1],
    refs: [{ y: samp, label: `sampling ${ui.fmt(samp, 2)}` }] });

  // numbers for the prose
  const S = (b, N) => data.summary.find(s => s.b === b && s.N === N);
  const firstFull = b => Ns.find(N => at(b, N).every(r => r.rank_1e8 === dim));
  const full = runs.filter(r => r.rank_1e8 === dim);
  const worst = full.reduce((a, r) => (r.ridge_psd_cos_Chat < a.ridge_psd_cos_Chat ? r : a));
  const fullGood = full.filter(r => r !== worst);
  const v = {
    nruns: String(runs.length),
    conv: runs.every(r => r.converged) ? "all" : `${runs.filter(r => r.converged).length} of ${runs.length}`,
    resmax: ui.fmt(Math.max(...runs.map(r => r.residual)), 2),
    full_b0: String(firstFull(bs[0])), full_b1: String(firstFull(bs[1])), full_b2: String(firstFull(bs[2])),
    edges40: ui.fmt(S(bs[0], Ns[0]).n_live[0], 2), edges500: ui.fmtInt(Math.round(S(bs[0], Ns[Ns.length - 1]).n_live[0])),
    K320: ui.fmtInt(Math.round(S(bs[0], 320)?.K?.[0] ?? NaN)), K500: ui.fmtInt(Math.round(S(bs[0], 500)?.K?.[0] ?? NaN)),
    cond40: ui.fmtFixed(S(bs[2], Ns[0]).log10_condition[0], 1), cond500: ui.fmtFixed(S(bs[2], Ns[Ns.length - 1]).log10_condition[0], 1),
    cosfull: ui.fmtFixed(Math.min(...fullGood.map(r => r.ridge_psd_cos_Chat)), 3),
    worstN: String(worst.N), worstb: String(worst.b), worstseed: String(worst.seed), worstrank4: String(worst.rank_1e4),
    worstcond: ui.fmt(worst.condition, 2), worstcos: ui.fmtFixed(worst.ridge_psd_cos_Chat, 3),
    fpmax: ui.fmt(Math.max(...runs.map(r => r.err_fixed_point)), 2),
    invmax: ui.fmt(Math.max(...runs.map(r => r.err_inverse_identifiable)), 2),
    samp: ui.fmt(samp, 2),
    cosSigma: ui.fmtFixed(full.reduce((a, r) => a + r.ridge_psd_cos_Sigma, 0) / full.length, 3),
  };
  ctx.panel.querySelectorAll("[data-nv]").forEach(el => { const k = el.dataset.nv; if (v[k] != null) el.textContent = v[k]; });
}
