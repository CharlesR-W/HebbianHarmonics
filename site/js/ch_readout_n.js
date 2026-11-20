// Chapter "readout", figure against N (data/big/readout_vs_N.json, hebb/big/readout_vs_n.py): independent equations
// (rank of the live-synapse equations, of 66) and the error of the recovered covariance, for three values of b.
import * as ui from "./ui.js";
import { lineChart } from "./charts.js";
import { catColor } from "./palette.js";

export async function nFigure(ctx) {
  const slot = ctx.slot("vsn");
  const r = await fetch("data/big/readout_vs_N.json");
  if (!r.ok) throw new Error(`readout_vs_N.json: ${r.status}`);
  const d = await r.json();
  slot.replaceChildren();
  const cfg = d.meta.config, Ns = cfg.Ns, bs = cfg.bs, dim = cfg.contrast_dim;
  const runsAt = (b, N) => d.runs.filter(u => u.b === b && u.N === N);
  const stat = (b, key, f) => Ns.map(N => f(runsAt(b, N).map(u => u[key])));
  const mean = v => v.reduce((a, c) => a + c, 0) / v.length;
  // geometric mean for the error (it spans eight decades), min-max band
  const gmean = v => Math.exp(mean(v.map(x => Math.log(x))));

  const grid = ui.h("div", { class: "ro-two" }); slot.append(grid);
  const box = (title) => { const b = ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", html: title })); const body = ui.h("div"); b.append(body); grid.append(b); ui.renderMath(b); return body; };
  const name = b => `b = ${b}`;
  // the five N values, equally spaced (they roughly double)
  const X = Ns.map((_, i) => i), xfmt = v => (Number.isInteger(v) && Ns[v] != null ? String(Ns[v]) : "");
  const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  const pow10 = v => { const e = Math.round(Math.log10(v)); if (Math.abs(v / 10 ** e - 1) > 1e-6) return ""; return e === 0 ? "1" : "10" + (e < 0 ? "⁻" : "") + String(Math.abs(e)).split("").map(c => SUP[c]).join(""); };
  lineChart(box("Independent equations from the live synapses"), {
    x: X, xrange: [-0.15, Ns.length - 0.85], yrange: [0, 72], height: 260,
    xlabel: "tested network sizes N (12 inputs)", ylabel: "independent equations", xname: "N", xfmt, yfmt: v => String(Math.round(v)),
    refs: [{ y: dim, label: `${dim} unknowns` }],
    series: bs.map((b, k) => ({ name: name(b), color: catColor(k), dots: true,
      y: stat(b, "rank_1e8", mean), band: [stat(b, "rank_1e8", v => Math.min(...v)), stat(b, "rank_1e8", v => Math.max(...v))] })),
  });
  lineChart(box("Error of the recovered second moment"), {
    x: X, ylog: true, xrange: [-0.15, Ns.length - 0.85], yrange: [1e-8, 2], height: 260,
    xlabel: "tested network sizes N (12 inputs)", ylabel: "relative error", xname: "N", xfmt, yfmt: pow10, legend: false,
    series: bs.map((b, k) => ({ name: name(b), color: catColor(k), dots: true,
      y: stat(b, "ridge_psd_relerr_Chat", gmean), band: [stat(b, "ridge_psd_relerr_Chat", v => Math.min(...v)), stat(b, "ridge_psd_relerr_Chat", v => Math.max(...v))] })),
  });

  // numbers for the text
  const put = (k, v) => ctx.article.querySelectorAll(`[data-nv=${k}]`).forEach(e => { e.textContent = v; });
  const sm = (b, N) => d.summary.find(s => s.b === b && s.N === N);
  const b0 = bs[0];
  put("edges40", ui.fmtInt(sm(b0, 40)?.n_live[0]));
  put("edges500", ui.fmtInt(sm(b0, 500)?.n_live[0]));
  put("K320", ui.fmtInt(sm(b0, 320)?.K[0]));
  const full = d.runs.filter(u => u.rank_1e8 === dim);
  const worst = full.reduce((a, u) => (u.ridge_psd_relerr_Chat > a.ridge_psd_relerr_Chat ? u : a), full[0]);
  put("worstat", `N = ${worst.N}, b = ${worst.b}`);
  put("worst", `seed ${worst.seed}, condition number ${ui.fmt(worst.condition, 2)}, only ${worst.rank_1e4} of 66 directions above $10^{-4}$ of the largest`);
  put("worstrel", ui.fmtPct(worst.ridge_psd_relerr_Chat));
  ui.renderMath(ctx.article.querySelector("#readout-fig-n"));
}
