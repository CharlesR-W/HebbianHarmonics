// Chapter "allocation" (Q-allocation, SPEC §3; findings G17–G20). Three parts, one module each:
//   ch_allocation_base.js     (a) base model: allocation set by the attachments, cut up by b (live + sweep)
//   ch_allocation_plastic.js  (b) plastic attachments under the energy rule drain to quiet inputs (live + sweep)
//   ch_allocation_sm.js       (c) the added similarity-matching principle: mass law, specialisation (live + sweep)
// ch_allocation_model.js holds the DOM-free model code. Every number in the prose is filled here from
// data/sweeps2/allocation.json (spans with data-fill) or computed live by the figures.
import * as ui from "./ui.js";
import { baseFigure, sweepCharts, randomFigure, randomStats, groupPlanted, meanSd } from "./ch_allocation_base.js";
import { plasticFigure, plasticTable } from "./ch_allocation_plastic.js";
import { smFigure, smTables, smRows } from "./ch_allocation_sm.js";

export async function init(panel, ctx) {
  // live figures first: they need no data
  baseFigure(ctx, ctx.slot("base"));
  plasticFigure(ctx, ctx.slot("plastic"));
  smFigure(ctx, ctx.slot("sm"));

  const r = await fetch("data/sweeps2/allocation.json", { cache: "no-cache" });
  if (!r.ok) throw new Error(`data/sweeps2/allocation.json: HTTP ${r.status}`);
  const data = await r.json();

  sweepCharts(data, ctx.slot("sweep-var"), ctx.slot("sweep-count"));
  randomFigure(data, ctx.slot("random"));
  plasticTable(data, ctx.slot("plastic-table"));
  smTables(data, ctx.slot("sm-tables"));
  fillProse(panel, data);
}

function fillProse(panel, data) {
  const put = (key, html) => panel.querySelectorAll(`[data-fill="${key}"]`).forEach(el => { el.innerHTML = html; ui.renderMath(el); });
  const pmText = ([m, s], d = 1) => (s > 0 ? `${ui.fmtFixed(m, d)} ± ${ui.fmtFixed(s, d)}` : ui.fmt(m, 3));

  // (a) variance sweep at the smallest b
  const gv = groupPlanted(data, "variance"), b0 = gv.bs[0], vlo = gv.vals[0], vhi = gv.vals[gv.vals.length - 1];
  const lo = gv.at(b0, vlo), hi = gv.at(b0, vhi);
  put("a-b", `$b = ${b0}$`);
  put("a-n", String(lo[0].n_block0));
  put("a-v-lo", String(vlo)); put("a-v-hi", String(vhi));
  put("a-k-lo", pmText(meanSd(lo.map(x => x.K_block0))));
  put("a-k-hi", pmText(meanSd(hi.map(x => x.K_block0))));
  put("a-size-hi", ui.fmt(meanSd(hi.map(x => x.mean_size_block0))[0], 2));
  const allA = data.a.planted, mixed = allA.reduce((s, x) => s + x.mixed_clusters, 0);
  put("a-runs", String(allA.length)); put("a-mixed", mixed === 0 ? "no" : String(mixed));
  const keep = allA.every(x => x.converged) ? "" : " (not every run converged)";
  if (keep) put("a-runs", `${allA.length}${keep}`);
  // random attachments: Spearman over multi-member components, pooled over seeds
  for (const rs of randomStats(data)) put(`a-sp-${rs.b}`, `${ui.fmtFixed(rs.rho, 2)} (${rs.n} components)`);

  // (c1) mass law with no ingredient, every setting
  const none = data.c.filter(x => x.ing === "none");
  const sci = x => (Math.abs(x) < 1e-300 ? "0" : ui.fmt(x, 1));
  put("c1-qdev", sci(Math.max(...none.map(x => x.q_dev))));
  put("c1-off", sci(Math.max(...none.map(x => x.offdiag))));
  put("c1-exc", sci(Math.max(...none.map(x => Math.abs(x.obj_main_excess)))));
  const pureBySetting = [...new Set(none.map(x => `${x.spec}|${x.nu}`))].map(k => { const rs = none.filter(x => `${x.spec}|${x.nu}` === k); return rs.reduce((s, x) => s + x.pure_frac, 0) / rs.length; });
  put("c1-pure-lo", ui.fmtPct(Math.min(...pureBySetting), 0)); put("c1-pure-hi", ui.fmtPct(Math.max(...pureBySetting), 0));
  // (c2) the closest count result: non-negative in the eigenbasis at the largest budget
  const nuMax = Math.max(...data.c.filter(x => x.spec === "lam8").map(x => x.nu));
  const best = smRows(data, "lam8", nuMax, "nonneg_eig");
  if (best.length) put("c2-best", ui.fmtFixed(best.reduce((s, x) => s + x.count_dev, 0) / best.length, 3));
}
