// Chapter "allocation" (Who gets the neurons?). Three parts, one module each:
//   ch_allocation_base.js     1. fixed attachments: the wiring decides, b cuts it up (live + variance sweep)
//   ch_allocation_plastic.js  2. learning attachments drain to the quiet inputs (live + sweep bars)
//   ch_allocation_sm.js       3. the added similarity-matching objective: mass law, counts (live + sweep)
// ch_allocation_model.js holds the DOM-free model code. Every number in the prose is filled here from
// data/sweeps2/allocation.json (spans with data-fill) or computed live by the figures.
import * as ui from "./ui.js";
import { baseFigure, sweepCharts, randomStats, groupPlanted, meanSd } from "./ch_allocation_base.js";
import { plasticFigure, plasticBars } from "./ch_allocation_plastic.js";
import { smFigure, smSweep, smRows } from "./ch_allocation_sm.js";

export async function init(panel, ctx) {
  // live figures first: they need no data
  baseFigure(ctx, ctx.slot("base"));
  plasticFigure(ctx, ctx.slot("plastic"));
  smFigure(ctx, ctx.slot("sm"));

  const r = await fetch("data/sweeps2/allocation.json", { cache: "no-cache" });
  if (!r.ok) throw new Error(`data/sweeps2/allocation.json: HTTP ${r.status}`);
  const data = await r.json();

  sweepCharts(data, ctx.slot("sweep-var"));
  const bars = plasticBars(data, ctx.slot("plastic-sweep"));
  smSweep(data, ctx.slot("sm-sweep"));
  fillProse(panel, data, bars);
}

function fillProse(panel, data, bars) {
  const put = (key, html) => panel.querySelectorAll(`[data-fill="${key}"]`).forEach(el => { el.innerHTML = html; ui.renderMath(el); });
  const pmText = ([m, s], d = 1) => (s > 0 ? `${ui.fmtFixed(m, d)} ± ${ui.fmtFixed(s, d)}` : ui.fmt(m, 3));

  // 1. variance sweep at the smallest b
  const gv = groupPlanted(data, "variance"), b0 = gv.bs[0], vlo = gv.vals[0], vhi = gv.vals[gv.vals.length - 1];
  const lo = gv.at(b0, vlo), hi = gv.at(b0, vhi);
  put("a-b", `$b = ${b0}$`);
  put("a-n", String(lo[0].n_block0));
  put("a-v-lo", String(vlo)); put("a-v-hi", String(vhi));
  put("a-k-lo", pmText(meanSd(lo.map(x => x.K_block0))));
  put("a-k-hi", pmText(meanSd(hi.map(x => x.K_block0))));
  const allA = data.a.planted, mixed = allA.reduce((s, x) => s + x.mixed_clusters, 0);
  put("a-runs", `${allA.length}${allA.every(x => x.converged) ? "" : " (not every run converged)"}`);
  put("a-mixed", mixed === 0 ? "no" : String(mixed));
  put("a-seeds", String(new Set(allA.map(x => x.seed)).size));
  // random attachments: Spearman over multi-member clusters, pooled over seeds
  for (const rs of randomStats(data)) put(`a-sp-${rs.b}`, ui.fmtFixed(rs.rho, 2));

  // 2. learning attachments: majority on the quietest block, over the linear settings other than the strongest ridge
  const aMax = Math.max(...bars.map(r => r.aA));
  const drain = bars.filter(r => r.aA < aMax), strong = bars.filter(r => r.aA === aMax);
  const qMaj = drain.map(r => r.maj[3]);
  put("b-q-lo", ui.fmtFixed(Math.min(...qMaj), 1)); put("b-q-hi", ui.fmtFixed(Math.max(...qMaj), 1));
  put("b-n", String(data.meta.config.b.N));
  put("b-amax", String(aMax));
  put("b-strong-q", ui.fmtFixed(Math.max(...strong.map(r => r.quiet)), 2));

  // 3. mass law with no ingredient, every setting
  const none = data.c.filter(x => x.ing === "none");
  const sci = x => (Math.abs(x) < 1e-300 ? "0" : ui.fmt(x, 1));
  put("c1-qdev", sci(Math.max(...none.map(x => x.q_dev))));
  const pureBySetting = [...new Set(none.map(x => `${x.spec}|${x.nu}`))].map(k => { const rs = none.filter(x => `${x.spec}|${x.nu}` === k); return rs.reduce((s, x) => s + x.pure_frac, 0) / rs.length; });
  put("c1-pure-lo", ui.fmtPct(Math.min(...pureBySetting), 0)); put("c1-pure-hi", ui.fmtPct(Math.max(...pureBySetting), 0));
  // the closest count result: non-negative in the eigenbasis at the largest budget
  const nuMax = Math.max(...data.c.filter(x => x.spec === "lam8").map(x => x.nu));
  const best = smRows(data, "lam8", nuMax, "nonneg_eig");
  if (best.length) {
    put("c2-best", ui.fmtFixed(best.reduce((s, x) => s + x.count_dev, 0) / best.length, 3));
    put("c2-best-q", ui.fmtFixed(best.reduce((s, x) => s + x.q_dev, 0) / best.length, 3));
  }
}
