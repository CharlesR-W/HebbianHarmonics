// Chapter "fusion" (an experiment sub-page): the two-neuron calibration, the live B = 3 network (with a second
// figure, w against D^2, driven by the same run), and a b-scan of that network computed in the page.
// Modules: ch_fusion_model.js (model side, no DOM), ch_fusion_draw.js (purpose-built charts),
// ch_fusion_pair.js, ch_fusion_sim.js, ch_fusion_scan.js (the figures).
import { initPair } from "./ch_fusion_pair.js";
import { initSim } from "./ch_fusion_sim.js";
import { initScan } from "./ch_fusion_scan.js";

export async function init(panel, ctx) {
  let J = null;
  try { const r = await fetch("data/sweeps2/fusion.json"); if (r.ok) J = await r.json(); } catch (e) { J = null; }

  const pair = initPair(ctx.slot("pair"), ctx, { loop: J ? J.pair.loop : null });
  const sim = initSim(ctx.slot("sim"), ctx, { fixedSlot: ctx.slot("fixed") });
  initScan(ctx.slot("scan"), ctx.slot("scan-text"), () => sim.params);
  ctx.onParams(params => pair.apply(params));
}
