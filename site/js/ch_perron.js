// Chapter "perron" (Perron clusters). Mounts the page's figures; each lives in its own module:
//   grid            ch_perron_tri.js    three inputs: finished runs over the predicted parts
//   phase           ch_perron_phase.js  the plane of θ and ρ12: predicted arrangement, 80 runs, one live run (tri.js mountLive)
//   fill            ch_perron_fill.js   16 hierarchical inputs, fresh starts: fraction of parts holding a group over θ and γ
//   land, norm      ch_perron_land.js   the landscape R_θ with the neurons' paths; subtractive vs multiplicative
//   tree            ch_perron_teach.js  A = Σ − θ and the correlation tree cut at θ (8 inputs)
//   big             ch_perron_big.js    groups vs parts vs eigenvectors in the precomputed 400-neuron runs
//   ramp            ch_perron_ramp.js   the precomputed θ ramps on 16 hierarchical inputs
// The three-input simulations use the v1 engine through ch_perron_model.js.
import * as ui from "./ui.js";
import { mountGrid } from "./ch_perron_tri.js";
import { mountPhase } from "./ch_perron_phase.js";
import { mountFill } from "./ch_perron_fill.js";
import { mountLandscape, mountNorm } from "./ch_perron_land.js";
import { mountTree } from "./ch_perron_teach.js";
import { mountBig } from "./ch_perron_big.js";
import { mountRamp } from "./ch_perron_ramp.js";

export async function init(panel, ctx) {
  const fail = (name, el, e) => { console.warn(`perron: ${name}`, e); if (el) el.replaceChildren(ui.h("p", { class: "u-help", text: `Could not build this figure: ${e.message}` })); };
  const mount = (name, fn) => { const el = ctx.slot(name); if (!el) return; try { const r = fn(el, ctx); if (r?.catch) r.catch(e => fail(name, el, e)); } catch (e) { fail(name, el, e); } };
  mount("grid", mountGrid);
  mount("phase", mountPhase);
  mount("land", mountLandscape);
  mount("tree", mountTree);
  mount("big", mountBig);
  mount("fill", mountFill);
  mount("ramp", (el, c) => mountRamp(el, c, { onData: fillRampText }));
  mount("norm", mountNorm);

  // scope of the exhaustive random-tree check quoted in the tree figure's details, read from c1_summary.json
  const c1 = ctx.slot("c1-text");
  if (c1) fetch("data/perron2/c1_summary.json").then(r => r.json()).then(d => {
    const ru = d.random_ultrametric, Bs = ru.B, fails = ru.unit.n_fail + ru.ultra_var.n_fail;
    c1.textContent = `On ${ru.n_trees} random hierarchical covariances with ${Math.min(...Bs)} to ${Math.max(...Bs)} inputs, ${ru.unit.n_theta / ru.n_trees} thresholds each, once with unit variances and once with unequal ones, the stable parts equalled the branches of the covariance tree ${fails === 0 ? "at every threshold tested" : `except at ${fails} thresholds`} (data/perron2/c1_summary.json).`;
  }).catch(e => console.warn("perron: c1_summary", e));

  // numbers quoted in the text next to the ramp figure, read from ramps.json
  function fillRampText({ splitTests }) {
    const cases = splitTests.cases;
    const half = cases.find(c => c.gamma === 0.5 && c.omega === 0.009 && c.kick === 0.001 && c.sizes_before.length === 1 && c.sizes.length > 1);
    const one = cases.filter(c => c.gamma === 1);
    const el = ctx.slot("split-text");
    if (!el || !half) return;
    el.textContent = `In a separate test we took the 300-neuron group just below the level 0.25 and moved θ two steps past it at once. ` +
      `With γ = 0.5 it split into ${half.sizes.join(" and ")} neurons, one group per child. With γ = 1 it stayed whole (${one.map(c => c.sizes.join("/")).every(s => s === "300") ? "all 300 neurons" : one.map(c => c.sizes.join("/")).join(", ")}) in all ${one.length} variants we tried, with floors down to a hundredth of the default.`;
  }
}
