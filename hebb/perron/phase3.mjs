// Precompute the runs drawn on the Perron page's phase diagram (site/data/perron/phase3.json).
// Three inputs with unit variances; g3 correlates 0.3 with g1 and with g2; the correlation rho of g1 and g2 and the
// threshold theta vary on a grid chosen to avoid exact ties (theta never equals 0.3 or a grid value of rho).
// Every run uses exactly the page's model code (site/js/ch_perron_model.js, the v1 engine): N = 60, gamma = 0.25,
// w_min = 1e-4, seed 0, until the largest weight change is below 1e-7 or 6,000 steps.
//
//   node hebb/perron/phase3.mjs          (about a minute on one core)
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as M from "../../site/js/ch_perron_model.js";
import * as T from "../../site/js/ch_perron_teach_model.js";

const R13 = 0.3, R23 = 0.3;
const THETAS = Array.from({ length: 8 }, (_, k) => +(0.08 + 0.1 * k).toFixed(2));
const RHOS = Array.from({ length: 10 }, (_, k) => +(-0.45 + 0.15 * k).toFixed(2));
const r4 = x => Math.round(x * 1e4) / 1e4;

const runs = [];
for (const rho of RHOS) for (const theta of THETAS) {
  const Sigma = M.sigmaOf({ v: [1, 1, 1], r: [rho, R13, R23] });
  const { final } = await M.run({ ...M.DEFAULTS, theta, Sigma });
  runs.push({ theta, rho, steps: final.steps, resid: final.resid, converged: final.resid < M.RES_TOL, n_dead: final.nDead,
    groups: final.groups.map(g => ({ n: g.n, h: g.h.map(r4) })) });
  process.stdout.write(".");
}
// The predicted arrangement over the plane: per rho row, the theta intervals on which the set of stable parts
// (exact enumeration, ch_perron_teach_model.js) is constant, located by bisection to 1e-4. A key lists the supports
// of the stable parts, inputs numbered from 0, e.g. "01|2" = the pair g1 g2 and g3 alone; "" = no stable part.
const MAP = { theta: [0, 0.8], rho: [-0.5, 0.9], rows: 281 };
const keyAt = (rho, theta) => T.stableParts(M.sigmaOf({ v: [1, 1, 1], r: [rho, R13, R23] }), 3, theta).map(p => p.S.join("")).sort().join("|");
const segments = [];
for (let j = 0; j < MAP.rows; j++) {
  const rho = MAP.rho[0] + (MAP.rho[1] - MAP.rho[0]) * j / (MAP.rows - 1);
  const xs = Array.from({ length: 161 }, (_, i) => MAP.theta[0] + (MAP.theta[1] - MAP.theta[0]) * i / 160);
  const ks = xs.map(x => keyAt(rho, x));
  const row = [[MAP.theta[0], ks[0]]];
  for (let i = 1; i < xs.length; i++) if (ks[i] !== ks[i - 1]) {
    let a = xs[i - 1], b = xs[i];
    while (b - a > 1e-4) { const m = (a + b) / 2; if (keyAt(rho, m) === ks[i - 1]) a = m; else b = m; }
    // a tie can leave a thin interval with a third key between the two; record what holds just past the break
    row.push([r4(b), keyAt(rho, b + 1e-5)]);
    if (row[row.length - 1][1] !== ks[i]) row.push([r4(Math.min(xs[i], b + 1e-3)), ks[i]]);
  }
  segments.push({ rho: r4(rho), seg: row });
}
const out = { map: { ...MAP, segments }, meta: { script: "hebb/perron/phase3.mjs", sigma: { variances: [1, 1, 1], r13: R13, r23: R23 }, thetas: THETAS, rhos: RHOS,
  model: { ...M.DEFAULTS, dt: M.DT, res_tol: M.RES_TOL, max_steps: M.MAX_STEPS } }, runs };
const path = fileURLToPath(new URL("../../site/data/perron/phase3.json", import.meta.url));
writeFileSync(path, JSON.stringify(out));
console.log(`\nwrote ${path}: ${runs.length} runs, ${runs.filter(r => r.converged).length} converged`);
