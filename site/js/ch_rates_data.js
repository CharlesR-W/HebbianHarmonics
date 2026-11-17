// Chapter "rates": data loading and the numbers the prose and tables quote, all read from data/nonlin/index.json.
export const BASE = "data/nonlin/";

const cache = new Map();
export function json(path) {
  if (!cache.has(path)) cache.set(path, fetch(BASE + path).then(r => { if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.json(); }));
  return cache.get(path);
}
export function bytes(path) {
  if (!cache.has(path)) cache.set(path, fetch(BASE + path).then(r => { if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.arrayBuffer(); }).then(b => new Uint8Array(b)));
  return cache.get(path);
}
export function image(path) {
  if (!cache.has(path)) cache.set(path, (async () => { const im = new Image(); im.src = BASE + path; await im.decode(); return im; })());
  return cache.get(path);
}

export const Q_GRID = [0.946, 0.967, 0.980, 0.987, 0.992];
export const qIndex = q => Q_GRID.findIndex(x => Math.abs(x - q) < 0.002);

// run selectors
export function pick(index, f) { return index.runs.filter(r => r.res === 14 && f(r)); }
export const isLin = r => r.rule === "linear";
export const tauOf = r => (r.rule === "linear" ? null : r.tau);
export const mainPart = r => (r.parts && r.parts.length ? r.parts[0] : null);

export function range(xs) {
  const v = xs.filter(Number.isFinite);
  return v.length ? [Math.min(...v), Math.max(...v)] : [NaN, NaN];
}

/** Numbers quoted in the tables, computed from index.json at runtime. */
export function summarise(index) {
  const R = index.runs.filter(r => r.res === 14);
  const g1 = R.filter(r => r.gamma === 1);
  const A = (exp, tau, seed) => g1.filter(r => r.exp === exp && (tau === undefined || r.tau === tau) && (seed === undefined || r.seed === seed));
  const lowQ = r => qIndex(r.q) >= 0 && qIndex(r.q) <= 2;
  const S = {};

  // part table rows (seed 0, q index 0..2)
  const row = (label, runs) => {
    const byQ = [0, 1, 2].map(k => runs.find(r => qIndex(r.q) === k));
    const parts = byQ.map(r => r && mainPart(r));
    return {
      label, cos: parts.map(p => p ? p.cos_to_linear_part : NaN),
      skew: range(parts.map(p => p ? p.skew : NaN)), kurt: range(parts.map(p => p ? p.excess_kurtosis : NaN)),
      excess: range(parts.map(p => p ? p.rate_var_over_gauss : NaN)),
    };
  };
  S.partRows = [
    row("linear rule", A("A_lin", undefined, 0)),
    row("rectified, $\\tau = 0$", A("A_rect", 0, 0)),
    row("rectified, $\\tau = 0.5$", A("A_rect", 0.5, 0)),
    row("rectified, $\\tau = 1$", A("A_rect", 1, 0)),
    row("rectified $\\tau = 0$ on Gaussian pixels, own $q$", A("Bown_rect", 0, 0)),
  ];
  const seed1 = A("A_rect", 0, 1).filter(lowQ).sort((a, b) => a.q - b.q).map(r => mainPart(r)?.cos_to_linear_part);
  S.seed1Cos = seed1;

  // skewness: linear vs rectified (seed 0, q index 0..2)
  S.skewLin = range(A("A_lin", undefined, 0).filter(lowQ).map(r => mainPart(r)?.skew));
  S.skewRect = range(A("A_rect", undefined, 0).filter(lowQ).map(r => mainPart(r)?.skew));
  S.cosRect = range(A("A_rect", undefined, undefined).filter(lowQ).map(r => mainPart(r)?.cos_to_linear_part));
  S.cosSurr = range(A("Bown_rect", 0, 0).filter(lowQ).map(r => mainPart(r)?.cos_to_linear_part));
  S.skewSurr = range(A("Bown_rect", 0, 0).filter(lowQ).map(r => mainPart(r)?.skew));
  const r28 = index.runs.filter(r => r.res === 28);
  S.res28 = { linLive: range(r28.filter(isLin).map(r => r.final.n_live_clusters)), rectLive: range(r28.filter(r => !isLin(r)).map(r => r.final.n_live_major)) };

  // survival at q = 0.987
  const q4 = r => qIndex(r.q) === 3;
  S.t1Alive987 = A("A_rect", 1, 0).find(q4)?.final.live_neurons;
  S.linDead987 = range(A("A_lin").filter(q4).map(r => r.final.n_dead));
  const surr = A("B_rect", 0, 0);
  S.surrDead967 = surr.find(r => qIndex(r.q) === 1)?.final.n_dead;

  // structure: live clusters
  S.rectMajorMax = Math.max(...R.filter(r => r.exp.endsWith("A_rect")).map(r => r.final.n_live_major));
  S.linLiveAtQ98 = range(A("A_lin").filter(r => qIndex(r.q) === 2).map(r => r.final.n_live_clusters));
  S.rectLiveAtQ98 = range(A("A_rect").filter(r => qIndex(r.q) === 2).map(r => r.final.n_live_clusters));
  const oneClusterRect = R.filter(r => r.exp.endsWith("A_rect") && r.final.n_live_clusters === 1 && r.final.n_live_major === 1);
  S.cloneL1 = Math.max(...oneClusterRect.map(r => r.final.within_l1_live).filter(Number.isFinite));
  S.nRuns = index.runs.length;

  // probes (tau = 0 rates of all N neurons)
  const pixelNets = R.filter(r => r.input === "pixels");
  S.probe = {
    rectOne: range(pixelNets.filter(r => !isLin(r) && r.final.n_live_clusters === 1).map(r => r.probes.trained_rate_t0_all)),
    linMany: range(pixelNets.filter(r => isLin(r) && r.final.n_live_clusters >= 3).map(r => r.probes.trained_rate_t0_all)),
    linOne: range(pixelNets.filter(r => isLin(r) && r.final.n_live_clusters === 1).map(r => r.probes.trained_rate_t0_all)),
    pixels: mean(pixelNets.map(r => r.probes.pixels)),
    untrained: mean(pixelNets.map(r => r.probes.init_rate_t0_all)),
    untrainedVolt: mean(pixelNets.map(r => r.probes.init_volt_all)),
    randrelu: mean(pixelNets.map(r => r.probes.randrelu_t0_all)),
  };

  // readout bias per tau (gamma = 1, seed 0, q index 0..2)
  S.readout = [0, 0.5, 1].map(tau => {
    const runs = A("A_rect", tau, 0).filter(lowQ).sort((a, b) => a.q - b.q);
    return { tau, ratio: runs.map(r => mainPart(r)?.energy_ratio_gauss_both), excess: runs.map(r => mainPart(r)?.rate_var_over_gauss),
      expected: runs.map(r => mainPart(r)?.energy_ratio_gauss_expected_cov) };
  });
  S.readoutSurr = range(g1.filter(r => (r.exp === "B_rect" || r.exp === "Bown_rect")).map(r => mainPart(r)?.energy_ratio_gauss_both));
  // the same surrogate nets without lateral plasticity (gamma = 0: input-synapse equations only)
  S.readoutSurrG0 = range(R.filter(r => r.gamma === 0 && (r.exp === "G0_B_rect" || r.exp === "G0_Bown_rect")).map(r => mainPart(r)?.energy_ratio_gauss_both));
  S.readoutLin = range(R.filter(isLin).map(r => mainPart(r)?.energy_ratio_ls_both));
  S.readoutG0 = range(R.filter(r => r.exp === "G0_A_rect" && r.tau === 0).map(r => mainPart(r)?.energy_ratio_gauss_both));
  return S;
}

export function mean(xs) { const v = xs.filter(Number.isFinite); return v.reduce((a, b) => a + b, 0) / (v.length || 1); }
