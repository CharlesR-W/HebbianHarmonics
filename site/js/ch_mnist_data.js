// Chapter "mnist": data loading (cached, lazy), sprite tiles and the numbers the prose quotes, all read from
// data/mnist/index.json (format: data/mnist/README.md).
import * as ui from "./ui.js";
import { P } from "./palette.js";

export const BASE = "data/mnist/";
const cache = new Map();
const get = (key, make) => { if (!cache.has(key)) cache.set(key, make()); return cache.get(key); };
const ok = (path, r) => { if (!r.ok) throw new Error(`${path}: ${r.status}`); return r; };
const fileOf = f => (typeof f === "string" ? f : f.file);

export const json = path => get("j:" + path, () => fetch(BASE + path).then(r => ok(path, r).json()));
export const bytes = path => get("b:" + path, () => fetch(BASE + path).then(r => ok(path, r).arrayBuffer()).then(b => new Uint8Array(b)));
export const f32 = path => get("f:" + path, () => fetch(BASE + path).then(r => ok(path, r).arrayBuffer()).then(b => new Float32Array(b)));
export const image = path => get("i:" + path, async () => { const im = new Image(); im.src = BASE + path; await im.decode(); return im; });

/** Per-run files, fetched once when the run is first shown. */
export function runData(run) {
  const F = run.files;
  return get("run:" + run.name, async () => {
    const [curves, neurons, readout, sim, umap] = await Promise.all([
      json(fileOf(F.curves)), json(fileOf(F.neurons)), json(fileOf(F.readout)), bytes(F.similarity.file), f32(F.umap.file)]);
    return { curves, neurons, readout, sim, umap };
  });
}

/** Draw tile i of a padded sprite ({tile_w, tile_h, pad, ncol}) into a res x res canvas. The exported tiles are
 *  block-constant upscales, so sampling at the native resolution recovers the pixels; CSS enlarges without smoothing. */
export function spriteCanvas(img, layout, i, px, res = 14) {
  const cv = document.createElement("canvas"); cv.width = res; cv.height = res;
  const c = cv.getContext("2d"); c.imageSmoothingEnabled = false;
  const { tile_w: tw, tile_h: th, pad, ncol } = layout;
  const x = pad + (i % ncol) * (tw + pad), y = pad + Math.floor(i / ncol) * (th + pad);
  c.drawImage(img, x, y, tw, th, 0, 0, res, res);
  Object.assign(cv.style, { width: px ? px + "px" : "100%", aspectRatio: "1", imageRendering: "pixelated", display: "block", borderRadius: "3px", boxShadow: `0 0 0 1px ${P.line}` });
  return cv;
}

/** A 14x14 canvas from a Float32Array through a palette LUT on [lo, hi]. */
export function arrayCanvas(a, lutTable, lo, hi, px, w = 14) {
  const cv = document.createElement("canvas"); cv.width = w; cv.height = w;
  const c = cv.getContext("2d"), im = c.createImageData(w, w), d = im.data, k = 255 / ((hi - lo) || 1);
  for (let p = 0; p < w * w; p++) { const q = Math.max(0, Math.min(255, Math.round((a[p] - lo) * k))) * 4; d[4 * p] = lutTable[q]; d[4 * p + 1] = lutTable[q + 1]; d[4 * p + 2] = lutTable[q + 2]; d[4 * p + 3] = 255; }
  c.putImageData(im, 0, 0);
  Object.assign(cv.style, { width: px ? px + "px" : "100%", aspectRatio: "1", imageRendering: "pixelated", display: "block", borderRadius: "3px", boxShadow: `0 0 0 1px ${P.line}` });
  return cv;
}

/** Decode snapshot k of H Sigma H^T (uint8 [S, n, n], value = -a + 2a*byte/255). */
export function decodeSim(u8, k, n, a) {
  const out = new Float32Array(n * n), off = k * n * n, s = 2 * a / 255;
  for (let i = 0; i < n * n; i++) out[i] = -a + s * u8[off + i];
  return out;
}

// ------------------------------------------------------------------ run naming and selection
export const LIFTS = [
  { value: "pixels", label: "pixels 14×14", short: "pixels" },
  { value: "rff64", label: "RFF D = 64", short: "RFF 64" },
  { value: "rff256", label: "RFF D = 256", short: "RFF 256" },
  { value: "rff1024", label: "RFF D = 1024", short: "RFF 1024" },
];
export const initKey = r => (r.init === "generic" ? "generic" : "heavy");
export const omega = r => Math.round(r.wmin * r.N * 1000) / 1000;
export const liftLabel = k => ({ pixels: "pixels 14×14", rff64: "RFF D = 64", rff256: "RFF D = 256", rff1024: "RFF D = 1024", "rff256bw0.5": "RFF D = 256, bandwidth ×0.5", rff256bw2: "RFF D = 256, bandwidth ×2" }[k] || k);
export const initLabel = r => (initKey(r) === "generic" ? "generic Exp init" : "heavy-tailed init");
export function runLabel(r) {
  const bits = [liftLabel(r.lift), `N = ${ui.fmtInt(r.N)}`, `θ = ${r.theta}`, initLabel(r)];
  if (r.mode === "mf") bits.push("mean-field"); else bits.push("minibatch 256");
  if (omega(r) !== 0.5) bits.push(`ω = ${omega(r)}`);
  if (r.seed) bits.push(`seed ${r.seed}`);
  return bits.join(", ");
}

// ------------------------------------------------------------------ numbers quoted in the prose
const range = xs => { const v = xs.filter(Number.isFinite); return v.length ? [Math.min(...v), Math.max(...v)] : [NaN, NaN]; };
export const f2 = x => (Number.isFinite(x) ? ui.fmtFixed(x, 2) : "n/a");
export const rg = ([a, b], f = f2) => (!Number.isFinite(a) ? "n/a" : f(a) === f(b) ? f(a) : `${f(a)}–${f(b)}`);

export function summarise(index) {
  const R = index.runs, by = Object.fromEntries(R.map(r => [r.name, r]));
  const H = by[index.headline], S = {};
  const live = r => r.final.n_live_clusters;
  S.nRuns = R.length;
  S.nFeatured = R.filter(r => r.featured).length;
  // headline structure
  const hs = H.cluster_sizes.map((s, k) => ({ s, e: H.cluster_energy[k], dead: H.dead_clusters.includes(k) }));
  const hl = hs.filter(c => !c.dead);
  S.hLive = live(H); S.hDead = H.final.n_dead; S.hN = H.N;
  S.hSizes = hl.map(c => c.s).sort((a, b) => b - a);
  S.hEnergy = range(hl.map(c => c.e));
  S.hSizeRatio = Math.max(...S.hSizes) / Math.min(...S.hSizes);
  S.hWithin = H.final.within_l1;
  // most live parts in any run (seed 0, sgd, omega 0.5)
  S.maxLiveSgd = Math.max(...R.filter(r => r.mode === "sgd").map(live));
  S.maxLiveAll = Math.max(...R.map(live));
  // RFF: number of major live parts
  const rff = R.filter(r => r.lift.startsWith("rff") && live(r) > 0);
  S.rffMajor = range(rff.map(r => r.final.n_live_major));
  // readout
  const ro = r => r.readout.ls_both, bl = r => r.baselines;
  S.hRo = ro(H); S.hBl = bl(H); S.hSupport = H.support; S.hRank = H.measurement_rank;
  const liveRuns = R.filter(r => live(r) > 0);
  S.roCosRange = range(liveRuns.map(r => ro(r).contrast_cos));
  S.roBwRange = range(liveRuns.map(r => ro(r).contrast_bw));
  S.blBwRange = range(liveRuns.flatMap(r => [bl(r).identity.contrast_bw, bl(r).diag.contrast_bw]));
  const r1024 = R.find(r => r.featured && r.lift === "rff1024");
  S.r1024 = r1024 ? { ls: ro(r1024).contrast_cos, diag: bl(r1024).diag.contrast_cos, id: bl(r1024).identity.contrast_cos, sup: r1024.support.ls_both?.cos_on_support } : null;
  const fpix = R.filter(r => r.featured && r.lift === "pixels" && live(r) > 0);
  S.supPix = range(fpix.map(r => r.support.ls_both?.cos_on_support));
  S.naiveWorse = liveRuns.filter(r => r.readout.naive_ib.contrast_cos < ro(r).contrast_cos).length;
  S.nLive = liveRuns.length;
  S.untrainedEqId = liveRuns.filter(r => Math.abs(r.readout.init_ls_both.contrast_cos - bl(r).identity.contrast_cos) < 0.02).length;
  return S;
}

/** Fill <span data-n="key"> placeholders in the article with values from S. */
export function fillNumbers(root, S) {
  const F = {
    "h-live": () => String(S.hLive), "h-dead": () => ui.fmtInt(S.hDead), "h-N": () => ui.fmtInt(S.hN),
    "h-sizes": () => S.hSizes.join(", "), "h-energy": () => rg(S.hEnergy), "h-ratio": () => `${Math.round(S.hSizeRatio)}×`,
    "h-within": () => ui.fmt(S.hWithin, 2),
    "n-runs": () => String(S.nRuns), "n-feat": () => String(S.nFeatured), "max-live": () => String(S.maxLiveSgd), "max-live-all": () => String(S.maxLiveAll),
    "ro-range": () => rg(S.roCosRange), "ro-bw": () => rg(S.roBwRange), "bl-bw": () => rg(S.blBwRange),
    "h-ro": () => f2(S.hRo.contrast_cos), "h-diag": () => f2(S.hBl.diag.contrast_cos), "h-id": () => f2(S.hBl.identity.contrast_cos),
    "h-rk": () => f2(S.hBl.rank_k_truncation.contrast_cos),
    "h-known": () => f2(S.hRo.known_relfro), "h-known-id": () => f2(S.hBl.identity.known_relfro), "h-known-diag": () => f2(S.hBl.diag.known_relfro), "h-known-rk": () => f2(S.hBl.rank_k_truncation.known_relfro),
    "r1024-ls": () => f2(S.r1024?.ls), "r1024-diag": () => f2(S.r1024?.diag),
    "h-sup-cos": () => f2(S.hSupport.ls_both.cos_on_support), "h-sup-n": () => String(S.hSupport.n_supported),
    "h-sup-mass": () => ui.fmtPct(S.hSupport.ls_both.frac_C_mass_on_support_block), "h-sig-mass": () => ui.fmtPct(S.hSupport.sigma_contrast_mass_on_support_block),
    "h-pe": () => ui.fmtPct(S.hSupport.ls_both.part_energy_relerr_median, 1),
    "sup-pix": () => rg(S.supPix),
    "h-rank": () => String(S.hRank.both.rank), "h-rank-ii": () => String(S.hRank.ii.rank), "h-rank-ib": () => String(S.hRank.ib.rank), "h-dim": () => ui.fmtInt(S.hRank.dim_sym),
    "naive-worse": () => `${S.naiveWorse} of ${S.nLive}`,
  };
  root.querySelectorAll("[data-n]").forEach(el => { const f = F[el.dataset.n]; if (f) el.textContent = f(); });
}
