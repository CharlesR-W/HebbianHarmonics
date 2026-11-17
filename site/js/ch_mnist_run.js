// Chapter "mnist": the training explorer. Run picker (lift, init, theta; N is 2048 for every fully exported run),
// one snapshot slider driving H Sigma H^T, the Hellinger-UMAP map and the neuron tiles, and the training curves.
import * as ui from "./ui.js";
import { heatmap, scatter, lineChart } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as D from "./ch_mnist_data.js";

const ICON_PLAY = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4.5 2.6v10.8L13 8z" fill="currentColor"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="3.5" y="2.8" width="3.2" height="10.4" rx=".8" fill="currentColor"/><rect x="9.3" y="2.8" width="3.2" height="10.4" rx=".8" fill="currentColor"/></svg>';

/** Colour of cluster rank `rank` (dead pool or past the 8 categorical colours: other). */
export const clusterColor = (rank, dead) => (dead || rank >= 8 ? P.other : catColor(rank));
export const clusterName = c => (c.dead ? "dead pool" : `c${c.rank + 1}`);

/** Heatmap groups (left band) from the displayed neurons' final clusters. */
export function groupsOf(nj) {
  const N = nj.neurons, n = N.ids.length, g = [];
  let s = 0;
  for (let i = 1; i <= n; i++) {
    if (i === n || N.cluster[i] !== N.cluster[s]) { g.push({ start: s, end: i, color: clusterColor(N.cluster_rank[s], N.dead[s]), label: "" }); s = i; }
  }
  return g;
}

export function runFigure(host, capHost, index, state, ctx) {
  host.replaceChildren();
  const F = state.featured;
  // ---- picker
  const top = ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "12px 28px", alignItems: "flex-end", marginBottom: "12px" } });
  const segLift = ui.segmented({ label: "lift", options: D.LIFTS.map(l => ({ value: l.value, label: l.label, title: F.some(r => r.lift === l.value) ? "" : "no full export for this lift; see the sweep above" })), value: state.run.lift,
    onChange: v => state.select(state.find(v, D.initKey(state.run), state.run.theta)) });
  D.LIFTS.forEach(l => segLift.setDisabled(l.value, !F.some(r => r.lift === l.value)));
  const segInit = ui.segmented({ label: "initial input weights", options: [{ value: "generic", label: "generic Exp" }, { value: "heavy", label: "heavy-tailed" }], value: D.initKey(state.run),
    onChange: v => state.select(state.find(state.run.lift, v, state.run.theta)) });
  const thetaBox = ui.h("div");
  let segTheta = null;
  const nNote = ui.h("div", { class: "u-field" }, ui.h("span", { class: "u-name", text: "neurons" }), ui.h("span", { style: { font: "500 13px/2 var(--sans)" }, html: "$N = 2048$ (all full exports)" }));
  top.append(segLift.el, segInit.el, thetaBox, nNote);
  ui.renderMath(nNote);
  const stats = ui.stats({ cols: 4, items: [
    { key: "live", label: "live parts", fmt: String },
    { key: "dead", label: "dead pool (neurons)", fmt: ui.fmtInt },
    { key: "ro", label: "readout, contrast-subspace cos", fmt: D.f2 },
    { key: "res", label: "mean residual |dw/dt|", fmt: v => ui.fmt(v, 2) },
  ] });
  Object.assign(stats.el.style, { marginBottom: "14px", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))" });
  // ---- snapshot bar
  let data = null, run = null, hm = null, sc = null, curves = [], tiles = null, token = 0, uRange = null;
  const play = ui.h("button", { type: "button", class: "u-play", "aria-label": "Play through training" });
  let S = 33, k = Math.max(0, ctx.param("snap", 32));
  const snap = ui.slider({ label: "training step", min: 0, max: S - 1, step: 1, value: Math.min(k, S - 1), fmt: i => stepLabel(i),
    onInput: i => { k = i; draw(); }, onChange: i => ctx.setParams({ snap: i }) });
  const tbar = ui.h("div", { style: { display: "flex", gap: "12px", alignItems: "center", flex: "1 1 340px", maxWidth: "520px" } }, play, ui.h("div", { style: { flex: 1 } }, snap.el));
  // ---- panels
  const hmBox = ui.h("div", { style: { minWidth: 0 } }), mapBox = ui.h("div", { style: { minWidth: 0 } }), tileBox = ui.h("div", { style: { minWidth: 0 } });
  const panelHead = html => ui.h("div", { class: "chart-title", html });
  const tileHead = panelHead("");
  const panels = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(290px, 1fr))", gap: "14px 26px", alignItems: "start" } },
    ui.h("div", { style: { minWidth: 0 } }, panelHead("voltage covariance $h_i^\\top\\Sigma_g h_j$"), hmBox),
    ui.h("div", { style: { minWidth: 0 } }, panelHead("2D map of the embeddings (Hellinger UMAP)"), mapBox),
    ui.h("div", { style: { minWidth: 0 } }, tileHead, tileBox));
  const curveGrid = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "8px 20px", marginTop: "16px" } });
  const curveBoxes = ["clusters", "neurons and synapses at the floor", "stationarity residual |dw/dt|", "readout of the instantaneous weights"].map(t => {
    const b = ui.h("div", { style: { minWidth: 0 } }); curveGrid.append(ui.h("div", { style: { minWidth: 0 } }, panelHead(t), b)); return b;
  });
  const row2 = ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "10px 24px", alignItems: "center", marginBottom: "12px" } }, ui.h("div", { style: { flex: "2 1 520px", minWidth: 0 } }, stats.el), tbar);
  stats.el.style.marginBottom = "0";
  host.append(top, row2, panels, curveGrid);
  ui.renderMath(panels);

  function stepLabel(i) { const st = run?.snapshot_steps; return st ? `${ui.fmtInt(st[i])}  (t = ${ui.fmt(st[i] * run.dt, 2)})` : String(i); }

  function buildTheta() {
    const opts = F.filter(r => r.lift === state.run.lift && D.initKey(r) === D.initKey(state.run)).map(r => r.theta).sort((a, b) => a - b);
    segTheta = ui.segmented({ label: "$\\theta$", options: opts.map(t => ({ value: t, label: String(t) })), value: state.run.theta,
      onChange: v => state.select(state.find(state.run.lift, D.initKey(state.run), v)) });
    thetaBox.replaceChildren(segTheta.el);
    ["generic", "heavy"].forEach(v => segInit.setDisabled(v, !F.some(r => r.lift === state.run.lift && D.initKey(r) === v)));
  }

  async function load(r) {
    const my = ++token;
    run = r;
    segLift.set(r.lift); segInit.set(D.initKey(r)); buildTheta();
    const f = r.final;
    stats.update({ live: f.n_live_clusters, dead: f.n_dead, ro: r.readout.ls_both.contrast_cos, res: f.res_mean });
    stats.set("live", f.n_live_clusters, `${f.n_live_major} with ≥ 1% of N`);
    stats.set("dead", f.n_dead, ui.fmtPct(f.n_dead / r.N, 0) + " of N");
    stats.set("ro", r.readout.ls_both.contrast_cos, `diag Σ ${D.f2(r.baselines.diag.contrast_cos)} · identity ${D.f2(r.baselines.identity.contrast_cos)}`);
    stats.set("res", f.res_mean, `max ${ui.fmt(f.res_max, 2)}`);
    [hmBox, mapBox, tileBox].forEach(b => b.replaceChildren(ui.h("p", { class: "mute", text: "loading…" })));
    const d = await D.runData(r);
    const tileImg = r.files.neuron_tiles ? await D.image(r.files.neuron_tiles.png) : null;
    const staImg = !r.files.neuron_tiles && r.files.cluster_sta ? await D.image(r.files.cluster_sta.png) : null;
    if (my !== token) return;
    data = { ...d, tileImg, staImg };
    S = r.snapshot_steps.length; k = Math.min(k, S - 1);
    snap.input.max = S - 1; snap.set(k);
    // fixed map frame over all snapshots
    const u = d.umap; let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < u.length; i += 2) { x0 = Math.min(x0, u[i]); x1 = Math.max(x1, u[i]); y0 = Math.min(y0, u[i + 1]); y1 = Math.max(y1, u[i + 1]); }
    const px = (x1 - x0) * 0.04, py = (y1 - y0) * 0.04; uRange = [[x0 - px, x1 + px], [y0 - py, y1 + py]];
    build();
    caption();
  }

  function build() {
    const r = run, d = data, n = r.n_shown, nj = d.neurons, N = nj.neurons;
    const clusters = Object.fromEntries(nj.clusters.map(c => [c.id, c]));
    const a = r.files.similarity.a;
    hm?.destroy(); sc?.destroy();
    hm = heatmap(hmBox, { values: D.decodeSim(d.sim, k, n, a), rows: n, cols: n, map: "div", range: [-a, a], size: 280, groups: { rows: groupsOf(nj) },
      zlabel: "$h_i^\\top\\Sigma_g h_j$", xlabel: "neuron j (final clusters by size)", ylabel: "neuron i",
      tooltip: (i, j, v) => ui.tipRows(`step ${ui.fmtInt(r.snapshot_steps[k])}`, [
        { name: `neuron ${N.ids[i]} (${clusterName(clusters[N.cluster[i]])})`, value: "", color: clusterColor(N.cluster_rank[i], N.dead[i]) },
        { name: `neuron ${N.ids[j]} (${clusterName(clusters[N.cluster[j]])})`, value: "", color: clusterColor(N.cluster_rank[j], N.dead[j]) },
        { name: "hᵢᵀΣg hⱼ", value: ui.fmtFixed(v, 3) }]) });
    const cols = N.ids.map((_, i) => clusterColor(N.cluster_rank[i], N.dead[i]));
    sc = scatter(mapBox, { x: frame(0), y: frame(1), color: cols, size: N.dead.map(x => (x ? 2.5 : 3.6)), equal: true, height: 280,
      xrange: uRange[0], yrange: uRange[1], xlabel: "UMAP 1", ylabel: "UMAP 2",
      tooltip: i => { const c = clusters[N.cluster[i]]; return ui.tipRows(`neuron ${N.ids[i]}`, [
        { name: "final cluster", value: clusterName(c), color: cols[i] }, { name: "cluster size", value: ui.fmtInt(c.size) },
        { name: "part energy hᵀΣg h", value: ui.fmtFixed(c.energy, 3) }, { name: "preferred digit", value: String(N.preferred_digit[i]) }]); } });
    // legend under the map: up to 8 live clusters + dead pool
    const live = nj.clusters.filter(c => !c.dead), dead = nj.clusters.find(c => c.dead);
    const items = live.slice(0, 8).map(c => ({ label: `c${c.rank + 1} · ${ui.fmtInt(c.size)}`, color: clusterColor(c.rank, false) }));
    if (live.length > 8) items.push({ label: `${live.length - 8} smaller parts`, color: P.other });
    if (dead) items.push({ label: `dead pool · ${ui.fmtInt(dead.size)}`, color: P.other, shape: "dot" });
    const lg = ui.legend({ items }); lg.el.classList.add("chart-legend"); lg.el.style.marginTop = "4px"; mapBox.append(lg.el);
    // tiles
    tileHead.innerHTML = d.tileImg ? "16 neurons' embeddings $h_i$" : "the parts, by their response";
    ui.renderMath(tileHead);
    if (d.tileImg) {
      tiles = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 58px))", gap: "5px" } });
      tileBox.replaceChildren(tiles, ui.h("div", { class: "u-help", style: { marginTop: "8px", maxWidth: "260px" }, text: "16 neurons spread through the display order; swatch: final cluster." }));
    } else {
      tiles = null;
      const lay = r.files.cluster_sta.layout, ids = r.files.cluster_sta.clusters;
      const row = ui.h("div", { style: { display: "flex", gap: "10px" } });
      ids.forEach((cid, i) => { const c = clusters[cid]; row.append(ui.h("div", { style: { width: "84px", font: "11.5px/1.3 var(--sans)", color: P.ink2 } }, D.spriteCanvas(d.staImg, lay, i), ui.h("div", { style: { marginTop: "3px" }, text: `${clusterName(c)} · ${ui.fmtInt(c.size)}` }))); });
      tileBox.replaceChildren(ui.h("p", { class: "u-help", style: { margin: "0 0 8px", maxWidth: "300px" }, html: `With random Fourier features $h_i$ is a vector over ${r.B} features, not an image. Each part is shown instead by its response-weighted average stimulus (final weights; red: brighter than the mean digit):` }), row);
      ui.renderMath(tileBox);
    }
    // curves
    curves.forEach(c => c.destroy());
    const cv = d.curves, x = cv.step.map(s => s + 1), common = { x, xlog: true, height: 150, xlabel: "training step + 1", xname: "step + 1", xfmt: v => ui.fmt(v, 2) };
    const frac = arr => arr.map(v => v / r.N);
    curves = [
      lineChart(curveBoxes[0], { ...common, ylog: true, series: [{ name: "live clusters", y: cv.n_live_clusters }, { name: "live, ≥ 1% of N", y: cv.n_live_major, dash: true }], yfmt: v => ui.fmt(v, 2) }),
      lineChart(curveBoxes[1], { ...common, yrange: [0, 1], yfmt: v => ui.fmtPct(v, 0), series: [{ name: "dead neurons / N", y: frac(cv.n_dead) }, { name: "input synapses at floor", y: cv.floor_ib }, { name: "lateral synapses at floor", y: cv.floor_ii }] }),
      lineChart(curveBoxes[2], { ...common, ylog: true, yfmt: v => ui.fmt(v, 1), series: [{ name: "mean", y: cv.res_mean }, { name: "mean over active edges", y: cv.res_active_mean }, { name: "max", y: cv.res_max, dash: true }] }),
      lineChart(curveBoxes[3], { ...common, yrange: [0, 1], yfmt: D.f2, series: [{ name: "contrast-subspace cos", y: cv.ro_both_contrast_cos }, { name: "scale-invariant cos", y: cv.ro_both_scale_cos },
        { name: `final, averaged weights (${D.f2(r.readout.ls_both.contrast_cos)})`, x: [1, x[x.length - 1]], y: [r.readout.ls_both.contrast_cos, r.readout.ls_both.contrast_cos], color: P.fg, dash: true, width: 1.2 },
        { name: `diag Σg (${D.f2(r.baselines.diag.contrast_cos)})`, x: [1, x[x.length - 1]], y: [r.baselines.diag.contrast_cos, r.baselines.diag.contrast_cos], color: P.mute2, dash: true, width: 1.2 }] }),
    ];
    draw();
  }
  function frame(c) { const n = run.n_shown, u = data.umap, off = k * n * 2, out = new Float32Array(n); for (let i = 0; i < n; i++) out[i] = u[off + 2 * i + c]; return out; }

  function draw() {
    if (!data) return;
    const r = run, n = r.n_shown;
    hm.update({ values: D.decodeSim(data.sim, k, n, r.files.similarity.a) });
    sc.update({ x: frame(0), y: frame(1) });
    if (tiles) {
      const lay = r.files.neuron_tiles.layout, rows = r.files.neuron_tiles.neurons, N = data.neurons.neurons;
      tiles.replaceChildren(...rows.map((pos, row) => {
        const w = ui.h("div", { style: { position: "relative" } }, D.spriteCanvas(data.tileImg, lay, row * lay.ncol + k));
        w.append(ui.h("i", { class: "ig-sw", style: { position: "absolute", left: 0, right: 0, bottom: "-3px", height: "2px", borderRadius: "1px", background: clusterColor(N.cluster_rank[pos], N.dead[pos]) } }));
        return w;
      }));
    }
    curves.forEach(c => c.setCursor(r.snapshot_steps[k] + 1));
  }

  function caption() {
    const r = run, tilesTxt = r.files.neuron_tiles
      ? "Right: the embeddings $h_i$ of 16 displayed neurons at this step as 14×14 images over the pixels, each tile on its own maximum (exported colour map: light = 0, dark = the tile's maximum), enlarged without smoothing; swatch: final cluster."
      : "Right: each final cluster's response-weighted average stimulus $\\mathbb E[r(x)(x - \\bar x)]/\\mathrm{sd}(r)$, $r(x) = c_k^\\top(g(x) - \\mu_g)$ over the test digits (red: brighter than average, each tile on its own symmetric scale).";
    D.caption(capHost,
      `at the chosen step, $h_i^\\top\\Sigma_g h_j$ of ${r.n_shown} displayed neurons in one fixed order (diverging map, one range for the whole run), the same neurons in a Hellinger-metric UMAP, ${r.files.neuron_tiles ? "16 neurons' $h_i$ as images" : "the parts' response-weighted average stimuli"}, and training curves against step + 1 (accent line: the chosen step). Colour everywhere: final cluster (grey: the dead pool, or past the eighth cluster). Stat tiles: final, averaged weights.`,
      `Left: rows and columns in one fixed order (final clusters by size, dead pool last; within a cluster by distance to its centroid), diverging map on a fixed symmetric range for the whole run; left band: final cluster. Middle: a 2D UMAP of the embeddings with the Hellinger metric (Euclidean distance between $\\sqrt{h_i}$), computed at the final step and warm-started backwards, each frame Procrustes-aligned to the next; axes fixed over the run. ${tilesTxt} Curves (log step axis): live clusters (all clusters except the dead pool, so every neuron is its own cluster at the start) and those with ≥ 1% of $N$; the fraction of neurons in the dead pool and of synapses at the floor; the stationarity residual, the projected mean-field drift $|dw/dt|$ at the snapshot weights; the masked least-squares readout of the instantaneous weights (contrast-subspace and scale-invariant cosines; these violate the equilibrium assumption before convergence), with dashed lines for the final readout on weights averaged over the last 600 steps and for diag $\\Sigma_g$.`,
      `run <code>${r.name}</code>: ${D.runLabel(r)}; $\\omega = N w_{\\min} = ${D.omega(r)}$, $dt = ${r.dt}$, ${ui.fmtInt(r.steps)} steps; $\\theta$ at the ${ui.fmtPct(r.theta_percentile, 1)} percentile of off-diagonal $\\Sigma_g$.`);
  }

  // play: one snapshot every 0.45 s while on screen
  let anim = null, acc = 0;
  const setIcon = () => { play.innerHTML = anim ? ICON_PAUSE : ICON_PLAY; };
  const stop = () => { anim?.stop(); anim = null; setIcon(); };
  play.addEventListener("click", () => {
    if (anim) { stop(); ctx.setParams({ snap: k }); return; }
    if (k >= S - 1) { k = 0; snap.set(0); draw(); }
    acc = 0;
    anim = ui.animate(host, dt => { acc += dt; if (acc < 0.45) return; acc = 0; if (k >= S - 1) { stop(); ctx.setParams({ snap: k }); return; } k += 1; snap.set(k); draw(); });
    setIcon();
  });
  setIcon();
  ctx.onHide(stop);
  ctx.onParams(p => { const s = parseInt(p.get("snap"), 10); if (Number.isFinite(s) && s !== k && s >= 0 && s < S) { k = s; snap.set(s); draw(); } });

  state.onRun(r => load(r));
  load(state.run);
}
