// Chapter "mnist": what the neurons became (parts as images, response-weighted averages, top digits, tuning,
// eigen-images of Sigma_g), the learned graph (UMAP or Tutte layout) and the allocation (size vs energy).
import * as ui from "./ui.js";
import { barChart, svgEl, scale, niceTicks, logTicks, hidpi } from "./charts.js";
import { P, lut } from "./palette.js";
import * as D from "./ch_mnist_data.js";
import { clusterColor, clusterName } from "./ch_mnist_run.js";

const small = { font: "11.5px/1.3 var(--sans)", color: "var(--ink-2)" };

// ------------------------------------------------------------------ parts as images
export function partsFigure(host, capHost, index, state, ctx) {
  let token = 0;
  async function load(r) {
    const my = ++token;
    host.replaceChildren(ui.h("p", { class: "mute", text: "loading…" }));
    const L = index.lifts[r.lift], F = r.files, pix = r.lift === "pixels";
    const eigFile = pix ? L.files.eigen_images : L.files.eigen_sta;
    const [d, parts, sta, top, eig, eigTop] = await Promise.all([D.runData(r), pix ? D.f32(F.cluster_part_arrays.file) : null, D.image(F.cluster_sta.png),
      D.image(F.cluster_top_digits.png), D.image(eigFile.png), D.image(L.files.eigen_top_digits.png)]);
    if (my !== token) return;
    const nj = d.neurons, clusters = nj.clusters, B = r.B;
    const staPos = Object.fromEntries(F.cluster_sta.clusters.map((id, i) => [id, i]));
    const topPos = Object.fromEntries(F.cluster_top_digits.clusters.map((id, i) => [id, i]));
    const tuneMax = Math.max(1, ...clusters.flatMap(c => c.tuning.map(Math.abs)), ...L.eigen_tuning.flat().map(Math.abs));
    const tuneRange = [-Math.ceil(tuneMax * 2) / 2, Math.ceil(tuneMax * 2) / 2];
    const seqLut = lut("seq");

    // layout: cards (left) and the detail panel (right)
    const cards = ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "10px 6px" } });
    const detail = ui.h("div", { style: { minWidth: 0, flex: "1 1 250px", maxWidth: "340px" } });
    const eigRow = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(8, minmax(0, 48px))", gap: "5px" } });
    const left = ui.h("div", { style: { minWidth: 0, flex: "1 1 520px" } },
      ui.h("div", { class: "chart-title", text: `Learned parts (${r.final.n_live_clusters} live clusters and ${r.final.n_dead ? "the dead pool" : "no dead pool"}), final weights` }),
      ui.h("div", { class: "u-help", style: { margin: "0 0 8px" }, html: pix ? "Top tile: the part $h_k$ over the 14×14 pixels. Bottom tile: its response-weighted average stimulus." : "Each tile: the part's response-weighted average stimulus (the part itself is a vector over features)." }),
      cards,
      ui.h("div", { class: "chart-title", style: { marginTop: "18px" }, text: pix ? "Top 16 eigenvectors of PΣgP, as images" : "Top 16 eigen-directions of PΣgP, as response-weighted average stimuli" }),
      eigRow);
    host.replaceChildren(ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "18px 32px", alignItems: "flex-start" } }, left, detail));
    ui.renderMath(left);

    const tile = (el, sel) => { el.style.cursor = "pointer"; el.addEventListener("pointerenter", sel); el.addEventListener("click", sel); el.addEventListener("focus", sel); el.tabIndex = 0; return el; };
    let selEl = null;
    const mark = el => { if (selEl) selEl.style.outline = ""; selEl = el; if (el) { el.style.outline = `2px solid ${P.fg}`; el.style.outlineOffset = "2px"; el.style.borderRadius = "4px"; } };

    clusters.forEach((c, ci) => {
      const col = clusterColor(c.rank, c.dead), w = pix ? 50 : 70;
      const card = ui.h("div", { style: { width: w + "px", display: "grid", gap: "3px" } });
      if (pix) {
        const a = parts.subarray(ci * B, (ci + 1) * B); let mx = 0; for (const v of a) mx = Math.max(mx, v);
        card.append(D.arrayCanvas(a, seqLut, 0, mx || 1));
      }
      card.append(D.spriteCanvas(sta, F.cluster_sta.layout, staPos[c.id]));
      card.append(ui.h("div", { style: { height: "3px", borderRadius: "2px", background: col } }));
      card.append(ui.h("div", { style: { ...small, fontVariantNumeric: "tabular-nums", color: c.dead ? P.mute : P.ink2 }, html: `<b>${clusterName(c)}</b><br>${ui.fmtInt(c.size)}` }));
      cards.append(tile(card, () => { mark(card); showCluster(c); }));
    });
    for (let i = 0; i < 16; i++) {
      const cell = ui.h("div", { style: { display: "grid", gap: "2px" } }, D.spriteCanvas(eig, eigFile.layout, i), ui.h("div", { style: { ...small, textAlign: "center" }, text: String(i + 1) }));
      eigRow.append(tile(cell, () => { mark(cell); showEig(i); }));
    }

    function digitsRow(img, lay, row) {
      const g = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(8, minmax(0, 36px))", gap: "3px" } });
      for (let j = 0; j < 8; j++) g.append(D.spriteCanvas(img, lay, row * lay.ncol + j, null, 28));
      return g;
    }
    function panel(title, sub, img, lay, row, tuning) {
      const bars = ui.h("div");
      detail.replaceChildren(
        ui.h("div", { class: "chart-title", html: title }), ui.h("div", { class: "u-help", style: { margin: "0 0 8px" }, html: sub }),
        ui.h("div", { style: { ...small, marginBottom: "4px" }, text: "8 test digits with the largest response" }), digitsRow(img, lay, row),
        ui.h("div", { style: { ...small, margin: "12px 0 2px" }, text: "mean z-scored response per digit class" }), bars);
      barChart(bars, { labels: [..."0123456789"], values: tuning, color: P.ink2, yrange: tuneRange, height: 130, yfmt: v => ui.fmt(v, 2), name: "mean z-response" });
      ui.renderMath(detail);
    }
    function showCluster(c) {
      const sub = c.dead ? `${ui.fmtInt(c.size)} neurons with every synapse at the floor, floating at the floor-weighted centroid; energy $h^\\top\\Sigma_g h = ${ui.fmtFixed(c.energy, 2)}$, far below $\\theta = ${r.theta}$. Not a learned part.`
        : `${ui.fmtInt(c.size)} neurons (clones); energy $h_k^\\top\\Sigma_g h_k = ${ui.fmtFixed(c.energy, 2)}$; ${pix ? `support ${c.support} pixels` : `effective support ${ui.fmt(c.entropy_eff, 2)} features`}; preferred digit ${c.preferred_digit}.`;
      panel(c.dead ? "The dead pool" : `Part ${clusterName(c)}`, sub, top, F.cluster_top_digits.layout, topPos[c.id], c.tuning);
    }
    function showEig(i) {
      panel(`Eigen-direction ${i + 1} of $P\\Sigma_g P$`, "Response $r(x) = v^\\top(g(x) - \\mu_g)$; the sign of an eigenvector is arbitrary.", eigTop, L.files.eigen_top_digits.layout, i, L.eigen_tuning[i]);
    }
    const first = cards.firstChild; mark(first); showCluster(clusters[0]);

    D.caption(capHost, `one card per final cluster (largest first; the dead pool last, a flagged cluster, not a learned part)${pix ? ": the part $h_k$ as a 14×14 image and" : ":"} its response-weighted average stimulus; the top 16 eigen-directions of $P\\Sigma_gP$; for the selected one, its top digits and digit tuning. Numbers: cluster size in neurons (all $N$).`,
      `${pix ? "Part $h_k$: the mean embedding of the cluster's members, a probability vector over the 196 pixels, as a 14×14 image, blue sequential map from 0 to the tile's own maximum. " : ""}Response-weighted average stimulus: $\\mathbb E[r(x)(x - \\bar x)]/\\mathrm{sd}(r)$ with $r(x) = h_k^\\top(g(x) - \\mu_g)$ over the 10,000 test digits, in pixel units, diverging map (red: brighter than the mean digit), each tile on its own symmetric scale. Colour bar: cluster (categorical by size; grey past the eighth and for the dead pool). Eigen-directions of $P\\Sigma_gP$ ($P = I - \\mathbf 1\\mathbf 1^\\top/B$), ${pix ? "as 14×14 images on their own symmetric scales (red +, blue −)" : "as response-weighted average stimuli"}. Right: the 8 test digits with the largest response, and the mean z-scored response per digit class (one range for the whole run). All tiles enlarged without smoothing.`,
      `run <code>${r.name}</code>; <code>data/mnist/runs/${r.name}/</code>, <code>data/mnist/lifts/${r.lift}/</code>.`);
  }
  state.onRun(load);
  load(state.run);
}

// ------------------------------------------------------------------ the learned graph
export function graphFigure(host, capHost, index, state, ctx) {
  let token = 0, layout = ctx.param("glay", "tutte"), gi = ctx.param("gstep", 5);
  async function load(r) {
    const my = ++token;
    host.replaceChildren(ui.h("p", { class: "mute", text: "loading…" }));
    const G = r.files.graph, [d, E] = await Promise.all([D.runData(r), D.f32(G.file)]);
    if (my !== token) return;
    const pix = r.lift === "pixels", n = r.n_shown, N = d.neurons.neurons, snaps = G.snapshots;
    const offs = []; snaps.reduce((o, s) => { offs.push(o); return o + s.n_edges; }, 0);
    gi = Math.max(0, Math.min(snaps.length - 1, gi));
    let lay = pix ? layout : "umap";
    const segStep = ui.segmented({ label: "step", options: snaps.map((s, i) => ({ value: i, label: ui.fmtInt(s.step) })), value: gi, onChange: v => { gi = v; ctx.setParams({ gstep: v }); draw(); } });
    const segLay = ui.segmented({ label: "layout", options: [{ value: "tutte", label: "Tutte (image plane)" }, { value: "umap", label: "UMAP" }], value: lay,
      onChange: v => { layout = lay = v; ctx.setParams({ glay: v }); draw(); } });
    if (!pix) segLay.setDisabled("tutte", true);
    const cv = ui.h("canvas", { style: { display: "block", touchAction: "pan-y" } }), info = ui.h("div", { class: "u-help", style: { marginTop: "6px" } });
    const box = ui.h("div", { class: "chart", style: { minWidth: 0 } }, cv);
    host.replaceChildren(ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "8px 18px", marginBottom: "8px" } }, segStep.el, segLay.el), box, info);
    // UMAP frame over all snapshots (as in the training figure)
    const u = d.umap; let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < u.length; i += 2) { x0 = Math.min(x0, u[i]); x1 = Math.max(x1, u[i]); y0 = Math.min(y0, u[i + 1]); y1 = Math.max(y1, u[i + 1]); }
    const cols = N.ids.map((_, i) => clusterColor(N.cluster_rank[i], N.dead[i]));
    let proj = null;
    function draw() {
      const W = Math.floor(box.clientWidth); if (W <= 0) return;
      const Hh = Math.min(W, 310), ctx2 = hidpi(cv, W, Hh), s = snaps[gi], pad = 14;
      ctx2.clearRect(0, 0, W, Hh);
      let X, Y, nodeXY;
      if (lay === "tutte") {
        const side = Math.min(W, Hh) - 2 * pad, ox = (W - side) / 2, sc = side / 14;
        X = v => ox + (v + 0.5) * sc; Y = v => pad + (v + 0.5) * sc;
        nodeXY = i => (i < n ? [X(s.tutte_xy[i][0]), Y(s.tutte_xy[i][1])] : [X(G.boundary_xy[i - n][0]), Y(G.boundary_xy[i - n][1])]);
      } else {
        const sx = (W - 2 * pad) / (x1 - x0), sy = (Hh - 2 * pad) / (y1 - y0), sc = Math.min(sx, sy), ox = (W - sc * (x1 - x0)) / 2, oy = (Hh - sc * (y1 - y0)) / 2;
        const off = s.umap_snapshot * n * 2;
        nodeXY = i => (i < n ? [ox + (u[off + 2 * i] - x0) * sc, Hh - oy - (u[off + 2 * i + 1] - y0) * sc] : null);
      }
      // edges bucketed by opacity (w / max w over the drawn edges)
      const o = offs[gi], m = s.n_edges; let wmax = 0, nII = 0, nIB = 0;
      for (let e = 0; e < m; e++) { const a = E[3 * (o + e)], b = E[3 * (o + e) + 1]; if (lay === "umap" && (a >= n || b >= n)) continue; wmax = Math.max(wmax, E[3 * (o + e) + 2]); }
      const buckets = Array.from({ length: 12 }, () => []);
      for (let e = 0; e < m; e++) {
        const a = E[3 * (o + e)], b = E[3 * (o + e) + 1], w = E[3 * (o + e) + 2];
        const ib = a >= n || b >= n; if (ib) nIB++; else nII++;
        if (lay === "umap" && ib) continue;
        const q = Math.min(11, Math.floor(w / (wmax || 1) * 12)); buckets[q].push(a, b);
      }
      ctx2.strokeStyle = P.edge || P.fg; ctx2.lineWidth = 0.8;
      buckets.forEach((bk, q) => { if (!bk.length) return; ctx2.globalAlpha = Math.max(0.03, (q + 1) / 12) * 0.55; ctx2.beginPath(); for (let t = 0; t < bk.length; t += 2) { const p1 = nodeXY(bk[t]), p2 = nodeXY(bk[t + 1]); ctx2.moveTo(p1[0], p1[1]); ctx2.lineTo(p2[0], p2[1]); } ctx2.stroke(); });
      ctx2.globalAlpha = 1;
      if (lay === "tutte") { // boundary pixels: small squares, ink outline
        ctx2.lineWidth = 0.8; ctx2.strokeStyle = P.fg; ctx2.fillStyle = P.surface;
        for (let b = 0; b < r.B; b++) { const [px, py] = nodeXY(n + b); ctx2.fillRect(px - 2, py - 2, 4, 4); ctx2.strokeRect(px - 2, py - 2, 4, 4); }
      }
      proj = new Float32Array(2 * n);
      const order = [...Array(n).keys()].sort((a, b) => (N.dead[b] - N.dead[a]));   // dead first, live on top
      for (const i of order) {
        const [px, py] = nodeXY(i); proj[2 * i] = px; proj[2 * i + 1] = py;
        ctx2.beginPath(); ctx2.arc(px, py, N.dead[i] ? 2.4 : 3.4, 0, 2 * Math.PI); ctx2.fillStyle = cols[i]; ctx2.fill(); ctx2.lineWidth = 0.8; ctx2.strokeStyle = P.surface; ctx2.stroke();
      }
      info.textContent = `Step ${ui.fmtInt(s.step)}: ${ui.fmtInt(m)} strongest active synapses of the displayed neurons (${ui.fmtInt(nII)} lateral, ${ui.fmtInt(nIB)} input)` + (lay === "umap" ? "; input synapses are not drawn in the UMAP layout (pixels are not placed in the map)." : ".");
    }
    cv.addEventListener("pointermove", ev => {
      if (!proj) return;
      const rc = cv.getBoundingClientRect(), x = ev.clientX - rc.left, y = ev.clientY - rc.top;
      let bi = -1, bd = 64; for (let i = 0; i < n; i++) { const dx = proj[2 * i] - x, dy = proj[2 * i + 1] - y, dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; bi = i; } }
      if (bi < 0) { ui.hideTip(); return; }
      const c = d.neurons.clusters.find(c => c.id === N.cluster[bi]);
      ui.showTip(ev.clientX, ev.clientY, ui.tipRows(`neuron ${N.ids[bi]}`, [{ name: "final cluster", value: clusterName(c), color: cols[bi] }, { name: "cluster size", value: ui.fmtInt(c.size) }]));
    });
    cv.addEventListener("pointerleave", () => ui.hideTip());
    new ResizeObserver(() => draw()).observe(box);
    draw();
    D.caption(capHost, `the ${n} displayed neurons (circles, colour: final cluster) and their strongest active synapses (ink, opacity $w/\\max w$)${pix ? ", in the image plane (Tutte) or in the UMAP map" : " in the UMAP map"}.`,
      `Active synapses: $w &gt; 2w_{\\min}$, at most 3,000 per step, strongest first; opacity relative to the largest drawn weight. ${pix ? "Tutte layout: each neuron at its harmonic position $\\sum_b h_{ib}\\,(\\text{column}_b, \\text{row}_b)$ in the 14×14 image plane, plus jitter (sd 0.12 pixel); squares: the 196 boundary pixels. " : ""}UMAP layout: the Hellinger-UMAP coordinates of that step (the frame of the training figure); lateral synapses only, since pixels are not placed in the map. Grey: the dead pool, or past the eighth cluster.`,
      `run <code>${r.name}</code>; <code>graph_edges.f32</code>.`);
  }
  state.onRun(load);
  load(state.run);
}

// ------------------------------------------------------------------ allocation: size against energy
export function allocFigure(host, capHost, index, state, ctx) {
  let token = 0;
  async function load(r) {
    const my = ++token;
    const d = await D.runData(r);
    if (my !== token) return;
    const cl = d.neurons.clusters, box = ui.h("div", { class: "chart", style: { minWidth: 0 } });
    const live = cl.filter(c => !c.dead), dead = cl.find(c => c.dead);
    const note = ui.h("div", { class: "u-help", style: { marginTop: "6px" } });
    host.replaceChildren(box, note);
    const es = cl.map(c => c.energy).concat([r.theta]), ss = cl.map(c => c.size);
    const xr = [Math.min(...es), Math.max(...es)], xp = (xr[1] - xr[0]) * 0.08 || 0.1; xr[0] -= xp; xr[1] += xp;
    const yr = [Math.max(0.8, Math.min(...ss) / 1.6), Math.max(...ss) * 1.6];
    function draw() {
      const W = Math.floor(box.clientWidth); if (W <= 0) return;
      box.replaceChildren();
      const Hh = 270, m = { l: 52, r: 14, t: 10, b: 38 };
      const svg = svgEl("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}` }, box);
      const X = scale("linear", xr[0], xr[1], m.l, W - m.r), Y = scale("log", yr[0], yr[1], Hh - m.b, m.t);
      const g = svgEl("g", { class: "grid" }, svg), ax = svgEl("g", { class: "axis" }, svg);
      const T = (x, y, s, a = {}) => { const t = svgEl("text", { x, y, ...a }, svg); t.textContent = s; return t; };
      logTicks(yr[0], yr[1]).forEach(v => { svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); T(m.l - 6, Y(v) + 3.5, ui.fmtInt(v), { "text-anchor": "end" }); });
      svgEl("line", { x1: m.l, x2: W - m.r, y1: Hh - m.b + .5, y2: Hh - m.b + .5 }, ax);
      niceTicks(xr[0], xr[1], Math.max(3, Math.round(W / 90))).forEach(v => { svgEl("line", { x1: X(v), x2: X(v), y1: Hh - m.b, y2: Hh - m.b + 4 }, ax); T(X(v), Hh - m.b + 15, ui.fmt(v, 2), { "text-anchor": "middle" }); });
      T((m.l + W - m.r) / 2, Hh - 4, "part energy hₖᵀΣg hₖ", { "text-anchor": "middle", class: "ttl" });
      T(0, 0, "neurons in the cluster", { "text-anchor": "middle", class: "ttl", transform: `translate(12,${(m.t + Hh - m.b) / 2}) rotate(-90)` });
      svgEl("line", { x1: X(r.theta), x2: X(r.theta), y1: m.t, y2: Hh - m.b, stroke: P.mute2, "stroke-dasharray": "3 3" }, svg);
      T(X(r.theta) + 4, m.t + 10, `θ = ${r.theta}`);
      const tip = c => ui.tipRows(clusterName(c), [{ name: "neurons", value: ui.fmtInt(c.size), color: clusterColor(c.rank, c.dead) }, { name: "energy hᵀΣg h", value: ui.fmtFixed(c.energy, 3) }, { name: "preferred digit", value: String(c.preferred_digit) }], c.dead ? "every synapse at the floor: not a learned part" : null);
      const hook = (el, c) => { el.addEventListener("pointermove", ev => ui.showTip(ev.clientX, ev.clientY, tip(c))); el.addEventListener("pointerleave", ui.hideTip); };
      if (dead) {
        const px = X(dead.energy), py = Y(dead.size);
        hook(svgEl("rect", { x: px - 5.5, y: py - 5.5, width: 11, height: 11, fill: P.other, stroke: P.fg, "stroke-width": 1 }, svg), dead);
        T(px + 9, py + 4, "dead pool", { style: "fill: var(--ink-2)" });
      }
      live.slice().reverse().forEach(c => hook(svgEl("circle", { cx: X(c.energy), cy: Y(c.size), r: 5.5, fill: clusterColor(c.rank, false), stroke: P.surface, "stroke-width": 1.2 }, svg), c));
    }
    new ResizeObserver(draw).observe(box);
    draw();
    const en = live.map(c => c.energy), sz = live.map(c => c.size);
    note.innerHTML = live.length > 1
      ? `Live parts: energy ${D.rg([Math.min(...en), Math.max(...en)])}, sizes ${ui.fmtInt(Math.min(...sz))}–${ui.fmtInt(Math.max(...sz))} neurons (${ui.fmt(Math.max(...sz) / Math.min(...sz), 2)}×).`
      : live.length === 1 ? `One live part holds ${ui.fmtInt(sz[0])} neurons: nothing to allocate.` : "No live parts.";
    D.caption(capHost, `one mark per final cluster: $x$ = energy $h_k^\\top\\Sigma_g h_k$ of its part (the mean embedding of its members), $y$ = its size in neurons (log axis; all $N$, <code>clusters[].size</code>). Circles: live parts (colour by size); grey square: the dead pool. Dashed: $\\theta$.`, null, `run <code>${r.name}</code>, final weights.`);
  }
  state.onRun(load);
  load(state.run);
}
