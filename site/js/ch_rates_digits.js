// Chapter "rates": figures built from the precomputed MNIST exports (parts across q, training film, samples).
import * as ui from "./ui.js";
import { heatmap } from "./charts.js";
import { P, catColor } from "./palette.js";
import * as D from "./ch_rates_data.js";

const RES = 14;

/** Draw tile i of a padded sprite (layout {tile_w, tile_h, pad, ncol}) into a RES x RES canvas (block-constant
 *  upscaled tiles, so nearest sampling recovers the pixels exactly); CSS enlarges it without smoothing. */
export function spriteCanvas(img, layout, i, px) {
  const cv = document.createElement("canvas"); cv.width = RES; cv.height = RES;
  const c = cv.getContext("2d"); c.imageSmoothingEnabled = false;
  const { tile_w: tw, tile_h: th, pad, ncol } = layout;
  const x = pad + (i % ncol) * (tw + pad), y = pad + Math.floor(i / ncol) * (th + pad);
  c.drawImage(img, x, y, tw, th, 0, 0, RES, RES);
  Object.assign(cv.style, { width: px ? px + "px" : "100%", aspectRatio: "1", imageRendering: "pixelated", display: "block", borderRadius: "3px", boxShadow: `0 0 0 1px ${P.line}` });
  return cv;
}

const fmt2 = x => (Number.isFinite(x) ? ui.fmtFixed(x, 2) : "n/a");

// ------------------------------------------------------------------ parts across q
export async function partsFigure(host, index) {
  const g1 = index.runs.filter(r => r.res === 14 && r.gamma === 1 && r.seed === 0);
  const ROWS = [
    { label: "linear rule", f: r => r.exp === "A_lin" },
    { label: "rectified, $\\tau = 0$", f: r => r.exp === "A_rect" && r.tau === 0 },
    { label: "rectified, $\\tau = 0.5$", f: r => r.exp === "A_rect" && r.tau === 0.5 },
    { label: "rectified, $\\tau = 1$", f: r => r.exp === "A_rect" && r.tau === 1 },
    { label: "Gaussian pixels, rectified $\\tau = 0$, own $q$", f: r => r.exp === "Bown_rect" && r.tau === 0 },
    { label: "Gaussian pixels, rectified $\\tau = 0$, digits' $\\theta$", f: r => r.exp === "B_rect" && r.tau === 0 },
  ];
  const narrow = host.clientWidth < 600;
  const grid = ui.h("div", { style: { display: "grid", gridTemplateColumns: narrow ? "repeat(5, minmax(0, 1fr))" : "minmax(150px, 190px) repeat(5, minmax(0, 1fr))", gap: narrow ? "4px 6px" : "8px 12px", alignItems: "start", maxWidth: "860px" } });
  host.replaceChildren(grid);
  const small = { font: "11.5px/1.3 var(--sans)", color: P.ink2 };
  if (!narrow) grid.append(ui.h("div"));
  D.Q_GRID.forEach(q => grid.append(ui.h("div", { style: { ...small, fontWeight: 600, textAlign: "center" }, text: `q = ${q.toFixed(3)}` })));
  for (const row of ROWS) {
    const lab = ui.h("div", { style: { font: "560 12.5px/1.35 var(--sans)", color: P.ink2, ...(narrow ? { gridColumn: "1 / -1", marginTop: "8px" } : { textAlign: "right", paddingTop: "22px" }) }, html: row.label });
    grid.append(lab);
    const runs = g1.filter(row.f);
    for (let k = 0; k < D.Q_GRID.length; k++) {
      const r = runs.find(x => D.qIndex(x.q) === k);
      const cell = ui.h("div", { style: { minWidth: 0, display: "flex", flexDirection: "column", alignItems: "center", gap: "3px" } });
      grid.append(cell);
      if (!r) { cell.append(ui.h("div", { style: small, text: "not run" })); continue; }
      const f = r.final, nParts = r.parts_final?.layout?.n || 0, main = Math.min(84, narrow ? 64 : 84);
      if (f.n_live_clusters === 0 || nParts === 0) {
        cell.append(ui.h("div", { style: { width: "100%", maxWidth: main + "px", aspectRatio: "1", background: P.wash, borderRadius: "3px", display: "grid", placeItems: "center", ...small, color: P.mute }, text: "all dead" }));
      } else {
        const img = await D.image(r.parts_final.png);
        const wrap = ui.h("div", { style: { width: "100%", maxWidth: main + "px" } }, spriteCanvas(img, r.parts_final.layout, 0));
        cell.append(wrap);
        if (nParts > 1) {
          const strip = ui.h("div", { style: { display: "flex", gap: "3px", width: "100%", maxWidth: main + "px" } });
          for (let i = 1; i < Math.min(4, nParts); i++) strip.append(ui.h("div", { style: { flex: "0 0 calc((100% - 6px) / 3)" } }, spriteCanvas(img, r.parts_final.layout, i)));
          cell.append(strip);
        }
      }
      cell.append(ui.h("div", { style: { ...small, textAlign: "center", fontVariantNumeric: "tabular-nums" }, text: `${f.n_live_clusters} live · ${ui.fmtInt(f.n_dead)} dead` }));
      cell.addEventListener("pointermove", ev => ui.showTip(ev.clientX, ev.clientY, cellTip(row, r)));
      cell.addEventListener("pointerleave", () => ui.hideTip());
    }
  }
  ui.renderMath(grid);
}

function cellTip(row, r) {
  const p = D.mainPart(r), f = r.final;
  const rows = [
    { name: "θ", value: ui.fmtFixed(r.theta, 3) }, { name: "q", value: ui.fmtFixed(r.q, 3) },
    { name: "live clusters (major)", value: `${f.n_live_clusters} (${f.n_live_major})` }, { name: "dead neurons", value: ui.fmtInt(f.n_dead) },
  ];
  if (p) rows.push(
    { name: "largest part: neurons", value: ui.fmtInt(p.size) }, { name: "width (pixels)", value: ui.fmt(p.width) },
    { name: "skewness of hᵀg", value: fmt2(p.skew) }, { name: "excess kurtosis", value: fmt2(p.excess_kurtosis) },
    { name: "cosine to the linear part", value: fmt2(p.cos_to_linear_part) }, { name: "fires for (τ = rule's)", value: ui.fmtPct(p.frac_active, 0) });
  return ui.tipRows(row.label.replace(/\$|\\/g, "").replace("tau", "τ").replace("theta", "θ"), rows, r.name);
}

// ------------------------------------------------------------------ training film
const FILM_OPTS = [
  { key: "A_rect_t0", label: "rectified τ = 0, q = 0.980" },
  { key: "A_rect_t1", label: "rectified τ = 1, q = 0.980" },
  { key: "Bown_rect_t0", label: "rectified τ = 0 on Gaussian pixels, own q = 0.980" },
  { key: "A_rect_t0_q0.946", label: "rectified τ = 0, q = 0.946" },
  { key: "B_rect_t0_q0.946", label: "rectified τ = 0 on Gaussian pixels, digits' θ at q = 0.946" },
];

export function filmFigure(host, index, ctx) {
  const byName = Object.fromEntries(index.runs.map(r => [r.name, r]));
  const lin = byName[index.featured.A_lin];
  let other = FILM_OPTS.some(o => o.key === ctx.param("film", "")) ? ctx.param("film", "") : "A_rect_t0";
  const steps = lin.snapshot_steps;
  let k = Math.max(0, Math.min(steps.length - 1, ctx.param("snap", 4)));
  host.replaceChildren();
  const bar = ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "10px 24px", alignItems: "end", marginBottom: "12px" } });
  host.append(bar);
  const sel = ui.select({ label: "compare the linear rule with", options: FILM_OPTS.map(o => ({ value: o.key, label: o.label })), value: other,
    onChange: v => { other = v; ctx.setParams({ film: v }); build(); } });
  const play = ui.h("button", { type: "button", class: "u-play", "aria-label": "Play through snapshots", style: { flex: "0 0 auto" } });
  const snap = ui.slider({ label: "training step", min: 0, max: steps.length - 1, step: 1, value: k, fmt: i => ui.fmtInt(steps[i]),
    onInput: i => { k = i; draw(); }, onChange: i => ctx.setParams({ snap: i }) });
  bar.append(ui.h("div", { style: { flex: "1 1 260px", maxWidth: "380px" } }, sel.el), ui.h("div", { style: { flex: "1 1 300px", maxWidth: "420px", display: "flex", gap: "10px", alignItems: "center" } }, play, ui.h("div", { style: { flex: 1 } }, snap.el)));
  const body = ui.h("div", { style: { display: "grid", gap: "18px" } });
  host.append(body);

  // play: one snapshot every 0.7 s, only while on screen (ui.animate)
  let anim = null, acc = 0;
  const ICON_PLAY = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4.5 2.6v10.8L13 8z" fill="currentColor"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="3.5" y="2.8" width="3.2" height="10.4" rx=".8" fill="currentColor"/><rect x="9.3" y="2.8" width="3.2" height="10.4" rx=".8" fill="currentColor"/></svg>';
  const setIcon = () => { play.innerHTML = anim ? ICON_PAUSE : ICON_PLAY; };
  const stop = () => { anim?.stop(); anim = null; setIcon(); };
  play.addEventListener("click", () => {
    if (anim) { stop(); return; }
    if (k === steps.length - 1) { k = 0; snap.set(0); draw(); }
    acc = 0;
    anim = ui.animate(host, dt => { acc += dt; if (acc < 700) return; acc = 0; if (k >= steps.length - 1) { stop(); ctx.setParams({ snap: k }); return; } k += 1; snap.set(k); draw(); });
    setIcon();
  });
  setIcon();
  ctx.onHide(stop);

  let panels = [], token = 0;
  async function build() {
    const my = ++token;
    body.replaceChildren(ui.h("p", { class: "mute", text: "loading…" }));
    const runs = [lin, byName[index.featured[other]]];
    const data = await Promise.all(runs.map(async r => ({
      r, img: await D.image(r.files.neurons_over_time.png),
      volt: await D.bytes(r.files.similarity_volt.file), rate: await D.bytes(r.files.similarity_rate.file),
    })));
    if (my !== token) return;
    const aV = Math.max(...runs.map(r => r.files.similarity_volt.a)), aR = Math.max(...runs.map(r => r.files.similarity_rate.a));
    body.replaceChildren();
    panels = data.map((d, pi) => {
      const r = d.r, n = r.n_shown;
      const title = pi === 0 ? "Linear rule, q = 0.980" : FILM_OPTS.find(o => o.key === other).label.replace(/^r/, "R");
      const head = ui.h("div", { class: "chart-title", text: title });
      const tiles = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(4, 38px)", gap: "3px" } });
      const tilesBox = ui.h("div", {}, ui.h("div", { class: "u-help", style: { marginBottom: "4px" }, html: "16 neurons, $h_i$" }), tiles);
      const vBox = ui.h("div", { style: { minWidth: 0 } }), rBox = ui.h("div", { style: { minWidth: 0 } });
      const row = ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "10px 26px", alignItems: "flex-start" } }, tilesBox,
        ui.h("div", { style: { minWidth: 0, flex: "1 1 220px", maxWidth: "300px" } }, ui.h("div", { class: "u-help", style: { marginBottom: "4px" }, html: "voltage covariance $h_i^\\top\\Sigma_g h_j$" }), vBox),
        ui.h("div", { style: { minWidth: 0, flex: "1 1 220px", maxWidth: "300px" } }, ui.h("div", { class: "u-help", style: { marginBottom: "4px" }, html: `rate covariance $c^2\\mathrm{Cov}(r_i, r_j)$${pi === 0 ? " (post hoc)" : ""}` }), rBox));
      body.append(ui.h("div", {}, head, row));
      ui.renderMath(row);
      const groups = groupsOf(r);
      const common = { rows: n, cols: n, map: "div", size: 230, groups: { rows: groups }, xlabel: "neuron j", ylabel: "neuron i" };
      const tipFn = (kind, a) => (i, j, v) => ui.tipRows(`${kind}, step ${ui.fmtInt(steps[k])}`, [{ name: `neurons ${r.shown_ids[i]}, ${r.shown_ids[j]}`, value: ui.fmtFixed(v, 3) }]);
      const hv = heatmap(vBox, { ...common, range: [-aV, aV], zlabel: "$h_i^\\top\\Sigma_g h_j$", tooltip: tipFn("voltage cov.") });
      const hr = heatmap(rBox, { ...common, range: [-aR, aR], zlabel: "$c^2\\mathrm{Cov}(r_i,r_j)$", tooltip: tipFn("rate cov.") });
      return { d, tiles, hv, hr };
    });
    draw();
  }
  function draw() {
    for (const p of panels) {
      const { r, img, volt, rate } = p.d, n = r.n_shown, lay = r.files.neurons_over_time.layout;
      p.tiles.replaceChildren(...Array.from({ length: 16 }, (_, row) => spriteCanvas(img, lay, row * lay.ncol + k, 38)));
      p.hv.update({ values: decode(volt, k, n, r.files.similarity_volt.a) });
      p.hr.update({ values: decode(rate, k, n, r.files.similarity_rate.a) });
    }
  }
  build();
  ctx.onParams(p => {
    const f = p.get("film"), s = parseInt(p.get("snap"), 10);
    if (Number.isFinite(s) && s !== k && s >= 0 && s < steps.length) { k = s; snap.set(s); if (!f || f === other) draw(); }
    if (f && f !== other && FILM_OPTS.some(o => o.key === f)) { other = f; sel.set(f); build(); }
  });
}

function decode(u8, k, n, a) {
  const out = new Float32Array(n * n), off = k * n * n, s = 2 * a / 255;
  for (let i = 0; i < n * n; i++) out[i] = -a + s * u8[off + i];
  return out;
}

function groupsOf(r) {
  const g = []; const lab = i => (r.shown_dead[i] ? -1 : r.shown_cluster_rank[i]);
  let start = 0;
  for (let i = 1; i <= r.n_shown; i++) {
    if (i === r.n_shown || lab(i) !== lab(start)) {
      const L = lab(start);
      g.push({ start, end: i, color: L < 0 || L >= 8 ? P.other : catColor(L), label: "" });
      start = i;
    }
  }
  return g;
}

// ------------------------------------------------------------------ samples
const SAMPLE_LABELS = [
  "real test digits", "true moments $\\mathcal N(\\mu, \\Sigma_g)$", "linear rule, least squares",
  "rectified $\\tau = 0$, inversion of the weights", "rectified, chord (nonlinearity ignored)",
  "rectified, inversion of the exact rate covariance", "rectified, trained on Gaussian pixels",
];
export async function samplesFigure(host, index) {
  const S = index.samples, img = await D.image(S.png), lay = S.layout;
  const grid = ui.h("div", { class: "img-grid ig-rows", style: { "--tile": "24px" } });
  S.rows.forEach((row, ri) => {
    const tiles = ui.h("div", { class: "ig-row-tiles" });
    for (let c = 0; c < lay.ncol; c++) tiles.append(ui.h("div", { class: "ig-tile" }, spriteCanvas(img, lay, ri * lay.ncol + c)));
    const neg = row.psd_negative_mass_clipped;
    const lab = (SAMPLE_LABELS[ri] || row.label) + (neg != null && neg > 1e-6 ? `<br><span class="mute">clipped ${ui.fmtPct(neg, 0)}</span>` : "");
    grid.append(ui.h("div", { class: "ig-row" }, ui.h("div", { class: "ig-row-label", html: lab, style: { whiteSpace: "normal", maxWidth: "160px" } }), tiles));
  });
  host.replaceChildren(grid);
  ui.renderMath(grid);
}
