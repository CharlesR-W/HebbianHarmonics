// Chapter "perron", section "Which parts get populated": the precomputed second pass on the Perron parts
// (site/data/perron2/, hebb/perron2/; format in data/perron2/README.md). initPerron2(ctx) fills, if present:
//   [data-slot="fig-cold"]  fresh starts: stable parts (grey staircase) and filled parts (dots per seed, mean line
//                           per γ) against θ, marks at γκ₀ (cold_starts.json; staircase from part_diagrams.json).
//                           URL keys `cs` (covariance), `cg` (γ or all).
//   [data-slot="fig-ramp"]  quasi-static θ ramps on hier16: the part diagram (drawPartDiagram) with the ramp's clusters
//                           overlaid, and below it the film of the ramp (filmPlayer), cursor and frame linked both ways
//                           (ramps.json, part_diagrams.json, ramp_hier16_*.json/.bin). URL key `rp` (ramp).
import * as ui from "./ui.js";
import { lineChart } from "./charts.js";
import { P, catColor } from "./palette.js";
import { filmPlayer } from "./filmplayer.js";
import { drawPartDiagram, supportName } from "./ch_perron_teach.js";
import * as M from "./ch_perron_teach_model.js";

const BASE = "data/perron2/";
const cache = new Map();
const getJSON = name => {
  if (!cache.has(name)) cache.set(name, fetch(BASE + name).then(r => { if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`); return r.json(); }));
  return cache.get(name);
};
const ALL_GAMMAS = [1, 0.5, 0.25, 0.1];                           // fixed colour per γ (categorical order)
const gColor = g => catColor(Math.max(0, ALL_GAMMAS.indexOf(g)));
const gLabel = g => `γ = ${g}`;

export function initPerron2(ctx) {
  const sc = ctx.slot("fig-cold"), sr = ctx.slot("fig-ramp");
  const fail = (el, e) => { if (el) el.replaceChildren(ui.h("p", { class: "u-help", text: `Could not load the data: ${e.message}` })); console.warn(e); };
  if (sc) coldFigure(sc, ctx).catch(e => fail(sc, e));
  if (sr) rampFigure(sr, ctx).catch(e => fail(sr, e));
}

// ======================================================================= fresh starts
const COVS = [
  { id: "hier16", label: "hierarchy (16)", diagram: "hier16", long: "16 inputs, binary hierarchy, correlations 0.8 / 0.5 / 0.25 / 0.05 by level" },
  { id: "block12", label: "blocks (3 × 4)", diagram: "block12", long: "12 inputs in three blocks of four" },
  { id: "block345", label: "blocks 3, 4, 5", diagram: null, long: "12 inputs in blocks of 3, 4 and 5, correlation 0.5 within a block and 0.1 across; 8 seeds, only the four settings shown were run" },
];
function block345() {
  const sizes = [3, 4, 5], bl = sizes.flatMap((s, k) => Array(s).fill(k)), B = bl.length, S = new Float64Array(B * B);
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) S[i * B + j] = i === j ? 1 : bl[i] === bl[j] ? 0.5 : 0.1;
  return { S, B };
}

async function coldFigure(slot, ctx) {
  const [cold, diag] = await Promise.all([getJSON("cold_starts.json"), getJSON("part_diagrams.json")]);
  const table = cold.table;
  const covs = COVS.filter(c => table.some(r => r.sigma === c.id));
  let cs = ctx.param("cs", "hier16"); if (!covs.some(c => c.id === cs)) cs = covs[0].id;
  let cg = String(ctx.param("cg", "all"));

  slot.replaceChildren();
  const ctl = ui.h("div", { class: "pt-ctl" });
  const covSeg = ui.segmented({ label: "covariance $\\Sigma_g$", options: covs.map(c => ({ value: c.id, label: c.label })), value: cs,
    onChange: v => { cs = v; ctx.setParams({ cs: v === "hier16" ? null : v }); buildGamma(); draw(); } });
  const gBox = ui.h("div");
  ctl.append(covSeg.el, gBox);
  const chartBox = ui.h("div"), lgBox = ui.h("div"), read = ui.h("div", { class: "pt-read" });
  slot.append(ctl, chartBox, lgBox, read);
  ui.renderMath(ctl);

  const gammasOf = s => ALL_GAMMAS.filter(g => table.some(r => r.sigma === s && r.gamma === g));
  function buildGamma() {
    const gs = gammasOf(cs);
    if (cg !== "all" && !gs.includes(+cg)) cg = "all";
    const seg = ui.segmented({ label: "lateral gain $\\gamma$", options: [{ value: "all", label: "all" }, ...gs.map(g => ({ value: String(g), label: String(g) }))], value: cg,
      onChange: v => { cg = v; ctx.setParams({ cg: v === "all" ? null : v }); draw(); } });
    gBox.replaceChildren(seg.el); ui.renderMath(gBox);
  }

  // staircase of the number of stable parts against θ
  const stairCache = {};
  function staircase(s) {
    if (stairCache[s]) return stairCache[s];
    const c = COVS.find(x => x.id === s), d = c.diagram && diag.diagrams.find(x => x.name === c.diagram);
    let th, n;
    if (d) { th = d.theta; n = d.parts.map(l => l.length); }
    else {
      // blocks: the stable set is constant between consecutive distinct entries of Σ (0.1, 0.5, 1), so one exact
      // enumeration inside each interval gives the whole staircase
      const { S, B } = block345(), lv = [0, 0.1, 0.5, 1];
      th = []; n = [];
      for (let k = 0; k + 1 < lv.length; k++) {
        const cnt = M.stableParts(S, B, (lv[k] + lv[k + 1]) / 2).length;
        th.push(lv[k] + 1e-6, lv[k + 1] - 1e-6); n.push(cnt, cnt);
      }
    }
    const x = [], y = [];
    th.forEach((t, k) => {
      if (k > 0 && n[k] !== n[k - 1]) { const m = (th[k - 1] + t) / 2; x.push(m, m); y.push(n[k - 1], n[k]); }
      x.push(t); y.push(n[k]);
    });
    return (stairCache[s] = { x, y, th, n });
  }

  let chart = null;
  function draw() {
    const rowsS = table.filter(r => r.sigma === cs).sort((a, b) => a.theta - b.theta);
    const gs = gammasOf(cs).filter(g => cg === "all" || +cg === g);
    const st = staircase(cs);
    const series = [{ name: "stable parts", x: st.x, y: st.y, color: P.mute, width: 2.4 }];
    const thetas = [...new Set(rowsS.map(r => r.theta))].sort((a, b) => a - b);
    const step = thetas.length > 1 ? Math.min(...thetas.slice(1).map((t, i) => t - thetas[i])) : 0.05;
    const nSeedMax = Math.max(...rowsS.map(r => r.n_populated.length));
    // one colour per γ; lines and dots of each γ nudged by a fixed small amount (vertically by γ, sideways by seed)
    // so that equal counts stay visible; all lines first, dots on top
    const dy = gs.length > 1 ? Math.max(0.1, 0.012 * (Math.max(...st.y) + 1)) : 0;
    const lines = [], dots = [];
    gs.forEach((g, gi) => {
      const rs = rowsS.filter(r => r.gamma === g), oy = (gi - (gs.length - 1) / 2) * dy;
      const px = [], py = [];
      rs.forEach(r => r.n_populated.forEach((v, s) => { px.push(r.theta + (s - (r.n_populated.length - 1) / 2) * step * 0.06); py.push(v + oy); }));
      const mean = rs.map(r => r.n_populated.reduce((a, b) => a + b, 0) / r.n_populated.length + oy);
      lines.push({ name: `${gLabel(g)}, mean`, x: rs.map(r => r.theta), y: mean, color: gColor(g), width: 1.6 });
      dots.push({ name: `${gLabel(g)}, seeds`, x: px, y: py, color: gColor(g), points: true, r: 2.6 });
    });
    series.push(...lines, ...dots);
    const kappa0 = rowsS[0]?.kappa0;
    const refs = gs.map(g => ({ x: g * kappa0, color: gColor(g) }));
    const ymax = Math.max(...st.y, ...rowsS.flatMap(r => r.n_populated), 1);
    const opts = { series, refs, xrange: [0, 1], yrange: [-0.4, Math.ceil(ymax + 0.5)], height: 300, xlabel: "pruning threshold θ",
      ylabel: "number of parts", legend: false, yfmt: v => (Number.isInteger(v) ? String(v) : ""),
      tooltip: xv => tipAt(xv, rowsS, gs, st) };
    if (!chart) chart = lineChart(chartBox, opts); else chart.update(opts);
    // legend: one entry per γ (mean line and dots in its colour), the staircase, the γκ₀ marks
    lgBox.replaceChildren(ui.legend({ items: [
      { label: "stable parts (exact)", color: P.mute, shape: "line" },
      ...gs.map(g => ({ label: `${gLabel(g)}: filled, per seed (dots) and mean (line)`, color: gColor(g), shape: "dot" })),
      { label: `dashed: $\\gamma\\kappa_0$ in the colour of $\\gamma$ ($\\kappa_0 = ${ui.fmt(kappa0, 3)}$)`, color: P.mute2, shape: "dash" },
    ] }).el);
    lgBox.firstChild.classList.add("chart-legend");
    const c = COVS.find(x => x.id === cs);
    read.innerHTML = `${c.long}. ${thetas.length} values of $\\theta$, ${nSeedMax} seeds each.${dy ? " Lines and dots of each $\\gamma$ are nudged up or down slightly, and seeds sideways, so that equal counts stay visible; hover for the exact values." : ""}`;
    ui.renderMath(lgBox); ui.renderMath(read);
  }
  function tipAt(xv, rowsS, gs, st) {
    const ths = [...new Set(rowsS.map(r => r.theta))];
    if (!ths.length) return null;
    const t = ths.reduce((a, b) => (Math.abs(b - xv) < Math.abs(a - xv) ? b : a));
    const k = st.th.reduce((bi, v, i) => (Math.abs(v - t) < Math.abs(st.th[bi] - t) ? i : bi), 0);
    const row0 = rowsS.find(r => r.theta === t);
    const rows = [{ name: "stable parts", value: String(row0 ? row0.n_stable : st.n[k]), color: P.mute }];
    for (const g of gs) {
      const r = rowsS.find(q => q.theta === t && q.gamma === g); if (!r) continue;
      const m = r.n_populated.reduce((a, b) => a + b, 0) / r.n_populated.length;
      rows.push({ name: `${gLabel(g)}: filled per seed`, value: `${r.n_populated.join(", ")} (mean ${ui.fmt(m, 2)})`, color: gColor(g) });
    }
    return ui.tipRows(`θ = ${ui.fmtFixed(t, 3)}`, rows);
  }
  buildGamma(); draw();
  ctx.onParams?.(p => {
    const a = p.get("cs"), b = p.get("cg");
    if (a && covs.some(c => c.id === a) && a !== cs) { cs = a; covSeg.set(a); buildGamma(); draw(); }
    if (b && b !== cg) { cg = b; buildGamma(); draw(); }
  });
}

// ======================================================================= ramps
const DIR_LABEL = { up: "up", down2: "down" };
async function rampFigure(slot, ctx) {
  const [ramps, diag, index] = await Promise.all([getJSON("ramps.json"), getJSON("part_diagrams.json"), getJSON("index.json").catch(() => [])]);
  const d = diag.diagrams.find(x => x.name === "hier16");
  const B = d.B, Sigma = Float64Array.from(d.Sigma.flat());
  const { linkage, leafOrder } = M.averageLinkage(Sigma, B);
  // part diagram: per θ the stable parts [support, c, margin, gap, h on the support] -> {S, c, margin, h (length B)}
  const parts = d.parts.map(list => list.map(([S, c, margin, , hS]) => { const h = new Array(B).fill(0); S.forEach((b, j) => (h[b] = hS[j])); return { S, c, margin, h }; }));
  const filmOf = (g, dir) => index.find(e => e.gamma === g && e.direction === dir)?.id || null;
  // ramp options: the four with films first, then the rest of ramps.json
  const opts = [];
  for (const run of ramps.runs) for (const dir of ["up", "down2"]) if (run[dir]) opts.push({ key: `${dir}_${run.gamma}`, gamma: run.gamma, dir, data: run[dir], film: filmOf(run.gamma, dir) });
  opts.sort((a, b) => (!!b.film - !!a.film) || (a.dir === b.dir ? b.gamma - a.gamma : a.dir === "up" ? -1 : 1));
  const labelOf = o => `${DIR_LABEL[o.dir]}, γ = ${o.gamma}${o.film ? "" : " (no film)"}`;
  const DEF = "up_1";
  let cur = opts.find(o => o.key === ctx.param("rp", DEF)) || opts[0];
  let k = cur.data.theta.length - 1;                                   // index into the ramp's steps

  slot.replaceChildren();
  const ctl = ui.h("div", { class: "pt-ctl" });
  const sel = ui.select({ label: "ramp", options: opts.map(o => ({ value: o.key, label: labelOf(o) })), value: cur.key,
    help: "Up: from a fresh start at θ = 0.03 to 0.95. Down: from a fresh start at θ = 0.82 (all 16 single inputs filled) to 0.03.",
    onChange: key => pick(key, true) });
  ctl.append(sel.el);
  const box = ui.h("div"), read = ui.h("div", { class: "pt-read" });
  const filmHead = ui.h("div", { class: "chart-title", style: { marginTop: "18px" } });
  const filmBox = ui.h("div"), noFilm = ui.h("p", { class: "u-help" });
  slot.append(ctl, box, read, filmHead, filmBox, noFilm);

  const overlayOf = o => o.data.theta.map((t, i) => ({ theta: t, parts: o.data.supports[i].map((S, j) => ({ support: S, n: o.data.sizes[i][j], stable: !!o.data.matches?.[i]?.[j] })) }));
  const nearestIdx = (arr, x) => arr.reduce((bi, v, i) => (Math.abs(v - x) < Math.abs(arr[bi] - x) ? i : bi), 0);
  const pd = drawPartDiagram(box, { thetas: d.theta, parts, B, leafOrder, linkage, vectors: false, overlay: overlayOf(cur), cursor: cur.data.theta[k],
    overlayLabel: "cluster at the end of a ramp step, on a stable part (size rises with its neurons)", onCursor: t => fromDiagram(t) });

  let fp = null, filmId = null;
  function setStep(i, { film = true } = {}) {
    k = Math.max(0, Math.min(cur.data.theta.length - 1, i));
    pd.setCursor(cur.data.theta[k]); readout();
    if (film && fp && filmId && fp.film?.id === filmId) {
      const fr = fp.film.frames, f = nearestIdx(fr.map(x => x.theta), cur.data.theta[k]);
      if (f !== fp.frame) fp.go(f);
    }
  }
  function fromDiagram(t) { setStep(nearestIdx(cur.data.theta, t)); }
  function readout() {
    const r = cur.data, sizes = r.sizes[k], sups = r.supports[k], m = r.matches?.[k] || [];
    const cl = sups.map((S, j) => `${supportName(S)}: ${sizes[j]}${m[j] ? "" : " (not on a stable part)"}`).join(", ");
    read.innerHTML = `<b>θ = ${ui.fmtFixed(r.theta[k], 3)}</b>, step ${k + 1} of ${r.theta.length}: ${r.n_stable[k]} stable part${r.n_stable[k] === 1 ? "" : "s"}, ` +
      `${r.n_populated[k]} filled (coverage ${ui.fmt(r.coverage[k], 2)}). Clusters (neurons): ${cl || "none"}${r.n_dead?.[k] ? `; ${r.n_dead[k]} dead` : ""}.`;
  }
  function pick(key, user) {
    const o = opts.find(x => x.key === key); if (!o) return;
    const prevTheta = cur.data.theta[k];
    cur = o; sel.set(key);
    if (user) ctx.setParams({ rp: key === DEF ? null : key });
    pd.update({ overlay: overlayOf(cur) });
    k = user ? nearestIdx(cur.data.theta, prevTheta) : cur.data.theta.length - 1;
    filmHead.textContent = o.film ? `The network along this ramp (${labelOf(o)}); drag the slider or the diagram` : "";
    if (o.film) {
      filmBox.hidden = false; noFilm.textContent = "";
      filmId = o.film; pendingTheta = cur.data.theta[k];
      if (!fp) fp = filmPlayer(filmBox, { base: BASE, id: o.film, picker: false, inputEdges: 0, countLabels: true, frameLabels: true, fps: 4,
        spacing: "One frame per step of the ramp, at its end.",
        frameLabel: fr => `θ = ${ui.fmtFixed(fr.theta, 3)}`,
        readout: (fr, f, film) => `frame ${f + 1} of ${film.F} · θ = ${ui.fmtFixed(fr.theta, 3)} · ${fr.K_major} cluster${fr.K_major === 1 ? "" : "s"} (${fr.sizes.join(" / ")}) · ${fr.populated.length} of ${fr.n_stable} stable parts filled · live edges ${ui.fmtInt(fr.n_live)}`,
        onFrame });
      else fp.load(o.film);
    } else {
      filmBox.hidden = true; filmId = null; fp?.pause();
      noFilm.textContent = `No film was recorded for this ramp (${labelOf(o).replace(" (no film)", "")}); the diagram and the line above show its clusters at each step.`;
      setStep(k, { film: false });
    }
    readout();
  }
  // a film that (re)loads shows its last frame first; its first frame event moves it to the step in view
  let pendingTheta = null;
  function onFrame(f, film) {
    if (film.id !== filmId) return;
    const fth = film.frames.map(x => x.theta);
    if (pendingTheta != null) { const g = nearestIdx(fth, pendingTheta); pendingTheta = null; if (g !== f) { fp.go(g); return; } }
    setStep(nearestIdx(cur.data.theta, fth[f]), { film: false });
  }
  pick(cur.key, false);
  ctx.onParams?.(p => { const v = p.get("rp"); if (v && v !== cur.key) pick(v, false); });
}
