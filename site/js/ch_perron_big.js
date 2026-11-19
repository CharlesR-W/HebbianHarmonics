// Chapter "perron": the precomputed big runs (site/data/big/, hebb/big/cov_runs.py, findings B1).
//   films3   the three B = 3, N = 300 films (filmplayer.js), with per-seed cluster sizes from cov_all.json
//   filmsB   the three B = 12-16, N = 400 films; picking one also picks its run in the parts figure
//   parts    per cluster: centroid vs predicted Perron vector vs best eigenvector of Sigma (unit length), for any of
//            the nine B = 12-16 runs at the default floor
//   table    the Perron report of all nine runs (cosines, stability test, off-support margin, best eigenvector)
// Every number shown is read from cov_all.json; the only thing computed here is the best eigenvector itself
// (cov_all.json stores its cosine, not the vector): the projection of the centroid on the eigenspace of Sigma
// (degenerate eigenvalues grouped) with the largest cosine, as in hebb/big/cov_runs.py.
import * as ui from "./ui.js";
import { svgEl, scale, niceTicks } from "./charts.js";
import { P, catColor, mixBary, boundaryColor } from "./palette.js";
import { eigh } from "./onsite_engine.js";
import { filmPlayer } from "./filmplayer.js";

const SUBS = "₀₁₂₃₄₅₆₇₈₉", sub = n => String(n).split("").map(c => SUBS[+c]).join("");
const gl = b => `g${sub(b + 1)}`;
const OMEGA = 0.009;
const SIGMA_NAME = { blocks12: "blocks (3 × 4)", hier16: "hierarchy (16)", ring16: "ring (16)" };
const SIGMA_LONG = {
  blocks12: "12 inputs in three blocks of four (correlation 0.5 within a block, 0.1 across), unit variances",
  hier16: "16 inputs in a binary hierarchy (correlations 0.8 / 0.5 / 0.25 / 0.05 by level), unit variances",
  ring16: "16 inputs on a ring, correlation exp(−d²/(2·1.5²)) at ring distance d",
};
const FILM_OF = { blocks12: "B_cov_blocks12_th0.35", hier16: "B_cov_hier16_th0.35", ring16: "B_cov_ring16_th0.5" };
const isMajor = (c, N) => c.size >= Math.max(2, 0.02 * N);
const runKey = r => `${r.name}_${r.theta}`;
const Bof = r => r.config.Sigma.length;

export async function initBig(ctx) {
  const data = await fetch("data/big/cov_all.json").then(r => { if (!r.ok) throw new Error(`cov_all.json: HTTP ${r.status}`); return r.json(); });
  const runs = data.runs;
  numbers(ctx, runs);
  seedTable(ctx, runs);

  // ---- B = 3 films
  const f3 = ctx.slot("films3");
  if (f3) {
    let last3 = null;
    filmPlayer(f3, { filter: e => e.rule === "cov" && e.family === "A", id: ctx.param("f3", "A_cov_ind_th0.4"), inputEdges: 300, countLabels: true,
      colors: film => { const H = film.H(film.F - 1), bc = [0, 1, 2].map(b => boundaryColor(b, 3)); return H ? Array.from({ length: film.N }, (_, i) => mixBary(Array.from(H.subarray(3 * i, 3 * i + 3)), bc)) : null; },
      // the first load restores the URL state; only later loads (the reader's picks) write it
      onFrame: (f, film) => { if (film.id !== last3) { const first = last3 === null; last3 = film.id; if (!first) ctx.setParams({ f3: film.id === "A_cov_ind_th0.4" ? null : film.id }); film3Note(ctx, film, runs); } } });
  }

  // ---- B = 12-16: parts figure first (the film picker drives it)
  const big = runs.filter(r => Bof(r) !== 3 && r.N === 400 && r.omega === OMEGA && r.seed === 0)
    .sort((a, b) => a.name.localeCompare(b.name) || a.theta - b.theta);
  const parts = partsFigure(ctx, big);
  partsTable(ctx, runs);
  const fB = ctx.slot("filmsB");
  if (fB) {
    let lastB = null;
    filmPlayer(fB, { filter: e => e.rule === "cov" && e.family === "B", id: ctx.param("fb", "B_cov_blocks12_th0.35"), inputEdges: 400, countLabels: true,
      onFrame: (f, film) => {
        if (film.id === lastB) return;
        const first = lastB === null; lastB = film.id;
        const r = big.find(x => x.film === film.id);
        if (first) { if (r && parts && ctx.param("pr", null) == null) parts.pick(runKey(r)); return; }   // keep a saved `pr`
        if (r && parts) parts.pick(runKey(r));
        ctx.setParams({ fb: film.id === "B_cov_blocks12_th0.35" ? null : film.id, pr: !r || runKey(r) === "blocks12_0.35" ? null : runKey(r) });
      } });
  }
}

// ---------------------------------------------------------------- numbers in the prose (data-pv="…")
function numbers(ctx, runs) {
  const N2 = r => Math.max(2, 0.02 * r.N);
  const major = runs.flatMap(r => r.clusters.filter(c => c.size >= N2(r)).map(c => ({ ...c, run: r })));
  const big = major.filter(c => Bof(c.run) !== 3);
  const v = {
    nruns: String(runs.length),
    nclusters: String(major.length),
    perronmin: ui.fmtFixed(Math.floor(Math.min(...major.map(c => c.perron_cos)) * 1e4) / 1e4, 4),
    topok: major.every(c => c.top_eig_ok) ? "every" : `${major.filter(c => c.top_eig_ok).length} of ${major.length}`,
    offmax: ui.fmt(Math.max(...major.map(c => c.offsupport_violation)), 2),
    spreadmax: ui.fmt(Math.max(...major.map(c => c.spread)), 1),
    nearestmin: ui.fmtFixed(Math.floor(Math.min(...major.map(c => c.nearest_stable_part_cos)) * 1e4) / 1e4, 4),
    eiglo: ui.fmtFixed(Math.min(...big.map(c => c.best_eig_cos)), 2),
    eighi: ui.fmtFixed(Math.max(...big.map(c => c.best_eig_cos)), 2),
  };
  ctx.panel.querySelectorAll("[data-pv]").forEach(el => { const k = el.dataset.pv; if (v[k] != null) el.textContent = v[k]; });
}

// sizes per seed for the B = 3 runs, one line per setting
function seedTable(ctx, runs) {
  const el = ctx.slot("films3-seeds"); if (!el) return;
  const set = [["ind3", 0.4, "independent, θ = 0.4"], ["ind3", 0.2, "independent, θ = 0.2"], ["pair3", 0.5, "ρ₁₂ = 0.6, θ = 0.5"]];
  const rows = set.map(([nm, th, lab]) => {
    const rs = runs.filter(r => r.name === nm && r.theta === th && r.omega === OMEGA).sort((a, b) => a.seed - b.seed);
    const cells = rs.map(r => r.clusters.filter(c => c.size >= Math.max(2, 0.02 * r.N)).map(c => `${c.size} at ${partName(c, Bof(r))}`).join(", "));
    return ui.h("tr", {}, ui.h("td", { text: lab }), ...cells.map(t => ui.h("td", { text: t })));
  });
  const head = ui.h("tr", {}, ui.h("th", { text: "setting" }), ...[0, 1, 2].map(s => ui.h("th", { text: `seed ${s}` })));
  el.replaceChildren(ui.h("div", { class: "table-scroll" }, ui.h("table", { class: "data" }, ui.h("thead", {}, head), ui.h("tbody", {}, ...rows))));
}
function partName(c, B) {
  const s = c.support;
  if (s.length === 1) return `the ${gl(s[0])} corner`;
  if (s.length === 2 && B === 3) return `the ${gl(s[0])}–${gl(s[1])} midpoint`;
  return `{${s.map(gl).join(" ")}}`;
}

function film3Note(ctx, film, runs) {
  const el = ctx.slot("films3-note"); if (!el) return;
  const r = runs.find(x => x.film === film.id);
  if (!r) { el.textContent = ""; return; }
  const cl = r.clusters.filter(c => c.size >= Math.max(2, 0.02 * r.N));
  let html = `<b>This run</b> (seed ${r.seed}): ${cl.map(c => `${c.size} neurons on ${partName(c, 3)}`).join(", ")}; converged at $t = ${ui.fmt(r.metrics.t, 3)}$.`;
  if (cl.length === 1) html += ` All ${r.N} neurons end on a single point at the corner, drawn as one dot at the corner and labelled with the count (the input square sits just outside the corner, joined to it by a leader).`;
  el.innerHTML = html; ui.renderMath(el);
}

// ---------------------------------------------------------------- best eigenvector (degenerate eigenspaces grouped)
function bestEigen(Sigma, c) {
  const B = c.length, { values, vectors } = eigh(Float64Array.from(Sigma.flat()), B);
  const nc = Math.hypot(...c), tol = 1e-8 * Math.max(...values.map(Math.abs));
  let best = { cos: -1 };
  for (let k = 0; k < B;) {
    let e = k + 1; while (e < B && Math.abs(values[e] - values[k]) <= tol) e++;
    const proj = new Float64Array(B);
    for (let q = k; q < e; q++) { let d = 0; for (let i = 0; i < B; i++) d += vectors[i * B + q] * c[i]; for (let i = 0; i < B; i++) proj[i] += d * vectors[i * B + q]; }
    const np = Math.hypot(...proj), cos = np / nc;
    if (cos > best.cos + 1e-12) best = { cos, vec: Array.from(proj, x => x / np), rank: k, mult: e - k };
    k = e;
  }
  return best;
}
const unit = v => { const n = Math.hypot(...v) || 1; return v.map(x => x / n); };

// ---------------------------------------------------------------- per-cluster small multiples
function partsFigure(ctx, big) {
  const slot = ctx.slot("parts"); if (!slot) return null;
  slot.replaceChildren();
  const def = ctx.param("pr", "blocks12_0.35");
  let cur = big.find(r => runKey(r) === def) || big[0];
  const sel = ui.select({ label: "run", options: big.map(r => ({ value: runKey(r), label: `${SIGMA_NAME[r.name]}, θ = ${r.theta}` })), value: runKey(cur),
    onChange: k => { pick(k); ctx.setParams({ pr: k === "blocks12_0.35" ? null : k }); } });
  sel.el.style.maxWidth = "360px";
  const sum = ui.h("p", { class: "u-help", style: { margin: "6px 0 10px" } });
  const grid = ui.h("div", { class: "bars-multi", style: { "--panel-w": "200px" } });
  const lg = ui.legend({ items: [
    { label: "cluster centroid (mean $h_i$ of the members)", color: P.ink2, shape: "square" },
    { label: "predicted part: Perron vector of $A_{SS}$ on the cluster's support", color: P.fg, shape: "dot" },
    { label: "best eigenvector of $\\Sigma_g$ (signed)", color: P.mute, shape: "dash" },
  ] });
  lg.el.classList.add("chart-legend");
  slot.append(sel.el, sum, lg.el, grid);
  ui.renderMath(slot);

  function pick(k) {
    const r = big.find(x => runKey(x) === k); if (!r) return;
    cur = r; sel.set(k); draw();
  }
  function draw() {
    const r = cur, B = Bof(r), Sigma = r.config.Sigma, labels = Array.from({ length: B }, (_, b) => gl(b));
    const cl = r.clusters.filter(c => isMajor(c, r.N));
    const panels = cl.map((c, k) => {
      const part = (r.metrics.stable_parts || []).find(s => s.S.join(",") === (c.nearest_stable_part_support || c.support).join(","));
      const ev = bestEigen(Sigma, c.centroid);
      return { c, k, bars: unit(c.centroid), dots: part ? unit(part.h) : null, line: ev.vec, ev };
    });
    let lo = 0, hi = 0;
    panels.forEach(p => [p.bars, p.dots || [], p.line].forEach(v => v.forEach(x => { lo = Math.min(lo, x); hi = Math.max(hi, x); })));
    const yr = [Math.min(0, Math.floor(lo * 10) / 10), Math.max(0.5, Math.ceil(hi * 10 + 0.5) / 10)];
    grid.replaceChildren(...panels.map(p => {
      const box = ui.h("div");
      box.append(ui.h("div", { class: "bm-title" }, ui.h("i", { class: "u-mk dot", style: { "--sw": catColor(p.k) } }),
        ui.h("span", { text: `c${p.k + 1}: ${p.c.size} neurons, support ${p.c.support.length === B ? "all" : p.c.support.length === 1 ? gl(p.c.support[0]) : `${gl(p.c.support[0])}…${gl(p.c.support[p.c.support.length - 1])}`}` })));
      box.append(ui.h("div", { class: "bm-sub", html: `cos with part ${ui.fmtFixed(p.c.perron_cos, 5)} · with eigenvector ${ui.fmtFixed(p.c.best_eig_cos, 3)}` }));
      const c = ui.h("div"); box.append(c);
      partPanel(c, { labels, bars: p.bars, dots: p.dots, line: p.line, color: catColor(p.k), yr, tip: b => ui.tipRows(`${labels[b]}, cluster c${p.k + 1}`, [
        { name: "centroid (unit length)", value: ui.fmtFixed(p.bars[b], 3), color: catColor(p.k), shape: "square" },
        { name: "Perron vector of A_SS", value: p.dots ? ui.fmtFixed(p.dots[b], 3) : "n/a", color: P.fg },
        { name: `best eigenvector (rank ${p.ev.rank + 1}${p.ev.mult > 1 ? `, ${p.ev.mult}-fold` : ""})`, value: ui.fmtFixed(p.line[b], 3), color: P.mute },
      ]) });
      return box;
    }));
    const m = r.metrics;
    sum.innerHTML = `${SIGMA_LONG[r.name]}; $\\theta = ${r.theta}$, $N = ${r.N}$, seed ${r.seed}. ` +
      `${m.n_stable_parts} stable part${m.n_stable_parts === 1 ? "" : "s"} exist (exact enumeration); ${cl.length} ${cl.length === 1 ? "is" : "are"} populated` +
      `${m.n_dead ? `, ${m.n_dead} neuron${m.n_dead === 1 ? "" : "s"} dead` : ""}${r.clusters.length > cl.length ? `, ${r.clusters.length - cl.length} straggler${r.clusters.length - cl.length === 1 ? "" : "s"} below 2% not shown` : ""}. ` +
      `$\\kappa_0 = ${ui.fmt(m.kappa0, 3)}$`;
    ui.renderMath(sum);
  }
  draw();
  return { pick };
}

// one small SVG panel: bars (centroid), ink dots joined by a thin line (Perron vector), dashed grey line with small
// dots (eigenvector), shared y range, zero line
function partPanel(el, o) {
  el.classList.add("chart");
  const ro = new ResizeObserver(() => render()); ro.observe(el);
  let Wd = 0;
  function render() {
    const W = Math.max(150, el.clientWidth || 200); if (Math.abs(W - Wd) < 1 && el.firstChild) return; Wd = W;
    const H = 116, m = { l: 30, r: 4, t: 6, b: 18 }, n = o.labels.length, bw = (W - m.l - m.r) / n;
    const Y = scale("linear", o.yr[0], o.yr[1], H - m.b, m.t), X = i => m.l + (i + 0.5) * bw;
    const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "centroid, Perron vector and eigenvector over the inputs" });
    const g = svgEl("g", { class: "grid" }, svg);
    niceTicks(o.yr[0], o.yr[1], 3).forEach(v => {
      svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g);
      const t = svgEl("text", { x: m.l - 4, y: Y(v) + 3.5, "text-anchor": "end" }, svg); t.textContent = ui.fmt(v, 2);
    });
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: P.mute2, "stroke-width": 1 }, svg);
    o.bars.forEach((v, i) => svgEl("rect", { x: X(i) - bw * 0.36, width: bw * 0.72, y: Math.min(Y(v), Y(0)), height: Math.abs(Y(v) - Y(0)), fill: o.color, opacity: 0.85 }, svg));
    const path = v => v.map((x, i) => (i ? "L" : "M") + X(i).toFixed(1) + "," + Y(x).toFixed(1)).join("");
    svgEl("path", { d: path(o.line), fill: "none", stroke: P.mute, "stroke-width": 1.3, "stroke-dasharray": "4 3" }, svg);
    o.line.forEach((x, i) => svgEl("circle", { cx: X(i), cy: Y(x), r: 1.8, fill: P.mute }, svg));
    if (o.dots) {
      svgEl("path", { d: path(o.dots), fill: "none", stroke: P.fg, "stroke-width": 0.8, opacity: 0.6 }, svg);
      o.dots.forEach((x, i) => svgEl("circle", { cx: X(i), cy: Y(x), r: Math.min(2.8, bw / 3 + 0.6), fill: P.fg, stroke: P.surface, "stroke-width": 0.8 }, svg));
    }
    const every = n > 12 ? 4 : n > 6 ? 2 : 1;
    o.labels.forEach((l, i) => { if (i % every === 0 || i === n - 1) { const t = svgEl("text", { x: X(i), y: H - 5, "text-anchor": "middle" }, svg); t.textContent = l; } });
    const hit = svgEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", ev => { const r = svg.getBoundingClientRect(), i = Math.max(0, Math.min(n - 1, Math.floor((ev.clientX - r.left - m.l) / bw))); ui.showTip(ev.clientX, ev.clientY, o.tip(i)); });
    hit.addEventListener("pointerleave", () => ui.hideTip());
    el.replaceChildren(svg);
  }
  render();
}

// ---------------------------------------------------------------- the Perron report, all B = 12-16 runs
function partsTable(ctx, runs) {
  const el = ctx.slot("parts-table"); if (!el) return;
  const big = runs.filter(r => Bof(r) !== 3 && r.seed === 0).sort((a, b) => a.name.localeCompare(b.name) || a.theta - b.theta || b.omega - a.omega);
  const rng = (xs, d) => { const lo = Math.min(...xs), hi = Math.max(...xs); return Math.abs(hi - lo) < 0.5 * 10 ** -d ? ui.fmtFixed(lo, d) : `${ui.fmtFixed(lo, d)}–${ui.fmtFixed(hi, d)}`; };
  const rows = [];
  let prev = null;
  for (const r of big) {
    const cl = r.clusters.filter(c => isMajor(c, r.N)), m = r.metrics, first = r.name !== prev; prev = r.name;
    rows.push(ui.h("tr", { style: r.omega === OMEGA ? null : { color: "var(--mute)" } },
      ui.h("td", { text: first && r.omega === OMEGA ? SIGMA_NAME[r.name] : "" }),
      ui.h("td", { class: "num", text: r.omega === OMEGA ? String(r.theta) : "" }),
      ui.h("td", { class: "num", text: r.omega === OMEGA ? "ω" : "ω/3" }),
      ui.h("td", { class: "num", text: String(m.n_stable_parts) }),
      ui.h("td", { class: "num", text: String(cl.length) }),
      ui.h("td", { class: "num", style: { whiteSpace: "nowrap" }, text: cl.length > 4 ? `${cl.slice(0, 3).map(c => c.size).join("/")}/… (${cl[cl.length - 1].size})` : cl.map(c => c.size).join("/") }),
      ui.h("td", { class: "num", text: ui.fmtFixed(Math.floor(Math.min(...cl.map(c => c.perron_cos)) * 1e4) / 1e4, 4) }),
      ui.h("td", { class: "num", text: cl.every(c => c.top_eig_ok) ? "all pass" : `${cl.filter(c => c.top_eig_ok).length} pass` }),
      ui.h("td", { class: "num", text: ui.fmt(Math.max(...cl.map(c => c.offsupport_violation)), 2) }),
      ui.h("td", { class: "num", text: rng(cl.map(c => c.best_eig_cos), 3) }),
    ));
  }
  const th = (t, title) => ui.h("th", { text: t, title });
  el.replaceChildren(ui.h("div", { class: "table-scroll" }, ui.h("table", { class: "data" },
    ui.h("thead", {}, ui.h("tr", {}, th("Σ_g"), th("θ"), th("floor"), th("stable parts", "exact enumeration of the parts of A that pass the stability test"),
      th("clusters", "clusters with at least 2% of the neurons"), th("sizes"), th("min cos(centroid, Perron)", "over the clusters"),
      th("top-eigenvalue test"), th("max off-support (Ah)_b", "≤ 0 is required of a part; more negative = larger margin"), th("best Σ_g eigenvector cos"))),
    ui.h("tbody", {}, ...rows))));
}
