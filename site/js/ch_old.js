// Chapter "old" (More on the covariance rule). Two precomputed figures over the covariance-rule runs in data/big/
// (hebb/big/cov_runs.py): a filmstrip of one run in the triangle (members of a group become identical copies) and
// the split of the neurons among the same groups for three random starts. Every number in the prose is read from
// data/big/cov_all.json here.
import * as ui from "./ui.js";
import { hidpi } from "./charts.js";
import { P, catColor } from "./palette.js";
import { loadFilm } from "./filmplayer.js";

const FILM = "A_cov_pair_th0.5";
const STRIP_T = [0, 0.2, 0.4, 0.8, 2];            // plus the final frame
const SUBS = "₀₁₂₃₄₅₆₇₈₉";
const gName = b => `g${SUBS[b + 1]}`;

export async function init(panel, ctx) {
  const [film, all] = await Promise.all([
    loadFilm(FILM),
    fetch("data/big/cov_all.json", { cache: "no-cache" }).then(r => { if (!r.ok) throw new Error(`data/big/cov_all.json: HTTP ${r.status}`); return r.json(); }),
  ]);
  filmstrip(ctx.slot("strip"), film);
  splitFigure(ctx.slot("split"), all);
  fillProse(panel, all, film);
}

// distinct positions of the neurons, to a tolerance in L1 on the embedding (greedy)
function distinct(coords, N, tol = 2e-3) {
  const reps = [];
  for (let i = 0; i < N; i++) {
    const a = coords.subarray(3 * i, 3 * i + 3);
    if (!reps.some(r => Math.abs(r[0] - a[0]) + Math.abs(r[1] - a[1]) + Math.abs(r[2] - a[2]) < tol)) reps.push(a);
  }
  return reps.length;
}

// ---------------------------------------------------------------- figure 1: filmstrip in the triangle
function filmstrip(el, film) {
  const N = film.N, labels = film.final.labels;
  const frames = STRIP_T.map(t => film.frames.reduce((best, f, i) => (Math.abs(f.t - t) < Math.abs(film.frames[best].t - t) ? i : best), 0));
  frames.push(film.F - 1);
  const data = frames.map(f => { const c = film.coord(f); return { f, t: film.frames[f].t, c, k: distinct(c, N) }; });
  const sizes = new Map(); labels.forEach(l => sizes.set(l, (sizes.get(l) || 0) + 1));
  const colOf = l => (l < 0 ? P.other : catColor(l));

  const row = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "8px 10px" } });
  const cells = data.map(d => {
    const cv = ui.h("canvas", { style: { display: "block", width: "100%" } });
    const lab = ui.h("div", { class: "u-help", style: { textAlign: "center", marginTop: "2px" } });
    const box = ui.h("div", { style: { minWidth: "0" } }, cv, lab);
    row.append(box);
    return { d, cv, lab, box };
  });
  const key = ui.legend({ items: [...[...sizes.keys()].sort((a, b) => a - b).map(l => ({ label: `group ${l + 1} at the end (${sizes.get(l)} neurons)`, color: colOf(l), shape: "dot" })),
    { label: "input", color: P.boundary, shape: "square" }] });
  el.replaceChildren(ui.h("div", { class: "panel-head", style: { marginBottom: "6px" } }, ui.spaceBadge("input", "input space")), row, key.el);

  function draw() {
    for (const { d, cv, lab, box } of cells) {
      const W = Math.floor(box.clientWidth); if (!W) continue;
      const side = W - 30, Hh = Math.round(side * 0.866 + 44), g = hidpi(cv, W, Hh);
      const V = [[15, 22 + side * 0.866], [15 + side, 22 + side * 0.866], [15 + side / 2, 22]];   // g1 left, g2 right, g3 top
      g.clearRect(0, 0, W, Hh);
      g.strokeStyle = P.line; g.lineWidth = 1; g.beginPath(); g.moveTo(...V[0]); g.lineTo(...V[1]); g.lineTo(...V[2]); g.closePath(); g.stroke();
      g.font = "600 11px Inter, system-ui, sans-serif"; g.textBaseline = "middle";
      V.forEach(([x, y], b) => {
        g.fillStyle = P.boundary; g.fillRect(x - 4, y - 4, 8, 8); g.lineWidth = 1.2; g.strokeStyle = P.fg; g.strokeRect(x - 4, y - 4, 8, 8);
        g.fillStyle = P.fg; g.textAlign = "center"; g.fillText(gName(b), x + (b === 2 ? 0 : 0), y + (b === 2 ? -12 : 12));
      });
      const xy = i => { let x = 0, y = 0, s = 0; for (let b = 0; b < 3; b++) { const h = d.c[3 * i + b]; x += h * V[b][0]; y += h * V[b][1]; s += h; } return [x / s, y / s]; };
      for (let i = 0; i < N; i++) {
        const [x, y] = xy(i);
        g.beginPath(); g.arc(x, y, 2.6, 0, 2 * Math.PI); g.fillStyle = colOf(labels[i]); g.globalAlpha = 0.75; g.fill(); g.globalAlpha = 1;
      }
      if (d.f === film.F - 1) {                       // final frame: label each group with its size
        const seen = new Set();
        g.font = "600 12px Inter, system-ui, sans-serif"; g.fillStyle = P.fg; g.textBaseline = "middle";
        for (let i = 0; i < N; i++) {
          const l = labels[i]; if (seen.has(l)) continue; seen.add(l);
          const [x, y] = xy(i); g.beginPath(); g.arc(x, y, 6, 0, 2 * Math.PI); g.fillStyle = colOf(l); g.fill(); g.strokeStyle = P.surface; g.lineWidth = 1.5; g.stroke();
          g.fillStyle = P.fg; g.textAlign = "left"; g.fillText(String(sizes.get(l)), x + 10, y + (y < 40 ? 4 : -10));
        }
      }
      lab.innerHTML = `<b>t = ${ui.fmt(d.t, 2)}</b> · ${d.k} distinct point${d.k === 1 ? "" : "s"}`;
    }
  }
  new ResizeObserver(draw).observe(row);
  draw();
}

// ---------------------------------------------------------------- figure 2: the split for three random starts
const GROUPS = [
  { name: "ind3", theta: 0.4, title: "three independent inputs, $\\theta = 0.4$" },
  { name: "pair3", theta: 0.5, title: "$g_1, g_2$ correlated ($\\rho = 0.6$), $\\theta = 0.5$" },
];
const OMEGA = 0.009;
const supKey = s => s.map(gName).join(" + ");
function splitFigure(el, all) {
  const supports = [];
  const runsOf = gp => all.runs.filter(r => r.name === gp.name && r.theta === gp.theta && r.omega === OMEGA).sort((a, b) => a.seed - b.seed);
  for (const gp of GROUPS) for (const r of runsOf(gp)) for (const c of r.clusters) { const k = supKey(c.support); if (!supports.includes(k)) supports.push(k); }
  supports.sort((a, b) => a.length - b.length || a.localeCompare(b));
  const colOf = k => catColor(supports.indexOf(k));
  const blocks = GROUPS.map(gp => {
    const runs = runsOf(gp);
    const lines = runs.map(r => {
      const N = r.N, segs = r.clusters.slice().sort((a, b) => supports.indexOf(supKey(a.support)) - supports.indexOf(supKey(b.support)));
      const html = segs.map(c => `<span title="${supKey(c.support)}: ${c.size} neurons" style="flex:${c.size} 0 0;background:${colOf(supKey(c.support))};height:100%;display:flex;align-items:center;justify-content:center;color:#fff;font:600 11.5px Inter,system-ui,sans-serif">${c.size}</span>`).join("");
      const dead = N - r.clusters.reduce((s, c) => s + c.size, 0);
      const conv = r.metrics.converged ? "" : " (not converged)";
      return `<div style="display:grid;grid-template-columns:110px 1fr;gap:4px 12px;align-items:center;margin:4px 0"><div style="font:12.5px Inter,system-ui,sans-serif;color:var(--ink-2)">start ${r.seed + 1}${conv}</div><div style="display:flex;height:22px;border-radius:3px;overflow:hidden;background:var(--wash)">${html}${dead > 0 ? `<span style="flex:${dead} 0 0;background:${P.other}"></span>` : ""}</div></div>`;
    }).join("");
    return `<div style="margin:6px 0 14px"><div class="chart-title" style="margin-bottom:2px">${gp.title} · ${runs[0].N} neurons</div>${lines}</div>`;
  }).join("");
  el.innerHTML = blocks;
  ui.legend({ parent: el, items: supports.map(k => ({ label: `group on ${k}`, color: colOf(k), shape: "square" })) });
  ui.renderMath(el);
}

function fillProse(panel, all, film) {
  const put = (key, html) => panel.querySelectorAll(`[data-fill="${key}"]`).forEach(el => { el.innerHTML = html; ui.renderMath(el); });
  const spread = Math.max(...all.runs.map(r => r.metrics.max_spread));
  put("o-runs", String(all.runs.length));
  put("o-conv", all.runs.every(r => r.metrics.converged) ? "all converged" : "not all converged");
  put("o-spread", ui.fmt(spread, 2));
  const Bs = all.runs.map(r => r.config.Sigma.length), Ns = [...new Set(all.runs.map(r => r.N))].sort((a, b) => a - b);
  put("o-B", `${Math.min(...Bs)} to ${Math.max(...Bs)}`);
  put("o-Ns", Ns.join(" or "));
  put("o-N", String(film.N));
  put("o-K", String(film.final.clusters.length));
  // floor sensitivity: same start (seed 0), floor omega 0.009 vs 0.003, on the two families of figure 2
  let dmax = 0;
  for (const gp of GROUPS) {
    const a = all.runs.find(r => r.name === gp.name && r.theta === gp.theta && r.seed === 0 && r.omega === OMEGA);
    const b = all.runs.find(r => r.name === gp.name && r.theta === gp.theta && r.seed === 0 && r.omega !== OMEGA);
    if (!a || !b) continue;
    for (const c of a.clusters) { const d = b.clusters.find(x => supKey(x.support) === supKey(c.support)); dmax = Math.max(dmax, d ? Math.abs(d.size - c.size) : c.size); }
    put("o-omega-lo", String(b.omega));
  }
  put("o-omega", String(OMEGA));
  put("o-floor-d", String(dmax));
}
