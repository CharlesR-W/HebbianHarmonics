// Chapter "many": how the picture scales with many inputs, and what "resolution" means.
//   * precomputed: data/sweeps2/fusion.json (N = 40 networks swept in b: cold start, continuation up and down)
//     and data/sweeps2/resolution.json (the three resolution statements);
//   * live: the population sim (ch_many_live.js) and the resolution figures, including the line along one mode
//     (ch_many_res.js). Model code without DOM is in ch_many_model.js.
// Every number in the prose is filled from the JSON at load time (spans with data-mv="key").
import * as ui from "./ui.js";
import { lineChart, hidpi, scale } from "./charts.js";
import { P, catColor } from "./palette.js";
import { buildLive, BR, BR_LABEL } from "./ch_many_live.js";
import { buildResolution } from "./ch_many_res.js";

export async function init(panel, ctx) {
  const loading = [...panel.querySelectorAll("[data-slot]")].map(s => { s.append(ui.h("p", { class: "u-help", text: "loading…" })); return s; });
  const [F, R] = await Promise.all(["fusion", "resolution"].map(n => fetch(`data/sweeps2/${n}.json`).then(r => {
    if (!r.ok) throw new Error(`data/sweeps2/${n}.json: HTTP ${r.status}`);
    return r.json();
  })));
  loading.forEach(s => s.replaceChildren());
  fillNumbers(panel, numbers(F, R));
  buildSweep(ctx, F);
  buildLive(ctx.slot("live"), ctx, F);
  buildResolution(ctx, R);
}

// ------------------------------------------------------------------ numbers quoted in the prose
const num = v => (v === "inf" ? Infinity : v);
const f3 = v => ui.fmt(v, 3), f2 = v => ui.fmt(v, 2);
function meanSd(xs) { const m = xs.reduce((a, b) => a + b, 0) / xs.length; const sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length); return [m, sd]; }

function numbers(F, R) {
  const c = F.meta.config, bg = c.b_grid, runs = F.runs, o = {};
  o.fN = c.N; o.fB = c.B; o.fs = c.s; o.fSlin = F.meta.seeds.linear.length; o.fSrect = F.meta.seeds.rect.length;
  o.fnb = bg.length; o.fbmin = f3(bg[0]); o.fbmax = f3(bg[bg.length - 1]);
  o.fconv = `${F.summary.n_converged} of ${F.summary.n_runs}`; o.fmaxres = f2(F.summary.max_residual);
  o.fdevlive = f2(F.summary.max_dev_live_converged); o.fslack = f2(F.summary.min_slack_dead_converged);
  o.fSe = c.S_eval; o.fS = c.S_train;
  const all = br => runs.flatMap(r => r[br].span_over_range.map(num));
  const allE = br => runs.flatMap(r => r[br].span_over_range_eval.map(num));
  o.fchain = f3(Math.max(...["cold", "up", "down"].flatMap(all))); o.fchainE = f3(Math.max(...["cold", "up", "down"].flatMap(allE)));
  // diameter at b = 0.1 (blocks, linear), alpha = 1 vs 0.1
  const kb = bg.findIndex(b => Math.abs(b - 0.1) < 1e-9);
  const sel = (sg, pot, al) => runs.filter(r => r.sigma === sg && r.pot === pot && r.alpha === al);
  const d = al => meanSd(sel("blocks", "linear", al).map(r => r.cold.diam_max[kb]));
  const [m1, s1] = d(1), [m01, s01] = d(0.1);
  o.fdiam1 = `${f2(m1)} ± ${f2(s1)}`; o.fdiam01 = `${f2(m01)} ± ${f2(s01)}`; o.fbk = f3(bg[kb]);
  // hysteresis windows: b where some seed's up and down partitions differ
  const win = (sg, al) => { const bs = bg.filter((_, k) => runs.some(r => r.sigma === sg && r.pot === "linear" && r.alpha === al && r.part_diff_up_down[k] > 0)); return bs.length ? `${f2(bs[0])}–${f2(bs[bs.length - 1])}` : "none"; };
  const pdmax = al => f2(Math.max(...runs.filter(r => r.pot === "linear" && r.alpha === al).flatMap(r => r.part_diff_up_down)));
  o.fpd01 = pdmax(0.1); o.fpd1 = pdmax(1);
  o.fwin01b = win("blocks", 0.1); o.fwin01p = win("power", 0.1); o.fwin1b = win("blocks", 1); o.fwin1p = win("power", 1);
  o.fk01 = f3(0.1 * c.deg / 4); o.fk1 = f3(1 * c.deg / 4);
  // which branch is lower: J(up) - J(down), seed means per configuration and b
  let best = { v: -Infinity }, worst = { v: Infinity };
  for (const sg of ["blocks", "power"]) for (const pot of ["linear", "rect"]) for (const al of c.alphas) {
    const rs = sel(sg, pot, al);
    bg.forEach((b, k) => {
      const v = rs.reduce((a, r) => a + r.up.J[k] - r.down.J[k], 0) / rs.length;
      if (v > best.v) best = { v, b, sg, pot, al };
      if (v < worst.v) worst = { v, b, sg, pot, al };
    });
  }
  const potName = p => (p === "rect" ? "rectifying" : "linear");
  o.fJbest = f2(best.v); o.fJbestcfg = `${best.sg}, ${potName(best.pot)}, α = ${best.al}, b = ${f2(best.b)}`;
  o.fJworst = f2(worst.v); o.fJworstcfg = `${worst.sg}, ${potName(worst.pot)}, α = ${worst.al}, b = ${f2(worst.b)}`;
  // nestedness
  let nb = { v: -1 }; runs.forEach(r => r.nest_cold.forEach((v, k) => { if (v > nb.v) nb = { v, r, k }; }));
  o.fnest = f2(nb.v); o.fnestcfg = `${nb.r.sigma}, ${potName(nb.r.pot)}, α = ${nb.r.alpha}, between b = ${f2(bg[nb.k])} and ${f2(bg[nb.k + 1])}`;
  o.fnestdown = f2(Math.max(...runs.flatMap(r => r.nest_down)));
  const a1 = runs.filter(r => r.alpha === 1);
  o.fnonmono = `${a1.filter(r => r.cold.K.some((v, k) => k > 0 && v > r.cold.K[k - 1])).length} of ${a1.length}`;

  // ---- resolution.json
  const p1 = R.p1.rows, col = j => Math.max(...p1.map(r => r[j]));
  o.r1chat = f2(col(3)); o.r1sig = ui.fmtPct(col(4), 1); o.r1eval = ui.fmtPct(col(5), 1); o.r1sum = f2(col(6));
  o.rS = R.meta.config.S_train; o.rSe = R.meta.config.S_eval; o.r1n = p1.length;
  const p2 = R.p2;
  o.r2n = p2.n_pairs; o.r2mc = p2.misclassified_by_Chat_threshold; o.r2ms = p2.misclassified_by_Sigma_threshold;
  o.r2b = f3(p2.b); o.r2a = f3(p2.alpha); o.rseeds = R.meta.seeds.length;
  const ratios = p2.nonlinear_rows.map(r => r[3] / r[2]);
  o.r2nllo = f2(Math.min(...ratios)); o.r2nlhi = f2(Math.max(...ratios));
  o.r2shlo = f3(1 / Math.sqrt(Math.max(...ratios))); o.r2shhi = f3(1 / Math.sqrt(Math.min(...ratios)));
  const rr = (al, b) => R.p3a.ratio_rows.find(r => r[0] === al && r[1] === b)?.[2];
  o.r3r01 = f3(rr(0.1, 0.1)); o.r3r001 = f3(rr(0.01, 0.1)); o.r3r1 = f3(rr(1, 0.1));
  const p3 = R.p3a.rows;
  const lams = p3a => [...new Set(p3a.map(r => r[1]))];
  { const l = lams(R.p3a.rows); o.r3lamratio = f2(Math.max(...l) / Math.min(...l)); }
  { const modes = new Set(p2.rows_seed0.map(r => r[0])); o.r2nmodes = modes.size; o.r2nsep = p2.rows_seed0.length / modes.size; }
  o.r3offmax = f2(Math.max(...p3.map(r => Math.abs(r[7] - r[6]) / r[8])));
  o.r3onmax = f2(Math.max(...p3.map(r => Math.abs(r[5] - r[4]) / r[8])));
  // population on a line
  const pop = R.pop;
  let cases = 0, below = 0, ex = { v: -1 }, bridge = 0;
  pop.forEach(p => p.cold.forEach((c2, k) => { cases++; if (c2[0] < c2[1]) below++; if (c2[0] - c2[1] > ex.v) ex = { v: c2[0] - c2[1], p, k }; bridge = Math.max(bridge, c2[2]); }));
  o.r4cases = cases; o.r4below = below; o.r4ex = ex.v; o.r4bridge = f3(bridge);
  const bgp = R.meta.config.population.b_grid;
  o.r4N = R.meta.config.population.N; o.r4nb = bgp.length;
  // condensation: alpha = 0.01, grid points where the seed-mean predicted K <= 1.8 and K > 1
  const meanAt = (a, al, k, f) => { const ps = pop.filter(p => p.a === a && p.alpha === al); return ps.reduce((s, p) => s + f(p, k), 0) / ps.length; };
  let condK = [], condSpan = 0;
  for (const a of [1, 3]) bgp.forEach((b, k) => {
    const pk = meanAt(a, 0.01, k, (p, kk) => p.cold[kk][1]), ck = meanAt(a, 0.01, k, (p, kk) => p.cold[kk][0]);
    if (pk <= 1.8) { condK.push(ck); pop.filter(p => p.a === a && p.alpha === 0.01).forEach(p => { condSpan = Math.max(condSpan, p.cold[k][3], p.cold[k][5]); }); }
  });
  o.r4condlo = f2(Math.min(...condK)); o.r4condhi = f2(Math.max(...condK)); o.r4condspan = f2(condSpan);
  // down-continuation coarser: mode 3, alpha 0.01, b closest to 0.006
  const k6 = bgp.reduce((bi, b, k) => (Math.abs(Math.log(b / 0.006)) < Math.abs(Math.log(bgp[bi] / 0.006)) ? k : bi), 0);
  o.r4b6 = f3(bgp[k6]);
  o.r4down = f2(meanAt(3, 0.01, k6, (p, k) => p.down[k][0])); o.r4cold = f2(meanAt(3, 0.01, k6, (p, k) => p.cold[k][0]));
  o.r4pred = f2(meanAt(3, 0.01, k6, (p, k) => p.cold[k][1]));
  o.r4up = f2(meanAt(3, 0.01, k6, (p, k) => p.up[k][0]));
  o.r4scal = f2(Math.max(...R.scaling.rows.map(r => r[2])));
  return o;
}

function fillNumbers(panel, o) {
  panel.querySelectorAll("[data-mv]").forEach(el => {
    const v = o[el.dataset.mv];
    el.textContent = v == null ? "n/a" : typeof v === "number" ? (Number.isInteger(v) ? String(v) : ui.fmt(v, 3)) : String(v);
  });
}

// ------------------------------------------------------------------ precomputed sweep (fusion.json)
const SG_OPTS = [{ value: "blocks", label: "blocks" }, { value: "power", label: "power law" }];
const POT_OPTS = [{ value: "linear", label: "linear" }, { value: "rect", label: "rectifier" }];

function buildSweep(ctx, F) {
  const bg = F.meta.config.b_grid;
  const st = {
    sg: ctx.param("sg", "blocks") === "power" ? "power" : "blocks",
    pot: ctx.param("pot", "linear") === "rect" ? "rect" : "linear",
    al: ctx.param("al", 1) === 0.1 ? 0.1 : 1,
  };
  const bars = [];
  function controlBar(parent) {
    const segs = {
      sg: ui.segmented({ label: "$\\Sigma_g$", options: SG_OPTS, value: st.sg, onChange: v => change({ sg: v }) }),
      pot: ui.segmented({ label: "potential", options: POT_OPTS, value: st.pot, onChange: v => change({ pot: v }) }),
      al: ui.segmented({ label: "$\\alpha$", options: [{ value: 0.1, label: "0.1" }, { value: 1, label: "1" }], value: st.al, onChange: v => change({ al: v }) }),
    };
    const el = ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "8px 22px", alignItems: "center", margin: "4px 0 10px", font: "13px/1.4 var(--sans)" } }, segs.sg, segs.pot, segs.al);
    parent.append(el); bars.push(segs);
  }
  function change(d) {
    Object.assign(st, d);
    bars.forEach(b => { b.sg.set(st.sg); b.pot.set(st.pot); b.al.set(st.al); });
    ctx.setParams({ sg: st.sg === "blocks" ? null : st.sg, pot: st.pot === "linear" ? null : st.pot, al: st.al === 1 ? null : st.al });
    draw();
  }
  const grid = (w = 400) => ui.h("div", { style: { display: "grid", gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, ${w}px), 1fr))`, gap: "16px 26px" } });
  const panel = (g, title, sub) => { const b = ui.h("div"); g.append(ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", html: title }), sub ? ui.h("div", { class: "u-help", style: { margin: "-2px 0 4px" }, html: sub }) : null, b)); return b; };

  // figure 1: K, largest, diameters
  const s1 = ctx.slot("sweep"); controlBar(s1);
  const g1 = grid(); s1.append(g1);
  const xr = [bg[0] * 0.85, bg[bg.length - 1] * 1.15], xlab = "fusion scale b";
  const cK = lineChart(panel(g1, "(a) Components $K$", "log scale; dashed: components with ≥ 2 members (cold start)"), { series: [], xlog: true, ylog: true, xrange: xr, xlabel: xlab, ylabel: "K", height: 230, xname: "b" });
  const cL = lineChart(panel(g1, "(b) Largest component", "neurons"), { series: [], xlog: true, xrange: xr, xlabel: xlab, ylabel: "neurons", height: 230, xname: "b" });
  const cD = lineChart(panel(g1, "(c) Widest component: $\\max$ member-to-prototype $D$", "lines: training bank; dots: evaluation bank (cold start)"), { series: [], xlog: true, xrange: xr, xlabel: xlab, ylabel: "diam_max (response units)", height: 230, xname: "b" });
  const cS = lineChart(panel(g1, "(d) Chain index: largest in-component $D$ over $\\sqrt{2b}$", "above 1: a component wider than one edge's range"), { series: [], xlog: true, xrange: xr, xlabel: xlab, ylabel: "span / √(2b)", height: 230, xname: "b", refs: [{ y: 1, label: "one edge's range" }] });
  ui.renderMath(s1);

  // figure 2: nestedness, up/down difference, J, partitions
  const s2 = ctx.slot("history"); controlBar(s2);
  const g2 = grid(300); s2.append(g2);
  const cN = lineChart(panel(g2, "(e) Refinement violations between neighbouring $b$", "fraction of pairs together at the smaller b, apart at the larger; 0 = nested"), { series: [], xlog: true, xrange: xr, xlabel: "b (geometric midpoint of the two grid values)", ylabel: "fraction of pairs", height: 230, xname: "b" });
  const cH = lineChart(panel(g2, "(f) Up and down partitions differ", "pairs grouped differently, both directions summed; 0 = same partition"), { series: [], xlog: true, xrange: xr, xlabel: xlab, ylabel: "pair mismatch", height: 230, xname: "b" });
  const cJ = lineChart(panel(g2, "(g) Which branch has the lower objective $J$", "above 0: the down (fused) branch is lower"), { series: [], xlog: true, xrange: xr, xlabel: xlab, ylabel: "J difference", height: 230, xname: "b" });
  ui.renderMath(s2);
  const partRow = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 300px), 1fr))", gap: "16px 26px", marginTop: "18px" } });
  s2.append(partRow);
  const pCold = partitionChart(panel(partRow, "(h) Components of each neuron, cold start", "seed 0; neurons sorted so nested components are contiguous"), bg);
  const pDown = partitionChart(panel(partRow, "(i) Components of each neuron, continuation down", "seed 0; sorted the same way, separately"), bg);
  ui.renderMath(partRow);

  function agg(rs, get) {
    const arrs = rs.map(r => get(r).map(num)), n = arrs[0].length;
    const mean = [], lo = [], hi = [];
    for (let k = 0; k < n; k++) { const v = arrs.map(a => a[k]); mean.push(v.reduce((a, b) => a + b, 0) / v.length); lo.push(Math.min(...v)); hi.push(Math.max(...v)); }
    return { mean, lo, hi };
  }
  const ser = (name, x, a, color, extra = {}) => ({ name, x, y: a.mean, band: [a.lo, a.hi], color, width: 1.8, ...extra });
  function draw() {
    const rs = F.runs.filter(r => r.sigma === st.sg && r.pot === st.pot && r.alpha === st.al);
    const br = ["cold", "up", "down"];
    const branches = get => br.map(b => ser(BR_LABEL[b], bg, agg(rs, r => get(r[b])), BR[b]));
    cK.update({ series: [...branches(x => x.K), { name: "multi-member, cold", x: bg, y: agg(rs, r => r.cold.K_multi).mean, color: BR.cold, dash: true, width: 1.3 }], yrange: [0.8, F.meta.config.N * 1.25] });
    cL.update({ series: branches(x => x.largest), yrange: [0, F.meta.config.N] });
    cD.update({ series: [...branches(x => x.diam_max), { name: "cold, evaluation bank", x: bg, y: agg(rs, r => r.cold.diam_max_eval).mean, color: BR.cold, points: true, r: 2.6 }] });
    cS.update({ series: branches(x => x.span_over_range) });
    const mid = bg.slice(0, -1).map((b, k) => Math.sqrt(b * bg[k + 1]));
    cN.update({ series: [ser("cold start", mid, agg(rs, r => r.nest_cold), BR.cold), ser("continuation down", mid, agg(rs, r => r.nest_down), BR.down)] });
    cH.update({ series: [ser("up vs down", bg, agg(rs, r => r.part_diff_up_down), catColor(6))] });
    const dJ = (a, b) => r => r[a].J.map((v, k) => v - r[b].J[k]);
    cJ.update({ series: [ser("J(up) − J(down)", bg, agg(rs, dJ("up", "down")), BR.up), ser("J(cold) − J(down)", bg, agg(rs, dJ("cold", "down")), BR.cold)] });
    const r0 = rs.find(r => r.seed === 0);
    pCold.update(r0.cold.lab); pDown.update(r0.down.lab);
  }
  draw();
}

// ------------------------------------------------------------------ partition diagram (canvas)
// Columns: the b grid; rows: neurons, sorted lexicographically by their component labels from the largest b to the
// smallest, so a nested sequence of partitions draws as contiguous blocks. Cell colour: the component's size rank in
// that column (largest = categorical 0, ...; ranks past 7 and ties fold to "other"); singletons in the wash colour.
function partitionChart(el, bg) {
  el.classList.add("chart");
  const cv = ui.h("canvas"); el.replaceChildren(cv);
  let labs = null, order = null, W = 0, geom = null;
  const m = { l: 38, r: 6, t: 4, b: 34 }, rowH = 5;
  function prep(L) {
    const N = L[0].length, nb = L.length;
    order = Array.from({ length: N }, (_, i) => i).sort((i, j) => { for (let k = nb - 1; k >= 0; k--) { const d = L[k][i] - L[k][j]; if (d) return d; } return i - j; });
  }
  function draw() {
    W = Math.floor(el.clientWidth); if (!W || !labs) return;
    const N = labs[0].length, nb = labs.length, H = m.t + N * rowH + m.b;
    const g = hidpi(cv, W, H); g.clearRect(0, 0, W, H);
    const cw = (W - m.l - m.r) / nb;
    geom = { cw, N, nb };
    for (let k = 0; k < nb; k++) {
      const L = labs[k], size = new Map();
      L.forEach(c => size.set(c, (size.get(c) || 0) + 1));
      const ranked = [...size.entries()].filter(([, s]) => s > 1).sort((a, b) => b[1] - a[1]);
      const rank = new Map(ranked.map(([c], r) => [c, r]));
      const x0 = m.l + k * cw;
      for (let r = 0; r < N; r++) {
        const c = L[order[r]], s = size.get(c);
        g.fillStyle = s === 1 ? P.wash : rank.get(c) < 8 ? catColor(rank.get(c)) : P.other;
        g.fillRect(x0 + 1, m.t + r * rowH, cw - 2, rowH);
        if (r > 0 && L[order[r - 1]] !== c) { g.fillStyle = P.surface; g.fillRect(x0 + 1, m.t + r * rowH - 0.5, cw - 2, 1); }
      }
    }
    g.font = "10.5px Inter, system-ui, sans-serif"; g.fillStyle = P.mute; g.textAlign = "center";
    const lw = Math.max(...bg.map(b => g.measureText(ui.fmt(b, 2)).width)) + 6, every = Math.max(1, Math.ceil(lw / cw));
    for (let k = 0; k < nb; k += every) g.fillText(ui.fmt(bg[k], 2), m.l + (k + 0.5) * cw, m.t + N * rowH + 13);
    g.fillStyle = P.ink2; g.font = "560 11.5px Inter, system-ui, sans-serif";
    g.fillText("fusion scale b", (m.l + W - m.r) / 2, H - 4);
    g.save(); g.translate(12, m.t + N * rowH / 2); g.rotate(-Math.PI / 2); g.fillText("neurons", 0, 0); g.restore();
  }
  new ResizeObserver(() => { if (Math.floor(el.clientWidth) !== W) draw(); }).observe(el);
  cv.addEventListener("pointermove", ev => {
    if (!geom) return;
    const r = cv.getBoundingClientRect(), k = Math.floor((ev.clientX - r.left - m.l) / geom.cw), row = Math.floor((ev.clientY - r.top - m.t) / rowH);
    if (k < 0 || k >= geom.nb || row < 0 || row >= geom.N) { ui.hideTip(); return; }
    const i = order[row], L = labs[k], c = L[i], s = L.filter(x => x === c).length;
    ui.showTip(ev.clientX, ev.clientY, ui.tipRows(`neuron ${i}, b = ${ui.fmt(bg[k], 3)}`, [{ name: "component size", value: String(s) }]));
  });
  cv.addEventListener("pointerleave", () => ui.hideTip());
  return { update(L) { labs = L; prep(L); draw(); } };
}
