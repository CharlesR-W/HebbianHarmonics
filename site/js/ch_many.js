// Chapter "many": how the picture scales with many inputs, and what "resolution" means.
//   * precomputed: data/sweeps2/fusion.json (N = 40 networks swept in b: cold start, continuation up and down)
//     and data/sweeps2/resolution.json (the three resolution statements);
//   * live: the population sim (ch_many_live.js) and the resolution figures, including the line along one mode
//     (ch_many_res.js). Model code without DOM is in ch_many_model.js.
// Every number in the prose is filled from the JSON at load time (spans with data-mv="key").
import * as ui from "./ui.js";
import { lineChart, hidpi } from "./charts.js";
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

function numbers(F, R) {
  // linear neurons only: the sweep also holds runs with rectifying neurons, which this page does not discuss
  const c = F.meta.config, bg = c.b_grid, runs = F.runs.filter(r => r.pot === "linear"), o = {};
  o.fN = c.N; o.fB = c.B; o.fs = c.s; o.fSlin = F.meta.seeds.linear.length;
  o.fnb = bg.length; o.fbmin = f3(bg[0]); o.fbmax = f3(bg[bg.length - 1]);
  o.fS = c.S_train;
  // convergence, recomputed over the linear runs (fusion.json's summary covers all runs)
  const BRS = ["cold", "up", "down"];
  let nRuns = 0, nConv = 0, maxRes = 0, maxDev = 0, minSlack = Infinity;
  for (const r of runs) for (const br of BRS) r[br].residual.forEach((res, k) => {
    nRuns++; maxRes = Math.max(maxRes, res);
    if (!r[br].converged[k]) return;
    nConv++; maxDev = Math.max(maxDev, num(r[br].dev_live[k]) || 0);
    const sl = num(r[br].slack_dead[k]); if (sl != null && Number.isFinite(sl)) minSlack = Math.min(minSlack, sl);
  });
  o.fconv = `${nConv} of ${nRuns}`; o.fmaxres = f2(maxRes); o.fdevlive = f2(maxDev); o.fslack = f2(minSlack);
  const all = br => runs.flatMap(r => r[br].span_over_range.map(num));
  o.fchain = f3(Math.max(...BRS.flatMap(all)));
  // hysteresis windows: b where some seed's up and down partitions differ
  const win = (sg, al) => { const bs = bg.filter((_, k) => runs.some(r => r.sigma === sg && r.alpha === al && r.part_diff_up_down[k] > 0)); return bs.length ? `${f2(bs[0])}–${f2(bs[bs.length - 1])}` : "none"; };
  o.fwin01b = win("blocks", 0.1); o.fwin01p = win("power", 0.1); o.fwin1b = win("blocks", 1); o.fwin1p = win("power", 1);
  // which branch is lower: J(up) - J(down), seed means per configuration and b
  const sel = (sg, al) => runs.filter(r => r.sigma === sg && r.alpha === al);
  let best = { v: -Infinity }, worst = { v: Infinity };
  for (const sg of ["blocks", "power"]) for (const al of c.alphas) {
    const rs = sel(sg, al);
    bg.forEach((b, k) => {
      const v = rs.reduce((a, r) => a + r.up.J[k] - r.down.J[k], 0) / rs.length;
      if (v > best.v) best = { v, b, sg, al };
      if (v < worst.v) worst = { v, b, sg, al };
    });
  }
  const sgName = sg => (sg === "power" ? "power law" : "blocks");
  o.fJbest = f2(best.v); o.fJbestcfg = `${sgName(best.sg)}, α = ${best.al}, b = ${f2(best.b)}`;
  o.fJworst = f2(-worst.v);   // the up branch is lower by this much o.fJworstcfg = `${sgName(worst.sg)}, α = ${worst.al}, b = ${f2(worst.b)}`;

  // ---- resolution.json
  o.rS = R.meta.config.S_train;
  const p2 = R.p2;
  o.r2n = p2.n_pairs; o.r2mc = p2.misclassified_by_Chat_threshold; o.r2ms = p2.misclassified_by_Sigma_threshold;
  o.r2b = f3(p2.b); o.r2a = f3(p2.alpha); o.rseeds = R.meta.seeds.length;
  const rr = (al, b) => R.p3a.ratio_rows.find(r => r[0] === al && r[1] === b)?.[2];
  o.r3r01 = f3(rr(0.1, 0.1)); o.r3r001 = f3(rr(0.01, 0.1)); o.r3r1 = f3(rr(1, 0.1));
  const p3 = R.p3a.rows;
  { const l = [...new Set(p3.map(r => r[1]))]; o.r3lamratio = f2(Math.max(...l) / Math.min(...l)); }
  { const modes = new Set(p2.rows_seed0.map(r => r[0])); o.r2nmodes = modes.size; o.r2nsep = p2.rows_seed0.length / modes.size; }
  o.r3offmax = f2(Math.max(...p3.map(r => Math.abs(r[7] - r[6]) / r[8])));
  o.r3onmax = f2(Math.max(...p3.map(r => Math.abs(r[5] - r[4]) / r[8])));
  // population on a line
  const pop = R.pop;
  let cases = 0, below = 0, ex = -1, bridge = 0;
  pop.forEach(p => p.cold.forEach(c2 => { cases++; if (c2[0] < c2[1]) below++; ex = Math.max(ex, c2[0] - c2[1]); bridge = Math.max(bridge, c2[2]); }));
  o.r4cases = cases; o.r4below = below; o.r4ex = ex; o.r4bridge = f3(bridge);
  const bgp = R.meta.config.population.b_grid;
  o.r4N = R.meta.config.population.N; o.r4nb = bgp.length;
  // condensation: alpha = 0.01, grid points where the seed-mean predicted K <= 1.8
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

function buildSweep(ctx, F) {
  const bg = F.meta.config.b_grid;
  // each figure has its own two controls (inputs, alpha) and URL keys; the sweep opens at alpha = 1 (chains are
  // widest there), the partitions at alpha = 0.1 (the hysteresis window is widest there)
  const readSt = (pre, al0) => ({ sg: ctx.param(pre + "sg", "blocks") === "power" ? "power" : "blocks", pot: "linear",
    al: ctx.param(pre + "al", al0) === 1 ? 1 : ctx.param(pre + "al", al0) === 0.1 ? 0.1 : al0 });
  const st = readSt("", 1), st2 = readSt("h", 0.1);
  function controlBar(parent, S, pre, al0, redraw) {
    const sg = ui.segmented({ label: "inputs", options: SG_OPTS, value: S.sg, onChange: v => { S.sg = v; write(); } });
    const al = ui.segmented({ label: "stiffness $\\alpha$", options: [{ value: 0.1, label: "0.1" }, { value: 1, label: "1" }], value: S.al, onChange: v => { S.al = v; write(); } });
    function write() { ctx.setParams({ [pre + "sg"]: S.sg === "blocks" ? null : S.sg, [pre + "al"]: S.al === al0 ? null : S.al }); redraw(); }
    parent.append(ui.h("div", { style: { display: "flex", flexWrap: "wrap", gap: "8px 22px", alignItems: "center", margin: "4px 0 10px", font: "13px/1.4 var(--sans)" } }, sg, al));
  }
  const grid = (w = 400) => ui.h("div", { style: { display: "grid", gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, ${w}px), 1fr))`, gap: "16px 26px" } });
  const panel = (g, title, sub) => { const b = ui.h("div"); g.append(ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", html: title }), sub ? ui.h("div", { class: "u-help", style: { margin: "-2px 0 4px" }, html: sub }) : null, b)); return b; };

  // figure 1: number of components and the chain index
  const s1 = ctx.slot("sweep"); controlBar(s1, st, "", 1, () => draw());
  const g1 = grid(320); s1.append(g1);
  const xr = [bg[0] * 0.85, bg[bg.length - 1] * 1.15], xlab = "fusion scale b";
  const cK = lineChart(panel(g1, "(a) Number of components $K$", "log scale; dashed: components with two or more members (cold start)"), { series: [], xlog: true, ylog: true, xrange: xr, xlabel: xlab, ylabel: "K", height: 240, xname: "b" });
  const cS = lineChart(panel(g1, "(b) Span of the widest component, in edge ranges $\\sqrt{2b}$", "above 1: a chain, wider than one edge's range"), { series: [], xlog: true, xrange: xr, xlabel: xlab, ylabel: "span / √(2b)", height: 240, xname: "b", refs: [{ y: 1, label: "one edge range" }] });
  ui.renderMath(s1);

  // figure 2: the partitions along b, cold start and continuation down (seed 0)
  const s2 = ctx.slot("history"); controlBar(s2, st2, "h", 0.1, () => draw());
  const partRow = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 300px), 1fr))", gap: "16px 26px", marginTop: "6px" } });
  s2.append(partRow);
  const pCold = partitionChart(panel(partRow, "From $W = 0$ at every $b$", "cold start"), bg);
  const pDown = partitionChart(panel(partRow, "Coming down from the fused state", "continuation down, carrying $W$"), bg);
  ui.renderMath(s2);

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
    cK.update({ series: [...branches(x => x.K), { name: "two or more members, cold", x: bg, y: agg(rs, r => r.cold.K_multi).mean, color: BR.cold, dash: true, width: 1.3 }], yrange: [0.8, F.meta.config.N * 1.25],
      refs: st.sg === "blocks" ? [{ y: 4, label: "4 planted blocks" }] : [] });
    cS.update({ series: branches(x => x.span_over_range) });
    const r0 = F.runs.find(r => r.sigma === st2.sg && r.pot === "linear" && r.alpha === st2.al && r.seed === 0);
    const ord = sharedOrder(r0.cold.lab, r0.down.lab), d = r0.cold.lab.map((L, k) => partitionsDiffer(L, r0.down.lab[k]));
    pCold.update(r0.cold.lab, ord, d); pDown.update(r0.down.lab, ord, d);
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
  let labs = null, order = null, diff = null, W = 0, geom = null;
  const m = { l: 38, r: 6, t: 14, b: 34 }, rowH = 5;
  function draw() {
    W = Math.floor(el.clientWidth); if (!W || !labs) return;
    const N = labs[0].length, nb = labs.length, H = m.t + N * rowH + m.b;
    const g = hidpi(cv, W, H); g.clearRect(0, 0, W, H);
    const cw = (W - m.l - m.r) / nb;
    geom = { cw, N, nb };
    for (let k = 0; k < nb; k++) {
      if (diff && diff[k]) {             // a column whose partition differs from the other panel's
        const xc = m.l + (k + 0.5) * cw;
        g.fillStyle = P.fg; g.beginPath(); g.moveTo(xc - 5, 2); g.lineTo(xc + 5, 2); g.lineTo(xc, 10); g.closePath(); g.fill();
      }
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
  return { update(L, ord, d) { labs = L; order = ord; diff = d; draw(); } };
}

// one neuron order for both panels: lexicographic in the first set of partitions (largest b first), then the second
function sharedOrder(A, B) {
  const N = A[0].length, nb = A.length;
  const cmp = (L, i, j) => { for (let k = nb - 1; k >= 0; k--) { const d = L[k][i] - L[k][j]; if (d) return d; } return 0; };
  return Array.from({ length: N }, (_, i) => i).sort((i, j) => cmp(A, i, j) || cmp(B, i, j) || i - j);
}
// true where two partitions of the same neurons differ (some pair together in one and apart in the other)
function partitionsDiffer(a, b) {
  const n = a.length;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if ((a[i] === a[j]) !== (b[i] === b[j])) return true;
  return false;
}
