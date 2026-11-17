// Chapter "homeo": the precomputed figures (site/data/sweeps/*.json, docs/findings.md). Every number shown is read
// from the JSON at runtime. Phase-diagram cells load their parameter state into the live explorer.
import * as ui from "./ui.js";
import { lineChart, svgEl, scale, niceTicks } from "./charts.js";
import { P, catColor, colorOf } from "./palette.js";

const SW = "data/sweeps/";
const f2 = x => ui.fmtFixed(x, 2), f3 = x => ui.fmt(x, 3);
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
const median = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const get = async f => (await fetch(SW + f)).json();
const loading = el => { if (el) el.replaceChildren(ui.h("p", { class: "mute", text: "loading…" })); };

export async function initMaps(ctx, data, live) {
  const slots = ["phase", "wmin", "omega", "alloc", "beta", "spread", "ingredients", "hyst", "div", "pnoise"].map(s => ctx.slot(s));
  slots.forEach(loading);
  const [phase, alloc, limit, ingr, hyst, jit, div, index] = await Promise.all(
    ["phase.json", "alloc.json", "alloc_limit.json", "alloc_ingredients.json", "hysteresis.json", "hysteresis_jitter.json", "diversity.json", "index.json"].map(get));
  phaseFigure(ctx.slot("phase"), phase, data, live);
  wminFigure(ctx.slot("wmin"), ctx.slot("omega"), phase, live);
  allocFigure(ctx.slot("alloc"), alloc);
  betaFigure(ctx.slot("beta"), limit, alloc);
  spreadFigure(ctx.slot("spread"), ingr, limit);
  ingredientsTable(ctx.slot("ingredients"), ingr, limit, alloc);
  hystFigure(ctx.slot("hyst"), hyst, jit, data);
  divFigure(ctx.slot("div"), div);
  await privateNoise(ctx.slot("pnoise"), index);
  // numbers quoted in the prose, filled from the JSON
  fillNumbers(ctx, { phase, alloc, limit, ingr, div });
  ctx.renderMath(ctx.article);
}

// ---------------------------------------------------------------------------------------- shared: a grid map
// Cells on true (possibly irregular) axes: edges at midpoints between grid values. x is theta (linear); y is linear or log.
function gridMap(el, { xs, ys, ylog = false, cell, color, label, tooltip, onClick, overlays, xlabel, ylabel, yfmt = v => ui.fmt(v, 2), height = 300 }) {
  el.classList.add("chart");
  const hostEl = ui.h("div"); el.replaceChildren(hostEl);
  let W = 0;
  const edges = (v, log) => {
    const t = log ? v.map(Math.log10) : v, e = [];
    e.push(t[0] - (t[1] - t[0]) / 2);
    for (let i = 0; i + 1 < t.length; i++) e.push((t[i] + t[i + 1]) / 2);
    e.push(t[t.length - 1] + (t[t.length - 1] - t[t.length - 2]) / 2);
    return log ? e.map(x => Math.pow(10, x)) : e;
  };
  const ex = edges(xs, false), ey = edges(ys, ylog);
  function draw() {
    W = Math.floor(el.clientWidth); if (!W) return;
    hostEl.replaceChildren();
    const m = { l: 52, r: 12, t: 10, b: 40 }, Hh = height;
    const X = scale("linear", ex[0], ex[ex.length - 1], m.l, W - m.r), Y = scale(ylog ? "log" : "linear", ey[0], ey[ey.length - 1], Hh - m.b, m.t);
    const svg = svgEl("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": `${ylabel} against ${xlabel}` }, hostEl);
    for (let j = 0; j < ys.length; j++) for (let i = 0; i < xs.length; i++) {
      const c = cell(i, j); if (!c) continue;
      const x0 = X(ex[i]), x1 = X(ex[i + 1]), y0 = Y(ey[j]), y1 = Y(ey[j + 1]);
      const r = svgEl("rect", { x: x0 + .5, y: y1 + .5, width: Math.max(0, x1 - x0 - 1), height: Math.max(0, y0 - y1 - 1), fill: color(c), style: onClick ? "cursor:pointer" : null }, svg);
      const lab = label(c);
      if (lab && x1 - x0 > 16 && y0 - y1 > 12) {
        const t = svgEl("text", { x: (x0 + x1) / 2, y: (y0 + y1) / 2 + 4, "text-anchor": "middle", "font-size": 11, fill: c.dark ? "#fff" : P.fg, "pointer-events": "none" }, svg);
        t.textContent = lab;
      }
      r.addEventListener("pointermove", ev => ui.showTip(ev.clientX, ev.clientY, tooltip(c, i, j)));
      r.addEventListener("pointerleave", () => ui.hideTip());
      if (onClick) r.addEventListener("click", () => onClick(c, i, j));
    }
    // axes
    const ax = svgEl("g", { class: "axis" }, svg);
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Hh - m.b + .5, y2: Hh - m.b + .5 }, ax);
    const xt = xs.filter((_, i) => i % Math.max(1, Math.ceil(xs.length * 36 / (W - m.l))) === 0);
    xt.forEach(v => { const t = svgEl("text", { x: X(v), y: Hh - m.b + 14, "text-anchor": "middle" }, svg); t.textContent = ui.fmt(v, 2); });
    ys.forEach(v => { const t = svgEl("text", { x: m.l - 6, y: Y(v) + 3.5, "text-anchor": "end" }, svg); t.textContent = yfmt(v); });
    const tx = svgEl("text", { x: (m.l + W - m.r) / 2, y: Hh - 6, "text-anchor": "middle", class: "ttl" }, svg); tx.textContent = xlabel;
    const ty = svgEl("text", { x: 0, y: 0, "text-anchor": "middle", class: "ttl", transform: `translate(12,${(m.t + Hh - m.b) / 2}) rotate(-90)` }, svg); ty.textContent = ylabel;
    // overlays: [{kind: "line", pts: [[x, y]...], dash, color, label}, {kind: "vline", x, label}]
    const clip = svgEl("clipPath", { id: "cp" + Math.random().toString(36).slice(2) }, svgEl("defs", {}, svg));
    svgEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: Hh - m.t - m.b }, clip);
    const og = svgEl("g", { "clip-path": `url(#${clip.id})`, "pointer-events": "none" }, svg);
    for (const o of overlays || []) {
      if (o.kind === "vline") {
        svgEl("line", { x1: X(o.x), x2: X(o.x), y1: m.t, y2: Hh - m.b, stroke: o.color || P.fg, "stroke-width": 1.2, "stroke-dasharray": o.dash || "2 3" }, og);
      } else {
        const d = o.pts.map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join("");
        svgEl("path", { d, fill: "none", stroke: P.surface, "stroke-width": 4.5, opacity: .8 }, og);
        svgEl("path", { d, fill: "none", stroke: o.color || P.fg, "stroke-width": 2, "stroke-dasharray": o.dash || null }, og);
      }
    }
  }
  const ro = new ResizeObserver(() => { if (Math.floor(el.clientWidth) !== W) draw(); }); ro.observe(el);
  return { redraw: draw };
}
const kColor = (K, kmax) => K === 0 ? P.wash : colorOf("seq", Math.log(1 + K) / Math.log(1 + kmax), 0, 1);
const isDark = (K, kmax) => K > 0 && Math.log(1 + K) / Math.log(1 + kmax) > 0.6;

// ---------------------------------------------------------------------------------------- phase: theta x gamma
function phaseFigure(el, phase, data, live) {
  const meta = phase.meta, thetas = meta.theta;
  const ctl = ui.h("div", { style: { display: "flex", gap: "18px", flexWrap: "wrap", alignItems: "end" } }), mapEl = ui.h("div"), keyEl = ui.h("div");
  el.replaceChildren(ctl, mapEl, keyEl);
  let cov = "block12";
  const seg = ui.segmented({ label: "covariance (synthetic Σ)", options: [{ value: "block12", label: "block12: 3 blocks of 4" }, { value: "hier8", label: "hier8: 8 inputs, 3 levels" }], value: cov, onChange: v => { cov = v; build(); } });
  ctl.append(seg.el);
  function build() {
    const c = phase.covs[cov], ax = c.axes.gamma, gam = ax.values, kmax = cov === "block12" ? 12 : 8;
    const S = data.covs[cov].Sigma, mBlock = (idx) => mean(idx.flatMap(a => idx.map(b => S[a][b])));
    const k0 = c.kappa0;
    // coarse-to-fine thresholds gamma * E_parent (findings F3), and the levels at which finer parts start to exist
    const parents = cov === "block12" ? [{ E: mBlock([0, 1, 2, 3]), lab: "γ·E_block" }] : [{ E: mBlock([0, 1, 2, 3]), lab: "γ·E_half" }, { E: mBlock([0, 1]), lab: "γ·E_pair" }];
    const levels = cov === "block12" ? [0.1, 0.5] : [0.05, 0.3, 0.6];
    const gl = [0, 3.2];
    const overlays = [
      { kind: "line", pts: gl.map(g => [g * k0, g]), dash: "6 4", color: P.fg },
      ...parents.map(pp => ({ kind: "line", pts: gl.map(g => [g * pp.E, g]), color: P.ink2 })),
      ...levels.map(x => ({ kind: "vline", x, color: P.mute })),
    ];
    gridMap(mapEl, { xs: thetas, ys: gam, height: 330, xlabel: "pruning threshold θ", ylabel: "lateral gain γ", yfmt: v => ui.fmt(v, 2),
      cell: (i, j) => { const ks = ax.K[j][i]; if (!ks) return null; const K = median(ks); return { K, ks, dead: ax.dead[j][i], res: ax.res[j][i], maxfrac: ax.maxfrac[j][i], theta: thetas[i], gamma: gam[j], dark: isDark(K, kmax) }; },
      color: q => kColor(q.K, kmax), label: q => q.K === 0 ? "†" : (q.ks.every(k => k === q.ks[0]) ? String(q.K) : `${q.K}*`),
      tooltip: q => ui.tipRows(`θ = ${f2(q.theta)}, γ = ${ui.fmt(q.gamma, 2)}`, [
        { name: "clusters K per seed", value: q.ks.join(", ") }, { name: "dead fraction", value: q.dead.map(x => ui.fmtFixed(x, 2)).join(", ") },
        { name: "largest cluster / N", value: q.maxfrac.map(x => ui.fmtFixed(x, 2)).join(", ") }, { name: "residual", value: ui.fmt(Math.max(...q.res), 2) }],
        "click: load into the explorer"),
      onClick: q => live.load({ cov, theta: q.theta, gamma: q.gamma, omega: +(meta.base.N * meta.base.wmin).toPrecision(2), n: meta.base.N, seed: 1 }, `phase diagram cell θ = ${f2(q.theta)}, γ = ${ui.fmt(q.gamma, 2)}`),
      overlays });
    keyEl.replaceChildren(ui.legend({ items: [
      { label: `θ = γκ₀ (κ₀ = ${f3(k0)}): one blob below it`, color: P.fg, shape: "dash" },
      ...parents.map(pp => ({ label: `θ = ${pp.lab} (E = ${f3(pp.E)}): coarse parts stay bound below it`, color: P.ink2, shape: "line" })),
      { label: `levels where finer parts start to exist: θ = ${levels.join(", ")}`, color: P.mute, shape: "dash" },
      { label: "† every neuron dead · * seeds disagree (median shown)", color: P.wash, shape: "square" },
    ] }).el);
  }
  build();
}

// ---------------------------------------------------------------------------------------- theta x wmin, and the omega collapse
function wminFigure(el, omegaEl, phase, live) {
  const meta = phase.meta, thetas = meta.theta, c = phase.covs.block12, ax = c.axes.wmin, wm = ax.values, N = meta.base.N;
  gridMap(el, { xs: thetas, ys: wm, ylog: true, height: 280, xlabel: "pruning threshold θ", ylabel: "floor w_min", yfmt: v => ui.fmt(v, 1),
    cell: (i, j) => { const ks = ax.K[j][i]; if (!ks) return null; const K = median(ks); return { K, ks, dead: ax.dead[j][i], theta: thetas[i], wmin: wm[j], dark: isDark(K, 12) }; },
    color: q => kColor(q.K, 12), label: q => q.K === 0 ? "†" : (q.ks.every(k => k === q.ks[0]) ? String(q.K) : `${q.K}*`),
    tooltip: q => ui.tipRows(`θ = ${f2(q.theta)}, w_min = ${ui.fmt(q.wmin, 1)} (ω = ${ui.fmt(N * q.wmin, 2)})`, [
      { name: "clusters K per seed", value: q.ks.join(", ") }, { name: "dead fraction", value: q.dead.map(x => ui.fmtFixed(x, 2)).join(", ") }], "click: load into the explorer"),
    onClick: q => live.load({ cov: "block12", theta: q.theta, gamma: 1, omega: +(N * q.wmin).toPrecision(2), n: N, seed: 1 }, `θ×w_min cell θ = ${f2(q.theta)}, ω = ${ui.fmt(N * q.wmin, 2)}`),
    overlays: [{ kind: "vline", x: 0.1, color: P.mute }, { kind: "vline", x: 0.5, color: P.mute }] });
  // extinction threshold (smallest theta at which every neuron is dead in every seed) against omega = N wmin
  const sets = [{ N: 60, a: c.axes.N60 }, { N: 120, a: ax }, { N: 240, a: c.axes.N240 }];
  const series = sets.map((s, k) => {
    const xs = [], ys = [];
    s.a.values.forEach((w, j) => {
      const i = thetas.findIndex((_, ii) => s.a.dead[j][ii] && s.a.dead[j][ii].every(d => d >= 0.999));
      xs.push(s.N * w); ys.push(i >= 0 ? thetas[i] : NaN);
    });
    return { name: `N = ${s.N}`, x: xs, y: ys, points: true, r: 4.5, color: catColor(k) };
  });
  lineChart(omegaEl, { series, xlog: true, yrange: [0.3, 0.85], height: 230, xlabel: "floor leak ω = N·w_min (log)", ylabel: "extinction θ", xname: "ω",
    yfmt: v => ui.fmt(v, 2), refs: [{ y: 0.8, label: "largest θ tested (0.8)" }] });
}

// ---------------------------------------------------------------------------------------- planted allocation curves
function allocFigure(el, alloc) {
  const ctl = ui.h("div", { style: { display: "flex", gap: "18px", flexWrap: "wrap", alignItems: "end" } }), ch = ui.h("div"), note = ui.h("div", { class: "u-help" });
  el.replaceChildren(ctl, ch, note);
  let fam = "var", gam = 0;
  const famS = ui.segmented({ label: "vary block 0's", options: [{ value: "var", label: "variance" }, { value: "size", label: "size" }, { value: "size_eqlam", label: "size, same λ" }], value: fam, onChange: v => { fam = v; draw(); } });
  const gS = ui.segmented({ label: "lateral gain γ", options: [0, 1].map(x => ({ value: x, label: String(x) })), value: gam, onChange: v => { gam = v; draw(); } });
  ctl.append(famS.el, gS.el);
  const chart = lineChart(ch, { series: [{ name: "", x: [1], y: [0] }], xlog: true, yrange: [0, 1], height: 280, ylabel: "fraction of N in block 0", xlabel: "", yfmt: v => ui.fmtPct(v, 0) });
  function draw() {
    const rows = alloc.planted.filter(r => r.family === fam && r.gamma === gam).sort((a, b) => a.theta - b.theta);
    const series = rows.map((r, k) => {
      const lo = r.frac0_mean.map((m, i) => Math.max(0, m - r.frac0_sd[i])), hi = r.frac0_mean.map((m, i) => Math.min(1, m + r.frac0_sd[i]));
      const tag = r.beta_mean == null ? "collective: all in one block" : `β = ${ui.fmtFixed(r.beta_mean, 1)}`;
      return { name: `θ = ${r.theta} (${tag})`, x: r.x, y: r.frac0_mean, band: [lo, hi], dots: true, color: catColor(k) };
    });
    const r0 = rows[0], ref = r0.lam0.map(l => l / (l + 3 * 2.2));
    series.push({ name: "proportional to λ (β = 1)", x: r0.x, y: ref, dash: true, color: P.mute2, width: 1.5 });
    const xl = fam === "var" ? "block 0's variance v₀ (log); its energy λ₀ = 2.2·v₀" : fam === "size" ? "block 0's size m₀ (log); λ₀ = 1 + 0.6(m₀ − 1)" : "block 0's size m₀ (log), variance adjusted so λ₀ = 2.2";
    chart.update({ series, xlabel: xl, xname: fam === "var" ? "v₀" : "m₀" });
    const miss = rows.map(r => `θ = ${r.theta}: ${r.missing.flat().filter(x => x > 0).length} of ${r.missing.flat().length} runs miss a block`).join("; ");
    note.innerHTML = `Seeds: ${alloc.meta.seeds.length} paired; largest residual ${ui.fmt(Math.max(...rows.map(r => r.res_max)), 1)}. Missing modes (a planted block with no neurons): ${miss}.`;
  }
  draw();
}

// ---------------------------------------------------------------------------------------- beta against theta
function betaFigure(el, limit, alloc) {
  const th = limit.meta.thetas.filter(t => t > 0);
  const pick = (shape, scale) => th.map(t => { const r = limit.rows.find(q => q.theta === t && q.shape === shape && q.scale === scale); return r && r.beta != null ? r.beta : NaN; });
  const planted = alloc.planted.filter(r => r.family === "var" && r.gamma === 0).sort((a, b) => a.theta - b.theta);
  const desc = alloc.descriptive_fits.filter(r => r.cov === "powerlaw_pos12" && r.gamma === 0 && r.fit && r.fit.lam_top);
  lineChart(el, { height: 280, yrange: [0, 11], xrange: [0, 0.52], xlabel: "pruning threshold θ", ylabel: "allocation exponent β", xname: "θ", yfmt: v => ui.fmt(v, 2),
    refs: [{ y: 1, label: "proportional (β = 1)" }],
    series: [
      { name: "diag(v), Exp init, N → ∞", x: th, y: pick("exp", 0.2), dots: true, color: P.cat[0] },
      { name: "same, init ×0.1 and ×10", x: th, y: pick("exp", 0.02), dash: true, width: 1.2, color: P.cat[0] },
      { name: "uniform init", x: th, y: pick("uniform", 0.2), dots: true, color: P.cat[1] },
      { name: "log-normal init (log-sd 1)", x: th, y: pick("lognormal", 0.2), dots: true, color: P.cat[2] },
      { name: "planted blocks, N = 150", x: planted.map(r => r.theta), y: planted.map(r => r.beta_mean), points: true, r: 5, color: P.cat[3] },
      { name: "power law (nonneg.), N = 120", x: desc.map(r => r.theta), y: desc.map(r => r.fit.lam_top.beta), points: true, r: 5, color: P.cat[4] },
    ] });
}

function spreadFigure(el, ingr, limit) {
  const thetas = [0.1, 0.2, 0.3];
  const series = thetas.map((t, k) => {
    const rows = ingr.rows.filter(r => r.tag === "init_logsigma" && r.theta === t).sort((a, b) => a.kw.logsig - b.kw.logsig);
    return { name: `θ = ${t}, log-normal`, x: rows.map(r => r.kw.logsig), y: rows.map(r => { const b = r.by_time["4000"].beta; return b == null ? NaN : b; }), dots: true, color: catColor(k) };
  });
  const sd = Math.PI / Math.sqrt(6);
  series.push({ name: "exponential init (log-sd π/√6)", x: thetas.map(() => sd), y: thetas.map(t => limit.rows.find(r => r.theta === t && r.shape === "exp" && r.scale === 0.2).beta), points: true, r: 5.5, color: P.fg });
  lineChart(el, { series, height: 280, yrange: [0, 12], xrange: [0, 3.2], xlabel: "spread of the initial input weights: sd of log W", ylabel: "allocation exponent β", xname: "log-sd", yfmt: v => ui.fmt(v, 2),
    refs: [{ y: 1, label: "β = 1" }] });
}

// ---------------------------------------------------------------------------------------- ingredients table
function ingredientsTable(el, ingr, limit, alloc) {
  const R = (tag, th, key, val) => ingr.rows.find(r => r.tag === tag && r.theta === th && r.kw[key] === val);
  const b = r => { const x = r && r.by_time[Object.keys(r.by_time).sort((a, c) => +a - +c).pop()]; return x && x.beta != null ? ui.fmtFixed(x.beta, 1) : "–"; };
  const fr = r => r ? r.by_time[Object.keys(r.by_time).sort((a, c) => +a - +c).pop()].frac : null;
  const L = (t, sh, sc) => limit.rows.find(r => r.theta === t && r.shape === sh && r.scale === sc);
  const pv = (fam, th, g) => alloc.planted.find(r => r.family === fam && r.theta === th && r.gamma === g);
  const loudOnly = f => f && f[f.length - 1] > 0.999 ? "all neurons to the loudest input" : null;
  const dead = f => f && f.every(x => x === 0) ? "every neuron dead" : null;
  const other = r => { const x = r.by_time["4000"]; return x.other != null ? ui.fmtPct(x.other, 0) : "–"; };
  const rows = [
    ["θ (with exponential init)", `θ = 0.02 → 0.45 (N → ∞, γ = 0)`, `${L(0.02, "exp", 0.2).beta.toFixed(1)} → ${L(0.45, "exp", 0.2).beta.toFixed(1)}`, "sets β: steep at low θ, softening towards the smallest variance; hard cutoff at c ≤ 0", "yes"],
    ["init scale", "mean 0.02 / 0.2 / 2, θ = 0.2", [0.02, 0.2, 2].map(s => L(0.2, "exp", s).beta.toFixed(1)).join(" / "), "none", "no"],
    ["init spread", "log-sd 0.5 / 1 / 2 / 3, θ = 0.2", [0.5, 1, 2, 3].map(s => b(R("init_logsigma", 0.2, "logsig", s))).join(" / "), "sets β; β ≈ 1 only by tuning the spread against θ", "yes"],
    ["init shape", "uniform vs exponential, θ = 0.2", `${L(0.2, "uniform", 0.2).beta.toFixed(1)} vs ${L(0.2, "exp", 0.2).beta.toFixed(1)}`, "narrower start, steeper filter", "yes"],
    ["lateral gain γ", "planted blocks, θ = 0.25: γ = 0 vs 1", `${pv("var", 0.25, 0).beta_mean.toFixed(1)} vs ${pv("var", 0.25, 1).beta_mean.toFixed(1)}`, "none above the collective threshold; below it, winner-take-all (θ = 0.05, 0.15 with γ = 1)", "only via the regime"],
    ["shared minibatch noise", "batch 8 / 32 / 128, θ = 0.2", [8, 32, 128].map(s => loudOnly(fr(R("noise_batch", 0.2, "batch", s))) ? "WTA" : b(R("noise_batch", 0.2, "batch", s))).join(" / "), "steeper, never towards proportional", "yes (worse)"],
    ["ρ = −w^p, p = 0.5", "θ = 0.1 / 0.2 / 0.3", [0.1, 0.2, 0.3].map(t => b(R("rho_power", t, "par", 0.5))).join(" / "), "a little flatter", "yes"],
    ["ρ = −w^p, p = 2, 3", "θ = 0.2", `${b(R("rho_power", 0.2, "par", 2))}, ${b(R("rho_power", 0.2, "par", 3))}`, `neurons left in mixed states: ${other(R("rho_power", 0.2, "par", 2))}, ${other(R("rho_power", 0.2, "par", 3))}`, "yes, not robust"],
    ["soft floor ρ = −w + ε/w", "ε = 10⁻³ / 10⁻² / 0.03, θ = 0.2", [0.001, 0.01, 0.03].map(e => { const f = fr(R("rho_softfloor", 0.2, "par", e)); return dead(f) ? "dead" : b(R("rho_softfloor", 0.2, "par", e)); }).join(" / "), "steeper, or lethal", "yes, not robust"],
    ["private (per-synapse) noise", "–", "not measured", "open; see the hook below", "?"],
  ];
  const tb = ui.h("table", { class: "data" },
    ui.h("thead", {}, ui.h("tr", {}, ...["ingredient", "setting", "β", "effect on allocation", "changes β?"].map(t => ui.h("th", { text: t })))),
    ui.h("tbody", {}, ...rows.map(r => ui.h("tr", {}, ui.h("td", { text: r[0] }), ui.h("td", { text: r[1] }), ui.h("td", { class: "num", text: r[2] }), ui.h("td", { text: r[3] }), ui.h("td", { text: r[4] })))));
  el.replaceChildren(ui.h("div", { class: "table-scroll" }, tb));
}

// ---------------------------------------------------------------------------------------- hysteresis loops
function hystFigure(el, hyst, jit, data) {
  const ctl = ui.h("div", { style: { display: "flex", gap: "18px", flexWrap: "wrap", alignItems: "end" } }), ch = ui.h("div");
  el.replaceChildren(ctl, ch);
  let cov = "block12", gam = 0, sig = 0;
  const cS = ui.segmented({ label: "covariance (synthetic Σ)", options: ["block12", "hier8"], value: cov, onChange: v => { cov = v; draw(); } });
  const gS = ui.segmented({ label: "lateral gain γ", options: [0, 1].map(x => ({ value: x, label: String(x) })), value: gam, onChange: v => { gam = v; draw(); } });
  const sS = ui.segmented({ label: "private jitter σ", options: [{ value: 0, label: "none" }, { value: 0.001, label: "10⁻³" }, { value: 0.01, label: "10⁻²" }], value: sig, onChange: v => { sig = v; draw(); } });
  ctl.append(cS.el, gS.el, sS.el);
  const chart = lineChart(ch, { series: [{ name: "", x: [0], y: [0] }], height: 280, xlabel: "pruning threshold θ (swept up, then down)", ylabel: "clusters K", xname: "θ", yrange: [0, 14] });
  function draw() {
    const L = sig ? jit.loops.find(l => l.cov === cov && l.gamma === gam && l.sigma === sig) : hyst.loops.find(l => l.cov === cov && l.gamma === gam && l.seed === 0);
    const up = L.path.filter(q => q.dir === "up"), dn = L.path.filter(q => q.dir === "down");
    const parts = data.covs[cov].parts.filter(r => r.theta >= 0.02 - 1e-9 && r.theta <= 0.8 + 1e-9 && Math.round(r.theta * 100) % 2 === 0);
    chart.update({ series: [
      { name: "θ rising", x: up.map(q => q.theta), y: up.map(q => q.K), dots: true, color: P.cat[0], r: 2.5 },
      { name: "θ falling", x: dn.map(q => q.theta), y: dn.map(q => q.K), dots: true, color: P.cat[1], dash: true, r: 2.5 },
      { name: "stable parts that exist", x: parts.map(r => r.theta), y: parts.map(r => r.n), color: P.mute2, width: 1.2 },
    ], yrange: [0, cov === "block12" ? 16 : 10] });
  }
  draw();
}

// ---------------------------------------------------------------------------------------- diversity
function divFigure(el, div) {
  const row = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "14px 26px" } });
  const a = ui.h("div"), b = ui.h("div"), note = ui.h("div", { class: "u-help", style: { marginTop: "6px" } });
  row.append(ui.h("div", {}, ui.h("div", { class: "chart-title", text: "(a) distinct functions per planted block (participation ratio)" }), a),
    ui.h("div", {}, ui.h("div", { class: "chart-title", text: "(b) clusters K the population splits into" }), b));
  el.replaceChildren(row, note);
  const ax = div.meta.axes, ser = (key, axis, k, name) => {
    const rows = div.axes[axis];
    return { name, x: rows.map(r => r.val), y: rows.map(r => mean(r[key])), band: [rows.map(r => Math.min(...r[key])), rows.map(r => Math.max(...r[key]))], dots: true, color: catColor(k) };
  };
  lineChart(a, { height: 230, xlabel: "connection probability p (1 = dense)", ylabel: "block-level PR", xname: "p", yrange: [0.9, 2.3], refs: [{ y: 1, label: "identical copies" }],
    series: [ser("block_pr", "p", 0, "γ = 1"), ser("block_pr", "p_gamma0", 1, "γ = 0")] });
  lineChart(b, { height: 230, xlabel: "connection probability p (1 = dense)", ylabel: "clusters K", xname: "p", yrange: [0, 25],
    series: [ser("K", "p", 0, "γ = 1"), ser("K", "p_gamma0", 1, "γ = 0")] });
  const maxL1 = axis => Math.max(...div.axes[axis].flatMap(r => r.l1));
  const prOK = ["wmin", "gamma", "batch", "N"].every(axn => div.axes[axn].every(r => r.pr.every(x => Math.abs(x - 1) < 1e-6)));
  note.innerHTML = `Dense wiring ($p = 1$): largest member-to-centroid $L_1$ distance over every $w_{\\min}$ (${ax.wmin.map(x => ui.fmt(x, 1)).join(", ")}): ${ui.fmt(maxL1("wmin"), 1)}; over $\\gamma$ (${ax.gamma.join(", ")}): ${ui.fmt(maxL1("gamma"), 1)}; over $N$ (${ax.N.join(", ")}): ${ui.fmt(maxL1("N"), 1)}; over minibatch sizes (${ax.batch.join(", ")}): ${ui.fmt(maxL1("batch"), 1)}. Within-cluster participation ratio ${prOK ? "exactly 1.000 in all of them" : "see JSON"}.`;
}

// ---------------------------------------------------------------------------------------- private noise: a hook
async function privateNoise(el, index) {
  const entry = index.files.find(f => f.file === "alloc_private_noise.json");
  if (!entry) {
    el.replaceChildren(ui.h("p", { class: "placeholder", html: `<span class="ph-tag">Hook</span> Allocation under <em>private</em> per-synapse noise has not been measured yet (findings, "Inconclusive"). A follow-up sweep is expected at <code>data/sweeps/alloc_private_noise.json</code>; when it is listed in <code>data/sweeps/index.json</code>, this space shows it.` }));
    return;
  }
  const d = await get("alloc_private_noise.json");
  const rows = (d.rows || []).filter(r => Number.isFinite(r.beta) && Number.isFinite(r.theta));
  el.replaceChildren(ui.h("p", { class: "u-help", text: `${entry.description} (script: ${entry.script || "n/a"}).` }));
  if (!rows.length) { el.append(ui.h("p", { class: "mute", text: "No rows with (theta, beta) in this file yet; see the JSON." })); return; }
  const sigmas = [...new Set(rows.map(r => r.sigma ?? r.kw?.sigma ?? 0))].sort((a, b) => a - b);
  const box = ui.h("div"); el.append(box);
  lineChart(box, { height: 240, xlabel: "pruning threshold θ", ylabel: "allocation exponent β", xname: "θ", refs: [{ y: 1, label: "β = 1" }],
    series: sigmas.map((s, k) => { const rr = rows.filter(r => (r.sigma ?? r.kw?.sigma ?? 0) === s).sort((a, b) => a.theta - b.theta); return { name: `σ = ${s}`, x: rr.map(r => r.theta), y: rr.map(r => r.beta), dots: true, color: catColor(k) }; }) });
}

// ---------------------------------------------------------------------------------------- numbers in the prose
function fillNumbers(ctx, { phase, alloc, limit, ingr, div }) {
  const L = (t, sh = "exp", sc = 0.2) => limit.rows.find(r => r.theta === t && r.shape === sh && r.scale === sc).beta;
  const pv = (th, g) => alloc.planted.find(r => r.family === "var" && r.theta === th && r.gamma === g);
  const lg = s => ingr.rows.find(r => r.tag === "init_logsigma" && r.theta === 0.2 && r.kw.logsig === s).by_time["4000"].beta;
  const vals = {
    "beta-lo": ui.fmtFixed(L(0.02), 1), "beta-hi": ui.fmtFixed(L(0.45), 1), "beta-04": ui.fmtFixed(L(0.4), 1),
    "beta-025": ui.fmtFixed(pv(0.25, 0).beta_mean, 1), "beta-025g1": ui.fmtFixed(pv(0.25, 1).beta_mean, 1),
    "beta-005": ui.fmtFixed(pv(0.05, 0).beta_mean, 1),
    "spread-3": ui.fmtFixed(lg(3), 2), "spread-05": ui.fmtFixed(lg(0.5), 1),
    "kappa-block": ui.fmt(phase.covs.block12.kappa0, 3),
    "pr-max": ui.fmtFixed(Math.max(...div.axes.p_gamma0.flatMap(r => r.block_pr)), 2),
  };
  ctx.article.querySelectorAll("[data-v]").forEach(e => { if (vals[e.dataset.v] != null) e.textContent = vals[e.dataset.v]; });
}
