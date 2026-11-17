// Chapter "method": fills the choices table and the sizes table from the sweep JSONs' own configs, and runs a set
// of solver and fixed-point checks live in the reader's browser (onsite_engine.js), on demand.
import * as ui from "./ui.js";
import * as O from "./onsite_engine.js";

const SWEEPS = ["fusion", "resolution", "diversity", "eigen", "allocation", "readout"];

export async function init(panel, ctx) {
  stackOnPhones(ctx);
  initChecks(ctx);
  const box = ctx.slot("sizes");
  box.textContent = "loading…";
  const data = {};
  await Promise.all(SWEEPS.map(async n => {
    try {
      const r = await fetch(`data/sweeps2/${n}.json`);
      if (!r.ok) return;
      const j = await r.json(); data[n] = j.meta;
      if (n === "resolution" && Array.isArray(j.pop)) data[n]._popAlphas = [...new Set(j.pop.map(x => x.alpha))].sort((x, y) => x - y);
    } catch (e) { /* reported below */ }
  }));
  fillValues(ctx, data);
  sizesTable(box, data);
}

// ------------------------------------------------------------ the choices table as stacked blocks on phones
// Four columns of prose do not fit a phone; below 700 px the same cells are shown one decision at a time.
function stackOnPhones(ctx) {
  const table = ctx.article.querySelector("#method-table"), wrap = table && table.closest(".table-scroll");
  if (!wrap) return;
  const heads = ["", "Default", "Alternative", "What it would change"];
  const cards = ui.h("div", { class: "method-cards" });
  for (const tr of table.querySelectorAll("tbody tr")) {
    const td = [...tr.children];
    const card = ui.h("div", { style: { borderTop: "1px solid var(--line)", padding: "12px 0 4px", font: "14px/1.5 var(--sans)" } });
    card.append(ui.h("div", { style: { fontWeight: "650", marginBottom: "4px" }, html: td[0].innerHTML }));
    for (let k = 1; k < td.length; k++) {
      const row = ui.h("p", { style: { margin: "0 0 8px" } });
      row.append(ui.h("span", { style: { fontSize: "11px", letterSpacing: ".06em", textTransform: "uppercase", color: "var(--mute)", marginRight: "6px" }, text: heads[k] }));
      const body = ui.h("span", { html: td[k].innerHTML });
      row.append(body); card.append(row);
    }
    cards.append(card);
  }
  wrap.after(cards);
  const narrow = window.matchMedia("(max-width: 700px)");
  const apply = () => { wrap.hidden = narrow.matches; cards.hidden = !narrow.matches; };
  apply(); narrow.addEventListener("change", apply);
}

// ------------------------------------------------------------ values quoted in the table, from the configs
function fillValues(ctx, d) {
  const f = d.fusion?.config, rect = f?.pots?.rect;
  const v = {
    s: f?.s, deg: f?.deg, klo: rect?.klo, khi: rect?.khi, st: rect?.sigma_t,
    Strain: f && ui.fmtInt(f.S_train), Seval: f && ui.fmtInt(f.S_eval),
    dt: f?.dt?.replace("alpha", "α"), runtol: f && ui.fmt(f.run_tol, 1), restol: d.fusion && ui.fmt(d.fusion.res_tol, 1),
    maxsteps: f && ui.fmtInt(f.max_steps), streams: d.fusion?.seed_streams,
  };
  // ranges over the network sweeps (allocation (c) is not a network and is left out)
  const nets = [];
  if (f) nets.push({ N: [f.N], B: f.B, seeds: [d.fusion.seeds.linear.length, d.fusion.seeds.rect.length], grid: f.b_grid.length });
  const r = d.resolution?.config;
  if (r) nets.push({ N: [r.population.N], B: r.B, seeds: [d.resolution.seeds.length], grid: r.population.b_grid.length });
  const dv = d.diversity?.config;
  if (dv) nets.push({ N: [dv.N], B: dv.B, seeds: [d.diversity.seeds.rank1.length, d.diversity.seeds.multi.length], grid: dv.b_grid.length });
  const e = d.eigen?.config;
  if (e) nets.push({ N: [e.N], B: e.B, seeds: [d.eigen.seeds.partB.length], grid: e.b_grid.length });
  const a = d.allocation?.config;
  if (a) nets.push({ N: [a.a.N_random, a.b.N], B: a.a.B, seeds: [d.allocation.seeds.a.length, d.allocation.seeds.b.length] });
  const ro = d.readout?.config;
  if (ro) nets.push({ N: ro.Ns, B: ro.B, seeds: [d.readout.seeds.length], grid: ro.b_grid.length });
  if (nets.length === SWEEPS.length) {
    const span = xs => { const lo = Math.min(...xs), hi = Math.max(...xs); return lo === hi ? String(lo) : `${lo} to ${hi}`; };
    const Ns = nets.flatMap(x => x.N), Bs = nets.map(x => x.B), Ss = nets.flatMap(x => x.seeds), Gs = nets.filter(x => x.grid).map(x => x.grid);
    Object.assign(v, { nrange: span(Ns), nmax: String(Math.max(...Ns)), brange: span(Bs), bmax: String(Math.max(...Bs)), seedrange: span(Ss), gridrange: span(Gs) });
  }
  ctx.article.querySelectorAll("[data-mm]").forEach(sp => {
    const x = v[sp.dataset.mm];
    sp.textContent = x == null ? "(not loaded)" : typeof x === "number" ? ui.fmt(x, 3) : x;
  });
}

function sizesTable(box, d) {
  const rng = a => (a && a.length ? `${a.length} (${ui.fmt(Math.min(...a), 2)}–${ui.fmt(Math.max(...a), 2)})` : "—");
  const set = a => (a && a.length ? a.map(x => ui.fmt(x, 2)).join(", ") : "—");
  const n = a => (Array.isArray(a) ? a.length : a);
  const rows = [];
  const f = d.fusion?.config;
  if (f) rows.push(["fusion", "Q-fusion: N neurons, cold / up / down", f.N, f.B, `${n(d.fusion.seeds.linear)} linear, ${n(d.fusion.seeds.rect)} rectifier`, rng(f.b_grid), set(f.alphas), `${ui.fmtInt(f.S_train)} / ${ui.fmtInt(f.S_eval)}`]);
  const r = d.resolution?.config;
  if (r) rows.push(["resolution", "planted pairs; a line of neurons along one mode", `2; ${r.population?.N ?? "—"}`, r.B, n(d.resolution.seeds), `${ui.fmt(r.b_activation, 2)}; ${rng(r.population?.b_grid)}`, `${ui.fmt(r.alpha_activation, 2)}; ${set(d.resolution._popAlphas)}`, `${ui.fmtInt(r.S_train)} / ${ui.fmtInt(r.S_eval)}`]);
  const dv = d.diversity?.config;
  if (dv) rows.push(["diversity", "rank-one drive, three potentials; multi-mode", dv.N, dv.B, `${n(d.diversity.seeds.rank1)}; ${n(d.diversity.seeds.multi)}`, rng(dv.b_grid), set(dv.alphas), `${ui.fmtInt(dv.S_train)} / ${ui.fmtInt(dv.S_eval)}`]);
  const e = d.eigen?.config;
  if (e) rows.push(["eigen", "population modes vs clusters vs Σ modes", e.N, e.B, n(d.eigen.seeds.partB), rng(e.b_grid), set([e.alpha]), `${ui.fmtInt(e.S_train)} / ${ui.fmtInt(e.S_eval)}`]);
  const a = d.allocation?.config;
  if (a) {
    rows.push(["allocation (a)", "base model, block attachments", `${a.a.N_random} (random)`, a.a.B, n(d.allocation.seeds.a), "—", set([a.a.alpha]), "—"]);
    rows.push(["allocation (b)", "plastic attachments", a.b.N, a.b.B, n(d.allocation.seeds.b), "—", "—", "—"]);
    const sp = a.c.spectra, Ns = a.c.N;
    rows.push(["allocation (c)", "added principle (not a network)", Object.keys(Ns).map(k => Ns[k]).join(", "), Object.keys(sp).map(k => sp[k].length).join(", "), d.allocation.seeds.c, "—", "—", "—"]);
  }
  const ro = d.readout?.config;
  if (ro) rows.push(["readout", "covariance from weights", set(ro.Ns), ro.B, n(d.readout.seeds), rng(ro.b_grid), set([ro.alpha]), `${ui.fmtInt(ro.S_train)} / —`]);
  const head = ["sweep", "what", "N", "B", "seeds", "values of b (range)", "α", "banks: training / evaluation"];
  const t = ui.h("table", { class: "data" });
  const tr = ui.h("tr"); head.forEach(hd => tr.append(ui.h("th", { text: hd }))); t.append(ui.h("thead", {}, tr));
  const tb = ui.h("tbody");
  rows.forEach(rw => { const r_ = ui.h("tr"); rw.forEach((c, k) => r_.append(ui.h("td", { class: k >= 2 && k !== 5 ? "num" : null, text: String(c) }))); tb.append(r_); });
  t.append(tb);
  const missing = SWEEPS.filter(s => !d[s]);
  box.replaceChildren(ui.h("div", { class: "table-scroll" }, t));
  if (missing.length) box.append(ui.h("p", { class: "u-help", text: `Not loaded: ${missing.join(", ")}.` }));
}

// ------------------------------------------------------------ live checks
const CHECK = { N: 40, B: 12, S: 256, Seval: 1024, b: 0.1, alpha: 1, klo: 3, khi: 0, sigmaT: 0.7, seed: 0, tol: 1e-10, stop: 1e-9, claim: 1e-6, fdStep: 1e-4, maxSteps: 4000 };

function initChecks(ctx) {
  const slot = ctx.slot("checks");
  const btn = ui.h("button", { type: "button", class: "u-chip", text: "Run the checks" });
  const status = ui.h("span", { class: "u-help", style: { marginLeft: "10px" }, text: "Not run yet. Takes a few seconds; the page stays responsive." });
  const out = ui.h("div");
  slot.append(ui.h("div", { style: { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "6px" } }, btn, status), out);
  let running = false;
  btn.addEventListener("click", async () => {
    if (running) return; running = true; btn.disabled = true;
    try { await runChecks(out, status); } catch (e) { console.error(e); status.textContent = `Failed: ${e.message}`; }
    running = false; btn.disabled = false; btn.textContent = "Run again";
  });
}

const nextFrame = () => new Promise(r => requestAnimationFrame(() => r()));

async function runChecks(out, status) {
  const c = CHECK;
  const { Sigma } = O.covBlocks({ sizes: [3, 3, 3, 3], within: 0.8, between: 0, variances: [2, 1, 0.5, 0.25] });
  const net = O.makeNet({ N: c.N, B: c.B, rng: O.makeRng(c.seed), s: 3, klo: c.klo, khi: c.khi, sigmaT: c.sigmaT });
  const G = O.sampleInputs(Sigma, c.B, c.S, O.makeRng(c.seed + 101));
  const Ge = O.sampleInputs(Sigma, c.B, c.Seval, O.makeRng(c.seed + 202));
  const sim = O.createSim({ net, G, S: c.S, b: c.b, alpha: c.alpha, dt: Math.min(0.5, 1 / c.alpha), solver: { tol: c.tol } });
  // slow rule to its fixed point, in chunks so the page keeps drawing
  const t0 = performance.now();
  let solveMs = 0;
  while (sim.steps < c.maxSteps && !(sim.lastChange < c.stop)) {
    const a = performance.now();
    sim.stepFor(30, Infinity, c.stop);
    solveMs += performance.now() - a;
    status.textContent = `Running the slow rule: ${sim.steps} steps, residual ${ui.fmt(sim.residual, 2)}`;
    await nextFrame();
  }
  const wall = performance.now() - t0;
  // fixed-point bookkeeping
  const N = c.N, W = net.W, D2 = sim.D2;
  let dev = 0, slack = Infinity, live = 0, strongest = null, nearestDead = null;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    const w = W[i * N + j], d2 = D2[i * N + j];
    if (w > 1e-9) { live++; dev = Math.max(dev, Math.abs(c.alpha * w - (c.b - d2 / 2))); if (!strongest || w > strongest.w) strongest = { i, j, w }; }
    else { slack = Math.min(slack, d2 - 2 * c.b); if (!nearestDead || d2 < nearestDead.d2) nearestDead = { i, j, d2 }; }
  }
  const comps = O.components(W, N);
  status.textContent = "Comparing the two solvers…"; await nextFrame();
  // fast solver against the per-sample reference port, cold start for both
  const fast = O.equilibrate(net, G, c.S, { tol: c.tol });
  const ex = O.equilibrateExact(net, G, c.S, { tol: c.tol });
  let dU = 0; for (let q = 0; q < fast.U.length; q++) dU = Math.max(dU, Math.abs(fast.U[q] - ex.U[q]));
  const indep = O.nodeResidual(net, fast.U, G, c.S);
  status.textContent = "Envelope check…"; await nextFrame();
  // envelope identity: central difference of the mean equilibrium energy in one conductance
  const Estar = () => { const e = O.equilibrate(net, G, c.S, { tol: c.tol }); return O.energy(net, e.U, G, c.S); };
  const fd = ({ i, j }) => {
    const w0 = W[i * N + j];
    W[i * N + j] = W[j * N + i] = w0 + c.fdStep; const ep = Estar();
    W[i * N + j] = W[j * N + i] = w0 - c.fdStep; const em = Estar();
    W[i * N + j] = W[j * N + i] = w0;
    return Math.abs((ep - em) / (2 * c.fdStep) - D2[i * N + j] / 2);
  };
  const envLive = strongest ? fd(strongest) : NaN, envDead = nearestDead ? fd(nearestDead) : NaN;
  // sampling: the same geometry on an independent evaluation bank
  const Ue = O.equilibrate(net, Ge, c.Seval, { tol: c.tol }).U, D2e = O.dist2(Ue, c.Seval, N);
  let samp = 0, mx = 0;
  for (let q = 0; q < N * N; q++) { samp = Math.max(samp, Math.abs(D2[q] - D2e[q])); mx = Math.max(mx, D2e[q]); }

  const rows = [
    ["Slow rule", `steps to max|ΔW|/dt < ${ui.fmt(c.stop, 1)}`, ui.fmtInt(sim.steps), `in ${ui.fmtInt(wall)} ms wall time, ${ui.fmtInt(sim.steps / (solveMs / 1000))} steps/s while stepping`],
    ["", "plasticity residual r = max|dw/dt| (projected)", ui.fmt(sim.residual, 2), `a fixed point is claimed below ${ui.fmt(c.claim, 1)}`],
    ["", "live edges: max |αw − (b − D²/2)|", ui.fmt(dev, 2), `${live} live edges, ${comps.K} components`],
    ["", "dead pairs: min (D² − 2b)", ui.fmt(slack, 2), "should be ≥ about −r"],
    ["Equilibrium solver", "fast path: max |∂E/∂u| over neurons and stimuli", ui.fmt(fast.residual, 2), `tolerance ${ui.fmt(c.tol, 1)}; independent recomputation ${ui.fmt(indep, 2)}`],
    ["", "reference port (per-sample Newton): max |∂E/∂u|", ui.fmt(ex.residual, 2), ""],
    ["", "max |U_fast − U_reference|", ui.fmt(dU, 2), "the two paths solve the same equations"],
    ["Envelope identity", "|∂E*/∂w − D²/2|, strongest live edge", ui.fmt(envLive, 2), `central difference, step ${ui.fmt(c.fdStep, 1)}`],
    ["", "the same, nearest dead pair", ui.fmt(envDead, 2), ""],
    ["Sampling", "max |D²(training) − D²(evaluation)| / max D²(evaluation)", ui.fmt(samp / mx, 2), "not a solver error: the two banks differ"],
  ];
  const t = ui.h("table", { class: "data" });
  const tr = ui.h("tr"); ["check", "quantity", "value", "note"].forEach(hd => tr.append(ui.h("th", { text: hd }))); t.append(ui.h("thead", {}, tr));
  const tb = ui.h("tbody");
  rows.forEach(rw => { const r = ui.h("tr"); rw.forEach((x, k) => r.append(ui.h("td", { class: k === 2 ? "num" : null, text: x }))); tb.append(r); });
  t.append(tb);
  out.replaceChildren(ui.h("div", { class: "table-scroll", style: { marginTop: "10px" } }, t));
  status.textContent = `Done (${ui.fmtInt(performance.now() - t0)} ms).`;
}
