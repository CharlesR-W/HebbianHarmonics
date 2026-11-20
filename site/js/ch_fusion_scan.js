// Chapter "fusion": the b-scan of the live network (cold starts and down-continuation at alpha = 1 and 0.1),
// computed in the page in slices once the figure comes on screen, and summarised in a table (inside Details).
import * as ui from "./ui.js";
import { lineChart } from "./charts.js";
import { P } from "./palette.js";
import * as M from "./ch_fusion_model.js";

export function initScan(slot, textSlot, getParams) {
  const head = ui.h("div", { style: { display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap", margin: "2px 0 10px" } });
  const status = ui.h("span", { class: "u-help", style: { marginTop: "0" } });
  head.append(status); slot.append(head);
  const row = ui.h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 300px), 1fr))", gap: "14px 24px" } });
  slot.append(row);
  const panel = title => { const b = ui.h("div"); row.append(ui.h("div", { style: { minWidth: "0" } }, ui.h("div", { class: "chart-title", html: title }), b)); return b; };
  const boxes = [panel("(a) Number of components $K$"), panel("(b) Span of the widest component, in edge ranges $\\sqrt{2b}$")];
  ui.renderMath(row);
  const xr = [M.SCAN_B[0] / 1.2, M.SCAN_B[M.SCAN_B.length - 1] * 1.2];
  const common = { xlog: true, xrange: xr, height: 210, xlabel: "fusion scale b", xname: "b", legend: false };
  const charts = [
    lineChart(boxes[0], { ...common, ylabel: "K", series: [] }),
    lineChart(boxes[1], { ...common, ylabel: "span/√(2b)", refs: [{ y: 1 }], series: [] }),
  ];
  const leg = ui.legend({ parent: slot, items: [
    { label: "$\\alpha = 1$, from $W = 0$", color: P.cat[0], shape: "line" }, { label: "$\\alpha = 1$, down-continuation", color: P.cat[0], shape: "dash" },
    { label: "$\\alpha = 0.1$, from $W = 0$", color: P.cat[1], shape: "line" }, { label: "$\\alpha = 0.1$, down-continuation", color: P.cat[1], shape: "dash" },
  ] });
  leg.el.style.marginTop = "8px";

  let res = null, job = null, net = null, loop = null, lastDraw = 0;
  const describe = p => `seed ${p.seed}, N = ${p.n}, s = ${p.s}, inputs ${M.sigmaById(p.sig).label}`;

  function start() {
    const p = getParams();
    net = { ...p }; res = {}; job = M.scanJob(p, res, 8);
    status.textContent = `Running on: ${describe(p)}…`;
    textSlot.innerHTML = "<p>The scan is running; the table appears when it has finished.</p>";
    if (loop) loop.stop();
    loop = ui.animate(slot, () => {
      if (!job) return;
      const t0 = performance.now();
      while (performance.now() - t0 < 22) {
        const r = job.next();
        if (r.done) { job = null; break; }
      }
      if (!job || performance.now() - lastDraw > 400) { lastDraw = performance.now(); draw(); }
      if (!job) { finish(); loop.stop(); loop = null; }
      else status.textContent = `Running on: ${describe(net)}… ${res.done} of ${res.total} fixed points`;
    });
  }
  function seriesFor(key) {
    const out = [];
    M.SCAN_ALPHAS.forEach((a, k) => {
      const r = res[a]; if (!r) return;
      for (const br of ["cold", "down"]) {
        const pts = r[br].filter(Boolean).filter(s => s.converged);
        out.push({ name: `α = ${a}, ${br === "cold" ? "from W = 0" : "down"}`, x: pts.map(s => s.b), y: pts.map(s => key(s)), color: P.cat[k], dash: br === "down", dots: true, width: br === "down" ? 1.6 : 2 });
      }
    });
    return out;
  }
  function draw() {
    if (!res) return;
    charts[0].update({ series: seriesFor(s => s.K) });
    charts[1].update({ series: seriesFor(s => s.spanRange) });
  }
  function finish() {
    const nBad = M.SCAN_ALPHAS.reduce((n, a) => n + ["cold", "down"].reduce((m, br) => m + res[a][br].filter(s => !s.converged).length, 0), 0);
    status.textContent = `Network: ${describe(net)}. ${res.total - nBad} of ${res.total} runs reached r < 10⁻⁶${nBad ? " (the others are left out of the plots)" : ""}.`;
    const notes = [];
    const rows = M.SCAN_ALPHAS.map(a => {
      // only fixed points (r < 1e-6) enter the summary; an unconverged grid point is a gap
      const c = res[a].cold, d = res[a].down, ok = s => s && s.converged;
      const cc = c.filter(ok);
      const b1 = cc.find(s => s.K === 1);
      let run = 0, best = 0, bestK = null;
      c.forEach((s, k) => { run = ok(s) && s.K >= 2 ? (k && ok(c[k - 1]) && s.K === c[k - 1].K ? run + 1 : 1) : 0; if (run > best) { best = run; bestK = s.K; } });
      let sr = 0, bsr = null; cc.forEach(s => { if (s.K < s.labels.length && s.spanRange > sr) { sr = s.spanRange; bsr = s.b; } });
      const N = c[0].labels.length, tight = cc.filter(s => s.K >= 2 && s.kMulti >= 2 && (s.K - s.kMulti) <= 0.2 * N && s.spanRange < 1);
      const pairs = c.map((s, k) => ({ b: s.b, kc: s.K, kd: d[k].K, both: ok(s) && ok(d[k]) })).filter(x => x.both);
      const diff = pairs.filter(x => x.kc !== x.kd);
      const worst = diff.reduce((m, x) => (!m || x.kc - x.kd > m.kc - m.kd ? x : m), null);
      for (const t of tight) {
        const k = c.indexOf(t), nb = [c[k - 1], c[k + 1]].filter(ok);
        notes.push(`At α = ${a} and b = ${ui.fmt(t.b, 2)} the cold start ends in ${t.kMulti} tight clusters and ${t.K - t.kMulti} lone neuron${t.K - t.kMulti === 1 ? "" : "s"} (K = ${t.K} components)` +
          (nb.length ? `; one grid step away it has ${nb.map(x => `K = ${x.K} (b = ${ui.fmt(x.b, 2)})`).join(" and ")}` : "") +
          (ok(d[k]) ? `, and the down-continuation at the same b has K = ${d[k].K}.` : "."));
      }
      let nest = 0, bn = null, nPairs = 0;
      for (let k = 1; k < c.length; k++) { if (!ok(c[k - 1]) || !ok(c[k])) continue; nPairs++; const v = M.pairViolation(c[k - 1].labels, c[k].labels); if (v > nest) { nest = v; bn = c[k].b; } }
      const na = "n/a (no converged runs)";
      return `<tr><td>${a}</td><td class="num">${cc.length ? `${cc[0].K} → ${b1 ? `1 at b = ${ui.fmt(b1.b, 2)}` : `${cc[cc.length - 1].K} at b = ${ui.fmt(cc[cc.length - 1].b, 2)}`}` : na}</td>` +
        `<td class="num">${best > 1 ? `${best} (K = ${bestK})` : "none"}</td>` +
        `<td class="num">${bsr != null ? `${ui.fmtFixed(sr, 2)} (b = ${ui.fmt(bsr, 2)})` : "–"}</td>` +
        `<td class="num">${tight.length ? tight.map(s => ui.fmt(s.b, 2)).join(", ") : "none"}</td>` +
        `<td class="num">${pairs.length ? `${diff.length} of ${pairs.length}${worst ? `; at b = ${ui.fmt(worst.b, 2)}: ${worst.kc} → ${worst.kd}` : ""}` : na}</td>` +
        `<td class="num">${nPairs ? `${ui.fmtFixed(nest, 2)}${bn != null && nest > 0 ? ` (b = ${ui.fmt(bn, 2)})` : ""}` : na}</td></tr>`;
    }).join("");
    textSlot.innerHTML = `<div class="table-scroll"><table class="data" style="min-width:680px"><thead><tr><th>α</th><th class="num">K from W = 0, smallest b → largest</th>` +
      `<th class="num">longest run of equal K ≥ 2 (grid points)</th><th class="num">widest span / √(2b)</th>` +
      `<th class="num">b with a few tight clusters</th><th class="num">b where K down ≠ K cold (of b with both converged); largest gap (cold → down)</th>` +
      `<th class="num">max non-nested fraction</th></tr></thead><tbody>${rows}</tbody></table></div>` +
      `<p class="u-help">Computed in this page for the network named above (${M.SCAN_B.length} values of b from ${M.SCAN_B[0]} to ${M.SCAN_B[M.SCAN_B.length - 1]}). "A few tight clusters": at least two components with two or more members, at most 20% of the neurons left alone, and no component wider than one edge range √(2b). Non-nested fraction: of the pairs in one component at one b, the fraction split at the next larger b (0 = refinement is nested). Only runs that reached r &lt; 10⁻⁶ enter this table; a grid point that did not is treated as a gap.</p>` +
      (notes.length ? `<p>${notes.join(" ")}</p>` : `<p>No value of b on the grid gives a few tight clusters, at either α.</p>`);
  }

  // start the first scan when the figure first comes on screen
  const io = new IntersectionObserver(es => { if (es[es.length - 1].isIntersecting && !res) { io.disconnect(); start(); } }, { rootMargin: "200px" });
  io.observe(slot);
  return {};
}
