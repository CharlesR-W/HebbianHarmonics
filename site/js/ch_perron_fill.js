// Chapter "perron", figure "which parts get filled" ([data-slot="fill"]): fresh starts on the 16-input hierarchy
// (site/data/perron2/cold_starts.json, hebb/perron2/cold.py). A map over threshold θ (columns) and lateral covariance
// gain γ (rows), each cell coloured by the fraction of the stable parts that hold a group (mean over seeds) and labelled
// "filled / stable"; the levels of the tree as dashed lines, and the binding line θ = γκ₀.
import * as ui from "./ui.js";
import { hidpi } from "./charts.js";
import { P, cmap, rgbToCss } from "./palette.js";
import { FONT } from "./ch_perron_tri.js";

const THETA_MAX = 0.78;          // above this the floor dominates (see the figure's details); not drawn
const LEVELS = [0.05, 0.25, 0.5];

export async function mountFill(slot, ctx) {
  const cold = await fetch("data/perron2/cold_starts.json").then(r => { if (!r.ok) throw new Error(`cold_starts.json: HTTP ${r.status}`); return r.json(); });
  const rows = cold.table.filter(r => r.sigma === "hier16" && r.theta <= THETA_MAX);
  const gammas = [...new Set(rows.map(r => r.gamma))].sort((a, b) => b - a);
  const thetas = [...new Set(rows.map(r => r.theta))].sort((a, b) => a - b);
  const cell = (g, t) => rows.find(r => r.gamma === g && r.theta === t);
  const kappa0 = rows[0].kappa0, seq = cmap("seq");
  slot.replaceChildren();
  const cv = ui.h("canvas", { style: { display: "block", width: "100%" } });
  const box = ui.h("div", {}, cv);
  const foot = ui.h("div", { class: "pc-foot" });
  ui.colorbar({ parent: foot, map: "seq", range: [0, 1], ticks: [0, 0.5, 1], width: 300, label: "fraction of the stable parts holding a group", fmt: x => ui.fmt(x) });
  ui.legend({ parent: foot, items: [{ label: "θ = γκ₀: below it the neurons bind into one group", color: P.accent, shape: "line" }, { label: "levels of the tree", color: P.mute2, shape: "dash" }] });
  const read = ui.h("div", { class: "pc-read" });
  slot.append(box, foot, read);
  const nSeeds = Math.max(...rows.map(r => r.n_populated.length));
  read.textContent = `Each cell: ${nSeeds} fresh starts of 300 neurons; the label is the number of stable parts holding a group (mean over the starts) out of the number of stable parts. κ₀ = ${ui.fmt(kappa0, 3)} is the mean covariance between inputs.`;

  let W = 0, Lay = null;
  function render() {
    W = Math.floor(box.clientWidth); if (W <= 0) return;
    const narrow = W < 560, m = { l: narrow ? 50 : 64, r: 8, t: 30, b: 40 }, rowH = narrow ? 34 : 42, H = m.t + rowH * gammas.length + m.b;
    const X = t => m.l + t / THETA_MAX * (W - m.l - m.r);
    Lay = { m, X, rowH, H };
    const g = hidpi(cv, W, H); g.clearRect(0, 0, W, H);
    const edges = thetas.map((t, i) => [i ? (thetas[i - 1] + t) / 2 : 0, i + 1 < thetas.length ? (t + thetas[i + 1]) / 2 : THETA_MAX]);
    gammas.forEach((gm, j) => {
      const y = m.t + j * rowH;
      thetas.forEach((t, i) => {
        const r = cell(gm, t); if (!r) return;
        const mean = r.n_populated.reduce((a, b) => a + b, 0) / r.n_populated.length, f = r.n_stable ? mean / r.n_stable : 0;
        const xa = Math.round(X(edges[i][0])), xb = Math.round(X(edges[i][1]));
        g.fillStyle = rgbToCss(seq(0.08 + 0.92 * f)); g.fillRect(xa, y + 1, xb - xa - 1, rowH - 2);
        if (xb - xa >= (narrow ? 13 : 24)) {
          const allSame = r.n_populated.every(v => v === r.n_populated[0]);
          g.fillStyle = f > 0.55 ? P.surface : P.fg; g.font = `${narrow ? 8.5 : 10.5}px ${FONT}`; g.textAlign = "center"; g.textBaseline = "middle";
          const lab = `${allSame ? r.n_populated[0] : ui.fmt(mean, 2)}`;
          if (narrow) { g.fillText(lab, (xa + xb) / 2, y + rowH / 2 - 6); g.fillText(`/${r.n_stable}`, (xa + xb) / 2, y + rowH / 2 + 6); }
          else g.fillText(`${lab}/${r.n_stable}`, (xa + xb) / 2, y + rowH / 2);
        }
      });
      g.fillStyle = P.ink2; g.font = `${narrow ? 10.5 : 11.5}px ${FONT}`; g.textAlign = "right"; g.textBaseline = "middle";
      g.fillText(`γ = ${gm}`, m.l - 6, y + rowH / 2);
    });
    // binding line θ = γκ₀, one segment per row, joined
    g.strokeStyle = P.accent; g.lineWidth = 2.2; g.beginPath();
    gammas.forEach((gm, j) => { const x = X(gm * kappa0), y0 = m.t + j * rowH, y1 = y0 + rowH; g[j ? "lineTo" : "moveTo"](x, y0); g.lineTo(x, y1); });
    g.stroke();
    // tree levels
    g.strokeStyle = P.mute2; g.lineWidth = 1; g.setLineDash([3, 3]);
    for (const l of LEVELS) { g.beginPath(); g.moveTo(X(l) + 0.5, m.t - 6); g.lineTo(X(l) + 0.5, m.t + rowH * gammas.length); g.stroke(); }
    g.setLineDash([]);
    // number of stable parts between levels, on top
    g.fillStyle = P.ink2; g.font = `${narrow ? 10 : 11}px ${FONT}`; g.textAlign = "center"; g.textBaseline = "alphabetic";
    const bounds = [0, ...LEVELS, THETA_MAX];
    for (let k = 0; k + 1 < bounds.length; k++) {
      const mid = (bounds[k] + bounds[k + 1]) / 2, r = rows.find(q => q.theta > bounds[k] && q.theta < bounds[k + 1]);
      if (r && X(bounds[k + 1]) - X(bounds[k]) > 40) g.fillText(`${r.n_stable} part${r.n_stable === 1 ? "" : "s"}`, X(mid), m.t - 10);
    }
    // θ axis
    const ya = m.t + rowH * gammas.length;
    g.strokeStyle = P.line; g.fillStyle = P.mute; g.font = `10.5px ${FONT}`; g.textBaseline = "top";
    for (const t of [0, 0.2, 0.4, 0.6]) { g.beginPath(); g.moveTo(X(t), ya); g.lineTo(X(t), ya + 4); g.stroke(); g.fillText(ui.fmt(t), X(t), ya + 6); }
    g.fillStyle = P.ink2; g.font = `560 11.5px ${FONT}`; g.textBaseline = "alphabetic"; g.fillText("pruning threshold θ", (m.l + W - m.r) / 2, H - 4);
  }
  new ResizeObserver(() => { const w = Math.floor(box.clientWidth); if (w > 0 && w !== W) requestAnimationFrame(render); }).observe(box);
  render();
  cv.addEventListener("pointermove", e => {
    if (!Lay) return;
    const rc = cv.getBoundingClientRect(), x = e.clientX - rc.left, y = e.clientY - rc.top, j = Math.floor((y - Lay.m.t) / Lay.rowH);
    if (j < 0 || j >= gammas.length || x < Lay.m.l) { ui.hideTip(); return; }
    const t = (x - Lay.m.l) / (W - Lay.m.l - Lay.m.r) * THETA_MAX, th = thetas.reduce((a, b) => (Math.abs(b - t) < Math.abs(a - t) ? b : a));
    const r = cell(gammas[j], th); if (!r) { ui.hideTip(); return; }
    ui.showTip(e.clientX, e.clientY, ui.tipRows(`θ = ${ui.fmtFixed(th, 3)}, γ = ${gammas[j]}`, [
      { name: "stable parts", value: String(r.n_stable) },
      { name: "holding a group, per start", value: r.n_populated.join(", ") },
      { name: "group sizes, first start", value: (r.sizes?.[0] || []).join(", ") }]));
  });
  cv.addEventListener("pointerleave", () => ui.hideTip());
}
