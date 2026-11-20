// Chapter "perron", figure "raising θ splits groups only if they can split" ([data-slot="ramp"]): the precomputed
// quasi-static θ ramps on 16 hierarchically correlated inputs (site/data/perron2/ramps.json, hebb/perron2/ramps.py)
// over the exact stable parts (part_diagrams.json). Two panels, lateral gain γ = 1 and γ = 0.1; one control, the
// direction of the ramp. Rows are inputs in tree order; light bars are the stable parts at each θ; dark bars are the
// parts that hold a group of neurons at the end of each ramp step.
import * as ui from "./ui.js";
import { hidpi, niceTicks } from "./charts.js";
import { P } from "./palette.js";
import { supportName, inputLabel } from "./ch_perron_teach.js";

const BASE = "data/perron2/";
const getJSON = name => fetch(BASE + name).then(r => { if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`); return r.json(); });
const GAMMAS = [1, 0.1];
const STEP_CAP = 40000;      // ramps.py: integration cap per θ step
const FONT = "Inter, system-ui, sans-serif";
// contiguous runs [[b0, b1], ...] of a support (inputs are in tree order)
const runsOf = S => { const r = [...S].sort((a, b) => a - b), out = []; for (let i = 0; i < r.length;) { let j = i; while (j + 1 < r.length && r[j + 1] === r[j] + 1) j++; out.push([r[i], r[j]]); i = j + 1; } return out; };

export async function mountRamp(slot, ctx, { onData } = {}) {
  const [ramps, diag] = await Promise.all([getJSON("ramps.json"), getJSON("part_diagrams.json")]);
  const d = diag.diagrams.find(x => x.name === "hier16");
  const B = d.B, levels = ramps.meta.levels;
  const runOf = g => ramps.runs.find(r => r.gamma === g);
  let dir = ctx.param("rd", "up") === "down2" ? "down2" : "up";

  slot.replaceChildren();
  const seg = ui.segmented({ label: "ramp", options: [{ value: "up", label: "raising θ (from 0.03)" }, { value: "down2", label: "lowering θ (from 0.82)" }], value: dir,
    onChange: v => { dir = v; ctx.setParams({ rd: v === "up" ? null : v }); panels.forEach(p => p.render()); readout(); } });
  const ctl = ui.h("div", { class: "pt-ctl" }, seg.el);
  const box = ui.h("div");
  const foot = ui.h("div", { class: "pc-foot" });
  ui.legend({ parent: foot, items: [
    { label: "stable part, empty", color: P.other, shape: "square" },
    { label: "stable part holding a group of neurons", color: P.fg, shape: "square" },
    { label: "group not on a stable part", color: P.accent, shape: "square" },
    { label: "▲ step stopped at the 40,000-step cap before settling", color: "transparent", shape: "dot" },
    { label: "levels of the tree", color: P.mute2, shape: "dash" }] });
  const read = ui.h("div", { class: "pc-read" });
  slot.append(ctl, box, foot, read);

  const panels = GAMMAS.map(g => panel(box, g));
  function panel(parent, g) {
    const title = ui.h("div", { class: "pc-panel-title", style: { marginTop: "6px" } });
    title.textContent = `lateral covariance gain γ = ${g}`;
    const cv = ui.h("canvas", { style: { display: "block", width: "100%", touchAction: "pan-y" } });
    const wrap = ui.h("div", {}, title, cv);
    parent.append(wrap);
    let W = 0, Lay = null;
    const run = () => runOf(g)[dir];
    function render() {
      W = Math.floor(wrap.clientWidth); if (W <= 0) return;
      const rowH = W < 560 ? 9 : 12, top = 24, x0 = 34, x1 = W - 8, plotH = rowH * B, Hh = top + plotH + 34;
      Lay = { rowH, top, x0, x1, plotH };
      const c = hidpi(cv, W, Hh); c.clearRect(0, 0, W, Hh);
      const X = t => x0 + t * (x1 - x0), Y = r => top + r * rowH;
      c.fillStyle = P.wash; c.fillRect(x0, top, x1 - x0, plotH);
      // stable parts: one column per θ of the part diagram
      const th = d.theta, n = th.length, inset = 1.5;
      for (let k = 0; k < n; k++) {
        const xa = Math.round(X(k ? (th[k - 1] + th[k]) / 2 : 0)), xb = Math.round(X(k < n - 1 ? (th[k] + th[k + 1]) / 2 : 1));
        c.fillStyle = P.other;
        for (const [S] of d.parts[k]) for (const [r0, r1] of runsOf(S)) c.fillRect(xa, Y(r0) + inset, xb - xa, (r1 - r0 + 1) * rowH - 2 * inset);
      }
      // groups at the end of each ramp step
      const r = run(), tt = r.theta, m = tt.length, sorted = [...tt].sort((a, b) => a - b);
      const half = (sorted[sorted.length - 1] - sorted[0]) / (m - 1) / 2;
      for (let i = 0; i < m; i++) {
        const xa = X(tt[i] - half), xb = X(tt[i] + half);
        r.supports[i].forEach((S, j) => {
          c.fillStyle = r.matches[i][j] ? P.fg : P.accent;
          for (const [r0, r1] of runsOf(S)) c.fillRect(xa, Y(r0) + inset + 1, xb - xa + 0.5, (r1 - r0 + 1) * rowH - 2 * inset - 2);
        });
      }
      // steps that used the whole 40,000-step budget without reaching the 1e-7 stopping target
      c.fillStyle = P.accent; c.font = `9px ${FONT}`; c.textAlign = "center"; c.textBaseline = "alphabetic";
      for (let i = 0; i < m; i++) if (r.steps[i] >= STEP_CAP && r.residual[i] >= 1e-7) c.fillText("▲", X(tt[i]), top - 1);
      // direction of the ramp
      const ax = dir === "up" ? [X(0.8), X(0.88)] : [X(0.88), X(0.8)], ay = top + plotH + 26;
      c.strokeStyle = P.ink2; c.fillStyle = P.ink2; c.lineWidth = 1.4;
      c.beginPath(); c.moveTo(ax[0], ay); c.lineTo(ax[1], ay); c.stroke();
      const sgn = Math.sign(ax[1] - ax[0]);
      c.beginPath(); c.moveTo(ax[1], ay); c.lineTo(ax[1] - 6 * sgn, ay - 4); c.lineTo(ax[1] - 6 * sgn, ay + 4); c.closePath(); c.fill();
      c.font = `11px ${FONT}`; c.textAlign = dir === "up" ? "left" : "right"; c.textBaseline = "middle";
      c.fillText("ramp", dir === "up" ? X(0.89) : X(0.79), ay);
      // levels of the tree
      c.strokeStyle = P.mute2; c.lineWidth = 1; c.setLineDash([3, 3]);
      for (const l of levels) { c.beginPath(); c.moveTo(X(l) + 0.5, top - 4); c.lineTo(X(l) + 0.5, top + plotH + 2); c.stroke(); }
      c.setLineDash([]);
      c.font = `10px ${FONT}`; c.fillStyle = P.mute; c.textAlign = "center"; c.textBaseline = "alphabetic";
      for (const l of levels) c.fillText(String(l), X(l), top - 12);
      // row labels and θ axis
      c.font = `${rowH < 10 ? 9 : 10}px ${FONT}`; c.fillStyle = P.ink2; c.textAlign = "right"; c.textBaseline = "middle";
      for (let b = 0; b < B; b += 2) c.fillText(inputLabel(b), x0 - 4, Y(b) + rowH / 2);
      const ya = top + plotH;
      c.strokeStyle = P.line; c.beginPath(); c.moveTo(x0, ya + 0.5); c.lineTo(x1, ya + 0.5); c.stroke();
      c.fillStyle = P.mute; c.font = `10.5px ${FONT}`; c.textAlign = "center"; c.textBaseline = "top";
      for (const t of niceTicks(0, 1, 6)) { c.beginPath(); c.moveTo(X(t), ya); c.lineTo(X(t), ya + 4); c.stroke(); c.fillText(ui.fmt(t), X(t), ya + 6); }
      c.fillStyle = P.ink2; c.font = `560 11px ${FONT}`; c.fillText("threshold θ", (x0 + x1) / 2, ya + 20);
    }
    cv.addEventListener("pointermove", e => {
      if (!Lay) return;
      const rc = cv.getBoundingClientRect(), x = e.clientX - rc.left, y = e.clientY - rc.top;
      if (x < Lay.x0 || x > Lay.x1 || y < Lay.top || y > Lay.top + Lay.plotH) { ui.hideTip(); return; }
      const t = (x - Lay.x0) / (Lay.x1 - Lay.x0), r = run();
      const lo = Math.min(...r.theta), hi = Math.max(...r.theta), hw = (hi - lo) / (r.theta.length - 1) / 2;
      if (t < lo - hw || t > hi + hw) { ui.showTip(e.clientX, e.clientY, ui.tipRows(`θ = ${ui.fmtFixed(t, 3)}`, [], "outside this ramp's range of θ")); return; }
      const i = r.theta.reduce((bi, v, k) => (Math.abs(v - t) < Math.abs(r.theta[bi] - t) ? k : bi), 0);
      const rows = r.supports[i].map((S, j) => ({ name: supportName(S), value: `${r.sizes[i][j]} neurons${r.matches[i][j] ? "" : ", not a stable part"}`, color: r.matches[i][j] ? P.fg : P.accent, shape: "square" }));
      ui.showTip(e.clientX, e.clientY, ui.tipRows(`θ = ${ui.fmtFixed(r.theta[i], 3)} (γ = ${g})`, rows, `${r.n_populated[i]} of ${r.n_stable[i]} stable parts hold a group${r.n_dead[i] ? `; ${r.n_dead[i]} neurons dead` : ""}`));
    });
    cv.addEventListener("pointerleave", () => ui.hideTip());
    new ResizeObserver(() => { const w = Math.floor(wrap.clientWidth); if (w > 0 && w !== W) requestAnimationFrame(render); }).observe(wrap);
    render();
    return { render };
  }

  // Per level of the tree (below the top one, where the floor dominates): the most common "filled of stable" count
  // over the ramp steps between that level and the next.
  function counts(g, which) {
    const r = runOf(g)[which], lv = [...levels].sort((a, b) => a - b), out = [];
    for (let k = 0; k + 1 < lv.length; k++) {
      const seen = new Map();
      r.theta.forEach((t, i) => { if (t > lv[k] && t < lv[k + 1]) { const key = `${r.n_populated[i]} of ${r.n_stable[i]}`; seen.set(key, (seen.get(key) || 0) + 1); } });
      if (seen.size) out.push([...seen].sort((a, b) => b[1] - a[1])[0][0]);
    }
    return out;
  }
  function readout() {
    let extra = "";
    if (dir === "down2") {
      const first = g => { const r = runOf(g).down2; const i = r.theta.findIndex((t, k) => k > 0 && r.n_populated[k] > 0 && r.n_stable[k] < r.n_stable[0]); return i >= 0 ? r.theta[i] : NaN; };
      extra = ` Lowering θ, the 16 single-input groups have merged into pairs by θ = ${ui.fmtFixed(first(1), 3)} (γ = 1) and ${ui.fmtFixed(first(0.1), 3)} (γ = 0.1), a little below the 0.8 level.`;
    }
    read.innerHTML = `Stable parts holding a group, between consecutive levels of the tree (most common count over the steps there): ` +
      `γ = 1: ${counts(1, dir).join(", ")}; γ = 0.1: ${counts(0.1, dir).join(", ")}.` + extra;
  }
  readout();
  onData?.({ counts, splitTests: ramps.split_tests });
  ctx.onParams?.(p => { const v = p.get("rd") === "down2" ? "down2" : "up"; if (v !== dir) { dir = v; seg.set(v); panels.forEach(q => q.render()); readout(); } });
}
