// Chapter "perron": two canvas figures in the triangle of mixtures (input space).
//   [data-slot="land"]  the landscape R_θ(h) = hᵀAh / hᵀh, A = Σ − θ11ᵀ, for the "all three correlated" covariance at
//                       the grid's four thresholds; rings at the stable parts; paths of the neurons of the same runs
//                       as the grid (shared through the runner's cache).
//   [data-slot="norm"]  subtractive θ against multiplicative normalisation: single neurons (input synapses only) from
//                       the same 40 random starts; where they end.
import * as ui from "./ui.js";
import { hidpi } from "./charts.js";
import { P, cmap, rgbToCss, mixBary } from "./palette.js";
import { eigh, makeRng } from "./engine.js";
import * as M from "./ch_perron_model.js";
import * as T from "./ch_perron_teach_model.js";
import { triGeom, BCOL, GL, FONT, fx, hstr, GRID_THETAS, gridParams, groupRadius, nearestPart } from "./ch_perron_tri.js";

const Rq = (S, th, h) => {
  let n = 0, d = 0;
  for (let a = 0; a < 3; a++) { d += h[a] * h[a]; for (let b = 0; b < 3; b++) n += h[a] * (S[a * 3 + b] - th) * h[b]; }
  return n / d;
};

// A canvas that redraws at its container's width (height = width * aspect).
function canvasPanel(parent, aspect, draw, onMove) {
  const cv = ui.h("canvas");
  parent.append(cv);
  let W = 0;
  const render = () => { W = Math.floor(parent.clientWidth); if (W <= 0) return; const H = Math.round(W * aspect); const g = hidpi(cv, W, H); g.clearRect(0, 0, W, H); draw(g, W, H); };
  new ResizeObserver(() => { const w = Math.floor(parent.clientWidth); if (w > 0 && w !== W) requestAnimationFrame(render); }).observe(parent);
  if (onMove) {
    cv.addEventListener("pointermove", e => { const r = cv.getBoundingClientRect(); onMove(e, e.clientX - r.left, e.clientY - r.top, W, Math.round(W * aspect)); });
    cv.addEventListener("pointerleave", () => ui.hideTip());
  }
  render();
  return { render, cv };
}

function frame(g, G, { sq = 9, font = 12, gap = 13 } = {}) {
  g.strokeStyle = P.mute2; g.lineWidth = 1.1;
  g.beginPath(); g.moveTo(...G.V[0]); g.lineTo(...G.V[1]); g.lineTo(...G.V[2]); g.closePath(); g.stroke();
  const cen = [(G.V[0][0] + G.V[1][0] + G.V[2][0]) / 3, (G.V[0][1] + G.V[1][1] + G.V[2][1]) / 3];
  g.font = `600 ${font}px ${FONT}`;
  G.V.forEach(([x, y], b) => {
    const dx = x - cen[0], dy = y - cen[1], n = Math.hypot(dx, dy), off = sq / 2 + gap, qx = x + off * dx / n, qy = y + off * dy / n;
    g.fillStyle = BCOL[b]; g.fillRect(qx - sq / 2, qy - sq / 2, sq, sq);
    g.strokeStyle = P.fg; g.lineWidth = 1; g.strokeRect(qx - sq / 2, qy - sq / 2, sq, sq);
    g.fillStyle = P.fg;
    if (b === 0) { g.textAlign = "left"; g.textBaseline = "middle"; g.fillText(GL[b], qx + sq / 2 + 4, qy); }
    else { g.textAlign = "center"; g.textBaseline = "top"; g.fillText(GL[b], qx, qy + sq / 2 + 2); }
  });
}
function ring(g, G, h, r = 10) { const [x, y] = G.at(h); g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.lineWidth = 1.6; g.strokeStyle = P.fg; g.stroke(); }
function dot(g, x, y, r, fill, stroke = P.surface) { g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fillStyle = fill; g.fill(); g.lineWidth = 1.2; g.strokeStyle = stroke; g.stroke(); }
function star(g, cx, cy, r) {
  g.beginPath();
  for (let k = 0; k < 10; k++) { const a = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r * 0.45 : r; g[k ? "lineTo" : "moveTo"](cx + rr * Math.cos(a), cy + rr * Math.sin(a)); }
  g.closePath(); g.fillStyle = P.surface; g.fill(); g.lineWidth = 1.6; g.strokeStyle = P.fg; g.stroke();
}

// ======================================================================= landscape
// the triangle of a landscape panel, 18 px lower to leave room for its title
function landGeom(W, H) {
  const G = triGeom(W, H - 34, 30);
  return { V: G.V.map(([x, y]) => [x, y + 18]), at: h => { const [x, y] = G.at(h); return [x, y + 18]; }, bary: (x, y) => G.bary(x, y - 18) };
}
export const LAND_RANGE = 1.6;     // fixed colour range [−1.6, 1.6] for every panel
export function mountLandscape(slot, ctx) {
  const S = M.sigmaOf(M.SIGMAS.tri), div = cmap("div");
  const row = ui.h("div", { class: "pc-row", style: { "--cols": GRID_THETAS.length } });
  const foot = ui.h("div", { class: "pc-foot" });
  ui.colorbar({ parent: foot, map: "div", range: [-LAND_RANGE, LAND_RANGE], ticks: [-1.5, -1, -0.5, 0, 0.5, 1, 1.5], width: 260,
    label: "$R_\\theta(h)$; contour lines every 0.1" });
  const lg = ui.legend({ parent: foot, items: [
    { label: "stable part", color: P.fg, shape: "ring" },
    { label: "path of a neuron, start to end", color: P.fg, shape: "line" },
    { label: "group at the end of the run", color: mixBary([0.45, 0.45, 0.1], BCOL), shape: "dot" }] });
  const read = ui.h("div", { class: "pc-read" });
  slot.replaceChildren(row, foot, read);
  const panels = GRID_THETAS.map(th => {
    const box = ui.h("div");
    row.append(box);
    const parts = T.stableParts(S, 3, th);
    const st = { th, parts, res: null, img: null, imgW: 0 };
    const pan = canvasPanel(box, 1.0, (g, W, H) => drawLand(g, W, H, st), (e, x, y, W, H) => {
      const h = landGeom(W, H).bary(x, y);
      if (Math.min(...h) < -0.01) { ui.hideTip(); return; }
      const hh = h.map(v => Math.max(0, v)), s = hh[0] + hh[1] + hh[2];
      const rows = [{ name: "mixture h", value: hstr(hh.map(v => v / s)) }, { name: "R_θ(h)", value: fx(Rq(S, th, hh), 3) }];
      ui.showTip(e.clientX, e.clientY, ui.tipRows(`θ = ${fx(th)}`, rows));
    });
    st.pan = pan;
    M.run(gridParams("tri", th), { record: 10 }).then(res => { st.res = res; pan.render(); summary(); });
    return st;
  });
  function drawLand(g, W, H, st) {
    const G = landGeom(W, H), G2 = G;
    // heat map with contour lines, computed once per size
    if (!st.img || st.imgW !== W) {
      const dpr = window.devicePixelRatio || 1, w = Math.round(W * dpr), hgt = Math.round(H * dpr);
      const off = document.createElement("canvas"); off.width = w; off.height = hgt;
      const oc = off.getContext("2d"), im = oc.createImageData(w, hgt), Gp = landGeom(W, H);
      const val = new Float32Array(w * hgt).fill(NaN);
      for (let py = 0; py < hgt; py++) for (let px = 0; px < w; px++) {
        const h = Gp.bary(px / dpr, py / dpr);
        if (h[0] < -0.002 || h[1] < -0.002 || h[2] < -0.002) continue;
        val[py * w + px] = Rq(S, st.th, h.map(v => Math.max(v, 0)));
      }
      for (let py = 0; py < hgt; py++) for (let px = 0; px < w; px++) {
        const v = val[py * w + px]; if (!Number.isFinite(v)) continue;
        let [r, gg, b] = div(0.5 + 0.5 * Math.max(-1, Math.min(1, v / LAND_RANGE)));
        const band = Math.floor(v / 0.1), vr = val[py * w + px + 1], vd = val[(py + 1) * w + px];
        if ((Number.isFinite(vr) && Math.floor(vr / 0.1) !== band) || (Number.isFinite(vd) && Math.floor(vd / 0.1) !== band)) {
          const k = band === -1 ? 0.45 : 0.75; r *= k; gg *= k; b *= k;           // the zero contour darker
        }
        const q = 4 * (py * w + px); im.data[q] = r; im.data[q + 1] = gg; im.data[q + 2] = b; im.data[q + 3] = 255;
      }
      oc.putImageData(im, 0, 0);
      st.img = off; st.imgW = W;
    }
    g.drawImage(st.img, 0, 0, W, H);
    frame(g, G);
    g.fillStyle = P.fg; g.font = `600 12.5px ${FONT}`; g.textAlign = "center"; g.textBaseline = "top";
    g.fillText(`θ = ${fx(st.th)}`, W / 2, 0);
    // paths: every third neuron
    if (st.res) {
      const path = st.res.path, N = st.res.final.H.length / 3;
      g.lineWidth = 1; g.strokeStyle = "rgba(22,22,26,0.5)";
      for (let i = 0; i < N; i += 3) {
        g.beginPath();
        path.forEach((Hf, f) => { const [x, y] = G2.at([Hf[3 * i], Hf[3 * i + 1], Hf[3 * i + 2]]); g[f ? "lineTo" : "moveTo"](x, y); });
        g.stroke();
        const [x0, y0] = G2.at([path[0][3 * i], path[0][3 * i + 1], path[0][3 * i + 2]]);
        g.beginPath(); g.arc(x0, y0, 1.8, 0, 2 * Math.PI); g.fillStyle = P.fg; g.fill();
      }
      for (const gr of st.res.final.groups) { const [x, y] = G2.at(gr.h); dot(g, x, y, groupRadius(gr.n, N), mixBary(gr.h, BCOL)); }
    }
    for (const p of st.parts) ring(g, G2, p.h, 10);
  }
  // Live check, shown under the figure: the local maxima of R_θ on a fine grid of the triangle are the stable parts.
  function summary() {
    if (panels.some(p => !p.res)) return;
    let dmax = 0;
    const out = panels.map(p => {
      const mx = gridMaxima(S, p.th, 160);
      const ok = mx.length === p.parts.length && p.parts.every(q => mx.some(m => nearestPart([q], m).d < 0.05));
      for (const q of p.parts) dmax = Math.max(dmax, Math.min(...mx.map(m => nearestPart([q], m).d)));
      return { th: p.th, n: mx.length, ok };
    });
    read.innerHTML = `Check, computed here: on a grid of ${ui.fmtInt(161 * 162 / 2)} mixtures (spacing 1/160), the peaks of $R_\\theta$ ${out.every(o => o.ok) ? "match the predicted parts one to one" : "<b>do not</b> all match the predicted parts"} (${out.map(o => `${o.n} at θ = ${fx(o.th)}`).join(", ")}); the largest distance from a part to its grid peak is ${ui.fmt(dmax, 2)} (sum of absolute differences).`;
    ui.renderMath(read);
  }
}

// Local maxima of R_θ on the grid {i/m} of the triangle (including its edges and corners).
export function gridMaxima(S, th, m) {
  const val = (i, j) => Rq(S, th, [i / m, j / m, (m - i - j) / m]);
  const nb = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]], out = [];
  for (let i = 0; i <= m; i++) for (let j = 0; j <= m - i; j++) {
    const v = val(i, j);
    if (nb.every(([a, b]) => { const p = i + a, q = j + b; return p < 0 || q < 0 || p + q > m || val(p, q) <= v + 1e-12; })) out.push([i / m, j / m, (m - i - j) / m]);
  }
  return out;
}

// ======================================================================= subtractive vs multiplicative
export const NORM_THETA = 0.55;
// One neuron with input synapses only, so h = w / Σ_b w_b. Subtractive: dw_b = (Σh)_b − θ − w_b (the page's rule, with
// its floor). Multiplicative: dw_b = (Σh)_b − α w_b with α = 1ᵀΣh / 1ᵀw, which holds the neuron's total weight fixed.
export function singleNeuron(S, w0, mode, { theta = NORM_THETA, steps = 4000, dt = 0.05, wmin = 1e-4, record = 20 } = {}) {
  const w = w0.slice(), path = [];
  for (let t = 0; t <= steps; t++) {
    const s = w[0] + w[1] + w[2], h = w.map(x => x / s);
    if (t % record === 0) path.push(h);
    const Sh = [0, 1, 2].map(a => S[a * 3] * h[0] + S[a * 3 + 1] * h[1] + S[a * 3 + 2] * h[2]);
    if (mode === "sub") for (let b = 0; b < 3; b++) w[b] = Math.max(wmin, w[b] + dt * (Sh[b] - theta - w[b]));
    else { const al = (Sh[0] + Sh[1] + Sh[2]) / s; for (let b = 0; b < 3; b++) w[b] = Math.max(0, w[b] + dt * (Sh[b] - al * w[b])); }
  }
  return path;
}
export function mountNorm(slot, ctx) {
  const S = M.sigmaOf(M.SIGMAS.tri), rng = makeRng(3);
  const starts = Array.from({ length: 40 }, () => [0, 1, 2].map(() => rng.exponential(0.2)));
  const runs = { sub: starts.map(w => singleNeuron(S, w, "sub")), mult: starts.map(w => singleNeuron(S, w, "mult")) };
  const parts = T.stableParts(S, 3, NORM_THETA);
  const { values, vectors } = eigh(S, 3);
  let v = [0, 1, 2].map(i => vectors[i * 3]); const sm = v[0] + v[1] + v[2]; v = v.map(x => x / sm);
  const row = ui.h("div", { class: "pc-row", style: { "--cols": 2 } });
  const TITLES = { sub: `Subtractive: θ = ${fx(NORM_THETA)} taken from every synapse's growth`, mult: "Multiplicative: each neuron's synapses scaled to a fixed total" };
  for (const mode of ["sub", "mult"]) {
    const box = ui.h("div");
    box.append(ui.h("div", { class: "pc-panel-title", text: TITLES[mode] }));
    row.append(box);
    canvasPanel(box, 0.86, (g, W, H) => {
      const G = triGeom(W, H - 14, 34);
      frame(g, G, { sq: 11, font: 14 });
      g.lineWidth = 1; g.strokeStyle = "rgba(22,22,26,0.32)";
      for (const p of runs[mode]) { g.beginPath(); p.forEach((h, f) => { const [x, y] = G.at(h); g[f ? "lineTo" : "moveTo"](x, y); }); g.stroke(); }
      for (const p of runs[mode]) { const [x, y] = G.at(p[0]); g.beginPath(); g.arc(x, y, 2, 0, 2 * Math.PI); g.fillStyle = P.mute; g.fill(); }
      if (mode === "sub") for (const q of parts) ring(g, G, q.h, 12);
      else { const [x, y] = G.at(v); star(g, x, y, 13); }
      for (const p of runs[mode]) { const h = p[p.length - 1], [x, y] = G.at(h); dot(g, x, y, 4.5, mixBary(h, BCOL)); }
    });
  }
  const foot = ui.h("div", { class: "pc-foot" });
  ui.legend({ parent: foot, items: [
    { label: "start", color: P.mute, shape: "dot" },
    { label: "path", color: P.fg, shape: "line" },
    { label: "end (colour: the mixture)", color: mixBary([0.5, 0.5, 0], BCOL), shape: "dot" },
    { label: "stable part of Σ − θ (left)", color: P.fg, shape: "ring" },
    { label: "☆ top eigenvector of Σ (right)", color: "transparent", shape: "dot" }] });
  const ends = mode => { const m = new Map(); for (const p of runs[mode]) { const k = hstr(p[p.length - 1]); m.set(k, (m.get(k) || 0) + 1); } return [...m]; };
  const read = ui.h("div", { class: "pc-read" });
  read.innerHTML = `Left: the 40 neurons end at ${ends("sub").map(([k, n]) => `${k} (${n})`).join(" and ")}, the stable parts. ` +
    `Right: they end at ${ends("mult").map(([k, n]) => `${k} (${n})`).join(", ")}; the top eigenvector of Σ, scaled to sum 1, is ${hstr(v)} (eigenvalue ${fx(values[0], 3)}).`;
  slot.replaceChildren(row, foot, read);
}
