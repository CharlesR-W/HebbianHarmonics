// The primer, part B (slides 5-7): a precomputed covariance-rule film (filmplayer.js), the stable Perron parts of a
// three-input covariance as θ moves (exact, computed here), and the distance rule on two neurons (shared drawing
// from part A).
import { filmPlayer } from "./filmplayer.js";
import { twoNeurons } from "./primer_a.js";
const NS = "http://www.w3.org/2000/svg";
const FONT = "var(--font-sans, Inter, sans-serif)";
const INCOL = ["var(--c-cat-0)", "var(--c-cat-1)", "var(--c-cat-2)"];

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function text(parent, x, y, s, extra = {}) {
  const t = el("text", { x, y, "font-size": 15, fill: "var(--fg)", style: `font-family:${FONT}`, ...extra }, parent);
  t.textContent = s;
  return t;
}

export function init(root, ctx) {
  initFilm(root, ctx);
  initPerron(root);
  twoNeurons(root.querySelector("#primer-b-rule-svg"), root.querySelector("#primer-b-rule-slider"), "dist");
}

/* ---------- Slide 5: the covariance rule at N = 300, replayed (lazy: loads when the slide comes near) ---------- */
function initFilm(root, ctx) {
  const host = root.querySelector(".tb-film");
  if (!host) return;
  let fp = null;
  const start = () => {
    if (fp) return;
    fp = filmPlayer(host, { filter: e => e.family === "A" && e.rule === "cov", id: "A_cov_ind_th0.4", height: 380, fps: 8 });
  };
  const sec = host.closest(".slide");
  if ("IntersectionObserver" in window && sec) {
    const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); start(); } }, { rootMargin: "400px 0px" });
    io.observe(sec);
  } else start();
  ctx.onHide?.(() => fp?.pause());
}

/* ---------- Slide 6: stable parts of A = Σ - θ11ᵀ on three inputs ---------- */
const SIG = [[1, 0.6, 0.25], [0.6, 1, 0.5], [0.25, 0.5, 1]];

// Eigen-decomposition of a small symmetric matrix (cyclic Jacobi). Returns {vals, vecs} with vecs[k] the k-th vector.
function eigSym(M) {
  const n = M.length, a = M.map(r => r.slice()), V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] ** 2;
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-30) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  const vals = a.map((r, i) => r[i]);
  const order = vals.map((v, i) => i).sort((i, j) => vals[j] - vals[i]);
  return { vals: order.map(i => vals[i]), vecs: order.map(i => V.map(r => r[i])) };
}

// Stable parts: supports S whose A_SS has a positive top eigenvector with c > 0, c strictly the top eigenvalue, and
// (A h)_b <= 0 off the support (the Perron-parts chapter's test; at equality the part is marginal).
function stableParts(theta) {
  const A = SIG.map(r => r.map(v => v - theta)), out = [];
  for (let mask = 1; mask < 8; mask++) {
    const S = [0, 1, 2].filter(b => mask & (1 << b));
    const { vals, vecs } = eigSym(S.map(i => S.map(j => A[i][j])));
    let v = vecs[0]; const sum = v.reduce((x, y) => x + y, 0); if (sum < 0) v = v.map(x => -x);
    if (vals[0] <= 1e-9 || v.some(x => x <= 1e-9) || (vals.length > 1 && vals[1] >= vals[0] - 1e-9)) continue;
    const h = [0, 0, 0], tot = v.reduce((x, y) => x + y, 0);
    S.forEach((b, k) => (h[b] = v[k] / tot));
    const viol = [0, 1, 2].some(b => !S.includes(b) && A[b].reduce((acc, x, c) => acc + x * h[c], 0) > 1e-12);
    if (!viol) out.push({ S, h });
  }
  return out;
}

function initPerron(root) {
  const svg = root.querySelector("#primer-b-perron-svg"), slider = root.querySelector("#primer-b-theta"), val = root.querySelector("#primer-b-thval");
  if (!svg || !slider) return;
  const C = [[320, 70], [88, 392], [552, 392]];
  const at = h => [h[0] * C[0][0] + h[1] * C[1][0] + h[2] * C[2][0], h[0] * C[0][1] + h[1] * C[1][1] + h[2] * C[2][1]];
  el("path", { d: `M${C[0]}L${C[1]}L${C[2]}Z`, fill: "none", stroke: "var(--line)", "stroke-width": 1.5 }, svg);
  const gS = el("g", {}, svg);
  C.forEach(([x, y], b) => {
    el("rect", { x: x - 10, y: y - 10, width: 20, height: 20, rx: 3, fill: INCOL[b] }, svg);
    text(svg, b === 0 ? x + 20 : x, b === 0 ? y + 6 : y + 34, `input ${b + 1}`, { "text-anchor": b === 0 ? "start" : b === 1 ? "start" : "end", fill: "var(--mute)" });
  });
  // Σ's top eigenvector (all entries positive here), normalised to sum 1
  const e = eigSym(SIG), v = e.vecs[0], sv = v.reduce((x, y) => x + y, 0), top = v.map(x => x / sv);
  const [ex, ey] = at(top);
  const star = (cx, cy, r) => {
    let d = "";
    for (let k = 0; k < 10; k++) { const a = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r * 0.45 : r; d += `${k ? "L" : "M"}${(cx + rr * Math.cos(a)).toFixed(1)},${(cy + rr * Math.sin(a)).toFixed(1)}`; }
    return d + "Z";
  };
  const gP = el("g", {}, svg);
  el("path", { d: star(ex, ey, 13), fill: "none", stroke: "var(--fg)", "stroke-width": 1.8 }, svg);
  text(svg, ex + 16, ey - 8, "top eigenvector of Σ", { "font-size": 13, fill: "var(--mute)" });
  const label = text(svg, 620, 26, "", { "font-weight": 600, "text-anchor": "end" });
  const sname = S => (S.length === 3 ? "all three inputs" : S.length === 2 ? `edge g${S[0] + 1}g${S[1] + 1}` : `corner g${S[0] + 1}`);
  function draw() {
    const theta = slider.value / 100, parts = stableParts(theta);
    gS.replaceChildren(); gP.replaceChildren();
    parts.forEach(({ S }, k) => {
      const col = `var(--c-cat-${3 + k})`;
      if (S.length === 2) el("line", { x1: C[S[0]][0], y1: C[S[0]][1], x2: C[S[1]][0], y2: C[S[1]][1], stroke: col, "stroke-width": 9, "stroke-opacity": 0.35, "stroke-linecap": "round" }, gS);
      else if (S.length === 1) el("circle", { cx: C[S[0]][0], cy: C[S[0]][1], r: 22, fill: col, "fill-opacity": 0.22 }, gS);
      else el("path", { d: `M${C[0]}L${C[1]}L${C[2]}Z`, fill: col, "fill-opacity": 0.08 }, gS);
    });
    parts.forEach(({ h }, k) => {
      const [x, y] = at(h);
      el("circle", { cx: x, cy: y, r: 11, fill: `var(--c-cat-${3 + k})`, stroke: "var(--bg, #fff)", "stroke-width": 2 }, gP);
    });
    label.textContent = `θ = ${theta.toFixed(2)}: ${parts.length} stable part${parts.length === 1 ? "" : "s"}`;
    val.textContent = parts.map(p => sname(p.S)).join(" · ");
  }
  slider.addEventListener("input", draw);
  draw();
}
