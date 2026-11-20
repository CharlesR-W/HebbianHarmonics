// The talk, part B (slides 5-6): grouping (toy), and the stable Perron parts of a three-input covariance as the
// threshold θ moves (computed exactly here).
const NS = "http://www.w3.org/2000/svg";
const FONT = "var(--font-sans, Inter, sans-serif)";
const CAT = i => `var(--c-cat-${i % 8})`;
const INCOL = ["var(--c-cat-0)", "var(--c-cat-1)", "var(--c-cat-2)"];

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function text(parent, x, y, s, extra = {}) {
  const t = el("text", { x, y, "font-size": 14, fill: "var(--mute)", style: `font-family:${FONT}`, ...extra }, parent);
  t.textContent = s;
  return t;
}
function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

export function init(root, ctx) {
  initScatter(root, ctx);
  initPerron(root);
}

/* ---------- Slide 5: toy grouping in the simplex ---------- */
function initScatter(root, ctx) {
  const svg = root.querySelector(".tb-scatter");
  if (!svg) return;
  const slider = root.querySelector(".tb-b"), bval = root.querySelector(".tb-bval");
  const A = [320, 34], B = [52, 446], C = [588, 446], S = 536; // side length in px = 1 unit
  el("path", { d: `M${A}L${B}L${C}Z`, fill: "none", stroke: "var(--line)", "stroke-width": 1.5 }, svg);
  text(svg, A[0], A[1] - 10, "input A", { "text-anchor": "middle", "font-size": 15 });
  text(svg, B[0] - 4, B[1] + 18, "input B", { "text-anchor": "start", "font-size": 15 });
  text(svg, C[0] + 4, C[1] + 18, "input C", { "text-anchor": "end", "font-size": 15 });
  const gEdge = el("g", {}, svg), gDot = el("g", {}, svg);
  const N = 40;
  const dots = Array.from({ length: N }, () => el("circle", { r: 6.5, stroke: "var(--bg, #fff)", "stroke-width": 1.5 }, gDot));
  let P = [], seed = 7, frame = 0, raf = 0;
  const bary = (a, b, c) => [a * A[0] + b * B[0] + c * C[0], a * A[1] + b * B[1] + c * C[1]];
  function reset() {
    const r = rng(seed++);
    const centres = [[0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7], [0.42, 0.42, 0.16], [0.2, 0.4, 0.4]];
    P = [];
    for (let i = 0; i < N; i++) {
      let w;
      if (r() < 0.65) {
        const c = centres[Math.floor(r() * centres.length)];
        w = c.map(v => Math.max(0.01, v + (r() - 0.5) * 0.36));
      } else w = [r(), r(), r()].map(v => -Math.log(v + 1e-9));
      const s = w[0] + w[1] + w[2];
      P.push(bary(w[0] / s, w[1] / s, w[2] / s));
    }
    frame = 0;
  }
  const bOf = v => 0.002 * Math.pow(100, v / 100); // 0.002 .. 0.2
  function step() {
    const r = Math.sqrt(2 * bOf(+slider.value)) * S, r2 = r * r;
    const next = P.map((p, i) => {
      let sx = 0, sy = 0, n = 0;
      for (let j = 0; j < N; j++) {
        const dx = P[j][0] - p[0], dy = P[j][1] - p[1];
        if (j !== i && dx * dx + dy * dy < r2) { sx += P[j][0]; sy += P[j][1]; n++; }
      }
      if (!n) return p;
      const k = 0.035;
      return [p[0] + k * (sx / n - p[0]), p[1] + k * (sy / n - p[1])];
    });
    P = next;
    return r;
  }
  function draw(r) {
    const r2 = r * r, parent = [...Array(N).keys()];
    const find = x => (parent[x] === x ? x : (parent[x] = find(parent[x])));
    gEdge.replaceChildren();
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const dx = P[j][0] - P[i][0], dy = P[j][1] - P[i][1], d2 = dx * dx + dy * dy;
      if (d2 < r2) {
        parent[find(i)] = find(j);
        el("line", { x1: P[i][0], y1: P[i][1], x2: P[j][0], y2: P[j][1], stroke: "var(--c-edge, #888)",
          "stroke-width": 1.4, "stroke-opacity": (0.15 + 0.6 * (1 - Math.sqrt(d2) / r)).toFixed(2) }, gEdge);
      }
    }
    const size = {}, colour = {};
    for (let i = 0; i < N; i++) size[find(i)] = (size[find(i)] || 0) + 1;
    Object.keys(size).filter(k => size[k] > 1).sort((a, b) => size[b] - size[a]).forEach((k, c) => (colour[k] = CAT(c)));
    let groups = 0, alone = 0;
    for (const k in size) size[k] > 1 ? groups++ : alone++;
    dots.forEach((d, i) => {
      d.setAttribute("cx", P[i][0].toFixed(1)); d.setAttribute("cy", P[i][1].toFixed(1));
      d.setAttribute("fill", colour[find(i)] || "var(--mute, #999)");
    });
    bval.textContent = `b = ${bOf(+slider.value).toFixed(3)} · ${groups} group${groups === 1 ? "" : "s"}, ${alone} alone`;
  }
  function loop() {
    raf = 0;
    if (!ctx.visible) return;
    const r = frame < 420 ? step() : Math.sqrt(2 * bOf(+slider.value)) * S;
    frame++;
    draw(r);
    if (frame < 420) raf = requestAnimationFrame(loop);
  }
  const start = () => { if (!raf) raf = requestAnimationFrame(loop); };
  root.querySelector(".tb-replay").addEventListener("click", () => { reset(); start(); });
  slider.addEventListener("input", () => { frame = Math.min(frame, 300); start(); });
  if (ctx.onShow) ctx.onShow(start);
  const sec = svg.closest(".slide");
  if (sec && "IntersectionObserver" in window) {
    new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { reset(); start(); } }, { threshold: 0.5 }).observe(sec);
  }
  reset(); draw(Math.sqrt(2 * bOf(+slider.value)) * S); start();
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
// (A h)_b <= 0 off the support (strictly: at equality the part is marginal and,
// as on the Perron page, not drawn).
function stableParts(theta) {
  const A = SIG.map(r => r.map(v => v - theta)), out = [];
  for (let mask = 1; mask < 8; mask++) {
    const S = [0, 1, 2].filter(b => mask & (1 << b));
    const { vals, vecs } = eigSym(S.map(i => S.map(j => A[i][j])));
    let v = vecs[0]; const sum = v.reduce((x, y) => x + y, 0); if (sum < 0) v = v.map(x => -x);
    if (vals[0] <= 1e-9 || v.some(x => x <= 1e-9) || (vals.length > 1 && vals[1] >= vals[0] - 1e-9)) continue;
    const h = [0, 0, 0], tot = v.reduce((x, y) => x + y, 0);
    S.forEach((b, k) => (h[b] = v[k] / tot));
    const viol = [0, 1, 2].some(b => !S.includes(b) && A[b].reduce((acc, x, c) => acc + x * h[c], 0) > -1e-12);
    if (!viol) out.push({ S, h });
  }
  return out;
}

function initPerron(root) {
  const svg = root.querySelector("#talk-b-perron-svg"), slider = root.querySelector("#talk-b-theta"), val = root.querySelector("#talk-b-thval");
  if (!svg || !slider) return;
  const C = [[320, 70], [88, 392], [552, 392]];
  const at = h => [h[0] * C[0][0] + h[1] * C[1][0] + h[2] * C[2][0], h[0] * C[0][1] + h[1] * C[1][1] + h[2] * C[2][1]];
  el("path", { d: `M${C[0]}L${C[1]}L${C[2]}Z`, fill: "none", stroke: "var(--line)", "stroke-width": 1.5 }, svg);
  const gS = el("g", {}, svg);
  C.forEach(([x, y], b) => {
    el("rect", { x: x - 10, y: y - 10, width: 20, height: 20, rx: 3, fill: INCOL[b] }, svg);
    text(svg, b === 0 ? x + 20 : x, b === 0 ? y + 6 : y + 34, `input ${b + 1}`, { "text-anchor": b === 0 ? "start" : b === 1 ? "start" : "end", "font-size": 15 });
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
  const label = text(svg, 620, 60, "", { "font-weight": 600, "text-anchor": "end", "font-size": 15, fill: "var(--fg)" });
  const sname = S => (S.length === 3 ? "all three inputs" : S.length === 2 ? `inputs ${S[0] + 1} + ${S[1] + 1}` : `input ${S[0] + 1} alone`);
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
    label.textContent = `θ = ${theta.toFixed(2)}: ${parts.length} Perron part${parts.length === 1 ? "" : "s"}`;
    val.textContent = parts.map(p => sname(p.S)).join(" · ");
  }
  slider.addEventListener("input", draw);
  draw();
}
