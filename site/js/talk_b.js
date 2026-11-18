// The talk, part B (slides 5-7): grouping, hierarchy, expansion code. Toy figures only; no model code.
const NS = "http://www.w3.org/2000/svg";
const FONT = "var(--font-sans, Inter, sans-serif)";
const CAT = i => `var(--c-cat-${i % 8})`;

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
  initTree(root);
  initFan(root, ctx);
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
    bval.textContent = `b = ${bOf(+slider.value).toFixed(3)} · ${groups} groups, ${alone} alone`;
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

/* ---------- Slide 6: tree of groups + block covariance ---------- */
function initTree(root) {
  const svg = root.querySelector(".tb-tree");
  if (!svg) return;
  const slider = root.querySelector(".tb-res"), val = root.querySelector(".tb-resval");
  const Y = [70, 185, 300];
  const nodes = [
    [{ x: 150, lo: 0, hi: 8 }],
    [{ x: 80, lo: 0, hi: 4 }, { x: 220, lo: 4, hi: 8 }],
    [{ x: 45, lo: 0, hi: 2 }, { x: 115, lo: 2, hi: 4 }, { x: 185, lo: 4, hi: 6 }, { x: 255, lo: 6, hi: 8 }],
  ];
  const colourOf = (lv, k) => (lv === 0 ? CAT(0) : lv === 1 ? CAT(k + 1) : CAT(k + 3));
  const gT = el("g", {}, svg);
  text(svg, 150, 26, "groups", { "text-anchor": "middle", fill: "var(--fg)", "font-size": 15, "font-weight": 600 });
  text(svg, 192, Y[0] + 5, "coarse (large b)", { "font-size": 13 });
  text(svg, 150, Y[2] + 46, "fine (small b)", { "text-anchor": "middle", "font-size": 13 });
  // covariance heatmap
  const X0 = 352, Y0 = 50, c = 32;
  const cov = (i, j) => (i === j ? 1 : (i >> 1) === (j >> 1) ? 0.75 : (i >> 2) === (j >> 2) ? 0.4 : 0.08);
  text(svg, X0 + 4 * c, 26, "input correlations", { "text-anchor": "middle", fill: "var(--fg)", "font-size": 15, "font-weight": 600 });
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++)
    el("rect", { x: X0 + j * c, y: Y0 + i * c, width: c - 1, height: c - 1, fill: "var(--c-cat-0)", "fill-opacity": cov(i, j).toFixed(2) }, svg);
  for (let i = 0; i < 8; i++) text(svg, X0 - 8, Y0 + i * c + 21, String(i + 1), { "text-anchor": "end", "font-size": 13 });
  const gBox = el("g", {}, svg);
  function draw() {
    const L = +slider.value;
    gT.replaceChildren(); gBox.replaceChildren();
    for (let lv = 0; lv < 3; lv++) nodes[lv].forEach((n, k) => {
      if (lv < 2) nodes[lv + 1].filter(m => m.lo >= n.lo && m.hi <= n.hi).forEach(m =>
        el("line", { x1: n.x, y1: Y[lv], x2: m.x, y2: Y[lv + 1], stroke: "var(--line)", "stroke-width": 2 }, gT));
    });
    for (let lv = 0; lv < 3; lv++) nodes[lv].forEach((n, k) => {
      const on = lv === L;
      el("circle", { cx: n.x, cy: Y[lv], r: 8 + 3.2 * (n.hi - n.lo), fill: on ? colourOf(lv, k) : "var(--wash, #eee)",
        stroke: on ? "var(--fg)" : "var(--line)", "stroke-width": on ? 2 : 1, opacity: lv > L ? 0.35 : 1 }, gT);
      if (on) el("rect", { x: X0 + n.lo * c - 2, y: Y0 + n.lo * c - 2, width: (n.hi - n.lo) * c + 3, height: (n.hi - n.lo) * c + 3,
        fill: "none", stroke: colourOf(lv, k), "stroke-width": 3.5, rx: 3 }, gBox);
    });
    val.textContent = `${nodes[L].length} group${L ? "s" : ""}`;
  }
  slider.addEventListener("input", draw);
  draw();
}

/* ---------- Slide 7: response curves as functions of one mode ---------- */
function initFan(root, ctx) {
  const svg = root.querySelector(".tb-fan");
  if (!svg) return;
  const cap = root.querySelector(".tb-fancap");
  const bH = root.querySelector(".tb-hope"), bM = root.querySelector(".tb-meas");
  const x0 = 60, x1 = 616, y0 = 24, y1 = 340, zmin = -2, zmax = 2;
  const X = z => x0 + (z - zmin) / (zmax - zmin) * (x1 - x0), Yp = v => y1 - v * (y1 - y0);
  el("line", { x1: x0, y1, x2: x1, y2: y1, stroke: "var(--mute)", "stroke-width": 1.5 }, svg);
  el("line", { x1: x0, y1: y0, x2: x0, y2: y1, stroke: "var(--mute)", "stroke-width": 1.5 }, svg);
  text(svg, (x0 + x1) / 2, 380, "shared input pattern z", { "text-anchor": "middle", fill: "var(--fg)", "font-size": 15 });
  text(svg, 22, (y0 + y1) / 2, "neuron response", { "text-anchor": "middle", fill: "var(--fg)", "font-size": 15,
    transform: `rotate(-90 22 ${(y0 + y1) / 2})` });
  const sig = u => 1 / (1 + Math.exp(-u)), sp = u => Math.log1p(Math.exp(u));
  const K = 7;
  // hoped-for: varied thresholds, gains and shapes
  const hope = [
    z => sig(4 * (z + 1.2)), z => sig(2.5 * (z - 0.2)), z => sig(6 * (z - 1.1)),
    z => sp(2 * (z + 0.3)) / sp(4.6), z => 1 - sig(3 * (z - 0.6)), z => Math.exp(-((z + 0.5) ** 2) / 0.35),
    z => Math.max(0, z - 1) / 1,
  ];
  // measured: one prototype shape, members within ~2% (rescaled)
  const proto = z => sp(1.6 * (z - 0.1)) / sp(1.6 * 1.9);
  const meas = Array.from({ length: K }, (_, k) => z => proto(z) * (1 + 0.025 * Math.sin(1.7 * k + 2.3 * z)) );
  const paths = Array.from({ length: K }, (_, k) => el("path", { fill: "none", stroke: CAT(k), "stroke-width": 3, "stroke-linecap": "round" }, svg));
  let t = 0, target = 0, raf = 0;
  function draw() {
    paths.forEach((p, k) => {
      let d = "";
      for (let i = 0; i <= 120; i++) {
        const z = zmin + (zmax - zmin) * i / 120;
        const v = (1 - t) * hope[k](z) + t * meas[k](z);
        d += `${i ? "L" : "M"}${X(z).toFixed(1)},${Yp(Math.min(1.02, Math.max(0, v))).toFixed(1)}`;
      }
      p.setAttribute("d", d);
    });
    cap.textContent = target
      ? "Measured (sketch of the result): each neuron rescaled; the shapes nearly coincide, differing by about 2% of variance."
      : "The hoped-for picture: neurons in one group respond to the same pattern, each with its own threshold and shape. Illustration.";
  }
  function anim() {
    raf = 0;
    const dt = target - t;
    t = Math.abs(dt) < 0.02 || !ctx.visible ? target : t + dt * 0.12;
    draw();
    if (t !== target) raf = requestAnimationFrame(anim);
  }
  function set(m) {
    target = m; bH.setAttribute("aria-pressed", String(!m)); bM.setAttribute("aria-pressed", String(!!m));
    if (!raf) raf = requestAnimationFrame(anim);
  }
  bH.addEventListener("click", () => set(0));
  bM.addEventListener("click", () => set(1));
  draw();
}
