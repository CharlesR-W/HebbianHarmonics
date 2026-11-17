// Component library for chapters: controls, transport, stat tiles, legends, colour bars, figure frames,
// tooltip, number formatting, URL-hash parsing and maths rendering. API and examples: DESIGN.md.
// Every control factory returns an object with `.el` (append it anywhere) plus get()/set(); `opts.parent`
// appends it for you. Callbacks: onInput (continuous, sliders only) and onChange (committed value).
import { gradientCss } from "./palette.js";

export const tabsMeta = [
  { id: "idea", label: "The idea" },
  { id: "fusion", label: "Fusion" },
  { id: "many", label: "Many inputs" },
  { id: "diversity", label: "Diversity" },
  { id: "allocation", label: "Allocation" },
  { id: "readout", label: "Readout" },
  { id: "old", label: "The old rule" },
  { id: "method", label: "Choices & caveats" },
].map(t => ({ ...t, title: t.id === "idea" ? "Hebbian Learning on a Nonlinear Resistor Network" : `${t.label} · Hebbian Learning on a Nonlinear Resistor Network` }));

// ---------------------------------------------------------------- basics
// #tab/anchor?k=v  ->  { tab, anchor, params: URLSearchParams }
export function parseHash(hash = location.hash) {
  const raw = hash.replace(/^#/, ""), qi = raw.indexOf("?");
  const path = qi < 0 ? raw : raw.slice(0, qi), qs = qi < 0 ? "" : raw.slice(qi + 1);
  const [tab, anchor] = decodeURIComponent(path).split("/");
  return { tab: tab || "", anchor: anchor || null, params: new URLSearchParams(qs) };
}

export function renderMath(el) {
  if (!el || !window.renderMathInElement) return;
  window.renderMathInElement(el, {
    delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true },
      { left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }],
    ignoredClasses: ["katex", "no-math", "tooltip"], throwOnError: false,
  });
}

// h("div", {class: "x", style: {"--sw": c}, onclick: fn, text: "hi", html: "<b>b</b>"}, child, "text", ...)
export function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "style" && typeof v === "object") { for (const [p, x] of Object.entries(v)) { if (p.startsWith("--")) e.style.setProperty(p, x); else e.style[p] = x; } }
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else if (k === "text") e.textContent = v;
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c.el instanceof Node ? c.el : c);
  return e;
}
const put = (comp, parent) => { if (parent) parent.append(comp.el); return comp; };
let uid = 0;
const nextId = p => `${p}-${++uid}`;

// Numbers: 3 significant figures by default, true minus sign, no trailing zeros, scientific outside [1e-3, 1e5).
export function fmt(v, sig = 3) {
  if (v == null || Number.isNaN(v)) return "n/a";
  if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
  if (v === 0) return "0";
  const a = Math.abs(v);
  let s;
  if (a >= 1e5 || a < 1e-3) { const [m, e] = v.toExponential(sig - 1).split("e"); s = `${+m}×10${sup(+e)}`; }
  else s = String(+v.toPrecision(sig));
  return s.replace("-", "−");
}
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const sup = n => String(n).split("").map(c => SUP[c]).join("");
export const fmtFixed = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d).replace("-", "−") : "n/a");
export const fmtPct = (v, d = 0) => (Number.isFinite(v) ? (100 * v).toFixed(d) + "%" : "n/a");
export const fmtInt = v => (Number.isFinite(v) ? Math.round(v).toLocaleString("en-GB") : "n/a");

// ---------------------------------------------------------------- tooltip (one global #tooltip element)
const tipEl = () => document.getElementById("tooltip");
// content: an HTML string or a Node. Positioned beside the pointer, kept on screen.
export function showTip(x, y, content) {
  const t = tipEl(); if (!t) return;
  if (typeof content === "string") t.innerHTML = content; else t.replaceChildren(content);
  t.hidden = false;
  const r = t.getBoundingClientRect(), W = window.innerWidth, H = window.innerHeight;
  let lx = x + 14, ly = y + 14;
  if (lx + r.width > W - 8) lx = x - r.width - 14;
  if (ly + r.height > H - 8) ly = y - r.height - 14;
  t.style.left = Math.max(8, lx) + "px"; t.style.top = Math.max(8, ly) + "px";
}
export function hideTip() { const t = tipEl(); if (t) t.hidden = true; }
// tipRows("neuron 12", [{name: "cluster", value: "2", color: "#2a78d6", shape: "dot"|"square"}], "optional note")
export function tipRows(head, rows = [], note) {
  const d = h("div");
  if (head != null) d.append(h("div", { class: "tt-head" }, head));
  for (const r of rows) {
    const k = h("span", { class: "tt-k" });
    if (r.color) k.append(h("i", { class: "tt-sw" + (r.shape === "square" ? " sq" : ""), style: { "--sw": r.color } }));
    k.append(r.name);
    d.append(h("div", { class: "tt-row" }, k, h("span", { class: "tt-v" }, r.value == null ? "" : String(r.value))));
  }
  if (note) d.append(h("div", { class: "tt-note", html: note }));
  return d;
}

export const reducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

// ---------------------------------------------------------------- visibility-aware animation loop
// Runs fn(dtSeconds) on every animation frame while `el` is on screen (and its tab is shown). Returns {stop()}.
export function animate(el, fn) {
  let on = false, raf = 0, last = 0, stopped = false;
  const frame = ts => { if (!on || stopped) return; const dt = last ? Math.min(0.1, (ts - last) / 1000) : 1 / 60; last = ts; fn(dt); raf = requestAnimationFrame(frame); };
  const io = new IntersectionObserver(es => {
    const vis = es[es.length - 1].isIntersecting;
    if (vis && !on) { on = true; last = 0; raf = requestAnimationFrame(frame); }
    if (!vis) { on = false; cancelAnimationFrame(raf); }
  });
  io.observe(el);
  return { stop() { stopped = true; on = false; cancelAnimationFrame(raf); io.disconnect(); }, get running() { return on; } };
}

// ---------------------------------------------------------------- slider
// slider({label, min, max, value, step?, log?, sig?, fmt?, unit?, help?, marks?: [{value, label}], onInput, onChange, parent})
// label/help may contain $TeX$. log: the track is logarithmic between min > 0 and max (values snap to `sig`
// significant digits, default 2). Double-click the readout to return to the initial value.
export function slider(o) {
  const id = nextId("sl"), log = !!o.log, N = 1000;
  const f = o.fmt || (v => fmt(v, o.sig || 3));
  const init = o.value;
  const toPos = v => log ? Math.log(v / o.min) / Math.log(o.max / o.min) * N : v;
  const fromPos = p => {
    if (!log) return +p;
    const v = o.min * Math.pow(o.max / o.min, p / N);
    return +v.toPrecision(o.sig || 2);
  };
  const input = h("input", { type: "range", id, min: log ? 0 : o.min, max: log ? N : o.max, step: log ? 1 : (o.step ?? (o.max - o.min) / 100) });
  const out = h("output", { for: id, title: "double-click to reset" });
  const el = h("div", { class: "u-slider" + (o.marks ? " has-marks" : "") },
    h("div", { class: "u-row" }, h("label", { for: id, class: "u-name", html: o.label }), out),
    h("div", { class: "u-track" }, input));
  if (o.marks) {
    const mk = h("div", { class: "u-marks", "aria-hidden": "true" });
    const lo = log ? 0 : o.min, hi = log ? N : o.max;
    for (const m of o.marks) {
      const pct = (toPos(m.value) - lo) / (hi - lo) * 100;
      mk.append(h("span", { class: "u-mark", style: { left: `calc(${pct}% + ${(0.5 - pct / 100) * 14}px)` }, title: m.title || "" }, h("i"), m.label ? h("b", { html: m.label }) : null));
    }
    el.querySelector(".u-track").append(mk);
  }
  if (o.help) el.append(h("div", { class: "u-help", html: o.help }));
  let val = o.value;
  const show = () => { out.innerHTML = f(val) + (o.unit ? `<span class="u-unit">${o.unit}</span>` : ""); };
  const set = (v, { silent = true } = {}) => {
    val = log ? Math.min(o.max, Math.max(o.min, v)) : v; input.value = toPos(val); show();
    if (!silent) { o.onInput?.(val); o.onChange?.(val); }
  };
  input.addEventListener("input", () => { val = fromPos(input.value); show(); o.onInput?.(val); });
  input.addEventListener("change", () => o.onChange?.(val));
  out.addEventListener("dblclick", () => set(init, { silent: false }));
  set(o.value);
  renderMath(el);
  return put({ el, get: () => val, set, input, setDisabled(d) { input.disabled = d; el.classList.toggle("disabled", d); } }, o.parent);
}

// ---------------------------------------------------------------- segmented control, select, toggle
// options: [{value, label, title?}] or plain values. label may contain $TeX$.
const normOpts = opts => opts.map(x => (typeof x === "object" ? x : { value: x, label: String(x) }));

export function segmented(o) {
  const opts = normOpts(o.options);
  const seg = h("div", { class: "seg", role: "radiogroup", "aria-label": (o.label || "").replace(/<[^>]+>|\$/g, "") });
  let val = o.value ?? opts[0].value;
  const btns = opts.map(op => {
    const b = h("button", { type: "button", role: "radio", title: op.title, html: op.label });
    b.addEventListener("click", () => { if (val !== op.value) { set(op.value); o.onChange?.(val); } });
    seg.append(b); return b;
  });
  const set = v => { val = v; btns.forEach((b, i) => b.setAttribute("aria-checked", String(opts[i].value === v))); };
  set(val);
  const el = h("div", { class: "u-field" + (o.stack ? " stack" : "") }, o.label ? h("span", { class: "u-name", html: o.label }) : null, seg);
  if (o.help) el.append(h("div", { class: "u-help", html: o.help }));
  renderMath(el);
  return put({ el, get: () => val, set, setDisabled(v, d) { const i = opts.findIndex(x => x.value === v); if (i >= 0) btns[i].disabled = d; } }, o.parent);
}

export function select(o) {
  const opts = normOpts(o.options), id = nextId("sel");
  const sel = h("select", { id });
  for (const op of opts) sel.append(h("option", { value: String(op.value), text: op.label.replace(/<[^>]+>/g, "") }));
  const byStr = s => opts.find(op => String(op.value) === s)?.value;
  const set = v => { sel.value = String(v); };
  sel.addEventListener("change", () => o.onChange?.(byStr(sel.value)));
  set(o.value ?? opts[0].value);
  const el = h("div", { class: "u-field" + (o.stack ? " stack" : "") }, o.label ? h("label", { for: id, class: "u-name", html: o.label }) : null, sel);
  if (o.help) el.append(h("div", { class: "u-help", html: o.help }));
  renderMath(el);
  return put({ el, get: () => byStr(sel.value), set, select: sel }, o.parent);
}

export function toggle(o) {
  const input = h("input", { type: "checkbox", role: "switch" });
  input.checked = !!o.value;
  input.addEventListener("change", () => o.onChange?.(input.checked));
  const el = h("label", { class: "u-toggle" }, input, h("span", { class: "u-switch", "aria-hidden": "true" }), h("span", { class: "u-toggle-label", html: o.label }));
  const wrap = o.help ? h("div", { class: "u-toggle-wrap" }, el, h("div", { class: "u-help", html: o.help })) : el;
  renderMath(wrap);
  return put({ el: wrap, get: () => input.checked, set: v => { input.checked = !!v; }, input }, o.parent);
}

// ---------------------------------------------------------------- preset / tour chips
// chips({label: "Tours", items: [{id, label, note}], active?: id, onPick(item), parent})
// Picking a chip marks it active and shows its note (HTML, $TeX$ ok). Call .moved() when the reader then changes
// any control: the chip unmarks, the note dims and offers "back to this view" (which re-fires onPick).
export function chips(o) {
  const row = h("div", { class: "u-chip-row", role: "group" });
  const note = h("div", { class: "u-chip-note", hidden: true });
  const el = h("div", { class: "u-chips" }, o.label ? h("span", { class: "u-name", html: o.label }) : null, row, note);
  let active = null, last = null;
  const btns = new Map();
  for (const it of o.items) {
    const b = h("button", { type: "button", class: "u-chip", "aria-pressed": "false", html: it.label, title: it.title || "" });
    b.addEventListener("click", () => { set(it.id); o.onPick?.(it); });
    btns.set(it.id, b); row.append(b);
  }
  const set = id => {
    active = id; if (id) last = id;
    btns.forEach((b, k) => b.setAttribute("aria-pressed", String(k === id)));
    const it = o.items.find(x => x.id === (id || last));
    note.hidden = !it || !it.note; note.classList.remove("moved");
    if (it && it.note) { note.innerHTML = it.note; renderMath(note); }
  };
  const moved = () => {
    if (!active) return;
    const it = o.items.find(x => x.id === active); active = null;
    btns.forEach(b => b.setAttribute("aria-pressed", "false"));
    if (!it || !it.note) return;
    note.classList.add("moved");
    const back = h("button", { type: "button", class: "u-link", text: "back to this view" });
    back.addEventListener("click", () => { set(it.id); o.onPick?.(it); });
    note.append(" ", back);
  };
  if (o.active) set(o.active);
  renderMath(el);
  return put({ el, set, moved, get: () => active, clear: () => { active = null; last = null; set(null); } }, o.parent);
}

// ---------------------------------------------------------------- transport
// transport({onTick(speed) -> frame|null, onSeek(frame, i), onReset() -> frame|null, onRestore?(frame),
//            speeds: [1, 4, 16], speed, timeFmt(t), autoplay (ignored under reduced motion), maxHistory: 400,
//            watch: Element, parent})
// A frame is {t, ...anything}; it is what you need to redraw that moment (positions, weights, ...).
// Each animation frame while playing and on screen: frame = onTick(speed); frames are kept in a history that is
// thinned by halves past maxHistory, so the scrubber always spans the whole run. Return {..., done: true} to pause
// (e.g. converged). Scrubbing calls onSeek(frame, i) only. Pressing play while scrubbed back: with onRestore,
// the sim is restored to that frame and history after it is dropped; without it, playback jumps to the live end.
export function transport(o) {
  const speeds = normOpts((o.speeds || [1, 4, 16]).map(s => (typeof s === "object" ? s : { value: s, label: `×${s}` })));
  const maxH = o.maxHistory || 400;
  const fmtT = o.timeFmt || (t => `t = ${fmt(t, 3)}`);
  const play = h("button", { type: "button", class: "u-play", "aria-label": "Play" });
  const step = h("button", { type: "button", class: "u-tbtn", title: "Advance one step", "aria-label": "Step", html: ICON.step });
  const reset = h("button", { type: "button", class: "u-tbtn", title: "Reset to the initial condition", "aria-label": "Reset", html: ICON.reset });
  const scrub = h("input", { type: "range", class: "u-scrub", min: 0, max: 0, step: 1, value: 0, "aria-label": "Scrub through the recorded run" });
  const time = h("span", { class: "u-time" }, "t = 0");
  const status = h("span", { class: "u-tstatus" });
  const spd = segmented({ options: speeds, value: o.speed ?? speeds[0].value, onChange: v => { speed = v; } });
  spd.el.classList.add("u-speed"); spd.el.title = "Integration steps per animation frame";
  const el = h("div", { class: "u-transport" }, play, step, reset, h("div", { class: "u-scrub-wrap" }, scrub, h("div", { class: "u-tline" }, time, status)), spd);
  let speed = spd.get(), playing = false, frames = [], pos = -1, onScreen = false, raf = 0, wantAuto = !!o.autoplay && !reducedMotion();

  const setPlayIcon = () => { play.innerHTML = playing ? ICON.pause : ICON.play; play.setAttribute("aria-label", playing ? "Pause" : "Play"); el.classList.toggle("playing", playing); };
  const showTime = f => { time.textContent = f ? fmtT(f.t) : ""; };
  const syncScrub = () => { scrub.max = Math.max(0, frames.length - 1); scrub.value = pos < 0 ? frames.length - 1 : pos; el.classList.toggle("scrubbed", pos >= 0); };
  const record = f => {
    if (!f) return;
    frames.push(f);
    if (frames.length > maxH) frames = frames.filter((_, i) => i % 2 === 0 || i === frames.length - 1);
    showTime(f); syncScrub();
  };
  const tick = () => {
    const f = o.onTick(speed);
    if (f) { record(f); if (f.done) { pause(); status.textContent = f.doneLabel || "converged"; } }
  };
  const loop = () => { if (!playing || !onScreen) { raf = 0; return; } tick(); if (playing) raf = requestAnimationFrame(loop); else raf = 0; };
  const kick = () => { if (playing && onScreen && !raf) raf = requestAnimationFrame(loop); };
  const resumeFromScrub = () => {
    if (pos < 0) return;
    if (o.onRestore) { frames = frames.slice(0, pos + 1); o.onRestore(frames[pos]); }
    else if (frames.length) o.onSeek?.(frames[frames.length - 1], frames.length - 1);
    pos = -1; syncScrub();
  };
  function startPlay() { resumeFromScrub(); status.textContent = ""; playing = true; setPlayIcon(); kick(); }
  function pause() { playing = false; setPlayIcon(); if (raf) { cancelAnimationFrame(raf); raf = 0; } }
  play.addEventListener("click", () => (playing ? pause() : startPlay()));
  step.addEventListener("click", () => { pause(); resumeFromScrub(); status.textContent = ""; tick(); });
  reset.addEventListener("click", () => api.reset());
  scrub.addEventListener("input", () => {
    pause(); const i = +scrub.value;
    pos = i >= frames.length - 1 ? -1 : i;
    const f = frames[i]; if (f) { showTime(f); o.onSeek?.(f, i); }
    el.classList.toggle("scrubbed", pos >= 0);
  });
  // keyboard: space toggles play when focus is inside the watched figure (but not in a text field)
  const watch = o.watch || el;
  watch.addEventListener("keydown", e => {
    if (e.key === " " && !e.target.closest("input[type=text], input[type=number], select, textarea, button")) { e.preventDefault(); playing ? pause() : startPlay(); }
  });
  const io = new IntersectionObserver(es => {
    onScreen = es[es.length - 1].isIntersecting;
    if (onScreen && wantAuto) { wantAuto = false; startPlay(); }
    if (onScreen) kick();
  }, { threshold: 0.05 });
  io.observe(watch);

  const api = {
    el,
    play: startPlay, pause, toggle: () => (playing ? pause() : startPlay()), step: () => step.click(),
    get playing() { return playing; },
    get frames() { return frames; },
    get speed() { return speed; },
    setSpeed(v) { speed = v; spd.set(v); },
    // reset(): calls onReset() and starts a fresh history from the frame it returns. Keeps the play state.
    reset() { frames = []; pos = -1; status.textContent = ""; const f = o.onReset?.(); if (f) { record(f); o.onSeek?.(f, 0); } else syncScrub(); },
    record, seek(i) { scrub.value = i; scrub.dispatchEvent(new Event("input")); },
    clearHistory() { frames = frames.length ? [frames[frames.length - 1]] : []; pos = -1; syncScrub(); },
    setStatus(s) { status.textContent = s || ""; },
    destroy() { pause(); io.disconnect(); },
  };
  setPlayIcon();
  return put(api, o.parent);
}
const ICON = {
  play: '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4.5 2.6v10.8L13 8z" fill="currentColor"/></svg>',
  pause: '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="3.5" y="2.8" width="3.2" height="10.4" rx=".8" fill="currentColor"/><rect x="9.3" y="2.8" width="3.2" height="10.4" rx=".8" fill="currentColor"/></svg>',
  step: '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3 3.2v9.6L10 8z" fill="currentColor"/><rect x="11" y="3.2" width="2.2" height="9.6" rx=".6" fill="currentColor"/></svg>',
  reset: '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3.2 8a4.8 4.8 0 1 0 1.5-3.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M2.6 2.2v3.6h3.6" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
};

// ---------------------------------------------------------------- control panel (primary + "more" disclosure)
// controls({title?, primary: [comp|Node], advanced?: [comp|Node], advancedLabel?, open?, parent})
// At most ~4 primary controls; everything else goes in `advanced` (a closed disclosure by default).
export function controls(o) {
  const body = h("div", { class: "u-ctl-body" });
  const el = h("div", { class: "u-controls" }, o.title ? h("div", { class: "u-ctl-title", html: o.title }) : null, body);
  const add = c => { body.append(c.el || c); return c; };
  (o.primary || []).forEach(add);
  let adv = null, advBody = null;
  const ensureAdv = () => {
    if (adv) return;
    advBody = h("div", { class: "u-ctl-body" });
    adv = h("details", { class: "u-adv" }, h("summary", { html: o.advancedLabel || "More controls" }), advBody);
    if (o.open) adv.open = true;
    el.append(adv);
  };
  const addAdvanced = c => { ensureAdv(); advBody.append(c.el || c); return c; };
  (o.advanced || []).forEach(addAdvanced);
  renderMath(el);
  return put({ el, add, addAdvanced }, o.parent);
}

// ---------------------------------------------------------------- stat tiles
// stats({items: [{key, label, unit?, fmt?, help?}], cols?, parent}) -> {set(key, value, sub?), update({key: value})}
export function stats(o) {
  const el = h("div", { class: "u-stats", style: o.cols ? { "--cols": o.cols } : null });
  const tiles = {};
  for (const it of o.items) {
    const v = h("span", { class: "u-stat-v" }, "–");
    const sub = h("span", { class: "u-stat-sub" });
    const t = h("div", { class: "u-stat", title: it.help || "" },
      h("span", { class: "u-stat-k", html: it.label }), h("span", { class: "u-stat-row" }, v, it.unit ? h("span", { class: "u-unit", html: it.unit }) : null), sub);
    tiles[it.key] = { v, sub, f: it.fmt || (x => (typeof x === "number" ? fmt(x) : String(x))) };
    el.append(t);
  }
  renderMath(el);
  const set = (key, value, subText) => { const t = tiles[key]; if (!t) return; t.v.textContent = value == null ? "–" : t.f(value); if (subText !== undefined) t.sub.innerHTML = subText || ""; };
  return put({ el, set, update(obj) { for (const k in obj) set(k, obj[k]); } }, o.parent);
}

// ---------------------------------------------------------------- legend and colour bar
// legend({items: [{label, color, shape: "dot"|"square"|"line"|"dash"|"band"|"ring", onHover?(on)}], title?, parent})
// Shapes follow the neuron grammar: "dot" = interior neuron, "square" = boundary neuron, "line" = synapse.
export function legend(o) {
  const el = h("div", { class: "u-legend" }, o.title ? h("span", { class: "u-legend-title", html: o.title }) : null);
  const set = items => {
    el.querySelectorAll(".u-key").forEach(k => k.remove());
    for (const it of items) {
      const k = h("span", { class: "u-key" }, h("i", { class: `u-mk ${it.shape || "dot"}`, style: { "--sw": it.color } }), h("span", { html: it.label }));
      if (it.onHover) { k.addEventListener("pointerenter", () => it.onHover(true)); k.addEventListener("pointerleave", () => it.onHover(false)); k.classList.add("hov"); }
      el.append(k);
    }
    renderMath(el);
  };
  set(o.items || []);
  return put({ el, set }, o.parent);
}

// colorbar({map: "seq"|"ink"|"div", range: [lo, hi], label, ticks?: [values], fmt?, width?, parent}) -> {set({range,label,ticks})}
export function colorbar(o) {
  const bar = h("div", { class: "u-cbar-bar" });
  const ticks = h("div", { class: "u-cbar-ticks" });
  const lab = h("span", { class: "u-cbar-label" });
  const el = h("div", { class: "u-cbar", style: o.width ? { width: typeof o.width === "number" ? o.width + "px" : o.width } : null }, lab, bar, ticks);
  let st = { ...o };
  const set = (p = {}) => {
    st = { ...st, ...p };
    bar.style.background = gradientCss(st.map || "seq");
    lab.innerHTML = st.label || ""; renderMath(lab);
    const [lo, hi] = st.range || [0, 1], f = st.fmt || (x => fmt(x, 2));
    const tv = st.ticks || (st.map === "div" ? [lo, 0, hi] : [lo, hi]);
    ticks.replaceChildren(...tv.map(v => h("span", { style: { left: `${(v - lo) / ((hi - lo) || 1) * 100}%` } }, f(v))));
  };
  set();
  return put({ el, set }, o.parent);
}

// ---------------------------------------------------------------- figure frame and callouts
// figure({title?, sub?, plotted, values?, size: ""|"wide"|"full", card?: true, id?, parent}) -> {el, body, setCaption}
// The caption is definitional: "Plotted:" says what every axis, mark and colour is; "Values:" lists only the
// main quantities (parameters, N, B, ...). No interpretation in captions.
export function figure(o) {
  const body = h("div", { class: "fig-body" });
  const cap = h("figcaption");
  const el = h("figure", { class: ["fig", o.size, o.card === false ? null : "card"].filter(Boolean).join(" "), id: o.id },
    o.title ? h("div", { class: "fig-title", html: o.title }) : null, o.sub ? h("div", { class: "fig-sub", html: o.sub }) : null, body, cap);
  const setCaption = ({ plotted, values, html } = {}) => {
    cap.innerHTML = html || `${plotted ? `<b>Plotted:</b> ${plotted}` : ""}${values ? ` <b>Values:</b> ${values}` : ""}`;
    cap.hidden = !cap.innerHTML.trim(); renderMath(cap);
  };
  setCaption(o);
  renderMath(el);
  return put({ el, body, caption: cap, setCaption }, o.parent);
}

// callout({kind: ""|"key"|"caveat"|"found", title?, html, parent})
export function callout(o) {
  const el = h("aside", { class: `callout ${o.kind || ""}`.trim() }, o.title ? h("p", { class: "callout-title", html: o.title }) : null, h("div", { html: o.html }));
  renderMath(el);
  return put({ el }, o.parent);
}

// simLayout({parent, side?: px}) -> {el, view, side, below}: the big view on the left, controls and readouts
// on the right (side column default 320 px); stacks (view first) below 900 px.
export function simLayout(o = {}) {
  const view = h("div", { class: "sim-view" }), side = h("div", { class: "sim-side" }), below = h("div", { class: "sim-below" });
  const el = h("div", { class: "sim", style: o.side ? { "--side-w": o.side + "px" } : null }, view, side, below);
  return put({ el, view, side, below }, o.parent);
}
