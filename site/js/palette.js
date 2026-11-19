// Colour grammar for JS. Values are read from the CSS custom properties in css/tokens.css (the single source of
// truth) when this module is first imported; stylesheets always load before module scripts run.
// See DESIGN.md "Colour" for which map encodes what.

const css = getComputedStyle(document.documentElement);
const v = name => css.getPropertyValue(name).trim();
const list = (prefix, n) => Array.from({ length: n }, (_, i) => v(`${prefix}${i}`));

export const P = {
  bg: v("--bg"), surface: v("--surface"), fg: v("--fg"), ink2: v("--ink-2"), mute: v("--mute"), mute2: v("--mute-2"),
  line: v("--line"), wash: v("--wash"), accent: v("--accent"),
  cat: list("--c-cat-", 8),          // clusters, series, boundary neurons (B <= 8): identity, fixed order
  other: v("--c-other"),             // unassigned / "other" / folded
  interior: v("--c-interior"), boundary: v("--c-boundary"), edge: v("--c-edge"),
  seq: list("--c-seq-", 9), ink: list("--c-ink-", 9), div: list("--c-div-", 9),
};

// ---- colour arithmetic
export function hexToRgb(hex) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
export const rgbToCss = ([r, g, b], a = 1) => a >= 1 ? `rgb(${r | 0},${g | 0},${b | 0})` : `rgba(${r | 0},${g | 0},${b | 0},${+a.toFixed(3)})`;
const toLin = c => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
const toSrgb = c => 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);

// ---- colour maps: t in [0, 1] -> [r, g, b] (piecewise linear in linear-light RGB through the token stops)
const MAPS = { seq: P.seq, ink: P.ink, div: P.div };
const cache = {};
function stops(name) {
  if (!cache[name]) cache[name] = (MAPS[name] || P.seq).map(h => hexToRgb(h).map(toLin));
  return cache[name];
}
export function cmap(name) {
  const s = stops(name), n = s.length - 1;
  return t => {
    t = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
    const x = t * n, i = Math.min(n - 1, Math.floor(x)), f = x - i, a = s[i], b = s[i + 1];
    return [toSrgb(a[0] + (b[0] - a[0]) * f), toSrgb(a[1] + (b[1] - a[1]) * f), toSrgb(a[2] + (b[2] - a[2]) * f)];
  };
}
// CSS colour for value x on map `name` over [lo, hi]. For "div", pass a symmetric range (e.g. [-m, m]).
export function colorOf(name, x, lo = 0, hi = 1) { return rgbToCss(cmap(name)((x - lo) / ((hi - lo) || 1))); }
// 256-entry RGBA lookup table (Uint8ClampedArray, 1024 bytes) for fast canvas rendering.
export function lut(name, n = 256) {
  const f = cmap(name), out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) { const c = f(i / (n - 1)); out.set([c[0], c[1], c[2], 255], i * 4); }
  return out;
}
// CSS linear-gradient string for a colour bar
export function gradientCss(name, dir = "to right") {
  const f = cmap(name);
  return `linear-gradient(${dir}, ${Array.from({ length: 9 }, (_, i) => rgbToCss(f(i / 8)) + ` ${(i / 8 * 100).toFixed(1)}%`).join(", ")})`;
}

// Categorical colour by index; past the end it folds to "other" (never cycled).
export const catColor = i => (i >= 0 && i < P.cat.length ? P.cat[i] : P.other);
// Boundary neuron b of B: its categorical hue when B <= 8, else ink.
export const boundaryColor = (b, B) => (B <= P.cat.length ? P.cat[b] : P.boundary);

// Barycentric colour: an interior neuron with harmonic embedding h (probability vector over boundary neurons) is
// drawn in the h-weighted mix of the boundary hues (linear light). Used when B <= 4. The centroid is a muted grey.
export function mixBary(h, colors) {
  const rgb = colors.map(c => hexToRgb(c).map(toLin));
  let r = 0, g = 0, b = 0, s = 0;
  for (let k = 0; k < h.length; k++) { const w = Math.max(0, h[k]); r += w * rgb[k][0]; g += w * rgb[k][1]; b += w * rgb[k][2]; s += w; }
  s = s || 1;
  return rgbToCss([toSrgb(r / s), toSrgb(g / s), toSrgb(b / s)]);
}
