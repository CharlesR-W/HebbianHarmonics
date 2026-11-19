# HebbianHarmonics site: design contract for chapter authors

Read this file and `SPEC.md`; nothing else is needed. Section 0 says what changed for v2; the rest is the
contract every chapter follows. Code examples below were written for v1 and use its names (`engine.js`, `theta`);
the APIs they show are unchanged. The `#perron` chapter (`content/perron.html`, `js/ch_perron.js`,
`js/ch_perron_model.js`; the v1 live figure, formerly in `#old`) is a complete chapter that uses most components (sim layout, transport, tours, controls,
stat tiles, legend, linked heatmap, side notes, callouts, a disclosure) and is the nearest working example.

## 0. What changed in v2 (2026-11-17)

- **Model.** The page is rebuilt around SPEC.md v2: on-site potentials, response functions $F_i$, the distance
  rule $dw_{ij}/dt = b - \alpha w_{ij} - \tfrac12 D_{ij}^2$. New chapters simulate with **`js/onsite_engine.js`**
  (pure ES module; reference `hebb/onsite.py`; parity tests `tests/onsite_*`) and load precomputed sweeps from
  `data/sweeps2/*.json` (each file carries its config). Use SPEC §1 notation in new chapters ($F_i$, $D_{ij}$,
  $b$, $\alpha$, $V_i$, $t_i$, $k_{lo}$, $k_{hi}$, $\sigma_t$, $s$, attachments $a_{ib}$); the v1 notation paragraph
  in section 4 applies to the old-rule chapter only.
- **Tabs** (2026-11-19 curation; see `docs/PLAN_curation.md`). Main path, in order (id: label): `primer` Primer
  (slides; marked with three squares) · `idea` The idea · `perron` Perron parts · `readout` Reading the weights ·
  `nonlinear` Nonlinear neurons. Then the **appendix**, drawn after a divider with a small "Appendix" label (class
  `tab-group`; appendix tabs carry class `tab-app`): `fusion` Fusion · `many` Many inputs · `diversity` Diversity ·
  `allocation` Allocation · `old` More on the covariance rule · `method` Choices & caveats. Kickers: "Part 2" …
  "Part 4" on the main path, "Appendix A" … "Appendix F" in the appendix. The empty hash shows `idea` (it carries
  the masthead, whose call to action points to the primer). Labels and groups are in `ui.js` `tabsMeta`; kickers,
  titles and the tab bar in `index.html` (`main.js` routes from `tabsMeta`, so adding a tab means editing those two
  places only). **Old addresses keep working:** `#talk…` redirects to `#primer…` (`tabAliases` in `ui.js`; ids
  `talk-*` became `primer-*`), and anchors that moved chapter are rewritten by `anchorMoves` in `main.js`
  (`#old/old-sim?tour=…` → `#perron/perron-sim?tour=…`; `#old?k=v` → `#perron?k=v`).
- **v1 survives in `#perron` (and the text of `#old`).** `js/engine.js` (with `ENGINE.md`, `hebb/core.py` and
  `tests/test_engine_parity.py`) is kept for the covariance-rule live figure; no other chapter should import it.
  **All data are synthetic** (MNIST was dropped). The other v1 chapters, their data and the v1
  experiment code were removed (git history keeps them); the v1 write-ups are in `docs/v1/`.
- **Placeholders.** Each new chapter starts as `content/<tab>.html` (the chapter's question and an
  "In preparation" note) and an empty `js/ch_<tab>.js`. Each author owns one disjoint pair of files, plus
  optional helper modules named `js/ch_<tab>_*.js` imported by `ch_<tab>.js` (as `ch_old_model.js`).
- **Ids are page-global.** Every chapter is loaded into the same document, so prefix element ids with the tab
  (`id="fusion-bistable"`, `id="readout-eq-rank"`); `#old` uses `old-`, `#perron` `perron-`, `#primer` `primer-`.

## 1. Files and who owns what

```
site/
  index.html            shell: banner, masthead, tabs, one empty <section> per chapter       (design system)
  css/tokens.css        ALL colours (single source of truth; JS reads it)                    (design system)
  css/site.css          type, masthead, tabs, prose, figures, callouts, side notes, tables   (design system)
  css/ui.css            components: controls, transport, tiles, legends, charts, 3D view     (design system)
  js/main.js            router, lazy chapter loading, maths, side notes, the ctx object       (design system)
  js/ui.js              components + helpers (sections 7)                                    (design system)
  js/charts.js          heatmap, filmstrip, lineChart, scatter, barChart, barMultiples, imageGrid (section 8)
  js/view3d.js          3D point/line viewer + simplex helpers (section 9)
  js/netview.js         neuron-space network diagram, seeded random layout (section 9b)
  js/filmplayer.js      player for the precomputed films in data/big/ (section 9c)
  js/palette.js         colour grammar for JS (reads css/tokens.css) (section 6)
  js/onsite_engine.js   the v2 simulation engine (separate owner; see SPEC.md §5)
  js/engine.js          the v1 engine, used only by the covariance-rule figure in #perron (see ENGINE.md)
  vendor/               KaTeX 0.16 and fonts (Source Serif 4, Inter, JetBrains Mono); nothing from a CDN
  content/<tab>.html    chapter prose fragment                                               (chapter author)
  js/ch_<tab>.js        chapter module                                                       (chapter author)
  data/sweeps2/         precomputed v2 sweeps (hebb/sweeps2/*.py)                            (data owner)
  data/big/             precomputed big-N films (hebb/big/*.py; format in data/big/README.md) (data owner)
  content/primer*.html, js/ch_primer.js, js/primer_[abc].js, css/primer.css   the Primer slides
```

A chapter author edits only `content/<tab>.html`, `js/ch_<tab>.js` and their own data files. If you need a new
component or a style change, ask for it to be added here rather than writing one-off CSS; small chapter-local
layout (a grid inside one figure) is fine as inline `style` or a `<style>`-free class built from existing ones.

Tabs: see section 0.
The chapter title and kicker ("Chapter 2 · live simulation") are in `index.html`; do not repeat them in the fragment.

### Running it

```bash
python3 -m http.server 8765 --bind 127.0.0.1 -d site      # then http://127.0.0.1:8765/#fusion
```
No build step. Everything is fetched with relative URLs, so it must be served over HTTP (not `file://`).
`http.server` sends no cache headers: after editing JS, reload with the cache disabled or add `?v=<n>` before the
`#` (e.g. `http://127.0.0.1:8765/?v=2#fusion`). Zero console errors is a requirement; check with
`playwright-cli console` (see the screenshot recipe in section 11).

## 2. How a chapter loads

When a tab is first shown, `main.js` fetches `content/<tab>.html` into the tab's `<article class="prose">`, renders
maths, numbers side notes and adds heading anchors, then does `import("./ch_<tab>.js")` and awaits
`init(panel, ctx)`. After `init` resolves it decorates the article again (so headings, maths and side notes that
your module added are handled too). Chapters load lazily, once; switching tabs does not re-run `init`.

**Fragments** are plain HTML: no `<html>`, `<head>`, `<body>` or `<script>`. Mark the places your module fills with
`data-slot="name"` and get them with `ctx.slot("name")`. Reference data with relative URLs (`data/sweeps2/resolution.json`).

**Chapter module** contract:

```js
// js/ch_many.js
import * as ui from "./ui.js";
import { heatmap, lineChart } from "./charts.js";
import { view3d } from "./view3d.js";
import { P, catColor } from "./palette.js";
import * as engine from "./engine.js";

export async function init(panel, ctx) {
  const theta = ctx.param("theta", 0.3);             // typed read from the URL with a default
  const hm = heatmap(ctx.slot("cov"), { map: "div", range: [-1, 1], zlabel: "$h_i^\\top\\Sigma_g h_j$" });
  ...
  ctx.onParams(p => applyState(p));                   // a deep link to this tab was followed while it is open
  ctx.onHide(() => view.setAutoRotate(false));        // tab hidden or browser tab hidden
}
```

`ctx` (one per chapter):

| member | meaning |
|---|---|
| `ctx.tab`, `ctx.panel`, `ctx.article` | tab id; its `<section>`; the `<article class="prose">` holding the fragment |
| `ctx.slot(name)` | `panel.querySelector('[data-slot="name"]')` |
| `ctx.params` | a fresh `URLSearchParams` of this chapter's URL state (`#many?b=24&theta=0.3` gives `b`, `theta`) |
| `ctx.param(name, fallback)` | typed read: number if `fallback` is a number, boolean (`1`/`true`) if boolean, else string |
| `ctx.setParams(obj, {push=false})` | merge into the URL; `null`/`undefined` deletes a key; numbers written to 4 significant digits, booleans as `1`/`0`. Uses `history.replaceState` (no history entry, no `hashchange`) unless `push: true` |
| `ctx.onParams(fn)` | `fn(URLSearchParams)` when the URL state changes from outside while this tab is shown (a deep link clicked, back/forward). Make it idempotent |
| `ctx.onShow(fn)` / `ctx.onHide(fn)` | tab (or browser tab) shown / hidden. `onShow` also fires immediately if the tab is already showing |
| `ctx.visible` | boolean |
| `ctx.renderMath(el)` | render `$…$` / `$$…$$` inside `el` (same as `ui.renderMath`) |

**URL state.** Write every reader-facing setting with `ctx.setParams` so any view can be shared as a link. Keep
keys short and lower-case (`theta`, `wmin`, `gamma`, `p`, `n`, `b`, `seed`, `preset`, `tour`). Read them once in
`init` (with defaults) and in `onParams`. When the reader returns to a chapter through its bare tab link, the
shell restores the chapter's last URL state for you. Hash grammar handled by the shell:
`#tab`, `#tab/anchor-id` (scroll to an element inside that chapter), `#tab?k=v&…` (state), `#anchor-id` (finds a
loaded chapter holding it). A link to the hash you are already on still re-applies its state.

## 3. Tone and captions

- **First person, exploratory.** "I expected one cluster per eigenvector; I got thresholded parts." Say what was
  tried, what happened, and what is still open. Where the note's conjectures fail or need amending, say so plainly
  and show the evidence (SPEC.md asks for honesty above all).
- **Captions are definitional and do not interpret.** A caption has a **Plotted:** part that defines every axis,
  mark, colour and line as a quantity with its symbol (and the sort order, projection, colour range), and an
  optional **Values:** part listing only the parameters needed to reproduce the figure ($B$, $N$, $\theta$, seed,
  ...). No "clearly", "as expected", "shows that"; interpretation belongs in the prose around the figure.
- **Synthetic data are labelled** in the figure title or caption: `<span class="synthetic-tag">synthetic Σ</span>`.
  Precomputed figures say where the numbers come from (`data/sweeps2/...`, `data/big/<id>`, run id).
- **Every live simulation states its mode** (mean-field with exact $\Sigma_g$, or minibatch with batch size).
- Numbers in prose that come from data are formatted like the UI (`ui.fmt`: 3 significant figures, true minus).
- Spelling: British ("colour", "normalise"), as in the rest of the page.

## 4. Page patterns (HTML in fragments)

```html
<p class="lede">The opening paragraph, slightly larger: what I wanted to know.</p>
<h2 id="pruning">Pruning is necessary</h2>          <!-- every h2/h3 gets an id; link as #three/pruning -->
<h3 id="pruning-why">Why</h3>
```

**Figures.** Width modifiers on `figure.fig`: none = prose width (700 px); `wide` = up to 1040 px; `full` = up to
1400 px (the sim). Add `card` for a white framed figure (use it for anything interactive). Side by side:
`<div class="fig-row">` around figures (stacks below ~660 px).

```html
<figure class="fig wide card" id="fig-readout">
  <div class="fig-title">Readout $\hat C$ against $\Sigma_g$ <span class="synthetic-tag">synthetic Σ</span></div>
  <div class="fig-sub">One short line on how to use it, e.g. "Hover a cell; the slider steps through training."</div>
  <div data-slot="readout"></div>
  <figcaption><b>Plotted:</b> … <b>Values:</b> …</figcaption>
</figure>
```

**The sim figure** (centrepiece pattern; built by `ui.simLayout`, section 7): view and transport on the left,
tours, controls, stat tiles and legend on the right (320 px), linked views underneath (`L.below`), all inside one
`figure.fig.full.card`. A run pauses whenever its figure is off screen, so put views that must follow the run
(heatmaps of $H\Sigma_g H^\top$, time series with a "now" cursor) inside the same figure, in `L.below`.

**Callouts.**
```html
<aside class="callout key"><p class="callout-title">Key point</p><p>The one thing to take away (max one per section).</p></aside>
<aside class="callout caveat"><p class="callout-title">Caveat</p><p>An honest limitation.</p></aside>
<aside class="callout"><p>A neutral note.</p></aside>
<aside class="callout found"><p class="callout-title">What we found</p>
  <ul><li><span class="verdict holds">holds</span>…</li>       <!-- verdict: holds | amended | fails | open -->
      <li><span class="verdict amended">amended</span>…</li></ul></aside>
```

**Side note** (margin note at ≥ 1240 px, numbered tap-to-open note below that; numbering is automatic):
```html
… keeps $L_{II}$ invertible<span class="sidenote">A neuron with every weight at the floor floats to the centroid.</span> and …
```

**Maths.** KaTeX auto-render: inline `$…$` or `\(…\)`, display `$$…$$` or `\[…\]`. A literal dollar is `\$`.
Numbered equation: `<div class="eq" id="eq-rule" data-num="1">$$…$$</div>`, cited as `<a href="#idea/eq-rule">(1)</a>`.
Notation (keep it uniform across chapters): $B$ boundary and $N$ interior neurons, boundary voltages $g$
(labels `g₁ g₂ …` in UI text), interior voltages $u = Hg$, $h_i$ the harmonic embedding (row $i$ of $H$),
$\Sigma_g$ the lifted input covariance, $\theta = -w_0/\eta$, $w_{\min}$, $\gamma$, $p$, $\hat C$ the readout.

**Maths and long derivations** go in a disclosure: `<details class="more"><summary>The maths: …</summary> … </details>`.

**Tables:** `<div class="table-scroll"><table class="data">…</table></div>`; numeric cells `class="num"` (right-aligned,
tabular figures).

**Inline keys:** `<span class="swatch" style="--sw: var(--c-cat-0)"></span>` (add class `round` for a dot).

**Deep links:** `<a class="explore-link" href="#homeo?theta=0.2&p=0.3">see this in the explorer</a>` (a pill with
an arrow; works within and across chapters). `class="explore-link big"` for the one primary call to action.

**Placeholders while drafting:** `<p class="placeholder"><span class="ph-tag">Placeholder</span> …</p>`.

## 5. Interaction conventions

- At most ~4 primary controls visible; everything else in the "More controls" disclosure (`ui.controls`).
- **Tours** (`ui.chips`) are named exact states with a one- or two-line definitional note. A tour is a URL:
  `#three?tour=<id>`; a tour id in the URL wins over individual parameters. When the reader then changes any
  control, call `chips.moved()` and drop `tour` from the URL (`ctx.setParams({..., tour: null})`).
- Every chart and the 3D view show a tooltip on hover (built in). Tooltips list quantities with their names; use
  `ui.tipRows`.
- Colour ranges are **fixed** across frames of one figure (pass `range`), so a change of colour is a change of value.
- Live sims start paused unless the figure is the chapter's centrepiece (`autoplay: true` starts it when it first
  scrolls into view). They pause off screen and when the tab is hidden (automatic).
- `prefers-reduced-motion` is respected for you: `transport` ignores `autoplay` and `view3d` ignores
  `autoRotate` under it (`ui.reducedMotion()` tells you, for anything of your own).

## 6. Colour grammar

One grammar for every chapter. Tokens live in `css/tokens.css` (CSS custom properties); `js/palette.js` exports
the same values as `P` (read from the CSS at import, so they cannot drift).

| role | CSS | JS | use |
|---|---|---|---|
| categorical (identity) | `--c-cat-0` … `--c-cat-7` | `P.cat[i]`, `catColor(i)` | clusters (ordered by size, largest = 0), series, boundary neurons when $B \le 8$. Fixed order, never cycled: past 8 everything folds to "other" |
| other / unassigned | `--c-other` | `P.other` | unassigned neurons, the folded tail, below-threshold |
| sequential `seq` | `--c-seq-0` … `-8` (blue) | `cmap("seq")` | generic magnitudes: counts, errors, phase diagrams |
| sequential `ink` | `--c-ink-0` … `-8` (paper → ink) | `cmap("ink")` | **weights / conductances only**; same metaphor as synapse opacity |
| diverging `div` | `--c-div-0` … `-8` (blue − / grey 0 / red +) | `cmap("div")` | covariance, similarity, signed loadings, eigenvectors. Always a symmetric range `[-m, m]` |
| text and surfaces | `--fg --ink-2 --mute --mute-2 --line --wash --bg --surface` | `P.fg …` | text never takes a data colour |
| accent | `--accent` | `P.accent` | caveats, the replay marker and the "now" cursor; never data |

**Neurons, everywhere:** an **interior neuron is a circle** (`P.interior` grey when not otherwise encoded), a
**boundary neuron is a square with an ink outline** labelled `g₁, g₂, …` (its categorical hue when $B \le 8$, ink
`P.boundary` when $B > 8$), a **synapse is an ink line whose opacity is $w_{ij}/\max w$**. Legends use the same
shapes (`shape: "dot" | "square" | "line"`).

**Barycentric colour** (small $B$): an interior neuron with embedding $h_i$ can be coloured `mixBary(h_i, bcols)`,
the $h$-weighted mix (in linear light) of the boundary hues, so a neuron at vertex $b$ has $b$'s colour and the
centroid is a muted grey. Use it when $B \le 4$; otherwise colour by cluster.

palette.js exports: `P`, `cmap(name) -> t => [r,g,b]`, `colorOf(name, x, lo, hi) -> css`, `lut(name, n=256) ->
Uint8ClampedArray(4n)`, `gradientCss(name, dir)`, `catColor(i)`, `boundaryColor(b, B)`, `mixBary(h, colors)`,
`hexToRgb`, `rgbToCss([r,g,b], a)`.

## 7. `js/ui.js`

Every control factory takes one options object and returns `{el, get(), set(v), …}`. Pass `parent` to append it, or
append `x.el` yourself. Labels, help text and option labels may contain `$TeX$`. Callbacks: `onInput(v)`
(continuous, sliders only) and `onChange(v)` (committed). `set(v)` never fires callbacks (a slider's
`set(v, {silent: false})` does).

**`slider({label, min, max, value, step?, log?, sig?, fmt?, unit?, help?, marks?, onInput?, onChange?, parent?})`**
Label left, live value right, track below. `log: true` makes the track logarithmic between `min > 0` and `max`
(values snap to `sig` significant digits, default 2). `marks: [{value, label}]` puts labelled ticks under the track
(e.g. where $\theta$ crosses a correlation). Double-clicking the value resets it. Extra: `.input`, `.setDisabled(b)`.
```js
const th = ui.slider({ label: "pruning threshold $\\theta$", min: -0.2, max: 0.6, step: 0.01, value: theta,
  fmt: x => ui.fmtFixed(x, 2), marks: [{ value: 0.1, label: "across" }, { value: 0.5, label: "within" }],
  onInput: x => { sim.setTheta(x); ctx.setParams({ theta: x, tour: null }); tours.moved(); } });
const wm = ui.slider({ label: "floor $w_{\\min}\\lambda/\\eta$", min: 1e-4, max: 0.1, log: true, value: 0.01, onChange: … });
```

**`segmented({label?, options, value, onChange, help?, stack?, parent?})`** options: `[{value, label, title?}]` or
plain values. Extra: `.setDisabled(value, bool)`.
```js
ui.segmented({ label: "mode", options: [{ value: "mf", label: "mean-field" }, { value: "mb", label: "minibatch" }], value: "mf", onChange: setMode });
```

**`select({label?, options, value, onChange, help?, stack?, parent?})`** same options; `get()` returns the original
(typed) value. Extra: `.select` (the element).

**`toggle({label, value, onChange, help?, parent?})`** a switch; `get()` → boolean.

**`chips({label?, items: [{id, label, note?, title?}], active?, onPick(item), parent?})`** preset / tour chips. Picking
marks the chip and shows its note under the row. Methods: `set(id)`, `moved()` (unmark; the note dims and offers
"back to this view", which calls `onPick` again), `clear()`, `get()`.
```js
const TOURS = [{ id: "collapse", label: "No pruning", note: "$\\theta \\le 0$: …", params: { theta: -0.1 } }, …];
const tours = ui.chips({ parent: L.side, label: "Tours", items: TOURS, active: ctx.param("tour", null),
  onPick: t => { applyParams(t.params); ctx.setParams({ ...t.params, tour: t.id }); } });
```

**`transport({onTick, onSeek, onReset?, onRestore?, speeds?, speed?, timeFmt?, autoplay?, maxHistory?, watch?, parent?})`**
Play/pause, step, reset, a scrubber over the recorded run, the time, and a speed control.
- `onTick(speed) -> frame | null` is called once per animation frame while playing **and on screen**. Advance the
  simulation `speed` steps (you decide what a step is), redraw, and return a frame `{t, …}` holding whatever you
  need to redraw that moment (e.g. `{t, H: Float32Array, W: Float32Array}`). Returned frames are recorded; past
  `maxHistory` (default 400) the history is thinned by halves, so the scrubber always spans the whole run.
  Return `{…, done: true, doneLabel?: "converged"}` to pause (e.g. at a fixed point).
- `onSeek(frame, i)` redraws a recorded frame (scrubbing). Don't advance the sim there.
- `onReset() -> frame` re-initialises and returns the first frame (the history restarts from it).
- `onRestore(frame)` (optional): pressing play while scrubbed back restores the sim to that frame and drops the
  history after it. Without it, play jumps back to the live end.
- `speeds`: numbers (`[1, 4, 16]`, shown as ×1 ×4 ×16) or `[{value, label}]`. `timeFmt(t)`: default `t = 1.23`.
- `autoplay: true` starts on first appearance. `watch`: the element whose visibility gates running (default the
  transport itself; pass the figure). Space toggles play while focus is in `watch`.
- Methods: `play() pause() toggle() step() reset() record(frame) seek(i) clearHistory() setStatus(text)
  setSpeed(v) destroy()`, getters `playing frames speed`.
```js
const tp = ui.transport({ parent: L.view, watch: fig, speeds: [1, 10, 100], autoplay: true,
  onTick: k => { sim.step(k); draw(sim); return { t: sim.t, H: sim.H.slice(), done: sim.converged() }; },
  onSeek: f => drawH(f.H),
  onRestore: f => sim.restore(f),
  onReset: () => { sim.init(params); draw(sim); return { t: 0, H: sim.H.slice() }; } });
```

**`controls({title?, primary: [comp|Node], advanced?: [comp|Node], advancedLabel?, open?, parent?})`** stacks
controls; `advanced` go in a closed "More controls" disclosure. Methods: `add(c)`, `addAdvanced(c)`.

**`stats({items: [{key, label, unit?, fmt?, help?}], cols? = 2, parent?})`** stat tiles (label, big value, unit,
optional sub-line). `set(key, value, subHtml?)`, `update({key: value, …})`. `fmt` defaults to `ui.fmt`.
```js
const st = ui.stats({ parent: L.side, items: [{ key: "k", label: "clusters", fmt: String }, { key: "bw", label: "Bures–Wasserstein", fmt: x => ui.fmtFixed(x, 3) }] });
st.update({ k: 3, bw: 0.214 }); st.set("k", 3, "of 4 planted");
```

**`legend({items: [{label, color, shape?: "dot"|"square"|"line"|"dash"|"band"|"ring", onHover?(on)}], title?, parent?})`**
`set(items)` replaces the keys.

**`colorbar({map: "seq"|"ink"|"div", range: [lo, hi], label?, ticks?: [values], fmt?, width?, parent?})`**
horizontal bar with tick values (default: ends, plus 0 for `div`). `set({range, label, ticks, map})`.

**`figure({title?, sub?, plotted?, values?, size?: ""|"wide"|"full", card? = true, id?, parent?})`** builds a
`figure.fig` from JS: `{el, body, caption, setCaption({plotted, values} | {html})}`. Use it for figures whose
caption depends on state (e.g. the current $B$).

**`callout({kind?: ""|"key"|"caveat"|"found", title?, html, parent?})`**.

**`simLayout({parent, side? = 320})`** → `{el, view, side, below}` (see section 4).

**Helpers:** `h(tag, attrs, ...children)` (attrs: `class`, `style` object incl. `--vars`, `text`, `html`,
`onclick`…; children may be Nodes, strings or components), `renderMath(el)`, `reducedMotion()`, `fmt(v, sig=3)` (true minus,
`×10ⁿ` outside [1e-3, 1e5), `n/a` for NaN), `fmtFixed(v, d=2)`, `fmtPct(v, d=0)`, `fmtInt(v)` (thousands
separators), `showTip(clientX, clientY, htmlOrNode)`, `hideTip()`,
`tipRows(head, [{name, value, color?, shape?: "dot"|"square"}], noteHtml?)`, `animate(el, fn(dt))` (rAF loop that
runs only while `el` is on screen; returns `{stop(), running}`), `parseHash()`, `tabsMeta`.

**Space badges and the synapse toggle** (section 9a): `spaceBadge(kind, text?)`, `panelHead(titleHtml, kind, text?)`,
`synapseToggle({checked = false, label = "show synapses", onChange(on), parent?})` → `{el, get(), set(on)}`.

## 8. `js/charts.js`

All charts: `const c = kind(el, opts); c.update(partialOpts); c.destroy()`. They fill the container's width, redraw
on resize and when a hidden tab becomes visible, are sharp on HiDPI screens, and show a tooltip on hover.
`update` merges into the previous options and redraws, so pass only what changed. Axis labels are plain text
(use Unicode: `θ`, `Σ`, `‖Δh‖₁`); colour-bar labels (`zlabel`) may contain `$TeX$`.

**`heatmap(el, {values, rows, cols, map?, range?, size?, square? = true, groups?, rowLabels?, colLabels?, xlabel?,
ylabel?, zlabel?, ticks?, colorbar? = true, tooltip?(i, j, v), onHover?(i, j, v), onClick?(i, j, v)})`**
Canvas, fast to ~1500 × 1500. `values`: row-major `Float32Array` (`rows*cols`) or an array of rows; NaN cells are
drawn in the wash colour. `map`: `"seq"` (default), `"ink"`, `"div"` or `t => [r,g,b]`. `range` is fixed when
given; without it, `[-m, m]` for `div` and `[min, max]` otherwise (pass a range for anything that animates).
`size`: max side in px (default 520). `groups: {rows: [{start, end, label?, color?}], cols?}` draws white
separators between groups, a coloured band on the left/bottom, and the group labels when cell labels do not fit
(`cols` defaults to `rows` for a square matrix). Cell labels show only when cells are ≥ 9 px (rows) / 11 px (cols).
`c.update(values)` or `c.update({values, range, groups})`; `c.range`.
```js
const hm = heatmap(L.below.firstChild, { map: "div", range: [-0.6, 0.6], size: 320, zlabel: "$h_i^\\top\\Sigma_g h_j$",
  xlabel: "neuron j (sorted by cluster)", ylabel: "neuron i" });
hm.update({ values: M, rows: N, cols: N, groups: { rows: clusters.map((c, k) => ({ start: c.start, end: c.end, color: catColor(k), label: `c${k + 1}` })) } });
```

**`filmstrip(el, {frames, frameLabels?, sliderLabel?, range?, thumbs? = true, fps? = 4, value?, onFrame?(i), …heatmap opts})`**
One heatmap stepped through snapshots with a slider and a play button, plus a row of thumbnails (click to jump).
The colour range is fixed across frames (computed over all frames when not given). Returns `{heatmap, go(i), index}`.
```js
filmstrip(ctx.slot("film"), { frames: sims, rows: 64, cols: 64, map: "div", range: [-1, 1], size: 300,
  frameLabels: steps.map(s => `step ${ui.fmtInt(s)}`), sliderLabel: "training step", zlabel: "$H\\Sigma_g H^\\top$" });
```

**`lineChart(el, {series: [{name, y, x?, color?, dash?, width?, band?: [lo[], hi[]], dots?, points?, r?}], x?, xlog?,
ylog?, xrange?, yrange?, xlabel?, ylabel?, height? = 220, xfmt?, yfmt?, refs?: [{x|y, label?}], cursor?, legend?, xname?})`**
SVG. Shared `x` or per-series `x`. Colours default to the categorical order. `band` shades a series' interval.
`points: true` draws a series as dots only (a scatter over a curve); `dots: true` adds dots to a line. `refs` are
dashed reference lines. Hover: crosshair and every series' value at the nearest x. Legend (with hover-to-isolate)
when there is more than one series. `c.setCursor(x)` moves an accent "now" line without a redraw (link it to the
transport time); `c.update({series})` for new data (throttle to every few ticks in a live sim).
```js
const lc = lineChart(box, { series: [{ name: "readout BW", y: [], x: [] }], ylog: true, xlabel: "time t", ylabel: "BW distance",
  refs: [{ y: baseline, label: "identity baseline" }] });
lc.update({ series: [{ name: "readout BW", x: ts, y: bw }] }); lc.setCursor(t);
```

**`scatter(el, {x, y, color?: css|css[], category?: int[], value?: number[], map?, range?, size?: px|px[], shape?:
("dot"|"square")[], anchors?: [{x, y, label, color}], xrange?, yrange?, equal?, xlabel?, ylabel?, height? = 320,
tooltip?(i), onHover?(i|null), onSelect?(i|null)})`**
Canvas. Colour by explicit CSS colours, by `category` (categorical order; `-1` = other) or by `value` through a
map. `equal: true` keeps one data unit the same length on both axes (use for embeddings). `anchors` are boundary
neurons (labelled squares). `c.highlight(i)` rings a point.

**`barChart(el, {labels, values?, series?: [{name, values, color?}], colors?, color?, name?, overlay?: {values, name?,
color?}, yrange?, xlabel?, ylabel?, height? = 180, yfmt?, tooltip?(i, s), labelEvery?, legend?})`**
SVG. One series (`values`, optional per-bar `colors`) or grouped `series`. Signed data get a zero line. `overlay`
draws a second quantity over the same categories as ink dots joined by a thin line (e.g. an eigenvector against
a cluster part).

**`barMultiples(el, {labels, panels: [{title, values, color?, sub?, overlay?}], yrange, valueName?, overlayName?,
height? = 96, panelWidth? = 180, labelEvery?})`**
Small multiples of bars over the same categories (one panel per cluster, bars over boundary neurons). Always pass
a shared `yrange`.

**`imageGrid(el, {arrays?, w?, h?, map?, range?, sprite?, indices?, labels?, colors?, tile? = 44, rows?, tooltip?(i),
onHover?(i|null), onClick?(i)})`**
Tiles from `Float32Array`s (`w*h`, row-major, through a colour map; `map` default `div`; `range` default one shared
range over all tiles, symmetric for `div`; `"tile"` gives each tile its own) or from a sprite PNG
(`sprite: {src, tileW, tileH, cols}` with `indices`). `labels` (small corner text) and `colors` (a swatch line under
each tile) are per tile. `rows: [{label, count}]` lays tiles out as labelled rows (`count` tiles per row, in order,
fixed size `tile`), e.g. one row per cluster or "linear" / "rectified". Returns `{ready (Promise), select(i), range}`.
```js
imageGrid(ctx.slot("parts"), { arrays: parts, w: 14, h: 14, map: "div", tile: 48,
  rows: [{ label: "linear", count: 8 }, { label: "rectified", count: 8 }], colors: parts.map((_, i) => catColor(i % 8)) });
imageGrid(ctx.slot("top"), { sprite: { src: "data/<run>/tiles.png", tileW: 28, tileH: 28, cols: 100 }, indices: top, labels: top.map(i => String(lab[i])) });
```

Low-level exports for chart-like figures of your own: `svgEl(tag, attrs, parent)`, `scale(kind, d0, d1, r0, r1)`
(`kind` "linear" | "log"; has `.inv`), `niceTicks(lo, hi, n)`, `logTicks(lo, hi)`, `hidpi(canvas, w, h) -> ctx2d`.
Reuse the CSS classes `.chart`, `.chart-title`, `.chart-legend`, and `ui.legend` / `ui.colorbar` so they match.

## 9. `js/view3d.js`

**Approach: hand-written on Canvas 2D, no three.js.** Reasons: the scene is simple (≤ ~300 points, ≤ ~5,000
segments, a few labelled anchors), and Canvas 2D gives antialiased hairlines and crisp text labels on the light
theme for free, HiDPI is one transform, and there is no 600 kB dependency. Synapses are bucketed by
(colour, opacity level) into a few dozen strokes per frame, points are depth-sorted and drawn far to near with a
surface ring and a mild depth fade. It draws only when something changed or while auto-rotating, and only while
on screen. Measured in headless Chrome **without a GPU** (software raster, worst case): the style guide's sim
(150 points, ~3,800 short synapses, live dynamics, linked heatmap) renders at 60 fps; a stress test of 300 points
and 5,000 long random edges spanning the view runs at ~28 fps (an empty canvas caps at ~55 fps there). With GPU
canvas (any laptop) this is well within a 60 fps budget. If a chapter must draw far more, thin edges by weight
first (drawing more than ~5,000 synapses is unreadable anyway).

**`view3d(el, opts)`** options: `height` (px, default 460; shrinks on phones to ~0.92 × width, min 280),
`yaw` (−0.5), `pitch` (0.95 rad, 0 = side view, π/2 = top view), `zoom` (1), `autoRotate` (false), `rotateSpeed`
(0.18 rad/s), `center` / `radius` (fixed framing; default: fitted once to the first points + anchors, then held so
moving points don't make the camera jitter; `v.fit()` refits), `hull` (true: grey lines between all anchors, i.e.
the simplex edges), `pointSize` (4 px), `edgeWidth` (1), `edgeAlpha` (0.6, the opacity of a weight-1 edge),
`edgeColor` (ink), `trailColor` (ink), `depthFade` (0.35), `hint` (true: "drag to orbit · scroll to zoom"),
`tooltip(index, kind)` → HTML or Node, `kind` = `"point"` | `"anchor"`. `badge`: `"input"` | `"function"` | `{kind, text}` draws a space badge top left (section 9a).

Methods:

| method | |
|---|---|
| `setPoints(xyz, colors?, sizes?)` | `xyz` Float32Array(3N). `colors`: one CSS colour or an array of N. `sizes`: px radius, number or array of N. Call every frame when positions change (cheap). With trails on, each call pushes the previous positions onto the trail |
| `setColors(colors)` | recolour without moving |
| `setEdges(pairs, alpha, {group?, palette?})` | `pairs` Uint32Array(2M) of point indices; **an index ≥ N means anchor (index − N)**, so interior–boundary synapses are `[i, N + b]`. `alpha` Float32Array(M) in [0, 1] (use $w_{ij}/\max w$; entries ≤ 0.004 are skipped). Optional per-edge colour: `group` Uint8Array(M) indexing `palette` (CSS colours) |
| `setAnchors([{xyz, label, color, size?}])` | boundary neurons: squares with ink outline, label pushed away from the centre |
| `setGuides([{a, b, label?, dash?, color?}])` | reference segments in scene coordinates (e.g. a height axis) |
| `setTrails(n)` / `clearTrails()` | keep the last `n` positions per point as fading ink trails (0 = off); clear on reset |
| `highlight(i \| [i…] \| null)` | ring points |
| `onHover(cb)` | `cb(pointIndex \| null, {kind, index} \| null)` |
| `onCamera(cb)` | `cb({yaw, pitch, zoom})` after the reader orbits/zooms (persist to the URL if you like) |
| `getCamera()`, `setCamera({yaw?, pitch?, zoom?})`, `setAutoRotate(on)`, `fit()`, `render()`, `setCorner(html)` (a small overlay label, top left), `destroy()` | |

Interaction: drag to orbit, wheel / pinch / `+` `−` keys to zoom, arrow keys to orbit when focused; buttons for
auto-rotate and reset view. On touch, vertical swipes scroll the page (horizontal drags orbit).

Helpers: `simplexVertices(B)` → `[[x,y,z]…]` for B = 2, 3 (equilateral triangle in z = 0, circumradius 1) or 4
(regular tetrahedron); throws for larger B (project instead, and say which projection in the caption).
`baryToXYZ(H, B, verts?, out?, lift?)` → Float32Array(3N) from row-major `H` (N × B); `lift` (Float32Array N) is
added to z, e.g. a neuron's voltage variance to give the flat B = 3 triangle a third dimension.

```js
const B = 3, verts = simplexVertices(B), bcol = [0, 1, 2].map(b => boundaryColor(b, B));
const v = view3d(L.view, { height: 480, tooltip: (i, kind) => kind === "point" ? ui.tipRows(`neuron ${i}`, …) : null });
v.setAnchors(verts.map((xyz, b) => ({ xyz, label: `g${"₁₂₃"[b]}`, color: bcol[b] })));
function draw(sim) {
  v.setPoints(baryToXYZ(sim.H, B, verts), rowsOf(sim.H).map(h => mixBary(h, bcol)));
  v.setEdges(sim.edgePairs, sim.edgeWeights.map(w => w / wmax));
}
```

## 9a. Neuron space and input space (the graphics convention)

**Every graphic is either neuron space or input / function space, and says so with a badge.**

- **Neuron space**: the network itself. Nodes are neurons (circles) and inputs (squares, ink outline), edges are
  synapses with width and opacity rising with $w/\max w$. Nodes sit at **seeded random positions** (uniform in the
  panel with a minimum spacing), so the physical layout is visibly unrelated to what a neuron computes. Never place
  neurons by their embedding or in a tidy hand-made layout. Draw it with `netview.js` (canvas) or, inside an SVG
  figure, with positions from `randomLayout` (same algorithm). Pies or other glyphs as nodes are fine.
- **Input space**: neurons at their embedding: rows $h_i$ of $H$ in the simplex for $B = 3$; a stated projection
  (classical MDS, PCA, a coordinate pair) for $B > 3$. Organised layouts stay. **Synapse lines are not drawn by
  default**; cluster membership is colour. Where the lines carry meaning, offer `ui.synapseToggle` ("show
  synapses", off by default).
- **Response-function space**: the same for nonlinear neurons (MDS of measured response distances $D$, PCA of
  responses); badge "response-function space". A completed-distance MDS is labelled an approximation.
- Plain charts (against $b$, $t$, $N$), tables and matrix heatmaps carry no badge.
- **Both views**: when the point is how the two relate (clusters form in input space while the network stays a
  tangle in neuron space), put them side by side in a `div.space-pair` (two columns, stacks on phones) with linked
  highlighting (hover a neuron in one, ring it in the other: `netview.setHighlight(i)`, `view3d.highlight(i)`) and
  the same cluster colours. Examples: `#perron-sim`, the film player.

**Badges.** Colours are tokens in `css/tokens.css` (`--space-neuron-*`, `--space-input-*`, `--space-function-*`;
chrome, never data colours), styled once in `css/ui.css` (`.space-badge`, with a dot, triangle or wave glyph).
In HTML: `<span class="space-badge" data-space="neuron">neuron space</span>` (`data-space` = `neuron` | `input` |
`function`), e.g. inside a `.fig-title` or a panel title. In JS: `ui.spaceBadge(kind, text?)`,
`ui.panelHead(titleHtml, kind, text?)` (a chart title with its badge), `view3d(el, {badge: "input"})`, and netview
draws its own (top left; `badge: false` turns it off).

## 9b. `js/netview.js`

**`netview(el, opts)`** appends a neuron-space network (canvas, HiDPI, redraws on resize and when its tab is shown).
Options: `N` neurons, `B = 0` inputs, `seed = 1` (layout seed: same `N`, `B`, `seed` → same picture; inputs take the
first `B` slots, so their places do not depend on `N`), `aspect = 1.5` (layout box; the panel is drawn at
`width / aspect` between `minHeight = 220` and `maxHeight = 460`, or at a fixed `height`), `radius` (px; default
from the spacing), `colors` (CSS or array of N; default `P.interior`), `inputColors` (default `boundaryColor(b, B)`),
`inputLabels` (default `g₁ g₂ …`), `labels` (text inside neurons when they are big enough), `edgeWidth = 3.2`
(px at $w = w_{\max}$), `denseAt = 150` (above this many edges, lines thin and fade so clusters stay legible),
`badge` (`"neuron space"`; `false` for none), `corner` (HTML note top right), `tooltip(i, kind)` → HTML | Node,
`onHover(i | null, kind)`, `onSelect(i | null, kind)` (`kind` = `"neuron"` | `"input"`), `glyph(ctx2d, i, x, y, r)`
→ `true` if it drew neuron `i` itself (pies).

| method | |
|---|---|
| `setColors(c)`, `setInputColors(c)` | one CSS colour or an array |
| `setSizes(m)` | radius multipliers (number or array of N) |
| `setLabels(l)`, `setInputLabels(l)` | |
| `setEdges(pairs, w, {wmax?})` | `pairs` flat `[i, j, …]` (an index ≥ N is input index − N, as in `view3d.setEdges`); `w` conductances, any scale, ≤ 0 skipped; width and opacity ∝ `w / wmax` (default `max w`; pass `wmax` to hold a scale across frames). Keep it to ≲ 1,500 edges (the strongest) |
| `setHighlight(i \| [i…] \| null, kind = "neuron")` | ring and darken its synapses; does **not** call `onHover`/`onSelect`, so link panels with it |
| `setSelected(i \| null, kind)` | a persistent dashed ring |
| `positions()` | `{neurons, inputs}` in [0, 1]² |
| `setCorner(html)`, `render()`, `destroy()`, `.el`, `.canvas` | |

`randomLayout(n, {seed = 1, aspect = 1.5, spacing = 0.62})` → `Float32Array(2n)` in [0, 1]²: dart throwing, each
point uniform and at least `spacing · sqrt(area / n)` from the earlier ones. Use it for SVG figures.

```js
import { netview } from "./netview.js";
const pair = ui.h("div", { class: "space-pair" }, leftBox, rightBox); L.view.append(pair);
const v = view3d(leftBox, { height: 440, badge: "input", tooltip: tipFor });
const nv = netview(rightBox, { N, B: 3, seed: 11, tooltip: (i, kind) => tipFor(i, kind === "input" ? "anchor" : "point"),
  onHover: (i, kind) => v.highlight(kind === "neuron" ? i : null) });
v.onHover(i => nv.setHighlight(i));
function draw(f) { nv.setColors(colors); nv.setEdges(pairs, weights); }        // same colours as the 3D view
```

## 9c. `js/filmplayer.js`

Replays a precomputed film from `data/big/` (format: `docs/PLAN_curation.md` "Film data format" and
`data/big/README.md`): lazy-loads `<id>.json` + `<id>.bin`, play / pause / scrub on the frame axis (frames are
spaced in log time; the label shows $t$), and draws the state twice with linked highlighting: input space (space
`"simplex"`: rows of $H$ in the triangle) or response-function space (`"pca"`: first two principal axes, fixed by
the final frame, with a "3-D (third axis)" toggle when the third axis carries ≥ 1% of the variance), and neuron
space (netview, layout seeded by `network_layout_seed`, the file's display edges plus the strongest input
synapses). Colour: cluster at the end of the run (`final.labels`, re-ranked by size). A readout line gives frame,
$t$, residual, live edges and components.

**`filmPlayer(el, {base = "data/big/", id?, filter?(indexEntry) → bool, picker = true, fps = 6, inputEdges = 600,
height = 420, onFrame?(f, film), colors?(film, clusterColors) → css[], countLabels = false})`** → `{load(id) → Promise, go(f), play(), pause(), frame, film, destroy()}`.
`colors` overrides the per-neuron colours (the perron chapter uses the barycentric mix for $B = 3$); `countLabels` labels
neurons that coincide on screen (a collapsed cluster) with their number. Linear runs (`nonlin = "linear"`) are badged
"input space" also when drawn on principal axes; nonlinear runs "response-function space". Without `id` it shows the first film of the index that passes `filter`; the picker (a select) offers the rest.
Also exported: `loadIndex(base)`, `loadFilm(id, base)` → film with `coord(f)`, `H(f)`, `edges(f)`, `inW(f)`
decoders, for chapter-specific figures over the same files.

```js
import { filmPlayer } from "./filmplayer.js";
filmPlayer(ctx.slot("film"), { filter: e => e.family === "A" && e.rule === "cov", id: "A_cov_pair_th0.5" });
```

## 10. Performance and lifecycle rules

- Never run a loop that is not gated on visibility. `ui.transport` and `view3d` handle this; for anything else use
  `ui.animate(el, fn)`.
- In `onTick`, update the 3D view every frame but throttle heavier linked views (an N × N heatmap, SVG line charts)
  to every 4–10 ticks; on `onSeek` update everything.
- Copy arrays into frames (`H.slice()`); the engine may reuse its buffers.
- Fetch precomputed data lazily inside `init` (it only runs when the tab is opened) and show a short "loading…"
  line in the slot meanwhile.
- `init` may be `async`; a thrown error is caught by the shell and shown in the chapter (and logged), so the rest
  of the page keeps working, but the page must have zero console errors when done.

## 11. Before handing in

1. Serve (`python3 -m http.server <port> --bind 127.0.0.1 -d site`), open your tab, interact with every control.
2. Zero console errors: `playwright-cli -s=<name> open http://127.0.0.1:<port>/#<tab> --browser=chrome`, then
   `playwright-cli -s=<name> console`.
3. Screenshots at 1440 × 900 and 390 × 844 (`resize`, then `screenshot --filename=/tmp/<tab>-d.png --full-page`);
   look at them for overlaps, clipped labels, raw TeX, and spacing. Close your session.
4. Every figure has a definitional caption; every synthetic figure says so; every live sim states its mode.
