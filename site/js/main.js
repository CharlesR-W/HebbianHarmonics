// Page shell: hash router, lazy chapter loading, maths, side notes, heading anchors, per-chapter ctx.
// Hash grammar:  #<tab>            a chapter
//                #<tab>/<id>       scroll to the element with that id inside the chapter
//                #<tab>?k=v&...    chapter state (read/write it through ctx, never directly)
//                #<id>             any id: finds the chapter holding it (after that chapter has loaded)
import { parseHash, renderMath, tabsMeta, tabAliases, HOME_TAB } from "./ui.js";

const TABS = tabsMeta.map(t => t.id);
// Anchors that moved or were removed: old deep links (#old/old-sim?tour=…, #talk/talk-hierarchy) go to the surviving
// place, always chapter-qualified so a cold load can lazy-load the chapter. [tab, anchor | null (chapter top)].
const anchorMoves = {
  old: { "old-watch": ["perron", "perron-sim"], "old-sim": ["perron", "perron-sim"], "old-collapse": ["perron", "perron-dynamics"],
    "old-parts": ["perron", "perron-parts"], "old-eq-part": ["perron", "perron-eq-part"] },
  perron: { "perron-cases": ["perron", "perron-parts"], "perron-collapse": ["perron", "perron-dynamics"], "perron-big": ["perron", "perron-which"],
    "perron-film3": ["perron", "perron-parts"], "perron-many": ["perron", "perron-which"], "perron-filmB": ["perron", "perron-which"] },
  talk: { "talk-hierarchy": ["primer", "primer-perron"], "talk-expansion": ["primer", "primer-nonlinear"],
    "talk-allocation": ["allocation", null] },
};
// Old addresses -> new: #talk… -> #primer… (ids inside were renamed talk-* -> primer-*); anchors that moved chapter
// (also as bare #id); #old?k=v (the old chapter's simulation state) -> #perron?k=v; a bare #<tab>-… id ->
// #<tab>/<tab>-… (so it loads that chapter). Rewrites the URL in place.
const bareMoves = Object.fromEntries(Object.values(anchorMoves).flatMap(m => Object.entries(m)));
const at = (t2, a2, qs) => `#${t2}${a2 ? "/" + a2 : ""}${qs}`;
function redirect() {
  const raw = location.hash.replace(/^#/, ""); if (!raw) return;
  const { tab, anchor, params } = parseHash();
  const qi = raw.indexOf("?"), qs = qi < 0 ? "" : raw.slice(qi);
  let to = null;
  if (anchor && anchorMoves[tab]?.[anchor]) to = at(...anchorMoves[tab][anchor], qs);
  else if (tabAliases[tab]) { const t2 = tabAliases[tab]; to = at(t2, anchor && anchor.replace(new RegExp(`^${tab}-`), `${t2}-`), qs); }
  else if (!anchor && tab === "old" && [...params.keys()].length) to = `#perron${qs}`;
  else if (!TABS.includes(tab) && bareMoves[tab]) to = at(...bareMoves[tab], qs);
  else if (!TABS.includes(tab)) {
    const m = /^([a-z]+)-(.*)$/.exec(tab);
    if (m && tabAliases[m[1]]) { const t2 = tabAliases[m[1]]; to = at(t2, `${t2}-${m[2]}`, qs); }
    else if (m && TABS.includes(m[1])) to = at(m[1], tab, qs);
  }
  if (to && to !== location.hash) history.replaceState(history.state, "", to);
}

const chapters = {};          // tab -> { loaded: Promise, ctx }
let current = null;

const panelOf = tab => document.getElementById(`tab-${tab}`);

function decorate(root, tab) {
  // heading anchors
  root.querySelectorAll("h2[id], h3[id]").forEach(hd => {
    if (hd.querySelector(":scope > .anchor")) return;
    const a = document.createElement("a"); a.className = "anchor"; a.href = `#${tab}/${hd.id}`; a.textContent = "#"; a.setAttribute("aria-label", "Link to this section");
    hd.prepend(a);
  });
  // side notes: numbered per chapter; margin notes on wide screens, tap-to-open below 1240 px
  let n = root.querySelectorAll(".sn-ref").length;
  root.querySelectorAll(".sidenote:not([data-sn])").forEach(sn => {
    n += 1; sn.dataset.sn = n;
    const b = document.createElement("button"); b.type = "button"; b.className = "sn-ref"; b.textContent = n; b.setAttribute("aria-label", `Side note ${n}`);
    const num = document.createElement("span"); num.className = "sn-num"; num.textContent = n; sn.prepend(num);
    sn.before(b);
    b.addEventListener("click", () => sn.classList.toggle("open"));
  });
  root.querySelectorAll("img:not([loading])").forEach(img => img.loading = "lazy");
  renderMath(root);
}

function makeCtx(tab, panel) {
  const hooks = { show: [], hide: [], params: [] };
  const ctx = {
    tab, panel, article: panel.querySelector("article.prose"),
    // current chapter state from the URL (a fresh URLSearchParams each call)
    get params() { const h = parseHash(); return h.tab === tab ? h.params : new URLSearchParams(); },
    // typed read with a fallback: ctx.param("theta", 0.3) -> number; ctx.param("preset", "blocks") -> string
    param(name, fallback) {
      const v = ctx.params.get(name);
      if (v == null || v === "") return fallback;
      if (typeof fallback === "number") { const x = Number(v); return Number.isFinite(x) ? x : fallback; }
      if (typeof fallback === "boolean") return v === "1" || v === "true";
      return v;
    },
    // merge into the URL state; null/undefined deletes a key. Uses replaceState (no history entry, no hashchange)
    // unless {push: true}. Numbers are written with up to 4 significant digits, booleans as 1/0.
    // Ignored while another chapter is shown (an async callback must not rewrite the URL to a hidden chapter).
    setParams(obj, { push = false } = {}) {
      if (current !== tab) return;
      const p = ctx.params;
      for (const [k, v] of Object.entries(obj)) {
        if (v == null) p.delete(k);
        else p.set(k, typeof v === "number" ? String(+v.toPrecision(4)) : typeof v === "boolean" ? (v ? "1" : "0") : String(v));
      }
      const qs = p.toString();
      const url = `#${tab}${qs ? "?" + qs : ""}`;
      if (url === location.hash) return;
      if (push) history.pushState(null, "", url); else history.replaceState(history.state, "", url);
      ctx._last = location.hash;
    },
    onShow(fn) { hooks.show.push(fn); if (current === tab) fn(); },
    onHide(fn) { hooks.hide.push(fn); },
    // fn(params) when the URL state changes from outside (a deep link clicked, back/forward) while this tab is shown
    onParams(fn) { hooks.params.push(fn); },
    get visible() { return current === tab && document.visibilityState === "visible"; },
    renderMath,
    slot(name) { return panel.querySelector(`[data-slot="${name}"]`); },
    _hooks: hooks, _last: null,
  };
  return ctx;
}

async function loadChapter(tab) {
  if (chapters[tab]) return chapters[tab].loaded;
  const panel = panelOf(tab), article = panel.querySelector("article.prose");
  const ctx = makeCtx(tab, panel);
  const rec = chapters[tab] = { ctx };
  rec.loaded = (async () => {
    try {
      const r = await fetch(`content/${tab}.html`, { cache: "no-cache" });
      if (!r.ok) throw new Error(`content/${tab}.html: HTTP ${r.status}`);
      article.innerHTML = await r.text();
    } catch (e) {
      article.innerHTML = `<p class="placeholder">This chapter's text has not been written yet (${e.message}).</p>`;
    }
    decorate(article, tab);
    try {
      const mod = await import(`./ch_${tab}.js`);
      if (typeof mod.init === "function") await mod.init(panel, ctx);
      decorate(article, tab);          // the chapter may have added headings, maths or side notes
    } catch (e) {
      console.error(`chapter ${tab}:`, e);
      const box = document.createElement("div"); box.className = "placeholder panel-error";
      box.textContent = `The interactive part of this chapter failed to load: ${e.message}`;
      article.prepend(box);
    }
  })();
  return rec.loaded;
}

async function route() {
  redirect();
  let { tab, anchor, params } = parseHash();
  if (!TABS.includes(tab)) {
    // a bare #id: look for it in the loaded chapters; otherwise the first chapter
    const el = tab && document.getElementById(tab), p = el && el.closest(".tab-panel");
    anchor = p ? tab : null; tab = p ? p.dataset.tab : HOME_TAB;
  }
  const changed = tab !== current, prev = current;
  if (changed && prev && chapters[prev]) chapters[prev].ctx._hooks.hide.forEach(f => f());
  current = tab;
  document.body.dataset.tab = tab;
  document.querySelectorAll(".tab-panel").forEach(p => p.hidden = p.dataset.tab !== tab);
  document.querySelectorAll(".tablist a").forEach(a => a.setAttribute("aria-selected", String(a.dataset.tab === tab)));
  document.title = tabsMeta.find(t => t.id === tab).title;
  const firstLoad = !chapters[tab];
  await loadChapter(tab);
  if (current !== tab) return;                       // the reader moved on while it loaded
  const ctx = chapters[tab].ctx;
  // returning to a chapter through its bare tab link: put its last state back in the URL
  if (!firstLoad && !anchor && ![...params.keys()].length && ctx._last && ctx._last !== location.hash) {
    history.replaceState(history.state, "", ctx._last);
  } else if (!firstLoad && location.hash !== ctx._last && [...params.keys()].length) {
    ctx._hooks.params.forEach(f => f(params));           // new state from outside (deep link, back/forward)
  }
  if ([...ctx.params.keys()].length || !ctx._last) ctx._last = location.hash;
  if (changed) ctx._hooks.show.forEach(f => f());
  const target = anchor && panelOf(tab).querySelector(`#${CSS.escape(anchor)}`);
  if (target) requestAnimationFrame(() => target.scrollIntoView({ block: "start" }));
  else if (changed) window.scrollTo(0, 0);
  // keep the selected tab visible in the scrolling tab strip on phones
  document.querySelector(`.tablist a[data-tab="${tab}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest" });
}

window.addEventListener("hashchange", route);
// a link to the hash we are already on (e.g. a tour link clicked twice) still re-applies its state
document.addEventListener("click", e => {
  const a = e.target.closest("a[href^='#']");
  if (a && a.getAttribute("href") === location.hash && a.getAttribute("href").includes("?")) {
    e.preventDefault(); const c = chapters[current]; if (c) c.ctx._last = null; route(); }
});
document.addEventListener("visibilitychange", () => {
  const c = current && chapters[current];
  if (!c) return;
  (document.visibilityState === "visible" ? c.ctx._hooks.show : c.ctx._hooks.hide).forEach(f => f());
});

function start() {
  if (window.renderMathInElement) route(); else window.addEventListener("load", route, { once: true });
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
