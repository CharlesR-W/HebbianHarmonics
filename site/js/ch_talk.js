// The talk: a slide-like overview. Three parts (talk_a/b/c), each an HTML fragment of <section class="slide">
// plus a module exporting init(root, ctx). This file loads them and adds keyboard stepping.
const PARTS = ["a", "b", "c"];

export async function init(panel, ctx) {
  await Promise.all(PARTS.map(async part => {
    const root = panel.querySelector(`.talk-part[data-part="${part}"]`);
    try {
      const r = await fetch(`content/talk_${part}.html`, { cache: "no-cache" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      root.innerHTML = await r.text();
      ctx.renderMath(root);
      const mod = await import(`./talk_${part}.js`);
      if (typeof mod.init === "function") await mod.init(root, ctx);
    } catch (e) {
      console.error(`talk part ${part}:`, e);
      root.insertAdjacentHTML("beforeend", `<p class="placeholder">Part ${part} failed to load: ${e.message}</p>`);
    }
  }));
  const slides = () => [...panel.querySelectorAll(".slide")];
  const count = panel.querySelector(".talk-count");
  slides().forEach((s, i) => { if (!s.querySelector(".slide-num")) { const n = document.createElement("div"); n.className = "slide-num"; n.textContent = String(i + 1).padStart(2, "0"); s.prepend(n); } });
  const currentIdx = () => {
    const mid = innerHeight * 0.4; let best = 0, bd = Infinity;
    slides().forEach((s, i) => { const r = s.getBoundingClientRect(); const d = Math.abs(r.top - 60); if (r.top < mid && d < bd || bd === Infinity) { bd = d; best = i; } });
    return best;
  };
  const upd = () => { if (ctx.visible) count.textContent = `${currentIdx() + 1} / ${slides().length}`; };
  const go = d => { const all = slides(); const i = Math.max(0, Math.min(all.length - 1, currentIdx() + d)); all[i].scrollIntoView({ behavior: "smooth", block: "start" }); };
  addEventListener("scroll", upd, { passive: true });
  addEventListener("keydown", e => {
    if (!ctx.visible || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target.closest("input, select, textarea, button, [contenteditable]")) return;
    if (["ArrowRight", "ArrowDown", "PageDown", "j"].includes(e.key)) { e.preventDefault(); go(1); }
    else if (["ArrowLeft", "ArrowUp", "PageUp", "k"].includes(e.key)) { e.preventDefault(); go(-1); }
  });
  ctx.onShow(upd); upd();
}
