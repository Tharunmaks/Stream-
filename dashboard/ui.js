"use strict";
// Shared helpers of the ExpertCloud pages: DOM builder (text is always inserted as text, never as HTML), icons, number formats,
// toasts, copy buttons, storage that tolerates blocked storage, and modals.
const $ = (s, r = document) => r.querySelector(s);
function h(tag, attrs, ...kids) {   // DOM builder; text always goes in as text nodes, never as HTML
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "style") e.style.cssText = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}
const ICONS = {
  home: '<path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/>',
  models: '<path d="M12 2l9 5v10l-9 5-9-5V7z"/><path d="M12 22V12M21 7l-9 5-9-5"/>',
  compute: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
  docs: '<path d="M4 4h10a4 4 0 0 1 4 4v12H8a4 4 0 0 1-4-4z"/><path d="M8 8h6M8 12h6"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
  db: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  card: '<rect x="2" y="5" width="20" height="14" rx="2"/><path d="M2 10h20M6 15h4"/>',
  stream: '<path d="M4 6h16M4 12h10M4 18h13"/><circle cx="19" cy="12" r="2"/>',
  grid: '<circle cx="5" cy="5" r="1.5"/><circle cx="12" cy="5" r="1.5"/><circle cx="19" cy="5" r="1.5"/><circle cx="5" cy="12" r="1.5"/><circle cx="12" cy="12" r="1.5"/><circle cx="19" cy="12" r="1.5"/><circle cx="5" cy="19" r="1.5"/><circle cx="12" cy="19" r="1.5"/><circle cx="19" cy="19" r="1.5"/>',
  bolt: '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
  overview: '<path d="M3 13h8V3H3zM13 21h8V11h-8zM3 21h8v-6H3zM13 3v6h8V3z"/>',
  playground: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
  nodes: '<rect x="3" y="4" width="18" height="7" rx="2"/><rect x="3" y="13" width="18" height="7" rx="2"/><path d="M7 7.5h.01M7 16.5h.01"/>',
  keys: '<circle cx="8" cy="15" r="4"/><path d="M10.8 12.2L20 3M16 7l3 3M14 9l2 2"/>',
  connect: '<path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/>',
  copy: '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/>',
  check: '<path d="M5 12l5 5L20 7"/>', cloud: '<path d="M17.5 19a4.5 4.5 0 1 0-1.4-8.8A6 6 0 1 0 6 16.9"/><path d="M12 13v8M9 18l3 3 3-3"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>', send: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  spark: '<path d="M12 3l1.9 5.8L20 10l-6.1 1.2L12 17l-1.9-5.8L4 10l6.1-1.2z"/>',
};
ICONS.users = ICONS.user; ICONS.billing = ICONS.card; ICONS.catalog = ICONS.models;
function icon(name) { const s = document.createElementNS("http://www.w3.org/2000/svg", "svg"); s.setAttribute("viewBox", "0 0 24 24"); s.setAttribute("class", "i"); s.innerHTML = ICONS[name]; return s; }
const fmt = n => n == null || isNaN(n) ? "—" : Math.abs(n) >= 1e9 ? (n / 1e9).toFixed(1) + "B" : Math.abs(n) >= 1e6 ? (n / 1e6).toFixed(1) + "M" : Math.abs(n) >= 1e4 ? (n / 1e3).toFixed(1) + "k" : Math.round(n).toLocaleString();
const mb = v => v == null || isNaN(v) ? "—" : v >= 1024 ? (v / 1024).toFixed(v >= 10240 ? 0 : 1) + " GB" : Math.round(v) + " MB";
function ago(t) { const s = Math.max(0, Date.now() / 1000 - t); return s < 60 ? "just now" : s < 3600 ? Math.floor(s / 60) + " min ago" : s < 86400 ? Math.floor(s / 3600) + " h ago" : Math.floor(s / 86400) + " d ago"; }
function dur(s) { s = Math.round(s || 0); return s < 60 ? s + "s" : s < 3600 ? Math.floor(s / 60) + "m" : s < 86400 ? Math.floor(s / 3600) + "h " + Math.floor(s % 3600 / 60) + "m" : Math.floor(s / 86400) + "d " + Math.floor(s % 86400 / 3600) + "h"; }
function toast(msg, kind = "") { const t = h("div", { class: "toast " + kind, role: "status" }, msg); $("#toasts").append(t); setTimeout(() => t.remove(), 4200); }
async function copy(text, btn) {
  try { await navigator.clipboard.writeText(text); }
  catch { const ta = h("textarea", { style: "position:fixed;opacity:0" }); ta.value = text; document.body.append(ta); ta.select(); document.execCommand("copy"); ta.remove(); }
  if (btn) { btn.replaceChildren(icon("check")); setTimeout(() => btn.replaceChildren(icon("copy")), 1400); } else toast("Copied", "ok");
}
function codeBlock(text) { const b = h("button", { class: "btn sm icon", type: "button", "aria-label": "Copy", onclick: () => copy(text, b) }, icon("copy")); return h("pre", { class: "code" }, text, b); }
function emptyState(ic, title, text, action) { return h("div", { class: "empty" }, h("div", { class: "ic" }, icon(ic)), h("h4", {}, title), h("p", {}, text), action || null); }
const store = {   // storage can be blocked (private windows); the page works without it
  get(k) { try { return localStorage.getItem(k) ?? sessionStorage.getItem(k); } catch { return null; } },
  set(k, v, keep) { try { (keep ? localStorage : sessionStorage).setItem(k, v); (keep ? sessionStorage : localStorage).removeItem(k); } catch {} },
  del(k) { try { localStorage.removeItem(k); sessionStorage.removeItem(k); } catch {} },
};

function modal(title, body, foot) {
  const bg = h("div", { class: "modal-bg" }), close = () => { bg.remove(); document.removeEventListener("keydown", esc); };
  const esc = e => { if (e.key === "Escape") close(); };
  bg.append(h("div", { class: "modal", role: "dialog", "aria-modal": "true" }, h("div", { class: "hd" }, h("h3", {}, title)), h("div", { class: "bd" }, body), h("div", { class: "ft" }, foot(close))));
  bg.addEventListener("click", e => { if (e.target === bg) close(); }); document.addEventListener("keydown", esc); document.body.append(bg);
}
