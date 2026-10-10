/* ===== core helpers ===== */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const k in (attrs || {})) {
    const v = attrs[k];
    if (k === 'class') e.className = v; else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, ''); else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
}
const store = {
  get(k, d) { try { const v = localStorage.getItem('es2.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('es2.' + k, JSON.stringify(v)); } catch (e) {} },
};
const fmtB = n => n >= 1e12 ? (n / 1e12).toFixed(1) + ' TB' : n >= 1e9 ? (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + ' GB' : n >= 1e6 ? (n / 1e6).toFixed(0) + ' MB' : n >= 1e3 ? (n / 1e3).toFixed(0) + ' KB' : n + ' B';
const fmtN = n => n >= 1e12 ? (n / 1e12).toFixed(1) + 'T' : n >= 1e9 ? (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(0) + 'M' : String(n);
const fmtT = s => s < 60 ? Math.round(s) + ' s' : s < 3600 ? Math.round(s / 60) + ' min' : s < 86400 ? (s / 3600).toFixed(1) + ' h' : (s / 86400).toFixed(1) + ' days';
const sleep = ms => new Promise(r => setTimeout(r, ms));
const copyText = async (t) => { try { await navigator.clipboard.writeText(t); toast('Copied', 'good'); } catch (e) { toast('Copy not allowed here', 'bad'); } };

/* toasts */
function toast(msg, kind, ms = 3200) {
  const box = $('#toasts'); if (!box) return;
  const t = h('div', { class: 'toast ' + (kind || '') }, msg);
  box.append(t);
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 320); }, ms);
}
/* bottom sheet */
function sheet(content) {
  const bg = h('div', { class: 'sheet-bg', onclick: e => { if (e.target === bg) close(); } });
  const sh = h('div', { class: 'sheet', role: 'dialog' }, content);
  bg.append(sh); document.body.append(bg);
  function close() { bg.remove(); }
  return { close, el: sh };
}
/* tiny markdown for answers */
function md(src) {
  const blocks = []; let s = esc(src).replace(/```(\w*)\n?([\s\S]*?)(```|$)/g, (m, lang, code) => { blocks.push([lang, code.replace(/\n$/, '')]); return '\u0000' + (blocks.length - 1) + '\u0000'; });
  s = s.replace(/`([^`\n]+)`/g, '<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>');
  const out = []; let list = null;
  for (const line of s.split('\n')) {
    let m;
    if ((m = line.match(/^\s*[-*] (.*)/)) || (m = line.match(/^\s*\d+[.)] (.*)/))) { const t = /^\s*\d/.test(line) ? 'ol' : 'ul'; if (!list) { list = t; out.push('<' + t + '>'); } out.push('<li>' + m[1] + '</li>'); continue; }
    if (list) { out.push('</' + list + '>'); list = null; }
    if ((m = line.match(/^#{1,4} (.*)/))) out.push('<h3>' + m[1] + '</h3>');
    else if (!line.trim()) out.push('');
    else out.push('<p>' + line + '</p>');
  }
  if (list) out.push('</' + list + '>');
  return out.join('').replace(/\u0000(\d+)\u0000/g, (m, i) => { const [l, c] = blocks[+i]; return '<pre><span class="lang">' + (l || 'code') + '</span><button class="cp" data-copy>copy</button><code>' + c + '</code></pre>'; });
}
document.addEventListener('click', e => {
  const b = e.target.closest('[data-copy]'); if (!b) return;
  const code = b.parentElement.querySelector('code'); copyText(code ? code.textContent : '');
});

/* router: one page at a time, hash based */
const PAGES = ['home', 'models', 'chat', 'terminal', 'testing', 'history', 'settings'];
const pageHooks = {};
function go(page, push = true) {
  if (!PAGES.includes(page)) page = 'home';
  for (const p of PAGES) {
    const sec = $('#' + p), on = p === page;
    if (sec) { sec.hidden = !on; if (on) { sec.style.animation = 'none'; void sec.offsetWidth; sec.style.animation = ''; } }
  }
  $$('.tab').forEach(t => t.setAttribute('aria-selected', t.dataset.page === page));
  if (push && location.hash !== '#' + page) { try { history.replaceState(null, '', '#' + page); } catch (e) { } }
  window.scrollTo({ top: 0 });
  document.body.dataset.page = page;
  if (pageHooks[page]) pageHooks[page]();
}
document.addEventListener('click', e => { const g = e.target.closest('[data-goto]'); if (g) { e.preventDefault(); go(g.dataset.goto); } });
