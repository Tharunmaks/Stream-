/* ===== privacy notice, "delete all my data", legal links ===== */
/* Everything this site stores lives on this device: localStorage keys "es2.*", one sessionStorage flag, the private file
   folder (OPFS) with saved models, and a service worker. No cookies, no analytics, no ads, no third-party scripts. */
async function wipeAllData() {
  try { Brain.close(); } catch (e) { }
  let failed = [];
  try { const rm = []; for (let i = 0; i < localStorage.length; i++) { const k = localStorage.key(i); if (k && k.indexOf('es2.') === 0) rm.push(k); } rm.forEach(k => localStorage.removeItem(k)); } catch (e) { failed.push('settings'); }
  try { sessionStorage.clear(); } catch (e) { }
  try { if (navigator.storage && navigator.storage.getDirectory) { const root = await navigator.storage.getDirectory(); const names = []; for await (const [n] of root.entries()) names.push(n); for (const n of names) await root.removeEntry(n, { recursive: true }); } } catch (e) { failed.push('saved models'); }
  try { if (navigator.serviceWorker) { const regs = await navigator.serviceWorker.getRegistrations(); for (const r of regs) await r.unregister(); } } catch (e) { }
  try { if (window.caches) { for (const k of await caches.keys()) await caches.delete(k); } } catch (e) { }
  return failed;
}
function privacyCard() {
  const items = [['Cookies', 'none'], ['Analytics, ads, trackers', 'none'], ['Third-party scripts or SDKs', 'none (fonts are built in)'], ['Our servers', 'none: this is a static page'], ['Network requests', 'huggingface.co only when you search, sign in or download; your own device when you connect it'], ['Stored on this device', 'settings, prompt history, saved models, Hugging Face token if you connected']];
  const del = h('button', { class: 'btn danger', onclick: () => {
    const s = sheet(h('div', { class: 'stack' }, h('h3', {}, 'Delete everything?'), h('p', {}, 'This removes your settings, prompt history, saved models, the Hugging Face token and the offline helper from this browser (the helper is set up again the next time you open the page). It cannot be undone. Files you keep outside this site are not touched.'),
      h('div', { class: 'row' }, h('button', { class: 'btn', onclick: () => s.close() }, 'Cancel'), h('button', { class: 'btn danger', onclick: async () => { s.close(); const f = await wipeAllData(); toast(f.length ? 'Deleted, except: ' + f.join(', ') : 'Everything was deleted from this browser.', f.length ? 'bad' : 'good', 5000); setTimeout(() => location.reload(), 1200); } }, 'Delete everything'))));
  } }, 'Delete all my data');
  return h('div', { class: 'card', id: 'privacy-card' }, h('span', { class: 'eyebrow' }, 'Privacy'), h('h3', {}, 'Your data stays on your device'),
    h('div', { class: 'table-box' }, ...items.map(([a, b]) => trow(a, b))),
    h('p', { class: 'dim', style: 'font-size:12.5px' }, 'Read the ', h('a', { href: 'legal.html#privacy' }, 'privacy policy'), ', ', h('a', { href: 'legal.html#terms' }, 'terms'), ' and ', h('a', { href: 'legal.html#cookies' }, 'cookie and storage list'), '. You can delete your data at any time (your right to erasure):'),
    h('div', { class: 'row' }, del, h('a', { class: 'btn', href: 'legal.html' }, 'Legal and accessibility page')));
}
function privacyNotice() {
  if (store.get('notice', 0)) return;
  const n = h('div', { class: 'notice', role: 'region', 'aria-label': 'Privacy notice' },
    h('p', {}, 'No cookies, no tracking, no ads. Your models, chats and settings stay on this device. We have no account system and run no server.'),
    h('div', { class: 'row' }, h('a', { class: 'btn sm', href: 'legal.html#privacy' }, 'Details'), h('button', { class: 'btn sm primary', onclick: () => { store.set('notice', 1); n.remove(); } }, 'OK')));
  document.body.append(n);
}
setTimeout(privacyNotice, 600);
/* decorative icons are hidden from screen readers; buttons already carry text or aria-label */
(function hideDecorativeSvgs() {
  const mark = root => (root.querySelectorAll ? root : document).querySelectorAll('svg:not([aria-label]):not([role]):not([aria-hidden])').forEach(s => s.setAttribute('aria-hidden', 'true'));
  mark(document);
  new MutationObserver(ms => { for (const m of ms) for (const n of m.addedNodes) if (n.nodeType === 1) { if (n.tagName === 'svg' && !n.hasAttribute('aria-label') && !n.hasAttribute('role')) n.setAttribute('aria-hidden', 'true'); else mark(n); } }).observe(document.body, { childList: true, subtree: true });
})();
