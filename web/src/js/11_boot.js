/* ===== boot ===== */
$$('.tab').forEach(t => t.addEventListener('click', () => go(t.dataset.page)));
addEventListener('hashchange', () => go(location.hash.slice(1), false));
addEventListener('unhandledrejection', e => { const m = e.reason && e.reason.message; if (m && !/AbortError/.test(m)) toast(m, 'bad', 5000); });
(async () => {
  chatChipUpdate();
  go((location.hash || '#home').slice(1), false);
  Lib.list().then(l => { $('#home-lib').textContent = l.filter(i => i.complete).length; }).catch(() => { });
  const nat = await Brain.detectNative();
  if (nat) toast('Connected to the native engine: ' + nat.model, 'good');
  Demo.probe();
  renderDoctor();
  if (nat) chatWelcome();
})();
