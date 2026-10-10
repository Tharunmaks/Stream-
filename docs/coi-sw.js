// ExpertStream: adds Cross-Origin isolation headers to this site's own pages so the
// engine can use SharedArrayBuffer and run on several CPU cores. Nothing else is changed.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', (e) => {
  const r = e.request;
  if (r.cache === 'only-if-cached' && r.mode !== 'same-origin') return;
  if (new URL(r.url).origin !== self.location.origin) return;
  e.respondWith(fetch(r).then((res) => {
    if (!res || res.status === 0 || res.type === 'opaque') return res;
    const h = new Headers(res.headers);
    h.set('Cross-Origin-Embedder-Policy', 'credentialless');
    h.set('Cross-Origin-Opener-Policy', 'same-origin');
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
  }).catch(() => fetch(r)));
});
