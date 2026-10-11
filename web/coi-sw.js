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
    h.set('X-Content-Type-Options', 'nosniff');
    h.set('Referrer-Policy', 'no-referrer');
    h.set('Cross-Origin-Resource-Policy', 'same-origin');
    h.set('Permissions-Policy', 'camera=(), microphone=(), geolocation=(), payment=(), usb=(), serial=(), bluetooth=(), accelerometer=(), gyroscope=(), magnetometer=(), clipboard-read=(), interest-cohort=()');
    /* the CSP header pins this app page's inline scripts by hash; other pages (craft.html) carry their own CSP meta */
    if (/\/(index\.html)?$/.test(new URL(r.url).pathname)) h.set('Content-Security-Policy', "default-src 'none'; script-src 'self' 'wasm-unsafe-eval' 'sha256-Eu57U32kC6G4T3baE6E/SaehzkDdqiilNciFvGJAkH8=' 'sha256-0P969QzR2eaZAWk5pVjV6eiOfHrc/VeHeNNBbgdMui4='; worker-src 'self' blob:; style-src 'unsafe-inline'; font-src data:; img-src 'self' data: blob: https://*.huggingface.co https://*.hf.co; connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co https://*.trycloudflare.com https://*.lhr.life http://127.0.0.1:* http://localhost:* http://*:8765; base-uri 'none'; form-action 'none'; object-src 'none'; frame-src 'none'; manifest-src 'none'; media-src 'none'; frame-ancestors 'none'");
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
  }).catch(() => fetch(r)));
});
