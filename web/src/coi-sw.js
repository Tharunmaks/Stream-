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
    if (/\/(index\.html)?$/.test(new URL(r.url).pathname)) h.set('Content-Security-Policy', "@@CSP@@");
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
  }).catch(() => fetch(r)));
});
