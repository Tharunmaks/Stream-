/* ===== hero: a 3D tower of experts drawn with a plain 2D canvas (no libraries, works everywhere) ===== */
(function () {
  const cv = $('#scene'); if (!cv || !cv.getContext) return; const g = cv.getContext('2d');
  const LAY = 12, PER = 36, pts = [], reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  for (let l = 0; l < LAY; l++) for (let e = 0; e < PER; e++) { const a = e / PER * Math.PI * 2 + l * .22, r = 2.5 + .35 * Math.sin(l / (LAY - 1) * Math.PI); pts.push({ x: Math.cos(a) * r, y: 3 - l * 6 / (LAY - 1), z: Math.sin(a) * r, l, glow: 0, k: 0 }); }
  let W = 0, H = 0, dpr = 1, rot = 0, layer = 0, t0 = 0, vis = true;
  const COL = [[57, 135, 229], [25, 176, 125], [232, 104, 58]];
  function size() { dpr = Math.min(devicePixelRatio || 1, 2); W = cv.clientWidth; H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0); }
  function pick(l) { for (let j = 0; j < 8; j++) { const p = pts[l * PER + (Math.random() * PER | 0)]; p.glow = 1; p.k = Math.random() < .5 ? 0 : Math.random() < .6 ? 1 : 2; } }
  function frame(now) {
    if (vis && document.body.dataset.page === 'home') {
      if (!W || W !== cv.clientWidth) size();
      if (now - t0 > 240) { t0 = now; layer = (layer + 1) % LAY; pick(layer); }
      rot += reduce ? 0 : .004; g.clearRect(0, 0, W, H);
      const cx = W < 640 ? W / 2 : W * .72, cy = H * .5, sc = Math.min(W, H) / 8.2, cr = Math.cos(rot), sr = Math.sin(rot), P = [];
      for (const p of pts) { const x = p.x * cr - p.z * sr, z = p.x * sr + p.z * cr, f = 9 / (9 + z); P.push({ p, X: cx + x * sc * f, Y: cy + p.y * sc * f * .95 + z * sc * .05, f, z }); }
      P.sort((a, b) => b.z - a.z);
      for (const q of P) { const p = q.p; if (p.glow > 0) p.glow = Math.max(0, p.glow - .012); const c = COL[p.k], a = .2 + .5 * q.f * .6 + p.glow * .6, s = (2.2 + p.glow * 3.2) * q.f * (sc / 40);
        g.fillStyle = p.glow > .02 ? `rgba(${c[0]},${c[1]},${c[2]},${Math.min(1, a + .2)})` : `rgba(80,100,140,${a * .8})`; g.beginPath(); g.arc(q.X, q.Y, Math.max(.8, s), 0, 6.283); g.fill();
        if (p.glow > .3) { g.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${p.glow * .15})`; g.beginPath(); g.arc(q.X, q.Y, s * 3, 0, 6.283); g.fill(); } }
      const y = cy + (3 - ((layer + Math.min(1, (now - t0) / 240)) * 6 / (LAY - 1))) * sc * .95, gr = g.createRadialGradient(cx, y, 0, cx, y, sc * .7); gr.addColorStop(0, 'rgba(255,240,210,.95)'); gr.addColorStop(.3, 'rgba(242,201,138,.35)'); gr.addColorStop(1, 'rgba(242,201,138,0)'); g.fillStyle = gr; g.beginPath(); g.arc(cx, y, sc * .7, 0, 6.283); g.fill();
    }
    requestAnimationFrame(frame);
  }
  document.addEventListener('visibilitychange', () => { vis = !document.hidden; });
  addEventListener('resize', () => { W = 0; });
  for (let l = 0; l < LAY; l++) pick(l); requestAnimationFrame(frame);
})();
