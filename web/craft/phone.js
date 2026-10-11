  // ---------------- phone planner: same sizing rules as craft/phone (backbone ladder, one 4 KB page per expert, 8 experts per word)
  var BB = @@BACKBONES@@, PAGE = 4096, PER_TOKEN = 3, FMIN = 1e9, FMAX = 70e9;
  var fT = 70e9;
  var fx = function (v) { return Math.round(1000 * (Math.log(v) - Math.log(FMIN)) / (Math.log(FMAX) - Math.log(FMIN))); };
  var fy = function (v) { return Math.exp(Math.log(FMIN) + (Math.log(FMAX) - Math.log(FMIN)) * v / 1000); };
  [["1B", 1e9], ["7B", 7e9], ["30B", 30e9], ["70B", 70e9]].forEach(function (p) {
    var b = document.createElement("button"); b.textContent = p[0]; b.dataset.t = p[1]; b.onclick = function () { fT = p[1]; fset(); }; $("f-presets").appendChild(b);
  });
  function fset() {
    var minutes = Math.max(5, +$("f-min").value || 600), tps = Math.max(1, +$("f-tps").value || 400), free = Math.max(1, +$("f-disk").value || 100);
    var tokens = tps * 60 * minutes, bb = BB[0];
    BB.forEach(function (r) { if (r[0] <= tokens / 20) bb = r; });
    var bp = bb[0], per = 2 * bb[3] * bb[1], shrunk = false, want = fT;
    var nk = Math.max(2, Math.round(Math.sqrt(Math.max(want - bp, 1e6) / per)));
    if (nk * nk * PAGE / 1e9 > free * 0.85) { nk = Math.max(2, Math.floor(Math.sqrt(free * 0.85 * 1e9 / PAGE))); shrunk = true; }
    var E = nk * nk, total = bp + E * per, active = bp + PER_TOKEN * per, visits = tokens * PER_TOKEN / E, file = E * PAGE / 1e9, wearTB = tps * 86400 * PER_TOKEN * PAGE / 1e12;
    $("f-range").value = fx(fT); $("f-size").textContent = fmt(fT);
    $("f-total").textContent = fmt(total); $("f-dense").textContent = fmt(bp) + " (" + bb[2] + " × " + bb[1] + ")"; $("f-exp").textContent = E.toLocaleString("en-US");
    $("f-active").textContent = fmt(active) + " (" + (100 * active / total).toFixed(4) + "%)"; $("f-tokens").textContent = fmt(tokens); $("f-visits").textContent = visits >= 10 ? visits.toFixed(0) : visits.toFixed(2);
    $("f-file").textContent = file >= 10 ? file.toFixed(0) + " GB" : file.toFixed(1) + " GB"; $("f-wear").textContent = wearTB.toFixed(1) + " TB";
    [].forEach.call($("f-presets").children, function (b) { b.setAttribute("aria-pressed", Math.abs(Math.log(+b.dataset.t / fT)) < 0.05 ? "true" : "false"); });
    var v = $("f-verdict"), msg = [];
    if (shrunk) msg.push("Your free storage fits about " + fmt(total) + " parameters, so the plan is cut to that.");
    if (visits < 1) { v.className = "verdict warn"; msg.push("In this time each expert is used about " + visits.toFixed(2) + " times, so most of the file stays unused. Train longer or choose a smaller size."); }
    else { v.className = "verdict good"; msg.push("Each expert is used about " + (visits >= 10 ? visits.toFixed(0) : visits.toFixed(1)) + " times."); }
    msg.push("Text quality follows the " + fmt(tokens) + " tokens seen and the " + fmt(bp) + " dense model, not the file size. Nonstop training writes about " + wearTB.toFixed(1) + " TB of flash a day (about " + (100 * wearTB / 500).toFixed(1) + "% of the life of a 500 TB flash, an assumed rating).");
    v.textContent = msg.join(" ");
  }
  $("f-range").addEventListener("input", function () { fT = fy(+this.value); fset(); });
  ["f-min", "f-tps", "f-disk"].forEach(function (id) { $(id).addEventListener("input", fset); });
  fset();
