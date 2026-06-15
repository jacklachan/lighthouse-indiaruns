/* Lighthouse sandbox — front-end interactions + motion.
   Talks to the FastAPI server (/api/rank, /api/sample). anime.js is vendored. */
(function () {
  "use strict";

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var hasAnime = typeof window.anime === "function";

  function esc(v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  /* ------------------------------------------------------------------ *
   *  Hero beam + decorative dots
   * ------------------------------------------------------------------ */
  function initBeam() {
    var dots = $("#dots");
    if (dots) {
      for (var i = 0; i < 14; i++) {
        var ang = Math.random() * Math.PI * 2;
        var rad = 40 + Math.random() * 78;
        var cx = 120 + Math.cos(ang) * rad;
        var cy = 120 + Math.sin(ang) * rad;
        var c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        c.setAttribute("cx", cx.toFixed(1));
        c.setAttribute("cy", cy.toFixed(1));
        c.setAttribute("r", (1.4 + Math.random() * 1.8).toFixed(1));
        c.setAttribute("fill", Math.random() > 0.5 ? "rgba(111,123,255,0.85)" : "rgba(246,181,61,0.85)");
        dots.appendChild(c);
        if (hasAnime) {
          window.anime({
            targets: c, opacity: [0.25, 1], r: [Number(c.getAttribute("r")), Number(c.getAttribute("r")) + 1.2],
            duration: 1600 + Math.random() * 1800, direction: "alternate", loop: true,
            easing: "easeInOutSine", delay: Math.random() * 1500
          });
        }
      }
    }
    if (hasAnime) {
      window.anime({ targets: ".beam-cone", rotate: "360deg", duration: 14000, loop: true, easing: "linear" });
    }
  }

  /* ------------------------------------------------------------------ *
   *  Count-ups + scroll reveal
   * ------------------------------------------------------------------ */
  function countUp(el, to, opts) {
    opts = opts || {};
    if (!hasAnime) { el.textContent = (opts.fmt ? opts.fmt(to) : to); return; }
    var obj = { v: 0 };
    window.anime({
      targets: obj, v: to, round: opts.round || 1, duration: opts.duration || 1400,
      easing: "easeOutExpo", delay: opts.delay || 0,
      update: function () { el.textContent = opts.fmt ? opts.fmt(obj.v) : String(obj.v); }
    });
  }

  function initHeroStats() {
    $$(".stat .num[data-count]").forEach(function (el, i) {
      var to = Number(el.getAttribute("data-count"));
      countUp(el, to, { delay: 300 + i * 140, fmt: function (n) { return Math.round(n).toLocaleString("en-US"); } });
    });
  }

  function initReveal() {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var el = e.target;
        var sibs = Array.prototype.filter.call(el.parentElement.children, function (c) {
          return c.classList.contains("reveal");
        });
        var idx = sibs.indexOf(el);
        if (hasAnime) {
          window.anime({ targets: el, translateY: [26, 0], opacity: [0, 1], delay: idx * 70, duration: 700, easing: "easeOutCubic" });
        } else { el.style.opacity = 1; el.style.transform = "none"; }
        io.unobserve(el);
      });
    }, { threshold: 0.12 });
    $$(".reveal").forEach(function (el) { io.observe(el); });
  }

  /* ------------------------------------------------------------------ *
   *  Tool: input mode, upload, ranking
   * ------------------------------------------------------------------ */
  var state = { mode: "sample", fileText: null, fileName: null, lastRows: null };

  function setMode(mode) {
    state.mode = mode;
    $$("#mode button").forEach(function (b) { b.classList.toggle("active", b.getAttribute("data-mode") === mode); });
    $("#upload-pane").style.display = mode === "upload" ? "block" : "none";
    $("#src-note").textContent = mode === "sample"
      ? "Scoring the preloaded sample of 100 candidates."
      : (state.fileName ? "Ready to score " + state.fileName + "." : "Drop or choose a JSONL file (≤100 candidates).");
  }

  function initModeToggle() {
    $$("#mode button").forEach(function (b) {
      b.addEventListener("click", function () { setMode(b.getAttribute("data-mode")); });
    });
  }

  function initUpload() {
    var dz = $("#dropzone"), input = $("#file-input");
    if (!dz) return;
    dz.addEventListener("click", function () { input.click(); });
    ["dragenter", "dragover"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add("drag"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove("drag"); });
    });
    dz.addEventListener("drop", function (e) {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) readFile(e.dataTransfer.files[0]);
    });
    input.addEventListener("change", function () { if (input.files[0]) readFile(input.files[0]); });
  }

  function readFile(file) {
    var reader = new FileReader();
    reader.onload = function () {
      state.fileText = reader.result;
      state.fileName = file.name;
      $("#file-name").textContent = "✓ " + file.name;
      $("#src-note").textContent = "Ready to score " + file.name + ".";
    };
    reader.readAsText(file);
  }

  function showError(msg) {
    var el = $("#error");
    el.textContent = "⚠ " + msg;
    el.classList.add("show");
  }
  function clearError() { $("#error").classList.remove("show"); }

  function setLoading(on) {
    $("#loading").classList.toggle("show", on);
    var btn = $("#rank-btn");
    btn.disabled = on;
    btn.textContent = on ? "Sweeping…" : "🔦 Rank candidates";
  }

  function rank() {
    clearError();
    var body;
    if (state.mode === "upload") {
      if (!state.fileText) { showError("Choose or drop a JSONL file first."); return; }
      body = { jsonl: state.fileText, use_sample: false };
    } else {
      body = { use_sample: true };
    }
    setLoading(true);
    $("#results").classList.remove("show");

    fetch("/api/rank", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (res) {
      return res.json().then(function (data) {
        if (!res.ok) throw new Error(data.detail || ("Request failed (" + res.status + ")"));
        return data;
      });
    }).then(function (data) {
      setLoading(false);
      render(data);
    }).catch(function (err) {
      setLoading(false);
      showError(err.message || "Ranking failed. Please try again.");
    });
  }

  /* ------------------------------------------------------------------ *
   *  Render results
   * ------------------------------------------------------------------ */
  function render(data) {
    state.lastRows = data.rows || [];
    var results = $("#results");
    results.classList.add("show");

    // metrics
    var s = data.summary || {};
    countUp($('[data-metric="ranked"]'), s.ranked || 0, {});
    countUp($('[data-metric="honeypots"]'), s.honeypots || 0, {});
    var top = $('[data-metric="top"]');
    if (s.top_score == null) { top.textContent = "—"; }
    else { countUp(top, s.top_score, { round: 1000, duration: 1200, fmt: function (n) { return n.toFixed(3); } }); }

    // table
    var tbody = $("#rows");
    tbody.innerHTML = "";
    (data.rows || []).forEach(function (r) {
      var tr = document.createElement("tr");
      var rankCls = r.rank === 1 ? "rank-pill top" : "rank-pill";
      var hp = r.honeypot ? ' <span class="hp-badge">⚠ honeypot</span>' : "";
      tr.innerHTML =
        '<td><span class="' + rankCls + '">' + r.rank + "</span></td>" +
        '<td><div class="cid">' + esc(r.candidate_id) + "</div>" +
          '<div class="score-bar"><span style="right:' + (100 - Math.max(0, Math.min(1, r.score)) * 100).toFixed(1) + '%"></span></div></td>' +
        '<td><div>' + esc(r.title || "—") + hp + '</div><div class="muted-cell">' + esc(r.country || "") + "</div></td>" +
        '<td class="muted-cell">' + (r.yrs != null ? r.yrs : "—") + "</td>" +
        '<td class="reason-cell">' + esc(r.reasoning || "") + "</td>";
      tbody.appendChild(tr);
    });
    if (hasAnime) {
      window.anime({ targets: "#rows tr", translateY: [14, 0], opacity: [0, 1], delay: window.anime.stagger(35), duration: 480, easing: "easeOutCubic" });
    }

    renderBreakdown(data.breakdown, data.weights || {});

    // reveal score bars after paint
    requestAnimationFrame(function () {
      $$("#rows .score-bar > span").forEach(function (sp, i) {
        var target = sp.style.right;
        sp.style.right = "100%";
        if (hasAnime) window.anime({ targets: sp, right: target, duration: 800, delay: 120 + i * 18, easing: "easeOutCubic" });
        else sp.style.right = target;
      });
    });

    results.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  var COMP_LABELS = {
    semantic_fit: "Semantic fit",
    role_coherence: "Role coherence",
    career_evidence: "Career evidence",
    experience_fit: "Experience fit",
    trust_skills: "Trust-weighted skills"
  };

  function renderBreakdown(b, weights) {
    var host = $("#breakdown-body");
    if (!b) { host.innerHTML = '<p class="note">No candidate to break down.</p>'; return; }
    var comps = b.components || {};
    var html = '<div class="note" style="margin-bottom:18px">Top candidate <strong style="color:var(--text)">' + esc(b.candidate_id) + "</strong> · base score " + esc(b.base) + "</div>";
    Object.keys(COMP_LABELS).forEach(function (k) {
      if (comps[k] == null) return;
      var pct = Math.max(0, Math.min(1, comps[k])) * 100;
      var w = weights[k] != null ? " · w " + weights[k] : "";
      html +=
        '<div class="comp-row">' +
          '<div class="top"><span class="name">' + COMP_LABELS[k] + "<small>" + w + "</small></span>" +
          '<span class="val">' + Number(comps[k]).toFixed(3) + "</span></div>" +
          '<div class="comp-track"><div class="comp-fill" data-pct="' + pct.toFixed(1) + '"></div></div>' +
        "</div>";
    });
    html +=
      '<div class="mult-grid">' +
        '<div class="mult"><div class="k">Gate multiplier</div><div class="v">×' + esc(b.gate_mult) + "</div></div>" +
        '<div class="mult"><div class="k">Behavior multiplier</div><div class="v">×' + esc(b.behavior_mult) + "</div></div>" +
        '<div class="mult"><div class="k">Honeypot</div><div class="v" style="color:' + (b.honeypot ? "var(--danger)" : "var(--good)") + '">' + (b.honeypot ? "flagged" : "clean") + "</div></div>" +
      "</div>";
    if (b.gate_reasons && b.gate_reasons.length) {
      html += '<div class="reasons-list">Gate notes:<ul>' + b.gate_reasons.map(function (g) { return "<li>" + esc(g) + "</li>"; }).join("") + "</ul></div>";
    }
    host.innerHTML = html;

    // animate the bars when the details is open
    var bd = $("#breakdown");
    var animateBars = function () {
      $$(".comp-fill", host).forEach(function (f, i) {
        var pct = f.getAttribute("data-pct") + "%";
        if (hasAnime) window.anime({ targets: f, width: pct, duration: 900, delay: i * 90, easing: "easeOutQuart" });
        else f.style.width = pct;
      });
    };
    bd.addEventListener("toggle", function () { if (bd.open) animateBars(); }, { once: false });
  }

  /* ------------------------------------------------------------------ *
   *  CSV download (built client-side, matching the app's columns)
   * ------------------------------------------------------------------ */
  function csvCell(v) {
    v = String(v == null ? "" : v);
    return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  }
  function downloadCsv() {
    if (!state.lastRows || !state.lastRows.length) return;
    var header = ["candidate_id", "rank", "score", "reasoning"];
    var lines = [header.join(",")];
    state.lastRows.forEach(function (r) {
      lines.push([csvCell(r.candidate_id), csvCell(r.rank), csvCell(r.score), csvCell(r.reasoning)].join(","));
    });
    var blob = new Blob([lines.join("\n")], { type: "text/csv" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url; a.download = "submission.csv"; a.click();
    URL.revokeObjectURL(url);
  }

  /* ------------------------------------------------------------------ */
  document.addEventListener("DOMContentLoaded", function () {
    initBeam();
    initHeroStats();
    initReveal();
    initModeToggle();
    initUpload();
    setMode("sample");
    $("#rank-btn").addEventListener("click", rank);
    $("#download-btn").addEventListener("click", downloadCsv);
  });
})();
