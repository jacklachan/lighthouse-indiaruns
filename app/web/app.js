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

  function gateShortLabel(reason) {
    // Map the long gate-reason text emitted by lighthouse/gates.py to a short
    // 1-2 word badge label. Keeps the full reason on hover via title=.
    var r = (reason || "").toLowerCase();
    if (r.indexOf("services/consulting") >= 0) return "services-only";
    if (r.indexOf("relocate") >= 0 || r.indexOf("visa") >= 0) return "location";
    if (r.indexOf("research-heavy") >= 0) return "research";
    if (r.indexOf("computer vision") >= 0 || r.indexOf("speech") >= 0) return "cv/speech";
    if (r.indexOf("langchain") >= 0 || r.indexOf("wrapper") >= 0) return "wrapper-only";
    if (r.indexOf("title-chaser") >= 0 || r.indexOf("job-hopping") >= 0) return "title-chaser";
    if (r.indexOf("non-engineering") >= 0) return "non-technical";
    return "gate";
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
  var state = {
    mode: "sample",
    fileText: null,
    fileName: null,
    lastRows: null,
    defaultWeights: null,
    weights: null,
    gates: [],          // [{key, label, desc}]
    skipGates: {},      // {key: bool}  true = disabled
    lastRankBody: null,
    rawById: null       // {candidate_id: raw record} for the detail modal (lazy)
  };

  /* Hand-tuned presets — each one snaps both the weight sliders and the
     gate toggles. Designed for the demo: a judge can see the ranking shift
     between "JD as written" and "I'd hire remotely / from services / etc."
     in one click. weights values reflect the JD defaults so the deltas are
     readable; gates list keys to BYPASS. */
  var PRESETS = {
    default: {
      label: "JD defaults",
      weights: { semantic_fit: 0.22, role_coherence: 0.26, career_evidence: 0.24, experience_fit: 0.10, trust_skills: 0.18 },
      skip: []
    },
    remote: {
      label: "Remote-hire",
      weights: { semantic_fit: 0.22, role_coherence: 0.26, career_evidence: 0.26, experience_fit: 0.10, trust_skills: 0.16 },
      skip: ["location_visa"]
    },
    junior: {
      label: "Junior-friendly",
      weights: { semantic_fit: 0.26, role_coherence: 0.22, career_evidence: 0.20, experience_fit: 0.04, trust_skills: 0.28 },
      skip: ["title_chaser"]
    },
    research: {
      label: "Research-heavy",
      weights: { semantic_fit: 0.28, role_coherence: 0.22, career_evidence: 0.18, experience_fit: 0.10, trust_skills: 0.22 },
      skip: ["research_only"]
    },
    services_ok: {
      label: "Services OK",
      weights: { semantic_fit: 0.22, role_coherence: 0.26, career_evidence: 0.24, experience_fit: 0.10, trust_skills: 0.18 },
      skip: ["services_only"]
    }
  };

  function setMode(mode) {
    state.mode = mode;
    state.rawById = null;   // source changed -> drop the detail-modal raw cache
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
      state.rawById = null;   // new upload -> drop the detail-modal raw cache
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

  function setLoading(on, quiet) {
    $("#loading").classList.toggle("show", on && !quiet);
    var btn = $("#rank-btn");
    btn.disabled = on;
    btn.textContent = on ? "Sweeping…" : "🔦 Rank candidates";
    // Quiet re-ranks (slider / gate / preset) skip the full sweep, so give them a
    // visible cue — otherwise a change that only moves deep ranks looks like a dead button.
    if (on && quiet) setRankStatus("re-ranking…", "busy");
  }

  function setRankStatus(text, cls) {
    var el = $("#rank-status");
    if (!el) return;
    el.textContent = text;
    el.className = "rank-status" + (cls ? " " + cls : "") + (text ? " show" : "");
  }

  // Diff two ranked lists by candidate_id -> rank. null on the first render
  // (nothing to compare against yet).
  function computeMovement(prev, next) {
    if (!prev || !prev.length) return null;
    var was = {};
    prev.forEach(function (r) { was[r.candidate_id] = r.rank; });
    var moved = 0, inTop = 0;
    next.forEach(function (r) {
      var p = was[r.candidate_id];
      if (p != null && p !== r.rank) { moved++; if (r.rank <= 20 || p <= 20) inTop++; }
    });
    return { moved: moved, inTop: inTop, was: was };
  }

  function rank(opts) {
    opts = opts || {};
    clearError();
    var body;
    if (state.mode === "upload") {
      if (!state.fileText) { showError("Choose or drop a JSONL file first."); return; }
      body = { jsonl: state.fileText, use_sample: false };
    } else {
      body = { use_sample: true };
    }
    if (state.weights && !weightsEqualDefaults(state.weights, state.defaultWeights)) {
      body.weights = state.weights;
    }
    var skipList = activeSkipGates();
    if (skipList.length) body.skip_gates = skipList;
    state.lastRankBody = body;
    setLoading(true, opts.quiet);
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
    var newRows = data.rows || [];
    var move = computeMovement(state.lastRows, newRows);   // null on first render
    var wasMap = move ? move.was : null;
    state.lastRows = newRows;
    var results = $("#results");
    results.classList.add("show");

    // Tell the user a re-rank happened and how much actually moved — deep-only
    // shuffles (gate toggles, gate-only presets) otherwise read as "nothing changed".
    if (move) {
      if (move.moved === 0) {
        setRankStatus("No movement — these settings don't reorder this sample", "flat");
      } else {
        setRankStatus("Re-ranked · " + move.moved + " moved" +
          (move.inTop ? " · " + move.inTop + " in top 20" : " (all below top 20)"), "ok");
      }
    } else {
      setRankStatus("", "");
    }

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
    newRows.forEach(function (r) {
      var tr = document.createElement("tr");
      tr.setAttribute("data-cid", r.candidate_id);   // click -> detail modal
      if (wasMap && wasMap[r.candidate_id] != null && wasMap[r.candidate_id] !== r.rank) {
        tr.setAttribute("data-moved", "");   // brief highlight on rows that shifted
      }
      var rankCls = r.rank === 1 ? "rank-pill top" : "rank-pill";
      var hp = r.honeypot ? ' <span class="hp-badge">⚠ honeypot</span>' : "";
      // Per-row gate badges: each fired hard-negative shown as a small chip
      // with the multiplier so a recruiter scanning the table can see WHICH
      // gates dampened each candidate without expanding the breakdown.
      var gateChips = "";
      if (r.gate_reasons && r.gate_reasons.length) {
        gateChips = ' <span class="gate-chips">' + r.gate_reasons.map(function (g) {
          var label = gateShortLabel(g);
          return '<span class="gate-chip" title="' + esc(g) + '">' + esc(label) + '</span>';
        }).join("") + "</span>";
      }
      tr.innerHTML =
        '<td><span class="' + rankCls + '">' + r.rank + "</span></td>" +
        '<td><div class="cid">' + esc(r.candidate_id) +
          ' <button class="similar-btn" data-cid="' + esc(r.candidate_id) + '" title="Find similar candidates">similar</button></div>' +
          '<div class="score-bar"><span style="right:' + (100 - Math.max(0, Math.min(1, r.score)) * 100).toFixed(1) + '%"></span></div></td>' +
        '<td><div>' + esc(r.title || "—") + hp + gateChips + '</div><div class="muted-cell">' + esc(r.country || "") + "</div></td>" +
        '<td class="muted-cell">' + (r.yrs != null ? r.yrs : "—") + "</td>" +
        '<td class="reason-cell">' + esc(r.reasoning || "") + "</td>";
      tbody.appendChild(tr);
    });
    $$("#rows .similar-btn").forEach(function (b) {
      b.addEventListener("click", function (e) { e.stopPropagation(); findSimilar(b.getAttribute("data-cid")); });
    });
    $$("#rows tr").forEach(function (tr) {
      tr.addEventListener("click", function () { openDetail(tr.getAttribute("data-cid")); });
    });
    if (hasAnime) {
      window.anime({ targets: "#rows tr", translateY: [14, 0], opacity: [0, 1], delay: window.anime.stagger(35), duration: 480, easing: "easeOutCubic" });
    }

    renderBreakdown(data.breakdown, data.weights || {});

    // Score bars keep the inline `right:X%` width set in the row HTML, so they
    // render filled immediately and survive every re-render. A CSS transition on
    // `.score-bar > span` handles the grow — no JS collapse-then-animate (that
    // left bars stuck at width 0 / colorless on quiet re-ranks).
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
   *  Candidate detail modal (click a row -> profile + score breakdown)
   * ------------------------------------------------------------------ */
  function parseJsonlMap(text) {
    var m = {};
    (text || "").split("\n").forEach(function (ln) {
      ln = ln.trim();
      if (!ln) return;
      try { var r = JSON.parse(ln); if (r && r.candidate_id) m[r.candidate_id] = r; } catch (e) { /* skip */ }
    });
    return m;
  }

  // Raw records for the current source, cached. Upload mode reuses the text the
  // user already provided; sample mode fetches /api/sample once (~74KB gzipped).
  function ensureRaws() {
    if (state.rawById) return Promise.resolve(state.rawById);
    if (state.mode === "upload" && state.fileText) {
      state.rawById = parseJsonlMap(state.fileText);
      return Promise.resolve(state.rawById);
    }
    return fetch("/api/sample")
      .then(function (res) { return res.text(); })
      .then(function (txt) { state.rawById = parseJsonlMap(txt); return state.rawById; })
      .catch(function () { state.rawById = {}; return state.rawById; });
  }

  function yrsFromMonths(m) {
    if (m == null || m < 0) return "";
    var y = Math.round((m / 12) * 10) / 10;
    return y + (y === 1 ? " yr" : " yrs");
  }

  function detailScoreHtml(row) {
    var comps = row.components || {};
    var html = '<div class="detail-section"><h4>Why this rank</h4>';
    html += '<p class="detail-reason">' + esc(row.reasoning || "") + "</p>";
    html += '<div class="detail-score">';
    Object.keys(COMP_LABELS).forEach(function (k) {
      if (comps[k] == null) return;
      var pct = Math.max(0, Math.min(1, comps[k])) * 100;
      html +=
        '<div class="comp-row"><div class="top"><span class="name">' + COMP_LABELS[k] + "</span>" +
        '<span class="val">' + Number(comps[k]).toFixed(3) + "</span></div>" +
        '<div class="comp-track"><div class="comp-fill" style="width:' + pct.toFixed(1) + '%"></div></div></div>';
    });
    html += "</div>";
    html +=
      '<div class="mult-grid">' +
        '<div class="mult"><div class="k">Base</div><div class="v">' + esc(row.base) + "</div></div>" +
        '<div class="mult"><div class="k">Gate ×</div><div class="v">×' + esc(row.gate_mult) + "</div></div>" +
        '<div class="mult"><div class="k">Behavior ×</div><div class="v">×' + esc(row.behavior_mult) + "</div></div>" +
        '<div class="mult"><div class="k">Honeypot</div><div class="v" style="color:' +
          (row.honeypot ? "var(--danger)" : "var(--good)") + '">' + (row.honeypot ? "flagged" : "clean") + "</div></div>" +
      "</div>";
    var notes = (row.gate_reasons || []).concat(row.behavior_facts || []);
    if (notes.length) {
      html += '<div class="reasons-list"><ul>' + notes.map(function (n) { return "<li>" + esc(n) + "</li>"; }).join("") + "</ul></div>";
    }
    return html + "</div>";
  }

  function detailProfileHtml(raw) {
    if (!raw) return '<div class="detail-section"><p class="note">Profile detail unavailable for this candidate.</p></div>';
    var p = raw.profile || {};
    var html = "";
    if (p.headline || p.summary) {
      html += '<div class="detail-section"><h4>Summary</h4>';
      if (p.headline) html += '<p class="detail-headline">' + esc(p.headline) + "</p>";
      if (p.summary) html += '<p class="detail-summary">' + esc(p.summary) + "</p>";
      html += "</div>";
    }
    var skills = raw.skills || [];
    if (skills.length) {
      html += '<div class="detail-section"><h4>Skills</h4><div class="chip-row">';
      skills.forEach(function (s) {
        html += '<span class="skill-chip">' + esc(s.name) + (s.proficiency ? "<small> · " + esc(s.proficiency) + "</small>" : "") + "</span>";
      });
      html += "</div></div>";
    }
    var hist = raw.career_history || [];
    if (hist.length) {
      html += '<div class="detail-section"><h4>Experience</h4>';
      hist.forEach(function (h) {
        var when = (h.start_date || "") + (h.end_date ? " – " + h.end_date : (h.is_current ? " – present" : ""));
        var dur = yrsFromMonths(h.duration_months);
        html += '<div class="career-row"><div><strong>' + esc(h.title || "—") + "</strong> · " + esc(h.company || "") + "</div>" +
          '<div class="muted-cell">' + esc(when) + (dur ? " · " + dur : "") + (h.industry ? " · " + esc(h.industry) : "") + "</div></div>";
      });
      html += "</div>";
    }
    var edu = raw.education || [];
    if (edu.length) {
      html += '<div class="detail-section"><h4>Education</h4>';
      edu.forEach(function (e) {
        html += '<div class="career-row"><div><strong>' + esc(e.degree || "—") + (e.field_of_study ? ", " + esc(e.field_of_study) : "") + "</strong></div>" +
          '<div class="muted-cell">' + esc(e.institution || "") + (e.tier ? " · " + esc(e.tier) : "") + "</div></div>";
      });
      html += "</div>";
    }
    var sig = raw.redrob_signals || {};
    var kvs = [];
    function pct(label, v) { if (typeof v === "number" && v >= 0) kvs.push([label, Math.round(v * 100) + "%"]); }
    pct("Recruiter response", sig.recruiter_response_rate);
    pct("Interview completion", sig.interview_completion_rate);
    if (sig.last_active_date) kvs.push(["Last active", String(sig.last_active_date)]);
    kvs.push(["Open to work", sig.open_to_work_flag ? "yes" : "no"]);
    if (typeof sig.notice_period_days === "number" && sig.notice_period_days >= 0) kvs.push(["Notice", sig.notice_period_days + " days"]);
    if (typeof sig.saved_by_recruiters_30d === "number" && sig.saved_by_recruiters_30d >= 0) kvs.push(["Recruiter saves (30d)", String(sig.saved_by_recruiters_30d)]);
    var verified = [sig.verified_email ? "email" : "", sig.verified_phone ? "phone" : ""].filter(Boolean).join(" + ");
    kvs.push(["Verified", verified || "no"]);
    html += '<div class="detail-section"><h4>Signals</h4><div class="signal-grid">';
    kvs.forEach(function (kv) { html += '<div class="signal"><span class="sk">' + esc(kv[0]) + '</span><span class="sv">' + esc(kv[1]) + "</span></div>"; });
    html += "</div></div>";
    return html;
  }

  function renderDetail(row, raw) {
    var p = (raw && raw.profile) || {};
    var name = p.anonymized_name || row.candidate_id;
    var head =
      '<div class="detail-head">' +
        '<div class="detail-rank">#' + row.rank + "</div>" +
        '<div class="detail-id"><div class="detail-name">' + esc(name) + "</div>" +
          '<div class="detail-cid">' + esc(row.candidate_id) + " · score " + esc(row.score) + "</div></div>" +
        (row.honeypot ? '<span class="hp-badge">⚠ honeypot</span>' : "") +
      "</div>" +
      '<div class="detail-sub">' + esc(row.title || p.current_title || "—") +
        (p.current_company ? " @ " + esc(p.current_company) : "") +
        (p.current_industry ? " · " + esc(p.current_industry) : "") + "</div>" +
      '<div class="detail-sub muted-cell">' + esc(p.location || p.country || "") +
        (row.yrs != null ? " · " + row.yrs + " yrs experience" : "") + "</div>";
    $("#detail-body").innerHTML = head + detailScoreHtml(row) + detailProfileHtml(raw);
  }

  function openDetail(cid) {
    if (!cid || !state.lastRows) return;
    var row = null;
    for (var i = 0; i < state.lastRows.length; i++) {
      if (state.lastRows[i].candidate_id === cid) { row = state.lastRows[i]; break; }
    }
    if (!row) return;
    $("#detail-body").innerHTML = '<p class="note">Loading…</p>';
    $("#detail-modal").hidden = false;
    document.body.classList.add("modal-open");
    ensureRaws().then(function (map) { renderDetail(row, map ? map[cid] : null); });
  }

  function closeDetail() {
    $("#detail-modal").hidden = true;
    document.body.classList.remove("modal-open");
  }

  function initDetailModal() {
    var modal = $("#detail-modal");
    if (!modal) return;
    $("#detail-close").addEventListener("click", closeDetail);
    modal.addEventListener("click", function (e) { if (e.target === modal) closeDetail(); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape" && !modal.hidden) closeDetail(); });
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

  /* ------------------------------------------------------------------ *
   *  JD-weight sliders (Discovery: re-rank against your own priorities)
   * ------------------------------------------------------------------ */
  function roundW(v) { return Math.round(v * 1000) / 1000; }

  function weightsEqualDefaults(a, b) {
    if (!a || !b) return true;
    var keys = Object.keys(b);
    for (var i = 0; i < keys.length; i++) {
      if (Math.abs((a[keys[i]] || 0) - (b[keys[i]] || 0)) > 1e-6) return false;
    }
    return true;
  }

  function activeSkipGates() {
    return Object.keys(state.skipGates).filter(function (k) { return state.skipGates[k]; });
  }

  function refreshWeightsStatus() {
    var sum = 0;
    var keys = Object.keys(state.weights || {});
    keys.forEach(function (k) { sum += state.weights[k]; });
    var skipCount = activeSkipGates().length;
    var skipBit = skipCount ? " · " + skipCount + " gate" + (skipCount === 1 ? "" : "s") + " off" : "";
    $("#weights-sum").textContent = "Σ " + sum.toFixed(3) + " (auto-normalized at scoring)" + skipBit;
    var modified = !weightsEqualDefaults(state.weights, state.defaultWeights) || skipCount > 0;
    var pill = $("#weights-status");
    pill.textContent = modified ? "custom" : "JD defaults";
    pill.classList.toggle("on", modified);
  }

  function renderSliders() {
    var host = $("#sliders");
    host.innerHTML = "";
    Object.keys(COMP_LABELS).forEach(function (k) {
      var v = state.weights[k] != null ? state.weights[k] : 0;
      var row = document.createElement("div");
      row.className = "slider-row";
      row.innerHTML =
        '<div class="slider-head">' +
          '<span class="slider-label">' + COMP_LABELS[k] + '</span>' +
          '<span class="slider-val" data-val="' + k + '">' + v.toFixed(2) + '</span>' +
        '</div>' +
        '<input type="range" min="0" max="0.5" step="0.01" value="' + v + '" data-key="' + k + '" />';
      host.appendChild(row);
    });
    $$("#sliders input[type=range]").forEach(function (inp) {
      inp.addEventListener("input", function () {
        var k = inp.getAttribute("data-key");
        var v = parseFloat(inp.value);
        state.weights[k] = roundW(v);
        $('[data-val="' + k + '"]').textContent = v.toFixed(2);
        highlightPreset(null);   // manual edit = no preset active
        refreshWeightsStatus();
        scheduleRerank();
      });
    });
  }

  var rerankTimer = null;
  function scheduleRerank() {
    // Any control change re-ranks. In upload mode with no file, rank() shows an
    // error and returns; in sample mode there is always something to re-rank.
    clearTimeout(rerankTimer);
    rerankTimer = setTimeout(function () { rank({ quiet: true }); }, 320);
  }

  function resetWeights() {
    applyPreset("default");
  }

  function renderGates() {
    var host = $("#gates");
    host.innerHTML = "";
    state.gates.forEach(function (g) {
      var off = !!state.skipGates[g.key];
      var row = document.createElement("label");
      row.className = "gate-row" + (off ? " off" : "");
      row.setAttribute("data-key", g.key);
      row.innerHTML =
        '<div class="gate-text">' +
          '<div class="gate-label">' + esc(g.label) + '</div>' +
          '<div class="gate-desc">' + esc(g.desc) + '</div>' +
        '</div>' +
        '<span class="toggle" role="switch" aria-checked="' + (!off) + '">' +
          '<input type="checkbox" data-key="' + esc(g.key) + '" ' + (off ? "" : "checked") + ' />' +
          '<span class="toggle-track"><span class="toggle-thumb"></span></span>' +
        '</span>';
      host.appendChild(row);
    });
    $$("#gates input[type=checkbox]").forEach(function (cb) {
      cb.addEventListener("change", function () {
        var k = cb.getAttribute("data-key");
        state.skipGates[k] = !cb.checked;
        cb.closest(".gate-row").classList.toggle("off", !cb.checked);
        refreshWeightsStatus();
        scheduleRerank();
      });
    });
  }

  function applyPreset(name) {
    var p = PRESETS[name];
    if (!p) return;
    state.weights = Object.assign({}, p.weights);
    state.skipGates = {};
    p.skip.forEach(function (k) { state.skipGates[k] = true; });
    renderSliders();
    renderGates();
    refreshWeightsStatus();
    highlightPreset(name);
    scheduleRerank();
  }

  function highlightPreset(name) {
    $$("#presets .chip").forEach(function (b) {
      b.classList.toggle("on", b.getAttribute("data-preset") === name);
    });
  }

  function initPresets() {
    $$("#presets .chip").forEach(function (b) {
      b.addEventListener("click", function () { applyPreset(b.getAttribute("data-preset")); });
    });
  }

  function initWeights() {
    Promise.all([
      fetch("/api/weights").then(function (r) { return r.json(); }),
      fetch("/api/gates").then(function (r) { return r.json(); })
    ]).then(function (results) {
      state.defaultWeights = results[0].default_weights || {};
      state.weights = Object.assign({}, state.defaultWeights);
      state.gates = results[1].gates || [];
      renderSliders();
      renderGates();
      refreshWeightsStatus();
      highlightPreset("default");
    }).catch(function () {
      // network/health issue — leave panels empty rather than crash the page
      $("#sliders").innerHTML = '<p class="note">Could not load default weights.</p>';
      $("#gates").innerHTML = '<p class="note">Could not load gate catalog.</p>';
    });
    $("#reset-weights").addEventListener("click", resetWeights);
    initPresets();
  }

  /* ------------------------------------------------------------------ *
   *  Discovery: facets + experience curve + similar-candidates
   * ------------------------------------------------------------------ */
  function rankBodyForCurrentSource() {
    if (state.mode === "upload") {
      if (!state.fileText) { showError("Choose or drop a JSONL file first."); return null; }
      return { jsonl: state.fileText, use_sample: false };
    }
    return { use_sample: true };
  }

  function findSimilar(cid) {
    var body = rankBodyForCurrentSource();
    if (!body) return;
    body.candidate_id = cid;
    body.limit = 10;
    setLoading(true);
    fetch("/api/similar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (res) {
      return res.json().then(function (data) {
        if (!res.ok) throw new Error(data.detail || "Similar lookup failed");
        return data;
      });
    }).then(function (data) {
      setLoading(false);
      renderSimilar(data);
    }).catch(function (err) {
      setLoading(false);
      showError(err.message || "Similar lookup failed");
    });
  }

  function renderSimilar(data) {
    var panel = $("#similar-panel");
    var target = data.target || {};
    $("#similar-target").textContent = target.candidate_id + " (" + (target.title || "—") + ")";
    var body = $("#similar-body");
    if (!data.rows || !data.rows.length) {
      body.innerHTML = '<p class="note">No similar candidates found in this batch.</p>';
    } else {
      var rows = data.rows.map(function (r) {
        var note = r.note ? '<div class="sim-note">' + esc(r.note) + '</div>' : "";
        return '<div class="similar-row">' +
          '<span class="sim-rank">' + r.rank + '</span>' +
          '<div class="sim-main">' +
            '<div><strong>' + esc(r.candidate_id) + '</strong> · ' +
            '<span class="muted-cell">' + esc(r.title || "—") + ' · ' + esc(r.country || "") + '</span></div>' +
            note +
          '</div>' +
          '<span class="sim-score">cos ' + r.similarity.toFixed(3) + '</span>' +
        '</div>';
      }).join("");
      body.innerHTML = rows;
    }
    panel.style.display = "";
    panel.open = true;
    panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function extractFacetsFromProse() {
    var text = $("#prose-input").value || "";
    if (text.trim().length < 50) {
      showError("Paste at least a short JD before extracting facets.");
      return;
    }
    setLoading(true);
    fetch("/api/facets_from_text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok) throw new Error(d.detail || "Extraction failed");
        return d;
      });
    }).then(function (d) {
      setLoading(false);
      $("#facets-input").value = (d.facets || []).join("\n");
      updateFacetCount();
    }).catch(function (err) { setLoading(false); showError(err.message); });
  }

  function searchPool() {
    var cid = ($("#pool-cid").value || "").trim();
    var limit = parseInt($("#pool-limit").value, 10) || 20;
    if (!cid) { showError("Enter a candidate_id."); return; }
    setLoading(true);
    fetch("/api/similar_pool", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ candidate_id: cid, limit: limit })
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok) throw new Error(d.detail || "Search failed");
        return d;
      });
    }).then(function (d) {
      setLoading(false);
      renderPoolResults(d);
    }).catch(function (err) { setLoading(false); showError(err.message); });
  }

  function renderPoolResults(d) {
    var box = $("#pool-results");
    if (!d.rows || !d.rows.length) {
      box.innerHTML = '<p class="note">No matches.</p>';
      return;
    }
    box.innerHTML =
      '<p class="note">Top ' + d.rows.length + ' look-alikes for ' + esc(d.target) +
      ' across ' + d.pool_size.toLocaleString("en-US") + ' candidates.</p>' +
      d.rows.map(function (r) {
        return '<div class="similar-row">' +
          '<span class="sim-rank">' + r.rank + '</span>' +
          '<div class="sim-main"><strong>' + esc(r.candidate_id) + '</strong></div>' +
          '<span class="sim-score">cos ' + r.similarity.toFixed(3) + '</span>' +
        '</div>';
      }).join("");
  }

  function initDiscovery() {
    Promise.all([
      fetch("/api/facets").then(function (r) { return r.json(); }),
      fetch("/api/experience").then(function (r) { return r.json(); })
    ]).then(function (results) {
      var f = results[0];
      $("#facets-input").value = (f.facets || []).join("\n");
      updateFacetCount();
      var e = results[1];
      renderExperienceGrid(e);
    }).catch(function () {
      $("#facets-input").placeholder = "Could not load JD facets.";
    });

    $("#facets-input").addEventListener("input", updateFacetCount);

    $("#save-facets").addEventListener("click", function () {
      var lines = $("#facets-input").value.split("\n").map(function (l) { return l.trim(); }).filter(Boolean);
      if (lines.length < 3 || lines.length > 20) {
        showError("Need 3–20 facets, one per line. You have " + lines.length + ".");
        return;
      }
      setLoading(true);
      fetch("/api/facets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ facets: lines })
      }).then(function (r) {
        return r.json().then(function (d) {
          if (!r.ok) throw new Error(d.detail || "Save failed");
          return d;
        });
      }).then(function () {
        $("#discovery-status").textContent = "Custom JD";
        setLoading(false);
        rank({ quiet: false });
      }).catch(function (err) { setLoading(false); showError(err.message); });
    });

    $("#save-experience").addEventListener("click", function () {
      var fields = ["band_min", "ideal_min", "ideal_max", "band_max"];
      var payload = {};
      for (var i = 0; i < fields.length; i++) {
        var v = parseFloat($("#exp-" + fields[i]).value);
        if (isNaN(v) || v < 0) { showError("All experience values must be non-negative numbers."); return; }
        payload[fields[i]] = v;
      }
      setLoading(true);
      fetch("/api/experience", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }).then(function (r) {
        return r.json().then(function (d) {
          if (!r.ok) throw new Error(d.detail || "Save failed");
          return d;
        });
      }).then(function () {
        $("#discovery-status").textContent = "Custom JD";
        setLoading(false);
        rank({ quiet: false });
      }).catch(function (err) { setLoading(false); showError(err.message); });
    });

    $("#extract-facets").addEventListener("click", extractFacetsFromProse);
    $("#pool-search").addEventListener("click", searchPool);

    $("#reset-discovery").addEventListener("click", function () {
      setLoading(true);
      fetch("/api/reset", { method: "POST" }).then(function () {
        return Promise.all([
          fetch("/api/facets").then(function (r) { return r.json(); }),
          fetch("/api/experience").then(function (r) { return r.json(); })
        ]);
      }).then(function (results) {
        $("#facets-input").value = (results[0].facets || []).join("\n");
        updateFacetCount();
        renderExperienceGrid(results[1]);
        $("#discovery-status").textContent = "Defaults loaded";
        setLoading(false);
        rank({ quiet: false });
      }).catch(function (err) { setLoading(false); showError(err.message); });
    });
  }

  var EXP_FIELDS = ["band_min", "ideal_min", "ideal_max", "band_max"];

  function renderExperienceGrid(e) {
    var labels = { band_min: "Band min", ideal_min: "Ideal min", ideal_max: "Ideal max", band_max: "Band max" };
    $("#experience-grid").innerHTML = EXP_FIELDS.map(function (f) {
      return '<div class="exp-row"><label>' + labels[f] + '</label>' +
        '<input type="number" id="exp-' + f + '" min="0" step="0.5" value="' + e[f] + '" /></div>';
    }).join("");
    $$("#experience-grid input[type=number]").forEach(function (inp) {
      inp.addEventListener("input", renderExperienceBand);
    });
    renderExperienceBand();
  }

  function fmtY(x) { return (Math.round(x * 10) / 10).toString().replace(/\.0$/, ""); }

  function expVals() {
    return EXP_FIELDS.map(function (f) {
      var el = $("#exp-" + f);
      var v = el ? parseFloat(el.value) : NaN;
      return isNaN(v) ? 0 : v;
    });
  }

  // Live picture of the experience curve: shade the acceptable [band_min,band_max]
  // span and the brighter ideal [ideal_min,ideal_max] zone on a 0..band_max track.
  function renderExperienceBand() {
    var host = $("#exp-band");
    if (!host) return;
    var v = expVals();
    var bmin = v[0], imin = v[1], imax = v[2], bmax = v[3];
    var top = Math.max(bmax, imax, bmin, 1) * 1.12;   // headroom so band_max isn't at the edge
    function pct(x) { return Math.max(0, Math.min(100, (x / top) * 100)); }
    var band = host.querySelector(".exp-band-fill");
    var ideal = host.querySelector(".exp-ideal-fill");
    band.style.left = pct(bmin) + "%";
    band.style.width = Math.max(0, pct(bmax) - pct(bmin)) + "%";
    ideal.style.left = pct(imin) + "%";
    ideal.style.width = Math.max(0, pct(imax) - pct(imin)) + "%";
    var maxEl = document.getElementById("exp-scale-max");
    if (maxEl) maxEl.textContent = fmtY(top) + " yrs";
    host.querySelector(".exp-caption").innerHTML =
      'Acceptable <strong>' + fmtY(bmin) + "–" + fmtY(bmax) + "</strong> yrs · ideal <strong>" +
      fmtY(imin) + "–" + fmtY(imax) + "</strong> yrs";
  }

  function updateFacetCount() {
    var el = $("#facet-count");
    if (!el) return;
    var n = ($("#facets-input").value || "").split("\n").map(function (l) { return l.trim(); }).filter(Boolean).length;
    el.textContent = n + (n === 1 ? " facet" : " facets");
  }

  /* ------------------------------------------------------------------ */
  document.addEventListener("DOMContentLoaded", function () {
    initBeam();
    initHeroStats();
    initReveal();
    initModeToggle();
    initUpload();
    setMode("sample");
    initWeights();
    initDiscovery();
    initDetailModal();
    $("#rank-btn").addEventListener("click", function () { rank(); });
    $("#download-btn").addEventListener("click", downloadCsv);
    rank();   // auto-rank the sample on load: table is live and every later tweak re-ranks
  });
})();
