# Changelog

All notable changes to this project are documented here. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed (Demo sample — alignment with the official submission)
- **`app/sample_candidates.jsonl` rebuilt** so the live HF Space ranks the
  actual top 80 of the 100K submission plus 20 visible traps (10 keyword-
  stuffers, 10 honeypots). The demo's ranks 1–80 are now byte-identical to
  `submission.csv` rows 1–80, so the live UI is an in-browser replay of the
  official ranking rather than a separate stress-test batch.
- **`non_technical_role` gate penalty tightened 0.25 → 0.05** in
  `artifacts/jd_rubric.json`. Non-engineering current-titles now sink ~40×
  below the lowest real engineer. Submission ranking is unchanged (real
  engineers never trip the gate).
- **`artifacts/sample_cand_emb.npy` regenerated** from the new sample via
  `scripts/build_sample_emb.py`.
- **`docs/sample_evidence.md` (new)** records the measured top-/bottom-band
  composition and reproduce commands.
- **README, in-app banner, and deck Results slide** rewritten to describe the
  new sample composition. Test count refs updated 63/124 → 127, coverage
  refs 86% → 87%.

### Added (Discovery surface — final pass)
- **`POST /api/similar_pool` — 100K look-alike.** Searches the full
  precomputed candidate-embedding pool (`artifacts/cand_emb.npy`) by cosine
  similarity. Doesn't require any upload, returns IDs + cosines. UI surfaces
  this via a new "🌐 Search the 100K pool" panel.
- **`POST /api/facets_from_text` — prose JD parser.** Takes raw JD prose and
  emits 3–15 facet sentences. Splits on bullet markers (`-`, `•`, numbered)
  with sentence-split fallback. Discovery UI shows a collapsible "📝 Paste a
  JD instead" textarea that auto-fills the facets editor.
- **`artifacts/sandbox_overrides.json` is now gitignored.** Live UI state
  (facet/experience overrides) persists across server restarts without
  leaking into commits.
- **Encoder pre-warm.** Lifespan handler launches a daemon thread that
  instantiates `SentenceTransformer` at server startup, so the first
  `/api/facets` call doesn't pay the ~10 s instantiation cost.
- **Per-batch embedding cache for `/api/similar`.** LRU(3) keyed by the
  sha1 of the JSONL text. Clicking 📡 on multiple rows of the same batch
  encodes only on the first click.
- **Similarity notes on look-alike rows.** Each `/api/similar` row now
  carries a grounded one-liner — shared skills (capped at 3), YoE delta,
  country mismatch — surfaced in the UI under each row.
- **Migrated from deprecated `@app.on_event("startup")` to FastAPI
  lifespan context manager.**

### Added (Discovery surface — earlier additions)
- **`POST /api/similar` — look-alike search.** Given a `candidate_id` from the
  current batch, returns the top-N most similar candidates by embedding cosine.
  Reuses the same encoder as `/api/rank`. Surfaced in the UI as a `📡` button
  next to each row in the ranked table.
- **`GET`/`POST /api/facets` — JD facet editing.** The recruiter can replace
  the 10 hand-authored facets with their own, and optionally pass a per-facet
  importance vector (`facet_weights`). The server re-encodes via the same
  sentence-transformer used for small-sample candidate embedding and swaps the
  facet matrix into live state. Population semantic bounds are dropped on
  edit so the in-batch fallback re-normalizes against the new facets.
- **`GET`/`POST /api/experience` — counterfactual experience curve.** Override
  `band_min`/`ideal_min`/`ideal_max`/`band_max` in the live rubric. Lets the
  judge ask "what if 3–7 yrs instead of 5–9 yrs?" without editing the artifact.
- **`POST /api/reset`** restores the rubric, facets, facet weights, and
  population semantic bounds to the committed artifact defaults — used by the
  UI "↺ Reset JD" button.
- **`raw_semantic_fit` accepts an optional `facet_weights` argument.** Each
  facet's cosine is multiplied by its weight before max + top-K aggregation,
  so a high-importance facet that the candidate hits hard can outvote a
  low-importance facet that they happen to match perfectly. Default `None`
  keeps every facet at weight 1.0 — identical to the previous path, so
  `rank.py` output is unchanged.

### Changed (engine — affects 100K scoring; regenerate `submission.csv`)
- **New `gate_unbacked_expertise` (0.65×).** Fires when a candidate lists 5+
  advanced/expert skills but none of them appear in any career role
  title/description or in the profile summary — the canonical keyword-stuffer
  pattern. Multiplicative penalty, not a zero (honest-but-sparse profiles
  with only 1–4 advanced skills are unaffected; if even one expert claim
  shows up in career text the gate stays silent). Bypass-able via the
  existing `skip_gates` Discovery hook.

### Changed (idea-level improvements — affect 100K scoring; regenerate `submission.csv`)
- **`career_evidence` is now recency-weighted.** The "consistently strong career"
  half (mean term, 35% of the score) now uses an exponential decay with a 4-yr
  half-life anchored on each role's end date. A current candidate with two
  recent ranking-system roles now beats an otherwise-identical candidate whose
  same work ended in 2014. The "single foundational role" half (max term, 65%)
  is left undecayed so veterans aren't punished for tenure — one shipped
  system from 2014 is still real evidence.
- **`semantic_fit` aggregates top-K facets instead of mean-over-all.** Old:
  `0.6*max + 0.4*mean(all 10 facets)`. New: `0.6*max + 0.4*mean(top 4 facets)`.
  A retrieval specialist hitting 3 facets strongly was previously dragged
  down by the 7 weak facets in their tail (e.g. the JD's "scrappy product
  engineering" facet). Top-K keeps depth visible without dropping the max
  signal. Specialist gain on a synthetic [0.9×3, 0.1×7] candidate: ~0.68 → ~0.82.
- **`reasoning._lift_clause` ("what would change this rank").** When the row
  directly above is within a closable deficit, the reasoning now surfaces the
  cheapest component bump that would overtake them, computed from the actual
  per-component leverage `w_i × gate_mult × behavior_mult`. Reads as
  *"One step from rank 11: +0.04 role_coherence would overtake CAND_XXX."*
  Wired through `app/server.py` via a new `above` field on the reasoning
  context. Honeypots get no lift clause (their final is zeroed).

### Fixed (second fairness pass — affects 100K scoring; regenerate `submission.csv`)
- `reasoning._contrastive_clause` now ranks components by *weighted contribution*
  (`gap × component_weight`) rather than the raw per-component delta. A 0.20 gap
  on `experience_fit` (w=0.10) contributes 0.020 to the final score; a 0.10 gap
  on `role_coherence` (w=0.26) contributes 0.026 — the latter is the real reason
  one candidate edges its peer, but the previous code named the former. The
  cutoff is also moved from raw 0.08 to contribution 0.02, so the clause only
  fires when the gap actually moved the score. (`rank.py` calls reasoning without
  context, so `submission.csv` is unaffected by this clause change; it only
  matters for the live sandbox where contrastive overlay is on.)
- `gate_research_only` tightened from `research >= 2 AND production == 0` to
  `research >= 3 AND production == 0 AND no_industry_product_role`. The
  industry-product check uses `is_services_company` plus a new academic-employer
  heuristic (IIT/IIM/IISc/"university"/"research lab"). Previously a published
  PhD ("phd" + "publication" = 2 research hits) at a real product company could
  trip the 0.55 multiplier just because their bio didn't literally use the word
  "production". Now only genuinely research-pure trajectories fire.
- `gate_services_only` ramped from a hard cliff at `services_fraction == 1.000`
  to a linear ramp on `[0.85, 1.000]`. 9/10 services + 1/10 product previously
  cleared the gate completely; now it sits ~2/3 of the way up the ramp
  (multiplier ~0.49 vs full-rubric 0.45). The 0.85 floor preserves the JD
  guidance that one+ real product role is rescuing.
- `gate_langchain_only_recent` now reads the candidate's profile summary as part
  of the "recent" text. A summary line like "Recently built LangChain prompt
  chains" is exactly the signal the gate is meant to catch even when no current
  role description repeats it.
- `_seniority_level` (used by `gate_title_chaser`) now matches keywords on
  word boundaries instead of plain substring. Pre-fix "leading product engineer"
  was scored as level 2 (`'lead' in title`), which could fake an escalation
  pattern across an otherwise-lateral career.
- `is_services_company` looks up the services-firm list by gate key instead of
  positional `hard_negatives[0]`, plus guards against an empty company name.
  The previous indexing was a latent bug waiting on a rubric reorder.
- `behavioral_modifier` no longer rewards a `last_active_date` that lies in the
  future (negative `days` ago) — the bonus branch was unreachable in clean data
  but synthetic profiles could game the +0.05 "active 0d ago" lift by stamping
  a far-future timestamp.
- `honeypot.detect` flags `last_active_date > REFERENCE_DATE` for the same
  reason it already flags `signup_date > REFERENCE_DATE`: a profile cannot have
  been active after the JD reference "now".
- `reasoning._concern_clause` now also surfaces `"slow average response"` as a
  concern. The behavioral facts emitted by the new `avg_response_time_hours`
  signal were never picked up by the reasoning filter.
- Updated `_near_miss_clause` services band to `[0.7, 0.85)` to match the new
  gate ramp, so "close call" no longer overlaps with "gate is dampening you".

### Added
- New sandbox UI: a custom dark/premium/animated frontend (`app/web/` — vanilla
  HTML/CSS/JS + vendored anime.js) served by a small FastAPI app (`app/server.py`) that
  wraps the unchanged `lighthouse/` ranker. Hero + story + pipeline sections and a full
  ranker tool (sample/upload → ranked table with grounded reasoning → top-candidate
  component breakdown → CSV download). JSON API: `/api/sample`, `/api/rank`, `/api/health`.
- FastAPI `TestClient` smoke tests (`tests/test_server.py`): static/index, health,
  sample, and input-validation (empty/invalid JSONL → 400). The encoder happy path is
  verified locally (needs `sentence-transformers` + a model download), not in CI.
- Continuous integration (`.github/workflows/ci.yml`): runs the test suite and the
  submission-format validator on every push and pull request to `main`.
- Project packaging and tool configuration (`pyproject.toml`): project metadata, optional
  dependency extras (`precompute`, `app`, `deck`, `dev`), and config for ruff, black,
  mypy, pytest, and coverage.
- Developer tooling manifest (`requirements-dev.txt`), kept separate from the minimal
  rank-time runtime.
- Project governance and policy docs: `LICENSE` (MIT), `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, and this changelog.
- Lint (ruff), format (black), and type (mypy) gates wired into CI, plus a coverage
  floor on the rank-time package (`fail_under = 85`).
- Expanded the test suite from 36 to 63 tests: `test_scoring.py` (normalize_semantic
  bounds/clipping plus a regression guard for the NumPy-2 degenerate-range crash),
  `test_loader.py` (defensive accessors, date parsing, malformed-line streaming, sparse
  profiles), and metric edge cases (zero-IDCG NDCG, zero-k precision). Coverage of the
  rank-time package rose from 78% to 86% (loader 53%→90%, scoring 79%→90%, metrics 100%).
- Vendored the 50-candidate published sample to `tests/fixtures/sample_candidates.json`
  so the full suite runs in CI without the untracked challenge bundle.
- README "How this maps to the brief" table — a rubric-to-repo index pointing each
  evaluated dimension at the file/evidence that satisfies it; refreshed the Tests
  section to cover the full gate (ruff/black/mypy/coverage) and 63 tests.

### Changed
- README: corrected the artifacts row (float16 `.npy`, no git-lfs) to match the
  reproduce note; both now agree the embeddings ship as a plain committed file.
- Documented the honeypot tenure-overflow thresholds (the `1.6x` overlap allowance and
  `+18mo` slack) so the rule reads as flagging impossibility, not concurrency; and
  clarified that the precomputed BM25 index is an optional lexical signal never loaded
  at rank-time. Comments only — no behavior change.

### Fixed
- `normalize_semantic` used the `ndarray.ptp()` method, which NumPy 2.0 removed (the repo
  pins `numpy==2.2.6`). This raised `AttributeError` in the degenerate-range branch
  (reachable in the small-N sandbox). Switched to `np.ptp(raw)` — numerically identical,
  no longer crashes. The 100K rank path never hits this branch, so `submission.csv` is
  unaffected.
- Type-only: added explicit `Optional` annotations on `None`-defaulted parameters in
  `scoring.py` / `ranker.py` (no runtime behavior change).
- **Fairness pass over the ranker (may shift individual scores in the 100K run; the
  qualitative headlines — top-1 still `CAND_0000031`, 0 honeypots in top-100, 0
  keyword-stuffers in top-25 — are preserved on the sample fixture):**
  - `behavioral_modifier` now guards `recruiter_response_rate >= 0` and
    `notice_period_days >= 0`. A `-1` sentinel in either field used to incorrectly
    apply the "low response" penalty or the "fast notice" bonus; both are now
    treated as neutral (matching the rubric's stated sentinel policy).
  - `gate_langchain_only_recent` now actually checks recency: wrapper terms only
    count when they appear in a role still active within the last 18 months. The
    rule field in `jd_rubric.json` always said this, but the code matched on
    full career text and ignored timestamps.
  - `classify_title` now lets a positive title term ("software engineer", "ml
    engineer", …) trump a negative substring when both are present in the same
    title — e.g. `"Marketing Software Engineer"` classifies as positive, not
    negative. A plain `"Marketing Manager"` (no positive term) still classifies
    as negative.
  - `honeypot.detect` no longer treats a missing `duration_months` field as
    "explicitly zero". The advanced/expert-with-zero-months rule now walks the
    raw skills list and requires the field to be present and equal to 0,
    so sparse-but-honest profiles are no longer at risk of a false positive.

### Notes
- The ranked output (`submission.csv`) and the rank-time runtime dependencies
  (`requirements.txt`) are unchanged — this release is repository-hardening only and does
  not affect ranking behavior.

## [0.1.0] — Submission baseline

### Added
- Five-component reasoning-first ranker (`lighthouse/`): semantic fit, role coherence,
  career evidence, experience fit, trust-weighted skills.
- JD-derived multiplicative hard-negative gates and an explainable honeypot filter.
- Fact-grounded reasoning generator and NDCG / MAP / P@k metrics.
- Offline precompute pipeline (`precompute.py`) and rank-time entry point (`rank.py`).
- Dual evaluation framework (`eval/`): self-labeled metrics plus a blind independent-label
  harness to guard against evaluation circularity.
- 36-test pytest suite, HuggingFace Spaces demo (`app/`), and submission deck (`deck/`).
