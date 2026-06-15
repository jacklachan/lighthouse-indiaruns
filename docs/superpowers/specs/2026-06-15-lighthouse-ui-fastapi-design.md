# Lighthouse Sandbox UI — FastAPI + Custom Animated Frontend (design)

**Date:** 2026-06-15
**Status:** Approved (user said "Approved start building")
**Topic:** Replace the Streamlit demo with a dark/premium/animated custom frontend served by FastAPI, deployed on the HF Docker Space `Auenchanters/lighthouse`.

## Goal

Improve the public demo UI for Lighthouse (Redrob India Runs Track-1 submission). The
current sandbox is a functional-but-plain Streamlit app. Replace it with a bespoke
dark, premium, animated single-page experience — a hero that sells the idea and a full
ranker tool — without changing any ranking behavior.

## Hard invariants (do-no-harm)

1. **The ranker core (`lighthouse/`) is untouched.** This is purely a presentation/
   serving layer that calls the same functions the Streamlit app called.
2. **`submission.csv` and the rank-time `requirements.txt` stay byte-identical.** New
   serving deps (`fastapi`, `uvicorn`) go only in the app/Space requirements, never in
   the frozen rank-time runtime.
3. **The functional surface is preserved exactly:** sample/upload candidates → rank →
   results table (rank, id, score, title, country, yrs, honeypot flag, grounded
   reasoning) → top-candidate component breakdown → CSV download. Numbers must match
   the Streamlit app (same normalization: divide final scores by max, round 6).

## Architecture

A small **FastAPI** app (`app/server.py`) that:

- serves the static frontend from `app/web/` (mounted at `/static`, index at `/`),
- exposes a thin JSON API that wraps the existing ranker.

The browser reads any uploaded file client-side (`FileReader`) and POSTs its text in a
JSON body, so the server needs **no multipart dependency** — only `fastapi` + `uvicorn`.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | the single-page UI (`app/web/index.html`) |
| `GET` | `/api/health` | liveness + artifact availability |
| `GET` | `/api/sample` | preloaded sample candidates (JSONL text) |
| `POST` | `/api/rank` | body `{jsonl?, use_sample?}` → ranked rows + summary + top breakdown + weights |

### Server data flow (mirrors `app/app.py` exactly)

1. Load `jd_rubric.json`, `jd_facet_emb.npy`, and `semantic_p5/p95` once at startup
   (`_facets_and_rubric` + `_build_art` with an empty precomputed set → every candidate
   is encoded on the fly by `bge-small`, same as today).
2. `_parse_jsonl(text)` → `raws` (cap 100, skip blank/bad lines).
3. `ranker.score_all(raws, art)` → records.
4. Normalize: `mx = max(final_score)`; divide each by `mx`, round 6 (identical to app).
5. `ranker.rank_records(records, top=len(records))` → assign ranks.
6. Build rows (rank, candidate_id, score, title, country, yrs, honeypot bool,
   `reasoning.generate(raw, rubric, rec)`).
7. Summary (`ranked`, `honeypots`, `top_score`) + `breakdown` of the top record
   (candidate_id, components, base, gate_mult, gate_reasons, behavior_mult, honeypot)
   + component `weights`.

### Error handling

- Empty/invalid input → `400` with a human message; frontend shows a styled error.
- Ranking exception → `500` with the message; frontend shows it without crashing.

## Frontend (dark, premium, animated)

Vanilla HTML + **Tailwind Play CDN** (utility layout) + a bespoke `styles.css` (glass,
gradients, the lighthouse-beam motif) + **anime.js (vendored** to `app/web/vendor/`, no
CDN dependency for the motion engine).

**Aesthetic:** near-black backdrop, indigo→amber "beam" accent, glassmorphic cards,
reui-style components (buttons/badges/cards/table) rebuilt in Tailwind/CSS.

**Sections:**

1. **Hero** — headline *"Keyword filters surface the loudest profiles. Lighthouse
   surfaces the right ones."*, sub-line, CTAs (→ tool, → GitHub), a sweeping
   lighthouse-beam animation, stat chips (100K pool · 0 honeypots in top-100 ·
   5-component score).
2. **The traps → the defense** — glass cards (keyword-stuffers, behavioral twins,
   honeypots, plain-language strong) with scroll-reveal.
3. **The pipeline** — the 5 components + gates + honeypot-zeroing + reasoning as an
   animated horizontal flow.
4. **The tool** — sample/upload toggle (drag-drop), "Rank candidates" button, animated
   loading beam, then: count-up metric cards, a staggered results table with honeypot
   badges + per-row reasoning, an expandable top-candidate breakdown with animated
   component bars (weighted), and a Download CSV button (CSV built client-side from the
   JSON: columns candidate_id, rank, score, reasoning).
5. **Footer** — credit + links.

**Motion (anime.js):** hero beam sweep + pulse; scroll-reveal (IntersectionObserver →
staggered translateY/opacity); metric count-ups; staggered table-row entrance; animated
component bars (width 0→value).

## Files

- **New:** `app/server.py`, `app/web/index.html`, `app/web/styles.css`,
  `app/web/app.js`, `app/web/vendor/anime.min.js`.
- **Changed:** `Dockerfile` (CMD → `uvicorn app.server:app`), `app/requirements.txt`
  (+`fastapi`, `uvicorn[standard]`; drop `streamlit`), `app/README.md` (Docker SDK note),
  `README.md` (repo `app/` row), `requirements-dev.txt` (+`fastapi`,`httpx` for the test).
- **Untouched:** all of `lighthouse/`, `submission.csv`, the rank-time `requirements.txt`,
  `artifacts/*`.

## Testing & verification

- `tests/test_server.py` — FastAPI `TestClient`, `importorskip("fastapi")`. Covers
  `GET /`, `GET /api/health`, `GET /api/sample`, and `POST /api/rank` with empty input →
  `400`. The encoder path (`/api/rank` happy path) needs `sentence-transformers` + a
  model download, so it is exercised in **local manual verification**, not CI.
- Full local gate: `ruff check .`, `black --check .`, `mypy lighthouse`, `pytest -q`,
  `git diff --exit-code submission.csv requirements.txt`.
- Local run: `uvicorn app.server:app --port 7860`, exercise sample-rank + upload + CSV in
  a browser.
- Deploy: sync `app/` + `Dockerfile` + `requirements.txt` + `README.md` into the HF Space
  clone, push, confirm clean build/run logs + HTTP 200 + a working `/api/rank`.

## Why this approach

Building in the GitHub repo as the single source of truth and deploying the identical
artifact to the Space removes the repo↔Space drift that caused the Task-1 ImportError.
FastAPI + vanilla frontend avoids a JS build toolchain (no node_modules in the Docker
image) while giving full control over the premium look the user asked for.
