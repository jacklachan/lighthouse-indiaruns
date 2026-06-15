---
title: Lighthouse Candidate Ranker
emoji: 🔦
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🔦 Lighthouse — HuggingFace Spaces sandbox

A live demo of the **Lighthouse** recruiter-grade candidate ranker (Redrob India Runs
challenge). A custom dark, animated frontend (FastAPI + vanilla JS + anime.js) lets you
use the preloaded 100-candidate sample or upload your own JSONL (≤100) and watch
Lighthouse rank them end-to-end on CPU — five-component fit score, JD hard-negative
gates, behavioral modifier, honeypot zeroing, and grounded reasoning — then download the
ranked CSV.

## How it works here vs. the official run

- **This Space:** uploaded candidates are not precomputed, so the small `BAAI/bge-small-en-v1.5`
  encoder runs at request time (a few seconds for ≤100 rows). This is the demo path.
- **Official 100K run:** embeddings are precomputed offline; `rank.py` then runs CPU-only,
  no network, < 5 minutes (see the GitHub repo).

## Architecture

- `app/server.py` — a small **FastAPI** app that serves the static frontend (`app/web/`)
  and a thin JSON API (`/api/sample`, `/api/rank`, `/api/health`) wrapping the unchanged
  `lighthouse/` ranker. The browser reads any uploaded file client-side and POSTs its
  text, so the server needs no multipart dependency.
- `app/web/` — the single-page UI: `index.html`, `styles.css`, `app.js`, and vendored
  `vendor/anime.min.js` (the only JS dependency).

## Deploy this Space yourself

1. Create a new **Docker** Space on HuggingFace.
2. Push this repository to the Space (it needs the `Dockerfile`, `app/` (server + web +
   `sample_candidates.jsonl` + `requirements.txt`), the `lighthouse/` package, and
   `artifacts/jd_rubric.json`, `artifacts/jd_facet_emb.npy`, `artifacts/precompute_meta.json`).
3. The Docker image runs `uvicorn app.server:app` on port 7860 and downloads `bge-small`
   on first request.

Run locally: `uvicorn app.server:app --host 0.0.0.0 --port 7860`.

The full architecture, evaluation, and reproduce commands are in the GitHub repository's
top-level `README.md`.
