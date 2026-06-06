---
title: Lighthouse Candidate Ranker
emoji: 🔦
colorFrom: indigo
colorTo: yellow
sdk: streamlit
sdk_version: 1.40.2
app_file: app/app.py
pinned: false
license: mit
---

# 🔦 Lighthouse — HuggingFace Spaces sandbox

A live demo of the **Lighthouse** recruiter-grade candidate ranker (Redrob India Runs
challenge). Upload up to 100 candidates (or use the preloaded 100-candidate sample) and
watch Lighthouse rank them end-to-end on CPU — five-component fit score, JD hard-negative
gates, behavioral modifier, honeypot zeroing, and grounded reasoning — then download the
ranked CSV.

## How it works here vs. the official run

- **This Space:** uploaded candidates are not precomputed, so the small `BAAI/bge-small-en-v1.5`
  encoder runs at request time (a few seconds for ≤100 rows). This is the demo path.
- **Official 100K run:** embeddings are precomputed offline; `rank.py` then runs CPU-only,
  no network, < 5 minutes (see the GitHub repo).

## Deploy this Space yourself

1. Create a new **Streamlit** Space on HuggingFace.
2. Push this repository to the Space (it needs `app/app.py`, the `lighthouse/` package,
   `artifacts/jd_rubric.json`, `artifacts/jd_facet_emb.npy`, and `app/sample_candidates.jsonl`).
3. Set **app_file** to `app/app.py` (already declared in this front-matter) and **requirements**
   to `app/requirements.txt`.
4. The Space will download `bge-small` on first run and is ready.

The full architecture, evaluation, and reproduce commands are in the GitHub repository's
top-level `README.md`.
