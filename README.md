# 🔦 Lighthouse — a recruiter-grade, reasoning-first candidate ranker

[![CI](https://github.com/jacklachan/lighthouse-indiaruns/actions/workflows/ci.yml/badge.svg)](https://github.com/jacklachan/lighthouse-indiaruns/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![Rank-time: CPU-only, no network](https://img.shields.io/badge/rank--time-CPU--only%20%7C%20no%20network%20%7C%20%3C5min-success.svg)](requirements.txt)

> *Keyword filters surface the loudest profiles. Lighthouse surfaces the right ones — and ignores the fakes that fool keyword filters.*

Submission for the **Redrob "Intelligent Candidate Discovery & Ranking" Challenge** (India Runs hackathon, Track 1 — Data & AI).

Lighthouse ranks the **top 100** best-fit candidates out of a **100,000** pool for one
nuanced Senior AI Engineer JD. It is built to beat the dataset's deliberate traps —
keyword-stuffers, behavioral twins, plain-language strong candidates, and ~80 honeypots —
by reasoning about the **gap between what the JD says and what it means**, not by counting
keywords.

---

## Headline evidence (label-independent — no trust in our labels required)

These three facts hold without believing any tier we authored. They are the strongest
signal in the submission; the metric tables in `eval/results.md` come **after**.

- **0 honeypots in the top-100** (independently audited over the full 100K; the dataset
  ships ~80 such impossible profiles, DQ threshold is >10%).
- **100/100 of the top-100 hold an AI/ML/IR/DS/Search/NLP-aligned title; 0 are
  non-technical.** The provided `sample_submission` (keyword count) puts HR Managers and
  Accountants at #1–20 — the exact trap the JD warns about.
- **Trap resistance:** the keyword baseline puts **13/32 keyword-stuffers** in its top-25;
  Lighthouse admits **0**. The plain-language Tier-5 `CAND_0000031` ranks **top-10** (#9).

Self-labeled NDCG/MAP numbers (composite **0.998 vs 0.553** baseline) live in
[`eval/results.md`](eval/results.md) §2 — read them as **internal consistency**, not
absolute accuracy. The *gap* to the baseline and the ablation deltas are the meaningful
signal there, not the saturated absolute.

---

## How this maps to the brief

A fast index from each thing the challenge evaluates to where this repo satisfies it.

| What the challenge evaluates | Where Lighthouse delivers it |
| --- | --- |
| Top-100 ranking from the 100K pool | `rank.py` → `submission.csv` (4 columns, validator-clean) |
| Honeypot avoidance (>10% in top-100 = DQ) | `lighthouse/honeypot.py` explainable filter — **0 honeypots** in the top-100, audited over the full 100K (`eval/results.md` §1) |
| Resisting keyword-stuffers & non-fits | `role_coherence` + `career_evidence` + JD-derived gates (`lighthouse/scoring.py`, `lighthouse/gates.py`); trap-resistance table in `eval/results.md` §1 |
| Ranking quality (label-independent) | **0 honeypots** in top-100; **100/100** AI/ML-aligned titles; **0/32** keyword-stuffers admitted vs baseline's **13/32** (`eval/results.md` §1) |
| Ranking quality (NDCG@10/50, MAP, P@10) | `lighthouse/metrics.py`; directional composite vs keyword baseline (`eval/results.md` §2 — read as gap, not absolute) |
| Grounded, explainable reasoning | `lighthouse/reasoning.py` — every claim grounded in the profile; `tests/test_reasoning.py` |
| CPU-only, no-network, < 5 min rank | `rank.py` imports only numpy/pandas/pyyaml (`requirements.txt`); declared in `submission_metadata.yaml` |
| Reproducibility & determinism | seeded (`SEED = 1729`), committed artifacts, `precompute.py`, and CI |
| Evaluation rigor (no circularity) | self-labeled **and** a blind independent-label harness (`eval/blind_compare.py`) |
| Code quality & testing | 63 tests + ruff/black/mypy + an 85% coverage floor, all enforced in CI (`.github/workflows/ci.yml`) |

---

## Reproduce the submission

```bash
# 1. Install rank-time deps (numpy / pandas / pyyaml are all rank.py needs)
pip install -r requirements.txt

# 2. (precompute — network OK, runs OUTSIDE the 5-min budget)
#    Builds embeddings, BM25 index, JD rubric and eval labels into artifacts/.
python precompute.py --candidates ./data/candidates.jsonl

# 3. Rank step — CPU only, no network, < 5 min, produces the CSV
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

# 4. Validate format
python validate_submission.py submission.csv     # -> "Submission is valid."
```

> `data/candidates.jsonl` is **gitignored** — the grader supplies the 100K pool at
> reproduction time. Precomputed artifacts (embeddings, indexes, the JD rubric) live in
> `artifacts/` so the rank step needs only local files. The committed
> `artifacts/cand_emb.npy` is ~75 MB (float16); if your host enforces the 100 MB git limit,
> it is comfortably under it (no git-lfs needed). If you prefer not to ship it, run
> `precompute.py` (network OK, ~130 min CPU one-time, or faster with
> `--model sentence-transformers/all-MiniLM-L6-v2`) to regenerate it.

**Precompute notes.** `precompute.py` downloads `bge-small` once (network OK — this is the
only network step and it is *outside* the rank budget) and encodes 100K candidates on CPU in
~130 min. The rank step then loads only `.npy`/`.json` artifacts and runs in well under 5
minutes on CPU with no network.

## Architecture (offline-LLM-augmented hybrid)

Two phases — see `artifacts/jd_rubric_rationale.md`, `eval/results.md`, and the deck for detail:

- **Offline precompute** (`precompute.py`): JD → structured rubric (`artifacts/jd_rubric.json`,
  authored by Claude reading the JD), candidate embeddings (`BAAI/bge-small-en-v1.5`, cached
  float16), a BM25 index, and a Claude-authored stratified eval label set.
- **Online rank** (`rank.py`, pure numpy/pandas): load artifacts → score every candidate on
  five components → apply hard-negative gates + behavioral modifier → zero out honeypots →
  sort with the spec tie-break → top 100 → grounded reasoning → `submission.csv`.

The five scoring components: `semantic_fit`, `role_coherence`, `career_evidence`,
`experience_fit`, `trust_skills`. Combined as
`final = base_weighted_sum × Π(hard-negative gates) × behavioral_modifier`, with honeypots
zeroed. **No candidate data hits any API at rank-time.**

## Results

The evidence that matters here is **label-independent** — it doesn't rely on trusting any labels
we authored. Full detail in [`eval/results.md`](eval/results.md).

**Label-independent (the real results):**

- **0 honeypots** in the top-100 (independently audited over the full 100K; DQ threshold is >10%).
- **All 100/100** of the top-100 hold an AI/ML/IR/DS/Search/NLP-aligned title; **0 are
  non-technical**. The provided `sample_submission` (keyword count) instead ranks HR Managers and
  Accountants at #1–20.
- **Trap resistance:** the keyword baseline puts **13/32 keyword-stuffers** in its top-25;
  Lighthouse admits **0**. The two planted non-fits rank low for the right reasons
  (`CAND_0000001` Toronto/no-relocate; `CAND_0000002` Operations-Manager trajectory); the
  plain-language Tier-5 `CAND_0000031` ranks in the top-10 (#9).

**Directional metrics (self-labeled — *not* a claim of absolute accuracy):** against a
Claude-authored 221-candidate proxy set, composite **0.998 vs 0.553** for the keyword baseline
(+0.445). The labeler and ranker share assumptions, so the near-perfect NDCG@10 reflects internal
consistency, **not** validated accuracy — the meaningful signal is the *gap* to the baseline and
the ablation deltas, not the absolute number. For an **independent** check, the repo ships a blind
human-labeling harness (`scripts/make_blind_eval.py` → a human fills tiers → `eval/blind_compare.py`).

Reproduce the eval:
```bash
python eval/build_labels.py --candidates ./data/candidates.jsonl   # Claude-authored proxy labels
python eval/evaluate.py --candidates ./data/candidates.jsonl       # -> eval/results.md
python scripts/make_blind_eval.py && python eval/blind_compare.py  # independent human check
```

## Tests & quality gates

```bash
pytest -q                        # 63 tests: honeypot detection, gates, scoring + the
                                 # normalize_semantic regression, loader robustness,
                                 # reasoning grounding, metrics, tie-break, CSV validity
pytest -q --cov=lighthouse       # 86% coverage on the rank-time package (CI floor: 85%)
ruff check . && black --check .  # lint + format
mypy lighthouse                  # static type check
```

Every gate above runs in CI on each push/PR to `main` (`.github/workflows/ci.yml`).

## Repo layout

| Path | What |
|---|---|
| `lighthouse/` | core package: `loader`, `features`, `scoring`, `gates`, `honeypot`, `reasoning`, `metrics` |
| `precompute.py` | builds all artifacts |
| `rank.py` | the single reproduce command |
| `artifacts/` | committed precomputed files (`jd_rubric.json`, float16 embeddings as plain `.npy` — no git-lfs) |
| `eval/` | Claude-authored labels, metrics, `results.md` (NDCG/MAP/P@10, ablation, baseline) |
| `tests/` | pytest: honeypot detection, monotonicity, tie-break, reasoning grounding, CSV validity |
| `app/` | HuggingFace Spaces sandbox — FastAPI + animated frontend (`server.py`, `web/`) wrapping the ranker |
| `deck/` | idea-submission deck → PDF |

## Compute environment

CPU-only, no network, ≤ 5 min, ≤ 16 GB RAM for the rank step. Declared in
`submission_metadata.yaml`.

## AI tool usage

Claude was used **offline** to (a) parse the JD into the structured rubric and (b) label the
stratified eval set, and as a coding assistant. **No candidate data was sent to any hosted LLM
at rank-time** — the ranking step is deterministic numpy/pandas over precomputed local
artifacts. See `submission_metadata.yaml`.
