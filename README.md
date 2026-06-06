# 🔦 Lighthouse — a recruiter-grade, reasoning-first candidate ranker

> *Keyword filters surface the loudest profiles. Lighthouse surfaces the right ones — and ignores the fakes that fool keyword filters.*

Submission for the **Redrob "Intelligent Candidate Discovery & Ranking" Challenge** (India Runs hackathon, Track 1 — Data & AI).

Lighthouse ranks the **top 100** best-fit candidates out of a **100,000** pool for one
nuanced Senior AI Engineer JD. It is built to beat the dataset's deliberate traps —
keyword-stuffers, behavioral twins, plain-language strong candidates, and ~80 honeypots —
by reasoning about the **gap between what the JD says and what it means**, not by counting
keywords.

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
> `artifacts/` so the rank step needs only local files.

## Architecture (offline-LLM-augmented hybrid)

Two phases — see [`docs`](#) and the deck for detail:

- **Offline precompute** (`precompute.py`): JD → structured rubric (`artifacts/jd_rubric.json`,
  authored by Claude reading the JD), candidate embeddings (`BAAI/bge-small-en-v1.5`, cached
  float16), a BM25 index, and a Claude-authored stratified eval label set.
- **Online rank** (`rank.py`, pure numpy/pandas): load artifacts → score every candidate on
  five components → apply hard-negative gates + behavioral modifier → zero out honeypots →
  sort with the spec tie-break → top 100 → grounded reasoning → `submission.csv`.

The five scoring components: `semantic_fit`, `role_coherence`, `career_evidence`,
`experience_fit`, `trust_skills`. **No candidate data hits any API at rank-time.**

## Repo layout

| Path | What |
|---|---|
| `lighthouse/` | core package: `loader`, `features`, `scoring`, `gates`, `honeypot`, `reasoning`, `metrics` |
| `precompute.py` | builds all artifacts |
| `rank.py` | the single reproduce command |
| `artifacts/` | committed precomputed files (`jd_rubric.json`, embeddings via git-lfs) |
| `eval/` | Claude-authored labels, metrics, `results.md` (NDCG/MAP/P@10, ablation, baseline) |
| `tests/` | pytest: honeypot detection, monotonicity, tie-break, reasoning grounding, CSV validity |
| `app/` | HuggingFace Spaces Streamlit sandbox |
| `deck/` | idea-submission deck → PDF |

## Compute environment

CPU-only, no network, ≤ 5 min, ≤ 16 GB RAM for the rank step. Declared in
`submission_metadata.yaml`.

## AI tool usage

Claude was used **offline** to (a) parse the JD into the structured rubric and (b) label the
stratified eval set, and as a coding assistant. **No candidate data was sent to any hosted LLM
at rank-time** — the ranking step is deterministic numpy/pandas over precomputed local
artifacts. See `submission_metadata.yaml`.
