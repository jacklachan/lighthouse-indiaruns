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

## Results (vs. naive keyword baseline)

Measured on a Claude-authored, stratified 221-candidate proxy label set (`eval/results.md`;
the official ground truth is hidden — the *relative* signals are the point):

| System | NDCG@10 | NDCG@50 | MAP | P@10 | Composite |
|---|---|---|---|---|---|
| **Lighthouse** | **1.000** | 0.993 | 0.998 | 1.000 | **0.998** |
| Keyword-count baseline | 0.577 | 0.563 | 0.438 | 0.600 | 0.553 |

- **+0.445 composite** over the baseline. The keyword baseline floods its top-25 with **13/32
  keyword-stuffers**; Lighthouse admits **0**.
- **0 honeypots** in the top-100 (DQ threshold is >10%). The two planted non-fits rank low for
  the right reasons (`CAND_0000001` Toronto/no-relocate; `CAND_0000002` Operations-Manager
  trajectory), and the plain-language Tier-5 `CAND_0000031` ranks at the top.
- Anti-trap logic is layered: see the ablation + trap-resistance tables in `eval/results.md`.

Reproduce the eval:
```bash
python eval/build_labels.py --candidates ./data/candidates.jsonl   # build Claude-authored labels
python eval/evaluate.py --candidates ./data/candidates.jsonl       # -> eval/results.md
```

## Tests

```bash
pytest -q     # 36 tests: honeypot detection, gates, scoring/monotonicity, tie-break,
              # reasoning grounding (no hallucination), metrics, CSV validity
```

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
