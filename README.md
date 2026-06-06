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
  plain-language Tier-5 `CAND_0000031` ranks #1.

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
