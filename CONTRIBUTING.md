# Contributing to Lighthouse

Thanks for your interest. Lighthouse is a reasoning-first candidate ranker built for the
Redrob / India Runs Track 1 challenge. This guide keeps changes safe and reproducible.

## Golden rule: the ranked output is a frozen, graded artifact

`submission.csv` is the graded deliverable and is produced from the grader-supplied
100K pool (which is **not** committed — see `.gitignore`). Do **not** edit it by hand or
regenerate it casually. Any change that could alter ranking order must be justified,
reproduced end-to-end, and reviewed.

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy==2.2.6 pandas==2.3.3 PyYAML==6.0.2   # rank-time runtime
pip install -r requirements-dev.txt                    # test + lint + type tooling
```

To rebuild artifacts or run the live app you also need the optional extras:

```bash
pip install -e ".[precompute]"   # torch + sentence-transformers + bm25
pip install -e ".[app]"          # streamlit
```

## Quality gates (run before every commit)

```bash
pytest -q                                   # all tests must pass
python validate_submission.py submission.csv  # must report "Submission is valid."
ruff check .                                 # lint
black --check .                              # format
mypy lighthouse                              # types
```

CI (`.github/workflows/ci.yml`) re-runs the test + validator gate on every push and PR
to `main`. A red CI is a blocker.

## Design constraints (do not regress these)

- **Rank-time is CPU-only, no network, < 5 min, ≤ 16 GB RAM.** `rank.py` may import only
  `numpy`, `pandas`, `pyyaml`. Heavy libraries (torch, sentence-transformers) belong in
  `precompute.py`/`app.py` behind lazy imports.
- **Determinism.** Seeded (`SEED = 1729`); the same input must yield the same output.
- **Explainability.** Gates, the honeypot filter, and reasoning must stay auditable —
  every claim in a candidate's reasoning must be grounded in that candidate's profile.

## Commit style

Small, focused commits. Describe *what changed and why*. Keep behavior-affecting changes
separate from formatting/typing/doc changes so review and `git bisect` stay clean.
