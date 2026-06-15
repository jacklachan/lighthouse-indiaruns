# Changelog

All notable changes to this project are documented here. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
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
  floor (`fail_under = 75`) on the rank-time package.

### Fixed
- `normalize_semantic` used the `ndarray.ptp()` method, which NumPy 2.0 removed (the repo
  pins `numpy==2.2.6`). This raised `AttributeError` in the degenerate-range branch
  (reachable in the small-N sandbox). Switched to `np.ptp(raw)` — numerically identical,
  no longer crashes. The 100K rank path never hits this branch, so `submission.csv` is
  unaffected.
- Type-only: added explicit `Optional` annotations on `None`-defaulted parameters in
  `scoring.py` / `ranker.py` (no runtime behavior change).

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
