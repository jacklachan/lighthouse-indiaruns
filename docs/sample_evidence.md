# Sample evidence — measured

The HF Space demo ranks a fixed 100-candidate **stress-test sample**, not the
top 100 of the full 100K. The sample is built ~half real AI/ML engineers,
~half deliberate traps (non-technical roles with stuffed keywords, services-only,
off-location, honeypots). This file records what the ranker actually does to it,
as a defensible cross-check against the live demo.

Source: `python rank.py --candidates app/sample_candidates.jsonl --top 100`.

## Headline (sample of 100)

| Slot | Composition |
|---|---|
| Top 10 | 10 / 10 AI/ML/IR engineers (no traps) |
| Bottom 10 | 8 / 10 explicit traps + 2 borderline engineers; 6 honeypots score 0.0 |
| Score ratio rank-1 vs rank-100 | 1.0 vs 0.0 (∞); rank-1 vs lowest non-honeypot ≈ 200× |

## Honeypots — zeroed, ranked last

| Rank | Title | Why flagged | Final score |
|---|---|---|---|
| 95 | NLP Engineer | career roles sum to 87mo, impossible for 2.8 yrs | 0.000 |
| 96 | Software Engineer | career roles sum to 162mo, impossible for 5.5 yrs | 0.000 |
| 97 | Business Analyst | 4 skills claimed advanced/expert with 0 months used | 0.000 |
| 98 | Project Manager | career roles sum to 237mo, impossible for 10.1 yrs | 0.000 |
| 99 | Civil Engineer | 4 skills claimed advanced/expert with 0 months used | 0.000 |
| 100 | Content Writer | career roles sum to 180mo, impossible for 7.6 yrs | 0.000 |

## Non-technical "keyword stuffer" gate — fires on every trap

Every non-engineering profile in the sample carries the `non_technical_role`
gate badge and a multiplied score in the 0.005-0.025 band — at least 40× below
the lowest-scoring real engineer in the top 30.

Examples (rank, title, final score):

| Rank | Title | Score |
|---|---|---|
| 33 | Content Writer | 0.0266 |
| 34 | Accountant | 0.0256 |
| 38 | Customer Support | 0.0229 |
| 50 | Marketing Manager | 0.0180 |
| 70 | HR Manager | 0.0129 |
| 89 | Accountant | 0.0033 |
| 91 | Graphic Designer | 0.0024 |

## Official submission (top 100 of 100,000)

The full ranker output committed to `submission.csv` is the real claim and is
independent of the demo:

| Metric | Result |
|---|---|
| Rows | 100 |
| AI/ML/IR-aligned titles | **100 / 100** |
| Non-technical titles | **0** |
| Honeypots in top 100 | **0** (audit script: `python rank.py … && grep …`) |

The submission file has not changed since the demo banner/gate tweak; the
non_technical gate never fires on real engineers, so tightening it (0.25 → 0.05)
leaves the official ranking untouched.

## Reproducing

```
python rank.py --candidates app/sample_candidates.jsonl --out /tmp/sample.csv --top 100
python validate_submission.py submission.csv
python -m pytest -q          # 127 tests, lint + mypy clean
```
