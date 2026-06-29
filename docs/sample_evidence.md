# Sample evidence — measured

The HF Space demo ranks a fixed 100-candidate **demo sample**
(`app/sample_candidates.jsonl`). The sample is now composed of:

- **80 candidates = the actual top 80 of the full 100K ranking** (i.e. `submission.csv` rows 1-80, in the same order produced by `rank.py`).
- **10 keyword-stuffer traps** — non-engineering current title (Accountant, Customer Support, HR Manager, Sales Executive, etc.) carrying AI/ML skills, sourced from the full 100K.
- **10 honeypots** — subtly impossible profiles (date contradictions, tenure-vs-experience overflow, expert-with-0-months) sourced from the full 100K.

This composition makes the demo's top 80 **literally identical** to the submission's top 80, while leaving 20 visible failure modes for judges to see the gates work.

Source: `python rank.py --candidates app/sample_candidates.jsonl --top 100`.

## Headline (demo sample)

| Slot | Composition | Score band |
|---|---|---|
| Ranks 1-80 | Real AI/ML/IR engineers (= submission's top 80) | 0.93 – 1.00 |
| Ranks 81-90 | Non-tech keyword-stuffers, every row with NON-TECHNICAL gate badge | 0.017 – 0.025 |
| Ranks 91-100 | Honeypots, every row with a flagged anomaly reason | 0.000 |

**Score ratio rank-1 vs lowest gated non-tech ≈ 40×.** Real engineers and traps are in clearly separated tiers.

## Honeypots — zeroed, ranked last

All 10 seeded honeypots score exactly 0.0 and occupy ranks 91-100. Examples:

- `4 skills claimed advanced/expert with 0 months used (e.g. ...)`
- `career roles sum to 237mo, impossible for 10.1 yrs experience`
- `role at <Company> ends before it starts`

## Non-technical "keyword stuffer" gate — fires on every trap

Every non-engineering profile in the sample carries the `non_technical_role` gate badge with reasoning `current role '<title>' is non-engineering with no AI/ML role in career history`. Mult: 0.05.

Examples (rank, title, score):

| Rank | Title | Score |
|---|---|---|
| 81 | Content Writer | 0.0242 |
| 82 | Sales Executive | 0.0203 |
| 83 | Customer Support | 0.0200 |
| 84 | Marketing Manager | 0.0195 |
| 85 | Operations Manager | 0.0173 |

## Official submission (top 100 of 100,000)

The full ranker output committed to `submission.csv`:

| Metric | Result |
|---|---|
| Rows | 100 |
| AI/ML/IR-aligned titles | **100 / 100** |
| Non-technical titles | **0** |
| Honeypots in top 100 | **0** |
| Wall-clock on 100K | **55 s** (CPU-only, no network) |

Because the demo sample's top 80 *is* the submission's top 80, the live demo at the HF Space is no longer just a stress-test visualization — it is a direct, in-browser replay of the first 80% of the official ranking.

## Reproducing

```
python rank.py --candidates "<path>/candidates.jsonl" --out submission.csv
python rank.py --candidates app/sample_candidates.jsonl --out /tmp/demo.csv
python validate_submission.py submission.csv
python -m pytest -q          # 127 tests, lint + mypy clean
```
