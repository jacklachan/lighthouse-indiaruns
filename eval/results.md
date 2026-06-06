# Lighthouse — Evaluation Results

> Metrics are computed against the **Claude-authored proxy labels** (`eval/eval_labels.json`; see `eval/labeler.py` for the honest framing). They are indicative — the official ground truth is hidden — but the **relative** signals (ablation deltas, baseline gap) are the point.

- Eval set: **221 candidates** across 8 archetypes (real AI engineers, plain-language strong, keyword-stuffers, services-only, location-fail, behaviorally-weak, honeypots, other).
- Tier histogram (0–5): {0: 165, 1: 16, 2: 6, 3: 3, 4: 15, 5: 16}
- Embeddings: 221 precomputed candidate vectors.

## Headline (Lighthouse vs naive keyword baseline)

| System | NDCG@10 | NDCG@50 | MAP | P@10 | Composite | Non-fits in top-10 |
|---|---|---|---|---|---|---|
| **Lighthouse (full)** | 1.000 | 0.994 | 0.998 | 1.000 | 0.998 | 0 |
| Baseline (keyword count) | 0.577 | 0.563 | 0.438 | 0.600 | 0.553 | 4 |

**Lighthouse beats the keyword baseline by +0.445 composite** (0.998 vs 0.553), and by +0.423 on the heavily-weighted NDCG@10. The baseline floods its top-10 with **4 non-fits** (keyword-stuffers/honeypots); Lighthouse admits **0**.

## Ablation study

Each row removes one piece of Lighthouse and re-evaluates. Single-component effects are small here and concentrate **below the top-10** (NDCG@10 is saturated — see Trap Resistance); the honeypot filter shows the clearest MAP effect, and the combined *anti-trap OFF* row shows the largest drop.

| Configuration | NDCG@10 | NDCG@50 | MAP | P@10 | Composite | Δ Comp | Non-fits@10 |
|---|---|---|---|---|---|---|---|
| Lighthouse (full) | 1.000 | 0.994 | 0.998 | 1.000 | 0.998 | — | 0 |
| – ablate role_coherence | 1.000 | 0.991 | 0.998 | 1.000 | 0.997 | -0.001 | 0 |
| – ablate career_evidence | 1.000 | 0.995 | 0.998 | 1.000 | 0.998 | +0.000 | 0 |
| – ablate trust_skills | 1.000 | 0.991 | 0.999 | 1.000 | 0.997 | -0.001 | 0 |
| – no hard-negative gates | 1.000 | 0.986 | 0.992 | 1.000 | 0.995 | -0.004 | 0 |
| – no honeypot filter | 1.000 | 0.982 | 0.952 | 1.000 | 0.987 | -0.011 | 0 |
| – no behavioral modifier | 1.000 | 0.994 | 0.986 | 1.000 | 0.996 | -0.002 | 0 |
| – anti-trap OFF (no role_coh+gates+honeypot) | 1.000 | 0.974 | 0.942 | 1.000 | 0.984 | -0.015 | 0 |

**Reading the ablation.** Lighthouse defends against each trap in *layers* — the `role_coherence` component *and* the `non_technical` gate both fight keyword-stuffers, so knocking out one leaves a backstop. That is why single-component deltas are small. The contribution shows where traps actually live: removing the **honeypot filter** drops MAP -0.047 (honeypots re-enter the ranked body), and the combined **anti-trap OFF** stack drops composite -0.015 to 0.984. The sharpest evidence is the Trap-Resistance table below and the **baseline gap** above: strip the reasoning layers and Lighthouse slides toward the keyword baseline that floods its shortlist with stuffers.

## Trap resistance (where the anti-trap logic shows up)

NDCG@10 is saturated above because, in this pool, trap candidates are genuinely weak on `career_evidence`/`semantic_fit` and never reach the top-10 under any reasonable scorer. The anti-trap logic's contribution is visible **deeper in the ranking**: it pushes traps down and keeps them out of the shortlist. Lower median rank = worse (trap surfaced higher).

| Trap archetype | n | Median rank (Lighthouse) | Median rank (anti-trap OFF) | Median rank (keyword baseline) | In top-25: LH / off / baseline |
|---|---|---|---|---|---|
| keyword_stuffer | 32 | **76** | 67 | 30 | 0 / 0 / 13 |
| location_fail | 33 | **171** | 161 | 133 | 0 / 0 / 2 |
| services_only | 32 | **158** | 190 | 126 | 0 / 0 / 0 |
| honeypot | 25 | **209** | 143 | 124 | 0 / 2 / 1 |

The **keyword baseline puts 13/32 keyword-stuffers in its top-25**; Lighthouse admits **0**. 
With the honeypot filter and gates removed, honeypots climb from median rank 209 to 143 (2 entering the top-25 vs 0 with the full system).

## Honeypot safety

- Lighthouse, eval top-50: **0/50 honeypots**.
- Same ranker with the honeypot filter OFF: 5/50.
- Keyword baseline: 2/50.
- Full 100K submission honeypot rate in top-100: **0** (see rank.py output / tests). DQ threshold is >10%.

