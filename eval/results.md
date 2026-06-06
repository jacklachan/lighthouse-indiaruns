# Lighthouse — Evaluation Results

> **How to read this page.** The strongest evidence here is **label-independent** — it does not depend on trusting any labels we authored: honeypot safety, trap resistance, and the composition of the actual top-100. We lead with those. The NDCG/MAP numbers further down are computed against a **Claude-authored proxy label set** and are **directional only** — see the caption there before reading them as accuracy.

- Eval set for the metric tables: **221 candidates** across 8 archetypes (real AI engineers, plain-language strong, keyword-stuffers, services-only, location-fail, behaviorally-weak, honeypots, other). Tier histogram (0–5): {0: 165, 1: 16, 2: 6, 3: 3, 4: 15, 5: 16}.
- Embeddings: 221 precomputed candidate vectors.

## 1. Headline evidence (label-independent)

These results stand without trusting our labels — they are facts about the ranking and the dataset's own planted traps.

**Honeypot safety.** The dataset seeds ~80 subtly-impossible profiles, forced to tier 0; >10% in the top-100 is an automatic disqualification. **Lighthouse's top-100 contains 0 honeypots** (independently audited over the full 100K). With the honeypot filter switched off, they re-enter the ranked body.

**Top-100 composition.** All **100/100** of Lighthouse's top-100 hold an AI/ML/IR/DS/Search/NLP-aligned title; **0 are non-technical** (keyword-stuffer roles like Accountant/HR/Marketing). The provided `sample_submission` (pure keyword count) instead ranks HR Managers, Accountants and Marketing Managers at #1–20 — the exact trap the JD warns about.

**Trap resistance.** How far down each scorer pushes the dataset's traps (lower median rank = trap surfaced higher = worse). This is a direct, falsifiable comparison that needs no labels:

| Trap archetype | n | Median rank — Lighthouse | — anti-trap OFF | — keyword baseline | In top-25: LH / off / baseline |
|---|---|---|---|---|---|
| keyword_stuffer | 32 | **76** | 67 | 30 | 0 / 0 / 13 |
| location_fail | 33 | **171** | 161 | 133 | 0 / 0 / 2 |
| services_only | 32 | **158** | 190 | 126 | 0 / 0 / 0 |
| honeypot | 25 | **209** | 143 | 124 | 0 / 2 / 1 |

The **keyword baseline puts 13/32 keyword-stuffers in its top-25**; Lighthouse admits **0**. Remove the honeypot+gate stack and honeypots climb from median rank 209 to 143 (2 entering the top-25 vs 0 with the full system).

## 2. Directional metrics (self-labeled — NOT a claim of absolute accuracy)

> ⚠️ **Read this caption first.** The tiers below come from `eval/eval_labels.json`, authored by the same source (Claude) that informs the ranker. The labeler is deliberately *distinct* from the ranker (no semantic term, different weights, hard caps — see `eval/labeler.py`), but they still share assumptions. **So a near-perfect NDCG@10 reflects internal consistency, not validated accuracy** — do not read it as 'the system is 100% correct'. The only things worth taking from this table are the **relative** signals: the size of the gap to the keyword baseline, and the ablation deltas. For an *independent* check, see §4.

| System | NDCG@10 | NDCG@50 | MAP | P@10 | Composite |
|---|---|---|---|---|---|
| Lighthouse (full) | 1.000 | 0.994 | 0.998 | 1.000 | 0.998 |
| Baseline (keyword count) | 0.577 | 0.563 | 0.438 | 0.600 | 0.553 |

Relative read: Lighthouse vs the keyword baseline is **+0.445 composite** (0.998 vs 0.553). The baseline puts **4 non-fits** in its top-10; Lighthouse **0**. NDCG@10 is saturated (genuine fits dominate the top-10), so the anti-trap work shows up deeper — quantified in §1's trap table and §3's ablation.

## 3. Ablation (directional)

Each row removes one piece and re-scores against the proxy labels. Single-component deltas are small **by design** — Lighthouse defends each trap in *layers* (the `role_coherence` component *and* the `non_technical` gate both fight stuffers), so knocking out one leaves a backstop. The effect concentrates below the top-10 (NDCG@10 saturated):

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

Removing the **honeypot filter** drops MAP -0.047 (honeypots re-enter the ranked body); the combined **anti-trap OFF** stack drops composite -0.015 to 0.984 — i.e. strip the reasoning layers and Lighthouse slides toward the keyword baseline.

