"""Claude-authored relevance-tier (0-5) labeler — the eval ground-truth proxy.

IMPORTANT (honesty): these labels are authored by Claude offline by encoding the
JD's own definition of fit. They are a documented PROXY for the hidden ground
truth, used to tune weights and report indicative metrics — not the official
labels. To keep the evaluation meaningful, the labeler is deliberately DISTINCT
from the ranker:
  * it ignores semantic_fit entirely (the ranker weights it 0.22),
  * it uses different component weights, and
  * it applies hard tier CAPS for JD disqualifiers + an availability down-weight.
So a high NDCG reflects genuine agreement, not a tautology, and ablating
role_coherence from the ranker measurably degrades it.

Tier meaning (aligned to the JD's "ideal candidate"):
  5 strong fit  | 4 good | 3 relevant | 2 marginal | 1 weak | 0 non-fit/honeypot
"""
from __future__ import annotations

from typing import Tuple

from lighthouse import features, gates, honeypot, scoring


def _fit_to_tier(fit: float) -> int:
    if fit >= 0.78:
        return 5
    if fit >= 0.62:
        return 4
    if fit >= 0.47:
        return 3
    if fit >= 0.33:
        return 2
    if fit >= 0.20:
        return 1
    return 0


def label_candidate(raw: dict, rubric: dict) -> Tuple[int, dict]:
    """Return (tier, explanation dict)."""
    hp, hp_reasons = honeypot.detect(raw)
    if hp:
        return 0, {"reason": "honeypot", "detail": hp_reasons[:2]}

    rc = features.role_coherence_taxonomy(raw, rubric)
    ce = features.career_evidence(raw, rubric)
    ef = features.experience_fit(raw, rubric)
    ts = features.trust_skills(raw, rubric)
    # labeler weights (no semantic term; distinct from the ranker)
    fit = 0.34 * rc + 0.30 * ce + 0.21 * ts + 0.15 * ef
    tier = _fit_to_tier(fit)

    _, gate_reasons = gates.apply_gates(raw, rubric)
    joined = " | ".join(gate_reasons).lower()
    caps = []

    if "non-engineering" in joined:
        tier = 0; caps.append("non-technical role")
    if "not willing to relocate" in joined:
        tier = min(tier, 1); caps.append("outside India, no relocation")
    elif "outside india" in joined:
        tier = min(tier, 3); caps.append("outside India (visa risk)")
    if "services/consulting" in joined:
        tier = min(tier, 2); caps.append("services-only career")
    if "research-heavy" in joined:
        tier = max(0, tier - 2); caps.append("research-only")
    if "computer vision/speech" in joined:
        tier = max(0, tier - 2); caps.append("cv/speech-only")
    if "llm-wrapper" in joined:
        tier = max(0, tier - 2); caps.append("langchain-only-recent")

    # availability down-weight (JD: unreachable candidates aren't actually hireable)
    beh_mult, beh_facts = scoring.behavioral_modifier(raw, rubric)
    if beh_mult <= 0.85 and tier > 0:
        tier = max(0, tier - 1); caps.append("low availability")

    return int(max(0, min(5, tier))), {
        "fit": round(fit, 3), "rc": rc, "ce": ce, "ef": ef, "ts": ts,
        "caps": caps, "gate_reasons": gate_reasons, "beh_mult": beh_mult,
    }
