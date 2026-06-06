"""The Lighthouse scoring model — the core IP.

Five explainable components in [0,1] are combined into a base score, then
modified multiplicatively:

    final = base_weighted_sum  ×  Π(hard-negative gates)  ×  behavioral_modifier
    final = 0                  if honeypot/anomaly detected

| component       | what it measures                                            | anti-trap role            |
|-----------------|-------------------------------------------------------------|---------------------------|
| semantic_fit    | cosine of candidate embedding vs JD facets                  | plain-language Tier-5s    |
| role_coherence  | is the title/trajectory actually AI/ML/IR/SWE-ranking?      | decisive vs keyword-stuff |
| career_evidence | did they BUILD ranking/search/recsys at product companies?  | rewards real builders     |
| experience_fit  | soft curve peaking 6-8 yrs, in-band 5-9                     | matches the JD band       |
| trust_skills    | skills weighted by proficiency×duration×endorse×assessment  | kills keyword stuffing    |

`semantic_fit` is supplied pre-normalized (population percentile-scaled) by the
caller; everything else is computed here from the rubric + raw fields.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Tuple

import numpy as np

from . import features, gates, honeypot, loader

REFERENCE_DATE = date(2026, 6, 6)


# ---------------------------------------------------------------------------
# semantic fit (embedding side)
# ---------------------------------------------------------------------------

def raw_semantic_fit(cand_emb: np.ndarray, facet_emb: np.ndarray) -> np.ndarray:
    """Per-candidate raw semantic fit = 0.6*max + 0.4*mean facet cosine.

    Both matrices are L2-normalized, so the dot product is cosine. Returns a
    1-D array aligned to cand_emb rows. (max rewards a strong single-facet
    match; mean rewards broad alignment.)
    """
    cos = cand_emb.astype(np.float32) @ facet_emb.T          # (N, F)
    return 0.6 * cos.max(axis=1) + 0.4 * cos.mean(axis=1)


def normalize_semantic(raw: np.ndarray) -> np.ndarray:
    """Percentile-clip raw semantic fit into [0,1] across the population."""
    if len(raw) == 0:
        return raw
    lo = np.percentile(raw, 5)
    hi = np.percentile(raw, 95)
    if hi - lo < 1e-6:
        return np.clip((raw - raw.min()) / (raw.ptp() + 1e-6), 0, 1)
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------------------
# behavioral modifier (multiplicative, clamped)
# ---------------------------------------------------------------------------

def behavioral_modifier(raw: dict, rubric: dict) -> Tuple[float, List[str]]:
    """Reward reachable/active candidates, penalize stale/unreachable. [floor,ceiling].

    Sentinels (-1 github, -1 offer, empty assessments) are NEUTRAL — only
    present signals move the modifier. Behavior modifies; it never drives.
    """
    b = rubric["behavioral"]
    sig = loader.get_signals(raw)
    delta = 0.0
    facts: List[str] = []

    la = loader.parse_date(sig.get("last_active_date"))
    if la:
        days = (REFERENCE_DATE - la).days
        if days <= b["active_recent_days"]:
            delta += 0.05; facts.append(f"active {days}d ago")
        elif days >= b["active_stale_days"]:
            delta -= 0.15; facts.append(f"inactive {days}d")
        elif days >= 120:
            delta -= 0.07; facts.append(f"last active {days}d ago")

    rr = sig.get("recruiter_response_rate")
    if isinstance(rr, (int, float)):
        if rr >= b["good_response_rate"]:
            delta += 0.04; facts.append(f"{rr:.0%} recruiter response")
        elif rr <= b["weak_response_rate"]:
            delta -= 0.12; facts.append(f"low {rr:.0%} recruiter response")
        elif rr <= 0.3:
            delta -= 0.05

    if sig.get("open_to_work_flag"):
        delta += 0.02
    if sig.get("verified_email") and sig.get("verified_phone"):
        delta += 0.02

    ic = sig.get("interview_completion_rate")
    if isinstance(ic, (int, float)) and ic > 0:
        if ic >= b["good_interview_completion"]:
            delta += 0.02
        elif ic < 0.3:
            delta -= 0.06; facts.append(f"low {ic:.0%} interview completion")

    notice = sig.get("notice_period_days")
    if isinstance(notice, (int, float)):
        if notice <= b["notice_preferred_days"]:
            delta += 0.02
        elif notice > b["notice_acceptable_days"]:
            delta -= 0.05; facts.append(f"{int(notice)}-day notice")

    mult = float(np.clip(1.0 + delta, b["modifier_floor"], b["modifier_ceiling"]))
    return round(mult, 4), facts


# ---------------------------------------------------------------------------
# component assembly + final score
# ---------------------------------------------------------------------------

def components(raw: dict, rubric: dict, semantic_fit: float) -> Dict[str, float]:
    tax = features.role_coherence_taxonomy(raw, rubric)
    # role_coherence blends taxonomy (dominant) with semantic fit
    role_coherence = round(0.7 * tax + 0.3 * semantic_fit, 4)
    return {
        "semantic_fit": round(float(semantic_fit), 4),
        "role_coherence": role_coherence,
        "career_evidence": features.career_evidence(raw, rubric),
        "experience_fit": features.experience_fit(raw, rubric),
        "trust_skills": features.trust_skills(raw, rubric),
    }


def base_score(comps: Dict[str, float], rubric: dict, drop: str = None) -> float:
    """Weighted sum of components. `drop` ablates one component (re-normalising)."""
    w = dict(rubric["component_weights"])
    w.pop("_comment", None)
    if drop and drop in w:
        w.pop(drop)
    total_w = sum(w.values())
    s = sum(comps[k] * wv for k, wv in w.items())
    return s / total_w if total_w else 0.0


def score_candidate(raw: dict, rubric: dict, semantic_fit: float, drop: str = None) -> dict:
    """Full scoring record for one candidate (feeds ranking + reasoning)."""
    comps = components(raw, rubric, semantic_fit)
    base = base_score(comps, rubric, drop=drop)

    hp, hp_reasons = honeypot.detect(raw)
    gate_mult, gate_reasons = gates.apply_gates(raw, rubric)
    beh_mult, beh_facts = behavioral_modifier(raw, rubric)

    if hp:
        final = 0.0
    else:
        final = base * gate_mult * beh_mult

    return {
        "candidate_id": loader.candidate_id(raw),
        "components": comps,
        "base": round(base, 4),
        "gate_mult": gate_mult,
        "gate_reasons": gate_reasons,
        "behavior_mult": beh_mult,
        "behavior_facts": beh_facts,
        "honeypot": hp,
        "honeypot_reasons": hp_reasons,
        "final_score": round(float(final), 6),
    }
