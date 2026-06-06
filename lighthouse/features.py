"""Deterministic per-candidate feature extraction.

These are the building blocks of the five scoring components that do NOT require
embeddings (role-taxonomy coherence, career evidence, experience fit, skill
trust). `semantic_fit` is computed in `scoring.py` from precomputed embeddings.

Every function is pure and deterministic. All text matching is lowercase
substring matching against term lists in the JD rubric, so the logic is fully
explainable and traceable back to the JD.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional

from . import loader

PROFICIENCY_WEIGHT = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return (s or "").lower()


def count_hits(text: str, terms: List[str]) -> int:
    t = _norm(text)
    return sum(1 for term in terms if term in t)


def any_hit(text: str, terms: List[str]) -> bool:
    t = _norm(text)
    return any(term in t for term in terms)


# ---------------------------------------------------------------------------
# title taxonomy
# ---------------------------------------------------------------------------

def classify_title(title: str, rubric: dict) -> str:
    """Return 'strong' | 'positive' | 'negative' | 'neutral' for a job title."""
    t = _norm(title)
    if not t:
        return "neutral"
    tax = rubric["role_taxonomy"]
    if any(term in t for term in tax["strong_positive_title_terms"]):
        return "strong"
    if any(term in t for term in tax["negative_title_terms"]):
        return "negative"
    if any(term in t for term in tax["positive_title_terms"]):
        return "positive"
    return "neutral"


_TITLE_SCORE = {"strong": 1.0, "positive": 0.7, "neutral": 0.4, "negative": 0.05}


def role_coherence_taxonomy(raw: dict, rubric: dict) -> float:
    """Coherence of titles with an AI/ML/IR/SWE-ranking trajectory, in [0,1].

    Blends the current title (weighted heavily) with the fraction of historical
    roles that are positive. This is the taxonomy half of `role_coherence`;
    `scoring.py` blends it with semantic fit.
    """
    p = loader.get_profile(raw)
    cur = classify_title(loader._s(p, "current_title"), rubric)
    cur_score = _TITLE_SCORE[cur]

    career = loader.get_career(raw)
    if career:
        hist_scores = [_TITLE_SCORE[classify_title(h["title"], rubric)] for h in career]
        hist = sum(hist_scores) / len(hist_scores)
    else:
        hist = cur_score
    return round(0.6 * cur_score + 0.4 * hist, 4)


# ---------------------------------------------------------------------------
# services / product company detection
# ---------------------------------------------------------------------------

def is_services_company(name: str, rubric: dict) -> bool:
    n = _norm(name)
    return any(c in n for c in rubric["hard_negatives"][0]["companies"])


def services_fraction(raw: dict, rubric: dict) -> float:
    career = loader.get_career(raw)
    if not career:
        return 0.0
    s = sum(1 for h in career if is_services_company(h["company"], rubric))
    return s / len(career)


# ---------------------------------------------------------------------------
# career evidence
# ---------------------------------------------------------------------------

def career_evidence(raw: dict, rubric: dict) -> float:
    """Did they actually BUILD ranking/search/recsys/retrieval at product cos? [0,1]

    Per role: term density over the description x a product-vs-services weight.
    Aggregate emphasises the single best role (JD: 'shipped at least one
    end-to-end ranking/search/recommendation system'), with a mean term so a
    consistently-strong career also benefits.
    """
    career = loader.get_career(raw)
    if not career:
        return 0.0
    terms = rubric["career_evidence_terms"]
    role_ev = []
    for h in career:
        hits = count_hits(h["description"] + " " + h["title"], terms)
        density = min(hits / 4.0, 1.0)
        if is_services_company(h["company"], rubric):
            product_w = 0.35
        elif h["company"]:
            product_w = 1.0
        else:
            product_w = 0.85
        role_ev.append(density * product_w)
    return round(0.65 * max(role_ev) + 0.35 * (sum(role_ev) / len(role_ev)), 4)


# ---------------------------------------------------------------------------
# experience fit (soft curve)
# ---------------------------------------------------------------------------

def experience_fit(raw: dict, rubric: dict) -> float:
    """Soft curve peaking inside ideal [6,8], in-band [5,9], gentle taper. [0,1]"""
    yoe = loader._f(loader.get_profile(raw), "years_of_experience")
    e = rubric["experience"]
    lo, hi = e["ideal_min"], e["ideal_max"]
    blo, bhi = e["band_min"], e["band_max"]
    if lo <= yoe <= hi:
        return 1.0
    if blo <= yoe < lo:
        return round(0.85 + 0.15 * (yoe - blo) / (lo - blo), 4)
    if hi < yoe <= bhi:
        return round(0.85 + 0.15 * (bhi - yoe) / (bhi - hi), 4)
    # outside band: decay (don't zero — strong signals can rescue)
    dist = (blo - yoe) if yoe < blo else (yoe - bhi)
    return round(max(0.15, 0.7 * math.exp(-dist / 3.0)), 4)


# ---------------------------------------------------------------------------
# skill trust
# ---------------------------------------------------------------------------

def trust_skills(raw: dict, rubric: dict) -> float:
    """Skills weighted by proficiency x duration x endorsements x assessment,
    counting only JD-relevant skills. Saturating. [0,1]

    This is what makes keyword-stuffing structurally worthless: an 'expert'
    skill with 0 months, no endorsements and a low Redrob assessment contributes
    almost nothing. Missing assessment is NEUTRAL (76% of pool has none).
    """
    skills = loader.get_skills(raw)
    if not skills:
        return 0.0
    sig = loader.get_signals(raw)
    assess = sig.get("skill_assessment_scores") or {}
    # build a case-insensitive assessment lookup
    assess_lc = {str(k).lower(): v for k, v in assess.items() if isinstance(v, (int, float))}

    relevant = set(rubric["jd_relevant_skills"])
    off = set(rubric["off_target_skills"])

    trust_sum = 0.0
    for sk in skills:
        name = _norm(sk["name"])
        if name in relevant:
            rel_w = 1.0
        elif name in off:
            rel_w = 0.15
        else:
            rel_w = 0.4
        if rel_w < 0.5:
            continue  # only JD-relevant skills build trust

        prof = PROFICIENCY_WEIGHT.get(sk["proficiency"], 0.4)
        dur_f = min(sk["duration_months"] / 24.0, 1.0)
        dur_w = 0.3 + 0.7 * dur_f
        end_f = min(sk["endorsements"] / 20.0, 1.0)
        end_w = 0.7 + 0.3 * end_f
        if name in assess_lc:
            assess_w = assess_lc[name] / 100.0
        else:
            assess_w = 0.65  # neutral when no assessment taken

        trust_sum += prof * dur_w * end_w * assess_w * rel_w

    # saturate: ~3 well-backed relevant skills -> ~0.78
    return round(1.0 - math.exp(-trust_sum / 2.0), 4)


# ---------------------------------------------------------------------------
# tenure stats (for title-chaser gate + reasoning)
# ---------------------------------------------------------------------------

def tenure_stats(raw: dict) -> Dict[str, float]:
    career = loader.get_career(raw)
    completed = [h["duration_months"] for h in career if not h["is_current"] and h["duration_months"] > 0]
    n_roles = len(career)
    avg_tenure = (sum(completed) / len(completed)) if completed else 0.0
    return {"n_roles": n_roles, "avg_tenure_months": round(avg_tenure, 1)}


# ---------------------------------------------------------------------------
# bundle: all non-embedding features for one candidate
# ---------------------------------------------------------------------------

def extract_features(raw: dict, rubric: dict) -> dict:
    """All deterministic (non-embedding) features for one candidate."""
    p = loader.get_profile(raw)
    feats = {
        "candidate_id": loader.candidate_id(raw),
        "yoe": loader._f(p, "years_of_experience"),
        "current_title": loader._s(p, "current_title"),
        "country": loader._s(p, "country"),
        "role_coherence_tax": role_coherence_taxonomy(raw, rubric),
        "career_evidence": career_evidence(raw, rubric),
        "experience_fit": experience_fit(raw, rubric),
        "trust_skills": trust_skills(raw, rubric),
        "services_fraction": services_fraction(raw, rubric),
    }
    feats.update(tenure_stats(raw))
    return feats
