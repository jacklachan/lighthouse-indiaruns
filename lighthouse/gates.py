"""Hard-negative gates — multiplicative penalties derived from the JD rubric.

Each gate encodes one disqualifier the JD *explicitly* names. Gates are
multiplicative (not hard zeros — that is reserved for honeypots), so a strong
candidate with one soft concern is dampened, not annihilated, and the reasoning
generator can name the concern honestly.

Every gate returns at most one (key, multiplier, reason). `apply_gates` returns
the product of multipliers and the list of fired reasons.
"""
from __future__ import annotations

from datetime import date
from typing import List, Tuple

from . import features, loader

_MIN_DATE = date(1900, 1, 1)


def _career_text(raw: dict) -> str:
    parts = [loader._s(loader.get_profile(raw), "summary")]
    for h in loader.get_career(raw):
        parts.append(h["title"])
        parts.append(h["description"])
    parts += [s["name"] for s in loader.get_skills(raw)]
    return " ".join(parts).lower()


def _gate(rubric: dict, key: str) -> dict:
    return next(g for g in rubric["hard_negatives"] if g["key"] == key)


def gate_services_only(raw: dict, rubric: dict, text: str) -> Tuple[float, str]:
    g = _gate(rubric, "services_only")
    if features.services_fraction(raw, rubric) >= 0.999 and loader.get_career(raw):
        cur = loader._s(loader.get_profile(raw), "current_company")
        return g["penalty"], f"entire career at services/consulting firms (e.g. {cur})"
    return 1.0, ""


def gate_location_visa(raw: dict, rubric: dict, text: str) -> Tuple[float, str]:
    g = _gate(rubric, "location_visa")
    p = loader.get_profile(raw)
    sig = loader.get_signals(raw)
    country = loader._s(p, "country").strip().lower()
    if country and country != "india":
        relocate = bool(sig.get("willing_to_relocate"))
        loc = loader._s(p, "location") or country.title()
        if not relocate:
            return g["penalty"], f"based in {loc} ({country.title()}) and not willing to relocate; no visa sponsorship"
        return g["soft_penalty"], f"based outside India ({loc}); relocation willing but visa/logistics risk"
    return 1.0, ""


def gate_research_only(raw: dict, rubric: dict, text: str) -> Tuple[float, str]:
    g = _gate(rubric, "research_only")
    research = features.count_hits(text, g["positive_terms"])
    production = features.count_hits(text, g["production_terms"])
    if research >= 2 and production == 0:
        return g["penalty"], "research-heavy profile with no evident production deployment"
    return 1.0, ""


def gate_cv_speech_only(raw: dict, rubric: dict, text: str) -> Tuple[float, str]:
    g = _gate(rubric, "cv_speech_only")
    domain = features.count_hits(text, g["domain_terms"])
    nlp_ir = features.count_hits(text, g["nlp_ir_terms"])
    if domain >= 2 and nlp_ir == 0:
        return g["penalty"], "primary expertise in computer vision/speech/robotics with no NLP/IR signal"
    return 1.0, ""


def gate_langchain_only_recent(raw: dict, rubric: dict, text: str) -> Tuple[float, str]:
    g = _gate(rubric, "langchain_only_recent")
    wrapper = features.count_hits(text, g["wrapper_terms"])
    depth = features.count_hits(text, g["depth_terms"])
    if wrapper >= 2 and depth == 0:
        return g["penalty"], "AI experience appears limited to recent LLM-wrapper tooling without classical ML/retrieval depth"
    return 1.0, ""


def _seniority_level(title: str) -> int:
    t = title.lower()
    if any(k in t for k in ("principal", "staff", "director", "vp", "head of", "distinguished")):
        return 3
    if "lead" in t or "manager" in t:
        return 2
    if "senior" in t or "sr." in t or "sr " in t:
        return 1
    return 0


def gate_title_chaser(raw: dict, rubric: dict, text: str) -> Tuple[float, str]:
    """Fire only on genuine title-chasing: short tenures AND an escalating
    seniority ladder. Lateral moves at short tenure (common for early-career ML
    engineers, e.g. RecSys -> Search -> NLP) are NOT title-chasing.
    """
    g = _gate(rubric, "title_chaser")
    stats = features.tenure_stats(raw)
    if not (stats["n_roles"] >= g["min_roles"] and 0 < stats["avg_tenure_months"] < g["max_avg_tenure_months"]):
        return 1.0, ""
    # titles in chronological (oldest-first) order
    career = sorted(loader.get_career(raw), key=lambda h: (h["start_date"] or _MIN_DATE))
    levels = [_seniority_level(h["title"]) for h in career]
    escalated = levels and (max(levels) - min(levels) >= 2) and levels[-1] >= levels[0] and levels[-1] >= 2
    if escalated:
        return g["penalty"], f"escalating titles while job-hopping every ~{stats['avg_tenure_months']:.0f} months across {stats['n_roles']} roles (title-chaser pattern)"
    return 1.0, ""


def gate_non_technical_role(raw: dict, rubric: dict, text: str) -> Tuple[float, str]:
    g = _gate(rubric, "non_technical_role")
    p = loader.get_profile(raw)
    cur_class = features.classify_title(loader._s(p, "current_title"), rubric)
    if cur_class == "negative":
        # whole trajectory non-technical? (no strong AI role anywhere)
        career = loader.get_career(raw)
        has_strong = any(features.classify_title(h["title"], rubric) == "strong" for h in career)
        if not has_strong:
            return g["penalty"], f"current role '{loader._s(p,'current_title')}' is non-engineering with no AI/ML role in career history"
    return 1.0, ""


GATES = [
    gate_non_technical_role,
    gate_services_only,
    gate_location_visa,
    gate_research_only,
    gate_cv_speech_only,
    gate_langchain_only_recent,
    gate_title_chaser,
]


def apply_gates(raw: dict, rubric: dict) -> Tuple[float, List[str]]:
    """Return (combined_multiplier, [fired reasons])."""
    text = _career_text(raw)
    mult = 1.0
    reasons: List[str] = []
    for fn in GATES:
        m, reason = fn(raw, rubric, text)
        if m < 1.0:
            mult *= m
            reasons.append(reason)
    return mult, reasons
