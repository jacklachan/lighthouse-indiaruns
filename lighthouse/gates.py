"""Hard-negative gates — multiplicative penalties derived from the JD rubric.

Each gate encodes one disqualifier the JD *explicitly* names. Gates are
multiplicative (not hard zeros — that is reserved for honeypots), so a strong
candidate with one soft concern is dampened, not annihilated, and the reasoning
generator can name the concern honestly.

Every gate returns at most one (key, multiplier, reason). `apply_gates` returns
the product of multipliers and the list of fired reasons.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from . import features, loader
from .constants import REFERENCE_DATE

_MIN_DATE = date(1900, 1, 1)
_RECENT_CUTOFF = REFERENCE_DATE - timedelta(days=18 * 30)

# Employer-name fragments and industry strings that indicate academia/research,
# used by the research_only gate so a postdoc at "IIT Bombay" is not counted
# as production deployment just because IIT is not on the services-firm list.
_ACADEMIC_NAME_TERMS = (
    "university",
    "institute of technology",
    "iit ",
    "iim ",
    "iisc",
    "academia",
    "research lab",
    "national lab",
)
_ACADEMIC_INDUSTRY_TERMS = ("academia", "education", "research", "university")


def _is_academic_employer(name: str, industry: str) -> bool:
    n = (name or "").lower() + " "  # trailing space lets "iit " match end-of-string
    i = (industry or "").lower()
    return any(t in n for t in _ACADEMIC_NAME_TERMS) or any(
        t in i for t in _ACADEMIC_INDUSTRY_TERMS
    )


def _career_text(raw: dict) -> str:
    parts = [loader._s(loader.get_profile(raw), "summary")]
    for h in loader.get_career(raw):
        parts.append(h["title"])
        parts.append(h["description"])
    parts += [s["name"] for s in loader.get_skills(raw)]
    return " ".join(parts).lower()


def _recent_career_text(raw: dict) -> str:
    """Concatenation of titles+descriptions from roles still active within the
    last ~18 months (current roles always count), plus the candidate's profile
    summary. Summary is included because it describes the candidate's *current*
    self-presentation — a "Recently built LangChain prompt chains" line in the
    summary is exactly the recent-wrapper signal the gate is meant to catch,
    even if it never appears in a role description.
    """
    parts: list[str] = [loader._s(loader.get_profile(raw), "summary")]
    for h in loader.get_career(raw):
        if h["is_current"]:
            parts.append(h["title"] + " " + h["description"])
            continue
        ed = h["end_date"]
        if ed and ed >= _RECENT_CUTOFF:
            parts.append(h["title"] + " " + h["description"])
    return " ".join(parts).lower()


def _gate(rubric: dict, key: str) -> dict:
    return next(g for g in rubric["hard_negatives"] if g["key"] == key)


def gate_services_only(raw: dict, rubric: dict, text: str) -> tuple[float, str]:
    """Multiplicative penalty for a career dominated by services/consulting firms.

    Old behavior was a cliff at 1.0 (10/10 services = 0.45; 9/10 = 1.0), so a
    single token product role neutralized the gate completely. The JD wording
    is "in their entire career" — but 9 services roles + 1 token product role
    is plainly closer to "entire" than to "balanced", so we ramp linearly:

      * services_fraction <  0.85           -> no penalty (one+ real product role)
      * services_fraction == 1.0            -> full rubric penalty (0.45)
      * services_fraction in [0.85, 1.0)    -> linear ramp between the two

    The 0.85 lower bound preserves the JD guidance that someone currently at a
    services firm with prior product experience should not be penalized.
    """
    g = _gate(rubric, "services_only")
    career = loader.get_career(raw)
    if not career:
        return 1.0, ""
    sf = features.services_fraction(raw, rubric)
    if sf < 0.85:
        return 1.0, ""
    cur = loader._s(loader.get_profile(raw), "current_company")
    if sf >= 0.999:
        return g["penalty"], f"entire career at services/consulting firms (e.g. {cur})"
    t = (sf - 0.85) / (1.0 - 0.85)
    mult = round(1.0 + (g["penalty"] - 1.0) * t, 4)
    return mult, f"{sf:.0%} of career at services/consulting firms (e.g. {cur})"


def gate_location_visa(raw: dict, rubric: dict, text: str) -> tuple[float, str]:
    g = _gate(rubric, "location_visa")
    p = loader.get_profile(raw)
    sig = loader.get_signals(raw)
    country = loader._s(p, "country").strip().lower()
    if country and country != "india":
        relocate = bool(sig.get("willing_to_relocate"))
        loc = loader._s(p, "location") or country.title()
        if not relocate:
            return (
                g["penalty"],
                f"based in {loc} ({country.title()}) and not willing to relocate; no visa sponsorship",
            )
        return (
            g["soft_penalty"],
            f"based outside India ({loc}); relocation willing but visa/logistics risk",
        )
    return 1.0, ""


def gate_research_only(raw: dict, rubric: dict, text: str) -> tuple[float, str]:
    """Fire only on genuinely research-pure profiles.

    Pre-fix: ``research >= 2 AND production == 0`` was trivially tripped by a
    real product engineer with a PhD ("PhD" + "publication" = 2 research hits),
    and the production check relied on candidates writing the literal words
    "production"/"shipped"/"deployed". A career at any non-services product
    company is itself production deployment, so we treat that as a structural
    production signal independent of vocabulary.
    """
    g = _gate(rubric, "research_only")
    research = features.count_hits(text, g["positive_terms"])
    production = features.count_hits(text, g["production_terms"])
    has_product_role = any(
        h["company"]
        and not features.is_services_company(h["company"], rubric)
        and not _is_academic_employer(h["company"], h["industry"])
        for h in loader.get_career(raw)
    )
    if research >= 3 and production == 0 and not has_product_role:
        return g["penalty"], "research-heavy profile with no evident production deployment"
    return 1.0, ""


def gate_cv_speech_only(raw: dict, rubric: dict, text: str) -> tuple[float, str]:
    g = _gate(rubric, "cv_speech_only")
    domain = features.count_hits(text, g["domain_terms"])
    nlp_ir = features.count_hits(text, g["nlp_ir_terms"])
    if domain >= 2 and nlp_ir == 0:
        return (
            g["penalty"],
            "primary expertise in computer vision/speech/robotics with no NLP/IR signal",
        )
    return 1.0, ""


def gate_langchain_only_recent(raw: dict, rubric: dict, text: str) -> tuple[float, str]:
    """Fire only when wrapper terms appear in *recent* (<=18mo) roles AND the
    full career text shows no classical-ML / retrieval depth. The recency arm
    enforces the rubric rule ('AI signals are recent') the old code skipped."""
    g = _gate(rubric, "langchain_only_recent")
    recent = _recent_career_text(raw)
    wrapper = features.count_hits(recent, g["wrapper_terms"])
    depth = features.count_hits(text, g["depth_terms"])
    if wrapper >= 2 and depth == 0:
        return (
            g["penalty"],
            "AI experience appears limited to recent LLM-wrapper tooling without classical ML/retrieval depth",
        )
    return 1.0, ""


_LEVEL_3_WORDS = ("principal", "staff", "director", "vp", "distinguished")
_LEVEL_2_WORDS = ("lead", "manager")
_LEVEL_1_WORDS = ("senior", "sr")
_WORD_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _has_word(text: str, word: str) -> bool:
    """Whole-word, case-insensitive match. Avoids 'lead' matching 'leading'."""
    pat = _WORD_RE_CACHE.get(word)
    if pat is None:
        pat = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
        _WORD_RE_CACHE[word] = pat
    return pat.search(text) is not None


def _seniority_level(title: str) -> int:
    """Coarse 0-3 seniority rank from a title.

    Pre-fix this used substring matching, so 'leading product engineer' or
    'engineering manager-track' would inflate the level via the 'lead'/'manager'
    substring. The title_chaser gate compares level deltas, so a single
    false-positive level shift could trip the escalation check on a steady
    lateral career. Word-boundary matching removes those false positives.
    """
    t = title.lower()
    if "head of" in t:
        return 3
    if any(_has_word(t, w) for w in _LEVEL_3_WORDS):
        return 3
    if any(_has_word(t, w) for w in _LEVEL_2_WORDS):
        return 2
    if any(_has_word(t, w) for w in _LEVEL_1_WORDS):
        return 1
    return 0


def gate_title_chaser(raw: dict, rubric: dict, text: str) -> tuple[float, str]:
    """Fire only on genuine title-chasing: short tenures AND an escalating
    seniority ladder. Lateral moves at short tenure (common for early-career ML
    engineers, e.g. RecSys -> Search -> NLP) are NOT title-chasing.
    """
    g = _gate(rubric, "title_chaser")
    stats = features.tenure_stats(raw)
    if not (
        stats["n_roles"] >= g["min_roles"]
        and 0 < stats["avg_tenure_months"] < g["max_avg_tenure_months"]
    ):
        return 1.0, ""
    # titles in chronological (oldest-first) order
    career = sorted(loader.get_career(raw), key=lambda h: (h["start_date"] or _MIN_DATE))
    levels = [_seniority_level(h["title"]) for h in career]
    escalated = (
        levels and (max(levels) - min(levels) >= 2) and levels[-1] >= levels[0] and levels[-1] >= 2
    )
    if escalated:
        return (
            g["penalty"],
            f"escalating titles while job-hopping every ~{stats['avg_tenure_months']:.0f} months across {stats['n_roles']} roles (title-chaser pattern)",
        )
    return 1.0, ""


_UNBACKED_DEFAULT_MIN = 5
_UNBACKED_DEFAULT_PENALTY = 0.65


def gate_unbacked_expertise(raw: dict, rubric: dict, text: str) -> tuple[float, str]:
    """Fire on the canonical keyword-stuffer trap: many advanced/expert skill
    claims with NONE of them appearing in any career role title/description or
    profile summary. A real practitioner uses their expert skills on the job
    and writes about it; a stuffer just lists keywords.

    Multiplicative penalty (not zero) — the dampen-don't-annihilate rule that
    distinguishes gates from honeypots. Configurable via the rubric:
    ``rubric['hard_negatives'][key='unbacked_expertise']`` may supply
    ``min_expert_claims`` and ``penalty``; defaults are 5 and 0.65.
    """
    g: dict = next(
        (x for x in rubric.get("hard_negatives", []) if x.get("key") == "unbacked_expertise"),
        {},
    )
    min_claims = int(g.get("min_expert_claims", _UNBACKED_DEFAULT_MIN))
    penalty = float(g.get("penalty", _UNBACKED_DEFAULT_PENALTY))

    raw_skills = raw.get("skills") or []
    expert_skills: list[str] = []
    for s in raw_skills:
        if not isinstance(s, dict):
            continue
        prof = (s.get("proficiency") or "").lower()
        if prof in ("advanced", "expert"):
            name = str(s.get("name") or "").strip()
            if name:
                expert_skills.append(name)
    if len(expert_skills) < min_claims:
        return 1.0, ""

    p = loader.get_profile(raw)
    haystack = (loader._s(p, "summary") + " ").lower()
    for h in loader.get_career(raw):
        haystack += (h["title"] + " " + h["description"] + " ").lower()
    ungrounded = [name for name in expert_skills if name.lower() not in haystack]
    if len(ungrounded) != len(expert_skills):
        return 1.0, ""
    return (
        penalty,
        f"{len(expert_skills)} expert-claimed skills never appear in any role "
        f"description or summary (e.g. {', '.join(expert_skills[:3])})",
    )


def gate_non_technical_role(raw: dict, rubric: dict, text: str) -> tuple[float, str]:
    g = _gate(rubric, "non_technical_role")
    p = loader.get_profile(raw)
    cur_class = features.classify_title(loader._s(p, "current_title"), rubric)
    if cur_class == "negative":
        # whole trajectory non-technical? (no strong AI role anywhere)
        career = loader.get_career(raw)
        has_strong = any(features.classify_title(h["title"], rubric) == "strong" for h in career)
        if not has_strong:
            return (
                g["penalty"],
                f"current role '{loader._s(p,'current_title')}' is non-engineering with no AI/ML role in career history",
            )
    return 1.0, ""


GATES: list[tuple[str, object]] = [
    ("non_technical_role", gate_non_technical_role),
    ("services_only", gate_services_only),
    ("location_visa", gate_location_visa),
    ("research_only", gate_research_only),
    ("cv_speech_only", gate_cv_speech_only),
    ("langchain_only_recent", gate_langchain_only_recent),
    ("title_chaser", gate_title_chaser),
    ("unbacked_expertise", gate_unbacked_expertise),
]

GATE_KEYS: set[str] = {key for key, _ in GATES}


def apply_gates(raw: dict, rubric: dict, skip: set[str] | None = None) -> tuple[float, list[str]]:
    """Return (combined_multiplier, [fired reasons]).

    ``skip`` is an optional set of gate keys to bypass — used by the Discovery
    UI so a recruiter can ask "what does the ranking look like if I'm willing
    to sponsor a visa?" without editing the rubric. When ``skip`` is None or
    empty the behavior is identical to the previous all-on path, so rank.py
    output is unchanged.
    """
    text = _career_text(raw)
    mult = 1.0
    reasons: list[str] = []
    skip = skip or set()
    for key, fn in GATES:
        if key in skip:
            continue
        m, reason = fn(raw, rubric, text)  # type: ignore[operator]
        if m < 1.0:
            mult *= m
            reasons.append(reason)
    return mult, reasons
