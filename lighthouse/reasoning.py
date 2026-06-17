"""Deterministic, fact-grounded reasoning generator (NO LLM at rank-time).

For each top-100 candidate we compose a 1-2 sentence justification that:
  * cites only REAL field values (years, title, named skills, signal numbers),
  * connects them to specific JD requirements,
  * honestly names gaps (gate concerns, behavioral risks),
  * varies by which factors dominate THIS candidate, and
  * matches its rank band in tone (confident high, "filler" honesty low).

Grounding guarantee: every skill or employer surfaced is pulled directly from
the candidate's own `skills` / `career_history` / `profile`, never invented.
`grounded_terms()` exposes exactly what was used so tests can verify it.

Contrastive / confidence overlays
---------------------------------
``generate(..., context=...)`` accepts an optional context dict for two
overlays that the brief asks for ("explainable") but a single per-candidate
template can't deliver:

* ``dominant_component`` — surfaces which of the five components actually
  drove this candidate's score, computed from the (weight × value)
  contribution.
* contrastive vs. a peer — when the caller passes a ``neighbor`` record (the
  candidate that ranked immediately above), we name the single largest
  component delta as the reason this one slots above / below.

Both are additive: when ``context`` is None the output is unchanged, so the
existing grounding tests are unaffected.
"""

from __future__ import annotations

from . import features, loader
from .constants import REFERENCE_DATE

# representative "build" words; only surfaced if present in the candidate's text
_EVIDENCE_WORDS = [
    ("ranking", "ranking"),
    ("learning to rank", "learning-to-rank"),
    ("recommendation", "recommendation"),
    ("recommender", "recommendation"),
    ("retrieval", "retrieval"),
    ("search", "search"),
    ("embedding", "embeddings"),
    ("semantic search", "semantic search"),
    ("personalization", "personalization"),
    ("information retrieval", "information retrieval"),
]


def _fmt_yoe(y: float) -> str:
    return f"{y:.1f}".rstrip("0").rstrip(".") + " yrs"


def extract_facts(raw: dict, rubric: dict, record: dict) -> dict:
    """Pull grounded facts used to compose reasoning."""
    p = loader.get_profile(raw)
    sig = loader.get_signals(raw)
    skills = loader.get_skills(raw)
    career = loader.get_career(raw)

    # JD-relevant skills the candidate actually has, strongest first
    relevant = set(rubric["jd_relevant_skills"])
    rel_present = [s for s in skills if s["name"].lower() in relevant]
    rel_present.sort(
        key=lambda s: (
            features.PROFICIENCY_WEIGHT.get(s["proficiency"], 0.4)
            * min(s["duration_months"] / 24.0, 1.0)
        ),
        reverse=True,
    )
    top_skills = [s["name"] for s in rel_present[:3]]

    # product (non-services) employers actually in the career history
    product_cos = []
    for h in career:
        if h["company"] and not features.is_services_company(h["company"], rubric):
            if h["company"] not in product_cos:
                product_cos.append(h["company"])

    # which "build" word is genuinely present in their text
    text = " ".join(
        [loader._s(p, "summary")] + [h["description"] + " " + h["title"] for h in career]
    ).lower()
    evidence_word = None
    for needle, label in _EVIDENCE_WORDS:
        if needle in text:
            evidence_word = label
            break

    la = loader.parse_date(sig.get("last_active_date"))
    days_active = (REFERENCE_DATE - la).days if la else None

    return {
        "yoe": loader._f(p, "years_of_experience"),
        "title": loader._s(p, "current_title"),
        "company": loader._s(p, "current_company"),
        "country": loader._s(p, "country"),
        "location": loader._s(p, "location"),
        "top_skills": top_skills,
        "product_cos": product_cos[:2],
        "evidence_word": evidence_word,
        "resp_rate": sig.get("recruiter_response_rate"),
        "days_active": days_active,
        "notice": sig.get("notice_period_days"),
        "relocate": sig.get("willing_to_relocate"),
        "n_relevant_skills": len(rel_present),
    }


def grounded_terms(raw: dict, rubric: dict, record: dict) -> dict[str, list[str]]:
    """The skills/companies the generator is allowed to mention (for tests)."""
    f = extract_facts(raw, rubric, record)
    return {"skills": f["top_skills"], "companies": f["product_cos"]}


def _positive_clause(f: dict, comps: dict, band: str) -> str:
    """Lead clause built from the candidate's strongest grounded evidence."""
    title = f["title"] or "Engineer"
    yoe = _fmt_yoe(f["yoe"])
    lead = f"{title} with {yoe}"

    # choose the dominant evidence to feature
    bits = []
    if f["evidence_word"] and comps["career_evidence"] >= 0.45:
        cos = f" at {', '.join(f['product_cos'])}" if f["product_cos"] else " at product companies"
        bits.append(f"built {f['evidence_word']} systems{cos}")
    elif f["product_cos"] and comps["role_coherence"] >= 0.6:
        bits.append(f"applied-ML track record at {', '.join(f['product_cos'])}")

    if f["top_skills"]:
        if len(f["top_skills"]) == 1:
            bits.append(f"hands-on with {f['top_skills'][0]}")
        else:
            bits.append(f"hands-on with {', '.join(f['top_skills'][:2])}")

    if not bits:
        # nothing strong to feature — keep it honest
        if comps["semantic_fit"] >= 0.5:
            bits.append("profile semantically adjacent to the retrieval/ranking mandate")
        else:
            bits.append("limited direct evidence of production retrieval/ranking work")

    body = "; ".join(bits)
    if band == "top":
        return f"{lead} — {body}."
    if band == "mid":
        return f"{lead}; {body}."
    # capitalize only the first letter (preserve proper-noun casing in company/skill names)
    body_cap = body[0].upper() + body[1:] if body else body
    return f"{lead}. {body_cap}."


def _concern_clause(f: dict, record: dict, band: str) -> str:
    """Honest concern/gap clause. Prefers the most material concern."""
    concerns = list(record.get("gate_reasons", []))
    concerns += [
        c
        for c in record.get("behavior_facts", [])
        if any(k in c for k in ("low", "inactive", "notice", "last active"))
    ]

    if record.get("honeypot"):
        return f" Flagged as anomalous: {record['honeypot_reasons'][0]}."

    if concerns:
        return " Concern: " + concerns[0] + "."

    # no concern -> reinforce with a real positive behavioral signal
    pos = []
    if isinstance(f["resp_rate"], (int, float)) and f["resp_rate"] >= 0.5:
        pos.append(f"{f['resp_rate']:.0%} recruiter response")
    if f["days_active"] is not None and f["days_active"] <= 60:
        pos.append(f"active {f['days_active']}d ago")
    if f["country"].lower() == "india":
        pos.append("India-based")
    elif f["relocate"]:
        pos.append("willing to relocate")
    if pos:
        s = ", ".join(pos)
        return " " + s[0].upper() + s[1:] + "."
    return ""


_COMPONENT_LABEL = {
    "semantic_fit": "semantic fit",
    "role_coherence": "role coherence",
    "career_evidence": "career evidence",
    "experience_fit": "experience fit",
    "trust_skills": "trust-weighted skills",
}


def _component_weights(rubric: dict) -> dict[str, float]:
    return {
        k: float(v) for k, v in rubric.get("component_weights", {}).items() if k in _COMPONENT_LABEL
    }


def _dominant_component(comps: dict, weights: dict) -> tuple[str, float] | None:
    """Return (key, weighted_contribution) of the component that drove the score,
    or None if no component is distinctly ahead. Distinct = strictly larger
    contribution than every other component by at least 5% of the leader."""
    contribs = [(k, weights.get(k, 0.0) * comps.get(k, 0.0)) for k in _COMPONENT_LABEL]
    contribs.sort(key=lambda x: x[1], reverse=True)
    if len(contribs) < 2 or contribs[0][1] <= 0:
        return None
    leader, second = contribs[0], contribs[1]
    if leader[1] - second[1] < 0.05 * leader[1]:
        return None
    return leader


def _confidence_clause(comps: dict, rubric: dict, record: dict, band: str) -> str:
    """Name the component that did the most work, with a value to anchor it."""
    if record.get("honeypot"):
        return ""
    weights = _component_weights(rubric)
    dom = _dominant_component(comps, weights)
    if not dom:
        return ""
    key, _ = dom
    val = comps.get(key, 0.0)
    # Only worth surfacing when the dominant component is actually strong.
    if val < 0.55:
        return ""
    label = _COMPONENT_LABEL[key]
    if band == "top":
        return f" Lifted by {label} ({val:.2f})."
    return f" Score driven by {label} ({val:.2f})."


def _near_miss_clause(raw: dict, rubric: dict, record: dict) -> str:
    """Surface a gate that almost fired but didn't, plus its margin.

    Useful for "close call" candidates: a recruiter scanning the table can
    see this row is one mid-tenure or one services-firm role away from a
    dampening penalty. Returns "" when nothing is close to firing.
    """
    if record.get("gate_reasons") or record.get("honeypot"):
        return ""  # already-explained concerns dominate
    msgs: list[str] = []
    # services_only nearly: >= 0.7 of roles at a services firm but not the
    # 0.999 the gate requires.
    sf = features.services_fraction(raw, rubric)
    if 0.7 <= sf < 0.999:
        msgs.append(f"services_fraction {sf:.2f} (gate fires at 1.00)")
    # title_chaser nearly: 3 roles with short avg tenure (gate wants >=4).
    stats = features.tenure_stats(raw)
    if stats["n_roles"] == 3 and 0 < stats["avg_tenure_months"] < 20:
        msgs.append(f"3 roles averaging {stats['avg_tenure_months']:.0f}mo (gate fires at 4 roles)")
    if not msgs:
        return ""
    return " Close call: " + msgs[0] + "."


def _contrastive_clause(record: dict, context: dict) -> str:
    """One-line ' vs. peer' grounded in the per-component delta."""
    neighbor = context.get("neighbor")
    if not isinstance(neighbor, dict):
        return ""
    n_id = neighbor.get("candidate_id")
    n_rank = neighbor.get("rank")
    if not n_id or n_rank is None:
        return ""

    my_comps = record.get("components", {})
    n_comps = neighbor.get("components", {})
    if not n_comps:
        return ""

    # Largest absolute per-component gap that points the right way (this
    # candidate is above neighbor when neighbor is ranked below it).
    direction = 1 if record.get("rank", 999) < n_rank else -1
    diffs = []
    for k in _COMPONENT_LABEL:
        d = (my_comps.get(k, 0.0) - n_comps.get(k, 0.0)) * direction
        diffs.append((k, d, my_comps.get(k, 0.0), n_comps.get(k, 0.0)))
    diffs.sort(key=lambda x: x[1], reverse=True)
    k, gap, mine, theirs = diffs[0]
    if gap < 0.08:
        return ""  # too close to call honestly
    label = _COMPONENT_LABEL[k]
    verb = "Edges" if direction > 0 else "Trails"
    return f" {verb} {n_id} (rank {n_rank}) on {label} ({mine:.2f} vs {theirs:.2f})."


def generate(raw: dict, rubric: dict, record: dict, context: dict | None = None) -> str:
    """Compose the grounded reasoning for one ranked candidate.

    ``context`` is optional; when provided it may contain a ``neighbor``
    record (the candidate immediately above this one) for a contrastive
    overlay. The base output is unchanged when ``context`` is ``None``.
    """
    rank = record.get("rank", 999)
    band = "top" if rank <= 10 else ("mid" if rank <= 50 else "low")
    f = extract_facts(raw, rubric, record)
    comps = record["components"]

    text = _positive_clause(f, comps, band) + _concern_clause(f, record, band)

    if band == "low" and not record.get("gate_reasons") and not record.get("honeypot"):
        text += " Adjacent fit included near the cutoff."

    if context is not None:
        text += _confidence_clause(comps, rubric, record, band)
        text += _near_miss_clause(raw, rubric, record)
        text += _contrastive_clause(record, context)

    return " ".join(text.split()).strip()
