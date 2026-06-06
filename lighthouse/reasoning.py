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
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Tuple

from . import features, loader

REFERENCE_DATE = date(2026, 6, 6)

# representative "build" words; only surfaced if present in the candidate's text
_EVIDENCE_WORDS = [
    ("ranking", "ranking"), ("learning to rank", "learning-to-rank"),
    ("recommendation", "recommendation"), ("recommender", "recommendation"),
    ("retrieval", "retrieval"), ("search", "search"), ("embedding", "embeddings"),
    ("semantic search", "semantic search"), ("personalization", "personalization"),
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
    rel_present.sort(key=lambda s: (features.PROFICIENCY_WEIGHT.get(s["proficiency"], 0.4)
                                    * min(s["duration_months"] / 24.0, 1.0)), reverse=True)
    top_skills = [s["name"] for s in rel_present[:3]]

    # product (non-services) employers actually in the career history
    product_cos = []
    for h in career:
        if h["company"] and not features.is_services_company(h["company"], rubric):
            if h["company"] not in product_cos:
                product_cos.append(h["company"])

    # which "build" word is genuinely present in their text
    text = " ".join([loader._s(p, "summary")] + [h["description"] + " " + h["title"] for h in career]).lower()
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


def grounded_terms(raw: dict, rubric: dict, record: dict) -> Dict[str, List[str]]:
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
    return f"{lead}. {body.capitalize()}."


def _concern_clause(f: dict, record: dict, band: str) -> str:
    """Honest concern/gap clause. Prefers the most material concern."""
    concerns = list(record.get("gate_reasons", []))
    concerns += [c for c in record.get("behavior_facts", [])
                 if any(k in c for k in ("low", "inactive", "notice", "last active"))]

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
        return " " + ", ".join(pos).capitalize() + "."
    return ""


def generate(raw: dict, rubric: dict, record: dict) -> str:
    """Compose the 1-2 sentence grounded reasoning for one ranked candidate."""
    rank = record.get("rank", 999)
    band = "top" if rank <= 10 else ("mid" if rank <= 50 else "low")
    f = extract_facts(raw, rubric, record)
    comps = record["components"]

    text = _positive_clause(f, comps, band) + _concern_clause(f, record, band)

    if band == "low" and not record.get("gate_reasons") and not record.get("honeypot"):
        text += " Adjacent fit included near the cutoff."

    return " ".join(text.split()).strip()
