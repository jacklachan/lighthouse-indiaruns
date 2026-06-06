"""Reasoning generator: grounding (no hallucination), variation, rank-consistency."""
import re

from lighthouse import loader, reasoning, scoring
from tests.conftest import make_candidate


def _record(raw, rubric, rank, semantic=0.6):
    rec = scoring.score_candidate(raw, rubric, semantic)
    rec["rank"] = rank
    return rec


# JD role-mandate words the generator may reference (as requirements, not as
# claimed candidate skills) even when the candidate lacks them.
JD_MANDATE_WORDS = {
    "retrieval", "ranking", "search", "embeddings", "recommendation",
    "semantic search", "information retrieval", "learning to rank",
    "vector search", "relevance", "personalization",
}


def _profile_text(raw):
    p = loader.get_profile(raw)
    parts = [loader._s(p, "headline"), loader._s(p, "summary"),
             loader._s(p, "current_title"), loader._s(p, "current_company")]
    for h in loader.get_career(raw):
        parts += [h["title"], h["company"], h["description"]]
    parts += [s["name"] for s in loader.get_skills(raw)]
    return " ".join(parts).lower()


def test_no_skill_hallucination(rubric, sample_candidates):
    """Every JD skill term named in the reasoning (excluding generic role-mandate
    references) must trace to SOMETHING in the candidate's profile text — title,
    skill, employer, or role description. Nothing is invented."""
    vocab = (set(rubric["ai_core_skills"]) | set(rubric["jd_relevant_skills"])) - JD_MANDATE_WORDS
    for cid, raw in sample_candidates.items():
        ptext = _profile_text(raw)
        rec = _record(raw, rubric, rank=5)
        text = reasoning.generate(raw, rubric, rec).lower()
        for skill in vocab:
            if re.search(r"\b" + re.escape(skill) + r"\b", text):
                assert skill in ptext, \
                    f"{cid}: reasoning names '{skill}' absent from profile text"


def test_claimed_skills_strictly_grounded(rubric, sample_candidates):
    """The strongest invariant: every skill/company the generator *claims* the
    candidate has is present on the candidate's profile."""
    for cid, raw in sample_candidates.items():
        own = {s["name"].lower() for s in loader.get_skills(raw)}
        rec = _record(raw, rubric, rank=5)
        terms = reasoning.grounded_terms(raw, rubric, rec)
        for sk in terms["skills"]:
            assert sk.lower() in own, f"{cid}: claimed skill '{sk}' not on profile"


def test_companies_are_grounded(rubric, sample_candidates):
    for cid, raw in sample_candidates.items():
        rec = _record(raw, rubric, rank=5)
        terms = reasoning.grounded_terms(raw, rubric, rec)
        career_cos = {h["company"] for h in loader.get_career(raw)}
        for co in terms["companies"]:
            assert co in career_cos, f"{cid}: company '{co}' not in career history"


def test_reasoning_mentions_real_yoe(rubric):
    c = make_candidate(profile={"years_of_experience": 7.0})
    rec = _record(c, rubric, rank=3)
    assert "7 yrs" in reasoning.generate(c, rubric, rec)


def test_variation_across_candidates(rubric, sample_candidates):
    texts = []
    for i, (cid, raw) in enumerate(sample_candidates.items()):
        rec = _record(raw, rubric, rank=i + 1)
        texts.append(reasoning.generate(raw, rubric, rec))
    # at least 90% unique among the sample
    assert len(set(texts)) >= 0.9 * len(texts)


def test_rank_consistency_concern_for_gated(rubric, sample_candidates):
    # CAND_0000001 (Toronto/no-relocate) should carry an honest concern
    raw = sample_candidates["CAND_0000001"]
    rec = _record(raw, rubric, rank=40)
    text = reasoning.generate(raw, rubric, rec)
    assert "Concern" in text or "relocate" in text.lower()


def test_low_rank_filler_tone(rubric):
    # a weak candidate at rank 95 should read as adjacent/filler, not glowing
    c = make_candidate(
        profile={"current_title": "Accountant", "years_of_experience": 12.0},
        career_history=[{"company": "Acme", "title": "Accountant",
                         "start_date": "2014-01-01", "end_date": None, "duration_months": 100,
                         "is_current": True, "industry": "Finance", "company_size": "201-500",
                         "description": "ledgers"}],
        skills=[],
    )
    rec = _record(c, rubric, rank=95, semantic=0.1)
    text = reasoning.generate(c, rubric, rec).lower()
    assert any(w in text for w in ("adjacent", "limited", "non-engineering", "filler", "concern"))
