"""Regression tests for the fairness pass over the ranker.

Each test pins one of the five corrections so a future refactor can't quietly
re-introduce the original unfair behavior.
"""

from __future__ import annotations

from lighthouse import features, gates, honeypot, scoring
from tests.conftest import make_candidate

# ---------------------------------------------------------------------------
# 1. behavioral_modifier: -1 sentinel on recruiter_response_rate is neutral
# ---------------------------------------------------------------------------


def test_behavior_treats_negative_response_rate_as_neutral(rubric):
    c = make_candidate(redrob_signals={"recruiter_response_rate": -1})
    mult, facts = scoring.behavioral_modifier(c, rubric)
    # Pre-fix the -1 would have been read as <= weak_response_rate and applied
    # -0.12. Now it should not appear in facts at all.
    assert not any("recruiter response" in f for f in facts), facts


def test_behavior_still_penalizes_a_real_weak_response_rate(rubric):
    """The guard must not regress the real penalty — a legitimate 0.05 still fires."""
    c = make_candidate(redrob_signals={"recruiter_response_rate": 0.05})
    _, facts = scoring.behavioral_modifier(c, rubric)
    assert any("low" in f and "recruiter response" in f for f in facts)


# ---------------------------------------------------------------------------
# 2. behavioral_modifier: -1 sentinel on notice_period_days is neutral
# ---------------------------------------------------------------------------


def test_behavior_treats_negative_notice_as_neutral(rubric):
    base_mult, _ = scoring.behavioral_modifier(make_candidate(), rubric)
    c = make_candidate(redrob_signals={"recruiter_response_rate": 0.4, "notice_period_days": -1})
    sentinel_mult, _ = scoring.behavioral_modifier(c, rubric)
    # Reference candidate has notice=30 (preferred). Sentinel must not produce
    # a HIGHER multiplier — the +0.02 fast-notice bonus must NOT fire on -1.
    assert sentinel_mult <= base_mult + 1e-6


# ---------------------------------------------------------------------------
# 3. gate_langchain_only_recent: now requires recent wrapper usage
# ---------------------------------------------------------------------------


def test_langchain_gate_does_not_fire_on_old_wrapper_role(rubric):
    """A 2019 role mentioning LangChain + LlamaIndex must NOT trigger the
    'AI experience is only recent LangChain' penalty — pre-fix it did."""
    c = make_candidate(
        career_history=[
            {
                "company": "Old Co",
                "title": "Engineer",
                "start_date": "2019-01-01",
                "end_date": "2020-01-01",
                "duration_months": 12,
                "is_current": False,
                "industry": "Tech",
                "company_size": "201-500",
                "description": "Explored LangChain and LlamaIndex briefly.",
            }
        ],
        skills=[],
    )
    text = gates._career_text(c)
    mult, reason = gates.gate_langchain_only_recent(c, rubric, text)
    assert mult == 1.0, f"old wrapper role fired the gate: {reason}"


def test_langchain_gate_still_fires_on_recent_wrapper_only_career(rubric):
    """A current role that ONLY mentions LangChain/LlamaIndex with no depth
    terms anywhere in the career must still get the 0.65 penalty."""
    c = make_candidate(
        profile={"current_title": "AI Engineer", "summary": ""},
        career_history=[
            {
                "company": "New Co",
                "title": "AI Engineer",
                "start_date": "2025-09-01",
                "end_date": None,
                "duration_months": 9,
                "is_current": True,
                "industry": "Tech",
                "company_size": "11-50",
                "description": "Built LangChain and LlamaIndex prompt chains.",
            }
        ],
        skills=[],
    )
    text = gates._career_text(c)
    mult, reason = gates.gate_langchain_only_recent(c, rubric, text)
    assert mult < 1.0
    assert "wrapper" in reason.lower() or "langchain" in reason.lower()


# ---------------------------------------------------------------------------
# 4. classify_title: positive trumps negative when both substrings present
# ---------------------------------------------------------------------------


def test_classify_title_positive_trumps_negative(rubric):
    assert features.classify_title("Marketing Software Engineer", rubric) == "positive"


def test_classify_title_pure_marketing_still_negative(rubric):
    assert features.classify_title("Marketing Manager", rubric) == "negative"


def test_classify_title_strong_term_still_wins(rubric):
    assert features.classify_title("Senior Machine Learning Engineer", rubric) == "strong"


# ---------------------------------------------------------------------------
# 5. honeypot: missing duration_months is not "explicitly zero"
# ---------------------------------------------------------------------------


def test_honeypot_ignores_skills_missing_duration_field():
    """Three advanced/expert skills with the duration_months field OMITTED
    must NOT trigger the 'claimed expertise with zero usage' flag — pre-fix
    they did because the loader collapsed missing -> 0."""
    raw = {
        "candidate_id": "CAND_TEST_SPARSE",
        "profile": {"years_of_experience": 6.0},
        "career_history": [],
        "education": [],
        "skills": [
            {"name": "NLP", "proficiency": "expert", "endorsements": 12},
            {"name": "Retrieval", "proficiency": "advanced", "endorsements": 5},
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 8},
        ],
        "redrob_signals": {},
    }
    is_hp, reasons = honeypot.detect(raw)
    assert not any("0 months used" in r for r in reasons), reasons
    assert not is_hp


def test_skip_gates_bypasses_location_visa_only(rubric):
    """skip_gates={'location_visa'} drops only that gate's penalty; the
    other gates still fire normally."""
    # A Toronto candidate who won't relocate -> location gate fires 0.30.
    c = make_candidate(
        profile={
            "country": "Canada",
            "location": "Toronto, Ontario",
            "current_title": "ML Engineer",
        },
        redrob_signals={"willing_to_relocate": False, "recruiter_response_rate": 0.6},
    )
    with_gates = scoring.score_candidate(c, rubric, semantic_fit=0.6)
    without_loc = scoring.score_candidate(c, rubric, semantic_fit=0.6, skip_gates={"location_visa"})
    assert with_gates["gate_mult"] < 1.0
    assert any("toronto" in r.lower() for r in with_gates["gate_reasons"])
    # bypassing location_visa removes that gate's contribution
    assert without_loc["gate_mult"] >= with_gates["gate_mult"]
    assert not any("toronto" in r.lower() for r in without_loc["gate_reasons"])
    # final_score reflects the higher base path
    assert without_loc["final_score"] > with_gates["final_score"]


def test_skip_gates_default_none_is_identical_to_omitting(rubric):
    """skip_gates=None must produce byte-identical records to omitting it,
    so rank.py output is guaranteed unchanged."""
    c = make_candidate()
    a = scoring.score_candidate(c, rubric, semantic_fit=0.5)
    b = scoring.score_candidate(c, rubric, semantic_fit=0.5, skip_gates=None)
    assert a == b


def test_honeypot_still_flags_explicit_zero_duration_expert_claims():
    """Three advanced/expert skills with duration_months EXPLICITLY = 0 must
    still trigger the flag."""
    raw = {
        "candidate_id": "CAND_TEST_FAKE",
        "profile": {"years_of_experience": 6.0},
        "career_history": [],
        "education": [],
        "skills": [
            {"name": "NLP", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {
                "name": "Retrieval",
                "proficiency": "advanced",
                "endorsements": 0,
                "duration_months": 0,
            },
            {
                "name": "Embeddings",
                "proficiency": "expert",
                "endorsements": 0,
                "duration_months": 0,
            },
        ],
        "redrob_signals": {},
    }
    is_hp, reasons = honeypot.detect(raw)
    assert is_hp
    assert any("0 months used" in r for r in reasons)
