"""Honeypot detection on planted-style impossibilities + real-candidate safety."""
from lighthouse import honeypot
from tests.conftest import make_candidate


def test_expert_with_zero_months_flags():
    c = make_candidate(skills=[
        {"name": "Embeddings", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "FAISS", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "NLP", "proficiency": "advanced", "endorsements": 0, "duration_months": 0},
    ])
    hp, reasons = honeypot.detect(c)
    assert hp
    assert any("0 months" in r for r in reasons)


def test_role_ends_before_it_starts_flags():
    c = make_candidate(career_history=[{
        "company": "Acme", "title": "ML Engineer", "start_date": "2023-01-01",
        "end_date": "2020-01-01", "duration_months": 12, "is_current": False,
        "industry": "Tech", "company_size": "51-200", "description": "work",
    }])
    hp, reasons = honeypot.detect(c)
    assert hp
    assert any("ends before" in r for r in reasons)


def test_duration_date_mismatch_flags():
    c = make_candidate(career_history=[{
        "company": "Acme", "title": "ML Engineer", "start_date": "2022-01-01",
        "end_date": "2022-06-01", "duration_months": 60, "is_current": False,  # 5mo span, 60mo claimed
        "industry": "Tech", "company_size": "51-200", "description": "work",
    }])
    hp, reasons = honeypot.detect(c)
    assert hp


def test_tenure_overflows_experience_flags():
    # 3 yrs experience but roles summing to 120 months
    c = make_candidate(
        profile={"years_of_experience": 3.0},
        career_history=[{
            "company": "A", "title": "ML Engineer", "start_date": "2016-01-01",
            "end_date": "2026-01-01", "duration_months": 120, "is_current": False,
            "industry": "Tech", "company_size": "51-200", "description": "work",
        }],
    )
    hp, reasons = honeypot.detect(c)
    assert hp
    assert any("impossible" in r or "sum" in r for r in reasons)


def test_education_invalid_flags():
    c = make_candidate(education=[{
        "institution": "X", "degree": "B.Tech", "field_of_study": "CS",
        "start_year": 2020, "end_year": 2016, "grade": None, "tier": "tier_2",
    }])
    hp, reasons = honeypot.detect(c)
    assert hp
    assert any("education" in r for r in reasons)


def test_clean_candidate_not_flagged():
    assert not honeypot.is_honeypot(make_candidate())


def test_real_strong_candidate_not_flagged(sample_candidates):
    # CAND_0000031 is the plain-language Tier-5; must never be a honeypot
    assert not honeypot.is_honeypot(sample_candidates["CAND_0000031"])


def test_real_nonfits_not_flagged_as_honeypot(sample_candidates):
    # The two planted non-fits are non-fits, not honeypots (dates are consistent)
    assert not honeypot.is_honeypot(sample_candidates["CAND_0000001"])
    assert not honeypot.is_honeypot(sample_candidates["CAND_0000002"])
