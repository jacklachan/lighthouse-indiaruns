"""Shared fixtures: load the JD rubric and the two planted sample candidates."""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
BUNDLE = os.path.join(
    ROOT, "[PUB] India_runs_data_and_ai_challenge",
    "India_runs_data_and_ai_challenge",
)


@pytest.fixture(scope="session")
def rubric():
    with open(os.path.join(ROOT, "artifacts", "jd_rubric.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_candidates():
    path = os.path.join(BUNDLE, "sample_candidates.json")
    if not os.path.exists(path):
        pytest.skip("sample_candidates.json not present")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {c["candidate_id"]: c for c in data}


def make_candidate(**over):
    """Minimal well-formed candidate; override pieces per test."""
    base = {
        "candidate_id": "CAND_9999999",
        "profile": {
            "anonymized_name": "Test Person", "headline": "Engineer",
            "summary": "An engineer.", "location": "Pune, Maharashtra",
            "country": "India", "years_of_experience": 7.0,
            "current_title": "Machine Learning Engineer", "current_company": "Swiggy",
            "current_company_size": "5001-10000", "current_industry": "Food Delivery",
        },
        "career_history": [{
            "company": "Swiggy", "title": "Machine Learning Engineer",
            "start_date": "2021-01-01", "end_date": None, "duration_months": 40,
            "is_current": True, "industry": "Food Delivery", "company_size": "5001-10000",
            "description": "Built ranking and retrieval systems with embeddings.",
        }],
        "education": [{"institution": "IIT", "degree": "B.Tech",
                       "field_of_study": "CS", "start_year": 2012, "end_year": 2016,
                       "grade": None, "tier": "tier_1"}],
        "skills": [{"name": "Embeddings", "proficiency": "expert",
                    "endorsements": 30, "duration_months": 48}],
        "redrob_signals": {
            "profile_completeness_score": 90, "signup_date": "2025-01-01",
            "last_active_date": "2026-05-20", "open_to_work_flag": True,
            "recruiter_response_rate": 0.8, "willing_to_relocate": True,
            "github_activity_score": 40, "offer_acceptance_rate": 0.6,
            "interview_completion_rate": 0.9, "verified_email": True,
            "verified_phone": True, "notice_period_days": 30,
            "skill_assessment_scores": {}, "preferred_work_mode": "hybrid",
        },
    }
    for k, v in over.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k].update(v)
        else:
            base[k] = v
    return base
