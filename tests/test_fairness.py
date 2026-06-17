"""Regression tests for the fairness pass over the ranker.

Each test pins one of the five corrections so a future refactor can't quietly
re-introduce the original unfair behavior.
"""

from __future__ import annotations

from lighthouse import features, gates, honeypot, reasoning, scoring
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


def _neutral_signals():
    """Strip all positive signals so the modifier has headroom — otherwise the
    default make_candidate candidate is already at the ceiling and new bonuses
    cannot register."""
    return {
        "last_active_date": "2026-02-01",  # ~125d ago: -0.07 (room to grow)
        "recruiter_response_rate": 0.4,  # in the neutral band
        "open_to_work_flag": False,
        "verified_email": False,
        "verified_phone": False,
        "interview_completion_rate": 0.5,
        "notice_period_days": 60,
        "skill_assessment_scores": {},
    }


def test_behavior_rewards_recruiter_saves(rubric):
    """Previously unused: saved_by_recruiters_30d. A bookmarked profile
    should now get a small positive lift."""
    base = _neutral_signals()
    cold = make_candidate(redrob_signals={**base, "saved_by_recruiters_30d": 0})
    hot = make_candidate(redrob_signals={**base, "saved_by_recruiters_30d": 40})
    cm, _ = scoring.behavioral_modifier(cold, rubric)
    hm, hfacts = scoring.behavioral_modifier(hot, rubric)
    assert hm > cm
    assert any("recruiter saves" in f for f in hfacts)


def test_behavior_rewards_high_profile_views(rubric):
    base = _neutral_signals()
    quiet = make_candidate(redrob_signals={**base, "profile_views_received_30d": 10})
    hot = make_candidate(redrob_signals={**base, "profile_views_received_30d": 250})
    assert (
        scoring.behavioral_modifier(hot, rubric)[0] > scoring.behavioral_modifier(quiet, rubric)[0]
    )


def test_behavior_penalizes_slow_avg_response_time(rubric):
    base = _neutral_signals()
    fast = make_candidate(redrob_signals={**base, "avg_response_time_hours": 6})
    slow = make_candidate(redrob_signals={**base, "avg_response_time_hours": 200})
    fm, _ = scoring.behavioral_modifier(fast, rubric)
    sm, sfacts = scoring.behavioral_modifier(slow, rubric)
    assert fm > sm
    assert any("slow average response" in f for f in sfacts)


def test_behavior_ignores_negative_sentinel_on_new_signals(rubric):
    """All four new signals must respect the -1 sentinel rule the rest
    of behavioral_modifier already follows."""
    c = make_candidate(
        redrob_signals={
            "saved_by_recruiters_30d": -1,
            "profile_views_received_30d": -1,
            "avg_response_time_hours": -1,
            "endorsements_received": -1,
        }
    )
    _, facts = scoring.behavioral_modifier(c, rubric)
    for forbidden in ("recruiter saves", "slow average response"):
        assert not any(forbidden in f for f in facts), facts


def test_behavior_ignores_future_last_active_date(rubric):
    """A future-dated last_active must not earn the +0.05 recent-activity bonus.
    Pre-fix days = REFERENCE_DATE - future_date was negative and slid into the
    'days <= active_recent_days' branch."""
    c = make_candidate(redrob_signals={"last_active_date": "2030-01-01"})
    _, facts = scoring.behavioral_modifier(c, rubric)
    assert not any("active" in f and "0d" in f for f in facts), facts


def test_honeypot_flags_future_last_active_date():
    raw = {
        "candidate_id": "CAND_FAKE",
        "profile": {"years_of_experience": 5.0},
        "career_history": [],
        "education": [],
        "skills": [],
        "redrob_signals": {"last_active_date": "2030-01-01"},
    }
    is_hp, reasons = honeypot.detect(raw)
    assert is_hp
    assert any("last_active_date" in r for r in reasons)


def test_unbacked_expertise_gate_fires_on_keyword_stuffer(rubric):
    """5+ expert/advanced skills with NONE appearing in any career description
    or summary -> classic keyword-stuffer pattern; gate dampens (not zeros)."""
    c = make_candidate(
        profile={"summary": "Backend engineer with data pipeline experience."},
        career_history=[
            {
                "company": "Acme",
                "title": "Backend Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "SaaS",
                "company_size": "1001-5000",
                "description": "Maintained Kafka and Spark Streaming pipelines for telemetry.",
            }
        ],
        skills=[
            {"name": "NLP", "proficiency": "advanced", "endorsements": 5, "duration_months": 12},
            {
                "name": "Embeddings",
                "proficiency": "expert",
                "endorsements": 8,
                "duration_months": 14,
            },
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 3, "duration_months": 6},
            {
                "name": "Pinecone",
                "proficiency": "advanced",
                "endorsements": 2,
                "duration_months": 8,
            },
            {
                "name": "Fine-tuning LLMs",
                "proficiency": "expert",
                "endorsements": 4,
                "duration_months": 10,
            },
        ],
    )
    text = gates._career_text(c)
    mult, reason = gates.gate_unbacked_expertise(c, rubric, text)
    assert mult < 1.0, (mult, reason)
    assert "expert-claimed" in reason


def test_unbacked_expertise_gate_silent_when_skill_appears_in_role(rubric):
    """If even one of the claimed expert skills surfaces in a role description
    the gate should NOT fire — honest candidates omit some terms from bios but
    not all of them."""
    c = make_candidate(
        skills=[
            {"name": "NLP", "proficiency": "advanced", "endorsements": 5, "duration_months": 12},
            {
                "name": "Embeddings",
                "proficiency": "expert",
                "endorsements": 8,
                "duration_months": 14,
            },
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 3, "duration_months": 6},
            {
                "name": "Pinecone",
                "proficiency": "advanced",
                "endorsements": 2,
                "duration_months": 8,
            },
            {
                "name": "Fine-tuning LLMs",
                "proficiency": "expert",
                "endorsements": 4,
                "duration_months": 10,
            },
        ],
        career_history=[
            {
                "company": "Swiggy",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "Internet",
                "company_size": "5001-10000",
                "description": "Owned NLP-based query understanding for search.",
            }
        ],
    )
    text = gates._career_text(c)
    mult, _ = gates.gate_unbacked_expertise(c, rubric, text)
    assert mult == 1.0


def test_unbacked_expertise_silent_below_threshold(rubric):
    """Fewer than 5 expert claims -> gate doesn't fire (sparse-but-honest)."""
    c = make_candidate(
        skills=[
            {"name": "NLP", "proficiency": "expert", "endorsements": 5, "duration_months": 12},
            {
                "name": "Embeddings",
                "proficiency": "expert",
                "endorsements": 8,
                "duration_months": 14,
            },
        ],
        career_history=[
            {
                "company": "Acme",
                "title": "Backend Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "SaaS",
                "company_size": "1001-5000",
                "description": "Built data pipelines.",
            }
        ],
    )
    text = gates._career_text(c)
    mult, _ = gates.gate_unbacked_expertise(c, rubric, text)
    assert mult == 1.0


def test_career_evidence_rewards_recent_consistency(rubric):
    """Same role mix, but one candidate's strong-term roles are recent and the
    other's are 10 years stale. The recent one must score strictly higher."""
    recent_career = [
        {
            "company": "Swiggy",
            "title": "ML Engineer",
            "start_date": "2024-01-01",
            "end_date": None,
            "duration_months": 24,
            "is_current": True,
            "industry": "Internet",
            "company_size": "5001-10000",
            "description": "Built ranking, search, embeddings, retrieval, personalization at scale.",
        },
        {
            "company": "Flipkart",
            "title": "ML Engineer",
            "start_date": "2022-01-01",
            "end_date": "2024-01-01",
            "duration_months": 24,
            "is_current": False,
            "industry": "E-commerce",
            "company_size": "5001-10000",
            "description": "Shipped ranking + recommendation systems.",
        },
    ]
    stale_career = [
        {
            "company": "Swiggy",
            "title": "Engineer",
            "start_date": "2014-01-01",
            "end_date": "2016-01-01",
            "duration_months": 24,
            "is_current": False,
            "industry": "Internet",
            "company_size": "5001-10000",
            "description": "Built ranking, search, embeddings, retrieval, personalization at scale.",
        },
        {
            "company": "Flipkart",
            "title": "Engineer",
            "start_date": "2016-01-01",
            "end_date": "2018-01-01",
            "duration_months": 24,
            "is_current": False,
            "industry": "E-commerce",
            "company_size": "5001-10000",
            "description": "Shipped ranking + recommendation systems.",
        },
    ]
    recent = features.career_evidence(make_candidate(career_history=recent_career), rubric)
    stale = features.career_evidence(make_candidate(career_history=stale_career), rubric)
    assert recent > stale


def test_career_evidence_one_strong_old_role_still_counts(rubric):
    """A single foundational shipped system from 10 years ago is real evidence;
    the max term is undecayed so a veteran is not punished for tenure."""
    veteran = make_candidate(
        career_history=[
            {
                "company": "Flipkart",
                "title": "ML Engineer",
                "start_date": "2012-01-01",
                "end_date": "2014-01-01",
                "duration_months": 24,
                "is_current": False,
                "industry": "E-commerce",
                "company_size": "5001-10000",
                "description": "Shipped ranking, search, retrieval, embeddings, recommendation, learning to rank.",
            },
            {
                "company": "Acme",
                "title": "Manager",
                "start_date": "2014-01-01",
                "end_date": None,
                "duration_months": 100,
                "is_current": True,
                "industry": "Consulting",
                "company_size": "201-500",
                "description": "team management",
            },
        ]
    )
    score = features.career_evidence(veteran, rubric)
    # max term ~ 1.0 * 0.65 = 0.65; recency-weighted mean term is small but
    # non-negative, so the overall score should still clearly reflect real
    # past evidence.
    assert score >= 0.55


def test_seniority_level_uses_word_boundaries(rubric):
    """Title-chaser gate compares ``_seniority_level`` deltas, so a substring
    false-positive on 'leading' could fake an escalation pattern."""
    assert gates._seniority_level("Leading Engineer (Specialist)") == 0
    assert gates._seniority_level("Lead Engineer") == 2
    assert gates._seniority_level("Senior Engineer") == 1
    assert gates._seniority_level("Sr. ML Engineer") == 1
    assert gates._seniority_level("Engineering Manager") == 2
    assert gates._seniority_level("Head of Search") == 3
    assert gates._seniority_level("Principal Scientist") == 3


def test_is_services_company_robust_to_rubric_reorder():
    """The lookup must work regardless of where services_only sits in the
    hard_negatives list — guards against a positional [0] regression."""
    rub = {
        "hard_negatives": [
            {"key": "title_chaser"},  # deliberately not services_only
            {"key": "services_only", "companies": ["foo services"]},
        ]
    }
    assert features.is_services_company("Foo Services Pvt Ltd", rub)
    assert not features.is_services_company("Flipkart", rub)


def test_honeypot_flags_future_signup_date():
    raw = {
        "candidate_id": "CAND_FAKE",
        "profile": {"years_of_experience": 5.0},
        "career_history": [],
        "education": [],
        "skills": [],
        "redrob_signals": {"signup_date": "2030-01-01"},
    }
    is_hp, reasons = honeypot.detect(raw)
    assert is_hp
    assert any("signup_date" in r for r in reasons)


def test_education_signal_tier1_ai_field(rubric):
    """A tier_1 institution + AI/CS field of study lifts role_coherence
    (signals previously ignored). Comparing two otherwise-identical
    candidates so the delta is the education term."""
    tier1_ai = make_candidate(
        profile={"current_title": "Software Engineer"},
        education=[
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "tier": "tier_1",
            }
        ],
    )
    no_edu = make_candidate(
        profile={"current_title": "Software Engineer"},
        education=[],
    )
    assert features.role_coherence_taxonomy(tier1_ai, rubric) > features.role_coherence_taxonomy(
        no_edu, rubric
    )


def test_education_signal_does_not_overpower_negative_title(rubric):
    """Education boost is capped by 0.10 so a tier_1 CS background cannot
    rescue a plainly non-engineering current title + history."""
    accountant_iit = make_candidate(
        profile={"current_title": "Accountant"},
        career_history=[
            {
                "company": "Acme",
                "title": "Accountant",
                "start_date": "2018-01-01",
                "end_date": None,
                "duration_months": 100,
                "is_current": True,
                "industry": "Finance",
                "company_size": "201-500",
                "description": "ledgers",
            }
        ],
        education=[
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "tier": "tier_1",
            }
        ],
    )
    score = features.role_coherence_taxonomy(accountant_iit, rubric)
    # Accountant cur + Accountant hist -> 0.05 each, plus 0.10*1.0 education.
    # 0.55*0.05 + 0.35*0.05 + 0.10*1.0 = 0.145. Still strongly sub-fit.
    assert score < 0.20


def test_cert_signal_picks_up_ai_certs(rubric):
    raw = make_candidate(
        certifications=[
            {"name": "AWS Certified Machine Learning", "issuer": "Amazon", "year": 2024},
            {"name": "Deep Learning Specialization", "issuer": "deeplearning.ai", "year": 2023},
        ]
    )
    assert features.cert_signal(raw, rubric) > 0.5


def test_cert_signal_ignores_unrelated_certs(rubric):
    raw = make_candidate(
        certifications=[
            {"name": "Scrum Master Certified", "issuer": "Scrum Alliance", "year": 2024},
            {"name": "PMP", "issuer": "PMI", "year": 2022},
        ]
    )
    assert features.cert_signal(raw, rubric) == 0.0


def test_trust_skills_bumped_by_certs(rubric):
    plain = make_candidate(certifications=[])
    credentialed = make_candidate(
        certifications=[
            {"name": "AWS Certified Machine Learning", "issuer": "Amazon", "year": 2024},
            {"name": "TensorFlow Developer Certificate", "issuer": "Google", "year": 2023},
        ]
    )
    p = features.trust_skills(plain, rubric)
    c = features.trust_skills(credentialed, rubric)
    assert c > p
    # bump is capped at +0.10
    assert (c - p) <= 0.10 + 1e-6


def test_near_miss_clause_fires_on_services_borderline(rubric):
    """A candidate with services_fraction in the near-miss band [0.7, 0.85)
    does NOT trip the ramped services_only gate but should be flagged as a
    close call in the reasoning."""
    services_terms = ["Infosys", "TCS", "Wipro", "Cognizant", "Accenture"]
    career = []
    for co in services_terms * 2:  # 10 roles, will overwrite some to product
        career.append(
            {
                "company": co,
                "title": "Software Engineer",
                "start_date": "2018-01-01",
                "end_date": "2019-01-01",
                "duration_months": 12,
                "is_current": False,
                "industry": "IT Services",
                "company_size": "1001-5000",
                "description": "",
            }
        )
    # 8/10 services, 2/10 product -> services_fraction = 0.8 (in near-miss band)
    career[-1]["company"] = "Swiggy"
    career[-2]["company"] = "Flipkart"
    c = make_candidate(career_history=career)
    rec = scoring.score_candidate(c, rubric, semantic_fit=0.5)
    rec["rank"] = 30
    neighbor = scoring.score_candidate(make_candidate(), rubric, semantic_fit=0.5)
    neighbor["rank"] = 29
    text = reasoning.generate(c, rubric, rec, context={"neighbor": neighbor})
    assert rec["gate_mult"] == 1.0, "near-miss test candidate should not have fired gate"
    assert "close call" in text.lower(), text
    assert "services_fraction" in text.lower()


def test_services_gate_ramps_on_partial_services_career(rubric):
    """9/10 services + 1/10 product was previously unpenalized; now sits on
    the ramp between full-rubric penalty (0.45) and no penalty (1.0)."""
    services_co = ["Infosys", "TCS", "Wipro", "Cognizant", "Accenture"]
    career = []
    for co in services_co * 2:
        career.append(
            {
                "company": co,
                "title": "Software Engineer",
                "start_date": "2018-01-01",
                "end_date": "2019-01-01",
                "duration_months": 12,
                "is_current": False,
                "industry": "IT Services",
                "company_size": "1001-5000",
                "description": "",
            }
        )
    career[-1]["company"] = "Swiggy"  # services_fraction = 0.9
    c = make_candidate(career_history=career)
    text = gates._career_text(c)
    mult, reason = gates.gate_services_only(c, rubric, text)
    # 0.9 sits 1/3 of the way up the [0.85, 1.0] ramp; multiplier should be
    # between the rubric penalty (0.45) and 1.0, strictly less than 1.
    assert 0.45 < mult < 1.0, (mult, reason)
    assert "services" in reason.lower()


def test_services_gate_silent_when_one_real_product_role(rubric):
    """JD says someone at a services firm with prior product experience is
    fine. 5/10 product roles -> services_fraction = 0.5 -> no penalty."""
    career = []
    for co in ["Infosys", "TCS", "Wipro", "Cognizant", "Accenture"]:
        career.append(
            {
                "company": co,
                "title": "Software Engineer",
                "start_date": "2018-01-01",
                "end_date": "2019-01-01",
                "duration_months": 12,
                "is_current": False,
                "industry": "IT Services",
                "company_size": "1001-5000",
                "description": "",
            }
        )
    for co in ["Swiggy", "Flipkart", "Zomato", "Razorpay", "PhonePe"]:
        career.append(
            {
                "company": co,
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "duration_months": 12,
                "is_current": False,
                "industry": "Internet",
                "company_size": "1001-5000",
                "description": "",
            }
        )
    c = make_candidate(career_history=career)
    text = gates._career_text(c)
    mult, _ = gates.gate_services_only(c, rubric, text)
    assert mult == 1.0


def test_research_gate_does_not_fire_on_phd_with_product_career(rubric):
    """A published PhD with a real product-co role must NOT trip the
    research_only gate. Pre-fix the gate fired on research>=2 (trivial:
    'PhD' + 'publication') with no 'production' literal in the bio,
    even when career history showed a clear product employer."""
    c = make_candidate(
        profile={
            "summary": "PhD in machine learning, published two papers on retrieval.",
            "current_title": "Applied Scientist",
            "current_company": "Flipkart",
        },
        career_history=[
            {
                "company": "Flipkart",
                "title": "Applied Scientist",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 30,
                "is_current": True,
                "industry": "E-commerce",
                "company_size": "5001-10000",
                "description": "Worked on ranking models and embeddings.",
            }
        ],
    )
    text = gates._career_text(c)
    mult, reason = gates.gate_research_only(c, rubric, text)
    assert mult == 1.0, f"PhD + product career fired research gate: {reason}"


def test_research_gate_still_fires_on_pure_academic(rubric):
    """Postdoc with thesis + publications + academic, no product career, no
    production language -> gate must still apply the 0.55 penalty."""
    c = make_candidate(
        profile={
            "summary": "Postdoc with thesis on language models, multiple publications.",
            "current_title": "Postdoc Researcher",
            "current_company": "IIT Bombay",
        },
        career_history=[
            {
                "company": "IIT Bombay",
                "title": "Postdoc Researcher",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 30,
                "is_current": True,
                "industry": "Academia",
                "company_size": "10001+",
                "description": "Academic research on retrieval models; published papers at top venues.",
            },
            {
                "company": "IIT Bombay",
                "title": "Research Scholar",
                "start_date": "2018-01-01",
                "end_date": "2022-01-01",
                "duration_months": 48,
                "is_current": False,
                "industry": "Academia",
                "company_size": "10001+",
                "description": "PhD work on retrieval; thesis defended.",
            },
        ],
    )
    text = gates._career_text(c)
    mult, reason = gates.gate_research_only(c, rubric, text)
    assert mult < 1.0
    assert "research" in reason.lower()


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
