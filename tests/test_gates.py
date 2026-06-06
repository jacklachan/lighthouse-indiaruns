"""Hard-negative gate behavior on planted patterns + real candidates."""
from lighthouse import gates
from tests.conftest import make_candidate


def test_no_gate_for_strong_india_candidate(rubric):
    mult, reasons = gates.apply_gates(make_candidate(), rubric)
    assert mult == 1.0
    assert reasons == []


def test_location_gate_outside_india_no_relocate(rubric):
    c = make_candidate(
        profile={"country": "Canada", "location": "Toronto"},
        redrob_signals={"willing_to_relocate": False},
    )
    mult, reasons = gates.apply_gates(c, rubric)
    assert mult < 0.5
    assert any("relocate" in r for r in reasons)


def test_location_gate_outside_india_willing(rubric):
    c = make_candidate(
        profile={"country": "Germany", "location": "Berlin"},
        redrob_signals={"willing_to_relocate": True},
    )
    mult, reasons = gates.apply_gates(c, rubric)
    assert 0.7 < mult < 0.95  # soft visa penalty only


def test_non_technical_role_gate(rubric):
    c = make_candidate(
        profile={"current_title": "Accountant"},
        career_history=[{
            "company": "Acme", "title": "Accountant", "start_date": "2018-01-01",
            "end_date": None, "duration_months": 60, "is_current": True,
            "industry": "Finance", "company_size": "201-500",
            "description": "Managed ledgers and audits.",
        }],
    )
    mult, reasons = gates.apply_gates(c, rubric)
    assert mult <= 0.25
    assert any("non-engineering" in r for r in reasons)


def test_services_only_gate(rubric):
    c = make_candidate(
        profile={"current_company": "Infosys", "current_title": "Software Engineer"},
        career_history=[
            {"company": "Infosys", "title": "Software Engineer", "start_date": "2019-01-01",
             "end_date": None, "duration_months": 50, "is_current": True,
             "industry": "IT Services", "company_size": "10001+", "description": "delivery work"},
            {"company": "TCS", "title": "Software Engineer", "start_date": "2016-01-01",
             "end_date": "2019-01-01", "duration_months": 36, "is_current": False,
             "industry": "IT Services", "company_size": "10001+", "description": "delivery work"},
        ],
    )
    mult, reasons = gates.apply_gates(c, rubric)
    assert mult <= 0.5
    assert any("services" in r for r in reasons)


def test_services_gate_not_fired_with_prior_product(rubric):
    # currently at a services firm but with a product-company role -> JD says fine
    c = make_candidate(
        profile={"current_company": "Wipro", "current_title": "Software Engineer"},
        career_history=[
            {"company": "Wipro", "title": "Software Engineer", "start_date": "2022-01-01",
             "end_date": None, "duration_months": 30, "is_current": True,
             "industry": "IT Services", "company_size": "10001+", "description": "work"},
            {"company": "Flipkart", "title": "Software Engineer", "start_date": "2018-01-01",
             "end_date": "2022-01-01", "duration_months": 48, "is_current": False,
             "industry": "E-commerce", "company_size": "5001-10000", "description": "ranking work"},
        ],
    )
    mult, reasons = gates.apply_gates(c, rubric)
    assert not any("services" in r for r in reasons)


def test_title_chaser_gate_fires_on_escalation(rubric):
    titles = ["ML Engineer", "Senior ML Engineer", "Staff ML Engineer", "Principal ML Engineer"]
    roles = []
    for i, (yr, t) in enumerate(zip([2019, 2021, 2022, 2024], titles)):
        roles.append({
            "company": f"Co{i}", "title": t, "start_date": f"{yr}-01-01",
            "end_date": f"{yr+1}-02-01", "duration_months": 13, "is_current": False,
            "industry": "Tech", "company_size": "201-500", "description": "ml work",
        })
    c = make_candidate(career_history=roles)
    mult, reasons = gates.apply_gates(c, rubric)
    assert any("title-chaser" in r for r in reasons)


def test_title_chaser_not_fired_on_lateral_moves(rubric):
    # short tenures but lateral titles -> NOT title-chasing
    titles = ["RecSys Engineer", "Search Engineer", "NLP Engineer", "Applied ML Engineer"]
    roles = []
    for i, (yr, t) in enumerate(zip([2019, 2021, 2022, 2024], titles)):
        roles.append({
            "company": f"Co{i}", "title": t, "start_date": f"{yr}-01-01",
            "end_date": f"{yr+1}-02-01", "duration_months": 14, "is_current": False,
            "industry": "Tech", "company_size": "201-500", "description": "ranking work",
        })
    c = make_candidate(career_history=roles)
    mult, reasons = gates.apply_gates(c, rubric)
    assert not any("title-chaser" in r for r in reasons)


def test_real_toronto_candidate_penalized(rubric, sample_candidates):
    # CAND_0000001: Toronto, won't relocate -> location gate must fire hard
    mult, reasons = gates.apply_gates(sample_candidates["CAND_0000001"], rubric)
    assert mult < 0.5
    assert any("relocate" in r for r in reasons)


def test_real_operations_manager_penalized(rubric, sample_candidates):
    # CAND_0000002: Operations Manager, non-technical trajectory
    mult, reasons = gates.apply_gates(sample_candidates["CAND_0000002"], rubric)
    assert mult < 0.5
