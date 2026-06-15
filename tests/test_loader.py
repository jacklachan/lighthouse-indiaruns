"""Defensive loader: safe accessors, date parsing, streaming, and sparse profiles.

These exercise the "never raise on a well-formed-but-sparse profile" guarantee the
loader documents — the path that lets the ranker survive the dataset's messy records.
"""

from lighthouse import loader


def test_safe_accessors_on_non_dict_return_default():
    assert loader._s(None, "x") == ""
    assert loader._f("not a dict", "x") == 0.0
    assert loader._i(None, "x") == 0
    assert loader._b([], "x") is False


def test_numeric_accessors_coerce_and_fall_back():
    d = {"a": "3.5", "b": "12", "c": "oops", "d": None}
    assert loader._f(d, "a") == 3.5
    assert loader._i(d, "b") == 12
    assert loader._f(d, "c") == 0.0  # unparseable -> default
    assert loader._i(d, "d") == 0  # None -> default
    assert loader._f(d, "missing", 1.0) == 1.0


def test_string_accessor_coerces_and_handles_none():
    assert loader._s({"n": 42}, "n") == "42"
    assert loader._s({"n": None}, "n", "def") == "def"
    assert loader._s({}, "missing", "fallback") == "fallback"


def test_bool_accessor_only_true_for_real_bool():
    assert loader._b({"x": True}, "x") is True
    assert loader._b({"x": 1}, "x") is False  # truthy int is not a bool
    assert loader._b({"x": "true"}, "x") is False


def test_parse_date_valid_and_invalid():
    assert loader.parse_date("2024-03-15").isoformat() == "2024-03-15"
    assert loader.parse_date("2024-03-15T08:00:00Z").isoformat() == "2024-03-15"
    assert loader.parse_date("") is None
    assert loader.parse_date(None) is None
    assert loader.parse_date("not-a-date") is None
    assert loader.parse_date(20240315) is None  # non-string


def test_iter_raw_skips_blank_and_malformed_lines(tmp_path):
    p = tmp_path / "cands.jsonl"
    p.write_text(
        '{"candidate_id": "CAND_1"}\n'
        "\n"  # blank line
        "{ not valid json }\n"  # malformed line
        '{"candidate_id": "CAND_2"}\n',
        encoding="utf-8",
    )
    rows = loader.load_all(str(p))
    assert [loader.candidate_id(r) for r in rows] == ["CAND_1", "CAND_2"]


def test_get_skills_filters_non_dicts_and_normalizes():
    raw = {"skills": [{"name": "Python", "proficiency": "EXPERT"}, "garbage", None]}
    skills = loader.get_skills(raw)
    assert len(skills) == 1
    assert skills[0]["name"] == "Python"
    assert skills[0]["proficiency"] == "expert"  # lowercased


def test_getters_on_missing_sections_return_empty():
    assert loader.get_skills({}) == []
    assert loader.get_career({}) == []
    assert loader.get_education({}) == []
    assert loader.get_signals({}) == {}
    assert loader.get_profile({}) == {}


def test_build_text_blob_sparse_and_populated():
    assert loader.build_text_blob({}) == ""
    raw = {"profile": {"headline": "Senior ML Engineer", "summary": "Builds rankers"}}
    blob = loader.build_text_blob(raw)
    assert "Senior ML Engineer" in blob
    assert "Builds rankers" in blob
