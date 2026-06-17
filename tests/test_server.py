"""Smoke tests for the FastAPI sandbox server.

These cover the static + metadata routes and input validation. The ``/api/rank``
happy path needs ``sentence-transformers`` plus a model download, so it is exercised
in local manual verification rather than CI; here we only assert that empty/invalid
input is rejected cleanly (no encoder involved).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from app.server import app  # noqa: E402

client = TestClient(app)


def test_index_serves_html() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Lighthouse" in r.text


def test_health_reports_artifacts() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["artifacts"] is True


def test_sample_returns_jsonl() -> None:
    r = client.get("/api/sample")
    assert r.status_code == 200
    first = r.text.strip().splitlines()[0]
    import json

    assert isinstance(json.loads(first), dict)


def test_rank_empty_input_is_400() -> None:
    r = client.post("/api/rank", json={"jsonl": "", "use_sample": False})
    assert r.status_code == 400


def test_rank_invalid_jsonl_is_400() -> None:
    r = client.post("/api/rank", json={"jsonl": "not-json\n{also bad", "use_sample": False})
    assert r.status_code == 400


def test_weights_endpoint_returns_defaults() -> None:
    r = client.get("/api/weights")
    assert r.status_code == 200
    body = r.json()
    assert "default_weights" in body
    w = body["default_weights"]
    assert set(w.keys()) == {
        "semantic_fit",
        "role_coherence",
        "career_evidence",
        "experience_fit",
        "trust_skills",
    }
    assert all(isinstance(v, (int, float)) and v >= 0 for v in w.values())
    assert sum(w.values()) > 0


def test_rank_rejects_unknown_weight_key() -> None:
    r = client.post(
        "/api/rank",
        json={"use_sample": True, "weights": {"semantic_fit": 0.2, "bogus": 0.1}},
    )
    assert r.status_code == 400
    assert "unknown" in r.json()["detail"].lower()


def test_rank_rejects_negative_weight() -> None:
    r = client.post(
        "/api/rank",
        json={
            "use_sample": True,
            "weights": {
                "semantic_fit": -0.1,
                "role_coherence": 0.0,
                "career_evidence": 0.0,
                "experience_fit": 0.0,
                "trust_skills": 0.0,
            },
        },
    )
    assert r.status_code == 400


def test_rank_rejects_zero_sum_weights() -> None:
    r = client.post(
        "/api/rank",
        json={
            "use_sample": True,
            "weights": {
                "semantic_fit": 0.0,
                "role_coherence": 0.0,
                "career_evidence": 0.0,
                "experience_fit": 0.0,
                "trust_skills": 0.0,
            },
        },
    )
    assert r.status_code == 400


def test_gates_endpoint_returns_catalog() -> None:
    r = client.get("/api/gates")
    assert r.status_code == 200
    body = r.json()
    assert "gates" in body
    keys = {g["key"] for g in body["gates"]}
    assert {
        "non_technical_role",
        "services_only",
        "location_visa",
        "research_only",
        "cv_speech_only",
        "langchain_only_recent",
        "title_chaser",
    } <= keys
    for g in body["gates"]:
        assert g["label"] and g["desc"]


def test_rank_rejects_unknown_gate_key() -> None:
    r = client.post(
        "/api/rank",
        json={"use_sample": True, "skip_gates": ["location_visa", "bogus"]},
    )
    assert r.status_code == 400
    assert "unknown" in r.json()["detail"].lower()


def test_rank_rejects_skip_gates_not_a_list() -> None:
    r = client.post(
        "/api/rank",
        json={"use_sample": True, "skip_gates": "location_visa"},
    )
    assert r.status_code in (400, 422)


def test_facets_endpoint_returns_defaults() -> None:
    r = client.get("/api/facets")
    assert r.status_code == 200
    body = r.json()
    assert "facets" in body and isinstance(body["facets"], list)
    assert "facet_weights" in body
    assert "default_facets" in body
    assert len(body["facets"]) == len(body["facet_weights"]) == len(body["default_facets"])


def test_facets_endpoint_rejects_too_few() -> None:
    r = client.post("/api/facets", json={"facets": ["only one"]})
    assert r.status_code == 400


def test_facets_endpoint_rejects_too_many() -> None:
    r = client.post("/api/facets", json={"facets": ["x"] * 30})
    assert r.status_code == 400


def test_facets_endpoint_rejects_weight_length_mismatch() -> None:
    r = client.post(
        "/api/facets",
        json={"facets": ["a", "b", "c"], "facet_weights": [1.0, 2.0]},
    )
    assert r.status_code == 400


def test_facets_endpoint_rejects_negative_weight() -> None:
    r = client.post(
        "/api/facets",
        json={"facets": ["a", "b", "c"], "facet_weights": [1.0, -0.5, 1.0]},
    )
    assert r.status_code == 400


def test_experience_endpoint_returns_defaults() -> None:
    r = client.get("/api/experience")
    assert r.status_code == 200
    body = r.json()
    for k in ("band_min", "band_max", "ideal_min", "ideal_max", "default"):
        assert k in body


def test_experience_endpoint_rejects_inverted_band() -> None:
    r = client.post(
        "/api/experience",
        json={"band_min": 10.0, "band_max": 4.0, "ideal_min": 5.0, "ideal_max": 7.0},
    )
    assert r.status_code == 400


def test_experience_endpoint_accepts_valid_override() -> None:
    r = client.post(
        "/api/experience",
        json={"band_min": 3.0, "band_max": 7.0, "ideal_min": 4.0, "ideal_max": 6.0},
    )
    assert r.status_code == 200
    # restore for downstream tests
    client.post("/api/reset")


def test_reset_endpoint_returns_ok() -> None:
    r = client.post("/api/reset")
    assert r.status_code == 200
    assert r.json() == {"reset": True}


def test_similar_endpoint_rejects_unknown_candidate() -> None:
    r = client.post(
        "/api/similar",
        json={"use_sample": True, "candidate_id": "CAND_DOES_NOT_EXIST", "limit": 5},
    )
    assert r.status_code == 404


def test_similar_endpoint_rejects_bad_limit() -> None:
    r = client.post(
        "/api/similar",
        json={"use_sample": True, "candidate_id": "CAND_0000031", "limit": 0},
    )
    assert r.status_code == 400


def test_gates_catalog_includes_unbacked_expertise() -> None:
    r = client.get("/api/gates")
    assert r.status_code == 200
    keys = {g["key"] for g in r.json()["gates"]}
    assert "unbacked_expertise" in keys


def test_facets_from_text_extracts_bullets() -> None:
    jd = (
        "Senior AI Engineer\n"
        "- Production experience with embeddings-based retrieval and dense vector search.\n"
        "- Built and shipped at least one end-to-end ranking or recommendation system.\n"
        "- Strong Python engineering with attention to code quality and testing.\n"
        "- Hybrid retrieval, learning-to-rank, and relevance evaluation frameworks.\n"
    )
    r = client.post("/api/facets_from_text", json={"text": jd})
    assert r.status_code == 200
    facets = r.json()["facets"]
    assert 3 <= len(facets) <= 15
    assert any("embeddings" in f.lower() for f in facets)


def test_facets_from_text_rejects_empty() -> None:
    r = client.post("/api/facets_from_text", json={"text": "Too short."})
    assert r.status_code == 400


def test_similar_pool_endpoint_returns_results() -> None:
    r = client.post("/api/similar_pool", json={"candidate_id": "CAND_0000031", "limit": 5})
    if r.status_code == 503:
        # cand_emb.npy not present in this checkout; skip.
        return
    assert r.status_code == 200
    body = r.json()
    assert body["target"] == "CAND_0000031"
    assert "pool_size" in body
    assert 1 <= len(body["rows"]) <= 5
    for row in body["rows"]:
        assert row["candidate_id"] != "CAND_0000031"
        assert 0.0 <= row["similarity"] <= 1.0


def test_similar_pool_rejects_unknown() -> None:
    r = client.post(
        "/api/similar_pool",
        json={"candidate_id": "CAND_BOGUS_ZZZ", "limit": 5},
    )
    # 404 when pool loaded, 503 when artifact missing — either is a valid rejection.
    assert r.status_code in (404, 503)


def test_similar_pool_rejects_bad_limit() -> None:
    r = client.post(
        "/api/similar_pool",
        json={"candidate_id": "CAND_0000031", "limit": 0},
    )
    assert r.status_code in (400, 503)
