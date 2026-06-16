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
