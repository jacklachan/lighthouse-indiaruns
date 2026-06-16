"""Lighthouse — FastAPI server for the candidate-ranking sandbox.

Replaces the previous Streamlit demo with a custom animated frontend (``app/web/``)
backed by a thin JSON API that runs the existing Lighthouse ranker on CPU. The ranking
logic in ``lighthouse/`` is unchanged — this module only serves and wraps it, so the
ranked numbers match the official path exactly.

Endpoints
  GET  /              the single-page UI (app/web/index.html)
  GET  /api/health    liveness + artifact availability
  GET  /api/sample    preloaded sample candidates (JSONL text)
  POST /api/rank      rank candidates -> ranked rows + summary + top breakdown

Run locally:  uvicorn app.server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")
WEB = os.path.join(HERE, "web")
SAMPLE = os.path.join(HERE, "sample_candidates.jsonl")

sys.path.insert(0, ROOT)
from lighthouse import loader, ranker, reasoning  # noqa: E402

MAX_CANDIDATES = 100
COMPONENT_KEYS = (
    "semantic_fit",
    "role_coherence",
    "career_evidence",
    "experience_fit",
    "trust_skills",
)

app = FastAPI(title="Lighthouse Candidate Ranker", version="1.0.0")

# Artifacts + the on-the-fly art object are built once, on first use.
_STATE: dict = {}


def _facets_and_rubric() -> tuple[dict, np.ndarray, float | None, float | None]:
    """Load the JD rubric, facet embeddings, and population semantic bounds."""
    rubric = json.load(open(os.path.join(ART, "jd_rubric.json"), encoding="utf-8"))
    facet_emb = np.load(os.path.join(ART, "jd_facet_emb.npy"))
    sem_lo = sem_hi = None
    meta_path = os.path.join(ART, "precompute_meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        sem_lo, sem_hi = meta.get("semantic_p5"), meta.get("semantic_p95")
    return rubric, facet_emb, sem_lo, sem_hi


def _build_art(rubric, facet_emb, sem_lo, sem_hi) -> dict:
    """Empty precomputed set -> every candidate is encoded on the fly.

    Carries the fixed population semantic bounds so small uploads score stably,
    identical to the Streamlit app's behavior.
    """
    dim = facet_emb.shape[1]
    return {
        "rubric": rubric,
        "ids": [],
        "id_to_row": {},
        "cand_emb": np.zeros((0, dim), dtype=np.float32),
        "facet_emb": facet_emb,
        "sem_lo": sem_lo,
        "sem_hi": sem_hi,
    }


def _state() -> dict:
    if not _STATE:
        rubric, facet_emb, sem_lo, sem_hi = _facets_and_rubric()
        _STATE["rubric"] = rubric
        _STATE["art"] = _build_art(rubric, facet_emb, sem_lo, sem_hi)
    return _STATE


def _parse_jsonl(text: str) -> list[dict]:
    """Parse up to MAX_CANDIDATES JSON objects (one per line); skip blanks/bad lines."""
    raws: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            raws.append(obj)
        if len(raws) >= MAX_CANDIDATES:
            break
    return raws


def _default_weights(rubric: dict) -> dict[str, float]:
    return {
        k: float(v) for k, v in rubric.get("component_weights", {}).items() if k in COMPONENT_KEYS
    }


def _validate_weights(weights: dict) -> dict[str, float]:
    """Return a sanitized weights dict or raise 400. Defaults missing keys to 0."""
    if not isinstance(weights, dict):
        raise HTTPException(status_code=400, detail="weights must be an object")
    cleaned: dict[str, float] = {}
    for k, v in weights.items():
        if k not in COMPONENT_KEYS:
            raise HTTPException(status_code=400, detail=f"unknown weight key: {k}")
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"weight {k} not numeric") from None
        if f < 0:
            raise HTTPException(status_code=400, detail=f"weight {k} must be >= 0")
        cleaned[k] = f
    for k in COMPONENT_KEYS:
        cleaned.setdefault(k, 0.0)
    if sum(cleaned.values()) <= 0:
        raise HTTPException(status_code=400, detail="weights must sum to > 0")
    return cleaned


def _reweight_records(records: list[dict], weights: dict[str, float]) -> None:
    """In-place: recompute base + final per record using custom component weights.

    Preserves the existing gate_mult, behavior_mult, honeypot zeroing — only the
    weighted sum of components is overridden. The ranker package is not touched.
    """
    total_w = sum(weights.values())
    for r in records:
        comps = r["components"]
        s = sum(comps[k] * weights[k] for k in COMPONENT_KEYS)
        base = s / total_w
        r["base"] = round(base, 4)
        if r.get("honeypot"):
            r["final_score"] = 0.0
        else:
            r["final_score"] = round(base * r["gate_mult"] * r["behavior_mult"], 6)


def _rank(raws: list[dict], weights: dict[str, float] | None = None) -> dict:
    """Run the full ranker and shape the result for the frontend.

    Mirrors ``app/app.py``: score -> normalize by max (round 6) -> rank -> rows +
    summary + top-candidate breakdown.

    When ``weights`` is supplied, the JD-default component weights are replaced
    by the caller's overrides (post-scoring re-weighting; lighthouse/ unchanged).
    """
    st = _state()
    rubric = st["rubric"]
    art = st["art"]

    records = ranker.score_all(raws, art)
    if weights is not None:
        _reweight_records(records, weights)
    mx = max((r["final_score"] for r in records), default=0.0)
    if mx > 0:
        for r in records:
            r["final_score"] = round(r["final_score"] / mx, 6)
    top = ranker.rank_records(records, top=len(records))

    raw_by_id = {loader.candidate_id(r): r for r in raws}
    rows = []
    for i, rec in enumerate(top):
        raw = raw_by_id[rec["candidate_id"]]
        p = loader.get_profile(raw)
        # Contrastive overlay: the candidate immediately above is this row's peer.
        neighbor = top[i - 1] if i > 0 else None
        context = {"neighbor": neighbor} if neighbor is not None else None
        rows.append(
            {
                "rank": rec["rank"],
                "candidate_id": rec["candidate_id"],
                "score": rec["final_score"],
                "title": loader._s(p, "current_title"),
                "country": loader._s(p, "country"),
                "yrs": loader._f(p, "years_of_experience"),
                "honeypot": bool(rec["honeypot"]),
                "reasoning": reasoning.generate(raw, rubric, rec, context=context),
            }
        )

    n_hp = sum(1 for r in top if r["honeypot"])
    best = top[0] if top else None
    breakdown = None
    if best is not None:
        breakdown = {
            "candidate_id": best["candidate_id"],
            "components": best["components"],
            "base": best["base"],
            "gate_mult": best["gate_mult"],
            "gate_reasons": best["gate_reasons"],
            "behavior_mult": best["behavior_mult"],
            "honeypot": best["honeypot"],
        }

    effective_weights = weights if weights is not None else _default_weights(rubric)
    return {
        "rows": rows,
        "summary": {
            "ranked": len(rows),
            "honeypots": n_hp,
            "top_score": rows[0]["score"] if rows else None,
        },
        "breakdown": breakdown,
        "weights": effective_weights,
        "default_weights": _default_weights(rubric),
    }


class RankRequest(BaseModel):
    jsonl: str | None = None
    use_sample: bool = False
    weights: dict[str, float] | None = None


@app.get("/api/health")
def health() -> dict:
    ok = os.path.exists(os.path.join(ART, "jd_rubric.json")) and os.path.exists(SAMPLE)
    return {"status": "ok" if ok else "degraded", "artifacts": bool(ok)}


@app.get("/api/sample", response_class=PlainTextResponse)
def sample() -> str:
    with open(SAMPLE, encoding="utf-8") as f:
        return f.read()


@app.get("/api/weights")
def weights_endpoint() -> dict:
    """JD-default component weights from the committed rubric."""
    return {"default_weights": _default_weights(_state()["rubric"])}


@app.post("/api/rank")
def rank_endpoint(req: RankRequest) -> dict:
    if req.use_sample:
        with open(SAMPLE, encoding="utf-8") as f:
            text = f.read()
    else:
        text = req.jsonl or ""

    raws = _parse_jsonl(text)
    if not raws:
        raise HTTPException(
            status_code=400,
            detail="No valid candidates found. Provide JSONL — one JSON object per line.",
        )
    weights = _validate_weights(req.weights) if req.weights is not None else None
    try:
        return _rank(raws, weights=weights)
    except HTTPException:
        raise
    except Exception as e:  # surface a clean message instead of a 500 stack
        raise HTTPException(status_code=500, detail=f"Ranking failed: {e}") from e


# Static assets last so the API routes above take precedence.
if os.path.isdir(WEB):
    app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(WEB, "index.html"))
