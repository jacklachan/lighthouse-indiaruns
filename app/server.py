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


def _rank(raws: list[dict]) -> dict:
    """Run the full ranker and shape the result for the frontend.

    Mirrors ``app/app.py``: score -> normalize by max (round 6) -> rank -> rows +
    summary + top-candidate breakdown.
    """
    st = _state()
    rubric = st["rubric"]
    art = st["art"]

    records = ranker.score_all(raws, art)
    mx = max((r["final_score"] for r in records), default=0.0)
    if mx > 0:
        for r in records:
            r["final_score"] = round(r["final_score"] / mx, 6)
    top = ranker.rank_records(records, top=len(records))

    raw_by_id = {loader.candidate_id(r): r for r in raws}
    rows = []
    for rec in top:
        raw = raw_by_id[rec["candidate_id"]]
        p = loader.get_profile(raw)
        rows.append(
            {
                "rank": rec["rank"],
                "candidate_id": rec["candidate_id"],
                "score": rec["final_score"],
                "title": loader._s(p, "current_title"),
                "country": loader._s(p, "country"),
                "yrs": loader._f(p, "years_of_experience"),
                "honeypot": bool(rec["honeypot"]),
                "reasoning": reasoning.generate(raw, rubric, rec),
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

    weights = {k: v for k, v in rubric.get("component_weights", {}).items() if k != "_comment"}
    return {
        "rows": rows,
        "summary": {
            "ranked": len(rows),
            "honeypots": n_hp,
            "top_score": rows[0]["score"] if rows else None,
        },
        "breakdown": breakdown,
        "weights": weights,
    }


class RankRequest(BaseModel):
    jsonl: str | None = None
    use_sample: bool = False


@app.get("/api/health")
def health() -> dict:
    ok = os.path.exists(os.path.join(ART, "jd_rubric.json")) and os.path.exists(SAMPLE)
    return {"status": "ok" if ok else "degraded", "artifacts": bool(ok)}


@app.get("/api/sample", response_class=PlainTextResponse)
def sample() -> str:
    with open(SAMPLE, encoding="utf-8") as f:
        return f.read()


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
    try:
        return _rank(raws)
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
