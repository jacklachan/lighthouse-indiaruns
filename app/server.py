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

import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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
OVERRIDES_PATH = os.path.join(ART, "sandbox_overrides.json")
_LOG = logging.getLogger("lighthouse.sandbox")
_EMB_CACHE_MAX = 3  # number of {jsonl_hash: embedding-matrix} entries to retain

sys.path.insert(0, ROOT)
from lighthouse import gates as _gates  # noqa: E402
from lighthouse import loader, ranker, reasoning  # noqa: E402

MAX_CANDIDATES = 100
COMPONENT_KEYS = (
    "semantic_fit",
    "role_coherence",
    "career_evidence",
    "experience_fit",
    "trust_skills",
)
GATE_KEYS = _gates.GATE_KEYS
# Human-readable labels + short descriptions for each gate, shown in the UI.
GATE_META: dict[str, dict[str, str]] = {
    "non_technical_role": {
        "label": "Non-technical role",
        "desc": "Current title + whole history non-engineering (the keyword-stuffer trap).",
    },
    "services_only": {
        "label": "Services-only career",
        "desc": "Every role at a consulting/services firm.",
    },
    "location_visa": {
        "label": "Location / visa",
        "desc": "Outside India; no visa sponsorship.",
    },
    "research_only": {
        "label": "Research-only",
        "desc": "Strong academic signal, no production deployment.",
    },
    "cv_speech_only": {
        "label": "CV / speech / robotics only",
        "desc": "Primary domain is CV/speech/robotics, no NLP / IR signal.",
    },
    "langchain_only_recent": {
        "label": "Recent LangChain only",
        "desc": "AI experience limited to recent LLM-wrapper tooling.",
    },
    "title_chaser": {
        "label": "Title-chaser",
        "desc": "Job-hops every ~18mo while titles escalate.",
    },
    "unbacked_expertise": {
        "label": "Unbacked expert skills",
        "desc": "5+ advanced/expert skills with none mentioned in career history or summary (keyword-stuffer trap).",
    },
}


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Boot-time wiring for the sandbox: load state, replay persisted overrides,
    and warm the encoder in a background thread so the first /api/facets call
    doesn't eat the ~10 s SentenceTransformer instantiation cost. All work is
    best-effort — a missing torch install or a malformed override file gets
    logged, not raised, so the server still serves the static UI."""
    try:
        _state()
    except Exception as ex:  # noqa: BLE001
        _LOG.warning("state init failed at startup: %s", ex)
        yield
        return
    try:
        _replay_overrides()
    except Exception as ex:  # noqa: BLE001
        _LOG.warning("override replay failed: %s", ex)

    def _warm() -> None:
        try:
            _get_encoder()
        except Exception as ex:  # noqa: BLE001
            _LOG.info("encoder warm skipped: %s", ex)

    threading.Thread(target=_warm, daemon=True, name="encoder-warm").start()
    yield


app = FastAPI(title="Lighthouse Candidate Ranker", version="1.0.0", lifespan=_lifespan)

# Artifacts + the on-the-fly art object are built once, on first use.
_STATE: dict = {}


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _facets_and_rubric() -> tuple[dict, np.ndarray, float | None, float | None]:
    """Load the JD rubric, facet embeddings, and population semantic bounds."""
    rubric = _read_json(os.path.join(ART, "jd_rubric.json"))
    facet_emb = np.load(os.path.join(ART, "jd_facet_emb.npy"))
    sem_lo = sem_hi = None
    meta_path = os.path.join(ART, "precompute_meta.json")
    if os.path.exists(meta_path):
        meta = _read_json(meta_path)
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


_STATE_LOCK = threading.Lock()


def _state() -> dict:
    # Guard on "art", not truthiness: _get_encoder() writes _STATE["_encoder"],
    # which would otherwise satisfy `if not _STATE` and skip the rubric/art load
    # (-> KeyError later). Lock + double-check prevents two first-requests from
    # double-loading. "art" is written LAST so any fast-path reader that sees it
    # sees a fully-populated _STATE.
    if "art" in _STATE:
        return _STATE
    with _STATE_LOCK:
        if "art" in _STATE:
            return _STATE
        rubric, facet_emb, sem_lo, sem_hi = _facets_and_rubric()
        _STATE["rubric"] = rubric
        # Snapshot for /api/reset.
        _STATE["default_rubric"] = json.loads(json.dumps(rubric))
        _STATE["default_facet_emb"] = facet_emb.copy()
        _STATE["default_facets"] = list(rubric.get("facets", []))
        _STATE["art"] = _build_art(rubric, facet_emb, sem_lo, sem_hi)
    return _STATE


_ENCODER_LOCK = threading.Lock()


def _encoder_model_name() -> str:
    meta_path = os.path.join(ART, "precompute_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f).get("model", "BAAI/bge-small-en-v1.5")
    return "BAAI/bge-small-en-v1.5"


def _get_encoder():
    """Lazy-load and cache the sentence-transformer. Thread-safe via lock so a
    concurrent first-call doesn't double-instantiate the ~130 MB model."""
    enc = _STATE.get("_encoder")
    if enc is not None:
        return enc
    with _ENCODER_LOCK:
        enc = _STATE.get("_encoder")
        if enc is not None:
            return enc
        from sentence_transformers import SentenceTransformer

        enc = SentenceTransformer(_encoder_model_name(), device="cpu")
        _STATE["_encoder"] = enc
        return enc


def _encode_facets(facets: list[str]) -> np.ndarray:
    """Encode JD facets with the cached precompute-time encoder.

    Bound to the sandbox path only — the offline rank-time runtime (rank.py +
    artifacts) never re-encodes facets.
    """
    model = _get_encoder()
    vecs = model.encode(facets, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Override persistence: survive a server restart by mirroring the user's
# facet/experience changes into a JSON file alongside the artifacts. Reset
# wipes the file. Engine code never reads this — it's purely sandbox state.
# ---------------------------------------------------------------------------


def _read_overrides_file() -> dict:
    if not os.path.exists(OVERRIDES_PATH):
        return {}
    try:
        with open(OVERRIDES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        _LOG.warning("could not read overrides file at %s", OVERRIDES_PATH)
        return {}


def _write_overrides_file(payload: dict) -> None:
    try:
        os.makedirs(ART, exist_ok=True)
        with open(OVERRIDES_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError:
        _LOG.warning("could not write overrides file at %s", OVERRIDES_PATH)


def _delete_overrides_file() -> None:
    try:
        if os.path.exists(OVERRIDES_PATH):
            os.remove(OVERRIDES_PATH)
    except OSError:
        _LOG.warning("could not delete overrides file at %s", OVERRIDES_PATH)


def _persist_current_overrides() -> None:
    """Snapshot current live overrides relative to defaults into the JSON file."""
    payload: dict = {}
    st = _STATE
    facets = st.get("facets")
    if facets and list(facets) != list(st.get("default_facets", [])):
        payload["facets"] = list(facets)
    fw = st["art"].get("facet_weights")
    if fw is not None:
        payload["facet_weights"] = list(fw)
    e_live = st["rubric"].get("experience", {})
    e_default = st.get("default_rubric", {}).get("experience", {})
    if any(
        e_live.get(k) != e_default.get(k)
        for k in ("band_min", "band_max", "ideal_min", "ideal_max")
    ):
        payload["experience"] = {
            k: e_live[k] for k in ("band_min", "band_max", "ideal_min", "ideal_max")
        }
    if payload:
        _write_overrides_file(payload)
    else:
        _delete_overrides_file()


def _replay_overrides() -> None:
    """Re-apply any persisted overrides on startup. Best-effort: a malformed
    payload is logged and ignored rather than failing the boot."""
    data = _read_overrides_file()
    if not data:
        return
    st = _state()  # forces default load
    try:
        if "facets" in data:
            facets = _validate_facets(data["facets"])
            emb = _encode_facets(facets)
            st["art"]["facet_emb"] = emb
            st["art"]["sem_lo"] = None
            st["art"]["sem_hi"] = None
            st["facets"] = facets
        if "facet_weights" in data:
            n = len(st.get("facets") or st["default_facets"])
            st["art"]["facet_weights"] = _validate_facet_weights(data["facet_weights"], n)
        if "experience" in data:
            e = data["experience"]
            st["rubric"]["experience"] = {**st["rubric"]["experience"], **e}
            st["art"]["rubric"] = st["rubric"]
        _LOG.info("replayed sandbox overrides from %s", OVERRIDES_PATH)
    except Exception as ex:  # noqa: BLE001
        _LOG.warning("failed to replay overrides (%s) — resetting to defaults", ex)
        st["art"]["facet_emb"] = st["default_facet_emb"].copy()
        st["art"]["facet_weights"] = None
        st["facets"] = list(st["default_facets"])
        st["rubric"] = json.loads(json.dumps(st["default_rubric"]))
        st["art"]["rubric"] = st["rubric"]
        _delete_overrides_file()


# ---------------------------------------------------------------------------
# Embedding cache for /api/similar: keep the last few encoded batches keyed
# by JSONL-content hash so a recruiter clicking 📡 on multiple rows of the
# same batch doesn't re-pay the per-call encoding latency.
# ---------------------------------------------------------------------------


def _jsonl_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _embed_cache_get(key: str) -> np.ndarray | None:
    cache = _STATE.setdefault("_emb_cache", [])
    for k, v in cache:
        if k == key:
            return v
    return None


def _embed_cache_put(key: str, value: np.ndarray) -> None:
    cache = _STATE.setdefault("_emb_cache", [])
    cache[:] = [(k, v) for k, v in cache if k != key]
    cache.insert(0, (key, value))
    del cache[_EMB_CACHE_MAX:]


_BULLET_SPLIT = re.compile(r"(?:^|\n)\s*(?:[-•\*·▪◦]|\d+[\.\)]|[a-z]\))\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[\.!?])\s+(?=[A-Z])")


def _facets_from_prose(text: str) -> list[str]:
    """Heuristic JD-prose → facet-list extractor.

    1. Split on bullet markers (-, •, 1., a)) if present — bullets are almost
       always one-facet-per-line in real JDs.
    2. Otherwise split on sentence terminators.
    3. Strip, drop near-empty (< 30 chars), drop boilerplate (< 5 words), cap
       individual facets at 400 chars, return at most 15.
    """
    text = (text or "").strip()
    if not text:
        return []
    chunks = _BULLET_SPLIT.split(text)
    if len(chunks) < 4:  # not bullet-formatted → fall back to sentence split
        chunks = []
        for line in text.splitlines():
            chunks.extend(_SENTENCE_SPLIT.split(line))
    out: list[str] = []
    for c in chunks:
        s = " ".join(c.split())  # collapse whitespace
        if len(s) < 30:
            continue
        if len(s.split()) < 5:
            continue
        if len(s) > 400:
            s = s[:397].rstrip() + "..."
        out.append(s)
        if len(out) >= 15:
            break
    return out


def _validate_facets(facets: list[str]) -> list[str]:
    if not isinstance(facets, list) or not (3 <= len(facets) <= 20):
        raise HTTPException(status_code=400, detail="facets must be a list of 3–20 strings")
    cleaned = []
    for f in facets:
        if not isinstance(f, str):
            raise HTTPException(status_code=400, detail="every facet must be a string")
        s = f.strip()
        if not s:
            raise HTTPException(status_code=400, detail="facet strings must not be empty")
        if len(s) > 400:
            raise HTTPException(status_code=400, detail="facet strings must be < 400 chars")
        cleaned.append(s)
    return cleaned


def _validate_facet_weights(weights: list[float] | None, n: int) -> list[float] | None:
    if weights is None:
        return None
    if not isinstance(weights, list) or len(weights) != n:
        raise HTTPException(status_code=400, detail=f"facet_weights must be a list of {n} numbers")
    out = []
    for w in weights:
        try:
            f = float(w)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="facet weight not numeric") from None
        if not math.isfinite(f) or f < 0:
            raise HTTPException(status_code=400, detail="facet weights must be finite and >= 0")
        out.append(f)
    if sum(out) <= 0:
        raise HTTPException(status_code=400, detail="facet weights must sum to > 0")
    return out


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
        if not math.isfinite(f) or f < 0:
            raise HTTPException(status_code=400, detail=f"weight {k} must be finite and >= 0")
        cleaned[k] = f
    for k in COMPONENT_KEYS:
        cleaned.setdefault(k, 0.0)
    if sum(cleaned.values()) <= 0:
        raise HTTPException(status_code=400, detail="weights must sum to > 0")
    return cleaned


def _validate_skip_gates(skip: object) -> set[str]:
    """Sanitize the caller-supplied list of gates to bypass."""
    if skip is None:
        return set()
    if not isinstance(skip, list):
        raise HTTPException(status_code=400, detail="skip_gates must be a list")
    cleaned: set[str] = set()
    for k in skip:
        if not isinstance(k, str):
            raise HTTPException(status_code=400, detail="skip_gates entries must be strings")
        if k not in GATE_KEYS:
            raise HTTPException(status_code=400, detail=f"unknown gate key: {k}")
        cleaned.add(k)
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


def _sample_emb_map() -> dict[str, np.ndarray]:
    """Precomputed embeddings for the fixed demo sample (``sample_cand_emb.npy``),
    sliced offline from the official ``cand_emb.npy`` by ``scripts/build_sample_emb.py``.

    Lets the sample / preset / weight / gate / experience-curve path rank with ZERO
    model inference: the vectors live in RAM and are never recomputed on CPU per
    request. This is what removes the ~95% CPU spike and sidesteps the encoder's
    meta-tensor load crash for the common path. Best-effort — a missing/mismatched
    file yields ``{}`` and the caller falls back to encoding.
    """
    cached = _STATE.get("_sample_emb")
    if cached is not None:
        return cached
    out: dict[str, np.ndarray] = {}
    path = os.path.join(ART, "sample_cand_emb.npy")
    try:
        if os.path.exists(path) and os.path.exists(SAMPLE):
            emb = np.load(path).astype(np.float32)
            with open(SAMPLE, encoding="utf-8") as f:
                ids = [loader.candidate_id(json.loads(ln)) for ln in f if ln.strip()]
            if emb.shape[0] == len(ids):
                out = {cid: emb[i] for i, cid in enumerate(ids)}
            else:
                _LOG.warning(
                    "sample_cand_emb rows %d != sample ids %d; ignoring", emb.shape[0], len(ids)
                )
    except Exception as ex:  # noqa: BLE001
        _LOG.warning("sample emb load failed: %s", ex)
    _STATE["_sample_emb"] = out
    return out


def _candidate_embeddings(raws: list[dict]) -> tuple[np.ndarray, dict[str, int]]:
    """Embedding matrix (row i -> raws[i]) + candidate_id -> row map for ``art``.

    Uses the RAM-resident precomputed sample vectors where available; only ids NOT
    in the sample (custom uploads) hit the encoder — and those are content-cached.
    Component weights, gate toggles and facet edits never change a candidate's
    embedding, so the sample path here is pure lookups: no model load, no CPU spike.
    Vectors are identical to encoding on the fly (same model, same build_text_blob),
    so the ranking is unchanged.
    """
    semb = _sample_emb_map()
    hits: dict[int, np.ndarray] = {}
    missing_idx: list[int] = []
    missing_blobs: list[str] = []
    for i, raw in enumerate(raws):
        v = semb.get(loader.candidate_id(raw))
        if v is not None:
            hits[i] = v
        else:
            missing_idx.append(i)
            missing_blobs.append(loader.build_text_blob(raw))

    enc = None
    if missing_blobs:
        key = _jsonl_hash("\x01".join(missing_blobs))
        enc = _embed_cache_get(key)
        if enc is None or enc.shape[0] != len(missing_blobs):
            model = _get_encoder()
            enc = model.encode(
                missing_blobs, normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)
            _embed_cache_put(key, enc)

    if hits:
        dim = int(next(iter(hits.values())).shape[0])
    elif enc is not None:
        dim = int(enc.shape[1])
    else:
        dim = int(_state()["art"]["facet_emb"].shape[1])

    out = np.zeros((len(raws), dim), dtype=np.float32)
    for i, vec in hits.items():
        out[i] = vec
    for j, i in enumerate(missing_idx):
        out[i] = enc[j]  # type: ignore[index]
    id_to_row = {loader.candidate_id(r): i for i, r in enumerate(raws)}
    return out, id_to_row


def _rank(
    raws: list[dict],
    weights: dict[str, float] | None = None,
    skip_gates: set[str] | None = None,
) -> dict:
    """Run the full ranker and shape the result for the frontend.

    Mirrors ``app/app.py``: score -> normalize by max (round 6) -> rank -> rows +
    summary + top-candidate breakdown.

    ``weights`` overrides the JD-default component weights for this request.
    ``skip_gates`` is a set of gate keys to bypass (Discovery UI; lets a
    recruiter ask "what if I'm willing to sponsor a visa?" without editing
    the rubric).
    """
    st = _state()
    rubric = st["rubric"]
    # Inject precomputed candidate embeddings so the sample path never loads the
    # encoder (removes the ~95% CPU spike + the meta-tensor crash). Per-call copy
    # picks up live facet edits (set_facets_endpoint mutates st["art"]["facet_emb"]);
    # the vectors are identical to on-the-fly encoding, so the ranking is unchanged.
    emb, id_to_row = _candidate_embeddings(raws)
    art = {**st["art"], "cand_emb": emb, "id_to_row": id_to_row}

    records = ranker.score_all(raws, art, skip_gates=skip_gates)
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
        # Contrastive + lift overlays. ``neighbor`` is the closest contender
        # BELOW so the clause reads "Edges CAND_X on role_coherence" (the brief's
        # money shot: why this one beats its twin). ``above`` is the row directly
        # above so the lift clause can name the cheapest component bump that
        # would overtake it ("+0.04 trust_skills would overtake CAND_Y").
        if i + 1 < len(top):
            neighbor = top[i + 1]
        elif i > 0:
            neighbor = top[i - 1]
        else:
            neighbor = None
        above = top[i - 1] if i > 0 else None
        if neighbor is None and above is None:
            context = None
        else:
            context = {"neighbor": neighbor, "above": above}
        rows.append(
            {
                "rank": rec["rank"],
                "candidate_id": rec["candidate_id"],
                "score": rec["final_score"],
                "title": loader._s(p, "current_title"),
                "country": loader._s(p, "country"),
                "yrs": loader._f(p, "years_of_experience"),
                "honeypot": bool(rec["honeypot"]),
                "gate_mult": rec["gate_mult"],
                "gate_reasons": rec["gate_reasons"],
                "reasoning": reasoning.generate(raw, rubric, rec, context=context),
                # Per-row scoring breakdown for the click-to-view detail modal
                # (already computed on rec; additive — ranking unchanged).
                "components": rec["components"],
                "base": rec["base"],
                "behavior_mult": rec["behavior_mult"],
                "behavior_facts": rec["behavior_facts"],
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
        "skip_gates": sorted(skip_gates) if skip_gates else [],
    }


class RankRequest(BaseModel):
    jsonl: str | None = None
    use_sample: bool = False
    weights: dict[str, float] | None = None
    skip_gates: list[str] | None = None


class SimilarRequest(BaseModel):
    jsonl: str | None = None
    use_sample: bool = False
    candidate_id: str
    limit: int = 10


class FacetsRequest(BaseModel):
    facets: list[str]
    facet_weights: list[float] | None = None


class ExperienceRequest(BaseModel):
    band_min: float
    band_max: float
    ideal_min: float
    ideal_max: float


class PoolSimilarRequest(BaseModel):
    candidate_id: str
    limit: int = 20


class FacetsFromTextRequest(BaseModel):
    text: str


@app.post("/api/facets_from_text")
def facets_from_text_endpoint(req: FacetsFromTextRequest) -> dict:
    """Parse a raw JD text into 3–15 facet sentences.

    Lets the Discovery UI accept JD prose ("paste a job ad") instead of
    requiring the recruiter to manually split it line-by-line. Returns the
    extracted list for review — the client then POSTs it to /api/facets to
    actually replace the live JD. Errors out cleanly when extraction yields
    too few usable sentences.
    """
    facets = _facets_from_prose(req.text)
    if len(facets) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Extracted only {len(facets)} usable facets — paste a longer JD or split it manually",
        )
    return {"facets": facets}


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


@app.get("/api/facets")
def facets_endpoint() -> dict:
    """Current JD facets and (optional) per-facet weights used by `semantic_fit`."""
    st = _state()
    facets = list(st.get("facets") or st["rubric"].get("facets", []))
    weights = list(st["art"].get("facet_weights") or [1.0] * len(facets))
    return {
        "facets": facets,
        "facet_weights": weights,
        "default_facets": list(st["default_facets"]),
    }


@app.post("/api/facets")
def set_facets_endpoint(req: FacetsRequest) -> dict:
    """Replace the JD facets (and optionally per-facet weights) in live state.

    Lets a recruiter rewrite the role description and immediately see ranking
    re-sort — the Discovery half of the brief. Encoding the new facets needs
    the same sentence-transformer the small-sample fallback uses, so the first
    call after model warmup is fast.
    """
    facets = _validate_facets(req.facets)
    weights = _validate_facet_weights(req.facet_weights, len(facets))
    try:
        emb = _encode_facets(facets)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"facet encoding failed: {e}") from e
    st = _state()
    # Replace facet embedding + label list. Drop the population semantic bounds
    # because they were calibrated against the OLD facets and would mis-normalize.
    st["art"]["facet_emb"] = emb
    st["art"]["facet_weights"] = weights
    st["art"]["sem_lo"] = None
    st["art"]["sem_hi"] = None
    st["facets"] = facets
    _persist_current_overrides()
    return {"facets": facets, "facet_weights": weights or [1.0] * len(facets)}


@app.get("/api/experience")
def experience_endpoint() -> dict:
    """Current experience curve from the rubric — feeds the `experience_fit` component."""
    st = _state()
    e = st["rubric"]["experience"]
    return {
        "band_min": e["band_min"],
        "band_max": e["band_max"],
        "ideal_min": e["ideal_min"],
        "ideal_max": e["ideal_max"],
        "default": {
            k: st["default_rubric"]["experience"][k]
            for k in ("band_min", "band_max", "ideal_min", "ideal_max")
        },
    }


@app.post("/api/experience")
def set_experience_endpoint(req: ExperienceRequest) -> dict:
    """Override the experience-fit curve in live state.

    Counterfactual lever: "what if the band is 3-7 instead of 5-9?" Validates
    that the band fully contains the ideal range and that all values are
    non-negative.
    """
    if not (0 <= req.band_min <= req.ideal_min <= req.ideal_max <= req.band_max):
        raise HTTPException(
            status_code=400,
            detail="require 0 <= band_min <= ideal_min <= ideal_max <= band_max",
        )
    st = _state()
    st["rubric"]["experience"] = {
        **st["rubric"]["experience"],
        "band_min": float(req.band_min),
        "band_max": float(req.band_max),
        "ideal_min": float(req.ideal_min),
        "ideal_max": float(req.ideal_max),
    }
    # Mirror into the art rubric so the engine sees the override too.
    st["art"]["rubric"] = st["rubric"]
    _persist_current_overrides()
    return {
        "band_min": req.band_min,
        "band_max": req.band_max,
        "ideal_min": req.ideal_min,
        "ideal_max": req.ideal_max,
    }


@app.post("/api/reset")
def reset_endpoint() -> dict:
    """Restore the live rubric + facet embeddings to the committed artifact defaults."""
    st = _state()
    st["rubric"] = json.loads(json.dumps(st["default_rubric"]))
    st["art"]["rubric"] = st["rubric"]
    st["art"]["facet_emb"] = st["default_facet_emb"].copy()
    st["art"]["facet_weights"] = None
    st["facets"] = list(st["default_facets"])
    # Re-read population semantic bounds.
    meta_path = os.path.join(ART, "precompute_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        st["art"]["sem_lo"] = meta.get("semantic_p5")
        st["art"]["sem_hi"] = meta.get("semantic_p95")
    _delete_overrides_file()
    return {"reset": True}


@app.get("/api/gates")
def gates_endpoint() -> dict:
    """Catalog of hard-negative gates the sandbox UI can toggle off."""
    return {
        "gates": [
            {"key": k, "label": GATE_META[k]["label"], "desc": GATE_META[k]["desc"]}
            for k in sorted(GATE_KEYS)
        ]
    }


def _read_jsonl_for_request(jsonl_field: str | None, use_sample: bool) -> list[dict]:
    """Resolve the request body to a parsed raw-candidate list, or raise 400.

    Used by both /api/rank and /api/similar so the parsing + emptiness handling
    stays in one place.
    """
    if use_sample:
        with open(SAMPLE, encoding="utf-8") as f:
            text = f.read()
    else:
        text = jsonl_field or ""
    raws = _parse_jsonl(text)
    if not raws:
        raise HTTPException(
            status_code=400,
            detail="No valid candidates found. Provide JSONL — one JSON object per line.",
        )
    return raws


def _similarity_note(target_raw: dict, other_raw: dict) -> str:
    """Tiny one-line hint of where two candidates align or diverge — built from
    grounded fields, never invented. Always returns something so the UI never
    has an empty column."""
    tp, op = loader.get_profile(target_raw), loader.get_profile(other_raw)
    t_skills = {s["name"].lower() for s in loader.get_skills(target_raw)}
    o_skills = {s["name"].lower() for s in loader.get_skills(other_raw)}
    shared = sorted(t_skills & o_skills)
    bits: list[str] = []
    if shared:
        bits.append("shared: " + ", ".join(shared[:3]))
    t_yoe = loader._f(tp, "years_of_experience")
    o_yoe = loader._f(op, "years_of_experience")
    if abs(t_yoe - o_yoe) >= 2:
        bits.append(f"YoE {o_yoe:.1f} vs {t_yoe:.1f}")
    t_country = loader._s(tp, "country") or "—"
    o_country = loader._s(op, "country") or "—"
    if t_country.lower() != o_country.lower():
        bits.append(f"country: {o_country}")
    return "; ".join(bits) if bits else "near-identical profile"


@app.post("/api/similar")
def similar_endpoint(req: SimilarRequest) -> dict:
    """Find candidates most similar to a target candidate by embedding cosine.

    Lets a recruiter say "show me others like CAND_0000031" — the JD-agnostic
    look-alike feature the Discovery brief implies. A per-batch embedding cache
    keyed by JSONL-content hash makes consecutive clicks instant after the
    first encode. Each returned row includes a grounded similarity note so the
    recruiter sees *why* the row is a look-alike (shared skills, YoE delta,
    different country) without expanding the table.
    """
    text = _jsonl_text_for_request(req.jsonl, req.use_sample)
    raws = _parse_jsonl(text)
    if not raws:
        raise HTTPException(
            status_code=400,
            detail="No valid candidates found. Provide JSONL — one JSON object per line.",
        )
    ids = [loader.candidate_id(r) for r in raws]
    if req.candidate_id not in ids:
        raise HTTPException(status_code=404, detail=f"candidate_id {req.candidate_id} not in batch")
    if req.limit < 1 or req.limit > MAX_CANDIDATES:
        raise HTTPException(status_code=400, detail="limit must be in [1, 100]")

    art = _state()["art"]
    cache_key = _jsonl_hash(text)
    emb = _embed_cache_get(cache_key)
    if emb is None:
        try:
            emb = ranker._embeddings_for(raws, art, None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"encoding failed: {e}") from e
        _embed_cache_put(cache_key, emb)

    target_idx = ids.index(req.candidate_id)
    # L2-normalized embeddings -> dot product equals cosine similarity.
    sims = emb @ emb[target_idx]
    order = np.argsort(-sims)

    target_raw = raws[target_idx]
    rows: list[dict] = []
    for k in order:
        if int(k) == target_idx:
            continue
        raw = raws[int(k)]
        p = loader.get_profile(raw)
        rows.append(
            {
                "rank": len(rows) + 1,
                "candidate_id": ids[int(k)],
                "similarity": round(float(sims[int(k)]), 4),
                "title": loader._s(p, "current_title"),
                "country": loader._s(p, "country"),
                "yrs": loader._f(p, "years_of_experience"),
                "note": _similarity_note(target_raw, raw),
            }
        )
        if len(rows) >= req.limit:
            break

    tp = loader.get_profile(target_raw)
    return {
        "target": {
            "candidate_id": req.candidate_id,
            "title": loader._s(tp, "current_title"),
            "country": loader._s(tp, "country"),
            "yrs": loader._f(tp, "years_of_experience"),
        },
        "rows": rows,
    }


_POOL_LOCK = threading.Lock()


def _get_pool() -> tuple[np.ndarray, list[str], dict[str, int]] | None:
    """Lazy-load the precomputed 100K candidate embeddings + id list. Returns
    None if either artifact is missing (CI / lean deploy). Cached in _STATE so
    the 100K × 384 float16 matrix is read from disk once per process.
    """
    pool = _STATE.get("_pool")
    if pool is not None:
        return pool
    with _POOL_LOCK:
        pool = _STATE.get("_pool")
        if pool is not None:
            return pool
        emb_path = os.path.join(ART, "cand_emb.npy")
        ids_path = os.path.join(ART, "candidate_ids.json")
        if not (os.path.exists(emb_path) and os.path.exists(ids_path)):
            return None
        emb = np.load(emb_path)
        with open(ids_path, encoding="utf-8") as f:
            ids = json.load(f)
        id_to_row = {cid: i for i, cid in enumerate(ids)}
        pool = (emb, ids, id_to_row)
        _STATE["_pool"] = pool
        return pool


@app.post("/api/similar_pool")
def similar_pool_endpoint(req: PoolSimilarRequest) -> dict:
    """Look-alike search over the full 100K candidate pool via precomputed
    embeddings — no JSONL upload required.

    Differs from /api/similar in that it operates on the precomputed
    `artifacts/cand_emb.npy` matrix instead of an uploaded batch, so a
    recruiter can ask "show me 20 candidates like CAND_0000031" without
    re-uploading data. Returns IDs + cosines only; titles/countries are not
    surfaced because the raw candidate data is gitignored (the grader supplies
    it). The UI can enrich by uploading the JSONL and calling /api/similar.
    """
    pool = _get_pool()
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="Pool embeddings not available — artifacts/cand_emb.npy missing",
        )
    emb, ids, id_to_row = pool
    if req.candidate_id not in id_to_row:
        raise HTTPException(
            status_code=404,
            detail=f"candidate_id {req.candidate_id} not in pool",
        )
    if req.limit < 1 or req.limit > 100:
        raise HTTPException(status_code=400, detail="limit must be in [1, 100]")

    idx = id_to_row[req.candidate_id]
    target = emb[idx].astype(np.float32)
    # Vectorised cosine on the L2-normalized matrix. float16 -> float32 for
    # numerical stability of the partial sort; 100K x 384 is ~150 MB intermediate.
    sims = (emb.astype(np.float32) @ target).astype(np.float32)
    k = min(req.limit + 1, len(ids))
    cand_indices = np.argpartition(-sims, kth=k - 1)[:k]
    cand_indices = cand_indices[np.argsort(-sims[cand_indices])]
    rows: list[dict] = []
    for i in cand_indices:
        if int(i) == idx:
            continue
        rows.append(
            {
                "rank": len(rows) + 1,
                "candidate_id": ids[int(i)],
                "similarity": round(float(sims[int(i)]), 4),
            }
        )
        if len(rows) >= req.limit:
            break
    return {"target": req.candidate_id, "rows": rows, "pool_size": len(ids)}


def _jsonl_text_for_request(jsonl_field: str | None, use_sample: bool) -> str:
    """Resolve raw JSONL text. Mirrors `_read_jsonl_for_request` but skips the
    parse step so /api/similar can hash the exact same text its cache was
    keyed against."""
    if use_sample:
        with open(SAMPLE, encoding="utf-8") as f:
            return f.read()
    return jsonl_field or ""


@app.post("/api/rank")
def rank_endpoint(req: RankRequest) -> dict:
    raws = _read_jsonl_for_request(req.jsonl, req.use_sample)
    weights = _validate_weights(req.weights) if req.weights is not None else None
    skip_gates = _validate_skip_gates(req.skip_gates) if req.skip_gates else set()
    try:
        return _rank(raws, weights=weights, skip_gates=skip_gates or None)
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
