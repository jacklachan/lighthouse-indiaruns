"""Rank-time engine: load artifacts, score every candidate, sort, take top-N.

Pure numpy/pandas/Python. No network, no model inference over the full pool
(embeddings are precomputed). The only place a model may load is the small-sample
fallback (`_embeddings_for`) used by the HuggingFace app when it is handed
candidates that were never precomputed — never on the 100K official path.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import loader, scoring


# ---------------------------------------------------------------------------
# artifact loading
# ---------------------------------------------------------------------------

def load_artifacts(art_dir: str) -> dict:
    rubric = json.load(open(os.path.join(art_dir, "jd_rubric.json"), encoding="utf-8"))
    ids = json.load(open(os.path.join(art_dir, "candidate_ids.json"), encoding="utf-8"))
    cand_emb = np.load(os.path.join(art_dir, "cand_emb.npy"))
    facet_emb = np.load(os.path.join(art_dir, "jd_facet_emb.npy"))
    id_to_row = {cid: i for i, cid in enumerate(ids)}
    # fixed population semantic bounds (p5/p95) so small batches score stably
    sem_lo = sem_hi = None
    meta_path = os.path.join(art_dir, "precompute_meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        sem_lo, sem_hi = meta.get("semantic_p5"), meta.get("semantic_p95")
    return {"rubric": rubric, "ids": ids, "cand_emb": cand_emb,
            "facet_emb": facet_emb, "id_to_row": id_to_row,
            "sem_lo": sem_lo, "sem_hi": sem_hi}


def _embeddings_for(raws: List[dict], art: dict, model_name: Optional[str]) -> np.ndarray:
    """Return an embedding matrix aligned to `raws`, using precomputed vectors
    where available and encoding any missing ones on the fly (small-sample only).
    """
    cand_emb = art["cand_emb"]
    dim = cand_emb.shape[1]
    out = np.zeros((len(raws), dim), dtype=np.float32)
    missing_idx, missing_blobs = [], []
    for i, raw in enumerate(raws):
        row = art["id_to_row"].get(loader.candidate_id(raw))
        if row is not None:
            out[i] = cand_emb[row].astype(np.float32)
        else:
            missing_idx.append(i)
            missing_blobs.append(loader.build_text_blob(raw))
    if missing_blobs:
        # small-sample fallback (e.g. HF app) — load the small encoder lazily
        from sentence_transformers import SentenceTransformer
        meta_path = os.path.join("artifacts", "precompute_meta.json")
        mn = model_name or (json.load(open(meta_path))["model"]
                            if os.path.exists(meta_path) else "BAAI/bge-small-en-v1.5")
        model = SentenceTransformer(mn, device="cpu")
        vecs = model.encode(missing_blobs, normalize_embeddings=True, convert_to_numpy=True)
        for j, i in enumerate(missing_idx):
            out[i] = vecs[j]
    return out


# ---------------------------------------------------------------------------
# scoring all candidates
# ---------------------------------------------------------------------------

def score_all(raws: List[dict], art: dict, drop: str = None,
              model_name: Optional[str] = None, use_gates: bool = True,
              use_honeypot: bool = True, use_behavior: bool = True) -> List[dict]:
    rubric = art["rubric"]
    emb = _embeddings_for(raws, art, model_name)
    sem_raw = scoring.raw_semantic_fit(emb, art["facet_emb"])
    sem_norm = scoring.normalize_semantic(sem_raw, art.get("sem_lo"), art.get("sem_hi"))
    records = []
    for raw, sf in zip(raws, sem_norm):
        records.append(scoring.score_candidate(
            raw, rubric, float(sf), drop=drop, use_gates=use_gates,
            use_honeypot=use_honeypot, use_behavior=use_behavior))
    return records


# ---------------------------------------------------------------------------
# ranking + ordering (spec-compliant)
# ---------------------------------------------------------------------------

def rank_records(records: List[dict], top: int = 100) -> List[dict]:
    """Sort by final_score desc, tie-break candidate_id ascending; take top-N.

    Guarantees the validator's constraints: scores non-increasing by rank and
    equal scores ordered by candidate_id ascending.
    """
    ordered = sorted(records, key=lambda r: (-r["final_score"], r["candidate_id"]))
    top_recs = ordered[:top]
    for i, r in enumerate(top_recs):
        r["rank"] = i + 1
    return top_recs
