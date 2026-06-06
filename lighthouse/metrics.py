"""Ranking metrics: NDCG@k, MAP, P@k — matching the challenge spec.

The composite the organizers use:
    composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10

Relevance tiers are 0-5. Conventions used here (documented in eval/results.md):
  * NDCG uses exponential gain  (2^rel - 1) / log2(rank+1)  — the common form.
  * "relevant" for P@k and MAP means tier >= 3 (the spec: 'P@10 counts tier-3+
    as relevant'); we apply the same threshold to MAP for consistency.
All functions take `rels`, the true relevance tiers in the system's ranked order.
"""
from __future__ import annotations

import math
from typing import List

RELEVANT_TIER = 3


def dcg_at_k(rels: List[float], k: int) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_at_k(rels: List[float], k: int) -> float:
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def precision_at_k(rels: List[float], k: int, threshold: int = RELEVANT_TIER) -> float:
    if k == 0:
        return 0.0
    topk = rels[:k]
    return sum(1 for r in topk if r >= threshold) / k


def average_precision(rels: List[float], threshold: int = RELEVANT_TIER) -> float:
    """AP over the ranked list; R = total relevant in the list."""
    n_rel = sum(1 for r in rels if r >= threshold)
    if n_rel == 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, r in enumerate(rels, start=1):
        if r >= threshold:
            hits += 1
            score += hits / i
    return score / n_rel


def evaluate_ranking(rels: List[float]) -> dict:
    """All metrics + the spec composite for one ranked list of true tiers."""
    ndcg10 = ndcg_at_k(rels, 10)
    ndcg50 = ndcg_at_k(rels, 50)
    mapv = average_precision(rels)
    p10 = precision_at_k(rels, 10)
    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * mapv + 0.05 * p10
    return {
        "NDCG@10": round(ndcg10, 4), "NDCG@50": round(ndcg50, 4),
        "MAP": round(mapv, 4), "P@10": round(p10, 4),
        "P@5": round(precision_at_k(rels, 5), 4),
        "composite": round(composite, 4),
    }
