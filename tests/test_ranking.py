"""Ranking order, tie-break, and end-to-end CSV validity (official validator)."""
import csv

import validate_submission
from lighthouse import ranker


def _recs(scores):
    return [{"candidate_id": f"CAND_{i:07d}", "final_score": s, "honeypot": False}
            for i, s in enumerate(scores, start=1)]


def test_scores_non_increasing_by_rank():
    import random
    random.seed(0)
    recs = _recs([round(random.random(), 4) for _ in range(300)])
    top = ranker.rank_records(recs, top=100)
    scores = [r["final_score"] for r in top]
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in top] == list(range(1, 101))


def test_tie_break_candidate_id_ascending():
    # all equal scores -> order must be candidate_id ascending
    recs = [{"candidate_id": cid, "final_score": 0.5, "honeypot": False}
            for cid in ["CAND_0000050", "CAND_0000003", "CAND_0000200", "CAND_0000001"]]
    top = ranker.rank_records(recs, top=4)
    assert [r["candidate_id"] for r in top] == \
        ["CAND_0000001", "CAND_0000003", "CAND_0000050", "CAND_0000200"]


def test_partial_ties_respect_secondary_order():
    recs = [
        {"candidate_id": "CAND_0000009", "final_score": 0.9, "honeypot": False},
        {"candidate_id": "CAND_0000002", "final_score": 0.8, "honeypot": False},
        {"candidate_id": "CAND_0000005", "final_score": 0.8, "honeypot": False},
        {"candidate_id": "CAND_0000001", "final_score": 0.7, "honeypot": False},
    ]
    top = ranker.rank_records(recs, top=4)
    assert [r["candidate_id"] for r in top] == \
        ["CAND_0000009", "CAND_0000002", "CAND_0000005", "CAND_0000001"]


def test_full_csv_passes_official_validator(tmp_path):
    import random
    random.seed(1)
    # 120 candidates with deliberate ties
    scores = [round(random.choice([0.9, 0.8, 0.8, 0.5, 0.5, 0.3, random.random()]), 4)
              for _ in range(120)]
    recs = _recs(scores)
    top = ranker.rank_records(recs, top=100)
    out = tmp_path / "submission.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in top:
            w.writerow([r["candidate_id"], r["rank"], f"{r['final_score']:.6f}", "grounded reason."])
    errors = validate_submission.validate_submission(str(out))
    assert errors == [], errors
