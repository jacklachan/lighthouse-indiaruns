"""Naive keyword-count baseline — reproduces the sample_submission's logic.

The provided sample_submission.csv ranks candidates by the number of "AI core
skills" they list, breaking ties by recruiter response rate. It is a deliberately
BAD ranking (it puts HR Managers and Accountants with 9 AI skills at the top).
We reproduce it here as the head-to-head comparison for results.md.
"""
from __future__ import annotations

from typing import List

from lighthouse import loader


def score_all(raws: List[dict], rubric: dict) -> List[dict]:
    core = set(rubric["ai_core_skills"])
    records = []
    for raw in raws:
        n_ai = sum(1 for s in loader.get_skills(raw) if s["name"].lower() in core)
        rr = loader.get_signals(raw).get("recruiter_response_rate") or 0.0
        # combine into a single monotone score (keyword count dominates)
        records.append({
            "candidate_id": loader.candidate_id(raw),
            "final_score": float(n_ai) + 0.001 * float(rr),
            "honeypot": False,
        })
    return records
