"""Build a BLIND independent-label sheet to validate Lighthouse against a human.

The eval in eval/results.md §2 uses Claude-authored labels — same source as the
ranker, so it can't catch shared blind spots. This harness fixes that: it writes
~20 stratified candidates to a CSV with a readable digest and an EMPTY `human_tier`
column, and deliberately **omits any Lighthouse score** so the human labels blind.
A teammate fills `human_tier` (0–5), then `eval/blind_compare.py` measures whether
Lighthouse agrees with the human (Spearman + NDCG).

Usage:
  python scripts/make_blind_eval.py --candidates ./data/candidates.jsonl --n 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

from lighthouse import SEED, loader, features
from eval.build_labels import archetype


def _digest(raw: dict, rubric: dict) -> dict:
    p = loader.get_profile(raw)
    sig = loader.get_signals(raw)
    skills = loader.get_skills(raw)
    career = loader.get_career(raw)
    rel = set(rubric["jd_relevant_skills"])
    rel_present = [s["name"] for s in skills if s["name"].lower() in rel][:6]
    roles = " | ".join(
        f"{h['title']} @ {h['company']} ({h['industry']}, {h['duration_months']}mo)"
        for h in career[:4])
    return {
        "candidate_id": loader.candidate_id(raw),
        "current_title": loader._s(p, "current_title"),
        "years_experience": loader._f(p, "years_of_experience"),
        "country": loader._s(p, "country"),
        "willing_to_relocate": sig.get("willing_to_relocate"),
        "headline": loader._s(p, "headline"),
        "summary": loader._s(p, "summary")[:400],
        "career_roles": roles,
        "jd_relevant_skills_listed": ", ".join(rel_present),
        "recruiter_response_rate": sig.get("recruiter_response_rate"),
        "last_active_date": sig.get("last_active_date"),
        "human_tier": "",   # <-- the human fills 0..5 (5=strong fit, 0=non-fit), BLIND
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="./data/candidates.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="eval/blind_eval_candidates.csv")
    args = ap.parse_args()

    rubric = json.load(open("artifacts/jd_rubric.json", encoding="utf-8"))
    random.seed(SEED + 7)   # different seed from the Claude label set

    buckets: dict = {}
    for raw in loader.iter_raw(args.candidates):
        buckets.setdefault(archetype(raw, rubric), []).append(raw)

    # spread the sample across archetypes so the human sees fits AND traps
    want = ["ai_engineer", "plain_language_strong", "keyword_stuffer",
            "services_only", "location_fail", "honeypot", "other"]
    per = max(1, args.n // len(want))
    chosen, seen = [], set()
    for a in want:
        for raw in random.sample(buckets.get(a, []), min(per, len(buckets.get(a, [])))):
            if raw["candidate_id"] not in seen:
                chosen.append(raw); seen.add(raw["candidate_id"])
    # top up to n from the whole pool
    allc = [r for v in buckets.values() for r in v]
    while len(chosen) < args.n and allc:
        raw = random.choice(allc)
        if raw["candidate_id"] not in seen:
            chosen.append(raw); seen.add(raw["candidate_id"])
    chosen = chosen[:args.n]
    random.shuffle(chosen)   # hide archetype ordering from the labeler

    rows = [_digest(r, rubric) for r in chosen]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} BLIND candidates -> {args.out}")
    print("Next: a human fills the `human_tier` column (0=non-fit … 5=strong fit) WITHOUT")
    print("looking at Lighthouse's output, then run:  python eval/blind_compare.py")


if __name__ == "__main__":
    main()
