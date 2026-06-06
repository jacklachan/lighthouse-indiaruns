"""Build a stratified eval sample across archetypes and label each tier 0-5.

Writes eval/eval_labels.json: {candidate_id: {tier, archetype, fit, caps}}.
This is the Claude-authored proxy ground truth (see eval/labeler.py).

Archetypes (the traps + the real fits the JD cares about):
  ai_engineer, plain_language_strong, keyword_stuffer, services_only,
  location_fail, behaviorally_weak, honeypot, other.

Usage:
  python eval/build_labels.py --candidates ./data/candidates.jsonl --per-bucket 30
"""
from __future__ import annotations

import argparse
import json
import os
import random

from lighthouse import SEED, features, honeypot, loader, scoring
from eval.labeler import label_candidate

ANCHORS = ["CAND_0000031", "CAND_0000001", "CAND_0000002"]


def _ai_skill_count(raw, rubric):
    core = set(rubric["ai_core_skills"])
    return sum(1 for s in loader.get_skills(raw) if s["name"].lower() in core)


def archetype(raw, rubric) -> str:
    if honeypot.is_honeypot(raw):
        return "honeypot"
    p = loader.get_profile(raw)
    sig = loader.get_signals(raw)
    country = loader._s(p, "country").lower()
    relocate = bool(sig.get("willing_to_relocate"))
    rc = features.role_coherence_taxonomy(raw, rubric)
    ce = features.career_evidence(raw, rubric)
    beh, _ = scoring.behavioral_modifier(raw, rubric)
    ai_sk = _ai_skill_count(raw, rubric)
    sf = features.services_fraction(raw, rubric)

    if country and country != "india" and not relocate:
        return "location_fail"
    if rc < 0.3 and ai_sk >= 6:
        return "keyword_stuffer"
    if sf >= 0.999:
        return "services_only"
    if rc >= 0.7 and ce >= 0.5 and country == "india":
        return "ai_engineer"
    if ce >= 0.5 and ai_sk < 3:
        return "plain_language_strong"
    if rc >= 0.6 and ce >= 0.35 and beh <= 0.85:
        return "behaviorally_weak"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="./data/candidates.jsonl")
    ap.add_argument("--per-bucket", type=int, default=30)
    ap.add_argument("--out", default="eval/eval_labels.json")
    args = ap.parse_args()

    rubric = json.load(open("artifacts/jd_rubric.json", encoding="utf-8"))
    random.seed(SEED)

    buckets: dict = {}
    anchors_raw = {}
    for raw in loader.iter_raw(args.candidates):
        cid = loader.candidate_id(raw)
        if cid in ANCHORS:
            anchors_raw[cid] = raw
        a = archetype(raw, rubric)
        buckets.setdefault(a, []).append(raw)

    print("Bucket sizes in pool:")
    for k in sorted(buckets):
        print(f"  {k:22s} {len(buckets[k]):>7d}")

    sampled = {}
    for a, items in buckets.items():
        # honeypots: take more; others: per-bucket cap
        cap = min(len(items), args.per_bucket if a != "honeypot" else 25)
        for raw in random.sample(items, cap):
            sampled[loader.candidate_id(raw)] = raw
    # ensure anchors are present
    for cid, raw in anchors_raw.items():
        sampled[cid] = raw

    labels = {}
    tier_hist = {i: 0 for i in range(6)}
    arche_of = {}
    for cid, raw in sampled.items():
        tier, info = label_candidate(raw, rubric)
        a = archetype(raw, rubric)
        labels[cid] = {"tier": tier, "archetype": a,
                       "fit": info.get("fit"), "caps": info.get("caps", [])}
        tier_hist[tier] += 1
        arche_of[cid] = a

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(labels, open(args.out, "w"), indent=2)

    print(f"\nLabeled {len(labels)} candidates -> {args.out}")
    print("Tier histogram:", tier_hist)
    print("\nAnchor labels:")
    for cid in ANCHORS:
        if cid in labels:
            print(f"  {cid}: tier {labels[cid]['tier']} ({labels[cid]['archetype']})")


if __name__ == "__main__":
    main()
