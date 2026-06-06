"""Lighthouse rank-time entry point.

    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

CPU only, no network, < 5 minutes, < 16 GB RAM. Loads precomputed artifacts
(embeddings, JD facets, rubric), scores every candidate with the five-component
model + gates + behavioral modifier, zeroes honeypots, ranks with the spec
tie-break, generates grounded reasoning, and writes a validator-clean CSV.
"""
from __future__ import annotations

import argparse
import csv
import time

from lighthouse import loader, ranker, reasoning


def main():
    ap = argparse.ArgumentParser(description="Lighthouse candidate ranker")
    ap.add_argument("--candidates", default="./data/candidates.jsonl")
    ap.add_argument("--out", default="./submission.csv")
    ap.add_argument("--artifacts", default="./artifacts")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--model", default=None, help="override embedder for small-sample fallback")
    ap.add_argument("--drop", default=None, help="ablate one component (eval only)")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading artifacts from {args.artifacts} ...")
    art = ranker.load_artifacts(args.artifacts)

    print(f"Loading candidates from {args.candidates} ...")
    raws = loader.load_all(args.candidates)
    print(f"  {len(raws):,} candidates loaded ({time.time()-t0:.1f}s)")

    print("Scoring ...")
    records = ranker.score_all(raws, art, drop=args.drop, model_name=args.model)
    # Order-preserving scale so the top score is 1.0 and all scores live in (0,1].
    mx = max((r["final_score"] for r in records), default=0.0)
    if mx > 0:
        for r in records:
            r["final_score"] = round(r["final_score"] / mx, 6)
    top = ranker.rank_records(records, top=args.top)
    print(f"  scored + ranked ({time.time()-t0:.1f}s)")

    # reasoning needs the raw record per candidate
    raw_by_id = {loader.candidate_id(r): r for r in raws}
    rows = []
    for rec in top:
        raw = raw_by_id[rec["candidate_id"]]
        why = reasoning.generate(raw, art["rubric"], rec)
        rows.append((rec["candidate_id"], rec["rank"], f"{rec['final_score']:.6f}", why))

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        w.writerows(rows)

    n_honeypot = sum(1 for r in top if r["honeypot"])
    print(f"Wrote {len(rows)} rows -> {args.out}")
    print(f"Honeypots in top-{args.top}: {n_honeypot}")
    print(f"Total wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
