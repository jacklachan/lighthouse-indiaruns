"""Compare Lighthouse against an INDEPENDENT human-labeled blind set.

Reads eval/blind_eval_candidates.csv after a human has filled `human_tier`,
scores those candidates with Lighthouse, and reports how well Lighthouse's
ranking agrees with the human's — the independent check that §2's self-labeled
metrics cannot provide. Writes eval/blind_results.md.

Usage:
  python eval/blind_compare.py --candidates ./data/candidates.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json

from lighthouse import loader, metrics, ranker

BLIND = "eval/blind_eval_candidates.csv"


def _spearman(rank_a, rank_b):
    """Spearman rho without scipy: Pearson on the rank vectors."""
    n = len(rank_a)
    if n < 2:
        return 0.0
    ma = sum(rank_a) / n
    mb = sum(rank_b) / n
    cov = sum((a - ma) * (b - mb) for a, b in zip(rank_a, rank_b))
    va = sum((a - ma) ** 2 for a in rank_a) ** 0.5
    vb = sum((b - mb) ** 2 for b in rank_b) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="./data/candidates.jsonl")
    ap.add_argument("--artifacts", default="./artifacts")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(BLIND, encoding="utf-8")))
    labeled = [r for r in rows if str(r.get("human_tier", "")).strip() != ""]
    if len(labeled) < 5:
        print(f"Only {len(labeled)} rows have a human_tier filled in {BLIND}.")
        print("Fill the human_tier column (0-5) for the sample, then re-run.")
        return
    human = {r["candidate_id"]: int(float(r["human_tier"])) for r in labeled}
    ids = set(human)

    art = ranker.load_artifacts(args.artifacts)
    raws = [r for r in loader.iter_raw(args.candidates) if loader.candidate_id(r) in ids]
    recs = ranker.score_all(raws, art)
    by_id = {r["candidate_id"]: r for r in recs}

    # Lighthouse order vs human tiers
    ordered = sorted(recs, key=lambda r: (-r["final_score"], r["candidate_id"]))
    rels_in_lh_order = [human[r["candidate_id"]] for r in ordered]
    ndcg = metrics.ndcg_at_k(rels_in_lh_order, len(rels_in_lh_order))
    ndcg10 = metrics.ndcg_at_k(rels_in_lh_order, 10)

    # Spearman between Lighthouse score and human tier
    common = [c for c in ids if c in by_id]
    lh_scores = [by_id[c]["final_score"] for c in common]
    hu = [human[c] for c in common]
    # convert to ranks for Spearman
    def to_ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0] * len(xs)
        for r, i in enumerate(order):
            ranks[i] = r
        return ranks
    rho = _spearman(to_ranks(lh_scores), to_ranks(hu))

    out = []
    out.append("## 4. Independent human validation (blind)\n")
    out.append(f"A human labeled **{len(human)}** candidates blind (tiers 0–5) in "
               f"`{BLIND}`, without seeing Lighthouse's scores — labels independent of the "
               f"ranker, unlike §2.\n")
    out.append(f"- **NDCG (full)** of Lighthouse's order vs the human tiers: **{ndcg:.3f}** "
               f"(NDCG@10 {ndcg10:.3f}).")
    out.append(f"- **Spearman ρ** (Lighthouse score vs human tier): **{rho:.3f}**.")
    out.append(f"- n = {len(human)} (small by design; independence matters more than size here).\n")
    open("eval/blind_results.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nWrote eval/blind_results.md. Paste/append it into eval/results.md as §4.")


if __name__ == "__main__":
    main()
