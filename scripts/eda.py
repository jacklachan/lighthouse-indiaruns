"""Exploratory data analysis over the full 100K candidate pool.

Run:  python scripts/eda.py --candidates ./data/candidates.jsonl
Writes a JSON summary to eval/eda_report.json and prints a human digest.

This is a read-only sanity pass: title/skill/location distributions, sentinel
coverage, experience band, and a first cut at how rare genuine fits are.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from lighthouse import loader

# Titles that plausibly indicate AI/ML/IR/SWE-ranking work (rough, for EDA only).
AI_TITLE_HINTS = (
    "machine learning", "ml engineer", "ai engineer", "applied scientist",
    "applied ml", "data scientist", "research engineer", "nlp",
    "recommendation", "search engineer", "ranking", "deep learning",
)
SERVICES_COS = ("tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
                "tech mahindra", "hcl", "mindtree", "ltimindtree")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="./data/candidates.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    n = 0
    titles = Counter()
    countries = Counter()
    skill_names = Counter()
    n_no_github = 0
    n_no_offer = 0
    n_empty_assess = 0
    yoe_buckets = Counter()
    n_relocate = 0
    n_open = 0
    n_india = 0
    n_ai_title = 0
    n_services_current = 0
    skills_per = []
    roles_per = []
    sum_skillmonths_gt_career = 0  # crude honeypot smell

    for raw in loader.iter_raw(args.candidates):
        n += 1
        if args.limit and n > args.limit:
            n -= 1
            break
        p = loader.get_profile(raw)
        sig = loader.get_signals(raw)
        title = loader._s(p, "current_title")
        titles[title] += 1
        country = loader._s(p, "country")
        countries[country] += 1
        if country.lower() == "india":
            n_india += 1
        tl = title.lower()
        if any(h in tl for h in AI_TITLE_HINTS):
            n_ai_title += 1
        cur_co = loader._s(p, "current_company").lower()
        if any(s in cur_co for s in SERVICES_COS):
            n_services_current += 1

        yoe = loader._f(p, "years_of_experience")
        yoe_buckets[min(int(yoe), 20)] += 1

        if loader._f(sig, "github_activity_score", 0) == -1:
            n_no_github += 1
        if loader._f(sig, "offer_acceptance_rate", 0) == -1:
            n_no_offer += 1
        assess = sig.get("skill_assessment_scores")
        if not assess:
            n_empty_assess += 1
        if sig.get("willing_to_relocate"):
            n_relocate += 1
        if sig.get("open_to_work_flag"):
            n_open += 1

        skills = loader.get_skills(raw)
        skills_per.append(len(skills))
        for sk in skills:
            skill_names[sk["name"]] += 1
        career = loader.get_career(raw)
        roles_per.append(len(career))
        skill_months = sum(sk["duration_months"] for sk in skills)
        if skill_months > yoe * 12 * 1.5 and yoe > 0:
            sum_skillmonths_gt_career += 1

    def pct(x):
        return round(100.0 * x / n, 2) if n else 0.0

    report = {
        "n_candidates": n,
        "top_titles": titles.most_common(25),
        "top_countries": countries.most_common(12),
        "n_india": n_india, "pct_india": pct(n_india),
        "n_ai_title": n_ai_title, "pct_ai_title": pct(n_ai_title),
        "n_services_current": n_services_current, "pct_services_current": pct(n_services_current),
        "sentinels": {
            "no_github_-1": n_no_github, "pct": pct(n_no_github),
            "no_offer_history_-1": n_no_offer, "pct_offer": pct(n_no_offer),
            "empty_skill_assessment": n_empty_assess, "pct_assess": pct(n_empty_assess),
        },
        "behavior": {
            "willing_to_relocate": n_relocate, "pct_relocate": pct(n_relocate),
            "open_to_work": n_open, "pct_open": pct(n_open),
        },
        "yoe_histogram": dict(sorted(yoe_buckets.items())),
        "skills_per_candidate_avg": round(sum(skills_per) / max(1, len(skills_per)), 2),
        "roles_per_candidate_avg": round(sum(roles_per) / max(1, len(roles_per)), 2),
        "top_skills": skill_names.most_common(40),
        "skillmonths_gt_1.5x_career": sum_skillmonths_gt_career,
        "pct_skillmonths_smell": pct(sum_skillmonths_gt_career),
    }

    with open("eval/eda_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"n = {n:,}")
    print(f"India: {report['pct_india']}%  | AI-ish title: {report['pct_ai_title']}%  | "
          f"current at services firm: {report['pct_services_current']}%")
    print(f"sentinels: no-github {report['sentinels']['pct']}%  "
          f"no-offer {report['sentinels']['pct_offer']}%  "
          f"empty-assessment {report['sentinels']['pct_assess']}%")
    print(f"willing_to_relocate {report['behavior']['pct_relocate']}%  "
          f"open_to_work {report['behavior']['pct_open']}%")
    print(f"avg skills/cand {report['skills_per_candidate_avg']}  "
          f"avg roles/cand {report['roles_per_candidate_avg']}")
    print(f"skill-months > 1.5x career length (honeypot smell): "
          f"{report['skillmonths_gt_1.5x_career']} ({report['pct_skillmonths_smell']}%)")
    print("\nTop 15 current titles:")
    for t, c in titles.most_common(15):
        print(f"  {c:6d}  {t}")
    print("\nReport written to eval/eda_report.json")


if __name__ == "__main__":
    main()
