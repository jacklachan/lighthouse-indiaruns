"""Honeypot / anomaly detection — explainable, rule-based.

The dataset seeds ~80 honeypots: "subtly impossible" profiles (tenure longer
than the company has existed; 'expert' in many skills with 0 months used; total
skill-months far exceeding career length; dates that don't add up). The spec
forces them to relevance tier 0 and disqualifies submissions with >10% of them
in the top 100. `rank.py` zeroes any candidate flagged here.

Design notes grounded in EDA over the real 100K:
  * The naive "sum(skill_months) > years_of_experience*12" check fires on 63% of
    the pool (skills are used concurrently), so it is USELESS as written. We only
    flag a *single* skill whose duration exceeds the whole career — that is
    genuinely impossible, not just overlapping.
  * Every rule returns a human-readable reason so flags are auditable and feed
    the reasoning generator.

Reference "now" is the JD reference date (2026-06-06).
"""
from __future__ import annotations

from datetime import date
from typing import List, Tuple

from . import loader

REFERENCE_DATE = date(2026, 6, 6)


def _months_between(d0: date, d1: date) -> int:
    return (d1.year - d0.year) * 12 + (d1.month - d0.month)


def detect(raw: dict) -> Tuple[bool, List[str]]:
    """Return (is_honeypot, reasons). Any single hard-impossible rule flags."""
    reasons: List[str] = []
    p = loader.get_profile(raw)
    yoe = loader._f(p, "years_of_experience")
    career = loader.get_career(raw)
    skills = loader.get_skills(raw)
    edu = loader.get_education(raw)

    # --- 1. claimed expertise with zero usage ---
    expert_zero = [s["name"] for s in skills
                   if s["proficiency"] in ("advanced", "expert") and s["duration_months"] == 0]
    if len(expert_zero) >= 3:
        reasons.append(
            f"{len(expert_zero)} skills claimed advanced/expert with 0 months used "
            f"(e.g. {', '.join(expert_zero[:3])})")

    # NOTE: a tempting rule — "a skill used more months than the whole career" —
    # was REMOVED after EDA: skill durations routinely exceed current-job YOE
    # (skills are learned before/outside a role), so that rule flagged 9% of the
    # pool. Genuine impossibility lives in the date math and expertise claims,
    # not in skill-vs-YOE.

    # --- 2 & 3. per-role date integrity ---
    total_role_months = 0
    for h in career:
        sd, ed = h["start_date"], h["end_date"]
        dm = h["duration_months"]
        total_role_months += dm
        if sd and sd > REFERENCE_DATE:
            reasons.append(f"role at {h['company']} starts in the future ({sd.isoformat()})")
        if sd and ed and ed < sd:
            reasons.append(f"role at {h['company']} ends before it starts")
        if sd and ed:
            span = _months_between(sd, ed)
            if span >= 0 and abs(span - dm) > 9:
                reasons.append(
                    f"role at {h['company']}: stated {dm}mo vs {span}mo implied by dates")

    # --- 5. total tenure overflows the career length ---
    if yoe > 0 and total_role_months > yoe * 12 * 1.6 + 18:
        reasons.append(
            f"career roles sum to {total_role_months}mo, impossible for {yoe:.1f} yrs experience")

    # --- 6. grossly impossible single-role tenure (dormant backstop) ---
    # EDA: the pool's longest legitimate single-role tenure is 228 months (19 yrs),
    # so a long tenure on its own is NOT evidence of fraud — we must not clip real
    # senior veterans. We only flag values BEYOND any plausible career (>300mo /
    # 25 yrs), which nothing in this pool hits; the genuinely impossible profiles
    # are caught by the date-integrity and tenure-vs-experience rules above.
    for h in career:
        if h["duration_months"] > 300:
            reasons.append(f"impossible {h['duration_months']}mo (> 25 yr) tenure at {h['company']}")
            break

    # --- 7. education dates invalid ---
    for e in edu:
        if e["start_year"] and e["end_year"] and e["end_year"] < e["start_year"]:
            reasons.append(
                f"education '{e['degree']}' ends ({e['end_year']}) before it starts ({e['start_year']})")

    return (len(reasons) > 0, reasons)


def is_honeypot(raw: dict) -> bool:
    return detect(raw)[0]
