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

from . import loader
from .constants import REFERENCE_DATE


def _months_between(d0: date, d1: date) -> int:
    return (d1.year - d0.year) * 12 + (d1.month - d0.month)


def detect(raw: dict) -> tuple[bool, list[str]]:
    """Return (is_honeypot, reasons). Any single hard-impossible rule flags."""
    reasons: list[str] = []
    p = loader.get_profile(raw)
    yoe = loader._f(p, "years_of_experience")
    career = loader.get_career(raw)
    edu = loader.get_education(raw)

    # --- 1. claimed expertise with zero usage ---
    # Walk the RAW skills list (not the loader-normalized one) so a candidate
    # whose record simply omits `duration_months` is treated as "unknown",
    # not "explicitly zero". The loader collapses missing -> 0 for
    # convenience, but that collapse is unfair here: an honest sparse profile
    # would otherwise get flagged as a honeypot. Only an EXPLICIT integer 0
    # alongside an advanced/expert claim is impossible-looking.
    raw_skills = raw.get("skills") or []
    expert_zero: list[str] = []
    for s in raw_skills:
        if not isinstance(s, dict):
            continue
        prof = (s.get("proficiency") or "").lower()
        if prof not in ("advanced", "expert"):
            continue
        dm = s.get("duration_months")
        if isinstance(dm, (int, float)) and dm == 0:
            expert_zero.append(str(s.get("name") or ""))
    if len(expert_zero) >= 3:
        reasons.append(
            f"{len(expert_zero)} skills claimed advanced/expert with 0 months used "
            f"(e.g. {', '.join(expert_zero[:3])})"
        )

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
                    f"role at {h['company']}: stated {dm}mo vs {span}mo implied by dates"
                )

    # --- 5. total tenure overflows the career length ---
    # Bound = yoe*12 * 1.6 + 18 (months). The 1.6x allowance lets legitimately
    # concurrent/overlapping roles (e.g. a contract or advisory role alongside a
    # full-time one) sum past the raw career length, and the +18mo absorbs rounding
    # and short pre-/post-career gaps. EDA: real profiles sit comfortably under this;
    # only fabricated histories (stacked full-time roles that could not have
    # co-occurred) overflow it, so the rule flags impossibility rather than overlap.
    if yoe > 0 and total_role_months > yoe * 12 * 1.6 + 18:
        reasons.append(
            f"career roles sum to {total_role_months}mo, impossible for {yoe:.1f} yrs experience"
        )

    # --- 6. grossly impossible single-role tenure (dormant backstop) ---
    # EDA: the pool's longest legitimate single-role tenure is 228 months (19 yrs),
    # so a long tenure on its own is NOT evidence of fraud — we must not clip real
    # senior veterans. We only flag values BEYOND any plausible career (>300mo /
    # 25 yrs), which nothing in this pool hits; the genuinely impossible profiles
    # are caught by the date-integrity and tenure-vs-experience rules above.
    for h in career:
        if h["duration_months"] > 300:
            reasons.append(
                f"impossible {h['duration_months']}mo (> 25 yr) tenure at {h['company']}"
            )
            break

    # --- 7. education dates invalid ---
    for e in edu:
        if e["start_year"] and e["end_year"] and e["end_year"] < e["start_year"]:
            reasons.append(
                f"education '{e['degree']}' ends ({e['end_year']}) before it starts ({e['start_year']})"
            )

    # --- 8. Redrob signup_date in the future ---
    # Pre-fix this signal was never inspected. A signup_date strictly after
    # the JD reference "now" is impossible — the candidate cannot have a
    # Redrob account that does not yet exist. Catches a class of synthetic
    # honeypots that were sliding through.
    sig = loader.get_signals(raw)
    signup = loader.parse_date(sig.get("signup_date"))
    if signup and signup > REFERENCE_DATE:
        reasons.append(f"Redrob signup_date {signup.isoformat()} is in the future")

    # --- 9. Redrob last_active_date in the future ---
    # Same logic as signup_date: a profile cannot have been "active" after the
    # JD reference "now". Catches synthetic profiles that game the activity
    # bonus by stamping a far-future last_active and would otherwise sneak the
    # +0.05 "active 0d ago" lift through behavioral_modifier.
    last_active = loader.parse_date(sig.get("last_active_date"))
    if last_active and last_active > REFERENCE_DATE:
        reasons.append(f"Redrob last_active_date {last_active.isoformat()} is in the future")

    return (len(reasons) > 0, reasons)


def is_honeypot(raw: dict) -> bool:
    return detect(raw)[0]
