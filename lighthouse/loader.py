"""Schema-tolerant loading of Redrob candidate records.

The challenge file is a 100K-line JSONL (~487 MB). We stream it line-by-line so
we never hold more than one raw record in memory at a time during parsing.

Every accessor here is defensive: optional fields may be missing, lists may be
empty, and the dataset uses sentinels (`github_activity_score == -1`,
`offer_acceptance_rate == -1`, empty `skill_assessment_scores`). Nothing in this
module should ever raise on a well-formed-but-sparse profile.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict, Iterator, List, Optional

# ---------------------------------------------------------------------------
# Low-level safe accessors
# ---------------------------------------------------------------------------

def _s(d: Optional[dict], key: str, default: str = "") -> str:
    if not isinstance(d, dict):
        return default
    v = d.get(key, default)
    return v if isinstance(v, str) else (default if v is None else str(v))


def _f(d: Optional[dict], key: str, default: float = 0.0) -> float:
    if not isinstance(d, dict):
        return default
    v = d.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(d: Optional[dict], key: str, default: int = 0) -> int:
    if not isinstance(d, dict):
        return default
    v = d.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _b(d: Optional[dict], key: str, default: bool = False) -> bool:
    if not isinstance(d, dict):
        return default
    v = d.get(key, default)
    return bool(v) if isinstance(v, bool) else default


def parse_date(s: Any) -> Optional[date]:
    """Parse an ISO date string; return None for missing/invalid/null."""
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Streaming reader
# ---------------------------------------------------------------------------

def iter_raw(path: str) -> Iterator[dict]:
    """Yield raw candidate dicts from a JSONL file, one line at a time."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines rather than aborting the whole run.
                continue


def load_all(path: str) -> List[dict]:
    """Load every raw candidate record into a list."""
    return list(iter_raw(path))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def get_skills(raw: dict) -> List[dict]:
    skills = raw.get("skills") or []
    out = []
    for sk in skills:
        if not isinstance(sk, dict):
            continue
        out.append({
            "name": _s(sk, "name"),
            "proficiency": _s(sk, "proficiency", "beginner").lower(),
            "endorsements": _i(sk, "endorsements"),
            "duration_months": _i(sk, "duration_months"),
        })
    return out


def get_career(raw: dict) -> List[dict]:
    hist = raw.get("career_history") or []
    out = []
    for h in hist:
        if not isinstance(h, dict):
            continue
        out.append({
            "company": _s(h, "company"),
            "title": _s(h, "title"),
            "start_date": parse_date(h.get("start_date")),
            "end_date": parse_date(h.get("end_date")),
            "duration_months": _i(h, "duration_months"),
            "is_current": _b(h, "is_current"),
            "industry": _s(h, "industry"),
            "company_size": _s(h, "company_size"),
            "description": _s(h, "description"),
        })
    return out


def get_education(raw: dict) -> List[dict]:
    edu = raw.get("education") or []
    out = []
    for e in edu:
        if not isinstance(e, dict):
            continue
        out.append({
            "institution": _s(e, "institution"),
            "degree": _s(e, "degree"),
            "field_of_study": _s(e, "field_of_study"),
            "start_year": _i(e, "start_year"),
            "end_year": _i(e, "end_year"),
            "tier": _s(e, "tier", "unknown"),
        })
    return out


def get_signals(raw: dict) -> dict:
    """Return the redrob_signals object (raw dict, with safe fallback)."""
    sig = raw.get("redrob_signals")
    return sig if isinstance(sig, dict) else {}


def get_profile(raw: dict) -> dict:
    p = raw.get("profile")
    return p if isinstance(p, dict) else {}


def candidate_id(raw: dict) -> str:
    return _s(raw, "candidate_id")


# ---------------------------------------------------------------------------
# Canonical text blob (used by both embeddings and BM25)
# ---------------------------------------------------------------------------

def build_text_blob(raw: dict) -> str:
    """Build one clean text blob per candidate for semantic + lexical matching.

    Order matters for readability but not for embeddings: headline + summary
    first (densest signal), then each role's title/description, then skill names.
    We deliberately repeat titles/descriptions verbatim — they are the strongest
    evidence of what the person actually *did*.
    """
    p = get_profile(raw)
    parts: List[str] = []

    headline = _s(p, "headline")
    if headline:
        parts.append(headline)

    summary = _s(p, "summary")
    if summary:
        parts.append(summary)

    cur_title = _s(p, "current_title")
    cur_co = _s(p, "current_company")
    if cur_title:
        parts.append(f"{cur_title} at {cur_co}".strip(" at"))

    for h in get_career(raw):
        seg = f"{h['title']} at {h['company']} ({h['industry']}). {h['description']}".strip()
        if seg:
            parts.append(seg)

    skill_names = [sk["name"] for sk in get_skills(raw) if sk["name"]]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    return "\n".join(parts).strip()
