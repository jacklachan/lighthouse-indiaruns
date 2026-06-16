"""Shared rank-time constants.

Centralizing these prevents the kind of bug where one module's hardcoded
reference date drifts from another's and date-based features (active-recent,
tenure overflow, langchain recency, days-active reasoning) start disagreeing
on what "now" means.
"""

from __future__ import annotations

from datetime import date

# The reference "now" the JD rubric was authored against. Every date-based
# feature (recency, tenure overflow, days-active behavior) measures against
# this date so the ranking is deterministic and replays identically year over
# year. Do not change without regenerating submission.csv.
REFERENCE_DATE: date = date(2026, 6, 6)
