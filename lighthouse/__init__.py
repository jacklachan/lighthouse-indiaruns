"""Lighthouse — a recruiter-grade, reasoning-first candidate ranker.

Keyword filters surface the loudest profiles. Lighthouse surfaces the right
ones — and ignores the fakes that fool keyword filters.
"""

__version__ = "1.0.0"

# Single source of truth for determinism across the whole pipeline.
SEED = 1729
