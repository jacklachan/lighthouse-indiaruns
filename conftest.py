"""Ensures the repo root is importable as `lighthouse` during pytest/dev."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
