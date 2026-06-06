"""Render submission.csv into a readable PDF (backup for the portal's PDF dropzone).

The canonical ranked artifact is submission.csv (the validator scores CSV). Some
portal widgets only accept PDF, so this produces deck/submission_top100.pdf — a
clean multi-page report of rank / candidate_id / score / title / grounded reasoning.

Usage:
  python scripts/make_submission_pdf.py --submission submission.csv
"""
from __future__ import annotations

import argparse
import csv
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INK = "#1f2a44"
ACCENT = "#c0392b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--out", default="deck/submission_top100.pdf")
    ap.add_argument("--per-page", type=int, default=13)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.submission, encoding="utf-8")))
    per = args.per_page
    pages = (len(rows) + per - 1) // per

    with PdfPages(args.out) as pdf:
        # cover
        fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait
        fig.text(0.5, 0.62, "🔦 Lighthouse", ha="center", fontsize=30, weight="bold", color=INK)
        fig.text(0.5, 0.56, "Ranked candidates — top 100", ha="center", fontsize=15, color=INK)
        fig.text(0.5, 0.50, "Redrob India Runs · Intelligent Candidate Discovery & Ranking",
                 ha="center", fontsize=10, color=INK)
        fig.text(0.5, 0.46, "Canonical machine-readable artifact: submission.csv "
                            "(this PDF is a human-readable rendering).",
                 ha="center", fontsize=8, color="#666666")
        plt.axis("off")
        pdf.savefig(fig); plt.close(fig)

        for pg in range(pages):
            chunk = rows[pg * per:(pg + 1) * per]
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.04, 0.03, 0.92, 0.94]); ax.axis("off")
            y = 1.0
            for r in chunk:
                rank, cid = r["rank"], r["candidate_id"]
                score = float(r["score"]); why = r["reasoning"]
                ax.text(0.0, y, f"#{rank}", fontsize=10, weight="bold", color=ACCENT,
                        transform=ax.transAxes, va="top")
                ax.text(0.06, y, f"{cid}   score {score:.4f}", fontsize=9.5, weight="bold",
                        color=INK, transform=ax.transAxes, va="top")
                wrapped = textwrap.fill(why, width=110)
                ax.text(0.06, y - 0.018, wrapped, fontsize=8, color="#222222",
                        transform=ax.transAxes, va="top")
                y -= 0.018 + 0.020 * (wrapped.count("\n") + 1) + 0.012
            fig.text(0.5, 0.012, f"Lighthouse top-100 — page {pg+1}/{pages}",
                     ha="center", fontsize=7, color="#888888")
            pdf.savefig(fig); plt.close(fig)

    import os
    print(f"Wrote {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB, {pages+1} pages)")


if __name__ == "__main__":
    main()
