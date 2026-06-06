"""Generate the Lighthouse system-architecture diagram (deck/assets/architecture.png)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2a44"
OFFLINE = "#e8edf7"
OFFLINE_E = "#3b5ba5"
ONLINE = "#fdf3df"
ONLINE_E = "#c9962f"
ACCENT = "#c0392b"

os.makedirs("deck/assets", exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 6.4), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, weight="bold" if bold else "normal", wrap=True)


def arrow(x1, y1, x2, y2, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.5, color=color))


# band labels
ax.add_patch(FancyBboxPatch((1, 55), 98, 42, boxstyle="round,pad=0.2,rounding_size=2",
                            fc="none", ec=OFFLINE_E, lw=1.2, linestyle=(0, (4, 3))))
ax.text(2.5, 94.5, "OFFLINE PRECOMPUTE  ·  network OK  ·  no time budget", fontsize=10,
        color=OFFLINE_E, weight="bold")
ax.add_patch(FancyBboxPatch((1, 3), 98, 44, boxstyle="round,pad=0.2,rounding_size=2",
                            fc="none", ec=ONLINE_E, lw=1.2, linestyle=(0, (4, 3))))
ax.text(2.5, 43.5, "RANK-TIME  ·  CPU only  ·  no network  ·  < 5 min  ·  numpy/pandas",
        fontsize=10, color=ONLINE_E, weight="bold")

# offline row
box(3, 79, 20, 9, "Job Description\n(.docx)", "#ffffff", OFFLINE_E, 9)
box(3, 64, 20, 9, "candidates.jsonl\n(100,000)", "#ffffff", OFFLINE_E, 9)
box(28, 79, 19, 9, "Claude (offline)\nJD → rubric", OFFLINE, ACCENT, 9, True)
box(28, 64, 19, 9, "text blobs +\nbge-small-en-v1.5", OFFLINE, OFFLINE_E, 9)
box(52, 84, 20, 8, "jd_rubric.json\n+ facet vecs", "#ffffff", OFFLINE_E, 8.5, True)
box(52, 72, 20, 8, "cand_emb.npy\n(fp16, ~75MB)", "#ffffff", OFFLINE_E, 8.5)
box(52, 60, 20, 8, "BM25 scores", "#ffffff", OFFLINE_E, 8.5)
box(76, 64, 21, 9, "Claude (offline)\neval labels 0–5", OFFLINE, ACCENT, 9, True)

arrow(23, 83.5, 28, 83.5); arrow(47, 83.5, 52, 86)
arrow(23, 68.5, 28, 68.5); arrow(47, 68.5, 52, 76); arrow(47, 66, 52, 64)
arrow(72, 64, 76, 67)

# bridge
box(38, 49.5, 24, 7, "artifacts/  (committed)", ONLINE, ONLINE_E, 9, True)
arrow(62, 80, 50, 56.5, OFFLINE_E)
arrow(62, 64, 52, 56.5, OFFLINE_E)

# rank-time row
box(3, 28, 15, 9, "load artifacts\n+ candidates", "#ffffff", ONLINE_E, 8.5)
box(20, 23, 21, 19,
    "score 5 components\n• semantic_fit\n• role_coherence\n• career_evidence\n"
    "• experience_fit\n• trust_skills", ONLINE, ONLINE_E, 8.2, True)
box(43, 30, 16, 9, "× hard-negative\ngates", ONLINE, ACCENT, 8.5, True)
box(43, 18, 16, 9, "× behavioral\nmodifier", ONLINE, ONLINE_E, 8.5)
box(61, 24, 15, 11, "honeypot →\nscore 0", ONLINE, ACCENT, 8.5, True)
box(78, 24, 9, 11, "sort +\ntie-break\ntop-100", "#ffffff", ONLINE_E, 8.2)
box(88, 24, 10, 11, "grounded\nreasoning", ONLINE, ONLINE_E, 8.2)
box(70, 7, 22, 7, "submission.csv  (top 100)", INK, INK, 9.5, True)
ax.text(81, 10.5, "submission.csv  ·  top 100", ha="center", va="center",
        fontsize=9.5, color="white", weight="bold")

arrow(18, 32.5, 20, 32.5)
arrow(41, 32.5, 43, 34.5); arrow(41, 30, 43, 23)
arrow(51, 30, 61, 30); arrow(51, 22.5, 60, 27)
arrow(76, 29.5, 78, 29.5); arrow(87, 29.5, 88, 29.5)
arrow(93, 24, 86, 14)
arrow(50, 49.5, 30, 37, ONLINE_E)

ax.text(50, 99.3, "Lighthouse — offline-LLM-augmented hybrid ranker",
        ha="center", fontsize=13, weight="bold", color=INK)

plt.tight_layout()
plt.savefig("deck/assets/architecture.png", bbox_inches="tight", facecolor="white")
print("wrote deck/assets/architecture.png")
