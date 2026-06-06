# Submission checklist — Redrob India Runs (Team UM)

Everything below is built and in this repo. This is the order to submit.

## 0. Pre-flight (verify locally, 2 min)

```bash
pip install -r requirements.txt
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv   # ~45 s, CPU
python validate_submission.py submission.csv                                  # "Submission is valid."
pytest -q                                                                     # 36 passed
```

## 1. Portal form (the upload page)

| Field | What to put |
|---|---|
| **Challenge** | Data & AI Challenge : Intelligent Candidate Discovery |
| **GitHub Repository** (must be public) | `https://github.com/jacklachan/lighthouse-indiaruns` — already public ✓ |
| **PPT/deck as PDF** | upload **`deck/Lighthouse_Redrob_Deck.pdf`** (≈0.5 MB ≤ 5 MB) |
| **Ranked output file** | upload **`submission.csv`** (canonical). The dropzone hints "PDF up to 5 MB" — if it hard-rejects the CSV, upload the backup **`deck/submission_top100.pdf`** instead. |

> **Note on the ranked-output dropzone:** the official spec scores a **CSV** (`validate_submission.py`
> checks CSV structure). The portal's "PDF" hint is most likely a generic widget label. Try the
> CSV first; the PDF (`deck/submission_top100.pdf`) is a human-readable rendering kept only as a
> fallback. If unsure, ask the organizers which they want — but never convert the CSV *away* from
> CSV as your primary artifact.

## 2. Portal metadata (have ready — mirrors `submission_metadata.yaml`)

- **Team name:** UM
- **Primary contact:** L Mohit Jain · mohitlalith07@gmail.com · +91-8660556007
- **Team members:** L Mohit Jain, Utkarsh Singh Yadav (usy.joseph@gmail.com)
- **Sandbox / demo link:** `https://huggingface.co/spaces/jacklachan/lighthouse` (Docker Space, RUNNING)
- **AI tools declared:** **Claude** — *keep this honest.* Claude was used offline for JD→rubric
  parsing + eval labeling + as a coding assistant; **no candidate data hit any API at rank-time**.
  (Declared AI use is *not* penalized; a declaration that contradicts the Stage-5 interview is.)
- **Compute env:** Local Windows 11, 16-core / 16 GB, Python 3.10.11, CPU-only ranking, no network.
- **Methodology summary:** see `submission_metadata.yaml` (≤200 words).

## 3. Before you click Submit

- [ ] GitHub repo is public and the latest commit is pushed.
- [ ] HF Space shows **Running** (open the sandbox link).
- [ ] Demo video recorded and its link added to the deck's "Submission Assets" slide (placeholder
      currently). *(Only outstanding manual item.)*
- [ ] `submission.csv` passes the validator.
- [ ] (Optional, strengthens Stage-4) a teammate filled `eval/blind_eval_candidates.csv` and you
      ran `python eval/blind_compare.py` — adds an independent human-validation number.

3 submissions max — this is the strong entry.
