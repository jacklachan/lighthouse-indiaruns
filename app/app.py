"""Lighthouse — HuggingFace Spaces / Streamlit sandbox.

Accepts up to 100 candidates (preloaded sample or uploaded JSONL), runs the full
Lighthouse ranker end-to-end on CPU, and shows the ranked table with scores +
grounded reasoning. Candidates uploaded here are not precomputed, so the small
bge-small encoder runs at request time (a few seconds for <=100 rows) — this is
the demo path; the official 100K run uses fully precomputed embeddings.

Run locally:  streamlit run app/app.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lighthouse import loader, ranker, reasoning  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")
SAMPLE = os.path.join(HERE, "sample_candidates.jsonl")

st.set_page_config(page_title="Lighthouse — Candidate Ranker", page_icon="🔦", layout="wide")


@st.cache_resource
def _facets_and_rubric():
    rubric = json.load(open(os.path.join(ART, "jd_rubric.json"), encoding="utf-8"))
    facet_emb = np.load(os.path.join(ART, "jd_facet_emb.npy"))
    return rubric, facet_emb


def _build_art(rubric, facet_emb):
    """Empty precomputed set -> every candidate is encoded on the fly."""
    dim = facet_emb.shape[1]
    return {"rubric": rubric, "ids": [], "id_to_row": {},
            "cand_emb": np.zeros((0, dim), dtype=np.float32), "facet_emb": facet_emb}


def _parse_jsonl(text: str):
    raws = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raws.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return raws


st.title("🔦 Lighthouse — recruiter-grade candidate ranker")
st.caption("Keyword filters surface the loudest profiles. Lighthouse surfaces the right ones — "
           "and ignores the fakes that fool keyword filters.")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Candidate source", ["Preloaded sample (100)", "Upload JSONL (≤100)"])
    st.markdown("---")
    st.markdown("**JD:** Senior AI Engineer @ Redrob AI — production embeddings/retrieval, "
                "ranking-eval, product (not services), 6–8 yrs ideal, Noida/Pune/India.")
    st.markdown("Ranking runs on **CPU, no hosted LLM**. The five-component score is gated by "
                "JD hard-negatives and a behavioral modifier; honeypots are zeroed.")

rubric, facet_emb = _facets_and_rubric()

raws = []
if mode.startswith("Preloaded"):
    if os.path.exists(SAMPLE):
        raws = _parse_jsonl(open(SAMPLE, encoding="utf-8").read())
        st.info(f"Loaded {len(raws)} preloaded sample candidates.")
    else:
        st.error("sample_candidates.jsonl not found.")
else:
    up = st.file_uploader("Upload candidates JSONL (one JSON candidate per line)", type=["jsonl", "json"])
    if up is not None:
        raws = _parse_jsonl(up.read().decode("utf-8"))
        st.info(f"Parsed {len(raws)} candidates.")

raws = raws[:100]

if raws and st.button("🔦 Rank candidates", type="primary"):
    with st.spinner(f"Encoding + scoring {len(raws)} candidates on CPU ..."):
        art = _build_art(rubric, facet_emb)
        records = ranker.score_all(raws, art)
        mx = max((r["final_score"] for r in records), default=0.0)
        if mx > 0:
            for r in records:
                r["final_score"] = round(r["final_score"] / mx, 6)
        top = ranker.rank_records(records, top=len(records))
        raw_by_id = {loader.candidate_id(r): r for r in raws}
        rows = []
        for rec in top:
            raw = raw_by_id[rec["candidate_id"]]
            p = loader.get_profile(raw)
            rows.append({
                "rank": rec["rank"],
                "candidate_id": rec["candidate_id"],
                "score": rec["final_score"],
                "title": loader._s(p, "current_title"),
                "country": loader._s(p, "country"),
                "yrs": loader._f(p, "years_of_experience"),
                "honeypot": "⚠️" if rec["honeypot"] else "",
                "reasoning": reasoning.generate(raw, rubric, rec),
            })
    df = pd.DataFrame(rows)
    n_hp = sum(1 for r in top if r["honeypot"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates ranked", len(rows))
    c2.metric("Honeypots flagged", n_hp)
    c3.metric("Top score", f"{rows[0]['score']:.3f}" if rows else "—")
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"score": st.column_config.NumberColumn(format="%.4f")})

    with st.expander("Inspect the top candidate's component breakdown"):
        best = top[0]
        st.json({"candidate_id": best["candidate_id"], "components": best["components"],
                 "base": best["base"], "gate_mult": best["gate_mult"],
                 "gate_reasons": best["gate_reasons"], "behavior_mult": best["behavior_mult"],
                 "honeypot": best["honeypot"]})

    st.download_button("⬇️ Download ranking CSV",
                       df[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False),
                       file_name="submission.csv", mime="text/csv")
elif not raws:
    st.warning("Choose the preloaded sample or upload a JSONL to begin.")
