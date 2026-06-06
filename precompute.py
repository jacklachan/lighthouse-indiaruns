"""Offline precompute — builds every artifact rank.py needs.

Network OK here; this runs OUTSIDE the 5-minute ranking budget. Produces:

  artifacts/candidate_ids.json   aligned list of candidate ids (file order)
  artifacts/cand_emb.npy         N x D float16, L2-normalized candidate embeddings
  artifacts/jd_facet_emb.npy     F x D float32, L2-normalized JD-facet embeddings
  artifacts/bm25_scores.npy      N float32, BM25(candidate blob vs JD query), 0-1 scaled
  artifacts/precompute_meta.json model name, dims, counts, date

The JD rubric (artifacts/jd_rubric.json) is authored separately by Claude and is
NOT regenerated here. Eval labels are built by eval/build_labels.py.

Usage:
  python precompute.py --candidates ./data/candidates.jsonl
  python precompute.py --candidates ./data/candidates.jsonl --limit 2000   # quick test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np

from lighthouse import SEED, loader

ART = "artifacts"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="./data/candidates.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="0 = all candidates")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--skip-bm25", action="store_true")
    ap.add_argument("--max-seq", type=int, default=256, help="max tokens per doc (CPU speed)")
    ap.add_argument("--out-dir", default=ART, help="artifact output directory")
    args = ap.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    np.random.seed(SEED)
    # Use all CPU threads for the transformer forward pass (precompute only).
    try:
        import torch
        torch.set_num_threads(os.cpu_count() or 8)
    except Exception:
        pass
    t0 = time.time()

    logpath = os.path.join(out, "precompute.log")
    logf = open(logpath, "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    rubric = json.load(open(os.path.join("artifacts", "jd_rubric.json"), encoding="utf-8"))

    # ---- 1. blobs + ids (streamed) ----
    log(f"[1/4] Building text blobs from {args.candidates} ...")
    ids, blobs = [], []
    for raw in loader.iter_raw(args.candidates):
        ids.append(loader.candidate_id(raw))
        blobs.append(loader.build_text_blob(raw))
        if args.limit and len(ids) >= args.limit:
            break
    n = len(ids)
    log(f"      {n:,} candidates, {time.time()-t0:.1f}s")

    # ---- 2. embeddings (checkpointed + resumable) ----
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device="cpu")
    model.max_seq_length = args.max_seq            # shorter context -> ~2x faster on CPU
    dim = model.get_sentence_embedding_dimension()

    partial = os.path.join(out, "_emb_partial.npy")
    progress = os.path.join(out, "_progress.json")
    emb = np.zeros((n, dim), dtype=np.float32)
    start = 0
    if os.path.exists(partial) and os.path.exists(progress):
        pj = json.load(open(progress))
        if pj.get("n") == n and pj.get("model") == args.model:
            saved = np.load(partial)
            k = min(len(saved), n)
            emb[:k] = saved[:k]
            start = pj.get("done", 0)
            log(f"      resuming from checkpoint at {start:,}/{n:,}")

    log(f"[2/4] Encoding with {args.model} (max_seq={args.max_seq}) from {start:,} ...")
    bs = args.batch_size
    for i in range(start, n, bs):
        chunk = blobs[i:i + bs]
        vecs = model.encode(chunk, batch_size=bs, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
        emb[i:i + len(chunk)] = vecs
        done = min(i + bs, n)
        if (i // bs) % 20 == 0 or done == n:
            rate = (done - start) / max(1e-6, time.time() - t0)
            eta = (n - done) / max(1e-6, rate)
            log(f"      {done:,}/{n:,}  ({rate:.0f}/s, eta {eta/60:.0f}m)")
        if (i // bs) % 50 == 0 and done < n:        # checkpoint every ~50 batches
            np.save(partial, emb[:done].astype(np.float16))
            json.dump({"n": n, "model": args.model, "done": done}, open(progress, "w"))

    emb16 = emb.astype(np.float16)

    facets = rubric["facets"]
    facet_text = [BGE_QUERY_PREFIX + f for f in facets] if "bge" in args.model.lower() else facets
    facet_emb = model.encode(facet_text, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    # ---- 3. SAVE CRITICAL ARTIFACTS NOW (before the optional, riskier BM25) ----
    log("[3/4] Saving embeddings + ids + facets + meta ...")
    json.dump(ids, open(os.path.join(out, "candidate_ids.json"), "w"))
    np.save(os.path.join(out, "cand_emb.npy"), emb16)
    np.save(os.path.join(out, "jd_facet_emb.npy"), facet_emb)
    meta = {
        "model": args.model, "dim": int(dim), "n_candidates": n,
        "n_facets": len(facets), "seed": SEED, "max_seq": args.max_seq,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1), "emb_dtype": "float16",
    }
    json.dump(meta, open(os.path.join(out, "precompute_meta.json"), "w"), indent=2)
    log(f"      saved cand_emb {emb16.shape} ({emb16.nbytes/1e6:.1f} MB)")

    # ---- 4. BM25 (optional; rank.py does not depend on it — non-fatal) ----
    bm25_scores = np.zeros(n, dtype=np.float32)
    if not args.skip_bm25:
        try:
            log("[4/4] Building BM25 index + scoring vs JD query ...")
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi([tokenize(b) for b in blobs])
            query_terms = set()
            for s in rubric["jd_relevant_skills"]:
                query_terms.update(tokenize(s))
            for f in facets:
                query_terms.update(tokenize(f))
            scores = bm25.get_scores(list(query_terms))
            mx = float(scores.max()) if scores.max() > 0 else 1.0
            bm25_scores = (scores / mx).astype(np.float32)
            log(f"      bm25 max={mx:.2f}")
        except Exception as e:
            log(f"      BM25 failed (non-fatal, saving zeros): {e}")
    np.save(os.path.join(out, "bm25_scores.npy"), bm25_scores)

    # cleanup checkpoints
    for f in (partial, progress):
        if os.path.exists(f):
            os.remove(f)
    log(f"DONE in {time.time()-t0:.1f}s. cand_emb {emb16.shape}, facets {facet_emb.shape}")


if __name__ == "__main__":
    main()
