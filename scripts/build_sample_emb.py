"""Slice the demo sample's embeddings out of the official cand_emb.npy.

The HF Space ranks a fixed 100-candidate sample. Encoding it at request time made
the Space re-run the bge-small model on CPU on every slider/preset change (~99% CPU
+ the meta-tensor load crash). The 100 sample ids are all present in the official
100K, so we just slice their precomputed vectors and ship them — the demo then ranks
the sample with zero model inference, byte-identical to the official path.

Run from repo root:  python scripts/build_sample_emb.py
Output: artifacts/sample_cand_emb.npy  (N, 384) float32, aligned to sample jsonl order.
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
SAMPLE = os.path.join(ROOT, "app", "sample_candidates.jsonl")
OUT = os.path.join(ART, "sample_cand_emb.npy")


def main() -> None:
    ids = [
        json.loads(line)["candidate_id"] for line in open(SAMPLE, encoding="utf-8") if line.strip()
    ]
    all_ids = json.load(open(os.path.join(ART, "candidate_ids.json"), encoding="utf-8"))
    idx = {c: i for i, c in enumerate(all_ids)}
    missing = [c for c in ids if c not in idx]
    if missing:
        raise SystemExit(f"{len(missing)} sample ids not in candidate_ids.json: {missing[:5]}")

    full = np.load(os.path.join(ART, "cand_emb.npy"), mmap_mode="r")
    emb = np.stack([np.asarray(full[idx[c]], dtype=np.float32) for c in ids])
    np.save(OUT, emb)
    print(f"wrote {OUT}  shape={emb.shape} dtype={emb.dtype}  ({emb.nbytes/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
