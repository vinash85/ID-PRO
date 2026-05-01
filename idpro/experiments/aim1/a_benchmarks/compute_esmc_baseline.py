"""
Compute ESM C mean-pool embeddings for the benchmark proteins so the
probe-training script can include an "ESM C only" baseline (query-agnostic,
no LLM reasoning).

Adds to idpro/data/probe/embeddings/rag_index.npz so the baseline is available
to all benchmark accessions too. The resulting index has both reference and
benchmark IDs.

Run:
    CUDA_VISIBLE_DEVICES=1 python scripts/compute_esmc_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.config import IDProConfig  # noqa: E402
from idpro.model.p2t.encoder import ProteinEncoder  # noqa: E402
from idpro.paths import AIM1_PROBE_DIR as DATA_DIR  # noqa: E402

EMB_DIR = DATA_DIR / "embeddings"
MAX_SEQ_LEN = 1000


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.open() if l.strip()]


def main():
    device = "cuda"
    cfg = IDProConfig()
    cfg.encoder.name = "esmc-600m"
    cfg.encoder.resolve()
    encoder = ProteinEncoder(cfg.encoder)
    encoder.load(device)

    # Load existing rag_index (has reference embeddings)
    rag_path = EMB_DIR / "rag_index.npz"
    existing = {}
    if rag_path.exists():
        npz = np.load(rag_path, allow_pickle=True)
        for i, acc in enumerate(npz["ids"].tolist()):
            existing[acc] = npz["embs"][i]
        print(f"Loaded {len(existing)} existing ESM C embeddings from rag_index.npz")

    # Compute for benchmark if missing
    bench_rows = load_jsonl(DATA_DIR / "benchmark.jsonl")
    todo = [r for r in bench_rows if r["accession"] not in existing]
    print(f"Benchmark rows: {len(bench_rows)}  need ESM C: {len(todo)}")

    bs = 16
    for i in range(0, len(todo), bs):
        batch = todo[i : i + bs]
        seqs = [r["sequence"][:MAX_SEQ_LEN] for r in batch]
        emb, mask = encoder.encode(seqs, device)
        for j, r in enumerate(batch):
            n = int(mask[j].sum().item())
            existing[r["accession"]] = emb[j, :n].mean(dim=0).float().cpu().numpy().astype(np.float32)
        if (i // bs) % 5 == 0:
            print(f"  {min(i + bs, len(todo))}/{len(todo)}")

    # L2-normalize everything (same convention as RAG build)
    all_ids = list(existing.keys())
    arr = np.stack([existing[a] for a in all_ids]).astype(np.float32)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
    np.savez(rag_path, embs=arr, ids=np.array(all_ids))
    print(f"Wrote {rag_path}   total={len(all_ids)}  dim={arr.shape[1]}")


if __name__ == "__main__":
    main()
