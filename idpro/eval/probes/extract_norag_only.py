"""
Extract A+B+C pool views (norag only) for the ESM3 ablation arms (S0 / S1).

For each protein we do ONE forward pass with rag_context=None and save:
  view_a_prompteol_l48, view_b_question_mean_l48, view_c_eos_l64

Outputs (default OUT_DIR):
  {which}_norag_embeddings.pt — dict: accession -> {labels, sequence_length,
                                                     view_a_..., view_b_..., view_c_...}

Run:
  CUDA_VISIBLE_DEVICES=0 \\
    /data/avi/.conda/envs/protein2text_env/bin/python \\
      idpro/scripts/extract_norag_only.py \\
        --ckpt idpro/checkpoints/esm3_S0_seqonly/stage4_step20000 \\
        --encoder esm3-1.4b \\
        --which reference \\
        --out-dir idpro/data/probe/embeddings_S0_norag

  Add  --structure-track  --structure-manifest idpro/data/structure_manifest.jsonl
  for the S1 arm.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from idpro.eval.probes.extract_probe_embeddings import (  # noqa: E402
    build_model, load_checkpoint, load_jsonl, extract_views,
    DATA_DIR, MAX_SEQ_LEN,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--which", choices=["reference", "benchmark", "dark"], required=True)
    ap.add_argument("--encoder", default="esm3-1.4b",
                    choices=["esmc-600m", "esmc-300m", "esm2-650m", "esm2-3b", "esm3-1.4b"])
    ap.add_argument("--structure-track", action="store_true")
    ap.add_argument("--structure-manifest", default="",
                    help="JSONL of {accession, pdb_path} — required for --structure-track")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Where to write {which}_norag_embeddings.pt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    structure_lookup: Dict[str, str] = {}
    if args.structure_track:
        if not args.structure_manifest:
            raise SystemExit("--structure-track requires --structure-manifest")
        with open(args.structure_manifest) as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                acc = rec.get("accession")
                pdb = rec.get("pdb_path")
                if acc and pdb:
                    structure_lookup[acc] = pdb
        print(f"Structure manifest: {len(structure_lookup):,} accessions", flush=True)

    rows = load_jsonl(DATA_DIR / f"{args.which}.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    out_path = args.out_dir / f"{args.which}_norag_embeddings.pt"

    results: Dict[str, Dict] = {}
    if out_path.exists():
        try:
            prev = torch.load(out_path, map_location="cpu", weights_only=False)
            if isinstance(prev, dict):
                results = prev
                print(f"Resuming: {len(results)} already done", flush=True)
        except Exception as e:
            print(f"Could not resume: {e}", flush=True)

    todo = [r for r in rows if r["accession"] not in results]
    print(f"To process: {len(todo)} / {len(rows)} (norag forward only)", flush=True)
    if not todo:
        return 0

    model = build_model(
        device,
        encoder_name=args.encoder,
        structure_track=args.structure_track,
        structure_manifest_path=args.structure_manifest,
    )
    load_checkpoint(model, args.ckpt, device)

    question = "What is the function of this protein?"
    save_every = 50
    t0 = time.time()
    for i, row in enumerate(todo):
        acc = row["accession"]
        seq = row["sequence"][:MAX_SEQ_LEN]
        pdb = structure_lookup.get(acc) if args.structure_track else None

        try:
            views = extract_views(model, seq, question, rag_context="",
                                  device=device, pdb_path=pdb)
        except Exception as e:
            print(f"  [error] {acc}: {e}", flush=True)
            continue

        rec = {
            "labels": row["labels"],
            "sequence_length": len(seq),
            **{k: v.to(torch.float16) for k, v in views.items()},
        }
        results[acc] = rec

        if (i + 1) % 5 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(todo) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(todo)}] rate={rate:.2f}/s  eta={eta/60:.1f}min", flush=True)

        if (i + 1) % save_every == 0:
            torch.save(results, out_path)

    torch.save(results, out_path)
    print(f"\nDone. {len(results)} proteins → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
