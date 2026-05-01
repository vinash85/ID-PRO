#!/usr/bin/env python3
"""
Filter existing QA jsonl files to the subset of proteins that have a structure
in the AlphaFold manifest.

When training ESM3 with the structure track populated, the QA pool must be
restricted to proteins for which a PDB exists. The existing JSONL rows already
carry an `id` field equal to the UniProt accession, so post-hoc filtering is
sufficient — no need to re-run generate_qa.py.

Usage (with env.sh sourced):
    python -m idpro.utils.filter_qa_by_structure \
        --manifest "$IDPRO_RUNS_ROOT/structure_manifest.jsonl" \
        --src-dir "$IDPRO_DATA_ROOT/preliminary_data/training_data/qa_stages" \
        --dst-dir "$IDPRO_REPO_ROOT/preliminary_data/training_data/qa_stages_struct"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_accession_set(manifest_path: Path) -> set[str]:
    accs: set[str] = set()
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            acc = rec.get("accession")
            if acc:
                accs.add(acc)
    return accs


def filter_jsonl(
    src: Path,
    dst: Path,
    keep: set[str],
    only_fullseq: bool = False,
) -> tuple[int, int]:
    """Filter `src` to `dst`, keeping rows whose `id` is in `keep`.

    If `only_fullseq` is True, also requires the row's `long_format_id` to
    correspond to a whole-protein QA category (`fullseq_allfeats` or
    `fullseq_domains`). Use this for stage 1 when training with the structure
    track on: fragment-classification QAs (TM helix, signal peptide, single
    domain etc.) pair a short substring with the full-protein PDB and trip
    ESM3's sequence-length-vs-structure assertion.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            if rec.get("id") not in keep:
                continue
            if only_fullseq:
                lfid = rec.get("long_format_id", "")
                if not (lfid.endswith("_fullseq_allfeats")
                        or lfid.endswith("_fullseq_domains")):
                    continue
            fout.write(line + "\n")
            n_out += 1
    return n_in, n_out


def main() -> int:
    import random
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--src-dir", required=True, type=Path,
                    help="Directory containing stageN/stageN_qa.jsonl")
    ap.add_argument("--dst-dir", required=True, type=Path,
                    help="Output directory; created if missing")
    ap.add_argument("--stages", default="1,4",
                    help="Comma-separated stage numbers to filter (default: 1,4)")
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="Fraction of structure-having accessions to keep (per-protein sample). Default 1.0 = no subsampling.")
    ap.add_argument("--sample-seed", type=int, default=42,
                    help="Random seed for the per-protein subsample.")
    ap.add_argument("--stage1-only-fullseq", action="store_true",
                    help="For stage 1, keep only fullseq_allfeats / fullseq_domains "
                         "QAs (drops fragment-classification rows that have "
                         "len(amino_seq) != PDB CA count).")
    args = ap.parse_args()

    if not (0.0 < args.sample_frac <= 1.0):
        raise SystemExit(f"--sample-frac must be in (0, 1]; got {args.sample_frac}")
    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if not args.src_dir.exists():
        raise SystemExit(f"src dir not found: {args.src_dir}")

    keep = load_accession_set(args.manifest)
    print(f"Manifest: {len(keep):,} accessions with structure")

    if args.sample_frac < 1.0:
        # Per-protein subsample: pick a deterministic subset of accessions.
        # Sort first to remove set-iteration nondeterminism, then shuffle with seed.
        accs_sorted = sorted(keep)
        rng = random.Random(args.sample_seed)
        rng.shuffle(accs_sorted)
        n_keep = int(round(len(accs_sorted) * args.sample_frac))
        keep = set(accs_sorted[:n_keep])
        print(f"Subsample: {len(keep):,} accessions kept "
              f"({100*args.sample_frac:.1f}%, seed={args.sample_seed})")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for stage in stages:
        src = args.src_dir / f"stage{stage}" / f"stage{stage}_qa.jsonl"
        dst = args.dst_dir / f"stage{stage}" / f"stage{stage}_qa.jsonl"
        if not src.exists():
            print(f"  [skip] {src} does not exist")
            continue
        only_fullseq = args.stage1_only_fullseq and stage == "1"
        n_in, n_out = filter_jsonl(src, dst, keep, only_fullseq=only_fullseq)
        pct = 100.0 * n_out / max(n_in, 1)
        tag = " [fullseq-only]" if only_fullseq else ""
        print(f"  stage{stage}{tag}: {n_out:,} / {n_in:,} kept ({pct:.1f}%) → {dst}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
