#!/usr/bin/env python3
"""
Build the ESM3 structure manifest: a JSONL mapping UniProt accession → PDB path.

Scans a directory of AlphaFold v6 PDB files named "AF-{accession}-F1-model_v6.pdb"
and writes one JSON record per line to the output manifest. Used by
`train_robust.py --structure-track --structure-manifest <path>` to look up
the structure for each training sample by its UniProt accession.

Usage:
  python idpro/scripts/build_structure_manifest.py \
      --pdb-dir preliminary_data/training_data/downloads/alphafold \
      --out     idpro/data/structure_manifest.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path


AF_NAME = re.compile(r"^AF-([A-Z0-9]+)-F1-model_v(\d+)\.pdb$")


def scan(pdb_dir: Path):
    n_total = 0
    n_skipped = 0
    seen = set()
    for path in sorted(pdb_dir.iterdir()):
        if not path.is_file():
            continue
        m = AF_NAME.match(path.name)
        if not m:
            n_skipped += 1
            continue
        accession = m.group(1)
        n_total += 1
        if accession in seen:
            # Multiple model versions for the same accession (shouldn't happen
            # with our downloader, but guard anyway). Take the lexicographically
            # later filename (later versions sort after earlier ones).
            continue
        seen.add(accession)
        yield accession, path
    print(f"  scanned {n_total} matching PDB files "
          f"({n_skipped} non-matching files skipped)", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb-dir", type=Path, required=True,
                   help="Directory containing AF-*-F1-model_v*.pdb files")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL manifest path")
    p.add_argument("--relative-to", type=Path, default=None,
                   help="Optional base path to write paths relative to "
                        "(e.g. the repo root). Default: store absolute paths.")
    args = p.parse_args()

    pdb_dir = args.pdb_dir.resolve()
    if not pdb_dir.is_dir():
        print(f"ERROR: --pdb-dir is not a directory: {pdb_dir}", file=sys.stderr)
        sys.exit(2)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(args.out, "w") as f:
        for accession, path in scan(pdb_dir):
            if args.relative_to:
                try:
                    rel = path.resolve().relative_to(args.relative_to.resolve())
                    pdb_path = str(rel)
                except ValueError:
                    pdb_path = str(path.resolve())
            else:
                pdb_path = str(path.resolve())
            f.write(json.dumps({"accession": accession, "pdb_path": pdb_path}) + "\n")
            n_written += 1

    print(f"  wrote {n_written:,} accessions → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
