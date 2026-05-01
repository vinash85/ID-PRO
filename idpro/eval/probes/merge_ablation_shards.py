"""
Merge sharded ablation-embedding files into a single combined file for the
probe training step.

Usage:
    python scripts/merge_ablation_shards.py --which reference
    python scripts/merge_ablation_shards.py --which benchmark
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from idpro.paths import PROBE_DIR  # noqa: E402

EMB_DIR = PROBE_DIR / "embeddings"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["reference", "benchmark"], required=True)
    args = ap.parse_args()

    pattern = str(EMB_DIR / f"ablation_{args.which}_embeddings_shard*.pt")
    shards = sorted(glob.glob(pattern))
    if not shards:
        print(f"No shards found for pattern: {pattern}")
        return 1
    print(f"Merging {len(shards)} shards for {args.which}:")

    merged = {}
    for s in shards:
        d = torch.load(s, map_location="cpu", weights_only=False)
        before = len(merged)
        merged.update(d)
        print(f"  {Path(s).name}: {len(d)} proteins  (total so far: {len(merged)})")
        if len(merged) - before < len(d):
            dup = len(d) - (len(merged) - before)
            print(f"    [warn] {dup} duplicate keys overwritten")

    out = EMB_DIR / f"ablation_{args.which}_embeddings.pt"
    torch.save(merged, out)
    print(f"\nWrote merged file: {out}   ({len(merged)} proteins)")


if __name__ == "__main__":
    raise SystemExit(main())
