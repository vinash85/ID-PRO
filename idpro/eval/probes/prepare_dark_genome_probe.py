"""
Prepare the 415 dark-genome proteins as a third probe test set.

Dark-genome proteins are UNANNOTATED by definition (no UniProt EC), but
many have WEAK labels from InterProScan automated calls: Pfam IDs, GO terms,
sometimes partial function text. We use these as a noisy proxy for
"the probe predicts something sensible."

Outputs: idpro/data/probe/dark.jsonl (same schema as reference.jsonl)

Run:
    python scripts/prepare_dark_genome_probe.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from idpro.paths import PROBE_DIR as DATA_DIR, DATA_ROOT  # noqa: E402

META = DATA_ROOT / "preliminary_data" / "dark_genome" / "dark_genome_metadata.tsv"

GO_F_RE = re.compile(r"F:([^;]+)")


def main():
    labels = json.loads((DATA_DIR / "labels.json").read_text())
    go_vocab = labels["go_f_vocab"]
    pf_vocab = labels["pfam_vocab"]

    rows = []
    with META.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if not r["sequence"]:
                continue
            pfams = set()
            for p in (r.get("pfam") or "").replace(" ", "").split(";"):
                if p:
                    pfams.add(p)
            go_f = set()
            for m in GO_F_RE.finditer(r.get("go_terms") or ""):
                go_f.add(m.group(1).strip())

            row = {
                "accession": r["accession"],
                "sequence": r["sequence"][:1000],
                "protein_name": r.get("protein_name", ""),
                "description": (r.get("function") or r.get("protein_name") or "")[:300],
                "labels": {
                    # is_enzyme: weak — proxy from "any Pfam family" (bad but workable).
                    #            We prefer the activity-word heuristic below.
                    "is_enzyme": int(
                        "activity" in (r.get("go_terms") or "").lower()
                        or any(
                            kw in (r.get("function") or "").lower()
                            for kw in ["catalyz", "enzym", "hydrolase", "transferase", "kinase"]
                        )
                    ),
                    "ec_l1": None,            # unknown — we DON'T have EC annotations
                    "go_f": [1 if t in go_f else 0 for t in go_vocab],
                    "pfam": [1 if p in pfams else 0 for p in pf_vocab],
                },
                "_weak_labels": {
                    "has_pfam_any": int(bool(pfams)),
                    "has_go_terms": int(bool(go_f)),
                    "n_pfam": len(pfams),
                    "n_go_f": len(go_f),
                    "raw_pfams": sorted(pfams),
                    "raw_go_f": sorted(go_f),
                },
            }
            rows.append(row)

    out = DATA_DIR / "dark.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Summary
    n_en = sum(r["labels"]["is_enzyme"] for r in rows)
    n_go = sum(1 for r in rows if r["_weak_labels"]["n_go_f"])
    n_pf = sum(1 for r in rows if r["_weak_labels"]["n_pfam"])
    n_topgo = sum(1 for r in rows if sum(r["labels"]["go_f"]) > 0)
    n_toppf = sum(1 for r in rows if sum(r["labels"]["pfam"]) > 0)
    print(f"Wrote {out}  N={len(rows)}")
    print(f"  weak is_enzyme=1: {n_en}/{len(rows)}")
    print(f"  have any Pfam annotation:  {n_pf}")
    print(f"  have any GO-F annotation:  {n_go}")
    print(f"  have top-20 Pfam hit:      {n_toppf}")
    print(f"  have top-20 GO-F hit:      {n_topgo}")


if __name__ == "__main__":
    main()
