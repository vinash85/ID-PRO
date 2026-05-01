"""
Prepare classification data + splits for the frozen-backbone probe experiment.

Outputs (to PROBE_SPLITS_DIR = datasets/probe_data/probe_splits/):
  reference.jsonl    — 3,000 characterized proteins (stratified by EC class).
                       Used BOTH as probe training data AND as the RAG source.
  benchmark.jsonl    — 669 benchmark proteins (held-out test set).
  labels.json        — label vocabularies (EC-L1 digits, top-20 GO-MF, top-20 Pfam).

Each row carries: accession, sequence, labels (is_enzyme, ec_l1, go_f_set, pfam_set),
and the UniProt function description (for RAG context).

Run:
    python idpro/experiments/aim1/probe_benchmarks/data_prep/prepare_reference_benchmark.py
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from idpro.paths import DATA_ROOT, PROBE_SPLITS_DIR  # noqa: E402

DATA_SRC = (
    DATA_ROOT
    / "preliminary_data"
    / "training_data"
    / "downloads"
    / "uniprot_bacteria_features"
    / "bacteria_all.jsonl"
)
BENCHMARK_SRC = (
    DATA_ROOT
    / "preliminary_data"
    / "benchmark"
    / "results"
    / "benchmark_results.jsonl"
)
# benchmark_results.jsonl has only (long_format_id, prompt, GT, pred); we need
# the FULL metadata (sequence, EC, GO, Pfam) for the same proteins. Pull from
# bacteria_all.jsonl by accession.

OUT_DIR = PROBE_SPLITS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_REFERENCE = 3_000
N_EC_PER_CLASS = 300   # 7 EC classes × 300 = 2,100 enzymes
N_NONENZYME = 900      # + 900 non-enzymes → 3,000 total, 70/30 enzyme/non-enzyme
TOP_K_GO = 20
TOP_K_PFAM = 20
MAX_SEQ_LEN = 1000  # protein sequence truncation
RANDOM_SEED = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_bacteria_index() -> dict:
    """Stream through the jsonl once, build accession → record dict."""
    idx = {}
    with DATA_SRC.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx[r["accession"]] = r
    print(f"  Loaded {len(idx)} bacteria records")
    return idx


def ec_level1(ec_list) -> Optional[int]:
    """Return the first-level EC digit (1-7), or None."""
    if not ec_list:
        return None
    m = re.match(r"^(\d+)\.", ec_list[0])
    if not m:
        return None
    try:
        v = int(m.group(1))
        return v if 1 <= v <= 7 else None
    except ValueError:
        return None


def make_labels(record: dict, go_vocab: List[str], pfam_vocab: List[str]) -> dict:
    ec = record.get("ec") or []
    go_f = set(record.get("go_f") or [])
    pfam = {x["id"] for x in (record.get("xrefs") or {}).get("pfam") or []}
    return {
        "is_enzyme": int(bool(ec)),
        "ec_l1": ec_level1(ec),  # 1-7 or None
        "go_f": [1 if t in go_f else 0 for t in go_vocab],
        "pfam": [1 if p in pfam else 0 for p in pfam_vocab],
    }


def build_vocab(records: List[dict]) -> tuple:
    """Top-K GO-F terms and Pfam IDs by frequency across the records."""
    gocount, pfcount = Counter(), Counter()
    for r in records:
        for t in (r.get("go_f") or []):
            gocount[t] += 1
        for x in (r.get("xrefs") or {}).get("pfam") or []:
            pfcount[x["id"]] += 1
    go_vocab = [t for t, _ in gocount.most_common(TOP_K_GO)]
    pf_vocab = [p for p, _ in pfcount.most_common(TOP_K_PFAM)]
    return go_vocab, pf_vocab


def make_rag_description(record: dict) -> str:
    """Short, RAG-ready function description."""
    parts = []
    name = record.get("protein_name") or ""
    if name:
        parts.append(name)
    ec = record.get("ec") or []
    if ec:
        parts.append(f"EC {ec[0]}")
    fn = record.get("cc_function") or ""
    if fn:
        parts.append(fn[:300])
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Build splits
# ---------------------------------------------------------------------------


def main() -> int:
    random.seed(RANDOM_SEED)
    print("Loading source data...")
    bact = load_bacteria_index()

    # Benchmark: recover UniProt accessions from long_format_id like "P0C2S4_0"
    print("Loading benchmark...")
    bench_ids = []
    with BENCHMARK_SRC.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            acc = r["long_format_id"].split("_")[0]
            bench_ids.append(acc)
    # Dedup preserving order
    seen = set()
    bench_ids = [a for a in bench_ids if not (a in seen or seen.add(a))]
    bench_records = [bact[a] for a in bench_ids if a in bact]
    print(f"  Benchmark: {len(bench_records)} / {len(bench_ids)} proteins matched")

    bench_ids_set = {r["accession"] for r in bench_records}

    # Reference: stratified sample from bacteria_all, excluding benchmark proteins.
    # Keep only proteins with sequence length in [50, MAX_SEQ_LEN] and at least one of
    # {EC, GO-F, Pfam} so we have something to classify against.
    print("Sampling reference set (stratified by EC class)...")
    by_class = defaultdict(list)
    for acc, r in bact.items():
        if acc in bench_ids_set:
            continue
        try:
            L = int(r.get("length", 0))
        except (ValueError, TypeError):
            continue
        if L < 50 or L > MAX_SEQ_LEN:
            continue
        ec1 = ec_level1(r.get("ec") or [])
        if ec1 is None:
            by_class["noenzyme"].append(r)
        else:
            by_class[ec1].append(r)

    for k, v in sorted(by_class.items(), key=lambda kv: str(kv[0])):
        print(f"    class {k}: {len(v)}")

    reference = []
    # Take roughly N_EC_PER_CLASS from each enzyme class
    for ec1 in range(1, 8):
        pool = by_class.get(ec1, [])
        if not pool:
            continue
        random.shuffle(pool)
        reference.extend(pool[:N_EC_PER_CLASS])
    # Add non-enzymes
    pool = by_class.get("noenzyme", [])
    random.shuffle(pool)
    reference.extend(pool[:N_NONENZYME])
    random.shuffle(reference)
    reference = reference[:N_REFERENCE]
    print(f"  Reference: {len(reference)} proteins")

    # Build vocab from the reference set (not including benchmark — clean split)
    go_vocab, pf_vocab = build_vocab(reference)
    print(f"  GO-F vocab (top {TOP_K_GO}):")
    for i, t in enumerate(go_vocab):
        print(f"    {i:2d}  {t}")
    print(f"  Pfam vocab (top {TOP_K_PFAM}):")
    for i, p in enumerate(pf_vocab):
        print(f"    {i:2d}  {p}")

    # Emit records with labels, sequences, RAG descriptions
    def emit(record: dict) -> dict:
        return {
            "accession": record["accession"],
            "sequence": record["sequence"][:MAX_SEQ_LEN],
            "protein_name": record.get("protein_name", ""),
            "description": make_rag_description(record),
            "labels": make_labels(record, go_vocab, pf_vocab),
        }

    # Write reference
    ref_path = OUT_DIR / "reference.jsonl"
    with ref_path.open("w") as f:
        for r in reference:
            f.write(json.dumps(emit(r)) + "\n")
    # Write benchmark
    bench_path = OUT_DIR / "benchmark.jsonl"
    with bench_path.open("w") as f:
        for r in bench_records:
            f.write(json.dumps(emit(r)) + "\n")

    # Label summary
    def counts(records):
        is_en = sum(1 for r in records if make_labels(r, go_vocab, pf_vocab)["is_enzyme"])
        ec_cnt = Counter()
        for r in records:
            v = ec_level1(r.get("ec") or [])
            ec_cnt[v if v is not None else "None"] += 1
        return is_en, ec_cnt

    ref_en, ref_ec = counts(reference)
    bench_en, bench_ec = counts(bench_records)
    print(f"\nReference: is_enzyme={ref_en}/{len(reference)}  EC L1 dist={dict(ref_ec)}")
    print(f"Benchmark: is_enzyme={bench_en}/{len(bench_records)}  EC L1 dist={dict(bench_ec)}")

    # Dump vocabs
    labels_path = OUT_DIR / "labels.json"
    labels_path.write_text(json.dumps({
        "go_f_vocab": go_vocab,
        "pfam_vocab": pf_vocab,
        "ec_l1_classes": list(range(1, 8)),
    }, indent=2))

    print()
    print(f"Wrote {ref_path}")
    print(f"Wrote {bench_path}")
    print(f"Wrote {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
