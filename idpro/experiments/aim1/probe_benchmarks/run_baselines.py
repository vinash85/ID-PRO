"""
One-stop runner for the Aim-1A baseline EC predictors. Replaces the pre-refactor
trio of `run_baselines_on_dark.py` + the two `parse_*` scripts.

Each method writes to BASELINE_PREDS_DIR/<method>_<split>_predictions.json,
where each value is a free-text string the strict-keyword scorer in
`utils.metrics` can score (`strict_class_scores` for natural-language methods;
`deepgometa_strict_scores` for raw GO-ID methods).

Methods
-------
  mmseqs       — MMseqs2 easy-search against the annotated bacteria reference.
                 Per-query top hit's annotation text becomes the prediction.
  deepfri      — DeepFRI sequence-CNN; we collect the top-10 GO-MF terms.
  deepgometa   — DeepGOMeta metagenome predictor; raw GO-IDs.
  clean        — Parse pre-existing CLEAN CSV outputs (run upstream).
  interlabelgo — Parse pre-existing InterLabelGO+ TSV outputs.

The runner methods (mmseqs, deepfri, deepgometa) shell out to env-specific
binaries; the parsers (clean, interlabelgo) only need the upstream output dirs.

Inputs
------
Splits come from PROBE_SPLITS_DIR/{split}.jsonl. We materialize a temporary
FASTA for each split as needed.

External-tool layout — defaults read from env vars; CLI flags override:
  IDPRO_BASELINE_TOOLS_DIR  (default: $IDPRO_DATA_ROOT/benchmark)
    └── DeepFRI/             (predict.py, trained_models/)
    └── deepgometa/          (predict.py, data/)
    └── CLEAN/app/results/inputs/{split}_maxsep.csv
    └── InterLabelGO/InterLabelGO+/work_{split}/InterLabelGO+.tsv
  IDPRO_BASELINE_BIN        (default: ~/.conda/envs/protein2text_env/bin)
    └── mmseqs, python

Reference build for MMseqs2 reads from $IDPRO_DATA_ROOT/preliminary_data/
training_data/downloads/uniprot_bacteria_features/bacteria_all.jsonl
(override with --bacteria-jsonl).

CLI
---
python idpro/experiments/aim1/probe_benchmarks/run_baselines.py \
    --methods mmseqs,deepfri,deepgometa --on benchmark,dark
python idpro/experiments/aim1/probe_benchmarks/run_baselines.py \
    --methods clean,interlabelgo --on benchmark,dark,reference
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.paths import (  # noqa: E402
    BASELINE_PREDS_DIR,
    DATA_ROOT,
    PROBE_SPLITS_DIR,
)

BASELINE_PREDS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TOOLS_DIR = Path(os.environ.get(
    "IDPRO_BASELINE_TOOLS_DIR", str(DATA_ROOT / "benchmark")))
DEFAULT_BIN = Path(os.environ.get(
    "IDPRO_BASELINE_BIN", str(Path.home() / ".conda" / "envs" / "protein2text_env" / "bin")))
DEFAULT_BACTERIA_JSONL = (
    DATA_ROOT / "preliminary_data" / "training_data" / "downloads"
    / "uniprot_bacteria_features" / "bacteria_all.jsonl"
)

ALL_METHODS = ["mmseqs", "deepfri", "deepgometa", "clean", "interlabelgo"]
ALL_SPLITS = ["reference", "benchmark", "dark"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _split_jsonl(split: str) -> Path:
    return PROBE_SPLITS_DIR / f"{split}.jsonl"


def _load_split(split: str) -> List[dict]:
    p = _split_jsonl(split)
    if not p.exists():
        raise SystemExit(
            f"Missing split {p}. Run "
            f"`data_prep/prepare_{'dark' if split == 'dark' else 'reference_benchmark'}.py` first."
        )
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _materialize_fasta(split: str, work_dir: Path, max_len: int = 1000) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    fa = work_dir / f"{split}.fasta"
    rows = _load_split(split)
    with fa.open("w") as f:
        for r in rows:
            seq = (r.get("sequence") or "")[:max_len]
            if seq:
                f.write(f">{r['accession']}\n{seq}\n")
    print(f"  fasta: wrote {fa} (n={len(rows)})")
    return fa


def _out_path(method: str, split: str) -> Path:
    return BASELINE_PREDS_DIR / f"{method}_{split}_predictions.json"


# ---------------------------------------------------------------------------
# MMseqs2
# ---------------------------------------------------------------------------


def _build_mmseqs_reference(args: argparse.Namespace) -> tuple[Path, Path]:
    ref_fasta = args.work_dir / "bacteria_annotated_ref.fasta"
    ref_json = args.work_dir / "bacteria_annotated_ref.json"
    if ref_fasta.exists() and ref_json.exists():
        return ref_fasta, ref_json
    src = Path(args.bacteria_jsonl)
    if not src.exists():
        raise SystemExit(f"MMseqs2 reference source missing: {src}")
    print(f"  building MMseqs2 reference from {src}")
    lookup: Dict[str, str] = {}
    n = 0
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with ref_fasta.open("w") as ff, src.open() as f:
        for line in f:
            r = json.loads(line)
            name = r.get("protein_name") or ""
            fn = r.get("cc_function") or ""
            ec = r.get("ec") or []
            go_f = r.get("go_f") or []
            desc_parts = [name, f"EC {','.join(ec)}" if ec else "", fn[:500], "; ".join(go_f)]
            desc = " | ".join(p for p in desc_parts if p).strip(" |")
            acc = r["accession"]
            lookup[acc] = desc
            seq = (r.get("sequence") or "")[:1000]
            if seq:
                ff.write(f">{acc}\n{seq}\n")
                n += 1
    ref_json.write_text(json.dumps(lookup))
    print(f"  wrote {ref_fasta} (n={n})")
    return ref_fasta, ref_json


def run_mmseqs(args: argparse.Namespace, splits: List[str]) -> None:
    print("=== MMseqs2 ===")
    bin_path = args.bin_dir / "mmseqs"
    if not bin_path.exists():
        raise SystemExit(f"mmseqs binary not found at {bin_path}")
    ref_fasta, ref_json = _build_mmseqs_reference(args)
    lookup = json.loads(ref_json.read_text())

    for split in splits:
        print(f"\n--- mmseqs / {split} ---")
        fa = _materialize_fasta(split, args.work_dir)
        out_tsv = args.work_dir / f"mmseqs_{split}_hits.tsv"
        tmp = args.work_dir / f"mmseqs_{split}_tmp"
        tmp.mkdir(exist_ok=True)
        cmd = (
            f"{bin_path} easy-search {fa} {ref_fasta} {out_tsv} {tmp} "
            f"-e 1e-3 --threads {args.threads} --max-seqs 1 "
            f"--format-output 'query,target,evalue,pident,qcov,tcov'"
        )
        t0 = time.time()
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=args.timeout)
        print(f"  done in {time.time() - t0:.1f}s rc={r.returncode}")
        if r.returncode != 0:
            print(r.stderr[-600:])
            continue

        preds: Dict[str, str] = {}
        with out_tsv.open() as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[0] not in preds:
                    preds[parts[0]] = lookup.get(parts[1], "")
        out = _out_path("mmseqs", split)
        out.write_text(json.dumps(preds, indent=2))
        print(f"  wrote {out}  hits={len(preds)}")


# ---------------------------------------------------------------------------
# DeepFRI
# ---------------------------------------------------------------------------


def run_deepfri(args: argparse.Namespace, splits: List[str]) -> None:
    print("=== DeepFRI ===")
    dfri = args.tools_dir / "DeepFRI"
    if not dfri.exists():
        raise SystemExit(f"DeepFRI dir not found at {dfri}")

    models_tar = dfri / "newest_trained_models.tar.gz"
    if not (dfri / "trained_models").exists() and models_tar.exists():
        print("  extracting DeepFRI models")
        subprocess.run(["tar", "xzf", str(models_tar), "-C", str(dfri)], check=False)

    py = args.bin_dir / "python"
    for split in splits:
        print(f"\n--- deepfri / {split} ---")
        fa = _materialize_fasta(split, args.work_dir)
        out_prefix = args.work_dir / f"deepfri_{split}"
        cmd = f"cd {dfri} && {py} predict.py --fasta_fn {fa} -ont mf -o {out_prefix}"
        t0 = time.time()
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=args.timeout)
        print(f"  done in {time.time() - t0:.1f}s rc={r.returncode}")
        if r.returncode != 0:
            print(r.stderr[-1500:])

        csv_path = Path(f"{out_prefix}_MF_predictions.csv")
        preds: Dict[str, str] = {}
        if csv_path.exists():
            per_prot: Dict[str, list] = defaultdict(list)
            with csv_path.open() as f:
                rdr = csv.reader(f)
                next(rdr, None)  # header
                for row in rdr:
                    if len(row) < 4:
                        continue
                    prot, go_id, score, go_name = row[0], row[1], row[2], row[3]
                    per_prot[prot].append((float(score), go_id, go_name))
            for prot, items in per_prot.items():
                items.sort(reverse=True)
                preds[prot] = " | ".join(f"{g} {n}" for (_s, g, n) in items[:10])
        out = _out_path("deepfri", split)
        out.write_text(json.dumps(preds, indent=2))
        print(f"  wrote {out}  n={len(preds)}")


# ---------------------------------------------------------------------------
# DeepGOMeta
# ---------------------------------------------------------------------------


def run_deepgometa(args: argparse.Namespace, splits: List[str]) -> None:
    print("=== DeepGOMeta ===")
    dgm = args.tools_dir / "deepgometa"
    if not dgm.exists():
        raise SystemExit(f"deepgometa dir not found at {dgm}")
    if not (dgm / "data").exists() or not list((dgm / "data").iterdir()):
        raise SystemExit(f"deepgometa data missing under {dgm / 'data'}")

    py = args.bin_dir / "python"
    for split in splits:
        print(f"\n--- deepgometa / {split} ---")
        fa = _materialize_fasta(split, args.work_dir)
        out_tsv = args.work_dir / f"deepgometa_{split}_predictions.tsv"
        cmd = f"cd {dgm} && {py} predict.py -if {fa} -of {out_tsv}"
        t0 = time.time()
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=args.timeout)
        print(f"  done in {time.time() - t0:.1f}s rc={r.returncode}")
        if r.returncode != 0:
            print(r.stderr[-1500:])

        preds: Dict[str, str] = {}
        if out_tsv.exists():
            with out_tsv.open() as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    prot = parts[0]
                    gos: List[str] = []
                    for p in parts[1:]:
                        for tok in p.split():
                            if tok.startswith("GO:"):
                                gos.append(tok.split("|")[0])
                    preds[prot] = " ".join(gos[:20])
        out = _out_path("deepgometa", split)
        out.write_text(json.dumps(preds, indent=2))
        print(f"  wrote {out}  n={len(preds)}")


# ---------------------------------------------------------------------------
# CLEAN parser
# ---------------------------------------------------------------------------

_EC_TO_NAME = {
    "1": "oxidoreductase", "2": "transferase", "3": "hydrolase",
    "4": "lyase", "5": "isomerase", "6": "ligase", "7": "translocase",
}


def _parse_clean_csv(csv_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with csv_path.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            pid = row[0]
            preds = []
            for tok in row[1:]:
                if "/" not in tok:
                    continue
                ec, score = tok.split("/", 1)
                ec = ec.replace("EC:", "").strip()
                try:
                    s = float(score)
                except ValueError:
                    s = 0.0
                preds.append((ec, s))
            parts = []
            for ec, s in preds:
                ec1 = ec.split(".")[0]
                name = _EC_TO_NAME.get(ec1, "")
                parts.append(f"EC {ec} ({name}, score={s:.3f})")
            out[pid] = "; ".join(parts)
    return out


def parse_clean(args: argparse.Namespace, splits: List[str]) -> None:
    print("=== CLEAN (parse) ===")
    base = args.tools_dir / "CLEAN" / "app" / "results" / "inputs"
    for split in splits:
        src = base / f"{split}_maxsep.csv"
        if not src.exists():
            print(f"  skip {split}: {src} missing")
            continue
        preds = _parse_clean_csv(src)
        out = _out_path("clean", split)
        out.write_text(json.dumps(preds, indent=2))
        print(f"  {split}: {len(preds)} → {out}")


# ---------------------------------------------------------------------------
# InterLabelGO+ parser
# ---------------------------------------------------------------------------

INTERLABELGO_TOP_K = 30
INTERLABELGO_MIN_SCORE = 0.05


def _parse_interlabelgo_tsv(tsv_path: Path) -> Dict[str, str]:
    bucket: Dict[str, list] = defaultdict(list)
    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                score = float(row["score"])
            except (KeyError, ValueError):
                continue
            if score < INTERLABELGO_MIN_SCORE:
                continue
            entry = row["EntryID"]
            bucket[entry].append((score, row["term"], row.get("go_term_name", "").strip(),
                                  row.get("aspect", "")))
    out: Dict[str, str] = {}
    for entry, terms in bucket.items():
        terms.sort(reverse=True)
        terms = terms[:INTERLABELGO_TOP_K]
        pid = entry.split("|")[1] if "|" in entry else entry
        out[pid] = "; ".join(f"{go} {name} ({asp}, score={s:.3f})"
                             for (s, go, name, asp) in terms)
    return out


def parse_interlabelgo(args: argparse.Namespace, splits: List[str]) -> None:
    print("=== InterLabelGO+ (parse) ===")
    base = args.tools_dir / "InterLabelGO" / "InterLabelGO+"
    for split in splits:
        src = base / f"work_{split}" / "InterLabelGO+.tsv"
        if not src.exists():
            print(f"  skip {split}: {src} missing")
            continue
        preds = _parse_interlabelgo_tsv(src)
        out = _out_path("interlabelgo", split)
        out.write_text(json.dumps(preds, indent=2))
        print(f"  {split}: {len(preds)} → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


METHOD_FNS = {
    "mmseqs": run_mmseqs,
    "deepfri": run_deepfri,
    "deepgometa": run_deepgometa,
    "clean": parse_clean,
    "interlabelgo": parse_interlabelgo,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="mmseqs,deepfri,deepgometa",
                    help=f"Comma-separated subset of {ALL_METHODS}")
    ap.add_argument("--on", default="benchmark,dark",
                    help=f"Comma-separated subset of {ALL_SPLITS}")
    ap.add_argument("--tools-dir", type=Path, default=DEFAULT_TOOLS_DIR,
                    help="Root containing DeepFRI/, deepgometa/, CLEAN/, InterLabelGO/")
    ap.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN,
                    help="Directory containing mmseqs + python from the tooling env")
    ap.add_argument("--bacteria-jsonl", type=Path, default=DEFAULT_BACTERIA_JSONL,
                    help="MMseqs2 reference: annotated bacteria jsonl")
    ap.add_argument("--work-dir", type=Path,
                    default=BASELINE_PREDS_DIR / "_work",
                    help="Scratch dir for FASTAs and intermediate hits")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=3600,
                    help="Per-tool subprocess timeout in seconds")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    splits = [s.strip() for s in args.on.split(",") if s.strip()]
    bad_m = [m for m in methods if m not in METHOD_FNS]
    bad_s = [s for s in splits if s not in ALL_SPLITS]
    if bad_m or bad_s:
        raise SystemExit(f"Unknown methods={bad_m} or splits={bad_s}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for m in methods:
        METHOD_FNS[m](args, splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
