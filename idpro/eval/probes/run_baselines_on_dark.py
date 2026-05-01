"""
Run each baseline method on the 415 dark-proteome proteins so we can do a
fair head-to-head spider plot.

Methods covered:
  - MMseqs2  (sequence similarity → transfer annotation from best hit in training set)
  - DeepFRI  (sequence-CNN function predictor; no structure needed in seq mode)
  - DeepGOMeta (metagenomic GO predictor; ESM-based sequence only)
  - BioReason-Pro (LLM reasoning; slow — skip or sample)

Writes results to $IDPRO_DATA_ROOT/benchmark/results/
under new filenames: {method}_dark_predictions.json

Run:
    CUDA_VISIBLE_DEVICES=1 python scripts/run_baselines_on_dark.py --methods mmseqs,deepfri,deepgometa
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from idpro.paths import DATA_ROOT as REPO  # noqa: E402

BENCH = REPO / "benchmark"
RESULTS = BENCH / "results"
DATA = BENCH / "data"
DARK_FASTA = DATA / "dark_proteins.fasta"
ENV_BIN = "/home/avi/.conda/envs/protein2text_env/bin"


# ---------------------------------------------------------------------------
# MMseqs2 — homology transfer from the training set (bacteria_all.jsonl)
# ---------------------------------------------------------------------------


def run_mmseqs_on_dark():
    """
    Build a reference FASTA from bacteria_all.jsonl (annotated proteins),
    then easy-search dark against it; transfer the reference protein's
    function text as the prediction.
    """
    print("\n=== MMseqs2 on dark ===")
    ref_fasta = DATA / "bacteria_annotated_ref.fasta"
    ref_json = DATA / "bacteria_annotated_ref.json"

    if not ref_fasta.exists():
        print("  Building reference FASTA + lookup from bacteria_all.jsonl ...")
        src = REPO / "preliminary_data" / "training_data" / "downloads" / \
              "uniprot_bacteria_features" / "bacteria_all.jsonl"
        lookup = {}
        n = 0
        with ref_fasta.open("w") as ff, src.open() as f:
            for line in f:
                r = json.loads(line)
                name = r.get("protein_name", "") or ""
                fn = r.get("cc_function", "") or ""
                ec = r.get("ec", []) or []
                go_f = r.get("go_f", []) or []
                # Build a descriptive text combining everything useful for keyword matching
                desc = " | ".join([name, f"EC {','.join(ec)}" if ec else "",
                                   fn[:500], "; ".join(go_f)]).strip(" |")
                acc = r["accession"]
                lookup[acc] = desc
                seq = (r.get("sequence") or "")[:1000]
                if seq:
                    ff.write(f">{acc}\n{seq}\n")
                    n += 1
        with ref_json.open("w") as f:
            json.dump(lookup, f)
        print(f"  wrote {ref_fasta} (n={n}) and {ref_json}")
    else:
        lookup = json.loads(ref_json.read_text())

    # Run easy-search
    out = RESULTS / "mmseqs_dark_hits.tsv"
    tmp = DATA / "mmseqs_dark_tmp"
    tmp.mkdir(exist_ok=True)
    t0 = time.time()
    cmd = (
        f"{ENV_BIN}/mmseqs easy-search "
        f"{DARK_FASTA} {ref_fasta} {out} {tmp} "
        f"-e 1e-3 --threads 16 --max-seqs 1 "
        f"--format-output 'query,target,evalue,pident,qcov,tcov'"
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1200)
    print(f"  mmseqs done in {time.time()-t0:.1f}s, rc={r.returncode}")
    if r.returncode != 0:
        print(r.stderr[-600:])

    # Parse hits → per-query prediction text
    preds = {}
    if out.exists():
        with out.open() as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    q, t = parts[0], parts[1]
                    if q not in preds:
                        preds[q] = lookup.get(t, "")

    out_json = RESULTS / "mmseqs_dark_predictions.json"
    out_json.write_text(json.dumps(preds, indent=2))
    print(f"  MMseqs2: {len(preds)} / 415 dark proteins got a hit (-> {out_json})")
    return preds


# ---------------------------------------------------------------------------
# DeepFRI on dark (sequence-CNN mode)
# ---------------------------------------------------------------------------


def run_deepfri_on_dark():
    print("\n=== DeepFRI on dark (sequence-CNN mode) ===")
    dfri = BENCH / "DeepFRI"
    if not dfri.exists():
        print("  DeepFRI directory missing; skipping")
        return {}

    # Check newest_trained_models archive
    models_tar = dfri / "newest_trained_models.tar.gz"
    models_dir = dfri / "trained_models"
    if not models_dir.exists() and models_tar.exists():
        print("  Extracting DeepFRI models ...")
        subprocess.run(["tar", "xzf", str(models_tar), "-C", str(dfri)], check=False)

    cfg = None
    for candidate in ["trained_models/model_config.json",
                      "newest_trained_models/model_config.json"]:
        p = dfri / candidate
        if p.exists():
            cfg = p
            break
    if cfg is None:
        print("  No DeepFRI model_config.json found; skipping")
        return {}

    out_prefix = RESULTS / "deepfri_dark"
    cmd = (
        f"cd {dfri} && {ENV_BIN}/python predict.py "
        f"--fasta_fn {DARK_FASTA} "
        f"-ont mf "
        f"-o {out_prefix} "
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
    print(f"  DeepFRI done in {time.time()-t0:.1f}s, rc={r.returncode}")
    if r.returncode != 0:
        print("  stderr tail:")
        print(r.stderr[-1500:])

    # Collect predictions: parse the _predictions.csv output
    preds = {}
    csv_path = RESULTS / "deepfri_dark_MF_predictions.csv"
    pred_scores_path = RESULTS / "deepfri_dark_MF_pred_scores.json"
    if csv_path.exists():
        # Format: Protein,GO_term,Score,GO_term_name
        import csv
        per_prot = {}
        with csv_path.open() as f:
            rdr = csv.reader(f)
            header = next(rdr, None)
            for row in rdr:
                if len(row) < 4:
                    continue
                prot, go_id, score, go_name = row[0], row[1], row[2], row[3]
                per_prot.setdefault(prot, []).append((float(score), go_id, go_name))
        for prot, items in per_prot.items():
            items.sort(reverse=True)
            # Top-10 GO terms; concat with GO-term-name so keyword matcher works.
            top = items[:10]
            text_parts = [f"{g} {n}" for (s, g, n) in top]
            preds[prot] = " | ".join(text_parts)
    out_json = RESULTS / "deepfri_dark_predictions.json"
    out_json.write_text(json.dumps(preds, indent=2))
    print(f"  DeepFRI: {len(preds)} / 415 dark proteins annotated -> {out_json}")
    return preds


# ---------------------------------------------------------------------------
# DeepGOMeta on dark
# ---------------------------------------------------------------------------


def run_deepgometa_on_dark():
    print("\n=== DeepGOMeta on dark ===")
    dgm = BENCH / "deepgometa"
    if not dgm.exists():
        print("  deepgometa directory missing; skipping")
        return {}

    if not (dgm / "data").exists() or not list((dgm / "data").iterdir()):
        print("  DeepGOMeta data not present; skipping")
        return {}

    out_dir = RESULTS / "deepgometa_dark_out"
    out_dir.mkdir(exist_ok=True)
    # Typical DeepGOMeta interface: -if <fasta>, writes predictions alongside input
    cmd = (
        f"cd {dgm} && {ENV_BIN}/python predict.py "
        f"-if {DARK_FASTA} "
        f"-of {out_dir / 'dark_predictions.tsv'}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
    print(f"  DeepGOMeta done in {time.time()-t0:.1f}s, rc={r.returncode}")
    if r.returncode != 0:
        print("  stderr tail:")
        print(r.stderr[-1500:])

    # Parse predictions → GO-id list per protein
    preds = {}
    tsv = out_dir / "dark_predictions.tsv"
    if tsv.exists():
        with tsv.open() as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                prot = parts[0]
                # Remaining fields: GO:xxxxxxx|score or just space-separated
                gos = []
                for p in parts[1:]:
                    for tok in p.split():
                        if tok.startswith("GO:"):
                            gos.append(tok.split("|")[0])
                preds[prot] = " ".join(gos[:20])

    out_json = RESULTS / "deepgometa_dark_predictions.json"
    out_json.write_text(json.dumps(preds, indent=2))
    print(f"  DeepGOMeta: {len(preds)} / 415 dark proteins -> {out_json}")
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="mmseqs,deepfri,deepgometa",
                    help="comma-separated list from {mmseqs, deepfri, deepgometa}")
    args = ap.parse_args()

    ms = [m.strip() for m in args.methods.split(",") if m.strip()]
    outs = {}
    if "mmseqs" in ms:
        outs["mmseqs"] = run_mmseqs_on_dark()
    if "deepfri" in ms:
        outs["deepfri"] = run_deepfri_on_dark()
    if "deepgometa" in ms:
        outs["deepgometa"] = run_deepgometa_on_dark()

    print("\n=== Summary ===")
    for m, p in outs.items():
        print(f"  {m}: {len(p)} predictions")


if __name__ == "__main__":
    main()
