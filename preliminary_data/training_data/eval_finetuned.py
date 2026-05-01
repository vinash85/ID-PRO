#!/usr/bin/env python3
"""
Evaluate the fine-tuned P2T model on the benchmark dataset.
Compare F1 between original P2T and fine-tuned P2T (with and without RAG).

The fine-tuned model saved full LLaMA weights. We load it as the model_base
for P2T (replacing the original LLaMA), with the original projector/resampler.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

P2T_DIR = Path("/data/asahu/projects/doe_genesis/Protein2Text")
PRELIM_DIR = Path("/data/asahu/projects/doe_genesis/preliminary_data")
FINETUNED_BASE = str(PRELIM_DIR / "training_data/finetuned_microbial_p2t")
ORIGINAL_BASE = "/data/ajararweh/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"
P2T_CHECKPOINT = str(P2T_DIR / "checkpoints/protein2text-llama3.1-8B-instruct-esm2-650M")
CONDA_ENV = "protein2text_env"

BENCHMARK = str(PRELIM_DIR / "benchmark/microbiome_benchmark.json")
BIOENERGY = str(PRELIM_DIR / "benchmark/bioenergy_enzymes.json")

EVAL_DIR = PRELIM_DIR / "training_data" / "eval_results"
EVAL_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(P2T_DIR))
sys.path.insert(0, str(Path("/data/asahu/projects/doe_genesis/autoresearch2")))
from prepare import compute_f1, compute_specificity, compute_coverage


def run_inference(input_file, output_file, model_base, gpu_id=1, temperature=0.3, max_tokens=512):
    """Run P2T inference with specified model base."""
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} conda run -n {CONDA_ENV} "
        f"python {P2T_DIR}/evaluation/inference_model.py "
        f"--input_file {input_file} --output_file {output_file} "
        f"--model_path {P2T_CHECKPOINT} --model_base {model_base} "
        f"--device cuda --temperature {temperature} --max_new_tokens {max_tokens}"
    )
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
    return result.returncode == 0, time.time() - t0


def load_jsonl(path):
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def evaluate(results_file):
    """Compute metrics on results."""
    results = load_jsonl(results_file)
    f1s = [compute_f1(r.get("Predicted", ""), r.get("Ground Truth", "")) for r in results]
    specs = [compute_specificity(r.get("Predicted", "")) for r in results]

    import numpy as np
    return {
        "n": len(results),
        "f1_mean": float(np.mean(f1s)),
        "f1_median": float(np.median(f1s)),
        "f1_std": float(np.std(f1s)),
        "coverage": sum(1 for s in specs if s > 0.3) / max(len(specs), 1),
        "specificity_mean": float(np.mean(specs)),
    }


def create_subset(input_file, output_file, n=50):
    """Create a subset for faster evaluation."""
    import random
    with open(input_file) as f:
        data = json.load(f)
    random.seed(42)
    subset = random.sample(data, min(n, len(data)))
    with open(output_file, 'w') as f:
        json.dump(subset, f, indent=2)
    return len(subset)


def main():
    print("=" * 60)
    print("EVALUATING FINE-TUNED P2T vs ORIGINAL")
    print("=" * 60)

    gpu_id = 1
    n_eval = 50  # samples per evaluation

    # Create evaluation subsets
    bench_subset = str(EVAL_DIR / "bench_subset.json")
    bio_subset = str(EVAL_DIR / "bio_subset.json")
    n1 = create_subset(BENCHMARK, bench_subset, n_eval)
    n2 = create_subset(BIOENERGY, bio_subset, n_eval)
    print(f"Benchmark subset: {n1}, Bioenergy subset: {n2}")

    results_table = []

    # ── Run 1: Original P2T on benchmark ──────────────────────────────
    print(f"\n--- Original P2T on benchmark (n={n_eval}) ---")
    out1 = str(EVAL_DIR / "original_benchmark.jsonl")
    success, dt = run_inference(bench_subset, out1, ORIGINAL_BASE, gpu_id)
    if success:
        m = evaluate(out1)
        print(f"  F1={m['f1_mean']:.4f}, Coverage={m['coverage']:.4f}, Time={dt:.0f}s")
        results_table.append({"model": "Original P2T", "dataset": "benchmark", **m, "time_s": dt})

    # ── Run 2: Fine-tuned P2T on benchmark ────────────────────────────
    print(f"\n--- Fine-tuned P2T on benchmark (n={n_eval}) ---")
    out2 = str(EVAL_DIR / "finetuned_benchmark.jsonl")
    success, dt = run_inference(bench_subset, out2, FINETUNED_BASE, gpu_id)
    if success:
        m = evaluate(out2)
        print(f"  F1={m['f1_mean']:.4f}, Coverage={m['coverage']:.4f}, Time={dt:.0f}s")
        results_table.append({"model": "Fine-tuned P2T", "dataset": "benchmark", **m, "time_s": dt})

    # ── Run 3: Original P2T on bioenergy ──────────────────────────────
    print(f"\n--- Original P2T on bioenergy (n={n_eval}) ---")
    out3 = str(EVAL_DIR / "original_bioenergy.jsonl")
    success, dt = run_inference(bio_subset, out3, ORIGINAL_BASE, gpu_id)
    if success:
        m = evaluate(out3)
        print(f"  F1={m['f1_mean']:.4f}, Coverage={m['coverage']:.4f}, Time={dt:.0f}s")
        results_table.append({"model": "Original P2T", "dataset": "bioenergy", **m, "time_s": dt})

    # ── Run 4: Fine-tuned P2T on bioenergy ────────────────────────────
    print(f"\n--- Fine-tuned P2T on bioenergy (n={n_eval}) ---")
    out4 = str(EVAL_DIR / "finetuned_bioenergy.jsonl")
    success, dt = run_inference(bio_subset, out4, FINETUNED_BASE, gpu_id)
    if success:
        m = evaluate(out4)
        print(f"  F1={m['f1_mean']:.4f}, Coverage={m['coverage']:.4f}, Time={dt:.0f}s")
        results_table.append({"model": "Fine-tuned P2T", "dataset": "bioenergy", **m, "time_s": dt})

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model':<20} {'Dataset':<12} {'F1':>8} {'Coverage':>10} {'Specificity':>12}")
    print("-" * 62)
    for r in results_table:
        print(f"{r['model']:<20} {r['dataset']:<12} {r['f1_mean']:>8.4f} {r['coverage']:>10.4f} {r['specificity_mean']:>12.4f}")

    # Compute improvements
    if len(results_table) >= 2:
        orig_bench = next((r for r in results_table if r["model"] == "Original P2T" and r["dataset"] == "benchmark"), None)
        ft_bench = next((r for r in results_table if r["model"] == "Fine-tuned P2T" and r["dataset"] == "benchmark"), None)
        if orig_bench and ft_bench:
            improvement = (ft_bench["f1_mean"] - orig_bench["f1_mean"]) / max(orig_bench["f1_mean"], 0.001) * 100
            print(f"\nBenchmark F1 improvement: {improvement:+.1f}%")

    if len(results_table) >= 4:
        orig_bio = next((r for r in results_table if r["model"] == "Original P2T" and r["dataset"] == "bioenergy"), None)
        ft_bio = next((r for r in results_table if r["model"] == "Fine-tuned P2T" and r["dataset"] == "bioenergy"), None)
        if orig_bio and ft_bio:
            improvement = (ft_bio["f1_mean"] - orig_bio["f1_mean"]) / max(orig_bio["f1_mean"], 0.001) * 100
            print(f"Bioenergy F1 improvement: {improvement:+.1f}%")

    # Save results
    with open(EVAL_DIR / "comparison_results.json", 'w') as f:
        json.dump(results_table, f, indent=2)
    print(f"\nResults saved to {EVAL_DIR / 'comparison_results.json'}")


if __name__ == "__main__":
    main()
