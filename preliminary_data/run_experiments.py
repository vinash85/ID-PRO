#!/usr/bin/env python3
"""
P2T Preliminary Data Experiments — Multi-GPU Runner
Runs Protein2Text inference on benchmark and dark genome proteins.
Supports parallel execution across GPUs.
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = Path("/data/asahu/projects/doe_genesis")
P2T_DIR = BASE_DIR / "Protein2Text"
PRELIM_DIR = BASE_DIR / "preliminary_data"
MODEL_PATH = str(P2T_DIR / "checkpoints/protein2text-llama3.1-8B-instruct-esm2-650M")
MODEL_BASE = "/data/ajararweh/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"
CONDA_ENV = "protein2text_env"


def run_p2t_inference(input_file, output_file, gpu_id, temperature=0.7, max_new_tokens=512, top_p=0.9, num_beams=1):
    """Run P2T inference on a single input file using specified GPU."""
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} conda run -n {CONDA_ENV} "
        f"python {P2T_DIR}/evaluation/inference_model.py "
        f"--input_file {input_file} "
        f"--output_file {output_file} "
        f"--model_path {MODEL_PATH} "
        f"--model_base {MODEL_BASE} "
        f"--device cuda "
        f"--temperature {temperature} "
        f"--max_new_tokens {max_new_tokens} "
        f"--top_p {top_p} "
        f"--num_beams {num_beams}"
    )
    print(f"[GPU {gpu_id}] Starting: {Path(input_file).name} -> {Path(output_file).name}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"[GPU {gpu_id}] FAILED after {dt:.0f}s: {result.stderr[-500:]}")
        return False, dt
    print(f"[GPU {gpu_id}] DONE in {dt:.0f}s: {Path(output_file).name}")
    return True, dt


def split_json_file(input_file, num_chunks, output_dir):
    """Split a JSON input file into chunks for parallel processing."""
    with open(input_file) as f:
        data = json.load(f)

    chunk_size = max(1, len(data) // num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(data)
        if start >= len(data):
            break
        chunk = data[start:end]
        chunk_file = output_dir / f"chunk_{i}.json"
        with open(chunk_file, 'w') as f:
            json.dump(chunk, f, indent=2)
        chunks.append(str(chunk_file))

    return chunks


def merge_results(chunk_outputs, final_output):
    """Merge JSONL result files."""
    with open(final_output, 'w') as out:
        for chunk_file in sorted(chunk_outputs):
            if os.path.exists(chunk_file):
                with open(chunk_file) as f:
                    for line in f:
                        out.write(line)


def run_benchmark(gpus=[0, 1]):
    """Run P2T benchmark on microbiome proteins using multiple GPUs."""
    input_file = PRELIM_DIR / "benchmark/microbiome_benchmark.json"
    if not input_file.exists():
        print(f"Benchmark input not found: {input_file}")
        return

    output_dir = PRELIM_DIR / "benchmark/results"
    output_dir.mkdir(exist_ok=True)
    chunks_dir = PRELIM_DIR / "benchmark/chunks"
    chunks_dir.mkdir(exist_ok=True)

    # Split data across GPUs
    chunk_files = split_json_file(input_file, len(gpus), chunks_dir)

    # Run in parallel
    futures = {}
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        for i, (chunk_file, gpu_id) in enumerate(zip(chunk_files, gpus)):
            output_file = str(output_dir / f"benchmark_results_gpu{gpu_id}.jsonl")
            future = executor.submit(
                run_p2t_inference, chunk_file, output_file, gpu_id,
                temperature=0.3, max_new_tokens=512
            )
            futures[future] = (gpu_id, output_file)

        for future in as_completed(futures):
            gpu_id, output_file = futures[future]
            success, dt = future.result()
            print(f"GPU {gpu_id}: {'SUCCESS' if success else 'FAILED'} ({dt:.0f}s)")

    # Merge results
    chunk_outputs = [str(output_dir / f"benchmark_results_gpu{g}.jsonl") for g in gpus]
    final_output = str(output_dir / "benchmark_results_all.jsonl")
    merge_results(chunk_outputs, final_output)
    print(f"Merged results: {final_output}")


def run_dark_genome(gpus=[0, 1]):
    """Run P2T on dark genome proteins using multiple GPUs."""
    input_file = PRELIM_DIR / "dark_genome/dark_genome_proteins.json"
    if not input_file.exists():
        print(f"Dark genome input not found: {input_file}")
        return

    output_dir = PRELIM_DIR / "dark_genome/results"
    output_dir.mkdir(exist_ok=True)
    chunks_dir = PRELIM_DIR / "dark_genome/chunks"
    chunks_dir.mkdir(exist_ok=True)

    chunk_files = split_json_file(input_file, len(gpus), chunks_dir)

    futures = {}
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        for i, (chunk_file, gpu_id) in enumerate(zip(chunk_files, gpus)):
            output_file = str(output_dir / f"dark_genome_results_gpu{gpu_id}.jsonl")
            future = executor.submit(
                run_p2t_inference, chunk_file, output_file, gpu_id,
                temperature=0.5, max_new_tokens=512
            )
            futures[future] = (gpu_id, output_file)

        for future in as_completed(futures):
            gpu_id, output_file = futures[future]
            success, dt = future.result()
            print(f"GPU {gpu_id}: {'SUCCESS' if success else 'FAILED'} ({dt:.0f}s)")

    chunk_outputs = [str(output_dir / f"dark_genome_results_gpu{g}.jsonl") for g in gpus]
    final_output = str(output_dir / "dark_genome_results_all.jsonl")
    merge_results(chunk_outputs, final_output)
    print(f"Merged results: {final_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["benchmark", "dark_genome", "both"], default="both")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU IDs")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    if args.task in ("benchmark", "both"):
        print("=" * 60)
        print("RUNNING BENCHMARK ON MICROBIOME PROTEINS")
        print("=" * 60)
        run_benchmark(gpus)

    if args.task in ("dark_genome", "both"):
        print("=" * 60)
        print("RUNNING DARK GENOME ANNOTATION")
        print("=" * 60)
        run_dark_genome(gpus)
