#!/usr/bin/env bash
# Launch wrapper for the 2-arm ESM3 structure ablation.
# See plans/ESM3_STRUCTURE_ABLATION_2ARM_PLAN.md.
#
# Usage:
#   bash idpro/training/launchers/run_esm3_ablation.sh S0 stage1
#   bash idpro/training/launchers/run_esm3_ablation.sh S0 stage4   # uses --resume
#   bash idpro/training/launchers/run_esm3_ablation.sh S1 stage1
#   bash idpro/training/launchers/run_esm3_ablation.sh S1 stage4   # uses --resume
#
# Single GPU:  CUDA_VISIBLE_DEVICES=2 bash ... S0 stage1
# Multi GPU:   GPUS=0,1 bash ... S0 stage1   (uses DeepSpeed ZeRO-2)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/env.sh"

ARM="${1:-}"
STAGE="${2:-}"

if [[ "$ARM" != "S0" && "$ARM" != "S1" ]]; then
    echo "ERROR: arm must be S0 or S1; got '$ARM'" >&2
    echo "Usage: $0 {S0|S1} {stage1|stage4}" >&2
    exit 1
fi
if [[ "$STAGE" != "stage1" && "$STAGE" != "stage4" ]]; then
    echo "ERROR: stage must be stage1 or stage4; got '$STAGE'" >&2
    echo "Usage: $0 {S0|S1} {stage1|stage4}" >&2
    exit 1
fi

# transformers 5.5.3 (has qwen3_5) + esm 3.2.1. This env was used to train the source-repo
# stage4_step80000 checkpoint (see saved llm_lora/config.json: transformers_version "5.5.3").
PYBIN=/data/avi/.conda/envs/protein2text_env/bin/python
# Multi-GPU launch via DeepSpeed: invoke the runner module directly (the
# `deepspeed` shell wrapper has a shebang into /home/avi which is not
# world-readable on this host).
DS_LAUNCH=("$PYBIN" -m deepspeed.launcher.runner)
# Avoid ~/.local site-packages collision (huggingface_hub is_offline_mode shadow).
export PYTHONNOUSERSITE=1
# Force unbuffered stdout/stderr so log lines flush immediately (helps when
# tailing logs of background processes through pipes/files).
export PYTHONUNBUFFERED=1

# ── Shared run-dir overrides (matched across arms) ────────────────────────
# 20% per-protein subsample (seed=42) — see plans/ESM3_STRUCTURE_ABLATION_2ARM_PLAN.md
QA_DIR_RUN="$REPO_ROOT/preliminary_data/training_data/qa_stages_struct_20pct_fullseq"
RESULTS_DIR_RUN="$IDPRO_RUNS_ROOT/training_results/esm3_ablation_$ARM"
export PYTHONHASHSEED=42
# PCI_BUS_ID ordering so CUDA_VISIBLE_DEVICES indices match nvidia-smi indices.
# Default FASTEST_FIRST puts H200 (GPU 3) and H100 NVL (GPU 2) before H100 PCIe
# (GPU 0/1) on this host, which silently steers training onto already-busy GPUs.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── Per-arm checkpoint dir ────────────────────────────────────────────────
if [[ "$ARM" == "S0" ]]; then
    CKPT_DIR_RUN="$IDPRO_RUNS_ROOT/checkpoints/esm3_S0_seqonly"
    STRUCT_FLAGS=()
else
    CKPT_DIR_RUN="$IDPRO_RUNS_ROOT/checkpoints/esm3_S1_struct"
    STRUCT_FLAGS=(--structure-track --structure-manifest "$IDPRO_RUNS_ROOT/structure_manifest.jsonl")
fi

mkdir -p "$RESULTS_DIR_RUN" "$CKPT_DIR_RUN"

# ── Repair `latest` symlink if stale (workaround for save_ckpt/DS race) ────
# Older save_ckpt() left `latest` pointing at the FIRST checkpoint while
# subsequent saves silently failed. Repoint to the highest-step checkpoint
# of the appropriate stage before launching, so --resume picks the right one.
prev_stage="stage1"
if [[ "$STAGE" == "stage4" ]]; then
    # Stage 4 resumes from the LATEST available checkpoint of EITHER stage —
    # prefer stage4 (its own resume) but fall back to stage1's last step.
    # `|| true` to swallow rc=2 when the glob has no matches (set -o
    # pipefail would otherwise abort the script via set -e on first stage 4).
    latest_step4=$( { ls -1d "$CKPT_DIR_RUN"/stage4_step* 2>/dev/null \
        || true; } | sed 's|.*/stage4_step||' | sort -n | tail -1)
    if [[ -n "$latest_step4" ]]; then
        prev_stage="stage4"
    fi
fi
latest_step=$( { ls -1d "$CKPT_DIR_RUN"/${prev_stage}_step* 2>/dev/null \
    || true; } | sed 's|.*/'"${prev_stage}"'_step||' | sort -n | tail -1)
if [[ -n "$latest_step" ]]; then
    target="${prev_stage}_step${latest_step}"
    current=""
    if [[ -L "$CKPT_DIR_RUN/latest" ]]; then
        current=$(readlink "$CKPT_DIR_RUN/latest")
    fi
    if [[ "$current" != "$target" ]]; then
        echo "[fix_latest] $CKPT_DIR_RUN/latest: '$current' → '$target'"
        rm -f "$CKPT_DIR_RUN/latest"
        (cd "$CKPT_DIR_RUN" && ln -s "$target" latest)
    fi
fi

# ── Stage flags ───────────────────────────────────────────────────────────
# Reduced max_steps for the 20% subset: 10K stage1, 20K stage4 (~17h/arm total)
COMMON_FLAGS=(
    --encoder esm3-1.4b
    --llm qwen3.5-27b
    --qa-dir "$QA_DIR_RUN"
    --ckpt-dir "$CKPT_DIR_RUN"
    --results-dir "$RESULTS_DIR_RUN"
)
if [[ "$STAGE" == "stage1" ]]; then
    # --resume is a no-op on a fresh checkpoint dir (load_ckpt returns 0,0,[]),
    # but lets us recover from a mid-run crash by picking up the latest stage1_step* dir.
    STAGE_FLAGS=(--stage 1 --max-steps 10000 --max-hours 12 --resume)
else
    STAGE_FLAGS=(--stage 4 --max-steps 20000 --max-hours 22 --resume)
fi

cd "$REPO_ROOT"

# Multi-GPU vs single-GPU launch. GPUS overrides CUDA_VISIBLE_DEVICES and
# triggers DeepSpeed ZeRO-2 when it contains >1 device.
GPUS_ARG="${GPUS:-$CUDA_VISIBLE_DEVICES}"
if [[ "$GPUS_ARG" == *,* ]]; then
    echo "================================================================"
    echo "  ESM3 ablation arm: $ARM   stage: $STAGE   [DeepSpeed ZeRO-2]"
    echo "  CKPT_DIR:    $CKPT_DIR_RUN"
    echo "  RESULTS_DIR: $RESULTS_DIR_RUN"
    echo "  GPUs:        $GPUS_ARG"
    echo "  Struct flags: ${STRUCT_FLAGS[*]:-<none>}"
    echo "================================================================"
    # The DeepSpeed launcher itself sets CUDA_VISIBLE_DEVICES from --include,
    # so we don't pre-export it here.
    #
    # Master port: derived from the first GPU index so that two arms running
    # in parallel (e.g. GPUS=0,1 and GPUS=2,3) don't collide on the default
    # port 29500. Override with MASTER_PORT=<n> if needed.
    first_gpu="${GPUS_ARG%%,*}"
    master_port="${MASTER_PORT:-$((29500 + first_gpu * 100))}"
    echo "  master_port: $master_port"
    exec "${DS_LAUNCH[@]}" \
        --include "localhost:$GPUS_ARG" \
        --master_port "$master_port" \
        idpro/training/train.py \
            "${COMMON_FLAGS[@]}" \
            "${STAGE_FLAGS[@]}" \
            "${STRUCT_FLAGS[@]}" \
            --deepspeed idpro/training/configs/ds_zero2.json
else
    export CUDA_VISIBLE_DEVICES="$GPUS_ARG"
    echo "================================================================"
    echo "  ESM3 ablation arm: $ARM   stage: $STAGE   [single-GPU]"
    echo "  CKPT_DIR:    $CKPT_DIR_RUN"
    echo "  RESULTS_DIR: $RESULTS_DIR_RUN"
    echo "  CUDA dev:    $CUDA_VISIBLE_DEVICES"
    echo "  Struct flags: ${STRUCT_FLAGS[*]:-<none>}"
    echo "================================================================"
    exec "$PYBIN" idpro/training/train.py \
        "${COMMON_FLAGS[@]}" \
        "${STAGE_FLAGS[@]}" \
        "${STRUCT_FLAGS[@]}"
fi
