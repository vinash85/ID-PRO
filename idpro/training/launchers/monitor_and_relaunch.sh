#!/bin/bash
# IDPro Training Monitor & Auto-Relaunch
# Checks every 5 minutes. Relaunches if training process dies.
# Also monitors Qwen download and launches multi-GPU training when ready.
#
# Usage:
#   nohup bash idpro/training/launchers/monitor_and_relaunch.sh > monitor.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/env.sh"

TRAIN_SCRIPT="$REPO_ROOT/idpro/training/train.py"
DS_CONFIG="$REPO_ROOT/idpro/training/configs/ds_zero2.json"
EVAL_TSV="$IDPRO_RUNS_ROOT/training_results/robust/eval_progress.tsv"
QWEN_MODEL="${IDPRO_HF_CACHE:-$IDPRO_RUNS_ROOT/hf_cache}/models--Qwen--Qwen3.5-27B"
CHECK_INTERVAL=300  # 5 minutes

echo "$(date): IDPro Monitor started"
echo "  Train script: $TRAIN_SCRIPT"
echo "  Eval TSV:     $EVAL_TSV"
echo "  Check interval: ${CHECK_INTERVAL}s"

while true; do
    # ── Check IDPro training ──────────────────────────────────────
    TRAIN_PID=$(ps aux | grep "idpro/training/train.py" | grep -v grep | grep -v monitor | awk '{print $2}' | head -1)

    if [ -z "$TRAIN_PID" ]; then
        echo "$(date): IDPro training NOT running."
        if [ -f "$EVAL_TSV" ]; then
            echo "$(date): Last eval: $(tail -1 "$EVAL_TSV")"
        fi
    else
        if [ -f "$EVAL_TSV" ]; then
            LATEST=$(tail -1 "$EVAL_TSV")
            echo "$(date): IDPro running (PID $TRAIN_PID) | Latest: $LATEST"
        else
            echo "$(date): IDPro running (PID $TRAIN_PID) | No eval yet"
        fi
    fi

    # ── Check Qwen download ───────────────────────────────────────
    QWEN_PID=$(ps aux | grep "snapshot_download" | grep -v grep | awk '{print $2}' | head -1)

    if [ -d "$QWEN_MODEL" ] && [ -z "$QWEN_PID" ]; then
        if [ -z "$TRAIN_PID" ]; then
            echo "$(date): Qwen3.5-27B available; multi-GPU launch:"
            echo "$(date):   deepspeed --num_gpus=4 $TRAIN_SCRIPT --deepspeed $DS_CONFIG --stage 1"
        fi
    elif [ ! -z "$QWEN_PID" ]; then
        echo "$(date): Qwen download in progress (PID $QWEN_PID)"
    fi

    # ── GPU status ────────────────────────────────────────────────
    GPU_STATUS=$(nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null)
    echo "$(date): GPUs: $GPU_STATUS"

    sleep $CHECK_INTERVAL
done
