#!/usr/bin/env bash
# Run stage1 → stage4 sequentially for one arm on a fixed GPU pair.
# Logs everything to $IDPRO_RUNS_ROOT/training_results/launch_logs/${ARM}_full.log.
#
# Usage:
#   bash idpro/training/launchers/run_esm3_arm_chain.sh S0 0,1
#   bash idpro/training/launchers/run_esm3_arm_chain.sh S1 2,3

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/env.sh"

ARM="${1:-}"
GPUS="${2:-}"

if [[ "$ARM" != "S0" && "$ARM" != "S1" ]]; then
    echo "ERROR: arm must be S0 or S1; got '$ARM'" >&2
    echo "Usage: $0 {S0|S1} <GPUS>   e.g.  $0 S0 0,1" >&2
    exit 1
fi
if [[ -z "$GPUS" ]]; then
    echo "ERROR: GPUS required (comma-separated, e.g. 0,1)" >&2
    exit 1
fi

LOGDIR="$IDPRO_RUNS_ROOT/training_results/launch_logs"
LOG="$LOGDIR/${ARM}_full.log"
mkdir -p "$LOGDIR"

run_stage() {
    local stage="$1"
    echo "[$(date)] === ${ARM} ${stage} start (GPUs $GPUS) ===" >> "$LOG"
    GPUS="$GPUS" bash "$SCRIPT_DIR/run_esm3_ablation.sh" "$ARM" "$stage" >> "$LOG" 2>&1
    local rc=$?
    echo "[$(date)] === ${ARM} ${stage} end rc=$rc ===" >> "$LOG"
    return $rc
}

fix_latest_symlink() {
    # Workaround: an older save_ckpt() in train.py raced with DeepSpeed's
    # internal `latest` file, leaving the symlink stuck at stage1_step2000.
    # Rebuild it to point at the highest stage1_step* directory before stage 4
    # so --resume picks up the correct checkpoint.
    local subdir
    if [[ "$ARM" == "S0" ]]; then
        subdir="esm3_S0_seqonly"
    else
        subdir="esm3_S1_struct"
    fi
    local ckpt_dir="$IDPRO_RUNS_ROOT/checkpoints/$subdir"
    if [[ ! -d "$ckpt_dir" ]]; then
        echo "[$(date)] fix_latest: ckpt_dir missing: $ckpt_dir" >> "$LOG"
        return
    fi
    local latest_step
    latest_step=$(ls -1d "$ckpt_dir"/stage1_step* 2>/dev/null \
        | sed 's|.*/stage1_step||' | sort -n | tail -1)
    if [[ -z "$latest_step" ]]; then
        echo "[$(date)] fix_latest: no stage1_step* dirs in $ckpt_dir" >> "$LOG"
        return
    fi
    local target="stage1_step${latest_step}"
    rm -f "$ckpt_dir/latest"
    (cd "$ckpt_dir" && ln -s "$target" latest)
    echo "[$(date)] fix_latest: $ckpt_dir/latest -> $target" >> "$LOG"
}

echo "[$(date)] ===== ${ARM} chain start (GPUs $GPUS) =====" >> "$LOG"
run_stage stage1
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "[$(date)] !!! stage1 failed rc=$rc — aborting chain" >> "$LOG"
    exit $rc
fi
fix_latest_symlink
run_stage stage4
rc=$?
echo "[$(date)] ===== ${ARM} chain end rc=$rc =====" >> "$LOG"
exit $rc
