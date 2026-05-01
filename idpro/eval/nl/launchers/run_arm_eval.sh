#!/usr/bin/env bash
# Run the full eval pipeline for one ablation arm (S0 or S1).
# Outputs go to $IDPRO_RUNS_ROOT/probe/embeddings/, then archive to embeddings_${ARM}/.
#
# Usage:
#   GPU=0 bash idpro/eval/nl/launchers/run_arm_eval.sh S0
#   GPU=0 bash idpro/eval/nl/launchers/run_arm_eval.sh S1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$REPO_ROOT/env.sh"

ARM="${1:-}"
GPU="${GPU:-0}"
if [[ "$ARM" != "S0" && "$ARM" != "S1" ]]; then
    echo "ERROR: arm must be S0 or S1" >&2; exit 1
fi

PYBIN=/data/avi/.conda/envs/protein2text_env/bin/python
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ "$ARM" == "S0" ]]; then
    CKPT="$IDPRO_RUNS_ROOT/checkpoints/esm3_S0_seqonly/stage4_step20000"
    STRUCT_FLAGS=()
else
    CKPT="$IDPRO_RUNS_ROOT/checkpoints/esm3_S1_struct/stage4_step20000"
    STRUCT_FLAGS=(--structure-track --structure-manifest "$IDPRO_RUNS_ROOT/structure_manifest.jsonl")
fi

EMB_DIR="$IDPRO_RUNS_ROOT/probe/embeddings"
ARCHIVE_DIR="$IDPRO_RUNS_ROOT/probe/embeddings_${ARM}"

# Fresh embeddings/ for this arm
rm -rf "$EMB_DIR"
mkdir -p "$EMB_DIR"

cd "$REPO_ROOT"

echo "================================================================"
echo "  EVAL arm=$ARM  CKPT=$CKPT  GPU=$GPU"
echo "  Struct: ${STRUCT_FLAGS[*]:-<none>}"
echo "================================================================"

for which in reference benchmark dark; do
    echo "--- extract: $which ---"
    "$PYBIN" idpro/eval/probes/extract_probe_embeddings.py \
        --ckpt "$CKPT" \
        --which "$which" \
        --encoder esm3-1.4b \
        "${STRUCT_FLAGS[@]}"
done

echo "--- evaluate_ec_classifier ---"
"$PYBIN" idpro/eval/probes/evaluate_ec_classifier.py

echo "--- evaluate_probe_on_dark ---"
"$PYBIN" idpro/eval/probes/evaluate_probe_on_dark.py

# Archive per-arm
echo "--- archiving to $ARCHIVE_DIR ---"
rm -rf "$ARCHIVE_DIR"
cp -r "$EMB_DIR" "$ARCHIVE_DIR"

echo "[DONE] $ARM eval pipeline. Outputs in $ARCHIVE_DIR"
