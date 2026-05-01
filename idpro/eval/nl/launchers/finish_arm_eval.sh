#!/usr/bin/env bash
# Resume an arm's eval pipeline AFTER extract has produced reference/benchmark/dark .pt
# files. Skips the wipe + extract phase. Runs only:
#   evaluate_ec_classifier.py
#   evaluate_probe_on_dark.py
# then copies embeddings/ → embeddings_${ARM}/.
#
# Usage:
#   GPU=0 bash idpro/eval/nl/launchers/finish_arm_eval.sh S0
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

EMB_DIR="$IDPRO_RUNS_ROOT/probe/embeddings"
ARCHIVE_DIR="$IDPRO_RUNS_ROOT/probe/embeddings_${ARM}"

# Sanity check
for f in reference_embeddings.pt benchmark_embeddings.pt dark_embeddings.pt; do
    [[ -f "$EMB_DIR/$f" ]] || { echo "ERROR: $EMB_DIR/$f missing — run extract first" >&2; exit 1; }
done

cd "$REPO_ROOT"

echo "================================================================"
echo "  FINISH EVAL arm=$ARM  GPU=$GPU"
echo "================================================================"

echo "--- evaluate_ec_classifier ---"
"$PYBIN" idpro/eval/probes/evaluate_ec_classifier.py

echo "--- evaluate_probe_on_dark ---"
"$PYBIN" idpro/eval/probes/evaluate_probe_on_dark.py

echo "--- archive → $ARCHIVE_DIR ---"
rm -rf "$ARCHIVE_DIR"
cp -r "$EMB_DIR" "$ARCHIVE_DIR"

echo "[DONE] $ARM eval pipeline. Outputs in $ARCHIVE_DIR"
