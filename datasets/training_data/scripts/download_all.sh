#!/bin/bash
# IDPro Training Data Download — Master Script
# Run all downloads in sequence. Each script is independent and resumable.
#
# Usage: bash scripts/download_all.sh
# Or run individual scripts: python scripts/download_uniprot.py
#
# Outputs land under <IDPRO_DATA_ROOT or repo/datasets>/training_data/.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DATA_DIR="${IDPRO_DATA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}/training_data"
mkdir -p "$TRAINING_DATA_DIR"
cd "$TRAINING_DATA_DIR"

echo "=========================================="
echo "IDPro Training Data Download"
echo "  output root: $TRAINING_DATA_DIR"
echo "=========================================="

echo ""
echo "[P1] Downloading UniProt bacterial protein features..."
python "$SCRIPT_DIR/download_uniprot.py" --taxonomy bacteria --output downloads/uniprot_bacteria_features/
python "$SCRIPT_DIR/download_uniprot.py" --taxonomy archaea --output downloads/uniprot_bacteria_features/

echo ""
echo "[P2] Downloading InterPro domain descriptions..."
python "$SCRIPT_DIR/download_interpro.py" --output downloads/interpro/

echo ""
echo "[P3] Downloading PROSITE motif patterns..."
python "$SCRIPT_DIR/download_prosite.py" --output downloads/prosite/

echo ""
echo "[P4] Downloading SIFTS + CATH structure mappings..."
python "$SCRIPT_DIR/download_structure.py" --output downloads/structure/

echo ""
echo "[P5] Downloading M-CSA catalytic mechanisms..."
python "$SCRIPT_DIR/download_mcsa.py" --output downloads/mcsa/

echo ""
echo "Building structured annotation records..."
python "$SCRIPT_DIR/build_records.py"

echo ""
echo "Generating QA data for all stages..."
python "$SCRIPT_DIR/generate_qa.py"

echo ""
echo "=========================================="
echo "DOWNLOAD COMPLETE"
echo "=========================================="
