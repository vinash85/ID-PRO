#!/bin/bash
# IDPro Training Data Download — Master Script
# Run all downloads in sequence. Each script is independent and resumable.
#
# Usage: bash scripts/download_all.sh
# Or run individual scripts: python scripts/download_uniprot.py

set -e
cd /data/asahu/projects/doe_genesis/preliminary_data/training_data

echo "=========================================="
echo "IDPro Training Data Download"
echo "=========================================="

echo ""
echo "[P1] Downloading UniProt bacterial protein features..."
python scripts/download_uniprot.py --taxonomy bacteria --output downloads/uniprot_bacteria_features/
python scripts/download_uniprot.py --taxonomy archaea --output downloads/uniprot_bacteria_features/

echo ""
echo "[P2] Downloading InterPro domain descriptions..."
python scripts/download_interpro.py --output downloads/interpro/

echo ""
echo "[P3] Downloading PROSITE motif patterns..."
python scripts/download_prosite.py --output downloads/prosite/

echo ""
echo "[P4] Downloading SIFTS + CATH structure mappings..."
python scripts/download_structure.py --output downloads/structure/

echo ""
echo "[P5] Downloading M-CSA catalytic mechanisms..."
python scripts/download_mcsa.py --output downloads/mcsa/

echo ""
echo "Building structured annotation records..."
python scripts/build_records.py

echo ""
echo "Generating QA data for all stages..."
python scripts/generate_qa.py

echo ""
echo "=========================================="
echo "DOWNLOAD COMPLETE"
echo "=========================================="
