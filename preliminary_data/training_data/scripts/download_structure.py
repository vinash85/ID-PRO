#!/usr/bin/env python3
"""
Download SIFTS (UniProt-PDB mapping) and CATH fold classifications.

Usage: python download_structure.py --output downloads/structure/
"""

import os
import json
import gzip
import argparse
import requests
from pathlib import Path


def download_sifts(output_dir):
    """Download SIFTS UniProt-PDB mapping."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gz_file = output_dir / "uniprot_pdb.tsv.gz"
    tsv_file = output_dir / "uniprot_pdb.tsv"

    if tsv_file.exists():
        print(f"  SIFTS already downloaded ({tsv_file.stat().st_size / 1e6:.1f} MB)")
        return str(tsv_file)

    url = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/uniprot_pdb.tsv.gz"
    print(f"  Downloading SIFTS mapping...")

    resp = requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()

    with open(gz_file, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024*1024):
            f.write(chunk)

    # Decompress
    with gzip.open(gz_file, 'rt') as gz, open(tsv_file, 'w') as out:
        out.write(gz.read())

    os.remove(gz_file)
    print(f"  SIFTS downloaded: {tsv_file.stat().st_size / 1e6:.1f} MB")
    return str(tsv_file)


def download_cath_list(output_dir):
    """Download CATH domain list with fold classifications."""
    output_dir = Path(output_dir)

    cath_file = output_dir / "cath_domain_list.txt"
    if cath_file.exists():
        print(f"  CATH already downloaded")
        return str(cath_file)

    url = "https://www.cathdb.info/version/latest/api/rest/id/all"
    print(f"  Downloading CATH domain list...")

    try:
        resp = requests.get(url, timeout=120, headers={"Accept": "text/plain"})
        resp.raise_for_status()
        with open(cath_file, "w") as f:
            f.write(resp.text)
        print(f"  CATH downloaded: {cath_file.stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        print(f"  CATH download failed: {e}")
        print(f"  Will use SIFTS PDB mapping + PDB secondary structure instead")

    return str(cath_file)


def parse_sifts(tsv_file, output_dir):
    """Parse SIFTS into a UniProt→PDB lookup."""
    output_dir = Path(output_dir)
    mapping = {}

    with open(tsv_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            # Header line: "SP_PRIMARY\tPDB"
            if line.startswith("SP_PRIMARY"):
                continue
            # Format: UniProt_ID\tpdb1;pdb2;pdb3
            parts = line.split("\t")
            if len(parts) >= 2:
                uniprot_id = parts[0].strip()
                pdb_ids = [p.strip() for p in parts[1].split(";") if p.strip()]
                if uniprot_id and pdb_ids:
                    mapping[uniprot_id] = [{"pdb": pdb_id} for pdb_id in pdb_ids]

    # Save as JSON
    output_file = output_dir / "uniprot_pdb_mapping.json"
    with open(output_file, "w") as f:
        json.dump(mapping, f)

    print(f"  Parsed SIFTS: {len(mapping)} UniProt entries with PDB structures")

    summary = {"total_uniprot_with_pdb": len(mapping), "file": str(output_file)}
    with open(output_dir / "sifts_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="downloads/structure/")
    args = parser.parse_args()

    tsv_file = download_sifts(args.output)
    parse_sifts(tsv_file, args.output)
    download_cath_list(args.output)
