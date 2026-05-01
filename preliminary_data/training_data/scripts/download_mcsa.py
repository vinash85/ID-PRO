#!/usr/bin/env python3
"""
Download M-CSA (Mechanism and Catalytic Site Atlas) catalytic mechanisms.

Usage: python download_mcsa.py --output downloads/mcsa/
"""

import json
import time
import argparse
import requests
from pathlib import Path

MCSA_API = "https://www.ebi.ac.uk/thornton-srv/m-csa/api"


def download_mcsa(output_dir):
    """Download all M-CSA entries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "mcsa_entries.jsonl"

    # Get all entries via paginated API
    print("  Fetching M-CSA entry list...")
    entries_list = []
    url = f"{MCSA_API}/entries/?format=json"

    while url:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            entries_list.extend(results)
            url = data.get("next")
            print(f"    Fetched {len(entries_list)} entries so far...")
            time.sleep(0.3)
        except Exception as e:
            print(f"  M-CSA API failed on page: {e}")
            break

    if not entries_list:
        print("  No M-CSA entries found. Skipping.")
        return

    print(f"  Found {len(entries_list)} M-CSA entries total")

    # Each entry from the list already has useful fields
    records = []
    for i, entry in enumerate(entries_list):
        record = {
            "mcsa_id": entry.get("mcsa_id"),
            "enzyme_name": entry.get("enzyme_name", ""),
            "uniprot_id": entry.get("reference_uniprot_id", ""),
            "description": entry.get("description", ""),
            "url": entry.get("url", ""),
        }
        records.append(record)

    # Save
    with open(output_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = {"total_entries": len(records), "file": str(output_file)}
    with open(output_dir / "mcsa_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  M-CSA complete: {len(records)} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="downloads/mcsa/")
    args = parser.parse_args()

    download_mcsa(args.output)
