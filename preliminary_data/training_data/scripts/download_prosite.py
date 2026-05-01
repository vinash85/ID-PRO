#!/usr/bin/env python3
"""
Download PROSITE motif patterns and descriptions.

Usage: python download_prosite.py --output downloads/prosite/
"""

import os
import json
import re
import argparse
import requests
from pathlib import Path


def download_prosite_dat(output_dir):
    """Download prosite.dat file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dat_file = output_dir / "prosite.dat"
    if dat_file.exists():
        print(f"  prosite.dat already exists ({dat_file.stat().st_size / 1e6:.1f} MB)")
        return str(dat_file)

    url = "https://ftp.expasy.org/databases/prosite/prosite.dat"
    print(f"  Downloading prosite.dat...")

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with open(dat_file, "w") as f:
        f.write(resp.text)

    print(f"  Downloaded: {dat_file.stat().st_size / 1e6:.1f} MB")
    return str(dat_file)


def parse_prosite_dat(dat_file, output_dir):
    """Parse prosite.dat into JSON records."""
    output_dir = Path(output_dir)
    entries = []
    current = {}

    with open(dat_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("ID"):
                current = {"id": line[5:].strip().rstrip(";")}
            elif line.startswith("AC"):
                current["accession"] = line[5:].strip().rstrip(";")
            elif line.startswith("DT"):
                pass
            elif line.startswith("DE"):
                current.setdefault("description", "")
                current["description"] += line[5:].strip() + " "
            elif line.startswith("PA"):
                current.setdefault("pattern", "")
                current["pattern"] += line[5:].strip()
            elif line.startswith("CC"):
                text = line[5:].strip()
                if text.startswith("/"):
                    pass  # metadata
                else:
                    current.setdefault("comment", "")
                    current["comment"] += text + " "
            elif line.startswith("DR"):
                # Cross-references to UniProt
                refs = line[5:].strip().rstrip(";").split(";")
                current.setdefault("uniprot_refs", [])
                for ref in refs:
                    ref = ref.strip()
                    if ref:
                        parts = ref.split(",")
                        if len(parts) >= 1:
                            current["uniprot_refs"].append(parts[0].strip())
            elif line.startswith("//"):
                if current.get("accession"):
                    current["description"] = current.get("description", "").strip()
                    current["comment"] = current.get("comment", "").strip()
                    current["type"] = "PATTERN" if current.get("pattern") else "PROFILE"
                    entries.append(current)
                current = {}

    # Save as JSONL
    output_file = output_dir / "prosite_entries.jsonl"
    with open(output_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Summary
    patterns = sum(1 for e in entries if e.get("pattern"))
    profiles = sum(1 for e in entries if not e.get("pattern"))
    with_desc = sum(1 for e in entries if e.get("description"))

    summary = {
        "total_entries": len(entries),
        "patterns": patterns,
        "profiles": profiles,
        "with_description": with_desc,
        "file": str(output_file),
    }
    with open(output_dir / "prosite_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Parsed: {len(entries)} entries ({patterns} patterns, {profiles} profiles)")
    print(f"  {with_desc} entries have descriptions")
    return entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="downloads/prosite/")
    args = parser.parse_args()

    dat_file = download_prosite_dat(args.output)
    parse_prosite_dat(dat_file, args.output)
