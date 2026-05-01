#!/usr/bin/env python3
"""
Download InterPro entry descriptions via FTP bulk files.

FTP source: https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/
Files:
  - entry.list       (2.7 MB)  — all entry IDs, types, names
  - interpro.xml.gz  (39 MB)   — full entry descriptions/abstracts
  - interpro2go      (2.9 MB)  — InterPro → GO mappings

Usage: python download_interpro.py --output downloads/interpro/
"""

import json
import gzip
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlretrieve

FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release"


def download_file(url, dest):
    """Download a file with progress."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  Already exists: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"  Downloading {url} ...")

    def report(block_num, block_size, total_size):
        if total_size > 0 and block_num % 200 == 0:
            pct = block_num * block_size / total_size * 100
            print(f"    {block_num * block_size / 1e6:.0f}/{total_size / 1e6:.0f} MB ({pct:.0f}%)")

    urlretrieve(url, str(dest), reporthook=report)
    print(f"  Downloaded: {dest.stat().st_size / 1e6:.1f} MB")


def parse_entry_list(filepath):
    """Parse entry.list TSV → dict of {accession: {type, name}}."""
    entries = {}
    with open(filepath) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                entries[parts[0]] = {"type": parts[1], "name": parts[2]}
    print(f"  entry.list: {len(entries)} entries")
    return entries


def parse_interpro2go(filepath):
    """Parse interpro2go → dict of {accession: [go_terms]}."""
    go_map = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith("!") or not line.strip():
                continue
            # Format: InterPro:IPR000003 Name > GO:term ; GO:0003677
            m = re.match(r'InterPro:(IPR\d+)\s+.+>\s+GO:(.+)\s+;\s+(GO:\d+)', line)
            if m:
                ipr_id, go_name, go_id = m.group(1), m.group(2).strip(), m.group(3)
                go_map.setdefault(ipr_id, []).append({"id": go_id, "name": go_name})
    print(f"  interpro2go: {len(go_map)} entries with GO terms")
    return go_map


def parse_interpro_xml(xml_gz_path, entry_list, go_map, output_file):
    """Stream-parse interpro.xml.gz to extract descriptions and member databases."""
    print(f"  Parsing {xml_gz_path} ...")

    total = 0
    with_desc = 0

    with gzip.open(xml_gz_path, "rb") as gz_f:
        with open(output_file, "w") as out_f:
            context = ET.iterparse(gz_f, events=("end",))

            for event, elem in context:
                if elem.tag == "interpro":
                    ipr_id = elem.get("id", "")
                    if not ipr_id:
                        elem.clear()
                        continue

                    total += 1

                    record = {
                        "accession": ipr_id,
                        "short_name": elem.get("short_name", ""),
                        "type": elem.get("type", ""),
                        "protein_count": int(elem.get("protein_count", "0")),
                    }

                    # Name
                    name_el = elem.find("name")
                    if name_el is not None and name_el.text:
                        record["name"] = name_el.text

                    # Abstract/description — collect all text content
                    abstract_el = elem.find("abstract")
                    if abstract_el is not None:
                        # Collect text from all <p> elements, stripping citations
                        desc_parts = []
                        for p_el in abstract_el.findall("p"):
                            text = ET.tostring(p_el, encoding="unicode", method="text")
                            text = re.sub(r'\s+', ' ', text).strip()
                            # Remove citation artifacts like ", , "
                            text = re.sub(r'\s*,\s*,\s*', ', ', text)
                            text = re.sub(r'\[\s*,?\s*\]', '', text)
                            text = re.sub(r'\s+', ' ', text).strip()
                            if text:
                                desc_parts.append(text)
                        if desc_parts:
                            record["description"] = " ".join(desc_parts)
                            with_desc += 1

                    # Member databases (Pfam, PROSITE, CDD, etc.)
                    member_dbs = {}
                    member_list = elem.find("member_list")
                    if member_list is not None:
                        for db_xref in member_list.findall("db_xref"):
                            db = db_xref.get("db", "")
                            dbkey = db_xref.get("dbkey", "")
                            name = db_xref.get("name", "")
                            member_dbs.setdefault(db, []).append({"id": dbkey, "name": name})
                    if member_dbs:
                        record["member_databases"] = member_dbs

                    # GO terms from interpro2go
                    if ipr_id in go_map:
                        record["go_terms"] = go_map[ipr_id]

                    # Extra info from entry.list
                    if ipr_id in entry_list:
                        if "name" not in record:
                            record["name"] = entry_list[ipr_id]["name"]

                    out_f.write(json.dumps(record) + "\n")

                    if total % 10000 == 0:
                        print(f"    Processed {total} entries...")

                    elem.clear()

    print(f"  Parsed: {total} entries, {with_desc} with descriptions")
    return total


def main(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download bulk files
    print("Downloading InterPro bulk files...")
    entry_list_file = output_dir / "entry.list"
    xml_gz_file = output_dir / "interpro.xml.gz"
    go_file = output_dir / "interpro2go"

    download_file(f"{FTP_BASE}/entry.list", entry_list_file)
    download_file(f"{FTP_BASE}/interpro.xml.gz", xml_gz_file)
    download_file(f"{FTP_BASE}/interpro2go", go_file)

    # Parse
    print("\nParsing entry list...")
    entry_list = parse_entry_list(entry_list_file)

    print("Parsing GO mappings...")
    go_map = parse_interpro2go(go_file)

    print("Parsing InterPro XML (descriptions + member databases)...")
    output_jsonl = output_dir / "interpro_entries.jsonl"
    total = parse_interpro_xml(xml_gz_file, entry_list, go_map, output_jsonl)

    # Summary
    summary = {
        "total_entries": total,
        "source_files": ["entry.list", "interpro.xml.gz", "interpro2go"],
        "output": str(output_jsonl),
    }
    with open(output_dir / "interpro_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: {total} InterPro entries → {output_jsonl}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="downloads/interpro/")
    args = parser.parse_args()

    main(args.output)
