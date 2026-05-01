#!/usr/bin/env python3
"""
Download UniProt reviewed bacterial/archaeal proteins via FTP bulk XML files,
then parse all feature annotations into JSONL.

FTP source: https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/
Files:
  - uniprot_sprot_bacteria.xml.gz  (~342 MB compressed, ~337K proteins)
  - uniprot_sprot_archaea.xml.gz   (~19 MB compressed, ~20K proteins)

Usage:
  python download_uniprot.py --taxonomy both --output downloads/uniprot_bacteria_features/
"""

import os
import json
import gzip
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlretrieve

FTP_BASE = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions"
NS = "{https://uniprot.org/uniprot}"

FILES = {
    "bacteria": "uniprot_sprot_bacteria.xml.gz",
    "archaea": "uniprot_sprot_archaea.xml.gz",
}


def download_file(url, dest):
    """Download a file with progress reporting."""
    if dest.exists():
        print(f"  Already downloaded: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"  Downloading {url} ...")

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded / total_size * 100
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            if block_num % 500 == 0:
                print(f"    {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)")

    urlretrieve(url, str(dest), reporthook=report)
    print(f"  Downloaded: {dest.stat().st_size / 1e6:.1f} MB")


def extract_features(entry):
    """Extract all feature annotations from a UniProt XML <entry> element."""
    record = {}

    # Accession
    acc_el = entry.find(f"{NS}accession")
    if acc_el is None:
        return None
    record["accession"] = acc_el.text

    # Sequence
    seq_el = entry.find(f"{NS}sequence")
    if seq_el is not None and seq_el.text:
        record["sequence"] = seq_el.text.replace("\n", "").replace(" ", "")
        record["length"] = len(record["sequence"])
    else:
        return None

    # Skip short sequences
    if record["length"] < 50:
        return None

    # Protein name
    protein_el = entry.find(f"{NS}protein")
    if protein_el is not None:
        rec_name = protein_el.find(f"{NS}recommendedName")
        if rec_name is None:
            rec_name = protein_el.find(f"{NS}submittedName")
        if rec_name is not None:
            full_name = rec_name.find(f"{NS}fullName")
            if full_name is not None:
                record["protein_name"] = full_name.text

    # Features (domains, motifs, active sites, binding, etc.)
    features = []
    for feat in entry.findall(f"{NS}feature"):
        f = {"type": feat.get("type", ""), "description": feat.get("description", "")}

        # Location
        loc = feat.find(f"{NS}location")
        if loc is not None:
            begin = loc.find(f"{NS}begin")
            end = loc.find(f"{NS}end")
            pos = loc.find(f"{NS}position")
            if begin is not None and end is not None:
                b = begin.get("position")
                e = end.get("position")
                if b and e:
                    f["start"] = int(b)
                    f["end"] = int(e)
            elif pos is not None:
                p = pos.get("position")
                if p:
                    f["position"] = int(p)

        # Evidence
        evidence = feat.get("evidence", "")
        if evidence:
            f["evidence"] = evidence

        features.append(f)

    record["features"] = features

    # Database cross-references
    xrefs = {"pfam": [], "interpro": [], "pdb": []}
    for dbref in entry.findall(f"{NS}dbReference"):
        db_type = dbref.get("type", "").lower()
        db_id = dbref.get("id", "")
        if db_type == "pfam":
            props = {p.get("type"): p.get("value") for p in dbref.findall(f"{NS}property")}
            xrefs["pfam"].append({"id": db_id, "name": props.get("entry name", "")})
        elif db_type == "interpro":
            props = {p.get("type"): p.get("value") for p in dbref.findall(f"{NS}property")}
            xrefs["interpro"].append({"id": db_id, "name": props.get("entry name", "")})
        elif db_type == "pdb":
            xrefs["pdb"].append(db_id)
        elif db_type == "go":
            props = {p.get("type"): p.get("value") for p in dbref.findall(f"{NS}property")}
            term = props.get("term", "")
            if term.startswith("F:"):
                record.setdefault("go_f", []).append(term[2:])
            elif term.startswith("P:"):
                record.setdefault("go_p", []).append(term[2:])
            elif term.startswith("C:"):
                record.setdefault("go_c", []).append(term[2:])

    record["xrefs"] = xrefs

    # EC numbers
    ec_nums = []
    if protein_el is not None:
        for rec_name_el in protein_el.findall(f"{NS}recommendedName") + protein_el.findall(f"{NS}submittedName") + protein_el.findall(f"{NS}alternativeName"):
            for ec_el in rec_name_el.findall(f"{NS}ecNumber"):
                if ec_el.text:
                    ec_nums.append(ec_el.text)
    if ec_nums:
        record["ec"] = ec_nums

    # Comments (function, catalytic activity, subcellular location)
    for comment in entry.findall(f"{NS}comment"):
        ctype = comment.get("type", "")
        if ctype == "function":
            texts = [t.text for t in comment.findall(f"{NS}text") if t.text]
            if texts:
                record["cc_function"] = " ".join(texts)
        elif ctype == "catalytic activity":
            reaction = comment.find(f"{NS}reaction")
            if reaction is not None:
                text_el = reaction.find(f"{NS}text")
                if text_el is not None and text_el.text:
                    record.setdefault("cc_catalytic_activity", []).append(text_el.text)
        elif ctype == "subcellular location":
            locs = []
            for subloc in comment.findall(f"{NS}subcellularLocation"):
                for loc_el in subloc.findall(f"{NS}location"):
                    if loc_el.text:
                        locs.append(loc_el.text)
            if locs:
                record["cc_subcellular_location"] = locs

    return record


def parse_xml_gz(xml_gz_path, output_jsonl, taxonomy_name):
    """Stream-parse a gzipped UniProt XML file into JSONL."""
    print(f"  Parsing {xml_gz_path} ...")

    total = 0
    kept = 0

    with gzip.open(xml_gz_path, "rb") as gz_f:
        with open(output_jsonl, "w") as out_f:
            # Use iterparse to stream through entries without loading entire XML
            context = ET.iterparse(gz_f, events=("end",))

            for event, elem in context:
                if elem.tag == f"{NS}entry":
                    total += 1
                    record = extract_features(elem)

                    if record:
                        out_f.write(json.dumps(record) + "\n")
                        kept += 1

                    # Free memory
                    elem.clear()

                    if total % 10000 == 0:
                        print(f"    {taxonomy_name}: processed {total}, kept {kept}")

    print(f"  {taxonomy_name}: {total} total entries, {kept} kept (length >= 50)")
    return kept


def download_and_parse(taxonomy_name, output_dir):
    """Download and parse one taxonomy division."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = FILES[taxonomy_name]
    gz_path = output_dir / filename
    jsonl_path = output_dir / f"{taxonomy_name}_all.jsonl"

    # Check if already parsed
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        lines = sum(1 for _ in open(jsonl_path))
        print(f"  {taxonomy_name} already parsed: {lines} proteins in {jsonl_path}")
        return lines

    # Download
    url = f"{FTP_BASE}/{filename}"
    download_file(url, gz_path)

    # Parse
    count = parse_xml_gz(gz_path, jsonl_path, taxonomy_name)

    # Save summary
    summary = {
        "taxonomy": taxonomy_name,
        "source": str(gz_path),
        "total_proteins": count,
        "file": str(jsonl_path),
    }
    with open(output_dir / f"{taxonomy_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", choices=["bacteria", "archaea", "both"], default="both")
    parser.add_argument("--output", type=str, default="downloads/uniprot_bacteria_features/")
    args = parser.parse_args()

    if args.taxonomy == "both":
        total = 0
        for tax in ["bacteria", "archaea"]:
            total += download_and_parse(tax, args.output)
        print(f"\nTotal: {total} proteins")
    else:
        download_and_parse(args.taxonomy, args.output)
