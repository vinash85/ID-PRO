#!/usr/bin/env python3
"""
Download reviewed bacterial/archaeal proteins from UniProt for Protein2Text benchmarking.

Creates:
  - microbiome_benchmark.json       : Full dataset in P2T inference format (~500-1000 proteins)
  - microbiome_benchmark_metadata.tsv: Metadata table for analysis
  - bioenergy_enzymes.json           : Subset of bioenergy-relevant proteins in P2T format
"""

import json
import csv
import random
import time
import sys
import os
from pathlib import Path
import requests
from collections import defaultdict

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"
# Default to the in-repo datasets/probe_data/benchmark dir the script lives in;
# overridable via IDPRO_DATA_ROOT (which gets the same probe_data/benchmark
# subpath appended).
_DATASETS_ROOT = os.environ.get("IDPRO_DATA_ROOT")
OUTPUT_DIR = str(
    Path(_DATASETS_ROOT) / "probe_data" / "benchmark"
    if _DATASETS_ROOT
    else Path(__file__).resolve().parent
)
FIELDS = "accession,protein_name,organism_name,sequence,cc_function,ec,cc_catalytic_activity,keyword"

# Diverse question templates for P2T format
QUESTION_TEMPLATES = [
    "<protein_sequence>\nWhat is the function of this protein?",
    "<protein_sequence>\nWhat enzymatic activity does this protein have?",
    "<protein_sequence>\nWhat biological process is this protein involved in?",
    "<protein_sequence>\nDescribe the molecular function of this protein.",
    "<protein_sequence>\nWhat is the substrate specificity of this protein?",
    "<protein_sequence>\nWhat role does this protein play in cellular metabolism?",
    "<protein_sequence>\nWhat reaction does this protein catalyze?",
    "<protein_sequence>\nWhat pathway is this protein a part of?",
    "<protein_sequence>\nDescribe the biochemical activity of this protein.",
    "<protein_sequence>\nWhat is the biological significance of this protein?",
]

# ----- Bioenergy-specific queries -----
BIOENERGY_QUERIES = {
    "cellulase": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND (protein_name:"cellulase" OR protein_name:"endoglucanase" OR protein_name:"cellobiohydrolase" OR protein_name:"beta-glucosidase" OR ec:3.2.1.4 OR ec:3.2.1.91 OR ec:3.2.1.21)',
        "size": 40,
        "label": "Cellulase / glycoside hydrolase",
    },
    "hemicellulase": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND (protein_name:"xylanase" OR protein_name:"mannanase" OR protein_name:"arabinofuranosidase" OR ec:3.2.1.8 OR ec:3.2.1.37 OR ec:3.2.1.78)',
        "size": 30,
        "label": "Hemicellulase",
    },
    "ligninase": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND (protein_name:"laccase" OR protein_name:"peroxidase" OR protein_name:"lignin" OR keyword:"Lignin degradation")',
        "size": 20,
        "label": "Ligninase / lignin-modifying",
    },
    "nitrogen_fixation": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND (protein_name:"nitrogenase" OR keyword:"Nitrogen fixation" OR protein_name:"nif")',
        "size": 30,
        "label": "Nitrogen fixation",
    },
    "biosynthetic_enzyme": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND (keyword:"Antibiotic biosynthesis" OR keyword:"Secondary metabolite biosynthesis" OR protein_name:"polyketide synthase" OR protein_name:"NRPS")',
        "size": 30,
        "label": "Biosynthetic enzyme",
    },
    "transporter": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND (keyword:"Sugar transport" OR keyword:"Ion transport" OR protein_name:"ABC transporter") AND cc_function:*',
        "size": 30,
        "label": "Transporter",
    },
    "methanogenesis": {
        "query": '(taxonomy_id:2157) AND reviewed:true AND existence:1 AND (keyword:"Methanogenesis" OR protein_name:"methyl-coenzyme M reductase" OR protein_name:"methane")',
        "size": 20,
        "label": "Methanogenesis",
    },
    "biofuel_pathway": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND (protein_name:"alcohol dehydrogenase" OR protein_name:"aldehyde dehydrogenase" OR protein_name:"butanol" OR ec:1.1.1.1 OR ec:1.1.1.2) AND cc_function:*',
        "size": 25,
        "label": "Biofuel pathway enzyme",
    },
}

# ----- General diversity queries for the broader benchmark -----
GENERAL_QUERIES = {
    "metabolic_diverse": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND (keyword:"Glycolysis" OR keyword:"Gluconeogenesis" OR keyword:"Tricarboxylic acid cycle" OR keyword:"Pentose shunt")',
        "size": 60,
        "label": "Central metabolism",
    },
    "oxidoreductase": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND ec:1.*',
        "size": 60,
        "label": "Oxidoreductase",
    },
    "transferase": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND ec:2.*',
        "size": 60,
        "label": "Transferase",
    },
    "hydrolase": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND ec:3.*',
        "size": 60,
        "label": "Hydrolase",
    },
    "lyase": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND ec:4.*',
        "size": 40,
        "label": "Lyase",
    },
    "isomerase": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND ec:5.*',
        "size": 30,
        "label": "Isomerase",
    },
    "ligase": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND ec:6.*',
        "size": 30,
        "label": "Ligase",
    },
    "dna_repair": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND keyword:"DNA repair"',
        "size": 30,
        "label": "DNA repair",
    },
    "stress_response": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND (keyword:"Stress response" OR keyword:"Heat shock")',
        "size": 30,
        "label": "Stress response",
    },
    "amino_acid_biosynthesis": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND keyword:"Amino-acid biosynthesis"',
        "size": 40,
        "label": "Amino acid biosynthesis",
    },
    "photosynthesis": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND keyword:"Photosynthesis"',
        "size": 25,
        "label": "Photosynthesis",
    },
    "lipid_metabolism": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND (keyword:"Lipid biosynthesis" OR keyword:"Fatty acid biosynthesis")',
        "size": 30,
        "label": "Lipid metabolism",
    },
    "cell_wall": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND keyword:"Cell wall biogenesis/degradation"',
        "size": 25,
        "label": "Cell wall biogenesis",
    },
    "signal_transduction": {
        "query": '(taxonomy_id:2 OR taxonomy_id:2157) AND reviewed:true AND existence:1 AND cc_function:* AND (keyword:"Two-component regulatory system" OR keyword:"Quorum sensing")',
        "size": 25,
        "label": "Signal transduction",
    },
}


def query_uniprot(query: str, size: int, fields: str = FIELDS) -> list:
    """Query UniProt REST API and return results."""
    params = {
        "query": query,
        "format": "json",
        "fields": fields,
        "size": min(size, 500),
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(UNIPROT_BASE, params=params, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", [])
            elif resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code} for query (attempt {attempt+1})")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e} (attempt {attempt+1})")
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))

    return []


def extract_function_text(entry: dict) -> str:
    """Extract function description from UniProt JSON entry."""
    comments = entry.get("comments", [])
    for comment in comments:
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                return " ".join(t.get("value", "") for t in texts).strip()
    return ""


def extract_catalytic_activity(entry: dict) -> str:
    """Extract catalytic activity from UniProt JSON entry."""
    comments = entry.get("comments", [])
    activities = []
    for comment in comments:
        if comment.get("commentType") == "CATALYTIC ACTIVITY":
            reaction = comment.get("reaction", {})
            name = reaction.get("name", "")
            if name:
                activities.append(name)
    return "; ".join(activities)


def extract_protein_name(entry: dict) -> str:
    """Extract recommended protein name."""
    prot_desc = entry.get("proteinDescription", {})

    # Try recommended name
    rec_name = prot_desc.get("recommendedName", {})
    if rec_name:
        full_name = rec_name.get("fullName", {})
        if isinstance(full_name, dict):
            return full_name.get("value", "")
        elif isinstance(full_name, str):
            return full_name

    # Try submitted name
    sub_names = prot_desc.get("submissionNames", [])
    if sub_names:
        full_name = sub_names[0].get("fullName", {})
        if isinstance(full_name, dict):
            return full_name.get("value", "")

    # Try alternative names
    alt_names = prot_desc.get("alternativeNames", [])
    if alt_names:
        full_name = alt_names[0].get("fullName", {})
        if isinstance(full_name, dict):
            return full_name.get("value", "")

    return entry.get("uniProtkbId", "Unknown")


def extract_organism(entry: dict) -> str:
    """Extract organism name."""
    org = entry.get("organism", {})
    return org.get("scientificName", "Unknown")


def extract_sequence(entry: dict) -> str:
    """Extract amino acid sequence."""
    seq = entry.get("sequence", {})
    return seq.get("value", "")


def extract_accession(entry: dict) -> str:
    """Extract primary accession."""
    return entry.get("primaryAccession", "")


def extract_ec_numbers(entry: dict) -> str:
    """Extract EC numbers."""
    prot_desc = entry.get("proteinDescription", {})
    rec_name = prot_desc.get("recommendedName", {})
    ec_numbers = rec_name.get("ecNumbers", [])
    return ", ".join(ec.get("value", "") for ec in ec_numbers)


def extract_keywords(entry: dict) -> list:
    """Extract keyword values."""
    return [kw.get("name", "") for kw in entry.get("keywords", [])]


def build_function_description(entry: dict) -> str:
    """Build a comprehensive function description from all available annotation."""
    parts = []

    func_text = extract_function_text(entry)
    if func_text:
        parts.append(func_text)

    cat_activity = extract_catalytic_activity(entry)
    if cat_activity and cat_activity not in func_text:
        parts.append(f"Catalytic activity: {cat_activity}")

    ec = extract_ec_numbers(entry)
    if ec and ec not in " ".join(parts):
        parts.append(f"EC number: {ec}")

    return " ".join(parts).strip()


def make_p2t_entry(entry: dict, idx: int, category: str = "") -> dict:
    """Convert a UniProt entry to P2T inference format."""
    accession = extract_accession(entry)
    protein_name = extract_protein_name(entry)
    sequence = extract_sequence(entry)
    function_desc = build_function_description(entry)

    if not sequence or not function_desc:
        return None

    # Pick a question template — cycle through them based on index
    q_template = QUESTION_TEMPLATES[idx % len(QUESTION_TEMPLATES)]

    p2t_entry = {
        "long_format_id": f"{accession}_{idx}",
        "id": accession,
        "protein": protein_name,
        "amino_seq": sequence,
        "category": category,
        "conversations": [
            {"from": "human", "value": q_template},
            {"from": "gpt", "value": function_desc},
        ],
    }
    return p2t_entry


def make_metadata_row(entry: dict, category: str) -> dict:
    """Build a metadata row for TSV output."""
    return {
        "accession": extract_accession(entry),
        "protein_name": extract_protein_name(entry),
        "organism": extract_organism(entry),
        "ec_number": extract_ec_numbers(entry),
        "function": extract_function_text(entry)[:500],
        "category": category,
        "sequence_length": len(extract_sequence(entry)),
    }


def main():
    random.seed(42)

    all_entries = []          # (entry_dict, category_label)
    bioenergy_entries = []    # (entry_dict, category_label)
    seen_accessions = set()

    # ---------- Download bioenergy-specific proteins ----------
    print("=" * 60)
    print("Downloading bioenergy-relevant proteins...")
    print("=" * 60)

    for key, qdef in BIOENERGY_QUERIES.items():
        print(f"\n  [{key}] {qdef['label']} (target: {qdef['size']})")
        results = query_uniprot(qdef["query"], qdef["size"])
        added = 0
        for r in results:
            acc = extract_accession(r)
            func = extract_function_text(r)
            seq = extract_sequence(r)
            if acc and func and seq and acc not in seen_accessions:
                seen_accessions.add(acc)
                bioenergy_entries.append((r, qdef["label"]))
                all_entries.append((r, qdef["label"]))
                added += 1
        print(f"    Retrieved {len(results)}, added {added} unique with function text")
        time.sleep(0.5)  # polite rate limiting

    print(f"\nBioenergy total: {len(bioenergy_entries)} proteins")

    # ---------- Download general diversity proteins ----------
    print("\n" + "=" * 60)
    print("Downloading general diversity proteins...")
    print("=" * 60)

    for key, qdef in GENERAL_QUERIES.items():
        print(f"\n  [{key}] {qdef['label']} (target: {qdef['size']})")
        results = query_uniprot(qdef["query"], qdef["size"])
        added = 0
        for r in results:
            acc = extract_accession(r)
            func = extract_function_text(r)
            seq = extract_sequence(r)
            if acc and func and seq and acc not in seen_accessions:
                seen_accessions.add(acc)
                all_entries.append((r, qdef["label"]))
                added += 1
        print(f"    Retrieved {len(results)}, added {added} unique with function text")
        time.sleep(0.5)

    print(f"\nTotal collected: {len(all_entries)} proteins")
    print(f"Bioenergy subset: {len(bioenergy_entries)} proteins")

    # ---------- Build P2T formatted datasets ----------
    print("\n" + "=" * 60)
    print("Building P2T formatted datasets...")
    print("=" * 60)

    # Full benchmark
    p2t_full = []
    metadata_rows = []
    for idx, (entry, cat) in enumerate(all_entries):
        p2t = make_p2t_entry(entry, idx, category=cat)
        if p2t:
            p2t_full.append(p2t)
            metadata_rows.append(make_metadata_row(entry, cat))

    # Bioenergy subset
    p2t_bioenergy = []
    for idx, (entry, cat) in enumerate(bioenergy_entries):
        p2t = make_p2t_entry(entry, idx, category=cat)
        if p2t:
            p2t_bioenergy.append(p2t)

    # ---------- Write outputs ----------
    print(f"\nWriting {len(p2t_full)} entries to microbiome_benchmark.json ...")
    with open(os.path.join(OUTPUT_DIR, "microbiome_benchmark.json"), "w") as f:
        json.dump(p2t_full, f, indent=2)

    print(f"Writing {len(metadata_rows)} rows to microbiome_benchmark_metadata.tsv ...")
    with open(os.path.join(OUTPUT_DIR, "microbiome_benchmark_metadata.tsv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Writing {len(p2t_bioenergy)} entries to bioenergy_enzymes.json ...")
    with open(os.path.join(OUTPUT_DIR, "bioenergy_enzymes.json"), "w") as f:
        json.dump(p2t_bioenergy, f, indent=2)

    # ---------- Summary statistics ----------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    cat_counts = defaultdict(int)
    for row in metadata_rows:
        cat_counts[row["category"]] += 1

    print(f"\n{'Category':<40} {'Count':>6}")
    print("-" * 48)
    for cat in sorted(cat_counts.keys()):
        print(f"  {cat:<38} {cat_counts[cat]:>6}")
    print("-" * 48)
    print(f"  {'TOTAL':<38} {sum(cat_counts.values()):>6}")

    # Sequence length stats
    seq_lengths = [row["sequence_length"] for row in metadata_rows]
    print(f"\nSequence length: min={min(seq_lengths)}, max={max(seq_lengths)}, "
          f"median={sorted(seq_lengths)[len(seq_lengths)//2]}, mean={sum(seq_lengths)//len(seq_lengths)}")

    # Organism diversity
    organisms = set(row["organism"] for row in metadata_rows)
    print(f"Unique organisms: {len(organisms)}")

    print(f"\nFiles written to: {OUTPUT_DIR}")
    print("  - microbiome_benchmark.json")
    print("  - microbiome_benchmark_metadata.tsv")
    print("  - bioenergy_enzymes.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
