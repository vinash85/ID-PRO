#!/usr/bin/env python3
"""
Download uncharacterized/hypothetical proteins from UniProt for Protein2Text annotation.

Targets:
1. Uncharacterized bacterial proteins (diverse taxa)
2. Metagenome-derived uncharacterized proteins
3. Proteins from bioenergy-relevant environments (soil, rhizosphere, compost, rumen)

Output:
- dark_genome_proteins.json  (P2T inference format)
- dark_genome_metadata.tsv   (metadata table)
"""

import json
import csv
import time
import random
import requests
import sys
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("/data/asahu/projects/doe_genesis/preliminary_data/dark_genome")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"

# Diverse P2T questions
P2T_QUESTIONS = [
    "What is the function of this protein?",
    "What enzymatic activity might this protein have?",
    "What biological process could this protein be involved in?",
    "What domain architecture does this protein have?",
    "What molecular function does this protein perform?",
    "Can you predict the cellular role of this protein?",
    "What pathway might this protein participate in?",
    "What structural features does this protein have?",
]

# Fields to retrieve from UniProt
UNIPROT_FIELDS = (
    "accession,protein_name,organism_name,organism_id,lineage,"
    "sequence,length,gene_names,cc_function,cc_subcellular_location,"
    "go_p,go_f,go_c,xref_pfam,xref_interpro,fragment"
)


def query_uniprot(query: str, size: int = 100, cursor: str = None) -> dict:
    """Query UniProt REST API and return JSON results."""
    params = {
        "query": query,
        "format": "json",
        "fields": UNIPROT_FIELDS,
        "size": min(size, 500),
    }
    if cursor:
        params["cursor"] = cursor

    for attempt in range(5):
        try:
            resp = requests.get(UNIPROT_API, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(5 * (attempt + 1))
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            time.sleep(5 * (attempt + 1))

    print(f"  FAILED after 5 attempts for query: {query[:80]}")
    return {"results": []}


def extract_text(obj, key_path=None):
    """Safely extract text from nested UniProt JSON objects."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return "; ".join(extract_text(x) for x in obj if x)
    if isinstance(obj, dict):
        # Try common UniProt JSON patterns
        if "value" in obj:
            return str(obj["value"])
        if "fullName" in obj:
            return extract_text(obj["fullName"])
        if "recommendedName" in obj:
            return extract_text(obj["recommendedName"])
        if "submissionNames" in obj:
            return extract_text(obj["submissionNames"])
    return str(obj) if obj else ""


def parse_protein_name(protein_desc):
    """Extract protein name from UniProt proteinDescription object."""
    if not protein_desc:
        return "Uncharacterized protein"
    # Try recommendedName first
    rec = protein_desc.get("recommendedName")
    if rec:
        fn = rec.get("fullName")
        if fn and isinstance(fn, dict):
            return fn.get("value", "Uncharacterized protein")
    # Try submissionNames
    sub = protein_desc.get("submissionNames")
    if sub and isinstance(sub, list) and len(sub) > 0:
        fn = sub[0].get("fullName")
        if fn and isinstance(fn, dict):
            return fn.get("value", "Uncharacterized protein")
    return "Uncharacterized protein"


def parse_entry(entry: dict) -> dict | None:
    """Parse a UniProt JSON entry into our internal format."""
    acc = entry.get("primaryAccession", "")
    if not acc:
        return None

    seq_obj = entry.get("sequence", {})
    sequence = seq_obj.get("value", "")
    length = seq_obj.get("length", len(sequence))

    if not sequence or length < 50:
        return None  # skip very short fragments

    protein_name = parse_protein_name(entry.get("proteinDescription"))
    organism = entry.get("organism", {}).get("scientificName", "Unknown")
    organism_id = entry.get("organism", {}).get("taxonId", 0)
    lineage = entry.get("organism", {}).get("lineage", [])

    # Gene names
    gene_names_list = entry.get("genes", [])
    gene_names = []
    for g in gene_names_list:
        if "geneName" in g:
            gene_names.append(g["geneName"].get("value", ""))
        for syn in g.get("synonyms", []):
            gene_names.append(syn.get("value", ""))
    gene_str = "; ".join(gene_names) if gene_names else ""

    # Function comment
    comments = entry.get("comments", [])
    function_text = ""
    subcell_text = ""
    for c in comments:
        ctype = c.get("commentType", "")
        if ctype == "FUNCTION":
            texts = c.get("texts", [])
            function_text = "; ".join(t.get("value", "") for t in texts)
        if ctype == "SUBCELLULAR LOCATION":
            locs = c.get("subcellularLocations", [])
            loc_names = []
            for loc in locs:
                if "location" in loc:
                    loc_names.append(loc["location"].get("value", ""))
            subcell_text = "; ".join(loc_names)

    # Cross-references (Pfam, InterPro)
    xrefs = entry.get("uniProtKBCrossReferences", [])
    pfam_ids = []
    interpro_ids = []
    for xr in xrefs:
        db = xr.get("database", "")
        xid = xr.get("id", "")
        if db == "Pfam":
            pfam_ids.append(xid)
        elif db == "InterPro":
            interpro_ids.append(xid)

    # GO terms
    go_terms = []
    for xr in xrefs:
        if xr.get("database") == "GO":
            props = xr.get("properties", [])
            for p in props:
                if p.get("key") == "GoTerm":
                    go_terms.append(p.get("value", ""))

    # Fragment status
    is_fragment = seq_obj.get("fragment", False) if isinstance(seq_obj.get("fragment"), bool) else False

    return {
        "accession": acc,
        "protein_name": protein_name,
        "organism": organism,
        "organism_id": organism_id,
        "lineage": lineage,
        "sequence": sequence,
        "length": length,
        "gene_names": gene_str,
        "function": function_text,
        "subcellular_location": subcell_text,
        "pfam": "; ".join(pfam_ids),
        "interpro": "; ".join(interpro_ids),
        "go_terms": "; ".join(go_terms),
        "is_fragment": is_fragment,
    }


def build_queries():
    """
    Build a list of (query_string, label, target_count) tuples.
    We query multiple categories to get diverse proteins.
    """
    queries = []

    # 1. Metagenome-derived uncharacterized proteins from bioenergy-relevant environments
    bioenergy_envs = [
        ("soil metagenome", "soil_metagenome", 60),
        ("rhizosphere metagenome", "rhizosphere_metagenome", 40),
        ("compost metagenome", "compost_metagenome", 30),
        ("gut metagenome", "gut_metagenome", 30),
        ("rumen metagenome", "rumen_metagenome", 25),
        ("marine metagenome", "marine_metagenome", 25),
        ("freshwater metagenome", "freshwater_metagenome", 20),
        ("hot springs metagenome", "hotspring_metagenome", 20),
        ("biogas reactor metagenome", "biogas_metagenome", 20),
        ("wastewater metagenome", "wastewater_metagenome", 15),
        ("hydrocarbon metagenome", "hydrocarbon_metagenome", 15),
        ("sediment metagenome", "sediment_metagenome", 15),
    ]

    for org_name, label, count in bioenergy_envs:
        q = (
            f'(organism_name:"{org_name}") '
            f'AND (protein_name:"Uncharacterized protein" OR protein_name:"hypothetical protein") '
            f"AND (length:[100 TO 800])"
        )
        queries.append((q, label, count))

    # 2. Diverse bacterial uncharacterized proteins (non-metagenome, for comparison)
    bacterial_taxa = [
        ("Proteobacteria", "proteobacteria", 20),
        ("Firmicutes", "firmicutes", 15),
        ("Actinobacteria", "actinobacteria", 15),
        ("Bacteroidetes", "bacteroidetes", 15),
        ("Cyanobacteria", "cyanobacteria", 10),
    ]

    for taxon, label, count in bacterial_taxa:
        q = (
            f'(taxonomy_name:"{taxon}") '
            f'AND (protein_name:"Uncharacterized protein") '
            f"AND (reviewed:false) "
            f"AND (length:[100 TO 800])"
        )
        queries.append((q, f"bacteria_{label}", count))

    return queries


def download_proteins():
    """Download proteins from UniProt across all query categories."""
    queries = build_queries()
    all_proteins = []
    seen_accessions = set()
    source_counts = defaultdict(int)

    for query_str, label, target in queries:
        print(f"\n--- Querying: {label} (target: {target}) ---")
        print(f"  Query: {query_str[:100]}...")

        result = query_uniprot(query_str, size=min(target * 2, 500))
        entries = result.get("results", [])
        print(f"  Got {len(entries)} raw results")

        count = 0
        for entry in entries:
            if count >= target:
                break
            parsed = parse_entry(entry)
            if parsed is None:
                continue
            if parsed["accession"] in seen_accessions:
                continue

            # Verify it's truly uncharacterized (weak/no annotation)
            has_function = bool(parsed["function"].strip())
            has_go = bool(parsed["go_terms"].strip())
            # We allow proteins with some Pfam/InterPro (partial annotation)
            # but skip those with known function text
            if has_function and "uncharacterized" not in parsed["function"].lower():
                continue

            parsed["source_category"] = label
            seen_accessions.add(parsed["accession"])
            all_proteins.append(parsed)
            count += 1
            source_counts[label] += 1

        print(f"  Kept {count} proteins from {label}")
        time.sleep(1)  # be polite to UniProt

    print(f"\n=== Total unique proteins: {len(all_proteins)} ===")
    print("Source breakdown:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")

    return all_proteins


def format_for_p2t(proteins: list[dict]) -> list[dict]:
    """Format proteins into Protein2Text inference JSON format."""
    p2t_data = []
    questions = P2T_QUESTIONS[:]

    for idx, prot in enumerate(proteins):
        # Cycle through diverse questions
        question = questions[idx % len(questions)]

        entry = {
            "long_format_id": f"{prot['accession']}_{idx}",
            "id": prot["accession"],
            "protein": prot["protein_name"],
            "amino_seq": prot["sequence"],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<protein_sequence>\n{question}",
                },
                {
                    "from": "gpt",
                    "value": "Unknown",
                },
            ],
        }
        p2t_data.append(entry)

    return p2t_data


def save_metadata(proteins: list[dict], path: Path):
    """Save metadata TSV."""
    fieldnames = [
        "accession", "protein_name", "organism", "organism_id",
        "source_category", "length", "gene_names", "function",
        "subcellular_location", "pfam", "interpro", "go_terms",
        "is_fragment", "sequence",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for prot in proteins:
            writer.writerow(prot)

    print(f"Saved metadata to {path}")


def main():
    print("=" * 60)
    print("Dark Genome Protein Downloader for Protein2Text")
    print("=" * 60)

    # Download
    proteins = download_proteins()

    if not proteins:
        print("ERROR: No proteins downloaded. Check network/API.")
        sys.exit(1)

    # Shuffle for diversity in the P2T file
    random.seed(42)
    random.shuffle(proteins)

    # Format for P2T
    p2t_data = format_for_p2t(proteins)

    # Save P2T JSON
    p2t_path = OUTPUT_DIR / "dark_genome_proteins.json"
    with open(p2t_path, "w") as f:
        json.dump(p2t_data, f, indent=2)
    print(f"\nSaved {len(p2t_data)} proteins to {p2t_path}")

    # Save metadata
    meta_path = OUTPUT_DIR / "dark_genome_metadata.tsv"
    save_metadata(proteins, meta_path)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total proteins: {len(proteins)}")

    # Length distribution
    lengths = [p["length"] for p in proteins]
    print(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.0f}")

    # Source distribution
    sources = defaultdict(int)
    for p in proteins:
        sources[p["source_category"]] += 1
    print("\nBy source:")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")

    # Metagenome vs cultured
    meta_count = sum(1 for p in proteins if "metagenome" in p.get("source_category", ""))
    bact_count = sum(1 for p in proteins if "bacteria_" in p.get("source_category", ""))
    print(f"\nMetagenome-derived: {meta_count}")
    print(f"Cultured bacteria: {bact_count}")

    # Annotation status
    has_pfam = sum(1 for p in proteins if p["pfam"])
    has_interpro = sum(1 for p in proteins if p["interpro"])
    has_go = sum(1 for p in proteins if p["go_terms"])
    no_annot = sum(1 for p in proteins if not p["pfam"] and not p["interpro"] and not p["go_terms"])
    print(f"\nPartial annotations:")
    print(f"  With Pfam domains: {has_pfam}")
    print(f"  With InterPro: {has_interpro}")
    print(f"  With GO terms: {has_go}")
    print(f"  Completely unannotated: {no_annot} ({100*no_annot/len(proteins):.1f}%)")

    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print("  - dark_genome_proteins.json  (P2T input)")
    print("  - dark_genome_metadata.tsv   (metadata)")


if __name__ == "__main__":
    main()
