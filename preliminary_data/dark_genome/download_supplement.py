#!/usr/bin/env python3
"""
Supplement the dark genome dataset with additional proteins from
categories that returned 0 results (soil, rhizosphere, compost, rumen, biogas)
using broader queries, and add more diversity overall.
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

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"

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

UNIPROT_FIELDS = (
    "accession,protein_name,organism_name,organism_id,lineage,"
    "sequence,length,gene_names,cc_function,cc_subcellular_location,"
    "go_p,go_f,go_c,xref_pfam,xref_interpro,fragment"
)


def query_uniprot(query, size=100):
    params = {
        "query": query,
        "format": "json",
        "fields": UNIPROT_FIELDS,
        "size": min(size, 500),
    }
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
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(5 * (attempt + 1))
    return {"results": []}


def parse_protein_name(protein_desc):
    if not protein_desc:
        return "Uncharacterized protein"
    rec = protein_desc.get("recommendedName")
    if rec:
        fn = rec.get("fullName")
        if fn and isinstance(fn, dict):
            return fn.get("value", "Uncharacterized protein")
    sub = protein_desc.get("submissionNames")
    if sub and isinstance(sub, list) and len(sub) > 0:
        fn = sub[0].get("fullName")
        if fn and isinstance(fn, dict):
            return fn.get("value", "Uncharacterized protein")
    return "Uncharacterized protein"


def parse_entry(entry):
    acc = entry.get("primaryAccession", "")
    if not acc:
        return None
    seq_obj = entry.get("sequence", {})
    sequence = seq_obj.get("value", "")
    length = seq_obj.get("length", len(sequence))
    if not sequence or length < 50:
        return None

    protein_name = parse_protein_name(entry.get("proteinDescription"))
    organism = entry.get("organism", {}).get("scientificName", "Unknown")
    organism_id = entry.get("organism", {}).get("taxonId", 0)
    lineage = entry.get("organism", {}).get("lineage", [])

    gene_names_list = entry.get("genes", [])
    gene_names = []
    for g in gene_names_list:
        if "geneName" in g:
            gene_names.append(g["geneName"].get("value", ""))
    gene_str = "; ".join(gene_names) if gene_names else ""

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

    xrefs = entry.get("uniProtKBCrossReferences", [])
    pfam_ids = []
    interpro_ids = []
    go_terms = []
    for xr in xrefs:
        db = xr.get("database", "")
        xid = xr.get("id", "")
        if db == "Pfam":
            pfam_ids.append(xid)
        elif db == "InterPro":
            interpro_ids.append(xid)
        elif db == "GO":
            for p in xr.get("properties", []):
                if p.get("key") == "GoTerm":
                    go_terms.append(p.get("value", ""))

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
        "is_fragment": False,
    }


def main():
    # Load existing proteins
    existing_path = OUTPUT_DIR / "dark_genome_proteins.json"
    with open(existing_path) as f:
        existing = json.load(f)
    seen_accessions = {e["id"] for e in existing}
    print(f"Existing proteins: {len(existing)} (skipping duplicates)")

    # Load existing metadata
    meta_path = OUTPUT_DIR / "dark_genome_metadata.tsv"
    existing_meta = []
    with open(meta_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            existing_meta.append(row)

    new_proteins = []
    source_counts = defaultdict(int)

    # Additional queries for missing/underrepresented categories
    # Use broader search terms and try taxonomy IDs for metagenome organisms
    supplement_queries = [
        # Soil-related: try broader organism search
        ('(organism_name:"soil") AND (protein_name:"Uncharacterized") AND (length:[100 TO 800])',
         "soil_metagenome", 50),
        # Rhizosphere
        ('(organism_name:"rhizosphere") AND (protein_name:"Uncharacterized") AND (length:[100 TO 800])',
         "rhizosphere_metagenome", 30),
        # Compost/thermophilic
        ('(organism_name:"compost") AND (protein_name:"Uncharacterized") AND (length:[100 TO 800])',
         "compost_metagenome", 20),
        ('(organism_name:"thermophilic") AND (protein_name:"Uncharacterized") AND (length:[100 TO 800])',
         "thermophilic", 15),
        # Rumen
        ('(organism_name:"rumen") AND (protein_name:"Uncharacterized") AND (length:[100 TO 800])',
         "rumen_metagenome", 25),
        # Biogas/anaerobic digestion
        ('(organism_name:"anaerobic digester") AND (protein_name:"Uncharacterized") AND (length:[100 TO 800])',
         "anaerobic_digester", 15),
        # Biofuel-relevant: lignocellulose degraders
        ('(organism_name:"Clostridiales") AND (protein_name:"Uncharacterized") AND (reviewed:false) AND (length:[150 TO 800])',
         "clostridiales_unchar", 20),
        # Methanogens (biogas relevant)
        ('(taxonomy_name:"Methanobacteria") AND (protein_name:"Uncharacterized") AND (length:[100 TO 800])',
         "methanobacteria", 15),
        # Cellulolytic bacteria
        ('(taxonomy_name:"Cellvibrio") AND (protein_name:"Uncharacterized") AND (length:[100 TO 800])',
         "cellvibrio", 10),
        # Candidate phyla radiation (CPR) - truly dark genome organisms
        ('(organism_name:"Candidatus") AND (protein_name:"Uncharacterized") AND (length:[100 TO 600])',
         "candidatus_cpr", 30),
        # Additional diverse metagenomes
        ('(organism_name:"activated sludge") AND (protein_name:"Uncharacterized") AND (length:[100 TO 800])',
         "activated_sludge", 15),
        ('(organism_name:"bioreactor") AND (protein_name:"Uncharacterized") AND (length:[100 TO 800])',
         "bioreactor", 15),
        # Hypothetical proteins specifically (different naming)
        ('(organism_name:"soil") AND (protein_name:"hypothetical") AND (length:[100 TO 800])',
         "soil_hypothetical", 30),
        ('(organism_name:"metagenome") AND (protein_name:"hypothetical") AND (length:[100 TO 800])',
         "metagenome_hypothetical", 30),
    ]

    for query_str, label, target in supplement_queries:
        print(f"\n--- Querying: {label} (target: {target}) ---")
        result = query_uniprot(query_str, size=min(target * 3, 500))
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
            # Skip if has known function
            if parsed["function"] and "uncharacterized" not in parsed["function"].lower():
                continue

            parsed["source_category"] = label
            seen_accessions.add(parsed["accession"])
            new_proteins.append(parsed)
            count += 1
            source_counts[label] += 1

        print(f"  Kept {count} proteins from {label}")
        time.sleep(1)

    print(f"\n=== New proteins downloaded: {len(new_proteins)} ===")

    if not new_proteins:
        print("No new proteins found. Exiting.")
        return

    # Merge with existing
    random.seed(42)
    random.shuffle(new_proteins)

    # Build combined P2T data
    all_p2t = list(existing)  # start with existing
    for idx, prot in enumerate(new_proteins):
        global_idx = len(existing) + idx
        question = P2T_QUESTIONS[global_idx % len(P2T_QUESTIONS)]
        entry = {
            "long_format_id": f"{prot['accession']}_{global_idx}",
            "id": prot["accession"],
            "protein": prot["protein_name"],
            "amino_seq": prot["sequence"],
            "conversations": [
                {"from": "human", "value": f"<protein_sequence>\n{question}"},
                {"from": "gpt", "value": "Unknown"},
            ],
        }
        all_p2t.append(entry)

    # Save combined P2T JSON
    p2t_path = OUTPUT_DIR / "dark_genome_proteins.json"
    with open(p2t_path, "w") as f:
        json.dump(all_p2t, f, indent=2)
    print(f"Saved {len(all_p2t)} total proteins to {p2t_path}")

    # Save combined metadata
    all_meta = list(existing_meta)
    fieldnames = [
        "accession", "protein_name", "organism", "organism_id",
        "source_category", "length", "gene_names", "function",
        "subcellular_location", "pfam", "interpro", "go_terms",
        "is_fragment", "sequence",
    ]
    for prot in new_proteins:
        row = {k: prot.get(k, "") for k in fieldnames}
        all_meta.append(row)

    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for row in all_meta:
            writer.writerow(row)
    print(f"Saved {len(all_meta)} total metadata rows to {meta_path}")

    # Summary
    print(f"\n{'='*60}")
    print("COMBINED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total proteins: {len(all_p2t)}")

    print("\nNew sources added:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")


if __name__ == "__main__":
    main()
