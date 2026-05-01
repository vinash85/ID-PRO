#!/usr/bin/env python3
"""
Build Structured Annotation Records from downloaded data.
Combines UniProt features + InterPro descriptions + PROSITE + SIFTS + M-CSA
into one JSON record per protein with IDENTIFY/LOCATE/RELATE/INFER/CONTEXTUALIZE fields.

Input format: FTP bulk XML-parsed JSONL (from download_uniprot.py).

Usage: python build_records.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict

BASE = Path("/data/asahu/projects/doe_genesis/preliminary_data/training_data")
DOWNLOADS = BASE / "downloads"
OUTPUT = BASE / "structured_records"

# ── Text sanitization — strip organism names and PubMed refs at record build time ──
_ORGANISM_RE = re.compile(
    r'\b(?:in\s+)?(?:Escherichia\s+coli|E\.\s*coli|Bacillus\s+subtilis|B\.\s*subtilis|'
    r'Salmonella\s+\w+|Pseudomonas\s+\w+|Staphylococcus\s+\w+|Streptococcus\s+\w+|'
    r'Mycobacterium\s+\w+|Helicobacter\s+pylori|Clostridium\s+\w+|Vibrio\s+cholerae|'
    r'Corynebacterium\s+\w+|Caulobacter\s+\w+|Thermus\s+\w+|Thermotoga\s+\w+|'
    r'Halobacterium\s+\w+|Methanothermobacter\s+\w+|Saccharomyces\s+\w+|'
    r'Drosophila\s+\w+|Caenorhabditis\s+\w+|Arabidopsis\s+\w+)\b',
    re.IGNORECASE
)
_PUBMED_RE = re.compile(r'\s*[\[\(]?\s*(?:PubMed|PMID)\s*[:\s]*\d+\s*[\]\)]?\s*', re.I)
_CITATION_RE = re.compile(r'\(\s*\)')


def sanitize_text(text):
    """Remove organism names, PubMed refs, and clean residual artifacts."""
    if not text:
        return text
    text = _PUBMED_RE.sub(' ', text)
    text = _ORGANISM_RE.sub('', text)
    text = _CITATION_RE.sub('', text)
    text = re.sub(r'  +', ' ', text).strip()
    return text


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_interpro_descriptions():
    """Load InterPro entry descriptions into lookups by accession."""
    desc_map = {}
    path = DOWNLOADS / "interpro" / "interpro_entries.jsonl"
    if not path.exists():
        print("  InterPro entries not found")
        return desc_map

    for entry in load_jsonl(str(path)):
        acc = entry.get("accession", "")
        desc_map[acc] = {
            "name": entry.get("name", ""),
            "description": entry.get("description", ""),
            "type": entry.get("type", ""),
            "go_terms": entry.get("go_terms", []),
            "member_databases": entry.get("member_databases", {}),
        }

    # Also build a Pfam→InterPro reverse lookup
    pfam_to_ipr = {}
    for acc, info in desc_map.items():
        for pfam_entry in info.get("member_databases", {}).get("PFAM", []):
            pfam_to_ipr[pfam_entry["id"]] = acc

    print(f"  Loaded {len(desc_map)} InterPro descriptions, {len(pfam_to_ipr)} Pfam→IPR mappings")
    return desc_map, pfam_to_ipr


def load_prosite_descriptions():
    """Load PROSITE motif descriptions."""
    desc_map = {}
    path = DOWNLOADS / "prosite" / "prosite_entries.jsonl"
    if not path.exists():
        return desc_map
    for entry in load_jsonl(str(path)):
        acc = entry.get("accession", "")
        desc_map[acc] = {
            "name": entry.get("id", entry.get("description", "")),
            "description": entry.get("comment", entry.get("description", "")),
            "pattern": entry.get("pattern", ""),
            "type": entry.get("type", ""),
        }
    print(f"  Loaded {len(desc_map)} PROSITE descriptions")
    return desc_map


def load_pdb_mapping():
    """Load UniProt→PDB mapping."""
    path = DOWNLOADS / "structure" / "uniprot_pdb_mapping.json"
    if not path.exists():
        return {}
    with open(path) as f:
        mapping = json.load(f)
    print(f"  Loaded {len(mapping)} PDB mappings")
    return mapping


def load_mcsa():
    """Load M-CSA catalytic mechanisms keyed by UniProt ID."""
    path = DOWNLOADS / "mcsa" / "mcsa_entries.jsonl"
    if not path.exists():
        return {}
    mcsa_map = {}
    for entry in load_jsonl(str(path)):
        uid = entry.get("uniprot_id", "")
        if uid:
            mcsa_map[uid] = entry
    print(f"  Loaded {len(mcsa_map)} M-CSA mechanisms")
    return mcsa_map


# Feature type mapping from XML lowercase types to our categories
DOMAIN_TYPES = {"domain", "repeat", "zinc finger region"}
MOTIF_TYPES = {"active site", "binding site", "short sequence motif", "DNA-binding region",
               "lipid moiety-binding region"}
STRUCTURAL_TYPES = {"signal peptide", "transmembrane region", "coiled-coil region",
                    "disulfide bond", "region of interest", "intramembrane region",
                    "propeptide", "topological domain"}
SKIP_TYPES = {"chain", "helix", "strand", "turn", "mutagenesis site", "modified residue",
              "sequence conflict", "compositionally biased region", "initiator methionine",
              "sequence variant", "non-terminal residue", "cross-link", "splice variant",
              "site"}


def parse_features(protein_data):
    """Parse UniProt feature annotations from FTP XML format."""
    features = protein_data.get("features", [])
    if not features:
        return None

    domains = []
    motifs = []
    structural = []

    for feat in features:
        ftype = feat.get("type", "")
        desc = feat.get("description", "")
        start = feat.get("start")
        end = feat.get("end")
        position = feat.get("position")

        if ftype in SKIP_TYPES:
            continue

        if ftype in DOMAIN_TYPES:
            entry = {"name": desc or ftype, "type": ftype.replace(" ", "_")}
            if start and end:
                entry["start"] = start
                entry["end"] = end
            elif position:
                entry["position"] = position
            domains.append(entry)

        elif ftype in MOTIF_TYPES:
            entry = {"name": desc or ftype, "type": ftype.replace(" ", "_")}
            if position:
                entry["position"] = position
            elif start and end:
                entry["start"] = start
                entry["end"] = end
            motifs.append(entry)

        elif ftype in STRUCTURAL_TYPES:
            # Use informative names instead of raw XML descriptions
            if ftype == "transmembrane region":
                dl = desc.lower()
                if dl in ("helical", "") or dl.startswith("helical;"):
                    # Extract helix number if present (e.g. "Helical; Name=2" → "Transmembrane helix 2")
                    m = re.search(r'name=(\d+)', dl)
                    name = f"Transmembrane helix {m.group(1)}" if m else "Transmembrane helix"
                elif dl.startswith("beta"):
                    name = "Transmembrane beta strand"
                else:
                    name = f"Transmembrane region ({desc})"
            elif ftype == "topological domain":
                name = f"Topological domain ({desc})" if desc else "Topological domain"
            elif ftype == "signal peptide":
                name = "Signal peptide"
            elif ftype == "intramembrane region":
                name = f"Intramembrane region" if desc.lower() in ("helical", "") else f"Intramembrane region ({desc})"
            else:
                name = desc or ftype.replace("_", " ").title()
            entry = {"name": name, "type": ftype.replace(" ", "_")}
            if start and end:
                entry["start"] = start
                entry["end"] = end
            elif position:
                entry["position"] = position
            structural.append(entry)

    if not domains and not motifs and not structural:
        return None

    return {"domains": domains, "motifs": motifs, "structural_features": structural}


def get_doe_application_context(text):
    """Determine DOE/JGI mission relevance from function text.

    DOE and JGI focus areas: bioenergy, carbon cycling, environmental genomics,
    bioremediation, nutrient cycling, plant-microbe interactions, extremophiles,
    and biosynthesis for bioproducts.
    """
    if not text:
        return ""
    t = text.lower()

    # Ordered by specificity — more specific patterns first to avoid false positives.
    # Each entry: (patterns, context_string, exclude_patterns)
    # exclude_patterns: if ANY of these match, skip this category (avoids false positives)
    keywords = [
        # ── Bioenergy: Lignocellulose degradation ──
        ([r"\bcellulase\b", r"\bcellulose degradation\b", r"\bendoglucanase\b",
          r"\bexoglucanase\b", r"\bbeta-glucosidase\b", r"\bcellobiose\b",
          r"\blignocellulose\b", r"\blignocellulosic\b"],
         "Relevant to cellulose degradation for biofuel production.",
         [r"\bglycogen\b", r"\bstarch\b"]),  # exclude glycogen/starch enzymes

        ([r"\bxylanase\b", r"\bxylan degradation\b", r"\bhemicellulose degradation\b",
          r"\barabinofuranosidase\b", r"\bxylose\b"],
         "Participates in hemicellulose degradation for biomass conversion.",
         []),

        ([r"\blignin degradation\b", r"\blignin peroxidase\b", r"\blignin modification\b",
          r"\bligninolytic\b", r"\bdelignification\b"],
         "Involved in lignin modification for biomass pretreatment.",
         []),

        ([r"\bfermentation\b", r"\bethanol production\b", r"\bbutanol production\b",
          r"\bacetone-butanol\b"],
         "Participates in fermentation pathways for biofuel production.",
         []),

        # ── Biogas & methane ──
        ([r"\bmethanogenesis\b", r"\bmethanotroph\b", r"\bmethyl-coenzyme m\b",
          r"\bmethane monooxygenase\b", r"\bmethane oxidation\b"],
         "Involved in methane metabolism for biogas production.",
         []),

        # ── Nitrogen cycling ──
        ([r"\bnitrogenase\b", r"\bnitrogen fixation\b", r"\bdinitrogen\b", r"\bnifh\b"],
         "Related to biological nitrogen fixation, a key process for sustainable agriculture and bioenergy crops.",
         []),
        ([r"\bdenitrification\b", r"\bnitrite reductase\b", r"\bnitrous oxide reductase\b",
          r"\bnitrate reductase\b"],
         "Involved in the nitrogen cycle (denitrification/nitrate reduction), relevant to understanding soil and environmental microbial communities.",
         []),
        ([r"\bnitrification\b", r"\bammonia oxidation\b", r"\bammonia monooxygenase\b"],
         "Involved in nitrification/ammonia oxidation, a key step in the global nitrogen cycle.",
         []),

        # ── Carbon cycling ──
        ([r"\bcarbon fixation\b", r"\brubisco\b", r"\bcalvin cycle\b",
          r"\bautotrophic co2\b", r"\bcarboxylase.*co2\b"],
         "Involved in carbon fixation, relevant to understanding the global carbon cycle.",
         [r"\boxaloacetate\b.*\btricarboxylic\b"]),  # exclude generic TCA

        ([r"\bphotosystem\b", r"\bphotosynthetic reaction center\b",
          r"\bchlorophyll biosynthesis\b", r"\bbacteriochlorophyll\b"],
         "Part of the photosynthetic apparatus, relevant to understanding microbial carbon fixation.",
         []),

        # ── Sulfur cycling ──
        ([r"\bsulfate reduction\b", r"\bsulfite reductase\b",
          r"\bdissimilatory sulfur\b", r"\bsulfur oxidation\b"],
         "Involved in the sulfur cycle, relevant to understanding biogeochemical processes in microbial communities.",
         []),

        # ── Bioremediation ──
        ([r"\bbioremediation\b", r"\bxenobiotic degradation\b",
          r"\baromatic compound degradation\b", r"\bpolycyclic aromatic\b",
          r"\bheavy metal resistance\b", r"\bmercury reductase\b",
          r"\barsenic resistance\b", r"\barsenate reductase\b"],
         "Relevant to bioremediation of environmental contaminants.",
         []),

        # ── Hydrogen production ──
        ([r"\bhydrogenase\b", r"\bhydrogen production\b", r"\bhydrogen evolution\b"],
         "Relevant to biological hydrogen production for clean energy.",
         [r"\belectron transport\b.*\bcytochrome\b"]),  # exclude generic ET chains

        # ── Biosynthesis / bioproducts ──
        ([r"\bpolyketide synthase\b", r"\bnonribosomal peptide\b",
          r"\bterpene synthase\b", r"\bterpene cyclase\b",
          r"\bisoprenoid biosynthesis\b", r"\bsiderophore biosynthesis\b"],
         "Part of a secondary metabolite biosynthetic pathway, relevant to bioproduct discovery.",
         []),

        # ── Plant-microbe interactions ──
        ([r"\bnodulation\b", r"\bnod factor\b", r"\brhizobium\b",
          r"\bmycorrhiza\b", r"\bplant growth promot\b"],
         "Involved in plant-microbe interactions relevant to sustainable agriculture.",
         []),

        # ── Extremophile biology ──
        ([r"\bthermostable\b", r"\bthermophilic\b", r"\bhyperthermophilic\b",
          r"\bhalophilic\b", r"\bacidophilic\b", r"\bpsychrophilic\b"],
         "From an extremophilic organism, potentially useful for industrial biotechnology applications.",
         []),
    ]

    for patterns, context, excludes in keywords:
        if any(re.search(p, t) for p in patterns):
            # Check exclusions
            if excludes and any(re.search(ep, t) for ep in excludes):
                continue
            return context
    return ""


def build_record(protein_data, interpro_desc, pfam_to_ipr, pdb_map, mcsa_map):
    """Build one structured annotation record from FTP XML format."""
    uid = protein_data.get("accession", "")
    sequence = protein_data.get("sequence", "")
    if not uid or not sequence:
        return None

    # Parse features
    parsed = parse_features(protein_data)
    if not parsed:
        return None

    # Get cross-references
    xrefs = protein_data.get("xrefs", {})
    pfam_ids = [x["id"] for x in xrefs.get("pfam", [])]
    interpro_ids = [x["id"] for x in xrefs.get("interpro", [])]
    pdb_ids = xrefs.get("pdb", [])

    # Enrich domain descriptions from InterPro
    # Strategy: match via InterPro cross-refs with strict name matching; no reuse of IPR IDs
    used_ipr_ids = set()
    used_pfam_ids = set()

    for domain in parsed["domains"]:
        best_desc = ""
        best_ipr = ""
        best_score = 0

        domain_name = domain["name"].lower().strip()
        if not domain_name:
            continue

        # Pass 1: Direct InterPro name match (strict — require significant word overlap)
        for ipr_id in interpro_ids:
            if ipr_id in used_ipr_ids or ipr_id not in interpro_desc:
                continue
            ipr = interpro_desc[ipr_id]
            ipr_name = ipr["name"].lower().strip()
            ipr_desc = ipr.get("description", "")
            if not ipr_name or not ipr_desc:
                continue

            # Compute word overlap score
            domain_words = set(domain_name.replace("-", " ").split())
            ipr_words = set(ipr_name.replace("-", " ").split())
            # Remove common short words
            stop = {"of", "the", "a", "an", "in", "and", "or", "to", "for", "type", "like", "family"}
            domain_words -= stop
            ipr_words -= stop
            if not domain_words or not ipr_words:
                continue

            overlap = domain_words & ipr_words
            score = len(overlap) / min(len(domain_words), len(ipr_words))

            # Require at least 50% word overlap
            if score >= 0.5 and score > best_score:
                best_score = score
                best_desc = ipr_desc
                best_ipr = ipr_id

        # Pass 2: If no direct match, try Pfam→IPR reverse lookup (one-to-one)
        if not best_desc:
            for pfam_id in pfam_ids:
                if pfam_id in used_pfam_ids:
                    continue
                if pfam_id in pfam_to_ipr:
                    ipr_id = pfam_to_ipr[pfam_id]
                    if ipr_id in used_ipr_ids or ipr_id not in interpro_desc:
                        continue
                    ipr = interpro_desc[ipr_id]
                    ipr_desc = ipr.get("description", "")
                    ipr_name = ipr.get("name", "").lower()
                    if not ipr_desc:
                        continue
                    # Looser match for Pfam: check if any domain word appears in IPR name
                    domain_words = set(domain_name.replace("-", " ").split()) - {"of", "the", "a", "type"}
                    if domain_words and any(w in ipr_name for w in domain_words if len(w) > 3):
                        best_desc = ipr_desc
                        best_ipr = ipr_id
                        used_pfam_ids.add(pfam_id)
                        break

        if best_desc:
            domain["interpro_id"] = best_ipr
            domain["interpro_description"] = sanitize_text(best_desc)[:500]
            used_ipr_ids.add(best_ipr)

    # For any remaining InterPro IDs not matched to a domain, attach as extra context
    # (useful for family-level annotations)
    extra_interpro = []
    for ipr_id in interpro_ids:
        if ipr_id not in used_ipr_ids and ipr_id in interpro_desc:
            ipr = interpro_desc[ipr_id]
            if ipr.get("description"):
                extra_interpro.append({
                    "interpro_id": ipr_id,
                    "name": ipr["name"],
                    "type": ipr["type"],
                    "description": sanitize_text(ipr["description"])[:300],
                })

    # Get function text (sanitize organism names and PubMed refs)
    function_text = sanitize_text(protein_data.get("cc_function", ""))

    # Get catalytic activity
    catalytic_list = protein_data.get("cc_catalytic_activity", [])
    catalytic_text = "; ".join(catalytic_list) if isinstance(catalytic_list, list) else str(catalytic_list)

    # Get EC numbers
    ec_list = protein_data.get("ec", [])
    ec = ", ".join(ec_list) if isinstance(ec_list, list) else str(ec_list) if ec_list else ""

    # Get GO terms
    go_f = protein_data.get("go_f", [])
    go_p = protein_data.get("go_p", [])
    go_c = protein_data.get("go_c", [])

    # Get subcellular location
    subcell = protein_data.get("cc_subcellular_location", [])
    subcell_text = ", ".join(subcell) if isinstance(subcell, list) else str(subcell) if subcell else ""

    # Build architecture string (sorted by position N→C)
    all_positioned = []
    for feat in parsed["domains"] + parsed["structural_features"]:
        s = feat.get("start")
        e = feat.get("end")
        if s and e:
            all_positioned.append((s, e, feat["name"]))
    all_positioned.sort(key=lambda x: x[0])
    architecture = " → ".join(f"[{name}, {s}-{e}]" for s, e, name in all_positioned)

    # M-CSA mechanism
    mechanism = ""
    mechanism_desc = ""
    if uid in mcsa_map:
        mcsa = mcsa_map[uid]
        mechanism = mcsa.get("enzyme_name", "")
        mechanism_desc = mcsa.get("description", "")

    # Combine function sources for bioenergy context
    combined_func = " ".join(filter(None, [
        function_text, catalytic_text, " ".join(go_f), " ".join(go_p)
    ]))

    # Build record
    record = {
        "uniprot_id": uid,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "IDENTIFY": {
            "domains": parsed["domains"],
            "motifs": parsed["motifs"],
            "structural_features": parsed["structural_features"],
        },
        "LOCATE": {
            "architecture": architecture,
        },
        "RELATE": {
            "has_pdb": uid in pdb_map or bool(pdb_ids),
            "mechanism": mechanism,
            "mechanism_description": mechanism_desc,
            "extra_interpro": extra_interpro,
        },
        "INFER": {
            "function": function_text,
            "catalytic_activity": catalytic_text,
            "ec": ec,
            "go_function": go_f,
            "go_process": go_p,
        },
        "CONTEXTUALIZE": {
            "application": get_doe_application_context(combined_func),
            "subcellular_location": subcell_text,
        },
        "cross_refs": {
            "pfam": pfam_ids,
            "interpro": interpro_ids,
            "pdb": pdb_ids if isinstance(pdb_ids, list) else [],
        }
    }

    return record


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading reference databases...")
    interpro_desc, pfam_to_ipr = load_interpro_descriptions()
    prosite_desc = load_prosite_descriptions()
    pdb_map = load_pdb_mapping()
    mcsa_map = load_mcsa()

    # Process UniProt proteins
    total = 0
    kept = 0
    n_domains = 0
    n_motifs = 0
    n_structural = 0
    n_with_function = 0
    n_with_ec = 0
    n_with_pdb = 0
    n_with_bioenergy = 0

    output_file = OUTPUT / "annotation_records.jsonl"

    with open(output_file, "w") as f_out:
        for taxonomy in ["bacteria", "archaea"]:
            input_file = DOWNLOADS / "uniprot_bacteria_features" / f"{taxonomy}_all.jsonl"
            if not input_file.exists():
                print(f"  {taxonomy} data not found, skipping")
                continue

            print(f"\nProcessing {taxonomy}...")
            with open(input_file) as f_in:
                for line in f_in:
                    total += 1
                    protein = json.loads(line)
                    record = build_record(protein, interpro_desc, pfam_to_ipr, pdb_map, mcsa_map)
                    if record:
                        f_out.write(json.dumps(record) + "\n")
                        kept += 1

                        # Stats
                        n_domains += len(record["IDENTIFY"]["domains"])
                        n_motifs += len(record["IDENTIFY"]["motifs"])
                        n_structural += len(record["IDENTIFY"]["structural_features"])
                        if record["INFER"]["function"]:
                            n_with_function += 1
                        if record["INFER"]["ec"]:
                            n_with_ec += 1
                        if record["RELATE"]["has_pdb"]:
                            n_with_pdb += 1
                        if record["CONTEXTUALIZE"]["application"]:
                            n_with_bioenergy += 1

                    if total % 50000 == 0:
                        print(f"    Processed {total}, kept {kept} ({100*kept/total:.0f}%)")

    print(f"\n{'='*60}")
    print(f"Complete: {kept}/{total} proteins have structured annotation records")
    print(f"  Domains:       {n_domains:>8d} total")
    print(f"  Motifs:        {n_motifs:>8d} total")
    print(f"  Structural:    {n_structural:>8d} total")
    print(f"  With function: {n_with_function:>8d} proteins")
    print(f"  With EC:       {n_with_ec:>8d} proteins")
    print(f"  With PDB:      {n_with_pdb:>8d} proteins")
    print(f"  Bioenergy:     {n_with_bioenergy:>8d} proteins")
    print(f"Saved to: {output_file}")

    summary = {
        "total_input": total,
        "records_built": kept,
        "total_domains": n_domains,
        "total_motifs": n_motifs,
        "total_structural": n_structural,
        "with_function": n_with_function,
        "with_ec": n_with_ec,
        "with_pdb": n_with_pdb,
        "with_bioenergy": n_with_bioenergy,
        "file": str(output_file),
    }
    with open(OUTPUT / "records_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
