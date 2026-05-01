#!/usr/bin/env python3
"""
Generate QA training data from Structured Annotation Records.
Produces Stage 1 (isolated), Stage 2-3 (multi-turn), and Stage 4 (single-turn CoT).

Output format: JSONL (one record per line) — handles large datasets.

Usage: python generate_qa.py [--stage 1|2|3|4|all]
"""

import json
import re
import random
import argparse
from pathlib import Path

BASE = Path("/data/asahu/projects/doe_genesis/preliminary_data/training_data")
RECORDS = BASE / "structured_records" / "annotation_records.jsonl"
QA_DIR = BASE / "qa_stages"

random.seed(42)

# ── Text Sanitization ─────────────────────────────────────────────────

# Organism names to strip from answers (per QA_GUIDELINES.md)
_ORGANISM_RE = re.compile(
    r'\b(?:in\s+)?(?:Escherichia\s+coli|E\.\s*coli|Bacillus\s+subtilis|B\.\s*subtilis|'
    r'Salmonella\s+(?:typhimurium|enterica)|Pseudomonas\s+(?:aeruginosa|putida)|'
    r'Staphylococcus\s+aureus|Streptococcus\s+(?:pneumoniae|pyogenes)|'
    r'Mycobacterium\s+(?:tuberculosis|smegmatis)|Helicobacter\s+pylori|'
    r'Clostridium\s+(?:difficile|perfringens)|Vibrio\s+cholerae|'
    r'Corynebacterium\s+glutamicum|Caulobacter\s+crescentus|'
    r'Thermus\s+thermophilus|Thermotoga\s+maritima|'
    r'Halobacterium\s+salinarum|Methanothermobacter\s+thermautotrophicus)\b',
    re.IGNORECASE
)

_PUBMED_RE = re.compile(
    r'\s*[\[\(]?\s*(?:PubMed|PMID)\s*[:\s]*\d+\s*[\]\)]?\s*',
    re.IGNORECASE
)

_CITATION_RE = re.compile(r'\s*\(\s*\)\s*')  # empty parens from stripped refs


def sanitize_text(text):
    """Remove organism names, PubMed refs, and clean up residual artifacts."""
    if not text:
        return text
    text = _PUBMED_RE.sub(' ', text)
    text = _ORGANISM_RE.sub('', text)
    text = _CITATION_RE.sub(' ', text)
    # Clean up double spaces and leading/trailing whitespace
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\n +', '\n', text)
    return text.strip()


# ── Question Templates ─────────────────────────────────────────────────

STAGE1_QUESTIONS = {
    "domain": [
        "What protein domain is this sequence?",
        "Identify the functional domain in this amino acid sequence.",
        "What type of protein domain does this sequence encode?",
    ],
    "motif": [
        "What functional motif is present in this sequence?",
        "Identify the conserved motif in this sequence.",
        "What catalytic or binding feature can you identify in this sequence?",
    ],
    "structural": [
        "What structural feature is this sequence?",
        "What can you identify about this sequence region?",
        "What type of structural element does this sequence represent?",
    ],
    "full_protein_domains": [
        "What functional domains are present in this protein sequence?",
        "Identify all protein domains in this amino acid sequence.",
        "What domains can be recognized from this protein sequence?",
    ],
    "full_protein_features": [
        "What sequence-derivable features are present in this protein?",
        "What functional and structural features can be identified in this sequence?",
    ],
}

STAGE2_QUESTIONS = [
    "What functional domains, motifs, and structural features are present in this protein sequence?",
    "How are these features spatially arranged along the sequence?",
    "How do these features relate to each other structurally and functionally?",
    "Based on all features, their spatial arrangement, and relationships, what is the biological function and mechanism?",
]

STAGE4_QUESTIONS = [
    "Based on the protein sequence, what is the biological function and potential application of this protein?",
    "Analyze this protein sequence: identify its domains, motifs, spatial arrangement, and predict its function.",
    "What can be inferred about this protein's function from its sequence-derivable features?",
]


def load_records():
    if not RECORDS.exists():
        print(f"No records file at {RECORDS}")
        return []
    records = []
    with open(RECORDS) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def format_domain_desc(domain):
    """Format a domain annotation for training."""
    name = domain.get("name", "Unknown domain")
    start = domain.get("start", "?")
    end = domain.get("end", "?")
    ipr_desc = domain.get("interpro_description", "")
    if start != "?" and end != "?":
        desc = f"{name} at residues {start}-{end}"
    else:
        desc = name
    if ipr_desc:
        # Take first 2 sentences, sanitize organism names and PubMed refs
        clean_desc = sanitize_text(ipr_desc)
        sentences = clean_desc.split(". ")
        desc += ". " + ". ".join(sentences[:2])
        if not desc.endswith("."):
            desc += "."
    return desc


def format_motif_desc(motif):
    """Format a motif annotation."""
    name = motif.get("name", "Unknown motif")
    pos = motif.get("position")
    start = motif.get("start")
    end = motif.get("end")
    mtype = motif.get("type", "").replace("_", " ")
    if pos:
        return f"{name} at position {pos} ({mtype})"
    elif start and end:
        return f"{name} at residues {start}-{end} ({mtype})"
    return f"{name} ({mtype})"


def aggregate_motifs(motifs):
    """Aggregate binding sites and other repetitive motifs into concise descriptions.
    E.g., 15 individual binding sites → '15 binding site residues at positions 22, 79, ...'
    """
    from collections import defaultdict
    groups = defaultdict(list)

    for m in motifs:
        mtype = m.get("type", "")
        name = m.get("name", "")
        pos = m.get("position") or m.get("start", "?")

        if mtype == "binding_site":
            # Group by ligand/description
            key = ("binding_site", name if name and name.lower() != "binding site" else "substrate/cofactor")
        elif mtype == "active_site":
            key = ("active_site", name if name else "catalytic residue")
        else:
            # Don't aggregate other motif types
            groups[("other", format_motif_desc(m))].append(pos)
            continue

        groups[key].append(pos)

    aggregated = []
    for (mtype, name), positions in groups.items():
        if mtype == "other":
            aggregated.append(name)
        elif len(positions) == 1:
            aggregated.append(f"{name} at position {positions[0]} ({mtype.replace('_', ' ')})")
        elif len(positions) <= 5:
            pos_str = ", ".join(str(p) for p in positions)
            aggregated.append(f"{name} at positions {pos_str} ({len(positions)} {mtype.replace('_', ' ')} residues)")
        else:
            # Show first 3 and last position
            shown = ", ".join(str(p) for p in positions[:3])
            aggregated.append(f"{name}: {len(positions)} {mtype.replace('_', ' ')} residues at positions {shown}, ... {positions[-1]}")

    return aggregated


def format_structural_desc(feat):
    """Format a structural feature."""
    name = feat.get("name", "Unknown")
    ftype = feat.get("type", "").replace("_", " ")
    start = feat.get("start", "?")
    end = feat.get("end", "?")
    if name == ftype or not name:
        name = ftype.title()
    if start != "?" and end != "?":
        return f"{name} at residues {start}-{end}"
    return name


def write_jsonl(items, path):
    """Write list of items to JSONL file."""
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


# ── Stage 1: Isolated Feature QA ──────────────────────────────────────

def generate_stage1(records, output_dir):
    """Generate isolated domain/motif/structural feature QA + full-protein domain QA."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    isolated_pairs = []
    fullseq_pairs = []

    for rec in records:
        seq = rec["sequence"]
        uid = rec["uniprot_id"]

        # Type A: Isolated domain subsequences
        for domain in rec["IDENTIFY"]["domains"]:
            start = domain.get("start")
            end = domain.get("end")
            if start and end and end > start:
                subseq = seq[start-1:end]  # 1-indexed to 0-indexed
                if len(subseq) < 20:
                    continue

                q = random.choice(STAGE1_QUESTIONS["domain"])
                a = f"Based on the sequence, this is a {format_domain_desc(domain)}"

                isolated_pairs.append({
                    "long_format_id": f"{uid}_domain_{start}_{end}",
                    "id": uid,
                    "protein": "Unknown",
                    "amino_seq": subseq,
                    "conversations": [
                        {"from": "human", "value": f"<protein_sequence>\n{q}"},
                        {"from": "gpt", "value": a}
                    ]
                })

        # Type A: Isolated structural feature subsequences
        for feat in rec["IDENTIFY"]["structural_features"]:
            start = feat.get("start")
            end = feat.get("end")
            if start and end and end > start:
                subseq = seq[start-1:end]
                if len(subseq) < 5:
                    continue

                q = random.choice(STAGE1_QUESTIONS["structural"])
                a = f"Based on the sequence, this is a {format_structural_desc(feat)}."

                isolated_pairs.append({
                    "long_format_id": f"{uid}_struct_{feat['type']}_{start}",
                    "id": uid,
                    "protein": "Unknown",
                    "amino_seq": subseq,
                    "conversations": [
                        {"from": "human", "value": f"<protein_sequence>\n{q}"},
                        {"from": "gpt", "value": a}
                    ]
                })

        # Type B: Full-protein domain identification
        domains = rec["IDENTIFY"]["domains"]
        if domains:
            q = random.choice(STAGE1_QUESTIONS["full_protein_domains"])
            parts = []
            for i, d in enumerate(domains):
                parts.append(f"{i+1}. {format_domain_desc(d)}")
            a = "Based on the sequence, this protein contains:\n" + "\n".join(parts)

            fullseq_pairs.append({
                "long_format_id": f"{uid}_fullseq_domains",
                "id": uid,
                "protein": "Unknown",
                "amino_seq": seq,
                "conversations": [
                    {"from": "human", "value": f"<protein_sequence>\n{q}"},
                    {"from": "gpt", "value": a}
                ]
            })

        # Type B: Full-protein all features
        all_feats = domains + rec["IDENTIFY"]["motifs"] + rec["IDENTIFY"]["structural_features"]
        if len(all_feats) >= 2:
            q = random.choice(STAGE1_QUESTIONS["full_protein_features"])
            parts = []
            for d in domains:
                parts.append(format_domain_desc(d))
            for desc in aggregate_motifs(rec["IDENTIFY"]["motifs"]):
                parts.append(desc)
            for s in rec["IDENTIFY"]["structural_features"]:
                parts.append(format_structural_desc(s))
            a = "Based on the sequence, this protein contains:\n" + "\n".join(f"  {i+1}. {p}" for i, p in enumerate(parts))

            fullseq_pairs.append({
                "long_format_id": f"{uid}_fullseq_allfeats",
                "id": uid,
                "protein": "Unknown",
                "amino_seq": seq,
                "conversations": [
                    {"from": "human", "value": f"<protein_sequence>\n{q}"},
                    {"from": "gpt", "value": a}
                ]
            })

    # Combine and shuffle
    all_pairs = isolated_pairs + fullseq_pairs
    random.shuffle(all_pairs)

    output_file = output_dir / "stage1_qa.jsonl"
    write_jsonl(all_pairs, output_file)

    print(f"Stage 1: {len(all_pairs)} QA pairs ({len(isolated_pairs)} isolated + {len(fullseq_pairs)} full-protein) → {output_file}")
    return len(all_pairs)


# ── Stage 2-3: Multi-Turn Decomposed QA ───────────────────────────────

def generate_stage23(records, output_dir, stage=2):
    """Generate multi-turn decomposed QA (4 turns: IDENTIFY→LOCATE→RELATE→INFER)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_items = []

    # Stage 2: simple proteins (1 domain, few motifs)
    # Stage 3: complex proteins (2+ domains, or 1 domain with 2+ structural features)
    for rec in records:
        n_domains = len(rec["IDENTIFY"]["domains"])
        n_motifs = len(rec["IDENTIFY"]["motifs"])
        n_structural = len(rec["IDENTIFY"]["structural_features"])
        total_features = n_domains + n_motifs + n_structural

        if stage == 2:
            if not (n_domains == 1 and total_features <= 5):
                continue
        else:  # stage 3
            if not (n_domains >= 2 or (n_domains >= 1 and n_structural >= 2)):
                continue

        seq = rec["sequence"]
        uid = rec["uniprot_id"]

        # Turn 1: IDENTIFY — list all features (with aggregated binding sites)
        identify_parts = []
        for d in rec["IDENTIFY"]["domains"]:
            identify_parts.append(format_domain_desc(d))
        for desc in aggregate_motifs(rec["IDENTIFY"]["motifs"]):
            identify_parts.append(desc)
        for s in rec["IDENTIFY"]["structural_features"]:
            identify_parts.append(format_structural_desc(s))

        if not identify_parts:
            continue

        identify_answer = "Based on the sequence, this protein contains:\n" + "\n".join(
            f"  {i+1}. {p}" for i, p in enumerate(identify_parts))

        # Turn 2: LOCATE — spatial arrangement
        arch = rec["LOCATE"]["architecture"]
        if arch:
            locate_answer = f"The features are arranged along the sequence (N→C terminus):\n  {arch}"
        else:
            locate_answer = "The features are arranged linearly along the protein sequence from N-terminus to C-terminus."

        # Turn 3: RELATE — structural/functional relationships
        relate_parts = []
        if rec["RELATE"].get("mechanism_description"):
            relate_parts.append(sanitize_text(rec["RELATE"]["mechanism_description"][:200]))
        elif rec["RELATE"].get("mechanism"):
            relate_parts.append(f"This protein functions as a {rec['RELATE']['mechanism']}.")
        for extra in rec["RELATE"].get("extra_interpro", [])[:2]:
            relate_parts.append(sanitize_text(f"{extra['name']}: {extra['description'][:150]}"))
        if rec["RELATE"]["has_pdb"]:
            relate_parts.append("Experimental 3D structure data is available for this protein.")

        if relate_parts:
            relate_answer = "Based on the spatial arrangement:\n" + "\n".join(f"  - {p}" for p in relate_parts)
        else:
            # Use GO terms and domain co-occurrence for relating
            go_parts = rec["INFER"].get("go_function", [])[:3]
            if go_parts:
                relate_answer = "Based on the identified features, the protein has the following functional attributes:\n" + "\n".join(f"  - {g}" for g in go_parts)
            else:
                relate_answer = "The relationship between the identified features suggests a coordinated functional role, where each domain/motif contributes to the overall protein function."

        # Turn 4: INFER — function, mechanism, application
        infer_parts = []
        if rec["INFER"]["function"]:
            func = sanitize_text(rec["INFER"]["function"])
            func = func.split(" (By similarity")[0].strip()
            infer_parts.append(func[:400])
        if rec["INFER"]["catalytic_activity"]:
            infer_parts.append(f"Catalytic activity: {rec['INFER']['catalytic_activity'][:200]}")
        if rec["INFER"]["ec"]:
            infer_parts.append(f"EC classification: {rec['INFER']['ec']}")
        if rec["CONTEXTUALIZE"]["application"]:
            infer_parts.append(rec["CONTEXTUALIZE"]["application"])
        if rec["CONTEXTUALIZE"]["subcellular_location"]:
            infer_parts.append(f"Subcellular location: {rec['CONTEXTUALIZE']['subcellular_location']}")

        if infer_parts:
            infer_answer = "Based on all identified features and their relationships:\n" + "\n".join(f"  {p}" for p in infer_parts)
        else:
            infer_answer = "The biological function of this protein requires further experimental characterization."

        qa_items.append({
            "long_format_id": f"{uid}_stage{stage}_multiturn",
            "id": uid,
            "protein": "Unknown",
            "amino_seq": seq,
            "conversations": [
                {"from": "human", "value": f"<protein_sequence>\n{STAGE2_QUESTIONS[0]}"},
                {"from": "gpt", "value": identify_answer},
                {"from": "human", "value": STAGE2_QUESTIONS[1]},
                {"from": "gpt", "value": locate_answer},
                {"from": "human", "value": STAGE2_QUESTIONS[2]},
                {"from": "gpt", "value": relate_answer},
                {"from": "human", "value": STAGE2_QUESTIONS[3]},
                {"from": "gpt", "value": infer_answer},
            ]
        })

    random.shuffle(qa_items)

    output_file = output_dir / f"stage{stage}_qa.jsonl"
    write_jsonl(qa_items, output_file)

    print(f"Stage {stage}: {len(qa_items)} multi-turn QA → {output_file}")
    return len(qa_items)


# ── Stage 4: Single-Turn CoT QA ──────────────────────────────────────

def generate_stage4(records, output_dir):
    """Generate single-turn CoT QA (all 5 steps in one answer)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_items = []

    for rec in records:
        seq = rec["sequence"]
        uid = rec["uniprot_id"]

        if not rec["IDENTIFY"]["domains"] and not rec["IDENTIFY"]["motifs"]:
            continue

        # Build full CoT answer
        cot_parts = []

        # [IDENTIFY]
        identify_lines = []
        for d in rec["IDENTIFY"]["domains"]:
            identify_lines.append(format_domain_desc(d))
        for desc in aggregate_motifs(rec["IDENTIFY"]["motifs"]):
            identify_lines.append(desc)
        for s in rec["IDENTIFY"]["structural_features"]:
            identify_lines.append(format_structural_desc(s))
        cot_parts.append("[IDENTIFY]\n" + "\n".join(f"  - {l}" for l in identify_lines))

        # [LOCATE] — always generate, even for single-domain proteins
        if rec["LOCATE"]["architecture"]:
            cot_parts.append(f"[LOCATE]\nN→C: {rec['LOCATE']['architecture']}")
        else:
            # Build LOCATE from whatever positional info is available
            positioned = []
            for d in rec["IDENTIFY"]["domains"]:
                s, e = d.get("start"), d.get("end")
                if s and e:
                    positioned.append((s, e, d["name"]))
            for sf in rec["IDENTIFY"]["structural_features"]:
                s, e = sf.get("start"), sf.get("end")
                if s and e:
                    positioned.append((s, e, sf["name"]))
            if positioned:
                positioned.sort()
                arch = " → ".join(f"[{name}, {s}-{e}]" for s, e, name in positioned)
                cot_parts.append(f"[LOCATE]\nN→C: {arch}")
            else:
                # Only motifs with point positions — list them
                point_features = []
                for m in rec["IDENTIFY"]["motifs"]:
                    pos = m.get("position")
                    if pos:
                        point_features.append((pos, m.get("name", "feature")))
                if point_features:
                    point_features.sort()
                    locs = ", ".join(f"{name} at position {pos}" for pos, name in point_features[:5])
                    cot_parts.append(f"[LOCATE]\nKey residues: {locs}")

        # [RELATE]
        relate_lines = []
        if rec["RELATE"].get("mechanism_description"):
            relate_lines.append(sanitize_text(rec["RELATE"]["mechanism_description"][:200]))
        elif rec["RELATE"].get("mechanism"):
            relate_lines.append(f"Functions as: {rec['RELATE']['mechanism']}")
        for extra in rec["RELATE"].get("extra_interpro", [])[:2]:
            relate_lines.append(sanitize_text(f"{extra['name']}: {extra['description'][:100]}"))
        if relate_lines:
            cot_parts.append("[RELATE]\n" + "\n".join(f"  - {l}" for l in relate_lines))

        # [INFER]
        infer_lines = []
        if rec["INFER"]["function"]:
            func = sanitize_text(rec["INFER"]["function"])
            func = func.split(" (By similarity")[0].strip()
            infer_lines.append(func[:400])
        if rec["INFER"]["catalytic_activity"]:
            infer_lines.append(f"Catalytic activity: {rec['INFER']['catalytic_activity'][:200]}")
        if rec["INFER"]["ec"]:
            infer_lines.append(f"EC: {rec['INFER']['ec']}")
        if infer_lines:
            cot_parts.append("[INFER]\n" + "\n".join(f"  {l}" for l in infer_lines))

        # [CONTEXTUALIZE]
        ctx_lines = []
        if rec["CONTEXTUALIZE"]["application"]:
            ctx_lines.append(rec["CONTEXTUALIZE"]["application"])
        if rec["CONTEXTUALIZE"]["subcellular_location"]:
            ctx_lines.append(f"Subcellular location: {rec['CONTEXTUALIZE']['subcellular_location']}")
        if ctx_lines:
            cot_parts.append("[CONTEXTUALIZE]\n" + "\n".join(f"  {l}" for l in ctx_lines))

        full_answer = "Based on the sequence:\n\n" + "\n\n".join(cot_parts)

        q = random.choice(STAGE4_QUESTIONS)
        qa_items.append({
            "long_format_id": f"{uid}_stage4_cot",
            "id": uid,
            "protein": "Unknown",
            "amino_seq": seq,
            "conversations": [
                {"from": "human", "value": f"<protein_sequence>\n{q}"},
                {"from": "gpt", "value": full_answer}
            ]
        })

    random.shuffle(qa_items)

    output_file = output_dir / "stage4_qa.jsonl"
    write_jsonl(qa_items, output_file)

    print(f"Stage 4: {len(qa_items)} CoT QA → {output_file}")
    return len(qa_items)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "3", "4", "all"])
    args = parser.parse_args()

    print("Loading structured annotation records...")
    records = load_records()
    print(f"Loaded {len(records)} records")

    if not records:
        print("No records found. Run build_records.py first.")
        return

    stages = ["1", "2", "3", "4"] if args.stage == "all" else [args.stage]
    totals = {}

    for stage in stages:
        if stage == "1":
            totals["stage1"] = generate_stage1(records, QA_DIR / "stage1")
        elif stage == "2":
            totals["stage2"] = generate_stage23(records, QA_DIR / "stage2", stage=2)
        elif stage == "3":
            totals["stage3"] = generate_stage23(records, QA_DIR / "stage3", stage=3)
        elif stage == "4":
            totals["stage4"] = generate_stage4(records, QA_DIR / "stage4")

    print(f"\n{'='*60}")
    print("QA Generation Summary:")
    for stage, count in totals.items():
        print(f"  {stage}: {count:>8d} QA items")
    print(f"  Total:  {sum(totals.values()):>8d}")

    with open(QA_DIR / "qa_summary.json", "w") as f:
        json.dump(totals, f, indent=2)


if __name__ == "__main__":
    main()
