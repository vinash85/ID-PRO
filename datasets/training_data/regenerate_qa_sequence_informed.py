#!/usr/bin/env python3
"""
Regenerate training data to be SEQUENCE-INFORMED.

Key principles:
1. Questions reference the sequence, not a protein name
2. Answers describe what can be inferred FROM the sequence
3. No protein names, PubMed IDs, accession numbers, or organism names in answers
4. Answers reference domains, motifs, structural features — things derivable from sequence
5. Include application context (bioenergy, biomanufacturing)
6. Break the name→function link
"""

import json
import re
import random
from pathlib import Path

DATA_DIR = Path("/data/asahu/projects/doe_genesis/preliminary_data/training_data")


def clean_function_text(text):
    """Remove database artifacts from function descriptions."""
    # Remove PubMed references
    text = re.sub(r'\(PubMed:\d+[^)]*\)', '', text)
    text = re.sub(r'PubMed:\d+', '', text)
    # Remove EC numbers in parentheses at start (keep inline mentions)
    text = re.sub(r'^\s*\([^)]*EC[^)]*\)\s*', '', text)
    # Remove protein names in parentheses
    text = re.sub(r'\([A-Z][a-z]*[A-Z][a-z]*\)', '', text)
    # Remove "from Organism" clauses
    text = re.sub(r'\bfrom\s+[A-Z][a-z]+\s+[a-z]+(\s+\(strain[^)]+\))?', '', text)
    # Remove accession-like patterns
    text = re.sub(r'\b[A-Z]\d{5}\b', '', text)
    text = re.sub(r'\b[A-Z]\d[A-Z0-9]{3}\d\b', '', text)
    # Remove "has the following function:" prefix
    text = re.sub(r'has the following function:\s*', '', text)
    # Remove protein name prefix like "Protein Name (Alias) from Organism"
    # Pattern: anything before "Catalyzes" or "Involved" or "Plays" or "Functions"
    match = re.search(r'(Catalyze|Involved|Play|Function|Acts|Binds|Transport|Hydrolyze|Required|Essential|Responsible|Mediates|Participates)', text)
    if match and match.start() > 20:
        text = text[match.start():]
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s*,\s*,', ',', text)
    text = re.sub(r'\(\s*\)', '', text)
    return text.strip()


def extract_ec_number(text):
    """Extract EC number from text."""
    match = re.search(r'EC[:\s]*(\d+\.\d+\.\d+\.\d+)', text)
    return match.group(1) if match else None


def extract_domains_from_answer(text):
    """Extract domain/family mentions from answer text."""
    domains = []
    # Look for common domain patterns
    patterns = [
        r'glycos[iy]l\s+hydrolase\s+family\s+\d+',
        r'GH\d+', r'GT\d+', r'PL\d+', r'CE\d+', r'AA\d+',
        r'kinase\s+domain', r'binding\s+domain',
        r'dehydrogenase', r'transferase', r'synthase', r'reductase',
        r'oxidase', r'hydrolase', r'lyase', r'isomerase', r'ligase',
        r'protease', r'peptidase', r'phosphatase',
        r'ABC\s+transporter', r'PTS\s+system',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            domains.append(m.group(0))
    return domains


def get_application_context(text):
    """Generate DOE-relevant application context based on function."""
    text_lower = text.lower()
    contexts = []

    if any(w in text_lower for w in ['cellulase', 'cellulose', 'glucan', 'glucosidase', 'endoglucanase']):
        contexts.append("This enzyme is relevant to cellulose degradation for biofuel production from plant biomass.")
    elif any(w in text_lower for w in ['xylan', 'xylose', 'hemicellulose', 'xylanase']):
        contexts.append("This enzyme participates in hemicellulose degradation, important for lignocellulosic biomass conversion.")
    elif any(w in text_lower for w in ['lignin', 'laccase', 'peroxidase']):
        contexts.append("This enzyme is involved in lignin modification, a key step in biomass pretreatment for biofuel production.")
    elif any(w in text_lower for w in ['nitrogen', 'nitrogenase', 'nif']):
        contexts.append("This enzyme is related to nitrogen fixation, critical for reducing fertilizer dependence in bioenergy crop production.")
    elif any(w in text_lower for w in ['methane', 'methanogenesis', 'methanol']):
        contexts.append("This enzyme is involved in methane metabolism, relevant to biogas production and greenhouse gas mitigation.")
    elif any(w in text_lower for w in ['ferment', 'ethanol', 'butanol']):
        contexts.append("This enzyme participates in fermentation pathways relevant to biofuel production.")
    elif any(w in text_lower for w in ['biosynthesis', 'polyketide', 'terpene']):
        contexts.append("This enzyme is part of a biosynthetic pathway that could be engineered for bioproduct manufacturing.")
    elif any(w in text_lower for w in ['transport', 'transporter', 'uptake', 'efflux']):
        contexts.append("This transport protein is important for understanding nutrient uptake and metabolite export in microbial cell factories.")
    elif any(w in text_lower for w in ['dehydrogenase', 'oxidoreductase', 'redox']):
        contexts.append("This redox enzyme is relevant to understanding electron transfer chains in microbial energy metabolism.")
    elif any(w in text_lower for w in ['sugar', 'glucose', 'galactose', 'sucrose', 'carbohydrate']):
        contexts.append("This enzyme is important for understanding and engineering carbohydrate metabolism in microbial systems.")
    else:
        contexts.append("Understanding this protein's function contributes to comprehensive annotation of microbial metabolic capabilities.")

    return contexts[0] if contexts else ""


# ── Question Templates (sequence-informed) ──────────────────────────

QUESTIONS = [
    "Based on the protein sequence, what is the likely biological function of this protein?",
    "Analyzing this protein sequence, what functional domains or motifs can be identified, and what do they suggest about the protein's role?",
    "From the amino acid sequence alone, what enzymatic activity would you predict for this protein?",
    "What can be inferred about this protein's function from its sequence features?",
    "Based on sequence analysis, what biological process is this protein likely involved in?",
    "Examining this protein sequence, what substrate specificity would you predict?",
    "What structural and functional properties can be predicted from this amino acid sequence?",
]

QUESTIONS_BIOENERGY = [
    "Based on the protein sequence, could this protein be relevant to bioenergy applications? What specific function do you predict?",
    "Analyzing this sequence, does this protein appear to be involved in biomass degradation, carbon metabolism, or other DOE-relevant processes?",
    "From the amino acid sequence, what role might this protein play in microbial community metabolism relevant to biomanufacturing?",
]


def make_sequence_informed_answer(raw_answer, ec_number=None):
    """Transform a database answer into a sequence-informed answer."""
    cleaned = clean_function_text(raw_answer)
    if not cleaned or len(cleaned) < 10:
        return None

    domains = extract_domains_from_answer(raw_answer)
    application = get_application_context(raw_answer)

    # Build sequence-informed answer
    parts = ["Based on the sequence,"]

    if domains:
        parts.append(f"this protein contains features characteristic of {', '.join(domains[:2])}.")
    else:
        parts.append("this protein")

    # Add cleaned functional description
    # Make it lowercase and remove leading articles
    func = cleaned
    if func[0].isupper() and not func.startswith(('DNA', 'RNA', 'ATP', 'NAD', 'GTP')):
        func = func[0].lower() + func[1:]

    # Remove "this protein" if already at start
    func = re.sub(r'^this protein\s*', '', func, flags=re.IGNORECASE)

    if domains:
        parts.append(f"The predicted function is: {func}")
    else:
        parts.append(f"is predicted to {func}" if not func.startswith(('catalyz', 'involv', 'play', 'act', 'bind', 'transport', 'function'))
                      else func)

    # Add EC number if available
    if ec_number:
        parts.append(f"(EC {ec_number}).")

    # Add application context
    if application:
        parts.append(application)

    answer = " ".join(parts)
    # Clean up
    answer = re.sub(r'\s+', ' ', answer).strip()
    answer = re.sub(r'\.\s*\.', '.', answer)
    answer = re.sub(r',\s*\.', '.', answer)

    return answer


def regenerate_dataset(input_file, output_file, include_bioenergy_questions=True):
    """Regenerate a QA dataset with sequence-informed format."""
    with open(input_file) as f:
        data = json.load(f)

    new_data = []
    skipped = 0

    for item in data:
        # Handle both formats (UniProt QA vs CAZy)
        if "conversations" in item:
            raw_answer = item["conversations"][1]["value"]
            amino_seq = item["amino_seq"]
            protein_id = item["id"]
        else:
            raw_answer = item.get("answer", item.get("protein_name", ""))
            amino_seq = item.get("sequence", item.get("amino_seq", ""))
            protein_id = item.get("uniprot_id", item.get("id", "unknown"))

        # Extract EC number before cleaning
        ec = extract_ec_number(raw_answer)

        # Generate sequence-informed answer
        answer = make_sequence_informed_answer(raw_answer, ec)
        if not answer or len(answer) < 30:
            skipped += 1
            continue

        # Pick a question
        if include_bioenergy_questions and random.random() < 0.3:
            question = random.choice(QUESTIONS_BIOENERGY)
        else:
            question = random.choice(QUESTIONS)

        new_data.append({
            "long_format_id": f"{protein_id}_seq_informed",
            "id": protein_id,
            "protein": "Unknown",  # Don't leak protein name
            "amino_seq": amino_seq,
            "conversations": [
                {"from": "human", "value": f"<protein_sequence>\n{question}"},
                {"from": "gpt", "value": answer}
            ]
        })

    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2)

    print(f"  Input: {len(data)}, Output: {len(new_data)}, Skipped: {skipped}")
    return new_data


def main():
    print("=" * 60)
    print("REGENERATING SEQUENCE-INFORMED TRAINING DATA")
    print("=" * 60)

    random.seed(42)

    # Regenerate UniProt combined
    print("\nProcessing UniProt combined...")
    regenerate_dataset(
        DATA_DIR / "uniprot/uniprot_combined_qa.json",
        DATA_DIR / "uniprot/uniprot_combined_qa_seq_informed.json"
    )

    # Regenerate CAZy
    print("\nProcessing CAZy bioenergy enzymes...")
    regenerate_dataset(
        DATA_DIR / "cazy/cazy_enzymes_qa.json",
        DATA_DIR / "cazy/cazy_enzymes_qa_seq_informed.json",
        include_bioenergy_questions=True
    )

    # Show examples
    print("\n" + "=" * 60)
    print("EXAMPLE SEQUENCE-INFORMED QA PAIRS")
    print("=" * 60)

    with open(DATA_DIR / "uniprot/uniprot_combined_qa_seq_informed.json") as f:
        new_data = json.load(f)

    for i in [0, 50, 200, 500]:
        if i < len(new_data):
            d = new_data[i]
            print(f"\nQ: {d['conversations'][0]['value']}")
            print(f"A: {d['conversations'][1]['value'][:300]}")
            print()


if __name__ == "__main__":
    main()
