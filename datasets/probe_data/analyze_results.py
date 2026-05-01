#!/usr/bin/env python3
"""
Analyze P2T benchmark results.
Compares predictions to ground truth for microbiome proteins.
Generates metrics and visualizations for the proposal.
"""

import os
import json
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def load_jsonl(filepath):
    """Load JSONL results file."""
    results = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def keyword_overlap(predicted, ground_truth):
    """Compute keyword overlap between prediction and ground truth."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
        "and", "or", "this", "that", "it", "for", "with", "on", "at", "by",
        "from", "as", "its", "has", "have", "can", "may", "could", "would",
        "protein", "proteins", "which", "their", "they", "been", "be", "not",
        "also", "such", "these", "those", "but", "if", "than", "no", "yes"
    }

    def extract_keywords(text):
        words = re.findall(r'[a-z]+', text.lower())
        return set(w for w in words if w not in stop_words and len(w) > 2)

    pred_kw = extract_keywords(predicted)
    truth_kw = extract_keywords(ground_truth)

    if not truth_kw:
        return 0.0

    overlap = pred_kw & truth_kw
    precision = len(overlap) / max(len(pred_kw), 1)
    recall = len(overlap) / len(truth_kw)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return f1


def assess_prediction_quality(predicted):
    """Assess qualitative aspects of a prediction."""
    scores = {}

    # Specificity: does it name specific proteins, domains, pathways?
    specific_terms = [
        "domain", "motif", "binding", "catalytic", "active site",
        "kinase", "phosphatase", "dehydrogenase", "transferase",
        "synthase", "reductase", "oxidase", "protease", "hydrolase",
        "pathway", "metabolism", "biosynthesis", "degradation",
        "membrane", "cytoplasm", "extracellular", "nuclear",
        "ATP", "NAD", "FAD", "CoA", "DNA", "RNA"
    ]
    specificity = sum(1 for term in specific_terms if term.lower() in predicted.lower())
    scores["specificity"] = min(specificity / 3, 1.0)  # normalize

    # Length: reasonable responses are 50-500 chars
    length = len(predicted)
    if length < 20:
        scores["length_quality"] = 0.2
    elif length < 50:
        scores["length_quality"] = 0.5
    elif length < 500:
        scores["length_quality"] = 1.0
    else:
        scores["length_quality"] = 0.8

    # Hallucination indicators
    hallucination_markers = [
        "I don't know", "I cannot", "not enough information",
        "unable to determine", "no information available"
    ]
    has_hedge = any(m in predicted.lower() for m in hallucination_markers)
    scores["confidence"] = 0.3 if has_hedge else 0.8

    # Bioenergy relevance
    bioenergy_terms = [
        "cellulase", "lignin", "cellulose", "hemicellulose", "biomass",
        "biofuel", "fermentation", "ethanol", "methane", "nitrogen",
        "fixation", "carbon", "photosynthesis", "depolymerization",
        "enzyme", "catalysis", "substrate", "metabolic", "biosynthetic"
    ]
    bioenergy = sum(1 for t in bioenergy_terms if t in predicted.lower())
    scores["bioenergy_relevance"] = min(bioenergy / 2, 1.0)

    return scores


def analyze_benchmark(results_file, output_dir):
    """Full analysis of benchmark results."""
    results = load_jsonl(results_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "total_proteins": len(results),
        "keyword_f1_scores": [],
        "specificity_scores": [],
        "bioenergy_relevance_scores": [],
        "predictions": []
    }

    for r in results:
        predicted = r.get("Predicted", "")
        ground_truth = r.get("Ground Truth", "")

        f1 = keyword_overlap(predicted, ground_truth)
        quality = assess_prediction_quality(predicted)

        metrics["keyword_f1_scores"].append(f1)
        metrics["specificity_scores"].append(quality["specificity"])
        metrics["bioenergy_relevance_scores"].append(quality["bioenergy_relevance"])

        metrics["predictions"].append({
            "id": r.get("long_format_id", ""),
            "prompt": r.get("Prompt", ""),
            "ground_truth": ground_truth[:200],
            "predicted": predicted[:200],
            "keyword_f1": f1,
            "specificity": quality["specificity"],
            "bioenergy_relevance": quality["bioenergy_relevance"],
            "confidence": quality["confidence"]
        })

    # Summary statistics
    f1_scores = metrics["keyword_f1_scores"]
    summary = {
        "total_proteins": len(results),
        "keyword_f1": {
            "mean": sum(f1_scores) / max(len(f1_scores), 1),
            "median": sorted(f1_scores)[len(f1_scores)//2] if f1_scores else 0,
            "above_0.1": sum(1 for s in f1_scores if s > 0.1) / max(len(f1_scores), 1),
            "above_0.2": sum(1 for s in f1_scores if s > 0.2) / max(len(f1_scores), 1),
            "above_0.3": sum(1 for s in f1_scores if s > 0.3) / max(len(f1_scores), 1),
        },
        "specificity": {
            "mean": sum(metrics["specificity_scores"]) / max(len(metrics["specificity_scores"]), 1),
            "high_specificity_frac": sum(1 for s in metrics["specificity_scores"] if s > 0.5) / max(len(metrics["specificity_scores"]), 1),
        },
        "bioenergy_relevance": {
            "mean": sum(metrics["bioenergy_relevance_scores"]) / max(len(metrics["bioenergy_relevance_scores"]), 1),
        }
    }

    # Save
    with open(output_dir / "benchmark_analysis.json", 'w') as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "benchmark_detailed.json", 'w') as f:
        json.dump(metrics["predictions"], f, indent=2)

    # Print report
    print("\n" + "=" * 60)
    print("BENCHMARK ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total proteins analyzed: {summary['total_proteins']}")
    print(f"\nKeyword F1 Score:")
    print(f"  Mean: {summary['keyword_f1']['mean']:.3f}")
    print(f"  Median: {summary['keyword_f1']['median']:.3f}")
    print(f"  >0.1: {summary['keyword_f1']['above_0.1']:.1%}")
    print(f"  >0.2: {summary['keyword_f1']['above_0.2']:.1%}")
    print(f"  >0.3: {summary['keyword_f1']['above_0.3']:.1%}")
    print(f"\nSpecificity:")
    print(f"  Mean: {summary['specificity']['mean']:.3f}")
    print(f"  High specificity (>0.5): {summary['specificity']['high_specificity_frac']:.1%}")
    print(f"\nBioenergy Relevance:")
    print(f"  Mean: {summary['bioenergy_relevance']['mean']:.3f}")

    # Print top 5 best and worst predictions
    sorted_preds = sorted(metrics["predictions"], key=lambda x: x["keyword_f1"], reverse=True)
    print(f"\n--- TOP 5 BEST PREDICTIONS ---")
    for p in sorted_preds[:5]:
        print(f"  [{p['id']}] F1={p['keyword_f1']:.3f}")
        print(f"    Truth: {p['ground_truth'][:100]}")
        print(f"    Pred:  {p['predicted'][:100]}")

    print(f"\n--- TOP 5 WORST PREDICTIONS ---")
    for p in sorted_preds[-5:]:
        print(f"  [{p['id']}] F1={p['keyword_f1']:.3f}")
        print(f"    Truth: {p['ground_truth'][:100]}")
        print(f"    Pred:  {p['predicted'][:100]}")

    return summary


def analyze_dark_genome(results_file, output_dir):
    """Analyze dark genome annotation results."""
    results = load_jsonl(results_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    bioenergy_relevant = []
    enzyme_predictions = []

    enzyme_keywords = [
        "cellulase", "ligninase", "hemicellulase", "laccase", "peroxidase",
        "dehydrogenase", "reductase", "oxidase", "transferase", "synthase",
        "kinase", "phosphatase", "protease", "hydrolase", "lyase", "isomerase",
        "catalase", "nitrogenase", "deaminase", "dehalogenase"
    ]

    for r in results:
        predicted = r.get("Predicted", "")
        quality = assess_prediction_quality(predicted)

        pred_info = {
            "id": r.get("long_format_id", ""),
            "prompt": r.get("Prompt", ""),
            "predicted": predicted[:500],
            "specificity": quality["specificity"],
            "bioenergy_relevance": quality["bioenergy_relevance"],
            "confidence": quality["confidence"],
            "predicted_enzymes": [kw for kw in enzyme_keywords if kw in predicted.lower()]
        }
        predictions.append(pred_info)

        if quality["bioenergy_relevance"] > 0:
            bioenergy_relevant.append(pred_info)

        if pred_info["predicted_enzymes"]:
            enzyme_predictions.append(pred_info)

    summary = {
        "total_dark_proteins": len(results),
        "specific_predictions": sum(1 for p in predictions if p["specificity"] > 0.3),
        "bioenergy_relevant": len(bioenergy_relevant),
        "enzyme_predictions": len(enzyme_predictions),
        "enzyme_types_found": dict(Counter(
            e for p in enzyme_predictions for e in p["predicted_enzymes"]
        )),
        "avg_specificity": sum(p["specificity"] for p in predictions) / max(len(predictions), 1),
        "avg_confidence": sum(p["confidence"] for p in predictions) / max(len(predictions), 1),
    }

    with open(output_dir / "dark_genome_analysis.json", 'w') as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "dark_genome_predictions.json", 'w') as f:
        json.dump(predictions, f, indent=2)

    with open(output_dir / "bioenergy_discoveries.json", 'w') as f:
        json.dump(bioenergy_relevant, f, indent=2)

    print("\n" + "=" * 60)
    print("DARK GENOME ANNOTATION RESULTS")
    print("=" * 60)
    print(f"Total dark proteins: {summary['total_dark_proteins']}")
    print(f"Specific predictions (specificity > 0.3): {summary['specific_predictions']}")
    print(f"Bioenergy-relevant predictions: {summary['bioenergy_relevant']}")
    print(f"Enzyme predictions: {summary['enzyme_predictions']}")
    print(f"Enzyme types found: {summary['enzyme_types_found']}")
    print(f"Avg specificity: {summary['avg_specificity']:.3f}")
    print(f"Avg confidence: {summary['avg_confidence']:.3f}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["benchmark", "dark_genome", "both"], default="both")
    parser.add_argument("--benchmark_results",
                        default="/data/asahu/projects/doe_genesis/preliminary_data/benchmark/results/benchmark_results_all.jsonl")
    parser.add_argument("--dark_genome_results",
                        default="/data/asahu/projects/doe_genesis/preliminary_data/dark_genome/results/dark_genome_results_all.jsonl")
    parser.add_argument("--output_dir",
                        default="/data/asahu/projects/doe_genesis/preliminary_data/reports")
    args = parser.parse_args()

    if args.task in ("benchmark", "both"):
        if os.path.exists(args.benchmark_results):
            analyze_benchmark(args.benchmark_results, args.output_dir)
        else:
            print(f"Benchmark results not found: {args.benchmark_results}")

    if args.task in ("dark_genome", "both"):
        if os.path.exists(args.dark_genome_results):
            analyze_dark_genome(args.dark_genome_results, args.output_dir)
        else:
            print(f"Dark genome results not found: {args.dark_genome_results}")
