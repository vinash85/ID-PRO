"""
What does an F1 score mean for protein-function predictions?

Token-F1 between a free-text prediction and a free-text ground truth has no
fixed scale: two correct descriptions of the same protein can have low overlap
because of paraphrase. This script anchors the metric by computing four
reference distributions on the 669-protein benchmark, then cross-checks against
*label-level AUC* on EC class and functional category — the operational signal
that actually drives experimental triage.

Inputs:
  BASELINE_PREDS_DIR/p2t_benchmark_predictions.jsonl   (P2T natural-language output)

Outputs (written to REPORTS_DIR):
  - f1_meaning_results.json   : machine-readable numbers
  - F1_MEANING_RESULTS.md     : human report
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.model.idpro.conformal import token_f1  # noqa: E402
from idpro.paths import BASELINE_PREDS_DIR, REPORTS_DIR  # noqa: E402

BENCHMARK_PATH = BASELINE_PREDS_DIR / "p2t_benchmark_predictions.jsonl"
REPORT_DIR = REPORTS_DIR

EC_PAT = re.compile(r"\b(\d+)\.(\d+)\.(\d+)\.(\d+)\b")

# Functional category lexicon. Keep small + canonical so the labels are robust.
CATEGORY_KEYWORDS = {
    "oxidoreductase": ["oxidoreductase", "oxidation", "reductase", "dehydrogenase"],
    "transferase": ["transferase", "kinase", "methyltransferase", "acetyltransferase"],
    "hydrolase": ["hydrolase", "hydrolysis", "endohydrolysis", "peptidase", "protease"],
    "transport": ["transport", "transporter", "permease", "channel", "import", "export"],
    "binding_metal": ["iron", "zinc", "copper", "magnesium", "manganese", "nickel"],
    "binding_nucleotide": ["atp", "gtp", "nadh", "nadph", "fad", "coenzyme"],
    "membrane": ["membrane", "transmembrane", "lipid bilayer"],
    "dna_rna": ["dna", "rna", "ribosom", "transcription", "translation"],
}


# ---------------------------------------------------------------------------
# Loading + EC extraction
# ---------------------------------------------------------------------------


def load_rows() -> list:
    rows = []
    with BENCHMARK_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_gt"] = r.get("Ground Truth") or ""
            r["_pred"] = r.get("Predicted") or ""
            r["_ec"] = extract_ec(r["_gt"])
            r["_ec_class"] = r["_ec"][0] if r["_ec"] else None
            r["_pred_ec"] = extract_ec(r["_pred"])
            r["_pred_ec_class"] = r["_pred_ec"][0] if r["_pred_ec"] else None
            rows.append(r)
    return rows


def extract_ec(text: str):
    m = EC_PAT.search(text or "")
    if not m:
        return None
    return tuple(int(x) for x in m.groups())


def label_categories(text: str) -> dict:
    t = (text or "").lower()
    return {cat: int(any(kw in t for kw in kws)) for cat, kws in CATEGORY_KEYWORDS.items()}


def category_score(text: str) -> dict:
    """Soft signal: count of keyword hits per category (>=0)."""
    t = (text or "").lower()
    return {cat: sum(t.count(kw) for kw in kws) for cat, kws in CATEGORY_KEYWORDS.items()}


# ---------------------------------------------------------------------------
# Reference F1 distributions
# ---------------------------------------------------------------------------


def f1_distribution_pairs(pairs: list, max_pairs: int = 5_000) -> np.ndarray:
    """Compute F1 over a list of (a, b) text pairs (capped for speed)."""
    if len(pairs) > max_pairs:
        idx = np.random.default_rng(0).choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    return np.array([token_f1(a, b) for a, b in pairs], dtype=float)


def reference_distributions(rows: list, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(rows)

    # Model F1: matched (pred_i, gt_i)
    model = np.array([token_f1(r["_pred"], r["_gt"]) for r in rows])

    # Floor 1 — random pairing of pred and gt across proteins
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    floor_random = np.array([token_f1(rows[i]["_pred"], rows[perm[i]]["_gt"]) for i in range(n)])

    # Group by EC class
    by_class = defaultdict(list)
    for r in rows:
        if r["_ec_class"] is not None:
            by_class[r["_ec_class"]].append(r["_gt"])

    # Ceiling — within-class GT-vs-GT pairs (same enzyme family)
    within_pairs = []
    for cls, gts in by_class.items():
        if len(gts) < 2:
            continue
        # Sample up to 200 pairs per class to bound compute.
        for _ in range(min(200, len(gts) * (len(gts) - 1) // 2)):
            i, j = rng.choice(len(gts), 2, replace=False)
            within_pairs.append((gts[i], gts[j]))
    ceiling_within = f1_distribution_pairs(within_pairs)

    # Floor 2 — across-class GT-vs-GT pairs (different enzyme family)
    classes = sorted(by_class.keys())
    across_pairs = []
    for _ in range(min(2_000, n * 5)):
        c1, c2 = rng.choice(classes, 2, replace=False)
        a = rng.choice(by_class[c1])
        b = rng.choice(by_class[c2])
        across_pairs.append((a, b))
    floor_diff_class = f1_distribution_pairs(across_pairs)

    return {
        "model": model,
        "ceiling_within_class": ceiling_within,
        "floor_diff_class": floor_diff_class,
        "floor_random": floor_random,
    }


# ---------------------------------------------------------------------------
# Label-level AUC (the "did we get the family right?" signal)
# ---------------------------------------------------------------------------


def auc_per_category(rows: list) -> dict:
    """For each functional category, compute AUC of category_score(pred) vs binary label in GT."""
    out = {}
    for cat in CATEGORY_KEYWORDS:
        y_true = np.array([label_categories(r["_gt"])[cat] for r in rows])
        y_score = np.array([category_score(r["_pred"])[cat] for r in rows], dtype=float)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            out[cat] = {"auc": None, "prevalence": float(y_true.mean()), "n_pos": int(y_true.sum())}
            continue
        out[cat] = {
            "auc": float(roc_auc_score(y_true, y_score)),
            "prevalence": float(y_true.mean()),
            "n_pos": int(y_true.sum()),
        }
    return out


def auc_ec_class_one_vs_rest(rows: list) -> dict:
    """For each EC class digit (1-7), AUC of (pred has class c) vs (gt has class c)."""
    out = {}
    has_gt = [r for r in rows if r["_ec_class"] is not None]
    if not has_gt:
        return out
    for cls in range(1, 8):
        y_true = np.array([1 if r["_ec_class"] == cls else 0 for r in has_gt])
        # Soft score: 1 if predicted EC class matches, else 0.5 if EC mentioned at all, else 0.
        y_score = np.array([
            1.0 if r["_pred_ec_class"] == cls
            else (0.5 if r["_pred_ec_class"] is not None else 0.0)
            for r in has_gt
        ])
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            out[str(cls)] = {"auc": None, "prevalence": float(y_true.mean())}
            continue
        out[str(cls)] = {
            "auc": float(roc_auc_score(y_true, y_score)),
            "prevalence": float(y_true.mean()),
            "n_pos": int(y_true.sum()),
        }
    return out


def ec_class_accuracy(rows: list) -> dict:
    """Top-1 EC class agreement, restricted to rows where both GT and prediction mention an EC."""
    both = [r for r in rows if r["_ec_class"] and r["_pred_ec_class"]]
    if not both:
        return {"n": 0, "accuracy": None}
    correct = sum(1 for r in both if r["_ec_class"] == r["_pred_ec_class"])
    return {"n": len(both), "accuracy": correct / len(both)}


# ---------------------------------------------------------------------------
# Simulation: F1 vs prediction-quality dial
# ---------------------------------------------------------------------------


def simulate_quality_trajectory(rows: list, levels: list, seed: int = 0) -> dict:
    """
    Build synthetic predictions at controlled quality levels.

    quality q in [0, 1]:
      with prob q,   keep original GT word (perfect substring)
      with prob 1-q, replace with a random word drawn from the corpus

    For each level we also extract the same labels and report AUC on EC class,
    so we can compare F1 trajectory vs AUC trajectory.
    """
    rng = np.random.default_rng(seed)
    corpus_words = []
    for r in rows:
        corpus_words.extend(re.findall(r"[a-zA-Z]+", r["_gt"]))
    corpus_words = np.array(corpus_words)

    out = []
    for q in levels:
        synth = []
        for r in rows:
            words = re.findall(r"[a-zA-Z]+", r["_gt"])
            if not words:
                synth.append("")
                continue
            mask = rng.random(len(words)) < q
            replaced = [w if m else rng.choice(corpus_words) for w, m in zip(words, mask)]
            # Keep EC mention with probability q (so AUC tracks quality too)
            ec_str = ""
            if r["_ec"] and rng.random() < q:
                ec_str = " EC " + ".".join(str(x) for x in r["_ec"])
            synth.append(" ".join(replaced) + ec_str)
        f1s = np.array([token_f1(s, r["_gt"]) for s, r in zip(synth, rows)])
        # Label AUC at this quality level
        synth_rows = [
            {**r, "_pred": s,
             "_pred_ec": extract_ec(s),
             "_pred_ec_class": (extract_ec(s) or (None,))[0]}
            for r, s in zip(rows, synth)
        ]
        cat_auc = auc_per_category(synth_rows)
        ec_auc = auc_ec_class_one_vs_rest(synth_rows)
        cat_mean = np.mean([v["auc"] for v in cat_auc.values() if v["auc"] is not None])
        ec_mean = np.mean([v["auc"] for v in ec_auc.values() if v["auc"] is not None])
        out.append({
            "quality": q,
            "f1_mean": float(f1s.mean()),
            "f1_median": float(np.median(f1s)),
            "cat_auc_mean": float(cat_mean),
            "ec_auc_mean": float(ec_mean),
        })
    return out


# ---------------------------------------------------------------------------
# Actionability rule
# ---------------------------------------------------------------------------


def actionability(
    model_f1: np.ndarray,
    ceiling: np.ndarray,
    floor: np.ndarray,
    cat_auc_mean: float,
    sim: list,
) -> dict:
    """
    Define an experimentally-actionable threshold by anchoring to the
    same-EC-class GT-vs-GT ceiling.

      normalized_F1 = (F1 - floor_med) / (ceiling_med - floor_med)
        in [0, 1]: 0 = chance, 1 = paraphrase ceiling.

      "Actionable" rule (Phase I target):
        F1 >= 0.30   AND   cat_AUC >= 0.65
        — empirically anchored: simulation shows F1=0.30 → cat_AUC ~0.65 and
          F1=0.40 → cat_AUC ~0.74. This brackets the band where label-level
          signal is detectable above chance and triage becomes useful.
    """
    ceil_med = float(np.median(ceiling))
    floor_med = float(np.median(floor))
    model_med = float(np.median(model_f1))
    norm = (model_med - floor_med) / max(1e-9, ceil_med - floor_med)
    target_f1 = 0.30
    target_cat_auc = 0.65
    frac_above_target = float((model_f1 >= target_f1).mean())

    # Find the simulation quality where F1 first crosses target_f1
    crossing = next(
        (s["quality"] for s in sim if s["f1_mean"] >= target_f1),
        None,
    )
    cat_at_crossing = next(
        (s["cat_auc_mean"] for s in sim if s["f1_mean"] >= target_f1),
        None,
    )
    return {
        "ceiling_median": ceil_med,
        "floor_median": floor_med,
        "model_median": model_med,
        "model_F1_normalized_to_ceiling": float(norm),
        "fraction_predictions_above_actionable_F1": frac_above_target,
        "target_F1_for_actionable": target_f1,
        "target_category_AUC_for_actionable": target_cat_auc,
        "model_category_AUC": float(cat_auc_mean),
        "simulation_quality_crossing_target_F1": crossing,
        "simulation_cat_AUC_at_crossing": cat_at_crossing,
        "current_model_meets_actionable": bool(
            model_med >= target_f1 and cat_auc_mean >= target_cat_auc
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def fmt_dist(name: str, x: np.ndarray) -> str:
    return (f"  {name:<24} n={len(x):4d}  mean={x.mean():.3f}  "
            f"median={np.median(x):.3f}  p25={np.percentile(x, 25):.3f}  "
            f"p75={np.percentile(x, 75):.3f}")


def main() -> int:
    rows = load_rows()
    print(f"Loaded {len(rows)} benchmark rows")
    print(f"  with EC class: {sum(1 for r in rows if r['_ec_class'])}")
    print(f"  prediction mentions EC: {sum(1 for r in rows if r['_pred_ec_class'])}")
    print()

    # 1. Reference distributions
    refs = reference_distributions(rows)
    print("F1 reference distributions")
    for name, arr in refs.items():
        print(fmt_dist(name, arr))
    print()

    # 2. Label-level AUC
    cat_auc = auc_per_category(rows)
    ec_auc = auc_ec_class_one_vs_rest(rows)
    ec_acc = ec_class_accuracy(rows)
    print("Functional-category AUC (pred score vs GT label)")
    for cat, v in cat_auc.items():
        if v["auc"] is None:
            print(f"  {cat:<22}  AUC=  N/A   prev={v['prevalence']:.2f}")
        else:
            print(f"  {cat:<22}  AUC={v['auc']:.3f}  prev={v['prevalence']:.2f}  n_pos={v['n_pos']}")
    cat_auc_mean = float(np.mean([v["auc"] for v in cat_auc.values() if v["auc"] is not None]))
    print(f"  {'mean':<22}  AUC={cat_auc_mean:.3f}")
    print()
    print("EC-class AUC (one-vs-rest over digits 1-7)")
    for cls, v in ec_auc.items():
        if v["auc"] is None:
            print(f"  class {cls}                AUC=  N/A   prev={v['prevalence']:.2f}")
        else:
            print(f"  class {cls}                AUC={v['auc']:.3f}  prev={v['prevalence']:.2f}  n_pos={v['n_pos']}")
    ec_auc_mean = float(np.mean([v["auc"] for v in ec_auc.values() if v["auc"] is not None]))
    print(f"  {'mean':<22}  AUC={ec_auc_mean:.3f}")
    print(f"  EC top-1 accuracy (rows with both EC mentions): "
          f"{ec_acc['accuracy']} on n={ec_acc['n']}")
    print()

    # 3. Simulation
    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sim = simulate_quality_trajectory(rows, levels)
    print("Simulation: F1 + AUC vs synthetic prediction quality")
    print(f"  {'q':>4}  {'F1_mean':>8}  {'F1_med':>8}  {'cat_AUC':>8}  {'ec_AUC':>8}")
    for s in sim:
        print(f"  {s['quality']:>4.1f}  {s['f1_mean']:>8.3f}  "
              f"{s['f1_median']:>8.3f}  {s['cat_auc_mean']:>8.3f}  "
              f"{s['ec_auc_mean']:>8.3f}")
    print()

    # 4. Actionability
    act = actionability(
        refs["model"],
        refs["ceiling_within_class"],
        refs["floor_random"],
        cat_auc_mean,
        sim,
    )
    print("Actionability check")
    for k, v in act.items():
        print(f"  {k:<32}  {v}")
    print()

    # Save
    REPORT_DIR.mkdir(exist_ok=True)
    out_json = {
        "n_rows": len(rows),
        "reference_distributions": {
            k: {
                "n": int(len(v)),
                "mean": float(v.mean()),
                "median": float(np.median(v)),
                "p25": float(np.percentile(v, 25)),
                "p75": float(np.percentile(v, 75)),
            }
            for k, v in refs.items()
        },
        "category_auc": cat_auc,
        "category_auc_mean": cat_auc_mean,
        "ec_class_auc": ec_auc,
        "ec_class_auc_mean": ec_auc_mean,
        "ec_class_top1_accuracy": ec_acc,
        "simulation": sim,
        "actionability": act,
    }
    out_path = REPORT_DIR / "f1_meaning_results.json"
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
