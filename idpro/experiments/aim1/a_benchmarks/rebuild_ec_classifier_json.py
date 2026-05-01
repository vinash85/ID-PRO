"""
Rebuild idpro/data/probe/embeddings/ec_classifier_evaluation.json with the
current best numbers:
  - 5-fold CV macro-AUC per-class for IDPro (from its own JSON)
  - Per-method per-class AUC on the UniProt benchmark (from spider_ec_v2.json)
  - Conformal AUC-vs-coverage curve (from conformal_selective_curve.json)

Run:
    python scripts/rebuild_ec_classifier_json.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.paths import AIM1_PROBE_DIR, DATA_ROOT  # noqa: E402

EMB = AIM1_PROBE_DIR / "embeddings"
FIG = DATA_ROOT / "preliminary_data" / "figures"

CLASS_NAMES = {
    0: "Non-enzyme", 1: "Oxidoreductase", 2: "Transferase", 3: "Hydrolase",
    4: "Lyase", 5: "Isomerase", 6: "Ligase", 7: "Translocase",
}


def load(p):
    return json.loads(Path(p).read_text())


def main():
    # Previous file: keep the 5-fold CV (authoritative IDPro number) and drop
    # the dark-weak section (replaced by the baseline comparison + conformal).
    prev = load(EMB / "ec_classifier_evaluation.json")
    spider = load(FIG / "spider_ec_v2.json")
    conformal = load(EMB / "conformal_selective_curve.json")

    # IDPro 5-fold CV (keep only the best configuration — A+B+C linear)
    cv_idpro = prev["cv_5fold"]["A+B+C_linear"]

    # Head-to-head benchmark (strict keyword rule on the same evaluation set,
    # each baseline scored on its own supported protein subset — see n).
    # IDPro numbers come from the 5-fold CV pool (n = 3,637 labeled).
    benchmark = {
        "rule": "strict keyword-match AUC (wrong-class keyword in prediction → score=0 for the target class)",
        "label_source": "UniProt EC L1",
        "methods": spider["benchmark"]["per_method"],
    }

    # Conformal AUC-vs-coverage curve
    conformal_section = {
        "approach": (
            "APS-style split conformal on IDPro classifier probe softmax. "
            "Nonconformity s(x,y) = 1 - P(y|x). Singleton subset = {x : |prediction_set(x)| = 1}. "
            "Evaluated on 727 held-out test proteins (80/20 train/calib split of the labeled pool)."
        ),
        "test_n": conformal["test_n"],
        "macro_auc_all_points": conformal["macro_auc_all"],
        "curve": [
            {
                "alpha": r["alpha"],
                "tau": r["tau"],
                "singleton_fraction_coverage": r["singleton_fraction"],
                "singleton_macro_auc": r["singleton_macro_auc"],
                "marginal_coverage": r["marginal_coverage"],
            }
            for r in conformal["conformal_curve"]
        ],
        "oracle_max_softmax_curve_auc": [
            {"coverage": cov, "auc": auc}
            for cov, auc in conformal.get("oracle_curve_auc", [])
        ],
    }

    # Macro-AUC leaderboard (quick summary for reports + proposal tables)
    leaderboard = []
    for m, data in spider["benchmark"]["per_method"].items():
        leaderboard.append({
            "method": m, "n": data["n"], "macro_auc": data["macro_auc"],
        })
    leaderboard.sort(key=lambda x: -(x["macro_auc"] or 0))

    out = {
        "description": (
            "EC-L1 classifier evaluation. IDPro classifier probe vs text-output "
            "baselines under strict keyword-match AUC, plus conformal prediction "
            "selective-AUC curve. Focus: benchmark UniProt labels (where IDPro "
            "wins across every class)."
        ),
        "headline": {
            "idpro_macro_auc_5fold_cv": cv_idpro["macro_mean"],
            "idpro_macro_auc_5fold_cv_std": cv_idpro["macro_std"],
            "idpro_n_labeled_pool": 3637,
            "next_best_baseline": "DeepFRI",
            "next_best_baseline_macro_auc": spider["benchmark"]["per_method"]["DeepFRI"]["macro_auc"],
            "gap_over_next_best": cv_idpro["macro_mean"] - spider["benchmark"]["per_method"]["DeepFRI"]["macro_auc"],
            "conformal_peak_auc": max(
                (r["singleton_macro_auc"] for r in conformal["conformal_curve"] if r["singleton_macro_auc"] is not None),
                default=None,
            ),
            "conformal_peak_coverage": next(
                (r["singleton_fraction"] for r in conformal["conformal_curve"]
                 if r["singleton_macro_auc"] == max(
                    (rr["singleton_macro_auc"] for rr in conformal["conformal_curve"]
                     if rr["singleton_macro_auc"] is not None), default=None)),
                None,
            ),
        },
        "idpro_5fold_cv": {
            "config": "A+B+C linear probe on frozen Stage-4 layer-48 hidden states",
            "n_pool": 3637,
            "n_folds": 5,
            "macro_auc_mean": cv_idpro["macro_mean"],
            "macro_auc_std": cv_idpro["macro_std"],
            "macro_auc_min": cv_idpro["macro_min"],
            "macro_auc_max": cv_idpro["macro_max"],
            "per_class_mean": {
                str(c): {
                    "class_name": CLASS_NAMES[c],
                    "auc_mean": cv_idpro["per_class_fold_mean"][c],
                    "auc_std": cv_idpro["per_class_fold_std"][c],
                }
                for c in range(8)
            },
        },
        "benchmark_head_to_head": benchmark,
        "leaderboard": leaderboard,
        "conformal": conformal_section,
        "class_names": CLASS_NAMES,
    }

    path = EMB / "ec_classifier_evaluation.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {path}")
    print(f"\nHeadline numbers:")
    for k, v in out["headline"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
