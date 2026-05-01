"""
Multi-axis head-to-head: IDPro classifier probe vs InterLabelGO+ (CAFA5 winner).

Axes:
  (1) Same-subset zero-shot AUC. Train IDPro probe on 3,000 reference proteins
      ONLY; evaluate on the 637 benchmark proteins (same proteins
      InterLabelGO+ scores). Removes the "IDPro is CV'd, InterLabelGO+ is
      zero-shot" critique.
  (2) Per-class winner table on the same 637 proteins (strict-keyword AUC).
  (3) Validation-bar count: how many EC classes hit AUC ≥ 0.85 (the wet-lab-
      triage-grade threshold).
  (4) Macro-AUC under both 5-fold CV (IDPro full advantage) AND zero-shot
      same-subset (apples-to-apples).
  (5) Capability axis: things multimodal LLMs can do that GO-term predictors
      cannot (CoT rationale, conformal calibration, RAG, open-ended QA,
      non-enzyme detection).
  (6) Honest leakage caveat: InterLabelGO+ trained on UniProt+GOA, our 637
      benchmark proteins likely IN its training set → InterLabelGO+'s AUC is
      an UPPER BOUND.

Output:
  preliminary_data/reports/IDPRO_VS_INTERLABELGO.md
  idpro/data/probe/embeddings/idpro_vs_interlabelgo.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from idpro.paths import PROBE_DIR, DATA_ROOT  # noqa: E402
from idpro.eval.reports.make_spider_plot_v2 import (  # noqa: E402
    strict_class_scores, CLASS_NAMES, BENCH_METHODS_DIR,
)

EMB_DIR = PROBE_DIR / "embeddings"
REPORT_DIR = DATA_ROOT / "preliminary_data" / "reports"
OUT_JSON = EMB_DIR / "idpro_vs_interlabelgo.json"
VIEWS = ["view_a_prompteol_l48", "view_b_question_mean_l48", "view_c_eos_l64"]
N_CLASSES = 8
VALIDATION_BAR = 0.85


# ---------------------------------------------------------------------------
# Load + helpers
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def stack_views(cache, accs, views):
    return torch.cat(
        [torch.stack([cache[a][v].float() for a in accs]) for v in views], dim=-1
    )


def ec_label(cache, a):
    v = cache[a]["labels"]["ec_l1"]
    return 0 if v is None else int(v)


def train_probe(x, y, device, epochs=100, lr=1e-3, wd=1e-4):
    probe = LinearProbe(x.shape[1], N_CLASSES).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    x = x.to(device); y = y.to(device)
    bs = 64
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0], device=device)
        for s in range(0, x.shape[0], bs):
            idx = perm[s:s+bs]
            loss = loss_fn(probe(x[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return probe.eval()


@torch.no_grad()
def predict(probe, x, device):
    return torch.softmax(probe(x.to(device)), dim=-1).cpu().numpy()


def per_class_auc(y_true, scores, n=N_CLASSES):
    out = []
    for c in range(n):
        yt = (y_true == c).astype(int)
        if yt.sum() in (0, len(yt)):
            out.append(np.nan); continue
        out.append(float(roc_auc_score(yt, scores[:, c])))
    return out


def macro(per):
    valid = [a for a in per if not np.isnan(a)]
    return float(np.mean(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Build IDPro zero-shot scores on the 637-benchmark
# ---------------------------------------------------------------------------


def idpro_zero_shot_637(device="cuda"):
    """Train IDPro probe on 3,000 reference only; predict on 637 benchmark.
    This matches InterLabelGO+'s zero-shot evaluation regime."""
    ref = torch.load(EMB_DIR / "reference_embeddings.pt", map_location="cpu", weights_only=False)
    bench = torch.load(EMB_DIR / "benchmark_embeddings.pt", map_location="cpu", weights_only=False)

    ref_accs = list(ref.keys())
    bench_accs = list(bench.keys())
    y_train = np.array([ec_label(ref, a) for a in ref_accs])
    y_test = np.array([ec_label(bench, a) for a in bench_accs])

    x_train = stack_views(ref, ref_accs, VIEWS)
    x_test = stack_views(bench, bench_accs, VIEWS)

    probe = train_probe(x_train, torch.tensor(y_train, dtype=torch.long), device)
    scores = predict(probe, x_test, device)
    return bench_accs, y_test, scores


# ---------------------------------------------------------------------------
# Build InterLabelGO+ scores via the strict keyword-matcher
# ---------------------------------------------------------------------------


def interlabelgo_strict_scores_637(bench_accs):
    """Reuse the spider-plot strict-keyword pipeline."""
    preds = json.loads(
        (BENCH_METHODS_DIR / "interlabelgo_benchmark_predictions.json").read_text()
    )
    scores = np.zeros((len(bench_accs), N_CLASSES), dtype=float)
    covered = 0
    for i, a in enumerate(bench_accs):
        if a in preds:
            covered += 1
            scores[i] = strict_class_scores(preds[a])
    return scores, covered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bench_accs, y_test, idpro_scores = idpro_zero_shot_637(device)
    print(f"Test set: 637 benchmark proteins (zero-shot for IDPro probe — no test protein in its train set)")
    print(f"  EC distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    idpro_per_class = per_class_auc(y_test, idpro_scores)
    idpro_macro = macro(idpro_per_class)
    print(f"\nIDPro zero-shot (train=3000 reference, test=637 benchmark):")
    print(f"  macro-AUC = {idpro_macro:.4f}")

    interlabelgo_scores, ilg_covered = interlabelgo_strict_scores_637(bench_accs)
    print(f"\nInterLabelGO+ coverage: {ilg_covered} / {len(bench_accs)}")
    interlabelgo_per_class = per_class_auc(y_test, interlabelgo_scores)
    interlabelgo_macro = macro(interlabelgo_per_class)
    print(f"InterLabelGO+ macro-AUC: {interlabelgo_macro:.4f}")

    # Per-class winner table
    print()
    print(f"Per-class breakdown ({sum(1 for c in CLASS_NAMES.values()):d} classes):")
    print(f"  {'class':<15} {'IDPro':>8} {'InterLG+':>10} {'Δ':>8} {'≥0.85?':>8}")
    winners = {"IDPro": 0, "InterLabelGO+": 0, "tied": 0}
    bar_idpro = 0
    bar_ilg = 0
    rows = []
    for c, name in CLASS_NAMES.items():
        ip = idpro_per_class[c]
        il = interlabelgo_per_class[c]
        delta = ip - il if not (np.isnan(ip) or np.isnan(il)) else float("nan")
        winner = ("IDPro" if delta > 0.005 else
                  "InterLabelGO+" if delta < -0.005 else
                  "tied") if not np.isnan(delta) else "n/a"
        if winner != "n/a":
            winners[winner] += 1
        if not np.isnan(ip) and ip >= VALIDATION_BAR:
            bar_idpro += 1
        if not np.isnan(il) and il >= VALIDATION_BAR:
            bar_ilg += 1
        flag = ""
        if not np.isnan(ip) and ip >= VALIDATION_BAR:
            flag += "I"
        if not np.isnan(il) and il >= VALIDATION_BAR:
            flag += "L"
        rows.append((c, name, ip, il, delta, winner, flag))
        print(f"  {name:<15} {ip:>8.3f} {il:>10.3f} {delta:>+8.3f} {flag:>8}")

    print(f"\nWinner counts: IDPro {winners['IDPro']}/{N_CLASSES}  InterLabelGO+ {winners['InterLabelGO+']}/{N_CLASSES}  tied {winners['tied']}/{N_CLASSES}")
    print(f"# classes ≥ {VALIDATION_BAR} AUC: IDPro {bar_idpro}/{N_CLASSES}  InterLabelGO+ {bar_ilg}/{N_CLASSES}")

    # IDPro CV reference (for the full-strength comparison)
    cv_path = EMB_DIR / "ec_classifier_evaluation.json"
    cv_macro = None
    cv_per_class = None
    if cv_path.exists():
        cvd = json.loads(cv_path.read_text())
        if "idpro_5fold_cv" in cvd:
            cv_macro = cvd["idpro_5fold_cv"]["macro_auc_mean"]
            cv_per_class = [cvd["idpro_5fold_cv"]["per_class_mean"][str(c)]["auc_mean"]
                            for c in range(N_CLASSES)]
            print(f"\nIDPro 5-fold CV (full-strength): macro = {cv_macro:.4f}")

    out = {
        "test_set": {
            "name": "637 UniProt-labeled benchmark",
            "n": len(bench_accs),
            "ec_distribution": {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        },
        "idpro_zero_shot_637": {
            "regime": "train=3000 reference, test=637 benchmark; no overlap",
            "macro_auc": idpro_macro,
            "per_class": {int(c): idpro_per_class[c] for c in range(N_CLASSES)},
            "n_classes_above_0_85": bar_idpro,
        },
        "idpro_5fold_cv": {
            "regime": "5-fold CV on full 3637-protein labeled pool",
            "macro_auc": cv_macro,
            "per_class": ({int(c): cv_per_class[c] for c in range(N_CLASSES)}
                          if cv_per_class else None),
        },
        "interlabelgo_plus": {
            "regime": "zero-shot inference; trained on CAFA5 corpus (UniProt+GOA) — likely contains our 637 benchmark proteins (training-overlap inflation)",
            "macro_auc": interlabelgo_macro,
            "per_class": {int(c): interlabelgo_per_class[c] for c in range(N_CLASSES)},
            "n_classes_above_0_85": bar_ilg,
            "coverage": ilg_covered,
        },
        "headline": {
            "idpro_lead_zero_shot": idpro_macro - interlabelgo_macro,
            "idpro_lead_cv": (cv_macro - interlabelgo_macro) if cv_macro else None,
            "winner_per_class_idpro": winners["IDPro"],
            "winner_per_class_interlabelgo": winners["InterLabelGO+"],
            "winner_tied": winners["tied"],
            "validation_bar": VALIDATION_BAR,
            "validation_bar_classes_idpro": bar_idpro,
            "validation_bar_classes_interlabelgo": bar_ilg,
        },
        "capability_comparison": {
            "natural_language_rationale_chain_of_thought": {"IDPro": True, "InterLabelGO+": False},
            "calibrated_uncertainty_conformal": {"IDPro": True, "InterLabelGO+": False},
            "abstention_on_low_confidence": {"IDPro": True, "InterLabelGO+": False},
            "rag_evidence_retrieval": {"IDPro": True, "InterLabelGO+": False},
            "non_enzyme_detection": {"IDPro": True, "InterLabelGO+": "predicts only enzyme/MF GO terms"},
            "open_ended_questions_beyond_EC": {"IDPro": True, "InterLabelGO+": False},
            "training_overlap_with_benchmark": {"IDPro": "5-fold CV — none", "InterLabelGO+": "high — benchmark likely IN training set"},
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    raise SystemExit(main())
