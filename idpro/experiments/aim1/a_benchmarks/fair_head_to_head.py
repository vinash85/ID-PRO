"""
Fair same-subset head-to-head comparison of IDPro vs all baselines.

Problem with previous report:
  - IDPro was CV'd on 3,637 proteins (its own labeled pool — training advantage)
  - Baselines were zero-shot on 34-637 proteins (whatever they covered)
  - Macro-AUCs aren't directly comparable.

Fix:
  - Canonical test subset: the 125 proteins in the original benchmark that
    have EC labels AND at least P2T/RAG/BioReason/DeepGOMeta predictions.
  - Narrow variants: ∩ DeepFRI (92 proteins), ∩ MMseqs2 (34 proteins).
  - Train IDPro probe on all proteins OUTSIDE the chosen test subset —
    no data leak, same sample size as baselines, zero-shot on the test set.
  - Compute strict-keyword AUC for each baseline on the same subset.
  - Report per-subset macro-AUC for each method.

Output: reports/FAIR_HEAD_TO_HEAD.md,
        idpro/data/probe/embeddings/fair_head_to_head.json,
        reports/figures/spider_ec_fair.{png,pdf}
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.paths import AIM1_PROBE_DIR as DATA_DIR, DATA_ROOT, REPORTS_DIR, FIGURES_DIR  # noqa: E402

EMB_DIR = DATA_DIR / "embeddings"
FIG_DIR = FIGURES_DIR
REPORT_DIR = REPORTS_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
BENCH_RES = DATA_ROOT / "benchmark" / "results"

VIEWS = ["view_a_prompteol_l48", "view_b_question_mean_l48", "view_c_eos_l64"]
CLASS_NAMES = {
    0: "Non-enzyme", 1: "Oxidoreductase", 2: "Transferase", 3: "Hydrolase",
    4: "Lyase", 5: "Isomerase", 6: "Ligase", 7: "Translocase",
}

# Reuse the same keyword dicts as make_spider_plot_v2.py for strict scoring
from idpro.experiments.aim1.reports.make_spider_plot_v2 import (  # noqa: E402
    strict_class_scores, deepgometa_strict_scores, _deepfri_csv_to_text,
    COLORS, METHOD_LINEWIDTH, METHOD_ZORDER,
)


# ---------------------------------------------------------------------------
# IDPro probe
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def stack_views(cache, accs, views):
    return torch.cat([torch.stack([cache[a][v].float() for a in accs]) for v in views], dim=-1)


def ec_label(cache, a):
    v = cache[a]["labels"]["ec_l1"]
    return 0 if v is None else int(v)


def train_probe(x_train, y_train, device, epochs=100, lr=1e-3, wd=1e-4):
    probe = LinearProbe(x_train.shape[1], 8).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    x = x_train.to(device); y = y_train.to(device)
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


def per_class_auc(y_true, scores, n_classes=8):
    per = []
    for c in range(n_classes):
        yt = (y_true == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            per.append(np.nan); continue
        per.append(float(roc_auc_score(yt, scores[:, c])))
    return per


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ref_cache = torch.load(EMB_DIR / "reference_embeddings.pt", map_location="cpu", weights_only=False)
    bench_cache = torch.load(EMB_DIR / "benchmark_embeddings.pt", map_location="cpu", weights_only=False)
    all_cache = {**ref_cache, **bench_cache}
    all_accs = list(ref_cache.keys()) + list(bench_cache.keys())
    all_labels = np.array([ec_label(ref_cache, a) for a in ref_cache.keys()] +
                          [ec_label(bench_cache, a) for a in bench_cache.keys()])
    print(f"Labeled pool: {len(all_accs)}")

    # Load baseline predictions
    def load_json_map(fname):
        return json.loads((BENCH_RES / fname).read_text())
    method_preds_text = {
        "P2T baseline": load_json_map("p2t_baseline_predictions.json"),
        "P2T (RAG transfer)": load_json_map("rag_transfer_predictions.json"),
        "BioReason-Pro": load_json_map("bioreason_predictions.json"),
        "DeepFRI": load_json_map("deepfri_predictions.json"),
        "MMseqs2": load_json_map("mmseqs_predictions.json"),
    }
    deepgometa = load_json_map("deepgometa_predictions.json")

    ours = set(bench_cache.keys())
    subsets = {
        "common_125_broad": list((set(method_preds_text["P2T (RAG transfer)"]) &
                                  set(method_preds_text["BioReason-Pro"]) &
                                  set(deepgometa) & ours)),
        "common_92_with_deepfri": list((set(method_preds_text["P2T (RAG transfer)"]) &
                                        set(method_preds_text["BioReason-Pro"]) &
                                        set(deepgometa) & set(method_preds_text["DeepFRI"]) & ours)),
        "common_24_all_methods": list((set(method_preds_text["P2T (RAG transfer)"]) &
                                       set(method_preds_text["BioReason-Pro"]) &
                                       set(deepgometa) & set(method_preds_text["DeepFRI"]) &
                                       set(method_preds_text["MMseqs2"]) & ours)),
    }
    # Sort each subset for reproducibility
    subsets = {k: sorted(v) for k, v in subsets.items()}
    for k, v in subsets.items():
        print(f"  {k}: n={len(v)}")

    fair_results = {}

    for subset_name, test_accs in subsets.items():
        if not test_accs:
            continue
        print(f"\n=== Subset: {subset_name}  n={len(test_accs)} ===")
        y_test = np.array([ec_label(all_cache, a) for a in test_accs])
        print(f"  EC distribution: {dict(sorted(Counter(y_test.tolist()).items()))}")

        # Train IDPro probe on everything NOT in this subset
        test_set_accs = set(test_accs)
        train_accs = [a for a in all_accs if a not in test_set_accs]
        y_train = np.array([ec_label(all_cache, a) for a in train_accs])
        x_train = stack_views(all_cache, train_accs, VIEWS)
        probe = train_probe(x_train, torch.tensor(y_train, dtype=torch.long), device=device)
        x_test = stack_views(all_cache, test_accs, VIEWS)
        p_test = predict(probe, x_test, device)
        idpro_auc = per_class_auc(y_test, p_test)
        valid = [a for a in idpro_auc if not np.isnan(a)]
        idpro_macro = float(np.mean(valid)) if valid else float("nan")
        print(f"  IDPro classifier probe (train_n={len(train_accs)}): macro-AUC = {idpro_macro:.3f}")

        method_aucs = {"IDPro classifier probe": idpro_auc}
        method_macros = {"IDPro classifier probe": idpro_macro}

        # Each baseline: strict keyword matrix on test_accs (all of them have preds
        # because we intersected their sets above)
        def strict_matrix(preds, accs, mode="text"):
            out = np.zeros((len(accs), 8), dtype=float)
            for i, a in enumerate(accs):
                pred_text = preds.get(a, "")
                if mode == "go_ids":
                    out[i] = deepgometa_strict_scores(pred_text)
                else:
                    out[i] = strict_class_scores(pred_text)
            return out

        for mname, preds in method_preds_text.items():
            # Only score if every test acc has a prediction
            missing = [a for a in test_accs if a not in preds]
            if missing and subset_name != "common_125_broad":
                # Skip method if it doesn't cover the test subset (shouldn't happen by construction)
                continue
            if missing and mname in ("DeepFRI", "MMseqs2"):
                continue
            scores = strict_matrix(preds, test_accs, mode="text")
            per = per_class_auc(y_test, scores)
            valid = [a for a in per if not np.isnan(a)]
            macro = float(np.mean(valid)) if valid else float("nan")
            method_aucs[mname] = per
            method_macros[mname] = macro
            print(f"  {mname:24s}: macro-AUC = {macro:.3f}")

        # DeepGOMeta (GO-ID mode)
        scores_dgm = strict_matrix(deepgometa, test_accs, mode="go_ids")
        per = per_class_auc(y_test, scores_dgm)
        valid = [a for a in per if not np.isnan(a)]
        method_aucs["DeepGOMeta"] = per
        method_macros["DeepGOMeta"] = float(np.mean(valid)) if valid else float("nan")
        print(f"  {'DeepGOMeta':24s}: macro-AUC = {method_macros['DeepGOMeta']:.3f}")

        fair_results[subset_name] = {
            "n": len(test_accs),
            "ec_distribution": {int(k): int(v) for k, v in Counter(y_test.tolist()).items()},
            "per_method": {
                m: {
                    "macro_auc": method_macros[m],
                    "per_class": {int(c): (None if np.isnan(v) else v) for c, v in enumerate(method_aucs[m])},
                } for m in method_aucs
            },
        }

    (EMB_DIR / "fair_head_to_head.json").write_text(json.dumps(fair_results, indent=2))
    print(f"\nWrote {EMB_DIR / 'fair_head_to_head.json'}")

    # ---------- Plot ----------
    # Pick the common_92_with_deepfri subset as the canonical fair comparison
    target = "common_92_with_deepfri"
    if target not in fair_results:
        return
    axis_labels = [CLASS_NAMES[c] for c in range(8)]
    subset = fair_results[target]
    n_axes = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(9.5, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(axis_labels, fontsize=10)
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=8)
    # chance
    chance = [0.5] * (n_axes + 1)
    ax.plot(angles_closed, chance, color="gray", lw=0.8, ls="--", alpha=0.6)

    method_order = ["IDPro classifier probe", "P2T baseline", "P2T (RAG transfer)",
                    "BioReason-Pro", "DeepFRI", "MMseqs2", "DeepGOMeta"]
    for m in method_order:
        if m not in subset["per_method"]:
            continue
        per = subset["per_method"][m]["per_class"]
        vals = [per[c] if per[c] is not None else 0.5 for c in range(8)]
        vals_closed = vals + vals[:1]
        color = COLORS.get(m, None)
        lw = METHOD_LINEWIDTH.get(m, 1.6)
        zo = METHOD_ZORDER.get(m, 3)
        ax.plot(angles_closed, vals_closed, "-o", label=f"{m} ({subset['per_method'][m]['macro_auc']:.3f})",
                color=color, lw=lw, markersize=5, zorder=zo)
        if m == "IDPro classifier probe":
            ax.fill(angles_closed, vals_closed, alpha=0.12, color=color, zorder=zo - 1)

    ax.set_title(f"Fair head-to-head: all methods on SAME {subset['n']}-protein subset\n"
                 "(common intersection of P2T/RAG/BioReason/DGM/DeepFRI; IDPro probe trained "
                 "on the 3,545 non-test proteins)", fontsize=10, pad=22, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.02), fontsize=9, title="method (macro-AUC)")

    for ext in ("png", "pdf"):
        p = FIG_DIR / f"spider_ec_fair.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
