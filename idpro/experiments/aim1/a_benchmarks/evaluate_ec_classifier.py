"""
Rigorous EC-L1 classifier evaluation:

  (A) 5-fold stratified CV on the full labeled pool (3,000 reference +
      637 benchmark = 3,637 proteins with UniProt EC labels) — reports
      macro-AUC, per-class AUC, macro-AUC mean ± std across folds.

  (B) Final probe (trained on full labeled pool) applied to dark-genome
      proteins with WEAK EC labels derived from GO molecular-function
      activity keywords. Scores AUC on 209 dark proteins (121 with specific
      enzyme class + 88 predicted non-enzyme).

Weak EC labels are noisy (InterProScan / GO-caller annotation, not
experimental), so dark AUCs are lower bounds on actual probe performance.

Run:
    python scripts/evaluate_ec_classifier.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.paths import AIM1_PROBE_DIR as DATA_DIR, DARK_GENOME_META  # noqa: E402

EMB_DIR = DATA_DIR / "embeddings"
DARK_META = DARK_GENOME_META

VIEWS = [
    "view_a_prompteol_l48",
    "view_b_question_mean_l48",
    "view_c_eos_l64",
]


# ---------------------------------------------------------------------------
# Weak EC label derivation for dark proteins
# ---------------------------------------------------------------------------

EC_KEYWORDS = {
    1: ["oxidoreductase activity", "dehydrogenase", "reductase", "oxidase", "oxygenase", "peroxidase"],
    2: ["transferase activity", "kinase activity", "methyltransferase", "acyltransferase", "glycosyltransferase"],
    3: ["hydrolase activity", "peptidase", "protease", "nuclease", "phosphatase", "esterase", "lipase", "glycosidase"],
    4: ["lyase activity", "decarboxylase", "aldolase", "dehydratase", "synthase"],
    5: ["isomerase activity", "racemase", "epimerase", "mutase"],
    6: ["ligase activity", "synthetase"],
    7: ["transporter activity", "transmembrane transport", "channel activity", "permease"],
}


def weak_ec_l1_from_go(go_terms: str) -> "int | None":
    """Return EC L1 (0..7) or None if ambiguous / no annotation."""
    if not go_terms:
        return None
    lo = go_terms.lower()
    matched = set()
    for ec, kws in EC_KEYWORDS.items():
        if any(kw in lo for kw in kws):
            matched.add(ec)
    if len(matched) == 0:
        return 0  # No enzyme activity term → non-enzyme (confident)
    if len(matched) == 1:
        return next(iter(matched))
    return None  # Conflicting multiple classes — exclude


def build_dark_ec_labels() -> dict:
    labels = {}
    with DARK_META.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            acc = r["accession"]
            go = r.get("go_terms") or ""
            lab = weak_ec_l1_from_go(go)
            if lab is not None:
                labels[acc] = lab
    return labels


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


class MLPProbe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)


def stack_views(cache, accs, views):
    return torch.cat([torch.stack([cache[a][v].float() for a in accs]) for v in views], dim=-1)


def train_ec_probe(x, y, kind="linear", device="cuda", epochs=100, lr=1e-3, wd=1e-4):
    in_dim = x.shape[1]
    probe = (LinearProbe(in_dim, 8) if kind == "linear" else MLPProbe(in_dim, 8)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    x = x.to(device); y = y.to(device)
    use_mini = x.shape[0] > 1024
    bs = 64
    for _ in range(epochs):
        probe.train()
        if use_mini:
            perm = torch.randperm(x.shape[0], device=device)
            for s in range(0, x.shape[0], bs):
                idx = perm[s:s+bs]
                loss = loss_fn(probe(x[idx]), y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        else:
            loss = loss_fn(probe(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return probe.eval()


@torch.no_grad()
def predict_ec(probe, x, device="cuda"):
    return torch.softmax(probe(x.to(device)), dim=-1).cpu().numpy()


def compute_ec_auc(y_true, y_score, n_classes=8):
    per = []
    for c in range(n_classes):
        yt = (y_true == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            per.append(None); continue
        per.append(float(roc_auc_score(yt, y_score[:, c])))
    good = [a for a in per if a is not None]
    return (float(np.mean(good)) if good else float("nan"), per)


def ec_label(cache, acc):
    v = cache[acc]["labels"]["ec_l1"]
    return 0 if v is None else int(v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ref_cache = torch.load(EMB_DIR / "reference_embeddings.pt", map_location="cpu", weights_only=False)
    bench_cache = torch.load(EMB_DIR / "benchmark_embeddings.pt", map_location="cpu", weights_only=False)
    dark_cache = torch.load(EMB_DIR / "dark_embeddings.pt", map_location="cpu", weights_only=False)

    ref_accs = list(ref_cache.keys())
    bench_accs = list(bench_cache.keys())
    dark_accs = list(dark_cache.keys())
    print(f"N: reference={len(ref_accs)}  benchmark={len(bench_accs)}  dark={len(dark_accs)}")

    # Pool reference + benchmark → labeled set with UniProt EC
    all_accs = ref_accs + bench_accs
    all_labels = np.array([ec_label(ref_cache, a) for a in ref_accs] +
                          [ec_label(bench_cache, a) for a in bench_accs])
    all_cache = {**ref_cache, **bench_cache}
    print(f"Combined labeled pool: n={len(all_accs)}")
    print(f"EC L1 distribution: "
          f"{dict(zip(*np.unique(all_labels, return_counts=True)))}")

    # Variants to test via CV: A+B+C linear (best linear) and A-only linear
    variants = [
        ("A+B+C_linear", VIEWS, "linear"),
        ("A+B+C_mlp", VIEWS, "mlp"),
        ("A_linear", [VIEWS[0]], "linear"),
    ]

    cv_results = {}
    for variant_name, views, kind in variants:
        print(f"\n=== (A) 5-fold stratified CV — {variant_name} ===")
        x_all = stack_views(all_cache, all_accs, views)
        y_all = torch.tensor(all_labels, dtype=torch.long)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        fold_macros = []
        fold_per_class = np.full((5, 8), np.nan)
        for fold, (tr, te) in enumerate(skf.split(x_all.numpy(), y_all.numpy())):
            probe = train_ec_probe(x_all[tr], y_all[tr], kind=kind, device=device)
            scores = predict_ec(probe, x_all[te], device)
            macro, per = compute_ec_auc(y_all[te].numpy(), scores)
            fold_macros.append(macro)
            for c in range(8):
                fold_per_class[fold, c] = per[c] if per[c] is not None else np.nan
            print(f"  fold {fold+1}: macro-AUC = {macro:.3f}")
        fold_macros = np.array(fold_macros)
        print(f"  macro-AUC: {fold_macros.mean():.3f} ± {fold_macros.std():.3f}  "
              f"range [{fold_macros.min():.3f}, {fold_macros.max():.3f}]")
        print(f"  per-class mean (±std across folds):")
        class_names = ["non-enzyme", "oxidored", "transfer", "hydrolase",
                       "lyase", "isomerase", "ligase", "translocase"]
        for c in range(8):
            col = fold_per_class[:, c]
            col = col[~np.isnan(col)]
            if len(col) > 0:
                print(f"    {c} ({class_names[c]:11s}): {col.mean():.3f} ± {col.std():.3f}  n_folds={len(col)}")
            else:
                print(f"    {c} ({class_names[c]:11s}): n/a")
        cv_results[variant_name] = {
            "macro_mean": float(fold_macros.mean()),
            "macro_std": float(fold_macros.std()),
            "macro_min": float(fold_macros.min()),
            "macro_max": float(fold_macros.max()),
            "per_class_fold_mean": [
                float(np.nanmean(fold_per_class[:, c])) if np.any(~np.isnan(fold_per_class[:, c])) else None
                for c in range(8)
            ],
            "per_class_fold_std": [
                float(np.nanstd(fold_per_class[:, c])) if np.any(~np.isnan(fold_per_class[:, c])) else None
                for c in range(8)
            ],
        }

    # (B) Dark-genome weak EC labels
    print("\n=== (B) Dark-genome EC-L1 AUC on weak GO-derived labels ===")
    dark_ec = build_dark_ec_labels()
    print(f"Dark proteins with weak EC label: {len(dark_ec)} / {len(dark_accs)}")
    dark_dist = {}
    for ec in dark_ec.values():
        dark_dist[ec] = dark_dist.get(ec, 0) + 1
    print(f"Weak EC distribution: {dict(sorted(dark_dist.items()))}")

    # Filter dark to those with weak labels AND embeddings
    dark_labeled = [a for a in dark_accs if a in dark_ec]
    y_dark = np.array([dark_ec[a] for a in dark_labeled])
    print(f"Dark labeled subset: {len(dark_labeled)} proteins")

    dark_results = {}
    for variant_name, views, kind in variants:
        # Train final probe on FULL labeled pool (ref + bench), apply to dark
        x_train = stack_views(all_cache, all_accs, views)
        y_train = torch.tensor(all_labels, dtype=torch.long)
        x_dark = stack_views(dark_cache, dark_labeled, views)

        probe = train_ec_probe(x_train, y_train, kind=kind, device=device)
        scores = predict_ec(probe, x_dark, device)
        macro, per = compute_ec_auc(y_dark, scores)
        print(f"  {variant_name:22s}  macro-AUC (weak) = {macro:.3f}")
        class_names = ["non-enzyme", "oxidored", "transfer", "hydrolase",
                       "lyase", "isomerase", "ligase", "translocase"]
        for c in range(8):
            n_pos = int((y_dark == c).sum())
            if n_pos == 0 or per[c] is None:
                continue
            print(f"    class {c} ({class_names[c]:11s}, n={n_pos:3d}): AUC = {per[c]:.3f}")
        dark_results[variant_name] = {
            "dark_macro_weak": float(macro),
            "dark_per_class_weak": per,
            "dark_n_labeled": int(len(dark_labeled)),
            "dark_label_dist": {int(k): int(v) for k, v in dark_dist.items()},
        }

    # Save everything
    out = {
        "cv_5fold": cv_results,
        "dark_weak": dark_results,
    }
    (EMB_DIR / "ec_classifier_evaluation.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {EMB_DIR / 'ec_classifier_evaluation.json'}")


if __name__ == "__main__":
    main()
