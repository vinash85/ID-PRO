"""
Compute downstream probe metrics from the per-arm no-RAG embeddings extracted by
extract_norag_only.py, in the same JSON shape as reports/norag_baseline_results.json.

For ARM in {S0, S1} this produces:
  - cv_5fold_pool: A_linear, A+B+C_linear, A+B+C_mlp on (reference + benchmark)
  - heldout_bench_ABC_mlp: is_enzyme / ec_l1 / go_f_top20 / pfam_top20 (train ref → test bench)
  - dark_train_ref_ABC_mlp: macro-AUC on dark for is_enzyme / ec_l1 / go_f_top20 / pfam_top20
  - dark_ec_argmax: count of A+B+C MLP EC argmax across the 8 EC L1 classes on dark
  - dark_per_class_cv_ABC_linear: per-class CV mean AUC (matches per-class table in report)

Run:
  CUDA_VISIBLE_DEVICES=0 python idpro/scripts/compute_norag_arm_results.py \
    --emb-dir idpro/data/probe/embeddings_S0_norag --out reports/norag_S0_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

VIEWS = ["view_a_prompteol_l48", "view_b_question_mean_l48", "view_c_eos_l64"]
EC_L1_NAMES = {
    0: "non-enzyme",
    1: "oxidoreductase",
    2: "transferase",
    3: "hydrolase",
    4: "lyase",
    5: "isomerase",
    6: "ligase",
    7: "translocase",
}


# ---------- probes ----------


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def stack_views(cache, accs, views):
    return torch.cat(
        [torch.stack([cache[a][v].float() for a in accs]) for v in views], dim=-1
    )


def train_classification(x, y, n_classes, kind, device, epochs=100, bs=64, lr=1e-3, wd=1e-4):
    in_dim = x.shape[1]
    probe = (LinearProbe(in_dim, n_classes) if kind == "linear" else MLPProbe(in_dim, n_classes)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    x = x.to(device)
    y = y.to(device)
    use_mini = x.shape[0] > 1024
    for _ in range(epochs):
        probe.train()
        if use_mini:
            perm = torch.randperm(x.shape[0], device=device)
            for s in range(0, x.shape[0], bs):
                idx = perm[s:s + bs]
                loss = loss_fn(probe(x[idx]), y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        else:
            loss = loss_fn(probe(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return probe.eval()


def train_multilabel(x, y, n_out, device, epochs=100, bs=64, lr=1e-3, wd=1e-4, binary=False):
    in_dim = x.shape[1]
    probe = MLPProbe(in_dim, n_out).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    x = x.to(device)
    y = y.to(device)
    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(x.shape[0], device=device)
        for s in range(0, x.shape[0], bs):
            idx = perm[s:s + bs]
            logits = probe(x[idx])
            if binary:
                loss = loss_fn(logits.squeeze(-1), y[idx].float())
            else:
                loss = loss_fn(logits, y[idx].float())
            opt.zero_grad(); loss.backward(); opt.step()
    return probe.eval()


@torch.no_grad()
def predict_softmax(probe, x, device):
    return torch.softmax(probe(x.to(device)), dim=-1).cpu().numpy()


@torch.no_grad()
def predict_sigmoid(probe, x, device, binary=False):
    logits = probe(x.to(device))
    if binary:
        return torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
    return torch.sigmoid(logits).cpu().numpy()


def per_class_auc_8(y_true, scores):
    per = []
    for c in range(8):
        yt = (y_true == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            per.append(None)
            continue
        per.append(float(roc_auc_score(yt, scores[:, c])))
    good = [a for a in per if a is not None]
    return (float(np.mean(good)) if good else float("nan"), per)


def multilabel_macro_auc(y_true, scores):
    """y_true: (N, K) {0,1}. scores: (N, K) float. Returns macro-AUC and per-class list."""
    K = y_true.shape[1]
    per = []
    for c in range(K):
        yt = y_true[:, c]
        if yt.sum() == 0 or yt.sum() == len(yt):
            per.append(None)
            continue
        per.append(float(roc_auc_score(yt, scores[:, c])))
    good = [a for a in per if a is not None]
    return (float(np.mean(good)) if good else float("nan"), per)


# ---------- main eval ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True, type=Path,
                    help="Directory containing {reference,benchmark,dark}_norag_embeddings.pt")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output JSON path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  emb_dir: {args.emb_dir}", flush=True)

    ref_path = args.emb_dir / "reference_norag_embeddings.pt"
    bench_path = args.emb_dir / "benchmark_norag_embeddings.pt"
    dark_path = args.emb_dir / "dark_norag_embeddings.pt"

    ref_cache = torch.load(ref_path, map_location="cpu", weights_only=False)
    bench_cache = torch.load(bench_path, map_location="cpu", weights_only=False)
    dark_cache = torch.load(dark_path, map_location="cpu", weights_only=False)
    ref_accs = list(ref_cache.keys())
    bench_accs = list(bench_cache.keys())
    dark_accs = list(dark_cache.keys())
    print(f"N: ref={len(ref_accs)} bench={len(bench_accs)} dark={len(dark_accs)}", flush=True)

    # ------------------------------------------------------------
    # (A) 5-fold stratified CV on the pooled (ref + bench) labeled set, EC L1
    # ------------------------------------------------------------
    pool_accs = ref_accs + bench_accs
    pool_cache = {**ref_cache, **bench_cache}
    pool_y = np.array([
        0 if pool_cache[a]["labels"]["ec_l1"] is None else int(pool_cache[a]["labels"]["ec_l1"])
        for a in pool_accs
    ])
    print(f"Pool n={len(pool_accs)}  EC distribution: "
          f"{dict(zip(*np.unique(pool_y, return_counts=True)))}", flush=True)

    cv_results = {}
    variants = [
        ("A_linear",      [VIEWS[0]], "linear"),
        ("A+B+C_linear",  VIEWS,      "linear"),
        ("A+B+C_mlp",     VIEWS,      "mlp"),
    ]

    for variant_name, views, kind in variants:
        print(f"\n=== CV5: {variant_name} ===", flush=True)
        x_pool = stack_views(pool_cache, pool_accs, views)
        y_pool = torch.tensor(pool_y, dtype=torch.long)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        fold_macros = []
        fold_per_class = np.full((5, 8), np.nan)
        for fold, (tr, te) in enumerate(skf.split(x_pool.numpy(), y_pool.numpy())):
            probe = train_classification(x_pool[tr], y_pool[tr], 8, kind, device)
            scores = predict_softmax(probe, x_pool[te], device)
            macro, per = per_class_auc_8(y_pool[te].numpy(), scores)
            fold_macros.append(macro)
            for c in range(8):
                fold_per_class[fold, c] = per[c] if per[c] is not None else np.nan
            print(f"  fold {fold+1}: macro-AUC={macro:.4f}", flush=True)
        fold_macros = np.array(fold_macros)
        per_class_mean = [
            float(np.nanmean(fold_per_class[:, c])) if np.any(~np.isnan(fold_per_class[:, c])) else None
            for c in range(8)
        ]
        cv_results[variant_name] = {
            "macro_auc_mean": float(fold_macros.mean()),
            "macro_auc_std": float(fold_macros.std()),
            "fold_aucs": [float(x) for x in fold_macros],
            "per_class_auc_mean": per_class_mean,
            "feature_dim": int(x_pool.shape[1]),
            "n_samples": int(len(pool_accs)),
        }
        print(f"  macro-AUC: {fold_macros.mean():.4f} ± {fold_macros.std():.4f}", flush=True)

    # ------------------------------------------------------------
    # (B) Held-out benchmark probe (train on ref → test on bench), A+B+C MLP
    # ------------------------------------------------------------
    print("\n=== Held-out bench (train ref → test bench), A+B+C MLP ===", flush=True)
    x_ref = stack_views(ref_cache, ref_accs, VIEWS)
    x_bench = stack_views(bench_cache, bench_accs, VIEWS)

    bench_results = {}

    # is_enzyme (binary)
    y_ref_e = torch.tensor([ref_cache[a]["labels"]["is_enzyme"] for a in ref_accs], dtype=torch.float32)
    y_bench_e = np.array([bench_cache[a]["labels"]["is_enzyme"] for a in bench_accs])
    probe = train_multilabel(x_ref, y_ref_e, 1, device, binary=True)
    scores = predict_sigmoid(probe, x_bench, device, binary=True)
    auc = float(roc_auc_score(y_bench_e, scores)) if len(np.unique(y_bench_e)) > 1 else float("nan")
    bench_results["is_enzyme"] = {"bench_auc": auc, "bench_per_class": [auc]}
    print(f"  is_enzyme: bench-AUC={auc:.4f}", flush=True)

    # ec_l1 (8-way)
    y_ref_ec = torch.tensor([
        0 if ref_cache[a]["labels"]["ec_l1"] is None else int(ref_cache[a]["labels"]["ec_l1"])
        for a in ref_accs
    ], dtype=torch.long)
    y_bench_ec = np.array([
        0 if bench_cache[a]["labels"]["ec_l1"] is None else int(bench_cache[a]["labels"]["ec_l1"])
        for a in bench_accs
    ])
    probe = train_classification(x_ref, y_ref_ec, 8, "mlp", device)
    scores = predict_softmax(probe, x_bench, device)
    macro, per = per_class_auc_8(y_bench_ec, scores)
    bench_results["ec_l1"] = {
        "bench_auc": macro,
        "bench_per_class": [a if a is not None else float("nan") for a in per],
    }
    print(f"  ec_l1: bench-AUC={macro:.4f}", flush=True)

    # go_f_top20 (multilabel)
    y_ref_go = torch.tensor([ref_cache[a]["labels"]["go_f"] for a in ref_accs], dtype=torch.float32)
    y_bench_go = np.array([bench_cache[a]["labels"]["go_f"] for a in bench_accs])
    probe = train_multilabel(x_ref, y_ref_go, 20, device, binary=False)
    scores = predict_sigmoid(probe, x_bench, device, binary=False)
    macro, per = multilabel_macro_auc(y_bench_go, scores)
    bench_results["go_f_top20"] = {
        "bench_auc": macro,
        "bench_per_class": [a if a is not None else float("nan") for a in per],
    }
    print(f"  go_f_top20: bench-AUC={macro:.4f}", flush=True)

    # pfam_top20 (multilabel)
    y_ref_pf = torch.tensor([ref_cache[a]["labels"]["pfam"] for a in ref_accs], dtype=torch.float32)
    y_bench_pf = np.array([bench_cache[a]["labels"]["pfam"] for a in bench_accs])
    probe = train_multilabel(x_ref, y_ref_pf, 20, device, binary=False)
    scores = predict_sigmoid(probe, x_bench, device, binary=False)
    macro, per = multilabel_macro_auc(y_bench_pf, scores)
    bench_results["pfam_top20"] = {
        "bench_auc": macro,
        "bench_per_class": [a if a is not None else float("nan") for a in per],
    }
    print(f"  pfam_top20: bench-AUC={macro:.4f}", flush=True)

    # ------------------------------------------------------------
    # (C) Dark genome (train on REF only, evaluate on dark), A+B+C MLP
    # Mirrors build_contrast_report.per_arm_predictions protocol.
    # ------------------------------------------------------------
    print("\n=== Dark eval (train ref → predict dark), A+B+C MLP ===", flush=True)
    x_dark = stack_views(dark_cache, dark_accs, VIEWS)

    dark_results = {}
    # is_enzyme
    y_dark_e = np.array([dark_cache[a]["labels"]["is_enzyme"] for a in dark_accs])
    probe = train_multilabel(x_ref, y_ref_e, 1, device, binary=True)
    scores_e = predict_sigmoid(probe, x_dark, device, binary=True)
    if len(np.unique(y_dark_e)) > 1:
        dark_results["is_enzyme"] = float(roc_auc_score(y_dark_e, scores_e))
    else:
        dark_results["is_enzyme"] = float("nan")
    print(f"  is_enzyme: dark-AUC={dark_results['is_enzyme']:.4f}", flush=True)

    # ec_l1 (multi-class) — for completeness; report uses argmax distribution
    probe_ec = train_classification(x_ref, y_ref_ec, 8, "mlp", device)
    scores_ec = predict_softmax(probe_ec, x_dark, device)
    # Dark ec_l1 labels are mostly None; only some have weak EC. Skip macro-AUC
    # (not directly comparable to existing report) — but compute argmax counts.
    ec_argmax = scores_ec.argmax(axis=1)
    argmax_counts = {EC_L1_NAMES[c]: int((ec_argmax == c).sum()) for c in range(8)}
    print(f"  ec_argmax counts: {argmax_counts}", flush=True)

    # go_f_top20
    y_dark_go = np.array([dark_cache[a]["labels"]["go_f"] for a in dark_accs])
    probe = train_multilabel(x_ref, y_ref_go, 20, device, binary=False)
    scores_go = predict_sigmoid(probe, x_dark, device, binary=False)
    macro, _ = multilabel_macro_auc(y_dark_go, scores_go)
    dark_results["go_f_top20"] = macro
    print(f"  go_f_top20: dark-AUC={macro:.4f}", flush=True)

    # pfam_top20
    y_dark_pf = np.array([dark_cache[a]["labels"]["pfam"] for a in dark_accs])
    probe = train_multilabel(x_ref, y_ref_pf, 20, device, binary=False)
    scores_pf = predict_sigmoid(probe, x_dark, device, binary=False)
    macro, _ = multilabel_macro_auc(y_dark_pf, scores_pf)
    dark_results["pfam_top20"] = macro
    print(f"  pfam_top20: dark-AUC={macro:.4f}", flush=True)

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------
    out = {
        "cv_5fold_pool": cv_results,
        "heldout_bench_ABC_mlp": bench_results,
        "dark_train_ref_ABC_mlp": dark_results,
        "dark_ec_argmax": argmax_counts,
        "n": {"reference": len(ref_accs), "benchmark": len(bench_accs), "dark": len(dark_accs)},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
