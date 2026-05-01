"""
Evaluate the frozen-IDPro classifier probe on the 415 dark-genome proteins.

Dark-genome proteins are unannotated in UniProt, but many carry WEAK labels
from InterProScan (Pfam IDs, GO terms). We report:

  1. Probe AUC on the weak labels (noisy — upper-bounded by InterProScan's
     own error rate — but strictly better than "no evaluation").
  2. Prediction distribution across EC classes (no ground truth) + confidence
     distribution vs benchmark proteins (expected: dark have lower confidence).
  3. Conformal-wrapped prediction sets (split conformal using the
     benchmark as calibration) — quantifies uncertainty per protein.

Run:
    python scripts/evaluate_probe_on_dark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.paths import AIM1_PROBE_DIR as DATA_DIR, REPORTS_DIR  # noqa: E402

EMB_DIR = DATA_DIR / "embeddings"
REPORT_DIR = REPORTS_DIR
REPORT_DIR.mkdir(parents=True, exist_ok=True)

VIEWS = [
    "view_a_prompteol_l48",
    "view_b_question_mean_l48",
    "view_c_eos_l64",
]


def stack_views(cache, accs, views, esmc_map=None):
    tensors = []
    for v in views:
        if v == "esmc_mean_pool":
            tensors.append(torch.stack([torch.from_numpy(esmc_map[a]).float() for a in accs]))
        else:
            tensors.append(torch.stack([cache[a][v].float() for a in accs]))
    return torch.cat(tensors, dim=-1)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
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


def train_probe(x_train, y_train, out_dim, task, kind="mlp", device="cuda", epochs=100, lr=1e-3, wd=1e-4):
    in_dim = x_train.shape[1]
    probe = (LinearProbe(in_dim, out_dim) if kind == "linear" else MLPProbe(in_dim, out_dim)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    x = x_train.to(device)
    y = y_train.to(device)
    if task == "is_enzyme":
        loss_fn = nn.BCEWithLogitsLoss()
    elif task == "ec_l1":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    use_mini = x.shape[0] > 1024
    bs = 64
    for _ in range(epochs):
        probe.train()
        if use_mini:
            perm = torch.randperm(x.shape[0], device=device)
            for s in range(0, x.shape[0], bs):
                idx = perm[s : s + bs]
                logits = probe(x[idx])
                if task == "is_enzyme":
                    loss = loss_fn(logits.squeeze(-1), y[idx].float())
                elif task == "ec_l1":
                    loss = loss_fn(logits, y[idx])
                else:
                    loss = loss_fn(logits, y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        else:
            logits = probe(x)
            if task == "is_enzyme":
                loss = loss_fn(logits.squeeze(-1), y.float())
            elif task == "ec_l1":
                loss = loss_fn(logits, y)
            else:
                loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return probe.eval()


@torch.no_grad()
def predict(probe, x, device="cuda", task="is_enzyme"):
    x = x.to(device)
    logits = probe(x)
    if task == "is_enzyme":
        return torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
    if task == "ec_l1":
        return torch.softmax(logits, dim=-1).cpu().numpy()
    return torch.sigmoid(logits).cpu().numpy()


def load_labels(cache, accs, kind):
    if kind == "is_enzyme":
        return torch.tensor([cache[a]["labels"]["is_enzyme"] for a in accs], dtype=torch.long)
    if kind == "ec_l1":
        v = [0 if cache[a]["labels"]["ec_l1"] is None else int(cache[a]["labels"]["ec_l1"]) for a in accs]
        return torch.tensor(v, dtype=torch.long)
    if kind == "go_f_top20":
        return torch.tensor([cache[a]["labels"]["go_f"] for a in accs], dtype=torch.float32)
    if kind == "pfam_top20":
        return torch.tensor([cache[a]["labels"]["pfam"] for a in accs], dtype=torch.float32)
    raise ValueError(kind)


def compute_auc(y_true, y_score, task):
    if task == "is_enzyme":
        if len(np.unique(y_true)) < 2:
            return float("nan"), [float("nan")]
        return roc_auc_score(y_true, y_score), [roc_auc_score(y_true, y_score)]
    if task == "ec_l1":
        per = []
        for c in range(8):
            yt = (y_true == c).astype(int)
            if yt.sum() in (0, len(yt)):
                per.append(float("nan")); continue
            per.append(roc_auc_score(yt, y_score[:, c]))
        good = [a for a in per if not np.isnan(a)]
        return (float(np.mean(good)) if good else float("nan"), per)
    # multilabel
    per = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() in (0, len(yt)):
            per.append(float("nan")); continue
        per.append(roc_auc_score(yt, y_score[:, i]))
    good = [a for a in per if not np.isnan(a)]
    return (float(np.mean(good)) if good else float("nan"), per)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ref_cache = torch.load(EMB_DIR / "reference_embeddings.pt", map_location="cpu", weights_only=False)
    bench_cache = torch.load(EMB_DIR / "benchmark_embeddings.pt", map_location="cpu", weights_only=False)
    dark_cache = torch.load(EMB_DIR / "dark_embeddings.pt", map_location="cpu", weights_only=False)

    # ESM C baseline embeddings
    npz = np.load(EMB_DIR / "rag_index.npz", allow_pickle=True)
    esmc_map = dict(zip(npz["ids"].tolist(), npz["embs"]))
    # Dark proteins aren't in ESM C index yet — add them.
    # (We need these if we want the ESM C baseline on dark; but we can skip for now.)
    have_esmc_for_dark = all(a in esmc_map for a in dark_cache)
    if not have_esmc_for_dark:
        print("Note: ESM C baseline unavailable for dark set (skipping ESM C variant on dark).")

    ref_accs = list(ref_cache.keys())
    bench_accs = list(bench_cache.keys())
    dark_accs = list(dark_cache.keys())
    print(f"N: reference={len(ref_accs)} benchmark={len(bench_accs)} dark={len(dark_accs)}")

    # Use the recommended A+B+C concat (per design doc + linear probe best).
    # Pair with MLP for classification strength, per results §3.
    variants = [
        ("A+B+C_mlp", VIEWS, "mlp"),
        ("A+B+C_linear", VIEWS, "linear"),
        ("A_prompteol_linear", [VIEWS[0]], "linear"),
    ]
    tasks = ["is_enzyme", "ec_l1", "go_f_top20", "pfam_top20"]
    task_dims = {"is_enzyme": 1, "ec_l1": 8, "go_f_top20": 20, "pfam_top20": 20}

    results_summary = {}

    print("\n=== Training probes on reference, applying to dark + benchmark ===")
    for variant_name, views, kind in variants:
        x_ref = stack_views(ref_cache, ref_accs, views)
        x_bench = stack_views(bench_cache, bench_accs, views)
        x_dark = stack_views(dark_cache, dark_accs, views)
        print(f"\n[{variant_name}]  dim={x_ref.shape[1]}")

        for task in tasks:
            y_ref = load_labels(ref_cache, ref_accs, task)
            y_bench = load_labels(bench_cache, bench_accs, task)
            y_dark = load_labels(dark_cache, dark_accs, task)

            probe = train_probe(
                x_ref, y_ref, task_dims[task], task=task, kind=kind, device=device
            )
            p_bench = predict(probe, x_bench, device, task)
            p_dark = predict(probe, x_dark, device, task)

            # AUC on benchmark (real GT)
            bench_auc, bench_per = compute_auc(y_bench.numpy(), p_bench, task)
            # AUC on dark (WEAK labels — noisy)
            dark_auc, dark_per = compute_auc(y_dark.numpy(), p_dark, task)

            # Confidence: max-softmax for multiclass, raw prob for binary/multilabel
            if task == "is_enzyme":
                bench_conf = p_bench
                dark_conf = p_dark
            elif task == "ec_l1":
                bench_conf = p_bench.max(axis=1)
                dark_conf = p_dark.max(axis=1)
            else:
                bench_conf = p_bench.max(axis=1)
                dark_conf = p_dark.max(axis=1)

            print(f"  {task:14s}  bench-AUC={bench_auc:.3f}  "
                  f"dark-AUC(weak)={dark_auc:.3f}  "
                  f"mean-conf bench={bench_conf.mean():.3f}  "
                  f"mean-conf dark={dark_conf.mean():.3f}")

            results_summary[(variant_name, task)] = {
                "bench_auc": bench_auc,
                "bench_auc_per_class": bench_per,
                "dark_auc_weak": dark_auc,
                "dark_auc_per_class_weak": dark_per,
                "bench_confidence_mean": float(bench_conf.mean()),
                "dark_confidence_mean": float(dark_conf.mean()),
                "bench_confidence_p25": float(np.percentile(bench_conf, 25)),
                "dark_confidence_p25": float(np.percentile(dark_conf, 25)),
                "bench_confidence_median": float(np.median(bench_conf)),
                "dark_confidence_median": float(np.median(dark_conf)),
                # For prediction-distribution + conformal
                "dark_preds": p_dark.tolist(),
                "bench_preds": p_bench.tolist(),
            }

    # ---- Conformal wrapper on EC L1 predictions ----
    print("\n=== Conformal prediction sets (EC L1, α=0.10) ===")
    from idpro.model.idpro.conformal import SplitConformalPredictor

    key = ("A+B+C_mlp", "ec_l1")
    if key in results_summary:
        rec = results_summary[key]
        bench_scores = 1 - np.max(np.array(rec["bench_preds"]), axis=1)  # 1 - max softmax
        cp = SplitConformalPredictor().calibrate(bench_scores)
        for alpha in [0.05, 0.10, 0.20]:
            tau = cp.threshold(alpha)
            # Count EC L1 classes that pass threshold per protein
            dark_probs = np.array(rec["dark_preds"])  # (N_dark, 8)
            # Nonconformity of class c for protein i = 1 - P(c|i)
            nc = 1 - dark_probs  # (N_dark, 8)
            set_sizes = (nc <= tau).sum(axis=1)
            bench_probs = np.array(rec["bench_preds"])
            nc_b = 1 - bench_probs
            set_sizes_b = (nc_b <= tau).sum(axis=1)
            print(f"  α={alpha:.2f}  tau={tau:.3f}  "
                  f"dark mean set size={set_sizes.mean():.2f}  "
                  f"bench mean set size={set_sizes_b.mean():.2f}  "
                  f"Δ={set_sizes.mean() - set_sizes_b.mean():+.2f}")
            results_summary[("conformal_ec_l1", alpha)] = {
                "tau": float(tau),
                "dark_set_sizes": set_sizes.tolist(),
                "bench_set_sizes": set_sizes_b.tolist(),
                "dark_mean_set_size": float(set_sizes.mean()),
                "bench_mean_set_size": float(set_sizes_b.mean()),
                "dark_empty_set_frac": float((set_sizes == 0).mean()),
                "bench_empty_set_frac": float((set_sizes_b == 0).mean()),
            }

    # ---- Prediction distribution on dark (EC L1) ----
    print("\n=== Dark-genome EC L1 prediction distribution (A+B+C MLP, argmax) ===")
    dark_probs = np.array(results_summary[("A+B+C_mlp", "ec_l1")]["dark_preds"])
    preds = dark_probs.argmax(axis=1)
    for c in range(8):
        n = int((preds == c).sum())
        name = {0: "non-enzyme", 1: "oxidoreductase", 2: "transferase", 3: "hydrolase",
                4: "lyase", 5: "isomerase", 6: "ligase", 7: "translocase"}[c]
        frac = n / len(preds)
        print(f"  class {c} ({name:15s}): n={n:3d} ({frac*100:.1f}%)")

    # Serialize (skip huge prediction arrays in summary, keep stats)
    out = {}
    for k, v in results_summary.items():
        sk = str(k)
        out[sk] = {kk: vv for kk, vv in v.items() if kk not in ("dark_preds", "bench_preds")}
    (EMB_DIR / "dark_probe_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {EMB_DIR / 'dark_probe_results.json'}")


if __name__ == "__main__":
    main()
