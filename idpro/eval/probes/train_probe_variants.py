"""
Train multiple classifier-probe variants on cached IDPro embeddings.

Variants span all sensible combinations of the three pooled views:
  A — PromptEOL colon position, layer 48
  B — mean over question token span, layer 48
  C — last-position (EOS) hidden state, layer 64

plus an ESM-C mean-pool protein-only baseline to show that query conditioning
matters.

For each variant we train three probes (in parallel on one GPU):
  is_enzyme      (binary)
  ec_l1          (7-way multi-class)
  go_f_top20     (20-way multi-label)
  pfam_top20     (20-way multi-label)

Labels/scores reported: macro-AUC, per-class AUC, AUC@FPR=0.1 where applicable.

Training uses the reference set (3000 proteins) → evaluates on benchmark (~637).

Run:
    python scripts/train_probe_variants.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from idpro.paths import PROBE_DIR as DATA_DIR, DATA_ROOT  # noqa: E402

EMB_DIR = DATA_DIR / "embeddings"
REPORT_DIR = DATA_ROOT / "preliminary_data" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

VIEWS = ["view_a_prompteol_l48", "view_b_question_mean_l48", "view_c_eos_l64"]


# ---------------------------------------------------------------------------
# Load cached embeddings → tensors aligned by order
# ---------------------------------------------------------------------------


def load_cache(path: Path) -> Dict[str, Dict]:
    return torch.load(path, map_location="cpu", weights_only=False)


def stack_view(cache: Dict[str, Dict], accessions: List[str], view: str) -> torch.Tensor:
    rows = [cache[a][view].float() for a in accessions]
    return torch.stack(rows, dim=0)


def stack_esmc_baseline(
    cache: Dict[str, Dict],
    accessions: List[str],
    esmc_embs: Dict[str, np.ndarray],
) -> torch.Tensor:
    """Protein-only baseline: ESM C mean-pool (query-agnostic)."""
    rows = [torch.from_numpy(esmc_embs[a]).float() for a in accessions]
    return torch.stack(rows, dim=0)


def load_labels(cache: Dict[str, Dict], accessions: List[str], kind: str):
    """Return a tensor with the right label shape for each task."""
    if kind == "is_enzyme":
        return torch.tensor([cache[a]["labels"]["is_enzyme"] for a in accessions], dtype=torch.long)
    if kind == "ec_l1":
        # Map None/Missing → 0 ("non-enzyme"); 1..7 → 1..7
        vals = []
        for a in accessions:
            v = cache[a]["labels"]["ec_l1"]
            vals.append(0 if v is None else int(v))
        return torch.tensor(vals, dtype=torch.long)
    if kind == "go_f_top20":
        return torch.tensor([cache[a]["labels"]["go_f"] for a in accessions], dtype=torch.float32)
    if kind == "pfam_top20":
        return torch.tensor([cache[a]["labels"]["pfam"] for a in accessions], dtype=torch.float32)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Probe architectures
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training loop (one probe)
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    variant: str
    task: str
    kind: str  # "linear" or "mlp"
    auc_macro: float
    auc_per_class: List[float]
    n_train: int
    n_test: int
    extra: Dict = field(default_factory=dict)


def compute_auc(y_true: np.ndarray, y_score: np.ndarray, task: str) -> Tuple[float, List[float]]:
    """Return (macro_auc, per_class_auc). y_true shape handled per task."""
    if task == "is_enzyme":
        # binary
        if len(np.unique(y_true)) < 2:
            return float("nan"), [float("nan")]
        auc = roc_auc_score(y_true, y_score)
        return auc, [auc]
    if task == "ec_l1":
        # multiclass one-vs-rest, classes 0..7 (0 = non-enzyme)
        per = []
        for c in range(8):
            yt = (y_true == c).astype(int)
            if yt.sum() == 0 or yt.sum() == len(yt):
                per.append(float("nan"))
                continue
            per.append(roc_auc_score(yt, y_score[:, c]))
        good = [a for a in per if not np.isnan(a)]
        macro = float(np.mean(good)) if good else float("nan")
        return macro, per
    if task in ("go_f_top20", "pfam_top20"):
        # multilabel: average AUC across labels that have both pos and neg examples.
        per = []
        for i in range(y_true.shape[1]):
            yt = y_true[:, i]
            if yt.sum() == 0 or yt.sum() == len(yt):
                per.append(float("nan"))
                continue
            per.append(roc_auc_score(yt, y_score[:, i]))
        good = [a for a in per if not np.isnan(a)]
        macro = float(np.mean(good)) if good else float("nan")
        return macro, per
    raise ValueError(task)


def train_one_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    task: str,
    kind: str,
    variant: str,
    device: str = "cuda",
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    verbose: bool = False,
) -> ProbeResult:
    in_dim = x_train.shape[1]
    if task == "is_enzyme":
        out_dim = 1
        loss_fn = nn.BCEWithLogitsLoss()
    elif task == "ec_l1":
        out_dim = 8
        loss_fn = nn.CrossEntropyLoss()
    elif task in ("go_f_top20", "pfam_top20"):
        out_dim = y_train.shape[1]
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(task)

    probe = (LinearProbe(in_dim, out_dim) if kind == "linear" else MLPProbe(in_dim, out_dim)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_test_cpu = y_test.cpu().numpy()

    # Simple full-batch training on small data, mini-batch if > 1024 samples.
    use_mini = x_train.shape[0] > 1024

    for epoch in range(epochs):
        probe.train()
        if use_mini:
            perm = torch.randperm(x_train.shape[0], device=device)
            for s in range(0, x_train.shape[0], batch_size):
                idx = perm[s : s + batch_size]
                logits = probe(x_train[idx])
                if task == "is_enzyme":
                    loss = loss_fn(logits.squeeze(-1), y_train[idx].float())
                elif task == "ec_l1":
                    loss = loss_fn(logits, y_train[idx])
                else:
                    loss = loss_fn(logits, y_train[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        else:
            logits = probe(x_train)
            if task == "is_enzyme":
                loss = loss_fn(logits.squeeze(-1), y_train.float())
            elif task == "ec_l1":
                loss = loss_fn(logits, y_train)
            else:
                loss = loss_fn(logits, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        logits = probe(x_test)
        if task == "is_enzyme":
            y_score = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        elif task == "ec_l1":
            y_score = torch.softmax(logits, dim=-1).cpu().numpy()
        else:
            y_score = torch.sigmoid(logits).cpu().numpy()

    auc_macro, per_class = compute_auc(y_test_cpu, y_score, task)
    return ProbeResult(
        variant=variant,
        task=task,
        kind=kind,
        auc_macro=auc_macro,
        auc_per_class=per_class,
        n_train=x_train.shape[0],
        n_test=x_test.shape[0],
    )


# ---------------------------------------------------------------------------
# Variant builders (feature concat)
# ---------------------------------------------------------------------------


def build_variant(
    cache: Dict[str, Dict],
    accs: List[str],
    views: List[str],
    esmc_embs: Optional[Dict[str, np.ndarray]] = None,
) -> torch.Tensor:
    tensors = []
    for v in views:
        if v == "esmc_mean_pool":
            tensors.append(stack_esmc_baseline(cache, accs, esmc_embs))
        else:
            tensors.append(stack_view(cache, accs, v))
    return torch.cat(tensors, dim=-1)


def make_variants() -> List[Tuple[str, List[str]]]:
    return [
        ("A_prompteol_l48",       ["view_a_prompteol_l48"]),
        ("B_question_l48",        ["view_b_question_mean_l48"]),
        ("C_eos_l64",             ["view_c_eos_l64"]),
        ("A+B",                   ["view_a_prompteol_l48", "view_b_question_mean_l48"]),
        ("A+C",                   ["view_a_prompteol_l48", "view_c_eos_l64"]),
        ("B+C",                   ["view_b_question_mean_l48", "view_c_eos_l64"]),
        ("A+B+C (recommended)",   ["view_a_prompteol_l48", "view_b_question_mean_l48", "view_c_eos_l64"]),
        ("ESMC_baseline",         ["esmc_mean_pool"]),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--kind", default="mlp", choices=["linear", "mlp", "both"])
    ap.add_argument("--allow-partial", action="store_true",
                    help="run even if only benchmark embeddings exist; use CV on benchmark")
    args = ap.parse_args()

    ref_cache_path = EMB_DIR / "reference_embeddings.pt"
    bench_cache_path = EMB_DIR / "benchmark_embeddings.pt"

    if not bench_cache_path.exists():
        print(f"Missing {bench_cache_path} — run extract_probe_embeddings.py first")
        return 1

    bench_cache = load_cache(bench_cache_path)
    bench_accs = list(bench_cache.keys())
    print(f"Benchmark: {len(bench_accs)} proteins")

    # ESM C baseline — build from the RAG index
    rag_npz = EMB_DIR / "rag_index.npz"
    esmc_embs = None
    if rag_npz.exists():
        npz = np.load(rag_npz, allow_pickle=True)
        esmc_ids = npz["ids"].tolist()
        esmc_arr = npz["embs"]
        esmc_embs = dict(zip(esmc_ids, esmc_arr))

    have_ref = ref_cache_path.exists()
    if have_ref:
        ref_cache = load_cache(ref_cache_path)
        ref_accs = list(ref_cache.keys())
        # Keep only accessions present in the ESM C index (needed for the baseline)
        if esmc_embs is not None:
            ref_accs = [a for a in ref_accs if a in esmc_embs]
        print(f"Reference: {len(ref_accs)} proteins (use as probe TRAIN)")
    else:
        print("No reference embeddings yet — falling back to 80/20 split on benchmark (use --allow-partial)")
        if not args.allow_partial:
            return 1
        ref_cache = None
        ref_accs = []

    # Determine train/test accessions
    if have_ref and len(ref_accs) >= 200:
        # Reference is the training set; benchmark is the test set.
        train_cache = ref_cache
        train_accs = ref_accs
        test_cache = bench_cache
        test_accs = bench_accs
    else:
        # Fallback: 80/20 split on benchmark only. Poorer science but workable.
        print("Fallback: 80/20 split on benchmark.")
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(bench_accs))
        split = int(len(bench_accs) * 0.8)
        train_accs = [bench_accs[i] for i in idx[:split]]
        test_accs = [bench_accs[i] for i in idx[split:]]
        train_cache = test_cache = bench_cache

    variants = make_variants()
    # Filter variants that need the ESM C baseline if we don't have it for BOTH splits.
    def esmc_available_for(accs): return esmc_embs is not None and all(a in esmc_embs for a in accs)
    keep = []
    for name, views in variants:
        if "esmc_mean_pool" in views:
            if not (esmc_available_for(train_accs) and esmc_available_for(test_accs)):
                print(f"  skip {name}: ESM C embeddings not available for all proteins")
                continue
        keep.append((name, views))
    variants = keep

    tasks = ["is_enzyme", "ec_l1", "go_f_top20", "pfam_top20"]
    kinds = ["mlp"] if args.kind == "mlp" else (["linear"] if args.kind == "linear" else ["linear", "mlp"])

    results: List[ProbeResult] = []
    t0 = time.time()

    for variant_name, views in variants:
        x_train = build_variant(train_cache, train_accs, views, esmc_embs)
        x_test = build_variant(test_cache, test_accs, views, esmc_embs)
        in_dim = x_train.shape[1]
        print(f"\n=== variant {variant_name}   dim={in_dim} "
              f"n_train={x_train.shape[0]} n_test={x_test.shape[0]} ===")
        for kind in kinds:
            for task in tasks:
                y_train = load_labels(train_cache, train_accs, task)
                y_test = load_labels(test_cache, test_accs, task)
                r = train_one_probe(
                    x_train, y_train, x_test, y_test,
                    task=task, kind=kind, variant=variant_name,
                    device=args.device, epochs=args.epochs,
                )
                results.append(r)
                print(f"  {kind:6s}  {task:14s}  macro-AUC={r.auc_macro:.3f}")

    dt = time.time() - t0
    print(f"\nTotal probe-training time: {dt:.1f}s")

    # Save JSON + markdown
    out_json = EMB_DIR / "probe_results.json"
    serialized = [
        {
            "variant": r.variant, "task": r.task, "kind": r.kind,
            "auc_macro": r.auc_macro, "auc_per_class": r.auc_per_class,
            "n_train": r.n_train, "n_test": r.n_test,
        } for r in results
    ]
    out_json.write_text(json.dumps(serialized, indent=2))
    print(f"Wrote {out_json}")

    # Pivot table: variant × task (macro-AUC), grouped by kind
    print("\n" + "=" * 72)
    print("Summary — macro-AUC")
    print("=" * 72)
    for kind in kinds:
        print(f"\n[{kind}]")
        header = f"  {'variant':<28}"
        for t in tasks:
            header += f" {t:>14}"
        print(header)
        for variant_name, _ in variants:
            row = f"  {variant_name:<28}"
            for t in tasks:
                v = next((r for r in results if r.variant == variant_name and r.task == t and r.kind == kind), None)
                row += f" {v.auc_macro if v else float('nan'):>14.3f}"
            print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
