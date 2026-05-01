"""
Train EC-L1 classifier probes for all three IDPro ablations and emit the
headline Δ AUC percentages for the proposal.

Variants evaluated (all on the same 3,637-protein UniProt-labeled pool,
5-fold stratified CV, A+B+C linear probe):

  Baseline           : full per-residue + RAG       (from reference_embeddings.pt / benchmark_embeddings.pt)
  A1.k32             : 32-token summary + RAG       (from ablation_*_embeddings.pt, *_k32 views)
  A1.k1              : 1-token summary  + RAG       (ablation_*_embeddings.pt, *_k1 views)
  A2.norag           : full per-residue + no-RAG    (ablation_*_embeddings.pt, *_norag views)
  A3a.base+evidence  : baseline views + 9-d pre-evidence + 9-d post-evidence concat

Outputs:
  idpro/data/probe/embeddings/ablation_results.json
  console-printed headline table

Run:
    python scripts/train_ablation_probes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from idpro.paths import PROBE_DIR  # noqa: E402

EMB_DIR = PROBE_DIR / "embeddings"

BASE_VIEWS = ["view_a_prompteol_l48", "view_b_question_mean_l48", "view_c_eos_l64"]
VIEW_SUFFIXES = {
    "baseline": "",                                # existing reference_/benchmark_ caches
    "A1.k32":   "_k32",
    "A1.k1":    "_k1",
    "A2.norag": "_norag",
}


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)


def train_probe(x, y, device="cuda", epochs=100, lr=1e-3, wd=1e-4):
    probe = LinearProbe(x.shape[1], 8).to(device)
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
def predict(probe, x, device="cuda"):
    return torch.softmax(probe(x.to(device)), dim=-1).cpu().numpy()


def macro_auc(y_true, scores, n_classes=8):
    per = []
    for c in range(n_classes):
        yt = (y_true == c).astype(int)
        if yt.sum() in (0, len(yt)):
            continue
        per.append(roc_auc_score(yt, scores[:, c]))
    return float(np.mean(per)) if per else float("nan")


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------


def build_feature_matrix(
    accs: List[str],
    ref_cache: dict,
    bench_cache: dict,
    abl_ref: dict,
    abl_bench: dict,
    variant: str,
    add_evidence: bool = False,
) -> torch.Tensor:
    """
    Assemble the A+B+C concat for the given variant. Uses the baseline caches
    when variant == "baseline", else the ablation caches.
    """
    suffix = VIEW_SUFFIXES[variant]
    per_protein_vecs = []
    for a in accs:
        if a in ref_cache:
            src_base = ref_cache[a]
            src_abl = abl_ref.get(a)
        else:
            src_base = bench_cache[a]
            src_abl = abl_bench.get(a)
        if variant == "baseline":
            parts = [src_base[v].float() for v in BASE_VIEWS]
        else:
            if src_abl is None:
                raise KeyError(f"Ablation embedding missing for {a} (variant {variant})")
            parts = [src_abl[v + suffix].float() for v in BASE_VIEWS]
        if add_evidence:
            if src_abl is None or "evidence_pre_mean_9d" not in src_abl:
                raise KeyError(f"Evidence logits missing for {a}")
            parts.append(src_abl["evidence_pre_mean_9d"].float())
            parts.append(src_abl["evidence_post_mean_9d"].float())
        per_protein_vecs.append(torch.cat(parts))
    return torch.stack(per_protein_vecs)


def ec_label(cache, a):
    v = cache[a]["labels"]["ec_l1"]
    return 0 if v is None else int(v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ref_cache   = torch.load(EMB_DIR / "reference_embeddings.pt",  map_location="cpu", weights_only=False)
    bench_cache = torch.load(EMB_DIR / "benchmark_embeddings.pt",  map_location="cpu", weights_only=False)
    abl_ref     = torch.load(EMB_DIR / "ablation_reference_embeddings.pt", map_location="cpu", weights_only=False)
    abl_bench   = torch.load(EMB_DIR / "ablation_benchmark_embeddings.pt", map_location="cpu", weights_only=False)

    all_accs = list(ref_cache.keys()) + list(bench_cache.keys())
    # Keep only proteins that have ablation embeddings for ALL four variants
    present = set(abl_ref.keys()) | set(abl_bench.keys())
    before = len(all_accs)
    all_accs = [a for a in all_accs if a in present]
    if len(all_accs) < before:
        print(f"  dropped {before - len(all_accs)} proteins missing from ablation caches")
    print(f"Pool size: {len(all_accs)}  (ref={len(ref_cache)}, bench={len(bench_cache)})")

    all_labels = np.array([
        ec_label(ref_cache,   a) if a in ref_cache else ec_label(bench_cache, a)
        for a in all_accs
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_idx = list(skf.split(np.zeros(len(all_accs)), all_labels))

    experiments = [
        ("baseline",           "baseline",        False),
        ("A1.k32",             "A1.k32",          False),
        ("A1.k1",              "A1.k1",           False),
        ("A2.norag",           "A2.norag",        False),
        ("A3a.base+evidence",  "baseline",        True),
    ]

    results = {}
    print()
    print("Training probes across 5-fold CV per variant ...")
    for name, variant, add_ev in experiments:
        try:
            X = build_feature_matrix(all_accs, ref_cache, bench_cache,
                                     abl_ref, abl_bench, variant, add_evidence=add_ev)
        except KeyError as e:
            print(f"  [skip] {name}: {e}")
            continue
        y = torch.tensor(all_labels, dtype=torch.long)
        fold_aucs = []
        for fi, (tr, te) in enumerate(fold_idx):
            probe = train_probe(X[tr], y[tr], device=device)
            scores = predict(probe, X[te], device=device)
            auc = macro_auc(y[te].numpy(), scores)
            fold_aucs.append(auc)
        mean, std = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
        results[name] = {
            "macro_auc_mean": mean,
            "macro_auc_std":  std,
            "fold_aucs":      fold_aucs,
            "feature_dim":    int(X.shape[1]),
            "n_samples":      int(X.shape[0]),
        }
        print(f"  {name:<22s}  dim={X.shape[1]:<6d}  macro-AUC = {mean:.4f} ± {std:.4f}")

    # Δ computations
    print()
    print("=" * 70)
    print("Headline Δ AUC (vs ablated variant)")
    print("=" * 70)
    base = results.get("baseline", {}).get("macro_auc_mean")
    if base is None:
        print("No baseline present; cannot compute Δ.")
        return 0

    headlines = []
    def emit(ablated_name: str, description: str, comparator: str = "baseline"):
        comp_val = results.get(comparator, {}).get("macro_auc_mean")
        abl_val  = results.get(ablated_name, {}).get("macro_auc_mean")
        if comp_val is None or abl_val is None:
            return
        # For A1/A2 we want comp - abl (positive = IDPro better)
        # For A3a we want abl - comp (positive = +evidence better)
        if ablated_name == "A3a.base+evidence":
            delta = abl_val - comp_val
            denom = comp_val
        else:
            delta = comp_val - abl_val
            denom = abl_val
        pct = 100.0 * delta / max(denom, 1e-9)
        line = (f"  {description:<60s}  Δ={delta:+.4f}  ({pct:+.2f}% relative)"
                f"  [{comp_val:.4f} vs {abl_val:.4f}]")
        headlines.append(line)
        print(line)

    emit("A1.k32",            "A1  Per-residue vs 32-token summary + RAG")
    emit("A1.k1",             "A1  Per-residue vs  1-token summary + RAG")
    emit("A2.norag",          "A2  With-RAG vs without-RAG (per-residue)")
    emit("A3a.base+evidence", "A3a Probe + evidence features vs probe alone")

    (EMB_DIR / "ablation_results.json").write_text(json.dumps(results, indent=2))
    print()
    print(f"Saved: {EMB_DIR / 'ablation_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
