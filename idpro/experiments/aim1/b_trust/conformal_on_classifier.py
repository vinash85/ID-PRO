"""
Wrap IDPro classifier-probe probabilities with split conformal prediction
(from `idpro/idpro/conformal.py`), then assess the HIGH-CONFIDENCE subset:
proteins for which the conformal prediction set contains a single class
(the strongest signal).

Pipeline:
  1. Train the A+B+C linear probe on 80% of the labeled pool; hold out 20%
     as the calibration set (no data leakage).
  2. For each test point (benchmark + dark proteome), compute the conformal
     prediction set at α ∈ {0.05, 0.10, 0.20}.
  3. Report:
       - coverage  = P(true class ∈ set)       (marginal guarantee ≥ 1-α)
       - set size  = mean number of classes in the set
       - singleton-set accuracy = when |set|=1, how often is it correct?
       - fraction of test points that are in the "high-confidence" (|set|=1) band
  4. Compare against standard argmax accuracy on the same splits.

Output: `preliminary_data/reports/CONFORMAL_CLASSIFIER_RESULTS.md` and
`idpro/data/probe/embeddings/conformal_classifier_results.json`.

Run:
    python scripts/conformal_on_classifier.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.model.idpro.conformal import SplitConformalPredictor  # noqa: E402
from idpro.paths import AIM1_PROBE_DIR as DATA_DIR, DATA_ROOT, REPORTS_DIR  # noqa: E402

EMB_DIR = DATA_DIR / "embeddings"
REPORT_DIR = REPORTS_DIR
REPORT_DIR.mkdir(parents=True, exist_ok=True)

VIEWS = ["view_a_prompteol_l48", "view_b_question_mean_l48", "view_c_eos_l64"]
CLASS_NAMES = {
    0: "Non-enzyme", 1: "Oxidoreductase", 2: "Transferase", 3: "Hydrolase",
    4: "Lyase", 5: "Isomerase", 6: "Ligase", 7: "Translocase",
}


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
    in_dim = x.shape[1]
    probe = LinearProbe(in_dim, 8).to(device)
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


def conformal_sets(cal_scores: np.ndarray, test_scores: np.ndarray, alpha: float):
    """
    APS-style conformal sets for multiclass: nonconformity(x, y) = 1 - P(y|x).
    For each test point, prediction set = { c : (1 - P(c|x)) <= tau }.
    """
    cp = SplitConformalPredictor().calibrate(cal_scores)
    tau = cp.threshold(alpha)
    # test_scores is (N, C); set membership mask is (N, C)
    nc = 1.0 - test_scores
    sets = nc <= tau  # (N, C) boolean
    return sets, float(tau)


def evaluate(y_true, probs, sets):
    """Report coverage, mean set size, singleton accuracy, argmax accuracy."""
    N, C = probs.shape
    set_sizes = sets.sum(axis=1)
    covered = np.array([sets[i, int(y_true[i])] for i in range(N)])
    coverage = float(covered.mean())
    mean_set = float(set_sizes.mean())
    singleton_mask = set_sizes == 1
    if singleton_mask.sum() == 0:
        singleton_acc = float("nan")
        singleton_frac = 0.0
    else:
        # singleton set = {argmax}; accuracy = P(argmax == true | singleton)
        argmax = probs.argmax(axis=1)
        correct = (argmax[singleton_mask] == y_true[singleton_mask]).sum()
        singleton_acc = float(correct / singleton_mask.sum())
        singleton_frac = float(singleton_mask.mean())
    argmax_all_acc = float((probs.argmax(axis=1) == y_true).mean())
    empty_frac = float((set_sizes == 0).mean())
    return {
        "coverage": coverage,
        "mean_set_size": mean_set,
        "singleton_fraction": singleton_frac,
        "singleton_accuracy": singleton_acc,
        "empty_fraction": empty_frac,
        "argmax_accuracy_all": argmax_all_acc,
    }


# ---------------------------------------------------------------------------
# Weak EC labels on dark (reused from evaluate_ec_classifier.py)
# ---------------------------------------------------------------------------


def build_dark_weak_labels():
    import csv, re
    EC_KEYWORDS = {
        1: ["oxidoreductase activity", "dehydrogenase", "reductase", "oxidase", "oxygenase", "peroxidase"],
        2: ["transferase activity", "kinase activity", "methyltransferase", "acyltransferase", "glycosyltransferase"],
        3: ["hydrolase activity", "peptidase", "protease", "nuclease", "phosphatase", "esterase", "lipase", "glycosidase"],
        4: ["lyase activity", "decarboxylase", "aldolase", "dehydratase", "synthase"],
        5: ["isomerase activity", "racemase", "epimerase", "mutase"],
        6: ["ligase activity", "synthetase"],
        7: ["transporter activity", "transmembrane transport", "channel activity", "permease"],
    }
    labels = {}
    meta = DATA_ROOT / "preliminary_data" / "dark_genome" / "dark_genome_metadata.tsv"
    with meta.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            go = (r.get("go_terms") or "").lower()
            if not go:
                continue
            matched = {ec for ec, kws in EC_KEYWORDS.items() if any(kw in go for kw in kws)}
            if not matched:
                labels[r["accession"]] = 0
            elif len(matched) == 1:
                labels[r["accession"]] = next(iter(matched))
    return labels


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

    # Combine reference + benchmark as labeled pool → train/cal split
    all_cache = {**ref_cache, **bench_cache}
    all_accs = ref_accs + bench_accs
    all_labels = np.array([ec_label(ref_cache, a) for a in ref_accs] +
                          [ec_label(bench_cache, a) for a in bench_accs])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(all_accs))
    n_cal = int(len(all_accs) * 0.2)
    cal_idx = perm[:n_cal]
    tr_idx = perm[n_cal:]
    cal_accs = [all_accs[i] for i in cal_idx]
    tr_accs = [all_accs[i] for i in tr_idx]
    y_cal = all_labels[cal_idx]
    y_tr = all_labels[tr_idx]
    print(f"Train: {len(tr_accs)}  Calibration: {len(cal_accs)}")

    # Train probe on train set
    x_tr = stack_views(all_cache, tr_accs, VIEWS)
    probe = train_probe(x_tr, torch.tensor(y_tr, dtype=torch.long), device=device)

    # Compute probs on calibration + benchmark + dark
    x_cal = stack_views(all_cache, cal_accs, VIEWS)
    p_cal = predict(probe, x_cal, device)

    # Nonconformity on calibration = 1 - P(true|x)
    nc_cal = 1 - np.array([p_cal[i, y_cal[i]] for i in range(len(y_cal))])

    # Benchmark test set: use bench proteins NOT in tr set.
    # Since we split all_accs (ref+bench), some benchmark proteins are in train
    # and some in calibration. For a fair test evaluation, use only the
    # calibration-held-out benchmark proteins.
    test_bench_accs = [a for a in bench_accs if a in cal_accs]
    y_test_bench = np.array([ec_label(bench_cache, a) for a in test_bench_accs])
    x_test_bench = stack_views(bench_cache, test_bench_accs, VIEWS)
    p_test_bench = predict(probe, x_test_bench, device)

    # Dark weak labels
    dark_weak = build_dark_weak_labels()
    test_dark_accs = [a for a in dark_accs if a in dark_weak]
    y_test_dark = np.array([dark_weak[a] for a in test_dark_accs])
    x_test_dark = stack_views(dark_cache, test_dark_accs, VIEWS)
    p_test_dark = predict(probe, x_test_dark, device)

    print(f"\nEvaluation splits:")
    print(f"  benchmark-held-out (from 20% calib split): {len(test_bench_accs)}")
    print(f"  dark with weak labels: {len(test_dark_accs)}")

    results = {"alphas": {}}
    print("\n=== Conformal metrics ===")
    print(f"  {'alpha':>6} | {'coverage':>9} {'mean_set':>9} {'singletonF':>11} {'singAcc':>8} {'empty':>7} | (benchmark)")
    for alpha in [0.05, 0.10, 0.20]:
        sets_bench, tau = conformal_sets(nc_cal, p_test_bench, alpha)
        m_bench = evaluate(y_test_bench, p_test_bench, sets_bench)
        print(f"  {alpha:>6.2f} | {m_bench['coverage']:>9.3f} "
              f"{m_bench['mean_set_size']:>9.3f} {m_bench['singleton_fraction']:>11.3f} "
              f"{m_bench['singleton_accuracy']:>8.3f} {m_bench['empty_fraction']:>7.3f} | tau={tau:.3f}")
        sets_dark, _ = conformal_sets(nc_cal, p_test_dark, alpha)
        m_dark = evaluate(y_test_dark, p_test_dark, sets_dark)
        print(f"         | {m_dark['coverage']:>9.3f} "
              f"{m_dark['mean_set_size']:>9.3f} {m_dark['singleton_fraction']:>11.3f} "
              f"{m_dark['singleton_accuracy']:>8.3f} {m_dark['empty_fraction']:>7.3f} | (dark)")
        results["alphas"][str(alpha)] = {
            "tau": tau,
            "benchmark": m_bench,
            "dark": m_dark,
        }

    # ---- High-confidence subset analysis (using α=0.10) ----
    print("\n=== High-confidence subset (α=0.10, singleton prediction sets) ===")
    alpha = 0.10
    cp_tau = results["alphas"]["0.1"]["tau"]
    sets_bench, _ = conformal_sets(nc_cal, p_test_bench, alpha)
    sets_dark, _ = conformal_sets(nc_cal, p_test_dark, alpha)
    bench_sizes = sets_bench.sum(axis=1)
    dark_sizes = sets_dark.sum(axis=1)

    print(f"  Benchmark set-size distribution: {dict(sorted(Counter(bench_sizes).items()))}")
    print(f"  Dark set-size distribution:      {dict(sorted(Counter(dark_sizes).items()))}")

    # Per-class accuracy within the singleton subset
    singleton_mask_bench = bench_sizes == 1
    if singleton_mask_bench.sum() > 0:
        argmax_bench = p_test_bench.argmax(axis=1)
        print(f"\n  Benchmark singleton-set accuracy by predicted class:")
        for c in range(8):
            pred_c_mask = (argmax_bench == c) & singleton_mask_bench
            if pred_c_mask.sum() == 0:
                continue
            correct = (y_test_bench[pred_c_mask] == c).sum()
            acc = correct / pred_c_mask.sum()
            print(f"    {c} {CLASS_NAMES[c]:15s}: {int(correct)}/{int(pred_c_mask.sum())} ({acc*100:.1f}%)")

    # Dark genome: no hard ground truth, but we can report how many get flagged
    # as "confident" (singleton set) vs "abstain" (empty set or |set|>1)
    print(f"\n  Dark-genome triage (α=0.10):")
    singleton_dark = int(dark_sizes == 1 .sum()) if False else int((dark_sizes == 1).sum())
    abstain_dark = int((dark_sizes != 1).sum())
    print(f"    High-confidence (|set|=1):   {singleton_dark}/{len(dark_sizes)} "
          f"({singleton_dark/len(dark_sizes)*100:.1f}%)")
    print(f"    Abstain / review (|set|≠1): {abstain_dark}/{len(dark_sizes)}")

    # Dark high-confidence predictions: which EC classes do they go to?
    argmax_dark = p_test_dark.argmax(axis=1)
    print(f"\n  High-confidence dark predictions by class:")
    hc_dark_pred_dist = Counter()
    for i, sz in enumerate(dark_sizes):
        if sz == 1:
            hc_dark_pred_dist[int(argmax_dark[i])] += 1
    for c in range(8):
        n = hc_dark_pred_dist.get(c, 0)
        if n == 0:
            continue
        # Weak-label agreement on this subset
        hc_idx = np.where(dark_sizes == 1)[0]
        mask = argmax_dark[hc_idx] == c
        if mask.sum() > 0:
            weak = y_test_dark[hc_idx][mask]
            weak_agree = (weak == c).sum()
            print(f"    {c} {CLASS_NAMES[c]:15s}: n={n:3d}  "
                  f"weak-label agreement: {int(weak_agree)}/{int(mask.sum())} "
                  f"({weak_agree/mask.sum()*100:.1f}%)")

    results["high_confidence"] = {
        "alpha": alpha,
        "tau": cp_tau,
        "benchmark": {
            "set_size_distribution": {int(k): int(v) for k, v in Counter(bench_sizes).items()},
            "singleton_fraction": float((bench_sizes == 1).mean()),
        },
        "dark": {
            "set_size_distribution": {int(k): int(v) for k, v in Counter(dark_sizes).items()},
            "singleton_fraction": float((dark_sizes == 1).mean()),
            "singleton_class_distribution": {int(k): int(v) for k, v in hc_dark_pred_dist.items()},
        },
    }

    # Save
    out_path = EMB_DIR / "conformal_classifier_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
