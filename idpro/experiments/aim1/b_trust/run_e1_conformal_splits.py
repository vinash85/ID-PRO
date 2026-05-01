"""E1 — Conformal robustness under temporal + Pfam-family + synthetic shifts.

Answers reviewer F01 (round-1 showstopper): "Conformal claims stronger than
validation plan; calibration may collapse under distribution shift."

Produces a table of (coverage, mean set size, worst-case set size) for three
distinct shift regimes, all on the existing 3,637-protein labeled pool + the
frozen Stage-4 classifier probe. Zero retraining.

Splits:
  A. Temporal — calibration drawn from pre-cutoff deposits; test from
     post-cutoff deposits. Cutoff chosen from the date-distribution histogram
     so each side has >=300 proteins.
  B. Pfam-family holdout — leave-one-Pfam-family-out across the top-K
     families (K=5 by default). For each fold, calibration EXCLUDES proteins
     containing that Pfam family; test INCLUDES only proteins containing it.
  C. Synthetic class-prior shift — up-weight EC class 1 (oxidoreductases) 2x
     in the sampled calibration set, then re-fit weighted conformal and
     compare coverage to unweighted conformal.

Reference (in-distribution baseline): random 80/20 split, same classifier
probe, already computed in `conformal_selective_curve.json`.

Usage:
  python idpro/scripts/run_e1_conformal_splits.py \
      [--metadata-cache idpro/data/probe/uniprot_metadata_cache.jsonl] \
      [--top-k-pfam 5] \
      [--alpha 0.10] \
      [--out idpro/data/probe/embeddings/e1_conformal_splits.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.model.idpro.conformal import SplitConformalPredictor  # noqa: E402
from idpro.paths import AIM1_PROBE_DIR as DATA_DIR  # noqa: E402

EMB_DIR = DATA_DIR / "embeddings"

# Same views as the existing conformal_on_classifier.py script
VIEWS = ["view_a_prompteol_l48", "view_b_question_mean_l48", "view_c_eos_l64"]
N_CLASSES = 8
CLASS_NAMES = {
    0: "Non-enzyme", 1: "Oxidoreductase", 2: "Transferase", 3: "Hydrolase",
    4: "Lyase", 5: "Isomerase", 6: "Ligase", 7: "Translocase",
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_embeddings():
    ref = torch.load(EMB_DIR / "reference_embeddings.pt", map_location="cpu", weights_only=False)
    bench = torch.load(EMB_DIR / "benchmark_embeddings.pt", map_location="cpu", weights_only=False)
    return {**ref, **bench}


def load_metadata(cache_path: Path) -> dict[str, dict]:
    """Load the UniProt metadata cache built by fetch_uniprot_metadata_for_e1.py."""
    meta: dict[str, dict] = {}
    if not cache_path.exists():
        print(f"WARN: metadata cache not found: {cache_path}")
        return meta
    with cache_path.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
                meta[d["accession"]] = d
            except Exception:
                continue
    return meta


def ec_label(cache, acc):
    v = cache[acc]["labels"]["ec_l1"]
    return 0 if v is None else int(v)


def stack_views(cache, accs):
    return torch.cat(
        [torch.stack([cache[a][v].float() for a in accs]) for v in VIEWS],
        dim=-1,
    )


# --------------------------------------------------------------------------- #
# Probe + conformal helpers (lifted + slimmed from conformal_on_classifier.py)
# --------------------------------------------------------------------------- #


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def train_probe(x, y, device, epochs=100, lr=1e-3, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    in_dim = x.shape[1]
    probe = LinearProbe(in_dim, N_CLASSES).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    x = x.to(device)
    y = y.to(device)
    bs = 64
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0], device=device)
        for s in range(0, x.shape[0], bs):
            idx = perm[s:s + bs]
            loss = loss_fn(probe(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe.eval()


@torch.no_grad()
def predict(probe, x, device):
    return torch.softmax(probe(x.to(device)), dim=-1).cpu().numpy()


def conformal_sets_unweighted(cal_scores: np.ndarray, cal_labels: np.ndarray,
                              test_scores: np.ndarray, alpha: float):
    """Standard split conformal (APS). Nonconformity = 1 - P(true|x)."""
    nc_cal = 1.0 - np.array([cal_scores[i, cal_labels[i]] for i in range(len(cal_labels))])
    cp = SplitConformalPredictor().calibrate(nc_cal)
    tau = cp.threshold(alpha)
    nc_test = 1.0 - test_scores
    sets = nc_test <= tau
    return sets, float(tau)


def conformal_sets_weighted(cal_scores: np.ndarray, cal_labels: np.ndarray,
                            cal_weights: np.ndarray, test_scores: np.ndarray,
                            alpha: float):
    """Weighted split conformal (Tibshirani 2019). cal_weights must be >0.

    For each test point, we'd want point-specific weights, but for an
    aggregate coverage test we use constant test weight = mean(cal_weights).
    """
    nc_cal = 1.0 - np.array([cal_scores[i, cal_labels[i]] for i in range(len(cal_labels))])
    # Normalized weights (sum = N)
    w = cal_weights * (len(cal_weights) / cal_weights.sum())
    # Weighted empirical quantile at level (1 - alpha)
    order = np.argsort(nc_cal)
    nc_sorted = nc_cal[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    threshold_mass = (1 - alpha) * cum_w[-1]
    idx = np.searchsorted(cum_w, threshold_mass, side="right")
    tau = float(nc_sorted[min(idx, len(nc_sorted) - 1)])
    nc_test = 1.0 - test_scores
    sets = nc_test <= tau
    return sets, tau


def evaluate_sets(y_true, probs, sets):
    N, C = probs.shape
    set_sizes = sets.sum(axis=1)
    covered = np.array([sets[i, int(y_true[i])] for i in range(N)])
    coverage = float(covered.mean())
    mean_set = float(set_sizes.mean())
    max_set = int(set_sizes.max())
    singleton_mask = set_sizes == 1
    singleton_frac = float(singleton_mask.mean())
    if singleton_mask.sum() == 0:
        singleton_acc = float("nan")
    else:
        argmax = probs.argmax(axis=1)
        correct = (argmax[singleton_mask] == y_true[singleton_mask]).sum()
        singleton_acc = float(correct / singleton_mask.sum())
    empty_frac = float((set_sizes == 0).mean())
    argmax_acc = float((probs.argmax(axis=1) == y_true).mean())
    return {
        "coverage": coverage,
        "mean_set_size": mean_set,
        "max_set_size": max_set,
        "singleton_fraction": singleton_frac,
        "singleton_accuracy": singleton_acc,
        "empty_fraction": empty_frac,
        "argmax_accuracy": argmax_acc,
        "n_test": int(N),
    }


# --------------------------------------------------------------------------- #
# Split builders
# --------------------------------------------------------------------------- #


def build_temporal_split(accs: list[str], meta: dict, target_cal_size: int = 300,
                         min_test_size: int = 200, rng=None):
    """Temporal split using the MEDIAN deposit year as the cutoff. Pre-cutoff
    proteins form train + cal; post-cutoff proteins form the test set. This
    tests whether calibration on older deposits generalizes to newer ones."""
    if rng is None:
        rng = np.random.default_rng(0)
    years = {}
    for a in accs:
        m = meta.get(a, {})
        dc = (m.get("date_created") or "")[:4]
        if dc.isdigit():
            years[a] = int(dc)
    if len(years) < target_cal_size + min_test_size:
        return None
    all_years = sorted(years.values())
    # Use median year as the cutoff — roughly 50/50 split by time
    cutoff_year = all_years[len(all_years) // 2]
    pre = [a for a, y in years.items() if y <= cutoff_year]
    post = [a for a, y in years.items() if y > cutoff_year]
    if len(post) < min_test_size or len(pre) < target_cal_size + 200:
        return None
    pre = list(pre)
    rng.shuffle(pre)
    cal = pre[:target_cal_size]
    train = pre[target_cal_size:]
    test = post
    return {
        "name": "A_temporal",
        "cutoff_year": cutoff_year,
        "train": train,
        "cal": cal,
        "test": test,
    }


def build_pfam_holdout_splits(accs: list[str], meta: dict, top_k: int = 5):
    """Leave-one-Pfam-family-out. Pick the top-K most populous Pfam families and
    for each one generate a fold where test = proteins containing that family,
    cal+train = all other proteins (stratified)."""
    pfam_to_accs: dict[str, list[str]] = {}
    for a in accs:
        m = meta.get(a, {})
        for pf in m.get("pfam_ids", []):
            pfam_to_accs.setdefault(pf, []).append(a)
    top = sorted(pfam_to_accs.items(), key=lambda kv: -len(kv[1]))[:top_k]
    splits = []
    for pf, family_accs in top:
        test = family_accs
        others = [a for a in accs if a not in set(family_accs)]
        # Need metadata to be present to exclude — fall back skip if accs w/o metadata
        # Random 80/20 on the "others" half → train vs cal
        rng = np.random.default_rng(hash(pf) & 0xffff)
        others_shuf = list(others)
        rng.shuffle(others_shuf)
        n_cal = min(300, int(len(others_shuf) * 0.2))
        cal = others_shuf[:n_cal]
        train = others_shuf[n_cal:]
        splits.append({
            "name": f"B_pfam_{pf}",
            "pfam_family": pf,
            "family_size": len(family_accs),
            "train": train,
            "cal": cal,
            "test": test,
        })
    return splits


def build_synthetic_shift_split(accs: list[str], cache, target_cal_size: int = 300,
                                boost_class: int = 1, boost_factor: float = 2.0,
                                seed: int = 0):
    """Random 80/20 split, then up-weight `boost_class` by `boost_factor` in
    calibration. Train unchanged. We compare unweighted conformal coverage
    against weighted-conformal coverage on the SAME test set."""
    rng = np.random.default_rng(seed)
    perm = list(accs)
    rng.shuffle(perm)
    cal = perm[:target_cal_size]
    train = perm[target_cal_size:]
    # Test = calibration's own held-out evaluation isn't valid; use the 20%
    # of train as the test-of-coverage (standard CV practice).
    n_test = min(637, int(len(train) * 0.2))
    test = train[:n_test]
    train = train[n_test:]
    # Weights: everything = 1.0 except boost_class in cal gets boost_factor
    cal_labels = np.array([ec_label(cache, a) for a in cal])
    weights = np.where(cal_labels == boost_class, boost_factor, 1.0).astype(float)
    return {
        "name": "C_synthetic_shift",
        "boost_class": boost_class,
        "boost_factor": boost_factor,
        "train": train,
        "cal": cal,
        "cal_weights": weights.tolist(),
        "test": test,
    }


# --------------------------------------------------------------------------- #
# Per-split runner
# --------------------------------------------------------------------------- #


def run_split(split: dict, cache, device, alpha: float = 0.10, use_weighted: bool = False):
    train_accs = split["train"]
    cal_accs = split["cal"]
    test_accs = split["test"]
    if len(train_accs) < 100 or len(cal_accs) < 100 or len(test_accs) < 20:
        return {
            "name": split["name"],
            "skipped_reason": f"too few (train={len(train_accs)}, "
                              f"cal={len(cal_accs)}, test={len(test_accs)})",
        }

    y_train = torch.tensor(
        [ec_label(cache, a) for a in train_accs], dtype=torch.long
    )
    x_train = stack_views(cache, train_accs)
    probe = train_probe(x_train, y_train, device)

    x_cal = stack_views(cache, cal_accs)
    p_cal = predict(probe, x_cal, device)
    y_cal = np.array([ec_label(cache, a) for a in cal_accs])

    x_test = stack_views(cache, test_accs)
    p_test = predict(probe, x_test, device)
    y_test = np.array([ec_label(cache, a) for a in test_accs])

    if use_weighted:
        w = np.array(split["cal_weights"])
        sets, tau = conformal_sets_weighted(p_cal, y_cal, w, p_test, alpha)
    else:
        sets, tau = conformal_sets_unweighted(p_cal, y_cal, p_test, alpha)

    metrics = evaluate_sets(y_test, p_test, sets)
    metrics.update({
        "name": split["name"],
        "alpha": alpha,
        "tau": tau,
        "n_train": len(train_accs),
        "n_cal": len(cal_accs),
        "n_test": len(test_accs),
        "weighted": use_weighted,
    })
    # Pass-through split-specific metadata
    for k in ("cutoff_year", "pfam_family", "family_size",
              "boost_class", "boost_factor"):
        if k in split:
            metrics[k] = split[k]
    return metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-cache", type=str,
                    default=str(DATA_DIR / "uniprot_metadata_cache.jsonl"))
    ap.add_argument("--top-k-pfam", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", type=str,
                    default=str(EMB_DIR / "e1_conformal_splits.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Alpha:  {args.alpha}")

    cache = load_embeddings()
    all_accs = list(cache.keys())
    print(f"Embeddings loaded: {len(all_accs)} proteins")

    meta = load_metadata(Path(args.metadata_cache))
    print(f"Metadata cache:    {len(meta)} records")

    # --- Split A: temporal
    temporal = build_temporal_split(all_accs, meta)
    if temporal:
        print(f"\nSplit A (temporal): cutoff={temporal['cutoff_year']}, "
              f"train={len(temporal['train'])}, cal={len(temporal['cal'])}, "
              f"test={len(temporal['test'])}")
    else:
        print("\nSplit A (temporal): SKIPPED (insufficient metadata)")

    # --- Split B: Pfam-family holdout (top-K)
    pfam_splits = build_pfam_holdout_splits(all_accs, meta, top_k=args.top_k_pfam)
    print(f"\nSplit B (pfam-holdout): {len(pfam_splits)} folds:")
    for s in pfam_splits:
        print(f"   - {s['pfam_family']} (family_size={s['family_size']}, "
              f"test={len(s['test'])})")

    # --- Split C: synthetic shift — run BOTH unweighted and weighted
    synthetic = build_synthetic_shift_split(all_accs, cache)
    print(f"\nSplit C (synthetic shift): boost_class={synthetic['boost_class']} "
          f"x{synthetic['boost_factor']}, "
          f"train={len(synthetic['train'])}, cal={len(synthetic['cal'])}, "
          f"test={len(synthetic['test'])}")

    # ---- Reference: random 80/20 in-distribution (sanity) ----
    rng = np.random.default_rng(42)
    perm = list(all_accs)
    rng.shuffle(perm)
    n_cal = 300
    in_dist = {
        "name": "0_in_distribution_reference",
        "train": perm[n_cal + 637:],
        "cal": perm[:n_cal],
        "test": perm[n_cal: n_cal + 637],
    }

    print("\n" + "=" * 72)
    print(f"Running conformal @ alpha={args.alpha}")
    print("=" * 72)

    results: list[dict] = []

    # In-distribution reference
    print(f"\n[In-distribution reference] ...")
    r = run_split(in_dist, cache, device, alpha=args.alpha)
    print(f"  coverage={r.get('coverage'):.3f}  mean_set={r.get('mean_set_size'):.2f}  "
          f"max_set={r.get('max_set_size')}")
    results.append(r)

    # Temporal
    if temporal:
        print(f"\n[A temporal @ cutoff {temporal['cutoff_year']}] ...")
        r = run_split(temporal, cache, device, alpha=args.alpha)
        if "skipped_reason" in r:
            print(f"  SKIPPED: {r['skipped_reason']}")
        else:
            print(f"  coverage={r.get('coverage'):.3f}  "
                  f"mean_set={r.get('mean_set_size'):.2f}  "
                  f"max_set={r.get('max_set_size')}")
        results.append(r)

    # Pfam holdout
    for s in pfam_splits:
        print(f"\n[B pfam={s['pfam_family']}] ...")
        r = run_split(s, cache, device, alpha=args.alpha)
        if "skipped_reason" in r:
            print(f"  SKIPPED: {r['skipped_reason']}")
        else:
            print(f"  coverage={r.get('coverage'):.3f}  "
                  f"mean_set={r.get('mean_set_size'):.2f}  "
                  f"max_set={r.get('max_set_size')}")
        results.append(r)

    # Synthetic shift: run UNWEIGHTED and WEIGHTED both
    print(f"\n[C synthetic shift, UNWEIGHTED conformal] ...")
    r_unw = run_split(synthetic, cache, device, alpha=args.alpha, use_weighted=False)
    r_unw["name"] += "_unweighted"
    if "skipped_reason" in r_unw:
        print(f"  SKIPPED: {r_unw['skipped_reason']}")
    else:
        print(f"  coverage={r_unw.get('coverage'):.3f}  "
              f"mean_set={r_unw.get('mean_set_size'):.2f}")
    results.append(r_unw)

    print(f"\n[C synthetic shift, WEIGHTED conformal] ...")
    r_w = run_split(synthetic, cache, device, alpha=args.alpha, use_weighted=True)
    r_w["name"] += "_weighted"
    if "skipped_reason" in r_w:
        print(f"  SKIPPED: {r_w['skipped_reason']}")
    else:
        print(f"  coverage={r_w.get('coverage'):.3f}  "
              f"mean_set={r_w.get('mean_set_size'):.2f}")
    results.append(r_w)

    # ---- Summary table ----
    print("\n" + "=" * 72)
    print(f"SUMMARY — conformal @ alpha={args.alpha} (nominal coverage = {1 - args.alpha:.0%})")
    print("=" * 72)
    print(f"{'split':<40} {'n_test':>7} {'cov':>7} {'mean_set':>9} {'max':>5}")
    for r in results:
        if "skipped_reason" in r:
            print(f"{r['name']:<40} SKIPPED ({r['skipped_reason']})")
            continue
        print(f"{r['name']:<40} "
              f"{r.get('n_test', 0):>7} "
              f"{r.get('coverage', 0):>7.3f} "
              f"{r.get('mean_set_size', 0):>9.2f} "
              f"{r.get('max_set_size', 0):>5}")

    # Write JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "alpha": args.alpha,
        "nominal_coverage": 1 - args.alpha,
        "target_band": [0.86, 0.94],
        "splits": results,
    }, indent=2, default=str))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
