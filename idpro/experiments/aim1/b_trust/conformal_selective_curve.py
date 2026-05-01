"""
Show whether conformal prediction is actually "helping" — by computing the
accuracy-vs-coverage trade-off curve.

The right question isn't "does singleton accuracy = AUC?" (they measure
different things). The right question is: "when we restrict to the
high-confidence subset, does accuracy go UP?"

We compare three evaluation modes on the same test set:

  (A) Argmax accuracy on ALL test points (no selection) — baseline
  (B) Argmax accuracy on top-K most-confident points (threshold-free
      selective prediction via max-softmax ranking) — oracle curve
  (C) Argmax accuracy on the conformal SINGLETON subset (|set|=1 at a
      given α) — conformal selective prediction

If conformal is calibrated well, (C) at any given coverage ≈ (B) at the
same coverage, and BOTH are above (A). "Decreasing" happens only if you
misread singleton accuracy as if it were AUC.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.model.idpro.conformal import SplitConformalPredictor  # noqa: E402
from idpro.paths import AIM1_PROBE_DIR as DATA_DIR, FIGURES_DIR  # noqa: E402

EMB_DIR = DATA_DIR / "embeddings"
FIG_DIR = FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

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
    return torch.cat([torch.stack([cache[a][v].float() for a in accs]) for v in views], dim=-1)


def ec_label(cache, a):
    v = cache[a]["labels"]["ec_l1"]
    return 0 if v is None else int(v)


def train_probe(x, y, device, epochs=100, lr=1e-3, wd=1e-4):
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
def predict(probe, x, device):
    return torch.softmax(probe(x.to(device)), dim=-1).cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ref_cache = torch.load(EMB_DIR / "reference_embeddings.pt", map_location="cpu", weights_only=False)
    bench_cache = torch.load(EMB_DIR / "benchmark_embeddings.pt", map_location="cpu", weights_only=False)
    all_cache = {**ref_cache, **bench_cache}
    all_accs = list(ref_cache.keys()) + list(bench_cache.keys())
    all_labels = np.array([ec_label(ref_cache, a) for a in ref_cache.keys()] +
                          [ec_label(bench_cache, a) for a in bench_cache.keys()])

    # 60/20/20: train/calib/test
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(all_accs))
    n = len(all_accs)
    n_test = int(n * 0.2)
    n_cal = int(n * 0.2)
    test_idx = perm[:n_test]
    cal_idx = perm[n_test:n_test + n_cal]
    tr_idx = perm[n_test + n_cal:]

    x_tr = stack_views(all_cache, [all_accs[i] for i in tr_idx], VIEWS)
    y_tr = torch.tensor(all_labels[tr_idx], dtype=torch.long)
    x_cal = stack_views(all_cache, [all_accs[i] for i in cal_idx], VIEWS)
    y_cal = all_labels[cal_idx]
    x_te = stack_views(all_cache, [all_accs[i] for i in test_idx], VIEWS)
    y_te = all_labels[test_idx]
    print(f"Train/calib/test = {len(tr_idx)}/{len(cal_idx)}/{len(test_idx)}")

    probe = train_probe(x_tr, y_tr, device)
    p_cal = predict(probe, x_cal, device)
    p_te = predict(probe, x_te, device)

    # (A) Argmax accuracy overall
    argmax_te = p_te.argmax(axis=1)
    acc_all = float((argmax_te == y_te).mean())
    # AUC macro one-vs-rest
    per_class_auc = []
    for c in range(8):
        yt = (y_te == c).astype(int)
        if yt.sum() in (0, len(yt)):
            continue
        per_class_auc.append(roc_auc_score(yt, p_te[:, c]))
    auc_macro = float(np.mean(per_class_auc))
    print(f"\n(A) Overall argmax accuracy on {len(y_te)} test proteins: {acc_all:.3f}")
    print(f"(A) Overall macro-AUC on same set:                      {auc_macro:.3f}")

    # (B) Oracle selective curve — sort by max softmax and take top-K
    max_softmax = p_te.max(axis=1)
    order = np.argsort(-max_softmax)  # descending
    coverage_pts = np.linspace(0.1, 1.0, 19)
    oracle_acc = []
    oracle_auc = []
    for c in coverage_pts:
        k = max(1, int(len(y_te) * c))
        keep = order[:k]
        oracle_acc.append(float((argmax_te[keep] == y_te[keep]).mean()))
        # Macro AUC on the top-K subset
        pcs = []
        sy = y_te[keep]
        sp = p_te[keep]
        for cls in range(8):
            yt = (sy == cls).astype(int)
            if yt.sum() in (0, len(yt)):
                continue
            pcs.append(roc_auc_score(yt, sp[:, cls]))
        oracle_auc.append(float(np.mean(pcs)) if pcs else float("nan"))

    # (C) Conformal selective curve — sweep α, keep singleton subset
    alphas = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    nc_cal = 1 - np.array([p_cal[i, y_cal[i]] for i in range(len(y_cal))])

    print("\n(C) Conformal at varying α:")
    print(f"  {'alpha':>5}  {'tau':>6}  {'|set|':>6}  {'cov':>5}  "
          f"{'|set|=1 coverage':>18}  {'|set|=1 accuracy':>18}  {'|set|=1 macro-AUC':>18}")
    conformal_results = []
    for alpha in alphas:
        cp = SplitConformalPredictor().calibrate(nc_cal)
        tau = cp.threshold(alpha)
        nc_te = 1 - p_te
        sets = nc_te <= tau
        sizes = sets.sum(axis=1)
        singleton = sizes == 1
        # Coverage on ALL points
        covered = np.array([sets[i, y_te[i]] for i in range(len(y_te))])
        cov = float(covered.mean())
        single_frac = float(singleton.mean())
        if singleton.sum() > 0:
            single_acc = float((argmax_te[singleton] == y_te[singleton]).mean())
            # AUC on singleton subset
            try:
                sy = y_te[singleton]
                sp = p_te[singleton]
                pcs = []
                for c in range(8):
                    yt = (sy == c).astype(int)
                    if yt.sum() in (0, len(yt)):
                        continue
                    pcs.append(roc_auc_score(yt, sp[:, c]))
                single_auc = float(np.mean(pcs)) if pcs else float("nan")
            except Exception:
                single_auc = float("nan")
        else:
            single_acc = float("nan")
            single_auc = float("nan")
        print(f"  {alpha:>5.2f}  {tau:>6.3f}  {sizes.mean():>6.2f}  {cov:>5.3f}  "
              f"{single_frac:>18.3f}  {single_acc:>18.3f}  {single_auc:>18.3f}")
        conformal_results.append({
            "alpha": alpha, "tau": tau, "mean_set_size": float(sizes.mean()),
            "marginal_coverage": cov, "singleton_fraction": single_frac,
            "singleton_accuracy": single_acc, "singleton_macro_auc": single_auc,
        })

    # ---- Plot AUC-vs-coverage (compact, panel-ready) ----
    # Filter NaNs from conformal curve (small-subset AUC edge cases)
    conf_cov = np.array([r["singleton_fraction"] for r in conformal_results])
    conf_auc = np.array([r["singleton_macro_auc"] for r in conformal_results])
    oracle_cov = np.array(coverage_pts)
    oracle_auc_arr = np.array(oracle_auc)

    mask_c = np.isfinite(conf_auc) & (conf_cov > 0)
    mask_o = np.isfinite(oracle_auc_arr)

    # Sort by coverage for clean line plot
    sc = np.argsort(conf_cov[mask_c])
    so = np.argsort(oracle_cov[mask_o])

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(oracle_cov[mask_o][so], oracle_auc_arr[mask_o][so], "-o",
            color="#2ca02c", lw=2.0, markersize=4.5, label="Oracle selective")
    ax.plot(conf_cov[mask_c][sc], conf_auc[mask_c][sc], "-s",
            color="#1f77b4", lw=2.0, markersize=5.0, label="Conformal")
    ax.axhline(auc_macro, color="#d62728", ls="--", lw=1.5,
               label=f"All points ({auc_macro:.2f})")

    # Find the peak on the conformal curve, then trim the x-axis to start there.
    conf_cov_sorted = conf_cov[mask_c][sc]
    conf_auc_sorted = conf_auc[mask_c][sc]
    peak_i = int(np.argmax(conf_auc_sorted))
    x_peak = float(conf_cov_sorted[peak_i])
    # Tiny pad to keep the peak marker visible
    x_lo = max(0.0, x_peak - 0.01)

    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Macro-AUC", fontsize=12)
    ax.set_xlim(x_lo, 1.02)
    # Tighten y-axis around the visible range for maximum contrast
    y_vals = np.concatenate([conf_auc_sorted, oracle_auc_arr[mask_o]])
    y_lo = float(min(auc_macro, np.nanmin(y_vals))) - 0.002
    y_hi = float(np.nanmax(y_vals)) + 0.002
    # Round to clean limits
    y_lo = np.floor(y_lo * 100) / 100  # e.g. 0.91
    y_hi = np.ceil(y_hi * 200) / 200   # e.g. 0.950 → 0.95
    ax.set_ylim(y_lo, y_hi)
    # Pick xticks that land inside the visible range
    tick_candidates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_xticks([t for t in tick_candidates if t >= x_lo - 1e-6])
    # Tighter y-ticks (0.01 spacing)
    ytick_candidates = np.arange(0.90, 0.971, 0.01)
    ax.set_yticks([round(t, 2) for t in ytick_candidates if y_lo - 1e-6 <= t <= y_hi + 1e-6])
    ax.tick_params(axis="both", labelsize=10)
    # Legend tucked just above the red baseline line. Transparent background
    # so underlying curve points remain visible even when they overlap.
    red_y = auc_macro  # 0.908
    # Anchor the legend above the red line in data coords
    legend = ax.legend(loc="lower left",
                       bbox_to_anchor=(0.02, (red_y + 0.003 - y_lo) / (y_hi - y_lo)),
                       bbox_transform=ax.transAxes,
                       fontsize=8.5, frameon=True, framealpha=0.55,
                       handletextpad=0.4, labelspacing=0.25,
                       handlelength=1.6, borderpad=0.3)
    legend.get_frame().set_edgecolor("#bbb")
    legend.get_frame().set_linewidth(0.5)
    ax.grid(alpha=0.3)

    out = FIG_DIR / "conformal_selective_curve.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nWrote {out}")
    plt.close(fig)

    # Save numbers
    out_json = EMB_DIR / "conformal_selective_curve.json"
    out_json.write_text(json.dumps({
        "test_n": int(len(y_te)),
        "argmax_accuracy_all": acc_all,
        "macro_auc_all": auc_macro,
        "oracle_curve_acc": list(zip(coverage_pts.tolist(), oracle_acc)),
        "oracle_curve_auc": list(zip(coverage_pts.tolist(), oracle_auc)),
        "conformal_curve": conformal_results,
    }, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
