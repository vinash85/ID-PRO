"""
End-to-end driver for the frozen-IDPro classifier-probe experiments (Aim 1A).

Replaces the pre-refactor scripts:
  train_probe_variants.py, evaluate_ec_classifier.py, evaluate_probe_on_dark.py,
  idpro_vs_interlabelgo.py, fair_head_to_head.py.

Same building blocks (`utils/{data,probes,metrics}.py`) under every mode:
  variants     — sweep view-combo × probe-kind × task; train on REFERENCE,
                 evaluate on a chosen test split. Replaces train_probe_variants.
  cv5fold      — 5-fold stratified CV on EC L1 over the REFERENCE+BENCHMARK
                 labeled pool. Replaces the CV half of evaluate_ec_classifier.
  dark         — train on REFERENCE → score BENCHMARK + DARK on all four tasks,
                 record confidence percentiles, conformal set sizes (EC L1).
                 Replaces evaluate_probe_on_dark + the dark-eval half of
                 evaluate_ec_classifier.
  zeroshot637  — train on REFERENCE only, evaluate on BENCHMARK; the
                 apples-to-apples regime against InterLabelGO+ (CAFA5 winner).
                 Replaces idpro_vs_interlabelgo.
  fair         — train on (REFERENCE+BENCHMARK) MINUS the test subset; evaluate
                 on the canonical 92/125-protein intersections of baseline
                 coverage. Replaces fair_head_to_head (probe side only — the
                 baselines themselves come from BASELINE_PREDS_DIR, written by
                 run_baselines.py).

CLI
---
python idpro/experiments/aim1/probe_benchmarks/run_probe.py variants \
    --probe mlp --tasks all --eval-on benchmark
python idpro/experiments/aim1/probe_benchmarks/run_probe.py cv5fold \
    --views A+B+C --probes linear,mlp
python idpro/experiments/aim1/probe_benchmarks/run_probe.py dark \
    --views A+B+C --probes mlp,linear --conformal-alphas 0.05,0.1,0.2
python idpro/experiments/aim1/probe_benchmarks/run_probe.py zeroshot637 \
    --views A+B+C
python idpro/experiments/aim1/probe_benchmarks/run_probe.py fair \
    --views A+B+C --baselines p2t,rag_transfer,bioreason,deepfri,mmseqs,deepgometa

All modes write JSON to PROBE_RESULTS_DIR/<mode>.json by default.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.paths import (  # noqa: E402
    BASELINE_PREDS_DIR,
    EXTRACTED_EMBEDDINGS_DIR,
    PROBE_RESULTS_DIR,
)
from idpro.experiments.aim1.probe_benchmarks.utils import (  # noqa: E402
    CLASS_NAMES,
    ESMC_VIEW,
    N_CLASSES,
    VIEWS,
    compute_auc,
    deepgometa_strict_scores,
    ec_label,
    iter_variants,
    load_emb_cache,
    load_esmc_index,
    load_labels,
    macro_from_per_class,
    per_class_auc,
    predict,
    stack_views,
    strict_class_scores,
    task_out_dim,
    train_probe,
)

PROBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# View-set parsing
# ---------------------------------------------------------------------------

VIEW_SHORT: Dict[str, str] = {"A": VIEWS[0], "B": VIEWS[1], "C": VIEWS[2], "ESMC": ESMC_VIEW}


def parse_views(spec: str) -> List[str]:
    """Accept either an iter_variants name ('A+B+C', 'ESMC_baseline', ...) or a
    plain '+'-separated short-form ('A', 'A+C', 'A+B+ESMC')."""
    spec = spec.strip()
    for name, views in iter_variants(include_esmc=True):
        if spec == name or spec.replace(" ", "") == name.replace(" ", ""):
            return views
    out: List[str] = []
    for tok in spec.split("+"):
        tok = tok.strip().upper()
        if tok not in VIEW_SHORT:
            raise ValueError(f"unknown view token {tok!r} in {spec!r}")
        out.append(VIEW_SHORT[tok])
    return out


def parse_csv(spec: str, allowed: List[str]) -> List[str]:
    items = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [s for s in items if s not in allowed]
    if bad:
        raise ValueError(f"unknown items {bad}; allowed {allowed}")
    return items


# ---------------------------------------------------------------------------
# Cache loaders (all modes need at least benchmark; reference + dark optional)
# ---------------------------------------------------------------------------


def _emb_path(which: str) -> Path:
    return EXTRACTED_EMBEDDINGS_DIR / f"{which}_embeddings.pt"


def _load_split(which: str, *, required: bool) -> Optional[Dict[str, dict]]:
    p = _emb_path(which)
    if not p.exists():
        if required:
            raise SystemExit(
                f"Missing embedding cache {p}. Run "
                f"`extract_embeddings.py views --which {which} --ckpt ...` first."
            )
        return None
    return load_emb_cache(p)


def _load_esmc(needed: bool = True) -> Optional[Dict[str, np.ndarray]]:
    rag = EXTRACTED_EMBEDDINGS_DIR / "rag_index.npz"
    idx = load_esmc_index(rag)
    if idx is None and needed:
        print(f"  note: no rag_index.npz at {rag} — ESM C baseline view unavailable")
    return idx


# ---------------------------------------------------------------------------
# Tasks helpers
# ---------------------------------------------------------------------------

ALL_TASKS = ["is_enzyme", "ec_l1", "go_f_top20", "pfam_top20"]


def _resolve_tasks(spec: str) -> List[str]:
    if spec == "all":
        return list(ALL_TASKS)
    return parse_csv(spec, ALL_TASKS)


def _train_eval(
    train_cache: Dict[str, dict],
    train_accs: List[str],
    test_cache: Dict[str, dict],
    test_accs: List[str],
    *,
    views: List[str],
    task: str,
    kind: str,
    device: str,
    epochs: int,
    esmc: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[float, List[float], np.ndarray]:
    """Train one probe and return (macro_auc, per_class_auc, raw_scores)."""
    x_tr = stack_views(train_cache, train_accs, views, esmc_embs=esmc)
    x_te = stack_views(test_cache, test_accs, views, esmc_embs=esmc)
    y_tr = load_labels(train_cache, train_accs, task)
    y_te = load_labels(test_cache, test_accs, task)

    probe = train_probe(
        x_tr, y_tr,
        out_dim=task_out_dim(task, y_tr),
        task=task, kind=kind, device=device, epochs=epochs,
    )
    scores = predict(probe, x_te, device, task)
    auc, per = compute_auc(y_te.numpy(), scores, task)
    return auc, per, scores


def _ec_score_array(cache: Dict[str, dict], accs: List[str]) -> np.ndarray:
    return np.array([ec_label(cache, a) for a in accs])


# ---------------------------------------------------------------------------
# Mode: variants
# ---------------------------------------------------------------------------


def cmd_variants(args: argparse.Namespace) -> int:
    train_cache = _load_split("reference", required=True)
    test_cache = _load_split(args.eval_on, required=True)
    esmc = _load_esmc(needed=True)

    train_accs = list(train_cache.keys())
    test_accs = list(test_cache.keys())
    if esmc is not None:
        # The ESM C variant requires every accession to be in the index.
        miss_tr = [a for a in train_accs if a not in esmc]
        miss_te = [a for a in test_accs if a not in esmc]
        esmc_ok = not (miss_tr or miss_te)
    else:
        esmc_ok = False
    print(f"train: {len(train_accs)} reference  test: {len(test_accs)} {args.eval_on}  "
          f"esmc_baseline={'on' if esmc_ok else 'off'}")

    variants = iter_variants(include_esmc=esmc_ok)
    tasks = _resolve_tasks(args.tasks)
    kinds = parse_csv(args.probe, ["linear", "mlp"])

    out: List[dict] = []
    t0 = time.time()
    for vname, views in variants:
        if ESMC_VIEW in views and not esmc_ok:
            continue
        for kind in kinds:
            for task in tasks:
                auc, per, _ = _train_eval(
                    train_cache, train_accs, test_cache, test_accs,
                    views=views, task=task, kind=kind,
                    device=args.device, epochs=args.epochs, esmc=esmc,
                )
                out.append({
                    "variant": vname, "kind": kind, "task": task,
                    "auc_macro": auc, "auc_per_class": per,
                    "n_train": len(train_accs), "n_test": len(test_accs),
                })
                print(f"  {vname:20s} {kind:6s} {task:14s}  macro-AUC={auc:.3f}")
    print(f"\nTotal: {time.time() - t0:.1f}s")

    out_path = Path(args.out) if args.out else PROBE_RESULTS_DIR / "variants.json"
    out_path.write_text(json.dumps({
        "config": {"eval_on": args.eval_on, "epochs": args.epochs, "tasks": tasks, "kinds": kinds},
        "results": out,
    }, indent=2))
    print(f"Wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Mode: cv5fold
# ---------------------------------------------------------------------------


def cmd_cv5fold(args: argparse.Namespace) -> int:
    ref = _load_split("reference", required=True)
    bench = _load_split("benchmark", required=True)
    cache = {**ref, **bench}
    accs = list(ref.keys()) + list(bench.keys())
    y = _ec_score_array(cache, accs)
    print(f"Combined labeled pool: {len(accs)}  EC dist: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    views = parse_views(args.views)
    kinds = parse_csv(args.probes, ["linear", "mlp"])

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(skf.split(np.zeros(len(accs)), y))

    cv_results: Dict[str, dict] = {}
    for kind in kinds:
        fold_macros: List[float] = []
        fold_per_class = np.full((args.n_splits, N_CLASSES), np.nan)
        for fold, (tr, te) in enumerate(splits):
            tr_accs = [accs[i] for i in tr]
            te_accs = [accs[i] for i in te]
            macro, per, _ = _train_eval(
                cache, tr_accs, cache, te_accs,
                views=views, task="ec_l1", kind=kind,
                device=args.device, epochs=args.epochs,
            )
            fold_macros.append(macro)
            for c in range(N_CLASSES):
                fold_per_class[fold, c] = per[c] if not np.isnan(per[c]) else np.nan
            print(f"  [{kind}] fold {fold+1}: macro-AUC = {macro:.3f}")
        fm = np.array(fold_macros)
        print(f"  [{kind}] macro-AUC: {fm.mean():.3f} ± {fm.std():.3f}  "
              f"range [{fm.min():.3f}, {fm.max():.3f}]")
        cv_results[kind] = {
            "views": views,
            "macro_mean": float(fm.mean()), "macro_std": float(fm.std()),
            "macro_min": float(fm.min()), "macro_max": float(fm.max()),
            "per_class_mean": [
                float(np.nanmean(fold_per_class[:, c])) if np.any(~np.isnan(fold_per_class[:, c])) else None
                for c in range(N_CLASSES)
            ],
            "per_class_std": [
                float(np.nanstd(fold_per_class[:, c])) if np.any(~np.isnan(fold_per_class[:, c])) else None
                for c in range(N_CLASSES)
            ],
            "fold_macros": [float(x) for x in fm],
        }

    out_path = Path(args.out) if args.out else PROBE_RESULTS_DIR / "cv5fold.json"
    out_path.write_text(json.dumps({
        "config": {"views": views, "kinds": kinds, "n_splits": args.n_splits, "seed": args.seed,
                   "epochs": args.epochs, "n_pool": len(accs)},
        "by_kind": cv_results,
    }, indent=2))
    print(f"Wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Mode: dark
# ---------------------------------------------------------------------------


def _confidence(scores: np.ndarray, task: str) -> np.ndarray:
    if task == "is_enzyme":
        return scores
    return scores.max(axis=1)


def _conformal_set_sizes(calib_probs: np.ndarray, target_probs: np.ndarray,
                         alpha: float) -> Tuple[float, np.ndarray]:
    """APS-style: nonconformity = 1 - P(y|x); threshold τ at the (1-α) quantile of
    calibration nonconformities. Returns (tau, set_sizes_for_target)."""
    nc_calib = 1.0 - calib_probs.max(axis=1)
    n = len(nc_calib)
    q_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
    q_idx = max(0, min(n - 1, q_idx))
    tau = float(np.sort(nc_calib)[q_idx])
    nc_target = 1.0 - target_probs
    sizes = (nc_target <= tau).sum(axis=1)
    return tau, sizes


def cmd_dark(args: argparse.Namespace) -> int:
    ref = _load_split("reference", required=True)
    bench = _load_split("benchmark", required=True)
    dark = _load_split("dark", required=True)
    esmc = _load_esmc(needed=False)

    ref_accs = list(ref.keys())
    bench_accs = list(bench.keys())
    dark_accs = list(dark.keys())
    print(f"N: ref={len(ref_accs)} bench={len(bench_accs)} dark={len(dark_accs)}")

    views = parse_views(args.views)
    kinds = parse_csv(args.probes, ["linear", "mlp"])
    tasks = _resolve_tasks(args.tasks)
    alphas = [float(x) for x in args.conformal_alphas.split(",") if x.strip()]

    out: Dict[str, Dict] = {}
    for kind in kinds:
        for task in tasks:
            x_ref = stack_views(ref, ref_accs, views, esmc_embs=esmc)
            x_b = stack_views(bench, bench_accs, views, esmc_embs=esmc)
            x_d = stack_views(dark, dark_accs, views, esmc_embs=esmc)
            y_ref = load_labels(ref, ref_accs, task)
            y_b = load_labels(bench, bench_accs, task)
            y_d = load_labels(dark, dark_accs, task)

            probe = train_probe(
                x_ref, y_ref, out_dim=task_out_dim(task, y_ref),
                task=task, kind=kind, device=args.device, epochs=args.epochs,
            )
            p_b = predict(probe, x_b, args.device, task)
            p_d = predict(probe, x_d, args.device, task)

            b_auc, b_per = compute_auc(y_b.numpy(), p_b, task)
            d_auc, d_per = compute_auc(y_d.numpy(), p_d, task)
            b_conf = _confidence(p_b, task)
            d_conf = _confidence(p_d, task)
            print(f"  {kind:6s} {task:14s}  bench-AUC={b_auc:.3f}  "
                  f"dark-AUC(weak)={d_auc:.3f}  "
                  f"conf bench={b_conf.mean():.3f}  conf dark={d_conf.mean():.3f}")

            entry = {
                "bench_auc": b_auc, "bench_per_class": b_per,
                "dark_auc_weak": d_auc, "dark_per_class_weak": d_per,
                "bench_conf_mean": float(b_conf.mean()),
                "bench_conf_p25": float(np.percentile(b_conf, 25)),
                "bench_conf_median": float(np.median(b_conf)),
                "dark_conf_mean": float(d_conf.mean()),
                "dark_conf_p25": float(np.percentile(d_conf, 25)),
                "dark_conf_median": float(np.median(d_conf)),
            }

            if task == "ec_l1":
                # Conformal sets calibrated on benchmark (real GT), applied to dark.
                conformal: Dict[str, dict] = {}
                argmax_dist = Counter(p_d.argmax(axis=1).tolist())
                entry["dark_argmax_distribution"] = {
                    int(c): int(argmax_dist.get(c, 0)) for c in range(N_CLASSES)
                }
                for alpha in alphas:
                    tau, sizes_d = _conformal_set_sizes(p_b, p_d, alpha)
                    _, sizes_b = _conformal_set_sizes(p_b, p_b, alpha)
                    conformal[str(alpha)] = {
                        "tau": tau,
                        "dark_mean_set_size": float(sizes_d.mean()),
                        "dark_empty_set_frac": float((sizes_d == 0).mean()),
                        "bench_mean_set_size": float(sizes_b.mean()),
                        "bench_empty_set_frac": float((sizes_b == 0).mean()),
                    }
                entry["conformal"] = conformal

            out[f"{kind}__{task}"] = entry

    out_path = Path(args.out) if args.out else PROBE_RESULTS_DIR / "dark.json"
    out_path.write_text(json.dumps({
        "config": {"views": views, "kinds": kinds, "tasks": tasks,
                   "alphas": alphas, "epochs": args.epochs},
        "results": out,
    }, indent=2))
    print(f"Wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Mode: zeroshot637 (apples-to-apples vs InterLabelGO+)
# ---------------------------------------------------------------------------


VALIDATION_BAR = 0.85


def cmd_zeroshot637(args: argparse.Namespace) -> int:
    ref = _load_split("reference", required=True)
    bench = _load_split("benchmark", required=True)
    ref_accs = list(ref.keys())
    bench_accs = list(bench.keys())

    views = parse_views(args.views)
    kind = args.probe
    print(f"train=ref({len(ref_accs)})  test=bench({len(bench_accs)})  views={views}  probe={kind}")

    macro, per, scores = _train_eval(
        ref, ref_accs, bench, bench_accs,
        views=views, task="ec_l1", kind=kind,
        device=args.device, epochs=args.epochs,
    )
    bar_count = sum(1 for v in per if not np.isnan(v) and v >= VALIDATION_BAR)
    print(f"  IDPro zero-shot macro-AUC = {macro:.4f}  classes ≥ {VALIDATION_BAR}: {bar_count}/{N_CLASSES}")
    for c, name in CLASS_NAMES.items():
        v = per[c]
        flag = " *" if not np.isnan(v) and v >= VALIDATION_BAR else ""
        print(f"    class {c} ({name:<14s}): {v:.3f}{flag}")

    # Optional baseline overlay (InterLabelGO+ / etc.) via strict-keyword scoring.
    overlays: Dict[str, dict] = {}
    if args.overlay_baselines:
        names = parse_csv(args.overlay_baselines,
                          ["interlabelgo", "p2t", "rag_transfer", "bioreason"])
        files = {
            "interlabelgo": "interlabelgo_benchmark_predictions.json",
            "p2t":          "p2t_benchmark_predictions.json",
            "rag_transfer": "rag_transfer_benchmark_predictions.json",
            "bioreason":    "bioreason_benchmark_predictions.json",
        }
        y_test = _ec_score_array(bench, bench_accs)
        for nm in names:
            p = BASELINE_PREDS_DIR / files[nm]
            if not p.exists():
                print(f"  [overlay] missing {p} — skip {nm}")
                continue
            preds = json.loads(p.read_text())
            mat = np.zeros((len(bench_accs), N_CLASSES))
            covered = 0
            for i, a in enumerate(bench_accs):
                if a in preds:
                    covered += 1
                    mat[i] = strict_class_scores(preds[a])
            ov_per = per_class_auc(y_test, mat, n_classes=N_CLASSES)
            ov_macro = macro_from_per_class(ov_per)
            ov_bar = sum(1 for v in ov_per if not np.isnan(v) and v >= VALIDATION_BAR)
            overlays[nm] = {
                "macro_auc": ov_macro,
                "per_class": [None if np.isnan(v) else v for v in ov_per],
                "coverage": covered,
                "n_classes_above_bar": ov_bar,
            }
            print(f"  [overlay] {nm:14s} macro={ov_macro:.4f} cov={covered}/{len(bench_accs)} ≥{VALIDATION_BAR}: {ov_bar}")

    out_path = Path(args.out) if args.out else PROBE_RESULTS_DIR / "zeroshot637.json"
    out_path.write_text(json.dumps({
        "config": {"views": views, "probe": kind, "epochs": args.epochs,
                   "n_train": len(ref_accs), "n_test": len(bench_accs),
                   "validation_bar": VALIDATION_BAR},
        "idpro": {
            "macro_auc": macro,
            "per_class": [None if np.isnan(v) else v for v in per],
            "n_classes_above_bar": bar_count,
        },
        "overlay_baselines": overlays,
    }, indent=2))
    print(f"Wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Mode: fair (apples-to-apples on baseline-coverage subsets)
# ---------------------------------------------------------------------------


def cmd_fair(args: argparse.Namespace) -> int:
    ref = _load_split("reference", required=True)
    bench = _load_split("benchmark", required=True)
    cache = {**ref, **bench}
    all_accs = list(ref.keys()) + list(bench.keys())
    print(f"Combined labeled pool: {len(all_accs)}")

    views = parse_views(args.views)
    kind = args.probe

    baseline_files = {
        "p2t":          "p2t_benchmark_predictions.json",
        "rag_transfer": "rag_transfer_benchmark_predictions.json",
        "bioreason":    "bioreason_benchmark_predictions.json",
        "deepfri":      "deepfri_benchmark_predictions.json",
        "mmseqs":       "mmseqs_benchmark_predictions.json",
        "deepgometa":   "deepgometa_benchmark_predictions.json",
    }
    requested = parse_csv(args.baselines, list(baseline_files))
    method_preds: Dict[str, Dict[str, str]] = {}
    for name in requested:
        p = BASELINE_PREDS_DIR / baseline_files[name]
        if not p.exists():
            raise SystemExit(f"Missing baseline predictions {p}; run run_baselines.py first.")
        method_preds[name] = json.loads(p.read_text())

    bench_keys = set(bench.keys())
    # Build subsets driven by which baselines were requested.
    def intersect(*names: str) -> List[str]:
        sets = [set(method_preds[n]) for n in names if n in method_preds]
        if not sets:
            return []
        return sorted(set.intersection(*sets) & bench_keys)

    subsets: Dict[str, List[str]] = {}
    broad = [b for b in ("p2t", "rag_transfer", "bioreason", "deepgometa") if b in method_preds]
    if broad:
        subsets["broad"] = intersect(*broad)
    if "deepfri" in method_preds:
        subsets["with_deepfri"] = intersect(*broad, "deepfri")
    if "mmseqs" in method_preds:
        narrow = list(broad)
        if "deepfri" in method_preds:
            narrow.append("deepfri")
        narrow.append("mmseqs")
        subsets["all_methods"] = intersect(*narrow)
    for k, v in subsets.items():
        print(f"  subset {k}: n={len(v)}")

    fair_results: Dict[str, dict] = {}
    for sname, test_accs in subsets.items():
        if not test_accs:
            continue
        print(f"\n=== subset {sname}  n={len(test_accs)} ===")
        y_test = _ec_score_array(cache, test_accs)
        print(f"  EC dist: {dict(sorted(Counter(y_test.tolist()).items()))}")

        train_accs = [a for a in all_accs if a not in set(test_accs)]
        macro, idpro_per, _ = _train_eval(
            cache, train_accs, cache, test_accs,
            views=views, task="ec_l1", kind=kind,
            device=args.device, epochs=args.epochs,
        )
        print(f"  IDPro probe (train_n={len(train_accs)}): macro-AUC = {macro:.3f}")

        per_method: Dict[str, dict] = {
            "IDPro_probe": {
                "macro_auc": macro,
                "per_class": {int(c): (None if np.isnan(v) else v) for c, v in enumerate(idpro_per)},
            }
        }
        for name in requested:
            preds = method_preds[name]
            mat = np.zeros((len(test_accs), N_CLASSES))
            for i, a in enumerate(test_accs):
                txt = preds.get(a, "")
                mat[i] = (deepgometa_strict_scores(txt) if name == "deepgometa"
                          else strict_class_scores(txt))
            mper = per_class_auc(y_test, mat, n_classes=N_CLASSES)
            mmacro = macro_from_per_class(mper)
            per_method[name] = {
                "macro_auc": mmacro,
                "per_class": {int(c): (None if np.isnan(v) else v) for c, v in enumerate(mper)},
            }
            print(f"  {name:14s}  macro-AUC = {mmacro:.3f}")

        fair_results[sname] = {
            "n": len(test_accs),
            "ec_distribution": {int(k): int(v) for k, v in Counter(y_test.tolist()).items()},
            "per_method": per_method,
        }

    out_path = Path(args.out) if args.out else PROBE_RESULTS_DIR / "fair.json"
    out_path.write_text(json.dumps({
        "config": {"views": views, "probe": kind, "epochs": args.epochs,
                   "baselines": requested},
        "subsets": fair_results,
    }, indent=2))
    print(f"Wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device", default="cuda")
        p.add_argument("--epochs", type=int, default=80)
        p.add_argument("--out", type=str, default="")

    p_v = sub.add_parser("variants", help="View-combo × probe-kind × task sweep")
    p_v.add_argument("--probe", default="mlp",
                     help="Comma-separated: linear,mlp (default mlp)")
    p_v.add_argument("--tasks", default="all",
                     help=f"'all' or comma-separated subset of {ALL_TASKS}")
    p_v.add_argument("--eval-on", default="benchmark",
                     choices=["benchmark", "dark"], help="test split")
    _common(p_v)
    p_v.set_defaults(func=cmd_variants)

    p_cv = sub.add_parser("cv5fold", help="5-fold stratified CV on EC L1")
    p_cv.add_argument("--views", default="A+B+C")
    p_cv.add_argument("--probes", default="linear,mlp")
    p_cv.add_argument("--n-splits", type=int, default=5)
    p_cv.add_argument("--seed", type=int, default=0)
    _common(p_cv)
    p_cv.set_defaults(func=cmd_cv5fold)

    p_d = sub.add_parser("dark", help="Train on reference; score bench + dark + conformal")
    p_d.add_argument("--views", default="A+B+C")
    p_d.add_argument("--probes", default="mlp,linear")
    p_d.add_argument("--tasks", default="all")
    p_d.add_argument("--conformal-alphas", default="0.05,0.1,0.2")
    _common(p_d)
    p_d.set_defaults(func=cmd_dark)

    p_z = sub.add_parser("zeroshot637", help="Train on reference, eval on benchmark (vs InterLabelGO+)")
    p_z.add_argument("--views", default="A+B+C")
    p_z.add_argument("--probe", default="linear", choices=["linear", "mlp"])
    p_z.add_argument("--overlay-baselines", default="",
                     help="Comma-separated: interlabelgo,p2t,rag_transfer,bioreason")
    _common(p_z)
    p_z.set_defaults(func=cmd_zeroshot637)

    p_f = sub.add_parser("fair", help="Same-subset comparison vs strict-keyword baselines")
    p_f.add_argument("--views", default="A+B+C")
    p_f.add_argument("--probe", default="linear", choices=["linear", "mlp"])
    p_f.add_argument("--baselines", default="p2t,rag_transfer,bioreason,deepfri,mmseqs,deepgometa")
    _common(p_f)
    p_f.set_defaults(func=cmd_fair)

    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
