"""
Build v2 spider plot comparing IDPro classifier probe to P2T baseline on the
415-protein dark-genome evaluation set, under a strict-keyword-match rule.

Strict keyword rule (user-specified):
  For each prediction text and each EC class C:
    - scan the text for any C-specific keyword  → C hit count
    - scan the text for keywords of OTHER classes → competing hit count
    - STRICT: if ANY competing (wrong-class) keyword appears, deem the
      prediction wrong for class C (force score_C = 0).
    - otherwise score_C = hit_count_C.

AUC: compute one-vs-rest ROC AUC per EC class using these strict scores
against the weak GO-derived EC labels also used by `run_probe.py cv5fold`.

Inputs (regenerable):
  EXTRACTED_EMBEDDINGS_DIR/{reference,benchmark,dark}_embeddings.pt
  PROBE_RESULTS_DIR/cv5fold.json   (5-fold CV per-class AUCs)
  BASELINE_PREDS_DIR/{method}_{split}_predictions.json (or .csv for DeepFRI)

Outputs:
  FIGURES_DIR/spider_ec_v2_twopanel.{png,pdf}
  FIGURES_DIR/spider_ec_v2.{png,pdf}
  FIGURES_DIR/spider_ec_v2.json
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.paths import (  # noqa: E402
    BASELINE_PREDS_DIR,
    DARK_GENOME_META,
    EXTRACTED_EMBEDDINGS_DIR,
    FIGURES_DIR,
    PROBE_RESULTS_DIR,
)

EMB_DIR = EXTRACTED_EMBEDDINGS_DIR
FIG_DIR = FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

DARK_META = DARK_GENOME_META
BASELINE_OUT = BASELINE_PREDS_DIR / "p2t_dark_predictions.jsonl"

# Other baselines (benchmark-only, no dark predictions)
BENCH_METHODS_DIR = BASELINE_PREDS_DIR
BENCH_METHOD_FILES = {
    "MMseqs2": "mmseqs_benchmark_predictions.json",
    "DeepFRI": "deepfri_benchmark_predictions.json",
    "DeepGOMeta": "deepgometa_benchmark_predictions.json",
    "BioReason-Pro": "bioreason_benchmark_predictions.json",
    "P2T (RAG transfer)": "rag_transfer_benchmark_predictions.json",
    # InterLabelGO+ is the CAFA5 winner (Liu et al. 2024). Its training set
    # covers most of UniProt+GOA so it likely has TRAINING-OVERLAP with our
    # 637-benchmark — i.e., its AUC is an UPPER BOUND on its true generalization.
    # IDPro is evaluated under 5-fold CV with no train/test overlap, so the
    # IDPro lead is conservative; the homology-controlled gap is likely larger.
    "InterLabelGO+": "interlabelgo_benchmark_predictions.json",
}

# Dark proteome predictions per method (where available).
DARK_METHOD_FILES: dict = {
    "InterLabelGO+": "interlabelgo_dark_predictions.json",
}

# Canonical enzyme-activity GO IDs → EC class (used for DeepGOMeta which
# outputs raw GO IDs rather than natural-language descriptions).
GO_TO_EC = {
    "GO:0016491": 1,  # oxidoreductase activity
    "GO:0016614": 1,  # oxidoreductase activity, acting on CH-OH
    "GO:0016740": 2,  # transferase activity
    "GO:0016301": 2,  # kinase activity
    "GO:0016772": 2,  # transferase activity, transferring phosphorus
    "GO:0016787": 3,  # hydrolase activity
    "GO:0008233": 3,  # peptidase activity
    "GO:0016788": 3,  # hydrolase activity, on ester bonds
    "GO:0016798": 3,  # hydrolase activity, on O-glycosyl bonds
    "GO:0016810": 3,  # hydrolase activity, acting on carbon-nitrogen
    "GO:0004518": 3,  # nuclease activity
    "GO:0016829": 4,  # lyase activity
    "GO:0016830": 4,  # carbon-carbon lyase activity
    "GO:0016831": 4,  # carboxy-lyase activity
    "GO:0016835": 4,  # carbon-oxygen lyase activity
    "GO:0016853": 5,  # isomerase activity
    "GO:0016854": 5,  # racemase / epimerase activity
    "GO:0016855": 5,  # racemase / epimerase acting on aa
    "GO:0016860": 5,  # intramolecular oxidoreductase activity
    "GO:0016874": 6,  # ligase activity
    "GO:0016875": 6,  # ligase activity, forming C-O bonds
    "GO:0022857": 7,  # transmembrane transporter activity
    "GO:0022804": 7,  # active transmembrane transporter activity
    "GO:0015075": 7,  # ion transmembrane transporter activity
    "GO:0005215": 7,  # transporter activity
    "GO:0008324": 7,  # cation transmembrane transporter activity
}


# ---------------------------------------------------------------------------
# EC L1 keyword dictionaries
# ---------------------------------------------------------------------------

# For LABELS (weak GO-derived), same dict as evaluate_ec_classifier.py:
LABEL_KEYWORDS = {
    1: ["oxidoreductase activity", "dehydrogenase", "reductase", "oxidase", "oxygenase", "peroxidase"],
    2: ["transferase activity", "kinase activity", "methyltransferase", "acyltransferase", "glycosyltransferase"],
    3: ["hydrolase activity", "peptidase", "protease", "nuclease", "phosphatase", "esterase", "lipase", "glycosidase"],
    4: ["lyase activity", "decarboxylase", "aldolase", "dehydratase", "synthase"],
    5: ["isomerase activity", "racemase", "epimerase", "mutase"],
    6: ["ligase activity", "synthetase"],
    7: ["transporter activity", "transmembrane transport", "channel activity", "permease"],
}

# For PREDICTIONS (baseline text output) — use a broader, natural-language
# keyword dictionary. A hit on any keyword signals the class claim.
# Keep per-class lists mutually exclusive where possible (to enforce strict rule).
PRED_KEYWORDS = {
    0: ["non-enzyme", "non enzyme", "structural", "regulatory", "binding protein",
        "transcription factor", "chaperone"],
    1: ["oxidoreductase", "dehydrogenase", "reductase", "oxidase", "oxygenase",
        "peroxidase", "redox"],
    2: ["transferase", "kinase", "phosphorylates", "methyltransferase",
        "acyltransferase", "glycosyltransferase", "phosphotransferase"],
    3: ["hydrolase", "hydrolyzes", "hydrolysis", "peptidase", "protease",
        "nuclease", "phosphatase", "esterase", "lipase", "glycosidase",
        "cellulase", "xylanase"],
    4: ["lyase", "decarboxylase", "aldolase", "dehydratase",
        "eliminates", "cleaves without hydrolysis"],
    5: ["isomerase", "racemase", "epimerase", "mutase", "isomerization"],
    6: ["ligase", "synthetase", "ligates", "forming a bond"],
    7: ["translocase", "transporter", "transmembrane transport", "channel",
        "permease", "pump", "import", "export", "abc transporter"],
}

CLASS_NAMES = {
    0: "Non-enzyme", 1: "Oxidoreductase", 2: "Transferase", 3: "Hydrolase",
    4: "Lyase", 5: "Isomerase", 6: "Ligase", 7: "Translocase",
}


def weak_ec_l1_from_go(go_terms: str):
    """Return EC L1 (0..7) or None. Same rule as evaluate_ec_classifier.py."""
    if not go_terms:
        return None
    lo = go_terms.lower()
    matched = set()
    for ec, kws in LABEL_KEYWORDS.items():
        if any(kw in lo for kw in kws):
            matched.add(ec)
    if not matched:
        return 0
    if len(matched) == 1:
        return next(iter(matched))
    return None


def build_weak_dark_labels() -> Dict[str, int]:
    labels = {}
    with DARK_META.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            lab = weak_ec_l1_from_go(r.get("go_terms") or "")
            if lab is not None:
                labels[r["accession"]] = lab
    return labels


# ---------------------------------------------------------------------------
# Strict per-class keyword score on a prediction string
# ---------------------------------------------------------------------------


def _count_hits(text: str, keywords: List[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def strict_class_scores(text: str) -> np.ndarray:
    """
    Returns an 8-d score vector where score[c] = hits(c-keywords) if
    NO other-class keyword appears in the text; else score[c] = 0.

    This is the user-specified strict rule: if the prediction mentions
    any OTHER EC class, it doesn't count for class c — no matter what
    else it says.
    """
    text_lower = (text or "").lower()
    hits = np.array([_count_hits(text_lower, PRED_KEYWORDS[c]) for c in range(8)], dtype=float)
    strict = np.zeros(8, dtype=float)
    # For class c, check if hits for any other class are > 0.
    # If so, strict[c] = 0; else strict[c] = hits[c].
    for c in range(8):
        other_max = float(np.delete(hits, c).max()) if hits.size > 1 else 0.0
        if other_max > 0:
            strict[c] = 0.0
        else:
            strict[c] = hits[c]
    return strict


# ---------------------------------------------------------------------------
# Evaluate baseline P2T text on dark proteins
# ---------------------------------------------------------------------------


def load_baseline_preds() -> Dict[str, str]:
    preds = {}
    with BASELINE_OUT.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            acc = r.get("long_format_id", "").split("_")[0]
            preds[acc] = r.get("Predicted") or ""
    return preds


def load_benchmark_baseline_preds() -> Dict[str, str]:
    """Benchmark set (UniProt-labeled)."""
    preds = {}
    p = PREL / "benchmark" / "results" / "benchmark_results.jsonl"
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            acc = r.get("long_format_id", "").split("_")[0]
            preds[acc] = r.get("Predicted") or ""
    return preds


def baseline_strict_matrix(accs: List[str], preds: Dict[str, str]) -> np.ndarray:
    """Return (N, 8) strict-keyword scores for each accession × EC class."""
    out = np.zeros((len(accs), 8), dtype=float)
    for i, a in enumerate(accs):
        out[i] = strict_class_scores(preds.get(a, ""))
    return out


def deepgometa_strict_scores(text: str) -> np.ndarray:
    """
    DeepGOMeta outputs GO IDs (e.g. 'GO:0016787 GO:0022857 ...').
    Map any enzyme-activity GO IDs in the prediction to EC class.
    Apply the same STRICT rule: if multiple EC classes are mentioned,
    score=0 for every class.
    """
    hits = np.zeros(8, dtype=float)
    for go_id in re.findall(r"GO:\d{7}", text or ""):
        ec = GO_TO_EC.get(go_id)
        if ec is not None:
            hits[ec] += 1
    # If proteins with no enzyme-activity GO IDs at all, that's the
    # non-enzyme hypothesis (class 0).
    if hits[1:].sum() == 0 and text:
        hits[0] = 1
    strict = np.zeros(8, dtype=float)
    for c in range(8):
        other_max = float(np.delete(hits, c).max()) if hits.size > 1 else 0.0
        strict[c] = 0.0 if other_max > 0 else hits[c]
    return strict


def generic_baseline_strict_matrix(
    accs: List[str],
    preds: Dict[str, str],
    mode: str = "text",
) -> np.ndarray:
    """mode=\"text\" uses natural-language keyword dict; mode=\"go_ids\" uses GO-ID map."""
    out = np.zeros((len(accs), 8), dtype=float)
    for i, a in enumerate(accs):
        if mode == "go_ids":
            out[i] = deepgometa_strict_scores(preds.get(a, ""))
        else:
            out[i] = strict_class_scores(preds.get(a, ""))
    return out


def load_benchmark_method_preds(filename: str) -> Dict[str, str]:
    p = BENCH_METHODS_DIR / filename
    d = json.loads(p.read_text())
    # values are strings directly (our inspection confirmed schema)
    return {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in d.items()}


def _deepfri_csv_to_text(csv_path: Path) -> Dict[str, str]:
    """Read DeepFRI per-protein predictions CSV → {protein_id: concat of top-10 GO/EC term names}."""
    import csv as _csv
    per_prot = {}
    with csv_path.open() as f:
        # Skip the leading banner '### Predictions made by DeepFRI.'
        first = f.readline()
        if first.startswith("###"):
            header = f.readline()
        else:
            header = first
        rdr = _csv.reader(f)
        for row in rdr:
            if len(row) < 4:
                continue
            prot, go_or_ec, score, name = row[0], row[1], row[2], row[3]
            try:
                s = float(score)
            except ValueError:
                continue
            per_prot.setdefault(prot, []).append((s, go_or_ec, name))
    out = {}
    for prot, items in per_prot.items():
        items.sort(reverse=True)
        top = items[:10]
        out[prot] = " | ".join(f"{g} {n}" for (s, g, n) in top)
    return out


# ---------------------------------------------------------------------------
# Get IDPro classifier scores from the already-trained probe
# ---------------------------------------------------------------------------


def idpro_probe_scores_on_dark(accs: List[str]) -> np.ndarray:
    """
    Retrain A-only linear probe on combined ref+bench labeled pool, apply to
    dark. Same as (B) path in evaluate_ec_classifier.py but returns scores
    aligned to our accs order.
    """
    import torch.nn as nn

    ref_cache = torch.load(EMB_DIR / "reference_embeddings.pt", map_location="cpu", weights_only=False)
    bench_cache = torch.load(EMB_DIR / "benchmark_embeddings.pt", map_location="cpu", weights_only=False)
    dark_cache = torch.load(EMB_DIR / "dark_embeddings.pt", map_location="cpu", weights_only=False)

    def ec_label(c, a):
        v = c[a]["labels"]["ec_l1"]
        return 0 if v is None else int(v)

    ref_accs = list(ref_cache.keys())
    bench_accs = list(bench_cache.keys())
    all_cache = {**ref_cache, **bench_cache}
    all_accs = ref_accs + bench_accs
    all_labels = np.array(
        [ec_label(ref_cache, a) for a in ref_accs] +
        [ec_label(bench_cache, a) for a in bench_accs]
    )

    VIEWS = ["view_a_prompteol_l48", "view_b_question_mean_l48", "view_c_eos_l64"]
    x_train = torch.cat(
        [torch.stack([all_cache[a][v].float() for a in all_accs]) for v in VIEWS],
        dim=-1,
    )
    y_train = torch.tensor(all_labels, dtype=torch.long)
    x_dark = torch.cat(
        [torch.stack([dark_cache[a][v].float() for a in accs]) for v in VIEWS],
        dim=-1,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe = nn.Linear(x_train.shape[1], 8).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    x_train = x_train.to(device); y_train = y_train.to(device)
    bs = 64
    for _ in range(100):
        perm = torch.randperm(x_train.shape[0], device=device)
        for s in range(0, x_train.shape[0], bs):
            idx = perm[s:s+bs]
            logits = probe(x_train[idx])
            loss = loss_fn(logits, y_train[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        scores = torch.softmax(probe(x_dark.to(device)), dim=-1).cpu().numpy()
    return scores


# ---------------------------------------------------------------------------
# AUC per class
# ---------------------------------------------------------------------------


def per_class_auc(y_true: np.ndarray, scores: np.ndarray, n_classes: int = 8):
    per = []
    for c in range(n_classes):
        yt = (y_true == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            per.append(np.nan)
            continue
        per.append(float(roc_auc_score(yt, scores[:, c])))
    return per


# ---------------------------------------------------------------------------
# Spider plot
# ---------------------------------------------------------------------------


COLORS = {
    "IDPro classifier probe": "#1f77b4",    # blue — our method
    "P2T baseline": "#d62728",              # red — closest text-only comparator
    "P2T (RAG transfer)": "#ff7f0e",        # orange
    "BioReason-Pro": "#9467bd",             # purple
    "DeepFRI": "#8c564b",                   # brown
    "MMseqs2": "#e377c2",                   # pink
    "DeepGOMeta": "#17becf",                # teal
    "CLEAN": "#2ca02c",                     # green — SOTA EC predictor (Yu et al. 2023)
    "InterLabelGO+": "#bcbd22",             # olive — CAFA5 winner (Liu et al. 2024)
}
METHOD_LINEWIDTH = {
    "IDPro classifier probe": 2.5,
}
METHOD_ZORDER = {
    "IDPro classifier probe": 10,
    "P2T baseline": 5,
}


def _draw_single_spider(ax, aucs_by_method, axis_labels, panel_title, panel_note):
    n_axes = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(axis_labels, fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=7)

    # Chance (AUC=0.5) reference
    chance = [0.5] * (n_axes + 1)
    ax.plot(angles_closed, chance, color="gray", lw=0.8, ls="--", alpha=0.6)

    for method, aucs in aucs_by_method.items():
        vals = [a if a is not None and not np.isnan(a) else 0.5 for a in aucs]
        vals_closed = vals + vals[:1]
        color = COLORS.get(method, None)
        lw = METHOD_LINEWIDTH.get(method, 1.6)
        zo = METHOD_ZORDER.get(method, 3)
        ax.plot(angles_closed, vals_closed, "-o", label=method,
                color=color, lw=lw, markersize=5 if lw > 2 else 3.5, zorder=zo)
        # Only fill the hero method to avoid visual clutter with many lines
        if method == "IDPro classifier probe":
            ax.fill(angles_closed, vals_closed, alpha=0.12, color=color, zorder=zo - 1)

    ax.set_title(panel_title, fontsize=12, pad=20, fontweight="bold")
    if panel_note:
        ax.text(0.5, -0.22, panel_note, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.5, color="#333",
                linespacing=1.3)


def _add_overall(aucs_by_method):
    """Append overall (macro) AUC — mean of non-NaN per-class values — as
    the 9th data point for every method. Returns new dict of 9-length lists."""
    out = {}
    for m, vals in aucs_by_method.items():
        clean = [v for v in vals if v is not None and not np.isnan(v)]
        macro = float(np.mean(clean)) if clean else float("nan")
        out[m] = list(vals) + [macro]
    return out


def spider_single_panel(
    axis_labels,
    aucs_by_method,
    out_path_stem,
):
    """Compact single-panel spider, title-less, n-free. Used for proposal panels."""
    n_axes = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    # Compact figure — legend to the right of the spider plot.
    fig, ax = plt.subplots(figsize=(5.2, 3.6), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(axis_labels, fontsize=9)
    ax.tick_params(axis="x", pad=0)  # pull axis labels closer to plot
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.5, 0.7, 0.9])
    ax.set_yticklabels(["0.5", "0.7", "0.9"], fontsize=7)

    # Chance reference
    chance = [0.5] * (n_axes + 1)
    ax.plot(angles_closed, chance, color="gray", lw=0.8, ls="--", alpha=0.6)

    for method, aucs in aucs_by_method.items():
        vals = [a if a is not None and not np.isnan(a) else 0.5 for a in aucs]
        vals_closed = vals + vals[:1]
        color = COLORS.get(method, None)
        lw = METHOD_LINEWIDTH.get(method, 1.6)
        zo = METHOD_ZORDER.get(method, 3)
        ax.plot(angles_closed, vals_closed, "-o", label=method,
                color=color, lw=lw, markersize=4 if lw > 2 else 3.0, zorder=zo)
        if method == "IDPro classifier probe":
            ax.fill(angles_closed, vals_closed, alpha=0.12, color=color, zorder=zo - 1)

    # Compact vertical legend on the right
    ax.legend(loc="center left", bbox_to_anchor=(1.18, 0.5),
              fontsize=8, frameon=False,
              handletextpad=0.35, labelspacing=0.35,
              handlelength=1.4, borderaxespad=0.0)

    # Tight layout — no title, minimal padding
    plt.subplots_adjust(left=0.0, right=0.72, top=1.0, bottom=0.0)

    for ext in ("png", "pdf"):
        p = out_path_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Wrote {p}")
    plt.close(fig)


def spider_two_panel(
    axis_labels,
    left_aucs, left_title, left_note,
    right_aucs, right_title, right_note,
    out_path_stem,
    suptitle,
):
    fig = plt.figure(figsize=(16, 8.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    _draw_single_spider(ax1, left_aucs, axis_labels, left_title, left_note)
    _draw_single_spider(ax2, right_aucs, axis_labels, right_title, right_note)

    # Shared legend — build from union of methods across both panels
    all_methods = []
    for d in (left_aucs, right_aucs):
        for m in d:
            if m not in all_methods:
                all_methods.append(m)
    handles, labels = [], []
    for m in all_methods:
        lw = METHOD_LINEWIDTH.get(m, 1.6)
        handles.append(plt.Line2D([], [], color=COLORS.get(m, "black"), lw=lw, marker="o"))
        labels.append(m)
    handles.append(plt.Line2D([], [], color="gray", ls="--", lw=0.8))
    labels.append("Chance (AUC = 0.5)")
    ncol = min(len(labels), 4)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.00)

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    for ext in ("png", "pdf"):
        p = out_path_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"Wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _bench_ec_labels(bench_cache):
    """Hard UniProt EC L1 labels from benchmark cache."""
    out = {}
    for a, v in bench_cache.items():
        ec = v["labels"]["ec_l1"]
        out[a] = 0 if ec is None else int(ec)
    return out


def _idpro_cv_per_class_auc_benchmark():
    """
    Pull 5-fold CV per-class AUCs from PROBE_RESULTS_DIR/cv5fold.json.
    Supports both the original schema (cv_5fold.A+B+C_linear.per_class_fold_mean)
    and the rebuilt schema (idpro_5fold_cv.per_class_mean[c].auc_mean).
    """
    candidates = [
        PROBE_RESULTS_DIR / "cv5fold.json",
        PROBE_RESULTS_DIR / "ec_classifier_evaluation.json",
    ]
    jp = next((p for p in candidates if p.exists()), None)
    if jp is None:
        return None
    d = json.loads(jp.read_text())

    # New schema (rebuild_ec_classifier_json.py)
    if "idpro_5fold_cv" in d:
        per = d["idpro_5fold_cv"].get("per_class_mean", {})
        out = [per.get(str(c), {}).get("auc_mean") for c in range(8)]
        if all(v is not None for v in out):
            return out

    # Legacy schema
    cv = d.get("cv_5fold", {}).get("A+B+C_linear", {})
    per = cv.get("per_class_fold_mean", None)
    if per and len(per) == 8 and all(v is not None for v in per):
        return per
    return None


def main():
    from collections import Counter
    axis_labels = [CLASS_NAMES[c] for c in range(8)]

    # --------- Panel B: DARK proteome (weak GO labels) ---------
    print("=" * 60)
    print("PANEL B: Dark proteome (weak GO-derived EC labels)")
    print("=" * 60)
    labels = build_weak_dark_labels()
    dark_cache = torch.load(EMB_DIR / "dark_embeddings.pt", map_location="cpu", weights_only=False)
    baseline_preds_dark = load_baseline_preds()
    have_dark = [a for a in labels if a in dark_cache and a in baseline_preds_dark]
    y_dark = np.array([labels[a] for a in have_dark])
    print(f"  n_eval = {len(have_dark)}")
    print(f"  EC distribution = {dict(sorted(Counter(y_dark.tolist()).items()))}")

    baseline_scores_dark = baseline_strict_matrix(have_dark, baseline_preds_dark)
    base_auc_dark = per_class_auc(y_dark, baseline_scores_dark)
    idpro_scores_dark = idpro_probe_scores_on_dark(have_dark)
    idpro_auc_dark = per_class_auc(y_dark, idpro_scores_dark)
    for c in range(8):
        n = int((y_dark == c).sum())
        ip = idpro_auc_dark[c]
        bp = base_auc_dark[c]
        print(f"  {c} {CLASS_NAMES[c]:15s} (n={n:3d})  "
              f"IDPro={ip if np.isnan(ip) else f'{ip:.3f}':<6s}  "
              f"baseline={bp if np.isnan(bp) else f'{bp:.3f}'}")
    valid_i = [a for a in idpro_auc_dark if not np.isnan(a)]
    valid_b = [a for a in base_auc_dark if not np.isnan(a)]
    print(f"  IDPro macro = {np.mean(valid_i):.3f}  |  baseline macro = {np.mean(valid_b):.3f}")

    # --------- Panel A: BENCHMARK (UniProt labels, all methods) ---------
    print()
    print("=" * 60)
    print("PANEL A: Benchmark (UniProt-labeled, hard EC labels)")
    print("=" * 60)
    bench_cache = torch.load(EMB_DIR / "benchmark_embeddings.pt", map_location="cpu", weights_only=False)
    bench_labels = _bench_ec_labels(bench_cache)

    # Load all competing baselines (their accession subsets differ)
    method_preds = {}
    method_preds["P2T baseline"] = load_benchmark_baseline_preds()  # full 669
    for mname, fn in BENCH_METHOD_FILES.items():
        try:
            method_preds[mname] = load_benchmark_method_preds(fn)
        except Exception as e:
            print(f"  skip {mname}: {e}")

    # Per-method evaluation on its own supported subset
    bench_aucs = {}
    bench_macro = {}
    bench_n = {}
    for mname, preds in method_preds.items():
        # Intersect with proteins we have UniProt EC labels for
        accs = [a for a in preds if a in bench_labels]
        if not accs:
            continue
        y = np.array([bench_labels[a] for a in accs])
        if mname == "DeepGOMeta":
            scores = generic_baseline_strict_matrix(accs, preds, mode="go_ids")
        else:
            scores = generic_baseline_strict_matrix(accs, preds, mode="text")
        per = per_class_auc(y, scores)
        valid = [a for a in per if not np.isnan(a)]
        bench_aucs[mname] = per
        bench_macro[mname] = float(np.mean(valid)) if valid else float("nan")
        bench_n[mname] = len(accs)

    # IDPro numbers (5-fold CV on the full 3,637-protein pool)
    idpro_auc_bench = _idpro_cv_per_class_auc_benchmark()
    if idpro_auc_bench is None:
        idpro_auc_bench = [np.nan] * 8
    valid_i_b = [a for a in idpro_auc_bench if not np.isnan(a)]
    bench_macro["IDPro classifier probe"] = float(np.mean(valid_i_b)) if valid_i_b else float("nan")
    bench_n["IDPro classifier probe"] = 3637  # 5-fold CV sample size
    bench_aucs["IDPro classifier probe"] = list(idpro_auc_bench)

    # Print table
    method_order = ["IDPro classifier probe", "InterLabelGO+",
                    "P2T baseline", "P2T (RAG transfer)",
                    "BioReason-Pro", "DeepFRI", "MMseqs2", "DeepGOMeta"]
    print(f"  {'method':<26} {'n':>4}  " + " ".join(f"{CLASS_NAMES[c][:8]:>8}" for c in range(8)) + "  macro")
    for mname in method_order:
        if mname not in bench_aucs:
            continue
        cells = []
        for c in range(8):
            a = bench_aucs[mname][c]
            cells.append(f"{'n/a':>8}" if np.isnan(a) else f"{a:>8.3f}")
        print(f"  {mname:<26} {bench_n[mname]:>4}  " + " ".join(cells) +
              f"  {bench_macro[mname]:.3f}")

    # --------- Plot ---------
    # Put IDPro first in the dict (ensures legend ordering)
    left_aucs = {"IDPro classifier probe": list(idpro_auc_bench)}
    for mname in method_order[1:]:
        if mname in bench_aucs:
            left_aucs[mname] = bench_aucs[mname]

    # Right panel — dark proteome. Include MMseqs2 and DeepFRI (we ran them).
    right_aucs = {"IDPro classifier probe": list(idpro_auc_dark),
                  "P2T baseline": list(base_auc_dark)}

    dark_n_by_method = {"IDPro classifier probe": len(have_dark),
                        "P2T baseline": len(have_dark)}
    dark_macro_by_method = {
        "IDPro classifier probe": float(np.mean(valid_i)) if valid_i else float("nan"),
        "P2T baseline": float(np.mean(valid_b)) if valid_b else float("nan"),
    }

    # Run remaining baselines on dark (if predictions exist from run_baselines.py)
    # InterLabelGO+ included with overlap caveat (see note at top); CLEAN excluded.
    extra_dark_files = {
        "MMseqs2": "mmseqs_dark_predictions.json",
        "DeepFRI": "deepfri_dark_MF_predictions.csv",
        "InterLabelGO+": "interlabelgo_dark_predictions.json",
    }
    for mname, fname in extra_dark_files.items():
        p = BENCH_METHODS_DIR / fname
        if not p.exists():
            continue
        if fname.endswith(".json"):
            dark_preds = json.loads(p.read_text())
        else:
            # DeepFRI CSV: concat top-10 GO+EC terms as text for strict-keyword scoring
            dark_preds = _deepfri_csv_to_text(p)
            # Combine with EC predictions if present
            ec_csv = BENCH_METHODS_DIR / "deepfri_dark_EC_predictions.csv"
            if ec_csv.exists():
                ec_text = _deepfri_csv_to_text(ec_csv)
                for prot, t in ec_text.items():
                    dark_preds[prot] = (dark_preds.get(prot, "") + " | " + t).strip(" |")
        # Score on intersection with our weak-labeled dark subset
        accs = [a for a in have_dark if a in dark_preds]
        if not accs:
            continue
        y = np.array([labels[a] for a in accs])
        scores = generic_baseline_strict_matrix(accs, dark_preds, mode="text")
        per = per_class_auc(y, scores)
        right_aucs[mname] = per
        dark_n_by_method[mname] = len(accs)
        valid = [a for a in per if not np.isnan(a)]
        dark_macro_by_method[mname] = float(np.mean(valid)) if valid else float("nan")

    # Build a compact, wrapped macro-AUC summary for Panel A
    macro_lines = [f"Macro-AUC by method (higher = better):"]
    for m in method_order:
        if m in bench_aucs:
            mark = "★" if m == "IDPro classifier probe" else " "
            macro_lines.append(f"  {mark} {m}: {bench_macro[m]:.3f}  (n={bench_n[m]})")
    bench_note = "\n".join(macro_lines)
    dark_method_order = ["IDPro classifier probe", "InterLabelGO+",
                         "P2T baseline", "P2T (RAG transfer)", "BioReason-Pro",
                         "DeepFRI", "MMseqs2", "DeepGOMeta"]
    dark_lines = [f"n = {len(have_dark)} dark proteins (weak GO-derived EC labels)",
                  "Macro-AUC by method (higher = better):"]
    for m in dark_method_order:
        if m in right_aucs:
            mark = "★" if m == "IDPro classifier probe" else " "
            dark_lines.append(
                f"  {mark} {m}: {dark_macro_by_method[m]:.3f}  (n={dark_n_by_method[m]})"
            )
    dark_note = "\n".join(dark_lines)

    spider_two_panel(
        axis_labels=axis_labels,
        left_aucs=left_aucs,
        left_title="A. Benchmark (UniProt EC labels)",
        left_note=bench_note,
        right_aucs=right_aucs,
        right_title="B. Dark proteome (weak GO-derived EC labels)",
        right_note=dark_note,
        out_path_stem=FIG_DIR / "spider_ec_v2_twopanel",
        suptitle="EC-class-level-1 AUC: IDPro classifier probe vs text-output baselines "
                 "(strict keyword-match rule; DeepGOMeta uses GO-ID→EC mapping)",
    )

    # Compact single-panel version for proposal panel layouts:
    # panel A only, no title, no n-label, legend only.
    # Include an "Overall" axis (mean of non-NaN per-class AUCs) so each
    # method's macro-AUC is visible at a glance alongside its class profile.
    axis_labels_ext = axis_labels + ["Overall"]
    left_aucs_ext = _add_overall(left_aucs)
    spider_single_panel(
        axis_labels=axis_labels_ext,
        aucs_by_method=left_aucs_ext,
        out_path_stem=FIG_DIR / "spider_ec_v2",
    )

    # Save numbers
    out = {
        "benchmark": {
            "per_method": {
                m: {
                    "n": bench_n.get(m, 0),
                    "macro_auc": bench_macro.get(m, None),
                    "per_class": {int(c): (None if np.isnan(v) else v) for c, v in enumerate(bench_aucs[m])},
                }
                for m in bench_aucs
            },
        },
        "dark": {
            "n_eval": len(have_dark),
            "ec_distribution": {int(k): int(v) for k, v in Counter(y_dark.tolist()).items()},
            "per_method": {
                m: {
                    "n": dark_n_by_method.get(m, 0),
                    "macro_auc": dark_macro_by_method.get(m, None),
                    "per_class": {int(c): (None if np.isnan(v) else v) for c, v in enumerate(right_aucs[m])},
                }
                for m in right_aucs
            },
        },
        "class_names": CLASS_NAMES,
    }
    (FIG_DIR / "spider_ec_v2.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {FIG_DIR / 'spider_ec_v2.json'}")


if __name__ == "__main__":
    main()
