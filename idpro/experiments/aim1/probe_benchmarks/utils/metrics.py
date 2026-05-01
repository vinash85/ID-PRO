"""Per-class / macro AUC, plus the EC-L1 weak-label and strict-keyword
scoring rules shared by the probe + baseline + reporting scripts."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# EC L1 keyword dictionaries
# ---------------------------------------------------------------------------

# Used to derive WEAK EC-L1 labels from GO annotation strings on dark proteins.
EC_LABEL_KEYWORDS: Dict[int, List[str]] = {
    1: ["oxidoreductase activity", "dehydrogenase", "reductase", "oxidase", "oxygenase", "peroxidase"],
    2: ["transferase activity", "kinase activity", "methyltransferase", "acyltransferase", "glycosyltransferase"],
    3: ["hydrolase activity", "peptidase", "protease", "nuclease", "phosphatase", "esterase", "lipase", "glycosidase"],
    4: ["lyase activity", "decarboxylase", "aldolase", "dehydratase", "synthase"],
    5: ["isomerase activity", "racemase", "epimerase", "mutase"],
    6: ["ligase activity", "synthetase"],
    7: ["transporter activity", "transmembrane transport", "channel activity", "permease"],
}

# Used to PREDICT EC-L1 from natural-language baseline outputs (broader vocab).
EC_PRED_KEYWORDS: Dict[int, List[str]] = {
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

# Canonical enzyme-activity GO IDs → EC class (used for tools that emit raw GO IDs).
GO_TO_EC: Dict[str, int] = {
    "GO:0016491": 1, "GO:0016614": 1,
    "GO:0016740": 2, "GO:0016301": 2, "GO:0016772": 2,
    "GO:0016787": 3, "GO:0008233": 3, "GO:0016788": 3, "GO:0016798": 3,
    "GO:0016810": 3, "GO:0004518": 3,
    "GO:0016829": 4, "GO:0016830": 4, "GO:0016831": 4, "GO:0016835": 4,
    "GO:0016853": 5, "GO:0016854": 5, "GO:0016855": 5, "GO:0016860": 5,
    "GO:0016874": 6, "GO:0016875": 6,
    "GO:0022857": 7, "GO:0022804": 7, "GO:0015075": 7, "GO:0005215": 7,
    "GO:0008324": 7,
}


# ---------------------------------------------------------------------------
# Weak label derivation
# ---------------------------------------------------------------------------


def weak_ec_l1_from_go(go_terms: str) -> Optional[int]:
    """Return EC L1 in {0..7} or None if multiple classes match (ambiguous)."""
    if not go_terms:
        return None
    lo = go_terms.lower()
    matched = {ec for ec, kws in EC_LABEL_KEYWORDS.items()
               if any(kw in lo for kw in kws)}
    if not matched:
        return 0
    if len(matched) == 1:
        return next(iter(matched))
    return None


# ---------------------------------------------------------------------------
# Strict per-class keyword score on a prediction string (text or GO IDs)
# ---------------------------------------------------------------------------


def _count_hits(text: str, keywords: List[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


def strict_class_scores(text: str) -> np.ndarray:
    """8-d strict-keyword score: hits(c) only counts if NO other-class
    keyword appears in the text. Used to score natural-language baselines."""
    text_lower = (text or "").lower()
    hits = np.array(
        [_count_hits(text_lower, EC_PRED_KEYWORDS[c]) for c in range(8)],
        dtype=float,
    )
    strict = np.zeros(8, dtype=float)
    for c in range(8):
        other_max = float(np.delete(hits, c).max()) if hits.size > 1 else 0.0
        strict[c] = 0.0 if other_max > 0 else hits[c]
    return strict


def deepgometa_strict_scores(text: str) -> np.ndarray:
    """8-d strict score for tools that emit raw GO IDs (DeepGOMeta).
    Falls back to class-0 (non-enzyme) when no enzyme-activity GO IDs hit."""
    hits = np.zeros(8, dtype=float)
    for go_id in re.findall(r"GO:\d{7}", text or ""):
        ec = GO_TO_EC.get(go_id)
        if ec is not None:
            hits[ec] += 1
    if hits[1:].sum() == 0 and text:
        hits[0] = 1
    strict = np.zeros(8, dtype=float)
    for c in range(8):
        other_max = float(np.delete(hits, c).max()) if hits.size > 1 else 0.0
        strict[c] = 0.0 if other_max > 0 else hits[c]
    return strict


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


def per_class_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_classes: int = 8,
) -> List[float]:
    """One-vs-rest AUC per class. Returns NaN for classes with only one label."""
    out: List[float] = []
    for c in range(n_classes):
        yt = (y_true == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            out.append(float("nan"))
            continue
        out.append(float(roc_auc_score(yt, scores[:, c])))
    return out


def macro_from_per_class(per: List[float]) -> float:
    valid = [a for a in per if not np.isnan(a)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task: str,
    n_classes: int = 8,
) -> Tuple[float, List[float]]:
    """Macro-AUC for any of the four classification tasks."""
    if task == "is_enzyme":
        if len(np.unique(y_true)) < 2:
            return float("nan"), [float("nan")]
        auc = float(roc_auc_score(y_true, y_score))
        return auc, [auc]
    if task == "ec_l1":
        per = per_class_auc(y_true, y_score, n_classes)
        return macro_from_per_class(per), per
    # multilabel
    per: List[float] = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            per.append(float("nan"))
            continue
        per.append(float(roc_auc_score(yt, y_score[:, i])))
    return macro_from_per_class(per), per
