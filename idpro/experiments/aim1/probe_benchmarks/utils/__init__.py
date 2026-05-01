"""Shared helpers for probe_benchmarks/ scripts.

Re-exports the most commonly used primitives so callers can write
`from idpro.experiments.aim1.probe_benchmarks.utils import LinearProbe, ...`.
"""
from .probes import LinearProbe, MLPProbe, train_probe, predict
from .data import (
    VIEWS,
    ESMC_VIEW,
    N_CLASSES,
    CLASS_NAMES,
    load_jsonl,
    load_emb_cache,
    load_esmc_index,
    stack_views,
    ec_label,
    load_labels,
    task_out_dim,
    task_loss,
    iter_variants,
)
from .metrics import (
    EC_LABEL_KEYWORDS,
    EC_PRED_KEYWORDS,
    GO_TO_EC,
    weak_ec_l1_from_go,
    strict_class_scores,
    deepgometa_strict_scores,
    per_class_auc,
    macro_from_per_class,
    compute_auc,
)

__all__ = [
    "LinearProbe", "MLPProbe", "train_probe", "predict",
    "VIEWS", "ESMC_VIEW", "N_CLASSES", "CLASS_NAMES", "load_jsonl",
    "load_emb_cache", "load_esmc_index", "stack_views", "ec_label",
    "load_labels", "task_out_dim", "task_loss", "iter_variants",
    "EC_LABEL_KEYWORDS", "EC_PRED_KEYWORDS", "GO_TO_EC",
    "weak_ec_l1_from_go", "strict_class_scores", "deepgometa_strict_scores",
    "per_class_auc", "macro_from_per_class", "compute_auc",
]
