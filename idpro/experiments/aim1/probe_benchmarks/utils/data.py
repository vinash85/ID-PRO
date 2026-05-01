"""Embedding-cache loaders + label/view stacking. The probe scripts all
operate on the .pt caches written by extract_embeddings.py and the .npz ESM C
mean-pool index, so this is a tiny adapter layer over those file formats."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


VIEWS: Tuple[str, str, str] = (
    "view_a_prompteol_l48",
    "view_b_question_mean_l48",
    "view_c_eos_l64",
)
ESMC_VIEW = "esmc_mean_pool"

N_CLASSES = 8
CLASS_NAMES: Dict[int, str] = {
    0: "Non-enzyme", 1: "Oxidoreductase", 2: "Transferase", 3: "Hydrolase",
    4: "Lyase", 5: "Isomerase", 6: "Ligase", 7: "Translocase",
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> List[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_emb_cache(path: Path) -> Dict[str, dict]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_esmc_index(rag_npz: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load ESM C mean-pool embeddings (RAG index) as accession → vector."""
    if not rag_npz.exists():
        return None
    npz = np.load(rag_npz, allow_pickle=True)
    return dict(zip(npz["ids"].tolist(), npz["embs"]))


# ---------------------------------------------------------------------------
# View stacking + variant builders
# ---------------------------------------------------------------------------


def stack_views(
    cache: Dict[str, dict],
    accs: List[str],
    views: List[str],
    esmc_embs: Optional[Dict[str, np.ndarray]] = None,
) -> torch.Tensor:
    """Concatenate the requested per-protein views into (N, sum(dim)).
    Pass `esmc_embs` to support the protein-only ESM C baseline view."""
    tensors = []
    for v in views:
        if v == ESMC_VIEW:
            if esmc_embs is None:
                raise ValueError("esmc_embs required when stacking esmc_mean_pool")
            tensors.append(torch.stack([torch.from_numpy(esmc_embs[a]).float() for a in accs]))
        else:
            tensors.append(torch.stack([cache[a][v].float() for a in accs]))
    return torch.cat(tensors, dim=-1)


def iter_variants(include_esmc: bool = True) -> List[Tuple[str, List[str]]]:
    """Standard variant grid used by `train_probe_variants`-style sweeps."""
    a, b, c = VIEWS
    variants: List[Tuple[str, List[str]]] = [
        ("A_prompteol_l48", [a]),
        ("B_question_l48", [b]),
        ("C_eos_l64", [c]),
        ("A+B", [a, b]),
        ("A+C", [a, c]),
        ("B+C", [b, c]),
        ("A+B+C", [a, b, c]),
    ]
    if include_esmc:
        variants.append(("ESMC_baseline", [ESMC_VIEW]))
    return variants


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def ec_label(cache: Dict[str, dict], acc: str) -> int:
    """Multi-class EC L1 label. None / no-EC → 0 (non-enzyme)."""
    v = cache[acc]["labels"]["ec_l1"]
    return 0 if v is None else int(v)


def load_labels(cache: Dict[str, dict], accs: List[str], task: str) -> torch.Tensor:
    if task == "is_enzyme":
        return torch.tensor(
            [cache[a]["labels"]["is_enzyme"] for a in accs], dtype=torch.long
        )
    if task == "ec_l1":
        return torch.tensor([ec_label(cache, a) for a in accs], dtype=torch.long)
    if task == "go_f_top20":
        return torch.tensor(
            [cache[a]["labels"]["go_f"] for a in accs], dtype=torch.float32
        )
    if task == "pfam_top20":
        return torch.tensor(
            [cache[a]["labels"]["pfam"] for a in accs], dtype=torch.float32
        )
    raise ValueError(f"unknown task: {task!r}")


def task_out_dim(task: str, y_train: torch.Tensor) -> int:
    if task == "is_enzyme":
        return 1
    if task == "ec_l1":
        return N_CLASSES
    return int(y_train.shape[1])  # multi-label dims come from the label tensor


def task_loss(task: str) -> str:
    return {
        "is_enzyme": "bce",
        "ec_l1": "ce",
        "go_f_top20": "bce",
        "pfam_top20": "bce",
    }[task]
