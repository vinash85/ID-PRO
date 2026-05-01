"""Resolved paths for IDPro. All values come from env vars set by env.sh.

Source the env.sh template, then `from idpro.paths import QA_DIR, CKPT_DIR, ...`.
Failing fast with a useful message if a required var is missing is intentional —
we'd rather crash at import time than write outputs to /tmp."""
import os
from pathlib import Path
from typing import Optional


def _req(var: str) -> Path:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"{var} is not set. Source env.sh from the repo root before running.")
    return Path(val)


def _opt(var: str, default: Optional[Path] = None) -> Optional[Path]:
    val = os.environ.get(var)
    return Path(val) if val else default


REPO_ROOT     = _opt("IDPRO_REPO_ROOT", Path(__file__).resolve().parent.parent)
DATA_ROOT     = _req("IDPRO_DATA_ROOT")
RUNS_ROOT     = _req("IDPRO_RUNS_ROOT")
QWEN_PATH     = _req("IDPRO_QWEN_PATH")
HF_CACHE      = _opt("IDPRO_HF_CACHE")
P2T_DIR       = _opt("IDPRO_P2T_DIR")

# Derived inputs
QA_DIR        = DATA_ROOT / "preliminary_data" / "training_data" / "qa_stages"
BENCHMARK     = DATA_ROOT / "preliminary_data" / "benchmark" / "microbiome_benchmark.json"
FEATURE_INDEX = DATA_ROOT / "feature_index.pkl"

# Derived outputs (per-run subdirs created by trainers)
CKPT_DIR        = RUNS_ROOT / "checkpoints"
RESULTS_DIR     = RUNS_ROOT / "training_results"
AIM1_PROBE_DIR  = RUNS_ROOT / "aim1" / "probe"
REPORTS_DIR     = REPO_ROOT / "reports"
FIGURES_DIR     = REPORTS_DIR / "figures"
