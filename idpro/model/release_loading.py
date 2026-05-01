"""Helpers for loading IDPro release artifacts from a local path or HF repo id.

The Aim 1 evaluation scripts take a ``--ckpt`` argument that historically
pointed at a local directory containing one of:

  - ``idpro_state.pt``                   (HF release format, written by
                                          tools/hf/export_for_hf.py)
  - ``trainable.pt``                     (source-repo training-script format)
  - ``mp_rank_00_model_states.pt``       (DeepSpeed ZeRO-2 checkpoint format)

This module lets those scripts also accept a Hugging Face repo id
(``"tumorailab/IDPRO-ESMC600M"``) — the helper detects that case, calls
``snapshot_download`` to materialise the repo locally, and returns the
local directory path. From there the rest of the loading code is unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


_STATE_FILE_PRIORITY = (
    "idpro_state.pt",
    "trainable.pt",
    "mp_rank_00_model_states.pt",
)


def _looks_like_hf_repo_id(spec: str) -> bool:
    """Heuristic: ``"<user-or-org>/<repo>"`` with no path separators beyond one slash."""
    if os.path.sep in spec and os.path.exists(spec):
        return False
    if spec.startswith((".", "/", "~")):
        return False
    parts = spec.split("/")
    return len(parts) == 2 and all(p and "." not in p[:1] for p in parts)


def resolve_release(
    spec: Union[str, os.PathLike],
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """Resolve ``spec`` to a local directory containing the IDPro release.

    Args:
        spec: Either a local path or a HF repo id (``"user/repo"``).
        revision: Optional HF revision (commit / tag / branch).
        cache_dir: Optional override for the HF download cache.

    Returns:
        Path to a local directory holding the release files.
    """
    s = os.fspath(spec)
    p = Path(s)
    if p.exists() and p.is_dir():
        return p
    if _looks_like_hf_repo_id(s):
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=s, revision=revision, cache_dir=cache_dir)
        return Path(local)
    raise FileNotFoundError(
        f"--ckpt='{s}' is neither an existing directory nor a recognised HF repo id."
    )


def find_state_file(release_dir: Path) -> Path:
    """Pick the best state file inside ``release_dir`` based on priority order."""
    for name in _STATE_FILE_PRIORITY:
        candidate = release_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No IDPro state file in {release_dir}. Looked for: {_STATE_FILE_PRIORITY}"
    )


def load_state_dict(release_dir: Path, *, map_location: str = "cpu") -> dict:
    """Load the trainable-only state dict from a release directory.

    Handles all three on-disk formats and returns a flat ``{name: tensor}`` dict.
    """
    import torch

    state_file = find_state_file(release_dir)
    blob = torch.load(state_file, map_location=map_location, weights_only=False)
    if state_file.name == "mp_rank_00_model_states.pt":
        # DeepSpeed wraps the state dict under a "module" key.
        return blob["module"]
    return blob
