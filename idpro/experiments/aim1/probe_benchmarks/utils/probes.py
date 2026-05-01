"""Linear and MLP probe heads + a single training/inference loop reused by
every probe script in `probe_benchmarks/` and `conformal/`."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _build(kind: str, in_dim: int, out_dim: int) -> nn.Module:
    if kind == "linear":
        return LinearProbe(in_dim, out_dim)
    if kind == "mlp":
        return MLPProbe(in_dim, out_dim)
    raise ValueError(f"unknown probe kind: {kind!r}")


def train_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    out_dim: int,
    task: str,
    kind: str = "linear",
    device: str = "cuda",
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    seed: Optional[int] = None,
) -> nn.Module:
    """Train one probe. `task` ∈ {is_enzyme, ec_l1, go_f_top20, pfam_top20}."""
    if seed is not None:
        torch.manual_seed(seed)

    if task == "is_enzyme":
        loss_fn = nn.BCEWithLogitsLoss()
    elif task == "ec_l1":
        loss_fn = nn.CrossEntropyLoss()
    elif task in ("go_f_top20", "pfam_top20"):
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"unknown task: {task!r}")

    probe = _build(kind, x_train.shape[1], out_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    x = x_train.to(device)
    y = y_train.to(device)
    use_mini = x.shape[0] > 1024

    for _ in range(epochs):
        probe.train()
        if use_mini:
            perm = torch.randperm(x.shape[0], device=device)
            for s in range(0, x.shape[0], batch_size):
                idx = perm[s:s + batch_size]
                logits = probe(x[idx])
                if task == "is_enzyme":
                    loss = loss_fn(logits.squeeze(-1), y[idx].float())
                else:
                    loss = loss_fn(logits, y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        else:
            logits = probe(x)
            if task == "is_enzyme":
                loss = loss_fn(logits.squeeze(-1), y.float())
            else:
                loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

    probe.eval()
    return probe


@torch.no_grad()
def predict(probe: nn.Module, x: torch.Tensor, device: str, task: str):
    """Return numpy scores. Binary → (N,); multiclass → (N,C) softmax;
    multilabel → (N,C) sigmoid."""
    logits = probe(x.to(device))
    if task == "is_enzyme":
        return torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
    if task == "ec_l1":
        return torch.softmax(logits, dim=-1).cpu().numpy()
    return torch.sigmoid(logits).cpu().numpy()
