"""
Per-residue MLP projector.
Maps encoder dim → LLM dim for each residue token independently.
"""

import torch
import torch.nn as nn

from ..config import ProjectorConfig


class ResidueProjector(nn.Module):
    """
    Per-residue MLP projection from encoder space to LLM space.
    Each residue is projected independently (no cross-residue interaction).

    Input: (B, T, encoder_dim)
    Output: (B, T, llm_dim)
    """

    def __init__(self, encoder_dim: int, llm_dim: int, config: ProjectorConfig):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim

        if config.num_layers == 1:
            self.proj = nn.Linear(encoder_dim, llm_dim)
        elif config.num_layers == 2:
            # Standard 2-layer MLP (same as LLaVA)
            act = nn.GELU() if config.activation == "gelu" else nn.SiLU()
            self.proj = nn.Sequential(
                nn.Linear(encoder_dim, llm_dim),
                act,
                nn.Linear(llm_dim, llm_dim),
            )
        else:
            layers = []
            dims = [encoder_dim] + [llm_dim] * config.num_layers
            act = nn.GELU() if config.activation == "gelu" else nn.SiLU()
            for i in range(config.num_layers):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < config.num_layers - 1:
                    layers.append(act)
            self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, encoder_dim) — per-residue embeddings
        Returns:
            (B, T, llm_dim) — projected tokens ready for LLM
        """
        return self.proj(x)
