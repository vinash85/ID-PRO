"""
Token-preserving residue adaptor.
Adds local protein context via 1D convolutions without changing token count.
Each residue sees its ±(kernel_size//2) neighbors after this module.
"""

import torch
import torch.nn as nn

from ...config import AdaptorConfig


class ResidueAdaptor(nn.Module):
    """
    Token-preserving adaptor using 1D convolutions.
    Input: (B, T, D) per-residue embeddings
    Output: (B, T, D) contextually enriched embeddings (same shape)

    Each conv layer gives each residue local context from its neighbors.
    Residual connections + LayerNorm for stable training.
    """

    def __init__(self, dim: int, config: AdaptorConfig):
        super().__init__()
        self.dim = dim
        self.config = config

        layers = []
        for i in range(config.num_layers):
            layers.append(ConvResidualBlock(
                dim=dim,
                kernel_size=config.kernel_size,
                dropout=config.dropout,
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) per-residue embeddings
            mask: (B, T) attention mask (1=real, 0=padding)
        Returns:
            (B, T, D) enriched embeddings
        """
        # Mask padding positions
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        for layer in self.layers:
            x = layer(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return x


class ConvResidualBlock(nn.Module):
    """Single conv + norm + residual block."""

    def __init__(self, dim: int, kernel_size: int = 7, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size // 2,
            groups=1,  # full mixing across channels
        )
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        residual = x
        # Conv expects (B, D, T)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.dropout(h)
        return self.norm(h + residual)
