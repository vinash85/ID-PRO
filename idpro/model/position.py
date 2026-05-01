"""
Position handling for multimodal protein + text sequences.

THE PROBLEM:
When we prepend 500-2000 protein residue tokens before the text,
the question text starts at position 500-2000 in the context.
But Qwen/LLaMA were trained with text starting at position 0.
The text decoder is optimized for low position IDs, not "token 1500+".

SOLUTIONS EXPLORED:

Option A: Shared Position Axis (naive, bad)
    positions: [0, 1, 2, ... 499, 500, 501, ... 700]
                 protein tokens     text tokens
    Problem: text at position 500+ is out of the decoder's comfort zone

Option B: Position Reset (LLaVA-style)
    positions: [0, 1, 2, ... 499, 0, 1, 2, ... 200]
                 protein tokens    text tokens (reset)
    Problem: attention can't distinguish "protein token 50" from "text token 50"

Option C: Separate Modality Position Embeddings (recommended by chatgpt.md)
    Protein tokens get protein-specific position encoding
    Text tokens get standard text position encoding (starting from 0)
    A modality embedding distinguishes the two

Option D: Offset-Aware RoPE (most elegant)
    Use Qwen's RoPE but with a learned offset:
    - Protein tokens: RoPE with positions [0, 1, ..., T-1]
    - Text tokens: RoPE with positions [0, 1, ..., Q-1] (reset to 0)
    - Cross-modal attention works because RoPE is relative

    Key insight: RoPE encodes RELATIVE positions, not absolute.
    So "protein token 50 attending to protein token 45" computes
    the same relative position as "text token 50 attending to text token 45".
    Cross-modal attention (text attending to protein) uses the difference
    between the two position systems — this is fine because attention
    learns the cross-modal relationship during training.

Option E: Dual-Stream with Cross-Attention
    Process protein and text in separate streams, combine via cross-attention.
    Most principled but requires architecture changes to the LLM.

RECOMMENDATION: Option D (Offset-Aware RoPE)
- Text positions always start at 0 → decoder operates in its trained regime
- Protein positions are separate → no interference with text positioning
- Cross-modal attention learns during fine-tuning
- Minimal code change: just modify the position_ids before LLM forward
- Compatible with Qwen/LLaMA RoPE without modifying the backbone
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MultimodalPositionManager:
    """
    Manages position IDs for the [protein | prot_end | rag_text | question | answer] layout.

    Creates position IDs where:
    - Protein tokens: positions 0..T-1 (independent axis)
    - Text tokens (rag + question + answer): positions 0..N-1 (reset to 0)

    The modality embedding (already in model.py) tells the LLM which tokens are protein
    vs text. The position reset ensures text operates in the decoder's trained regime.
    """

    @staticmethod
    def create_position_ids(
        protein_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        total_length: int,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Create position IDs with modality-aware reset.

        Args:
            protein_lengths: (B,) number of protein tokens per sample (includes PROT_END)
            text_lengths: (B,) number of text tokens per sample (rag + question + answer)
            total_length: total padded sequence length
            device: target device

        Returns:
            position_ids: (B, total_length) — protein positions then reset text positions
        """
        batch_size = protein_lengths.shape[0]
        position_ids = torch.zeros(batch_size, total_length, dtype=torch.long, device=device)

        for i in range(batch_size):
            prot_len = protein_lengths[i].item()
            text_len = text_lengths[i].item()

            # Protein positions: 0, 1, 2, ..., prot_len-1
            position_ids[i, :prot_len] = torch.arange(prot_len, device=device)

            # Text positions: reset to 0, 1, 2, ..., text_len-1
            position_ids[i, prot_len:prot_len + text_len] = torch.arange(text_len, device=device)

            # Padding positions: 0 (doesn't matter, masked by attention_mask)

        return position_ids

    @staticmethod
    def create_position_ids_continuous(
        protein_lengths: torch.Tensor,
        total_length: int,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Simple continuous positions (fallback / for comparison).
        positions: [0, 1, 2, ..., total_length-1]
        """
        return torch.arange(total_length, device=device).unsqueeze(0).expand(
            protein_lengths.shape[0], -1
        )

    @staticmethod
    def create_position_ids_interleaved(
        protein_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        total_length: int,
        protein_position_scale: float = 0.1,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Scaled protein positions + normal text positions.

        Protein positions are scaled down so that a 500-residue protein
        occupies position range [0, 50] instead of [0, 500], keeping
        text positions in a more natural range.

        This is a softer version of the reset: instead of jumping from
        position 499 to 0, we go from position 49 to 50.
        """
        batch_size = protein_lengths.shape[0]
        position_ids = torch.zeros(batch_size, total_length, dtype=torch.long, device=device)

        for i in range(batch_size):
            prot_len = protein_lengths[i].item()
            text_len = text_lengths[i].item()

            # Protein positions: scaled down
            prot_positions = torch.arange(prot_len, device=device, dtype=torch.float32)
            prot_positions = (prot_positions * protein_position_scale).long()
            position_ids[i, :prot_len] = prot_positions

            # Text positions: continue from where protein left off
            text_start = prot_positions[-1].item() + 1 if prot_len > 0 else 0
            position_ids[i, prot_len:prot_len + text_len] = (
                torch.arange(text_len, device=device) + text_start
            )

        return position_ids


class ProteinPositionEncoding(nn.Module):
    """
    Learned position encoding specifically for protein residues.
    Added to protein token embeddings BEFORE they enter the LLM.

    This gives the model an explicit signal of "where in the protein"
    each residue is, independent of the LLM's own position encoding.

    Two components:
    1. Absolute position: residue number in the sequence
    2. Relative position: fraction through the protein (0.0 = N-terminus, 1.0 = C-terminus)
    """

    def __init__(self, llm_dim: int, max_len: int = 8192):
        super().__init__()
        # Sinusoidal position encoding (no learnable params, works for any length)
        # Stored as bf16 to match model dtype
        self.register_buffer(
            "sin_pos",
            self._sinusoidal_encoding(max_len, llm_dim).to(torch.bfloat16),
            persistent=False,
        )
        # Learned modality-aware position scale
        self.position_scale = nn.Parameter(torch.ones(1))
        # Relative position encoding (N-term to C-term)
        self.relative_proj = nn.Linear(1, llm_dim, bias=False)
        nn.init.zeros_(self.relative_proj.weight)  # start with no contribution

    @staticmethod
    def _sinusoidal_encoding(max_len: int, dim: int) -> torch.Tensor:
        """Standard sinusoidal positional encoding."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, dim)

    def forward(self, protein_tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Add protein-specific position information to residue tokens.

        Args:
            protein_tokens: (B, T, llm_dim) — projected protein tokens
            mask: (B, T) — 1 for real residues, 0 for padding

        Returns:
            (B, T, llm_dim) — position-enriched protein tokens
        """
        B, T, D = protein_tokens.shape

        # Sinusoidal absolute position (scaled)
        abs_pos = self.sin_pos[:, :T, :D] * self.position_scale

        # Relative position (0 to 1 through the protein)
        if mask is not None:
            # Actual lengths per sample
            seq_lens = mask.sum(dim=1, keepdim=True).float()  # (B, 1)
            positions = torch.arange(T, device=protein_tokens.device).float().unsqueeze(0)  # (1, T)
            relative = positions / seq_lens.clamp(min=1)  # (B, T), values 0..1
        else:
            relative = torch.arange(T, device=protein_tokens.device).float().unsqueeze(0) / T

        relative = relative.unsqueeze(-1).to(dtype=protein_tokens.dtype)  # (B, T, 1), match model dtype
        rel_pos = self.relative_proj(relative)  # (B, T, llm_dim)

        abs_pos = abs_pos.to(dtype=protein_tokens.dtype)

        return protein_tokens + abs_pos + rel_pos
