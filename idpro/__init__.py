"""
IDPro: Illuminating the Dark PROteome
Per-residue multimodal protein function prediction with chain-of-thought reasoning.

Architecture:
  ESM C/ESM2 (frozen) → Residue Adaptor (1D Conv) → Projector (MLP) → Qwen3.5/LLaMA (LoRA)

Training stages:
  1. Alignment: Train adaptor + projector (LLM frozen)
  2. Structure: LLM LoRA on structural context
  3. Composition: Multi-turn decomposed reasoning
  4. Reasoning: Single-turn CoT (IDENTIFY→LOCATE→RELATE→INFER→CONTEXTUALIZE)
"""

__version__ = "0.1.0"
