"""
IDPro model configuration.
Encoder-agnostic: swap ESM C ↔ ESM2 by changing encoder_name and encoder_dim.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from .paths import QWEN_PATH


@dataclass
class EncoderConfig:
    """Protein encoder configuration."""
    name: str = "esmc-600m"          # "esmc-600m", "esmc-300m", "esm2-650m", "esm2-3b", "esm3-1.4b"
    dim: int = 1152                   # ESM C 600M=1152, ESM2-650M=1280, ESM3-1.4B=1536
    freeze: bool = True               # Always freeze encoder
    select_layer: int = -1            # Which layer's output to use (-1 = last)

    # Backend dispatch — auto-set by resolve() from PRESETS
    backend: str = "esmc"             # "esmc" | "esm2" | "esm3"

    # ESM3 structure-track knobs (no-ops for esmc/esm2)
    structure_track: bool = False     # If True, populate ESM3 structure track from PDB
    structure_manifest_path: str = "" # JSONL: {"accession": ..., "pdb_path": ...}

    # Predefined encoder configs
    PRESETS = {
        "esmc-600m": {"name": "EvolutionaryScale/esmc-600m-2024-12", "dim": 1152, "backend": "esmc"},
        "esmc-300m": {"name": "EvolutionaryScale/esmc-300m-2024-12", "dim": 960,  "backend": "esmc"},
        "esm2-650m": {"name": "facebook/esm2_t33_650M_UR50D",        "dim": 1280, "backend": "esm2"},
        "esm2-3b":   {"name": "facebook/esm2_t36_3B_UR50D",          "dim": 2560, "backend": "esm2"},
        "esm3-1.4b": {"name": "esm3-sm-open-v1",                     "dim": 1536, "backend": "esm3"},
    }

    def resolve(self):
        """Fill in preset values."""
        if self.name in self.PRESETS:
            preset = self.PRESETS[self.name]
            self.name = preset["name"]
            self.dim = preset["dim"]
            self.backend = preset.get("backend", self.backend)
        return self


@dataclass
class AdaptorConfig:
    """Token-preserving residue adaptor (1D Conv)."""
    num_layers: int = 2               # Number of conv layers
    kernel_size: int = 7              # Local context window (±3 residues)
    dropout: float = 0.0
    # Input/output dim matches encoder dim — set automatically


@dataclass
class ProjectorConfig:
    """Per-residue MLP projector."""
    num_layers: int = 2               # 2-layer MLP (like LLaVA)
    activation: str = "gelu"
    # Input dim = encoder dim, output dim = llm dim — set automatically


@dataclass
class LLMConfig:
    """Language model backbone configuration."""
    name: str = "Qwen/Qwen3.5-27B"
    dim: int = 3584                   # Hidden dim of the LLM
    max_context: int = 262144         # Max context window

    # LoRA config
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )

    # Training precision
    dtype: str = "bf16"               # "bf16" or "fp16"
    use_qlora: bool = False           # bf16 LoRA recommended for Qwen3.5

    # Predefined LLM configs
    PRESETS = {
        "qwen3.5-27b": {"name": str(QWEN_PATH), "dim": 5120},
        "qwen3-14b":   {"name": "Qwen/Qwen3-14B-Base", "dim": 5120},
        "llama3.1-8b": {"name": "meta-llama/Meta-Llama-3.1-8B-Instruct", "dim": 4096},
    }

    def resolve(self):
        short = self.name.lower().replace("/", "").replace(".", "").replace("_", "").replace("-", "")
        for key, preset in self.PRESETS.items():
            key_normalized = key.replace(".", "").replace("_", "").replace("-", "")
            if key_normalized in short or short in key_normalized:
                self.name = preset["name"]
                self.dim = preset["dim"]
                break
        return self


@dataclass
class RAGConfig:
    """RAG retrieval configuration."""
    enabled: bool = True
    k: int = 5                        # Number of neighbors to retrieve
    embedding_index_path: str = ""    # Path to precomputed ESM embedding index
    max_context_tokens: int = 512     # Max tokens for RAG context


@dataclass
class TrainingConfig:
    """Training hyperparameters per stage."""
    stage: int = 1                    # 1=alignment, 2=structure, 3=composition, 4=reasoning

    # What to train per stage
    train_adaptor: bool = True
    train_projector: bool = True
    train_llm_lora: bool = False
    freeze_encoder: bool = True

    # Hyperparameters
    lr: float = 2e-3
    batch_size: int = 4
    grad_accum_steps: int = 8
    epochs: int = 1
    max_seq_len: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"

    # Data mixing (anti-forgetting replay)
    replay_fraction: float = 0.0      # Fraction of data from previous stages

    # RAG
    rag_fraction: float = 0.0         # Fraction of samples with RAG context

    # Stage presets
    # Stage 0: Low-level features (AA composition, hydrophobicity, charge)
    # Stage 1: Domain/motif recognition from subsequences
    # Stage 2: Structural context (fold, TM, disorder)
    # Stage 3: Multi-turn decomposed reasoning (IDENTIFY→LOCATE→RELATE→INFER)
    # Stage 4: Single-turn CoT (internalized reasoning)
    # Stage 5: DPO (prefer correct over hallucinated)
    # Stage 6: GRPO (optimize composite reward)
    STAGE_PRESETS = {
        0: {"train_adaptor": True, "train_projector": True, "train_llm_lora": False,
            "lr": 2e-3, "batch_size": 16, "epochs": 1, "rag_fraction": 0.0,
            "replay_fraction": 0.0, "max_seq_len": 256},
        1: {"train_adaptor": True, "train_projector": True, "train_llm_lora": False,
            "lr": 2e-3, "batch_size": 8, "epochs": 1, "rag_fraction": 0.0,
            "replay_fraction": 0.0, "max_seq_len": 512},
        2: {"train_adaptor": False, "train_projector": False, "train_llm_lora": True,
            "lr": 2e-5, "batch_size": 4, "epochs": 2, "rag_fraction": 0.5,
            "replay_fraction": 0.0, "max_seq_len": 2048},
        3: {"train_adaptor": True, "train_projector": True, "train_llm_lora": True,
            "lr": 1e-5, "batch_size": 2, "epochs": 3, "rag_fraction": 0.7,
            "replay_fraction": 0.2, "max_seq_len": 4096},
        4: {"train_adaptor": False, "train_projector": False, "train_llm_lora": True,
            "lr": 5e-6, "batch_size": 2, "epochs": 2, "rag_fraction": 0.8,
            "replay_fraction": 0.1, "max_seq_len": 4096},
        5: {"train_adaptor": False, "train_projector": False, "train_llm_lora": True,
            "lr": 5e-7, "batch_size": 2, "epochs": 1, "rag_fraction": 0.8,
            "replay_fraction": 0.1, "max_seq_len": 4096},
        6: {"train_adaptor": False, "train_projector": False, "train_llm_lora": True,
            "lr": 1e-7, "batch_size": 1, "epochs": 1, "rag_fraction": 0.8,
            "replay_fraction": 0.1, "max_seq_len": 4096},
    }

    def apply_stage_preset(self):
        if self.stage in self.STAGE_PRESETS:
            for k, v in self.STAGE_PRESETS[self.stage].items():
                setattr(self, k, v)
        return self


@dataclass
class IDProConfig:
    """Full IDPro model configuration."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    adaptor: AdaptorConfig = field(default_factory=AdaptorConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Special tokens
    prot_start_token: str = "<PROT_START>"
    prot_end_token: str = "<PROT_END>"
    rag_start_token: str = "<RAG_START>"
    rag_end_token: str = "<RAG_END>"

    def resolve(self):
        self.encoder.resolve()
        self.llm.resolve()
        return self

    def validate(self):
        """Check config consistency."""
        assert self.encoder.dim > 0, "Encoder dim must be positive"
        assert self.llm.dim > 0, "LLM dim must be positive"
        assert self.adaptor.kernel_size % 2 == 1, "Kernel size must be odd"
        return True
