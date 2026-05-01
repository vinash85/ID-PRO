"""
Protein encoder wrapper. Supports ESM C, ESM2, and ESM3 with a unified interface.
Always frozen — provides per-residue embeddings.

For ESM3, an optional `structures` kwarg routes through the structure track
when `config.structure_track=True`. Pass a list of PDB file paths (one per
sequence, or None to mask the structure track for that sample).
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from ..config import EncoderConfig


class ProteinEncoder(nn.Module):
    """
    Unified wrapper for protein encoders (ESM C, ESM2, ESM3).
    Produces per-residue embeddings: (batch_size, seq_len, encoder_dim).
    Always frozen — never trained.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.encoder = None
        self.tokenizer = None
        # ESM3-specific handles, populated by _load_esm3()
        self._ESMProtein = None
        self._LogitsConfig = None
        self._loaded = False

    # ── Backend dispatch ──────────────────────────────────────────────

    def _backend(self) -> str:
        """Resolve backend, falling back to name-sniffing for legacy configs."""
        b = getattr(self.config, "backend", "") or ""
        if b in ("esmc", "esm2", "esm3"):
            return b
        # Backwards-compat: infer from name
        n = self.config.name.lower()
        if "esm3" in n:
            return "esm3"
        if "esmc" in n or "esm-c" in n:
            return "esmc"
        if "esm2" in n:
            return "esm2"
        raise ValueError(f"Unknown encoder backend for name: {self.config.name}")

    def load(self, device: str = "cuda"):
        """Lazy-load the encoder model."""
        if self._loaded:
            return

        backend = self._backend()
        if backend == "esmc":
            self._load_esmc(self.config.name, device)
        elif backend == "esm2":
            self._load_esm2(self.config.name, device)
        elif backend == "esm3":
            self._load_esm3(self.config.name, device)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        self._loaded = True
        struct_str = ""
        if backend == "esm3":
            struct_str = f", structure_track={self.config.structure_track}"
        print(f"[Encoder] Loaded {self.config.name} "
              f"(backend={backend}, dim={self.config.dim}, frozen{struct_str})")

    # ── Loaders ───────────────────────────────────────────────────────

    def _load_esmc(self, name: str, device: str):
        """Load ESM C model."""
        try:
            from esm.models.esmc import ESMC
            from esm.tokenization import EsmSequenceTokenizer

            short_name = name.lower()
            if "600m" in short_name:
                model_name = "esmc_600m"
            elif "300m" in short_name:
                model_name = "esmc_300m"
            else:
                model_name = name

            self.encoder = ESMC.from_pretrained(model_name).to(device)
            self.tokenizer = EsmSequenceTokenizer()
            self.encoder.eval()
        except ImportError:
            raise ImportError(
                "ESM C requires the `esm` package. Install with: pip install esm"
            )

    def _load_esm2(self, name: str, device: str):
        """Load ESM2 model via HuggingFace transformers."""
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.encoder = AutoModel.from_pretrained(name).to(device)
        self.encoder.eval()

    def _load_esm3(self, name: str, device: str):
        """Load ESM3 model from the EvolutionaryScale `esm` SDK.

        ESM3 is a multi-track encoder (sequence + structure + function).
        We always feed the sequence track. The structure track is populated
        per-sample at encode time when config.structure_track=True and a PDB
        path is provided in the encode() `structures` kwarg.
        """
        try:
            from esm.models.esm3 import ESM3
            from esm.sdk.api import ESMProtein, LogitsConfig
        except ImportError:
            raise ImportError(
                "ESM3 requires the EvolutionaryScale `esm` package. "
                "Install with: pip install esm"
            )

        self.encoder = ESM3.from_pretrained(name).to(device)
        self.encoder.eval()
        self._ESMProtein = ESMProtein
        self._LogitsConfig = LogitsConfig
        # NOTE on `_structure_encoder` lazy init and --resume: ESM3's
        # `_structure_encoder` is lazily instantiated inside encode() the
        # first time `input.coordinates is not None`. For S1, this means
        # the saved DeepSpeed `module` dict has 34 SE keys but the saved
        # `frozen_param_fragments` does NOT (SE was registered as a Module
        # only after first encode()). Forcing instantiation here would make
        # DS's load_module_state_dict() KeyError trying to look those keys
        # up in `frozen_param_fragments`. Since SE is frozen and its weights
        # are identical to whatever from_pretrained produces, we just leave
        # it lazy: first encode() will re-instantiate it with the same
        # weights. The saved SE keys in `module` are silently skipped via
        # load_module_strict=False in train_robust.py:load_ckpt().

    # ── Encode ────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(
        self,
        sequences: List[str],
        device: str = "cuda",
        structures: Optional[List[Optional[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode protein sequences into per-residue embeddings.

        Args:
            sequences:  List of amino-acid strings.
            device:     Target device.
            structures: Only used by ESM3. List parallel to `sequences` with
                        either a PDB file path (use that protein's structure)
                        or None (mask the structure track for that sample).

        Returns:
            embeddings:     (B, max_seq_len, encoder_dim) per-residue, bf16
            attention_mask: (B, max_seq_len) 1 for real residues, 0 for padding
        """
        if not self._loaded:
            self.load(device)

        backend = self._backend()
        if backend == "esmc":
            return self._encode_esmc(sequences, device)
        if backend == "esm2":
            return self._encode_esm2(sequences, device)
        if backend == "esm3":
            return self._encode_esm3(sequences, device, structures=structures)
        raise ValueError(f"Unknown backend: {backend}")

    def _encode_esmc(self, sequences: List[str], device: str):
        """Encode with ESM C."""
        tokens = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            output = self.encoder(tokens["input_ids"])

        embeddings = output.embeddings if hasattr(output, 'embeddings') else output.last_hidden_state

        # ESM C adds <cls> at start and <eos> at end — strip both
        embeddings = embeddings[:, 1:-1, :]
        mask = tokens.get("attention_mask", torch.ones(embeddings.shape[:2], device=device))
        mask = mask[:, 1:-1]

        return embeddings.to(torch.bfloat16), mask

    def _encode_esm2(self, sequences: List[str], device: str):
        """Encode with ESM2."""
        tokens = self.tokenizer(
            sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            output = self.encoder(**tokens)

        embeddings = output.last_hidden_state
        embeddings = embeddings[:, 1:-1, :]
        mask = tokens["attention_mask"][:, 1:-1]

        return embeddings.to(torch.bfloat16), mask

    def _encode_esm3(
        self,
        sequences: List[str],
        device: str,
        structures: Optional[List[Optional[str]]] = None,
    ):
        """Encode with ESM3 (per-residue embeddings, optional structure track).

        We loop one protein at a time because ESM3's SDK builds an `ESMProtein`
        per sample (heterogeneous structure availability + variable length).
        Outputs are then padded into a (B, max_L, 1536) tensor matching the
        ESM C / ESM2 contract.
        """
        if self._ESMProtein is None or self._LogitsConfig is None:
            raise RuntimeError("ESM3 not loaded — call .load() first")

        # Force eval mode: the VQ-VAE structure-encoder codebook in esm 3.2.1
        # asserts `False, "Not implemented"` when self.training=True (codebook.py:81),
        # and DeepSpeed's engine.train() flips submodule training flags including
        # this frozen encoder. Re-assert eval here on every encode call.
        self.encoder.eval()

        use_struct = bool(self.config.structure_track)
        if structures is None:
            structures = [None] * len(sequences)
        if len(structures) != len(sequences):
            raise ValueError(
                f"structures has {len(structures)} entries, sequences has {len(sequences)}"
            )

        per_sample = []
        for seq, pdb_path in zip(sequences, structures):
            if use_struct and pdb_path:
                # from_pdb populates sequence + coordinates; override sequence in
                # case the PDB has gaps or differs from the canonical UniProt seq.
                protein = self._ESMProtein.from_pdb(pdb_path)
                protein.sequence = seq
            else:
                protein = self._ESMProtein(sequence=seq)

            with torch.no_grad():
                tensor = self.encoder.encode(protein)
                output = self.encoder.logits(
                    tensor, self._LogitsConfig(return_embeddings=True)
                )

            emb = output.embeddings  # [L+2, 1536] or [1, L+2, 1536]
            if emb.dim() == 2:
                emb = emb.unsqueeze(0)
            # Strip BOS/EOS to match the ESMC/ESM2 convention
            emb = emb[:, 1:-1, :]
            per_sample.append(emb)

        max_len = max(e.shape[1] for e in per_sample)
        B = len(per_sample)
        D = per_sample[0].shape[-1]
        padded = torch.zeros(B, max_len, D, device=device, dtype=torch.bfloat16)
        mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, e in enumerate(per_sample):
            L = e.shape[1]
            padded[i, :L, :] = e[0].to(device=device, dtype=torch.bfloat16)
            mask[i, :L] = 1
        return padded, mask

    @property
    def dim(self):
        return self.config.dim
