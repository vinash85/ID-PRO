"""
IDPro: Full multimodal protein→text model.

Architecture:
  Protein Sequence → Encoder (frozen) → Adaptor (1D Conv) → Projector (MLP) → LLM (LoRA)
  RAG context is tokenized as text and concatenated between protein tokens and the question.

The model handles:
  1. Encoding protein sequences into per-residue embeddings
  2. Adapting and projecting residue tokens into LLM space
  3. Building the combined input: [protein_tokens | rag_tokens | question_tokens]
  4. Generating chain-of-thought answers via the LLM
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any

from ..config import IDProConfig
from .encoder import ProteinEncoder
from .adaptor import ResidueAdaptor
from .projector import ResidueProjector
from .position import MultimodalPositionManager, ProteinPositionEncoding
from .evidence import EvidenceSpanHead, EvidenceConfig


class IDProModel(nn.Module):
    """
    Full IDPro multimodal model.

    Forward flow:
      1. Protein sequences → encoder → per-residue embeddings (B, T, encoder_dim)
      2. Embeddings → adaptor → local-context-enriched embeddings (B, T, encoder_dim)
      3. Embeddings → projector → LLM-space tokens (B, T, llm_dim)
      4. Concatenate: [prot_tokens | prot_end_embed | rag_text_embeds | question_embeds]
      5. Forward through LLM → generate answer

    The LLM sees protein tokens as if they were a different "language" — each residue
    is one token, and the LLM's attention can attend to specific residues.
    """

    def __init__(self, config: IDProConfig):
        super().__init__()
        self.config = config

        # Protein encoder (always frozen, lazy-loaded)
        self.encoder = ProteinEncoder(config.encoder)

        # Residue adaptor (token-preserving local context)
        self.adaptor = ResidueAdaptor(
            dim=config.encoder.dim,
            config=config.adaptor,
        )

        # Per-residue projector (encoder_dim → llm_dim)
        self.projector = ResidueProjector(
            encoder_dim=config.encoder.dim,
            llm_dim=config.llm.dim,
            config=config.projector,
        )

        # LLM backbone and tokenizer (loaded separately)
        self.llm = None
        self.tokenizer = None

        # Modality embedding: learned embedding to distinguish protein tokens from text
        self.protein_modality_embed = nn.Parameter(
            torch.zeros(1, 1, config.llm.dim)
        )

        # Special token embeddings
        self.prot_end_embed = nn.Parameter(
            torch.randn(1, 1, config.llm.dim) * 0.02
        )

        # Protein-specific position encoding (adds residue position info before LLM)
        self.protein_position = ProteinPositionEncoding(
            llm_dim=config.llm.dim,
            max_len=8192,
        )

        # Position handling strategy
        self.position_strategy = "reset"  # "reset", "continuous", "scaled"

        # Evidence span heads — multi-level evidence prediction
        self.evidence_config = EvidenceConfig()

        # Option A: Pre-LLM evidence head (on adaptor output, before LLM)
        # Sees only sequence-derived features — like a neural InterPro
        self.evidence_head_pre = EvidenceSpanHead(
            llm_dim=config.encoder.dim,  # operates on encoder dim, before projection
            config=EvidenceConfig(loss_weight=0.05),
        )

        # Option B: Post-LLM evidence head (on intermediate LLM layer)
        # Sees cross-modal reasoning: protein + RAG + question context
        self.evidence_head_post = EvidenceSpanHead(
            llm_dim=config.llm.dim,  # operates on LLM hidden dim
            config=self.evidence_config,
        )

    def load_llm(self, device: str = "cuda", dtype=torch.bfloat16):
        """Load the LLM backbone with optional LoRA."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[LLM] Loading {self.config.llm.name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm.name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                self.config.prot_start_token,
                self.config.prot_end_token,
                self.config.rag_start_token,
                self.config.rag_end_token,
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

        # Load model
        # Support multi-GPU: if device is "auto" or a dict, use it directly
        if device == "auto" or isinstance(device, dict):
            dm = device
        else:
            dm = {"": device}

        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": dm,
            "trust_remote_code": True,
        }

        if self.config.llm.use_qlora:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm.name, **load_kwargs
        )

        # Resize token embeddings for special tokens
        self.llm.resize_token_embeddings(len(self.tokenizer))

        print(f"[LLM] Loaded. Vocab size: {len(self.tokenizer)}")

    def apply_lora(self):
        """Apply LoRA adapters to the LLM."""
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=self.config.llm.lora_r,
            lora_alpha=self.config.llm.lora_alpha,
            target_modules=self.config.llm.lora_target_modules,
            lora_dropout=self.config.llm.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(self.llm, lora_config)

        trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.llm.parameters())
        print(f"[LoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    def encode_protein(
        self,
        sequences: List[str],
        device: str = "cuda",
        structures: Optional[List[Optional[str]]] = None,
    ):
        """
        Encode protein sequences into LLM-ready tokens.

        Args:
            sequences:  List of amino-acid strings.
            device:     Target device.
            structures: Optional. ESM3 only — list of PDB paths (or None) parallel
                        to `sequences`. Ignored by ESMC/ESM2 backends.

        Returns:
            protein_tokens: (B, T, llm_dim) — per-residue tokens for LLM
            protein_mask: (B, T) — attention mask
        """
        # Step 1: ESM encoding (frozen)
        embeddings, mask = self.encoder.encode(sequences, device, structures=structures)

        # Ensure consistent dtype (bf16) throughout the bridge
        target_dtype = next(self.adaptor.parameters()).dtype
        embeddings = embeddings.to(dtype=target_dtype)

        # Step 2: Residue adaptor (local context)
        embeddings = self.adaptor(embeddings, mask)

        # Step 3: Pre-LLM evidence head (Option A)
        # Runs on adaptor output BEFORE projection — sees only sequence features
        pre_evidence_logits = self.evidence_head_pre(embeddings, mask)

        # Step 4: Project to LLM dim
        protein_tokens = self.projector(embeddings)

        # Step 6: Add modality embedding (distinguishes protein from text tokens)
        protein_tokens = protein_tokens + self.protein_modality_embed

        # Step 7: Add protein-specific position encoding
        protein_tokens = self.protein_position(protein_tokens, mask)

        return protein_tokens, mask, pre_evidence_logits

    def build_inputs(
        self,
        protein_tokens: torch.Tensor,
        protein_mask: torch.Tensor,
        questions: List[str],
        rag_contexts: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        device: str = "cuda",
    ) -> Dict[str, torch.Tensor]:
        """
        Build combined input for the LLM.

        Layout: [protein_tokens] [PROT_END] [rag_context] [question] [answer]

        For training: answer tokens have labels, everything else is masked (-100).
        For inference: no answer tokens, generate from the question.
        """
        batch_size = protein_tokens.shape[0]
        llm_dim = self.config.llm.dim

        # Get LLM's embedding layer for text tokens
        embed_fn = self.llm.get_input_embeddings()
        if hasattr(embed_fn, 'weight'):
            embed_fn_call = embed_fn
        else:
            embed_fn_call = embed_fn

        all_input_embeds = []
        all_attention_masks = []
        all_labels = []

        for i in range(batch_size):
            parts_embeds = []
            parts_labels = []

            # 1. Protein tokens
            n_prot = protein_mask[i].sum().int().item()
            prot = protein_tokens[i, :n_prot]  # (n_prot, llm_dim)
            parts_embeds.append(prot)
            parts_labels.append(torch.full((n_prot,), -100, device=device))

            # 2. PROT_END token
            parts_embeds.append(self.prot_end_embed.squeeze(0))  # (1, llm_dim)
            parts_labels.append(torch.full((1,), -100, device=device))

            # 3. RAG context (if provided)
            if rag_contexts and rag_contexts[i]:
                rag_text = f"{self.config.rag_start_token} {rag_contexts[i]} {self.config.rag_end_token} "
                rag_ids = self.tokenizer.encode(rag_text, add_special_tokens=False, return_tensors="pt")[0].to(device)
                rag_embeds = embed_fn_call(rag_ids)  # (n_rag, llm_dim)
                parts_embeds.append(rag_embeds)
                parts_labels.append(torch.full((len(rag_ids),), -100, device=device))

            # 4. Question
            q_ids = self.tokenizer.encode(questions[i], add_special_tokens=False, return_tensors="pt")[0].to(device)
            q_embeds = embed_fn_call(q_ids)
            parts_embeds.append(q_embeds)
            parts_labels.append(torch.full((len(q_ids),), -100, device=device))

            # 5. Answer (for training only)
            if answers and answers[i]:
                a_ids = self.tokenizer.encode(
                    answers[i] + self.tokenizer.eos_token,
                    add_special_tokens=False,
                    return_tensors="pt"
                )[0].to(device)
                a_embeds = embed_fn_call(a_ids)
                parts_embeds.append(a_embeds)
                parts_labels.append(a_ids)  # Train on answer tokens

            # Concatenate all parts
            combined_embeds = torch.cat(parts_embeds, dim=0)  # (total_len, llm_dim)
            combined_labels = torch.cat(parts_labels, dim=0)  # (total_len,)

            all_input_embeds.append(combined_embeds)
            all_labels.append(combined_labels)

        # Pad to same length
        max_len = max(e.shape[0] for e in all_input_embeds)

        padded_embeds = torch.zeros(batch_size, max_len, llm_dim, device=device, dtype=protein_tokens.dtype)
        padded_masks = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
        padded_labels = torch.full((batch_size, max_len), -100, device=device, dtype=torch.long)

        for i in range(batch_size):
            seq_len = all_input_embeds[i].shape[0]
            padded_embeds[i, :seq_len] = all_input_embeds[i]
            padded_masks[i, :seq_len] = 1
            padded_labels[i, :seq_len] = all_labels[i]

        # Build position IDs based on strategy
        protein_lengths = torch.tensor(
            [protein_mask[i].sum().int().item() + 1  # +1 for PROT_END
             for i in range(batch_size)],
            device=device
        )
        text_lengths = torch.tensor(
            [padded_masks[i].sum().item() - protein_lengths[i].item()
             for i in range(batch_size)],
            device=device
        )

        if self.position_strategy == "reset":
            position_ids = MultimodalPositionManager.create_position_ids(
                protein_lengths, text_lengths, max_len, device
            )
        elif self.position_strategy == "scaled":
            position_ids = MultimodalPositionManager.create_position_ids_interleaved(
                protein_lengths, text_lengths, max_len,
                protein_position_scale=0.1, device=device
            )
        else:  # continuous
            position_ids = MultimodalPositionManager.create_position_ids_continuous(
                protein_lengths, max_len, device
            )

        return {
            "inputs_embeds": padded_embeds,
            "attention_mask": padded_masks,
            "position_ids": position_ids,
            "labels": padded_labels if answers else None,
        }

    def forward(
        self,
        sequences: List[str],
        questions: List[str],
        rag_contexts: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        device: str = "cuda",
        structures: Optional[List[Optional[str]]] = None,
    ):
        """
        Full forward pass: encode protein → build combined input → LLM forward.

        For training: returns loss.
        For inference: returns logits.

        `structures`: ESM3 only — list of PDB paths parallel to `sequences`,
        or None to disable. Ignored when the encoder backend is ESM C / ESM2.
        """
        # Encode protein sequences (includes pre-LLM evidence)
        protein_tokens, protein_mask, pre_evidence_logits = self.encode_protein(
            sequences, device, structures=structures
        )

        # Build combined inputs
        inputs = self.build_inputs(
            protein_tokens, protein_mask,
            questions, rag_contexts, answers, device
        )

        # Forward through LLM with custom position IDs
        # Request hidden states for post-LLM evidence head
        outputs = self.llm(
            inputs_embeds=inputs["inputs_embeds"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
            output_hidden_states=True,
        )

        # Build protein mask for the combined sequence
        protein_evidence_mask = torch.zeros_like(inputs["attention_mask"])
        for i in range(len(sequences)):
            n_prot = protein_mask[i].sum().int().item()
            protein_evidence_mask[i, :n_prot] = 1

        # Post-LLM evidence head (Option B): intermediate LLM layer
        if outputs.hidden_states is not None:
            tap_idx = self.evidence_config.tap_layer  # -16 → layer 48 of 64
            intermediate_hidden = outputs.hidden_states[tap_idx]
            post_evidence_logits = self.evidence_head_post(intermediate_hidden, protein_evidence_mask)
        else:
            post_evidence_logits = None

        # Attach evidence outputs for loss computation and analysis
        outputs.pre_evidence_logits = pre_evidence_logits    # Option A: sequence-only
        outputs.post_evidence_logits = post_evidence_logits  # Option B: reasoning-enriched
        outputs.protein_mask = protein_mask
        outputs.protein_evidence_mask = protein_evidence_mask

        return outputs

    @torch.no_grad()
    def generate(
        self,
        sequences: List[str],
        questions: List[str],
        rag_contexts: Optional[List[str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        device: str = "cuda",
        structures: Optional[List[Optional[str]]] = None,
    ) -> List[str]:
        """
        Generate answers for protein function prediction.

        `structures`: ESM3 only — list of PDB paths parallel to `sequences`,
        or None to disable. Ignored when the encoder backend is ESM C / ESM2.

        Returns: List of generated text strings.
        """
        # Encode protein sequences
        protein_tokens, protein_mask, pre_evidence = self.encode_protein(
            sequences, device, structures=structures
        )

        # Build inputs (no answers)
        inputs = self.build_inputs(
            protein_tokens, protein_mask,
            questions, rag_contexts, answers=None, device=device
        )

        # Generate
        gen_kwargs = {
            "inputs_embeds": inputs["inputs_embeds"],
            "attention_mask": inputs["attention_mask"],
            "position_ids": inputs["position_ids"],
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "top_p": 0.9 if temperature > 0 else None,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        output_ids = self.llm.generate(**gen_kwargs)

        # Decode generated tokens
        # Note: with inputs_embeds, generate() may return only new tokens
        # (no input prefix) or input+output depending on the model.
        # Handle both cases by checking output length vs input length.
        generated = []
        input_len = inputs["attention_mask"][0].sum().item()
        for i in range(len(sequences)):
            if output_ids.shape[1] > input_len:
                # Output includes input tokens — skip them
                gen_tokens = output_ids[i, input_len:]
            else:
                # Output is only generated tokens — use all
                gen_tokens = output_ids[i]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            generated.append(text)

        return generated

    def set_training_stage(self, stage: int):
        """Configure trainable parameters based on training stage."""
        config = self.config.training
        config.stage = stage
        config.apply_stage_preset()

        # Encoder is always frozen
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Adaptor
        for param in self.adaptor.parameters():
            param.requires_grad = config.train_adaptor

        # Projector
        for param in self.projector.parameters():
            param.requires_grad = config.train_projector

        # Modality embedding and special tokens — always trainable
        self.protein_modality_embed.requires_grad = True
        self.prot_end_embed.requires_grad = True

        # LLM LoRA
        if config.train_llm_lora and not hasattr(self.llm, 'peft_config'):
            self.apply_lora()

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Stage {stage}] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    def save_checkpoint(self, path: str):
        """Save trainable parameters only (adaptor, projector, LoRA, special tokens)."""
        import os
        os.makedirs(path, exist_ok=True)

        # Save adaptor + projector + special tokens
        bridge_state = {
            "adaptor": self.adaptor.state_dict(),
            "projector": self.projector.state_dict(),
            "protein_modality_embed": self.protein_modality_embed.data,
            "prot_end_embed": self.prot_end_embed.data,
        }
        torch.save(bridge_state, os.path.join(path, "bridge.pt"))

        # Save LoRA if applied
        if hasattr(self.llm, 'save_pretrained'):
            self.llm.save_pretrained(os.path.join(path, "llm_lora"))

        # Save config
        import json
        from dataclasses import asdict
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        print(f"[Checkpoint] Saved to {path}")

    def load_checkpoint(self, path: str, device: str = "cuda"):
        """Load trainable parameters from checkpoint."""
        import os

        # Load bridge
        bridge_path = os.path.join(path, "bridge.pt")
        if os.path.exists(bridge_path):
            bridge_state = torch.load(bridge_path, map_location=device)
            self.adaptor.load_state_dict(bridge_state["adaptor"])
            self.projector.load_state_dict(bridge_state["projector"])
            self.protein_modality_embed.data = bridge_state["protein_modality_embed"]
            self.prot_end_embed.data = bridge_state["prot_end_embed"]

        # Load LoRA
        lora_path = os.path.join(path, "llm_lora")
        if os.path.exists(lora_path) and self.llm is not None:
            from peft import PeftModel
            self.llm = PeftModel.from_pretrained(self.llm, lora_path)

        print(f"[Checkpoint] Loaded from {path}")
