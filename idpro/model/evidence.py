"""
Evidence Span Head: per-residue functional role prediction.

Operates on LLM hidden states of protein tokens AFTER cross-modal processing.
This means the span head benefits from:
  - RAG context (retrieved similar proteins)
  - Question context (what the user is asking about)
  - LLM's parametric biomedical knowledge
  - Cross-attention between protein and text tokens

The span head predicts functional roles for each residue:
  domain_start, domain_interior, domain_end,
  active_site, binding_site, signal_peptide, transmembrane, none

Placed at an intermediate LLM layer (not the last) to capture
cross-modal reasoning without text-generation bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class EvidenceConfig:
    """Configuration for the evidence span head."""
    num_labels: int = 9
    hidden_dim: int = 512            # Internal hidden dim of the classifier
    dropout: float = 0.1
    loss_weight: float = 0.1         # Weight of evidence loss vs generation loss
    tap_layer: int = -16             # Which LLM layer to tap (-16 = 16 from end)
    # With Qwen3.5-27B (64 layers), -16 means layer 48

    LABEL_NAMES: List[str] = None

    def __post_init__(self):
        if self.LABEL_NAMES is None:
            self.LABEL_NAMES = [
                "none",              # 0: no functional role
                "domain_start",      # 1: first residue of a domain
                "domain_interior",   # 2: interior of a domain
                "domain_end",        # 3: last residue of a domain
                "active_site",       # 4: catalytic residue
                "binding_site",      # 5: substrate/cofactor binding
                "signal_peptide",    # 6: part of signal peptide
                "transmembrane",     # 7: transmembrane region
                "motif",             # 8: short functional motif
            ]
            self.num_labels = len(self.LABEL_NAMES)


class EvidenceSpanHead(nn.Module):
    """
    Per-residue functional role classifier.

    Input: LLM hidden states for protein tokens (after cross-modal processing)
    Output: per-residue logits over functional role labels

    Architecture:
      LLM hidden state (llm_dim) → LayerNorm → Linear → GELU → Dropout → Linear → logits (num_labels)

    The head is deliberately simple — the heavy lifting is done by the LLM.
    We just need a lightweight classifier on top of the LLM's enriched representations.
    """

    def __init__(self, llm_dim: int, config: EvidenceConfig):
        super().__init__()
        self.config = config
        self.llm_dim = llm_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_labels),
        )

        # Class weights for imbalanced labels (most residues are "none")
        # Upweight rare but important classes — float32 for loss computation
        self.register_buffer(
            "class_weights",
            torch.tensor([
                0.1,   # none — very common, downweight
                2.0,   # domain_start — rare, important
                0.5,   # domain_interior — common within domains
                2.0,   # domain_end — rare, important
                5.0,   # active_site — very rare, very important
                3.0,   # binding_site — rare, important
                2.0,   # signal_peptide — moderately rare
                2.0,   # transmembrane — moderately rare
                3.0,   # motif — rare
            ]),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify each protein residue's functional role.

        Args:
            hidden_states: (B, total_seq_len, llm_dim) — LLM hidden states
            protein_mask: (B, total_seq_len) — 1 for protein tokens, 0 for text/padding

        Returns:
            logits: (B, total_seq_len, num_labels) — per-position logits
                    (only protein positions are meaningful; text positions are masked)
        """
        logits = self.classifier(hidden_states)

        # Zero out non-protein positions (text tokens shouldn't have evidence labels)
        logits = logits * protein_mask.unsqueeze(-1)

        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss over protein residues only.

        Args:
            logits: (B, seq_len, num_labels)
            labels: (B, seq_len) — ground truth label indices (0-8)
            protein_mask: (B, seq_len) — 1 for protein tokens

        Returns:
            scalar loss
        """
        # Flatten
        logits_flat = logits.view(-1, self.config.num_labels)
        labels_flat = labels.view(-1)
        mask_flat = protein_mask.view(-1).bool()

        # Only compute loss on protein positions
        if mask_flat.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        logits_masked = logits_flat[mask_flat]
        labels_masked = labels_flat[mask_flat]

        loss = F.cross_entropy(
            logits_masked,
            labels_masked,
            weight=self.class_weights.to(logits.device),
            ignore_index=-100,
        )

        return loss * self.config.loss_weight

    def predict(
        self,
        logits: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> List[Dict]:
        """
        Convert logits to predicted evidence spans.

        Returns list (per sample) of dicts with predicted features:
        [
            {"type": "domain", "start": 45, "end": 320, "confidence": 0.92},
            {"type": "active_site", "position": 170, "confidence": 0.98},
            ...
        ]
        """
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)  # (B, seq_len)

        batch_results = []

        for i in range(preds.shape[0]):
            n_prot = protein_mask[i].sum().int().item()
            sample_preds = preds[i, :n_prot].detach().cpu().numpy()
            sample_probs = probs[i, :n_prot].detach().cpu().numpy()

            features = []
            current_domain = None

            for pos in range(n_prot):
                label = int(sample_preds[pos])
                conf = float(sample_probs[pos, label])
                label_name = self.config.LABEL_NAMES[label]

                if label_name == "domain_start":
                    current_domain = {"type": "domain", "start": pos + 1, "confidence": conf}  # 1-indexed
                elif label_name == "domain_end" and current_domain:
                    current_domain["end"] = pos + 1
                    features.append(current_domain)
                    current_domain = None
                elif label_name == "domain_interior" and current_domain:
                    current_domain["confidence"] = min(current_domain["confidence"], conf)
                elif label_name in ("active_site", "binding_site", "motif"):
                    features.append({
                        "type": label_name,
                        "position": pos + 1,  # 1-indexed
                        "confidence": conf,
                    })
                elif label_name == "signal_peptide":
                    # Aggregate consecutive signal peptide residues
                    if features and features[-1].get("type") == "signal_peptide":
                        features[-1]["end"] = pos + 1
                        features[-1]["confidence"] = min(features[-1]["confidence"], conf)
                    else:
                        features.append({
                            "type": "signal_peptide",
                            "start": pos + 1,
                            "end": pos + 1,
                            "confidence": conf,
                        })
                elif label_name == "transmembrane":
                    if features and features[-1].get("type") == "transmembrane":
                        features[-1]["end"] = pos + 1
                    else:
                        features.append({
                            "type": "transmembrane",
                            "start": pos + 1,
                            "end": pos + 1,
                            "confidence": conf,
                        })

            # Close any unclosed domain
            if current_domain:
                current_domain["end"] = n_prot
                features.append(current_domain)

            batch_results.append(features)

        return batch_results

    def format_evidence_text(self, features: List[Dict]) -> str:
        """
        Convert predicted features to a text string matching the IDENTIFY format.
        This can be compared against the model's text-based IDENTIFY output
        for consistency checking.
        """
        if not features:
            return "No significant functional features detected."

        parts = []
        for f in features:
            ftype = f["type"]
            conf = f.get("confidence", 0)

            if ftype == "domain":
                parts.append(
                    f"Predicted domain at residues {f['start']}-{f['end']} "
                    f"(confidence: {conf:.0%})"
                )
            elif ftype in ("active_site", "binding_site", "motif"):
                parts.append(
                    f"Predicted {ftype.replace('_', ' ')} at residue {f['position']} "
                    f"(confidence: {conf:.0%})"
                )
            elif ftype in ("signal_peptide", "transmembrane"):
                parts.append(
                    f"Predicted {ftype.replace('_', ' ')} at residues {f['start']}-{f['end']} "
                    f"(confidence: {conf:.0%})"
                )

        return "Evidence from span head:\n" + "\n".join(f"  - {p}" for p in parts)


def create_evidence_labels(
    protein_length: int,
    features: List[Dict],
) -> torch.Tensor:
    """
    Create per-residue evidence labels from structured annotation records.

    Args:
        protein_length: number of residues
        features: list of feature dicts from the annotation record
            Each has: type, start/end or position

    Returns:
        labels: (protein_length,) tensor of label indices
    """
    labels = torch.zeros(protein_length, dtype=torch.long)  # 0 = none

    label_map = {
        "domain": {"start": 1, "interior": 2, "end": 3},
        "active_site": 4,
        "binding_site": 5,
        "signal_peptide": 6,
        "transmembrane": 7,
        "motif": 8,
    }

    for feat in features:
        ftype = feat.get("type", "")
        start = feat.get("start")
        end = feat.get("end")
        position = feat.get("position")

        if ftype in ("domain", "repeat", "zinc_finger_region"):
            if start and end and start <= protein_length and end <= protein_length:
                labels[start - 1] = 1  # domain_start (1-indexed → 0-indexed)
                labels[end - 1] = 3    # domain_end
                if end - start > 1:
                    labels[start:end - 1] = 2  # domain_interior

        elif ftype in ("active_site",):
            pos = position or start
            if pos and pos <= protein_length:
                labels[pos - 1] = 4

        elif ftype in ("binding_site",):
            pos = position or start
            if pos and pos <= protein_length:
                labels[pos - 1] = 5

        elif ftype in ("signal_peptide",):
            if start and end:
                s, e = max(start - 1, 0), min(end, protein_length)
                labels[s:e] = 6

        elif ftype in ("transmembrane_region", "transmembrane"):
            if start and end:
                s, e = max(start - 1, 0), min(end, protein_length)
                labels[s:e] = 7

        elif ftype in ("short_sequence_motif", "motif", "dna_binding", "nucleotide_binding"):
            if position:
                if position <= protein_length:
                    labels[position - 1] = 8
            elif start and end:
                s, e = max(start - 1, 0), min(end, protein_length)
                labels[s:e] = 8

    return labels
