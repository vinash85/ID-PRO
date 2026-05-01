"""
Dataset classes for IDPro training stages.
Handles both single-turn and multi-turn QA formats.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset


class IDProDataset(Dataset):
    """
    Dataset for IDPro training.
    Handles Stage 1-4 data formats: single-turn QA and multi-turn conversations.

    Each item returns:
        sequence: str — amino acid sequence
        question: str — the question to ask
        answer: str — the ground truth answer
        rag_context: str — optional RAG context
    """

    def __init__(
        self,
        data_path: str,
        stage: int = 1,
        rag_contexts: Optional[Dict[str, str]] = None,
        rag_fraction: float = 0.0,
        max_seq_len: int = 2048,
    ):
        self.stage = stage
        self.rag_contexts = rag_contexts or {}
        self.rag_fraction = rag_fraction
        self.max_seq_len = max_seq_len

        # Load data (supports JSON and JSONL)
        self.data = self._load(data_path)
        print(f"[Dataset] Loaded {len(self.data)} items from {data_path} (stage {stage})")

    def _load(self, path):
        path = Path(path)
        if path.suffix == ".jsonl":
            items = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
            return items
        else:
            with open(path) as f:
                return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        sequence = item.get("amino_seq") or item.get("sequence", "")
        protein_id = item.get("id") or item.get("uniprot_id", "")

        # Extract question and answer from conversations
        convs = item.get("conversations", [])

        if self.stage <= 3 and len(convs) > 2:
            # Multi-turn: return all turns
            turns = []
            for i in range(0, len(convs), 2):
                q = convs[i]["value"].replace("<protein_sequence>\n", "")
                a = convs[i + 1]["value"] if i + 1 < len(convs) else ""
                turns.append({"question": q, "answer": a})
            question = turns[0]["question"]
            answer = "\n\n".join(f"Turn {i+1}: {t['answer']}" for i, t in enumerate(turns))
        elif convs:
            question = convs[0]["value"].replace("<protein_sequence>\n", "")
            answer = convs[1]["value"] if len(convs) > 1 else ""
        else:
            question = item.get("question", "What is the function of this protein?")
            answer = item.get("answer", "")

        # RAG context (probabilistic)
        rag_context = ""
        if self.rag_fraction > 0 and random.random() < self.rag_fraction:
            rag_context = self.rag_contexts.get(protein_id, "")

        return {
            "sequence": sequence[:self.max_seq_len],
            "question": question,
            "answer": answer,
            "rag_context": rag_context,
            "protein_id": protein_id,
        }


class MultiStageDataset(Dataset):
    """
    Combines datasets from multiple stages with configurable mixing ratios.
    Used for anti-forgetting replay.
    """

    def __init__(self, datasets: List[Dataset], weights: Optional[List[float]] = None):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for d in datasets:
            total += len(d)
            self.cumulative_sizes.append(total)
        self.total = total

        # Weights for sampling (if provided)
        if weights:
            assert len(weights) == len(datasets)
            self.weights = weights
        else:
            self.weights = [len(d) / total for d in datasets]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                local_idx = idx - (self.cumulative_sizes[i - 1] if i > 0 else 0)
                return self.datasets[i][local_idx]
        return self.datasets[-1][idx - self.cumulative_sizes[-2]]
