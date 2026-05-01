"""
RAG (Retrieval-Augmented Generation) evidence layer.
Uses ESM embeddings (from the encoder) to find similar characterized proteins
and build context for the LLM.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class RAGIndex:
    """
    Precomputed embedding index for RAG retrieval.
    Stores ESM embeddings of characterized proteins with their function descriptions.
    Uses numpy for simplicity — replace with FAISS for production scale.
    """

    def __init__(self):
        self.ids: List[str] = []
        self.embeddings: np.ndarray = None  # (N, dim)
        self.descriptions: Dict[str, str] = {}
        self._normalized = False

    def build(self, proteins: List[Dict], encoder, device: str = "cuda", batch_size: int = 16):
        """
        Build the index from a list of characterized proteins.

        Args:
            proteins: List of dicts with 'id', 'sequence', 'description'
            encoder: ProteinEncoder instance
            device: GPU device
            batch_size: Batch size for encoding
        """
        import torch

        all_embeddings = []
        self.ids = []

        for i in range(0, len(proteins), batch_size):
            batch = proteins[i:i + batch_size]
            sequences = [p["sequence"][:1022] for p in batch]

            embeddings, masks = encoder.encode(sequences, device)
            # Mean pool per-residue embeddings → one vector per protein
            for j in range(len(batch)):
                n = masks[j].sum().int().item()
                pooled = embeddings[j, :n].mean(dim=0).cpu().numpy()
                all_embeddings.append(pooled)
                self.ids.append(batch[j]["id"])
                self.descriptions[batch[j]["id"]] = batch[j].get("description", "")

            if (i // batch_size) % 20 == 0:
                print(f"  RAG index: {min(i + batch_size, len(proteins))}/{len(proteins)}")

        self.embeddings = np.array(all_embeddings)
        self._normalize()
        print(f"  RAG index built: {len(self.ids)} proteins, dim={self.embeddings.shape[1]}")

    def _normalize(self):
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-10)
        self._normalized = True

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(str(path / "embeddings.npy"), self.embeddings)
        with open(path / "metadata.json", "w") as f:
            json.dump({"ids": self.ids, "descriptions": self.descriptions}, f)
        print(f"  RAG index saved: {path}")

    def load(self, path: str):
        path = Path(path)
        self.embeddings = np.load(str(path / "embeddings.npy"))
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        self.ids = meta["ids"]
        self.descriptions = meta["descriptions"]
        self._normalized = True
        print(f"  RAG index loaded: {len(self.ids)} proteins")

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        exclude_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve top-k most similar proteins.

        Args:
            query_embedding: (dim,) mean-pooled embedding of query protein
            k: Number of neighbors
            exclude_id: Exclude this protein ID (for leave-one-out)

        Returns:
            List of dicts with 'id', 'description', 'similarity'
        """
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-10)
        similarities = self.embeddings @ query_norm

        # Sort descending
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            pid = self.ids[idx]
            if pid == exclude_id:
                continue
            results.append({
                "id": pid,
                "description": self.descriptions.get(pid, ""),
                "similarity": float(similarities[idx]),
            })
            if len(results) >= k:
                break

        return results

    def build_context(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        exclude_id: Optional[str] = None,
        style: str = "expert",
    ) -> str:
        """
        Build RAG context string from retrieved neighbors.

        Args:
            query_embedding: Mean-pooled embedding of query protein
            k: Number of neighbors
            exclude_id: Protein ID to exclude
            style: Context formatting style

        Returns:
            Context string to inject into the prompt
        """
        neighbors = self.retrieve(query_embedding, k, exclude_id)

        if not neighbors:
            return ""

        parts = []
        for n in neighbors:
            sim = n["similarity"]
            desc = n["description"][:300]

            if style == "expert":
                parts.append(
                    f"A protein with {sim:.0%} structural similarity is known to be: {desc}"
                )
            elif style == "rich":
                parts.append(
                    f"Characterized protein (embedding similarity={sim:.2f}):\n  Function: {desc}"
                )
            elif style == "minimal":
                parts.append(desc)

        return "\n".join(parts)
