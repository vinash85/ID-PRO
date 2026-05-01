"""
Extract layer-48 (and layer-64) hidden-state pool views from frozen IDPro
for the downstream classifier probe experiment.

For each protein we record three pool views:
  A — PromptEOL colon position, layer 48    (multimodal-reasoning compressed)
  B — mean over question token span, layer 48  (question-conditioned)
  C — last-position hidden state, layer 64   (EOS, standard decoder pool)

Inputs built as:
  [Protein tokens] [PROT_END] [RAG top-k neighbors] [Question] "In one word, the function is:"

Outputs (to idpro/data/probe/embeddings/):
  reference_embeddings.pt  — dict: accession -> {view_a, view_b, view_c, labels}
  benchmark_embeddings.pt  — same schema for held-out test set
  rag_index.npz            — ESM C mean-pool embeddings of the reference set (for RAG)

Run:
    CUDA_VISIBLE_DEVICES=1 python scripts/extract_probe_embeddings.py \
        --ckpt checkpoints/robust/stage4_step80000 \
        --which reference
    CUDA_VISIBLE_DEVICES=1 python scripts/extract_probe_embeddings.py \
        --ckpt checkpoints/robust/stage4_step80000 \
        --which benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.config import IDProConfig  # noqa: E402
from idpro.model import IDProModel  # noqa: E402
from idpro.paths import AIM1_PROBE_DIR as DATA_DIR  # noqa: E402

OUT_DIR = DATA_DIR / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Layer taps. Qwen3.5-27B has 64 layers → hidden_states has 65 entries (embedding + 64).
# hidden_states[48] is the output of layer 48 (1-indexed), i.e. index 48 in the list.
TAP_LAYER_MID = 48
TAP_LAYER_LAST = 64  # == outputs.hidden_states[-1]

PROMPTEOL_TEMPLATE = " In one word, the enzymatic function of this protein is:"

MAX_SEQ_LEN = 1000
RAG_K = 5
RAG_CTX_MAX_CHARS = 1200  # cap RAG text to control LLM context length


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def build_model(
    device: str = "cuda",
    encoder_name: str = "esmc-600m",
    structure_track: bool = False,
    structure_manifest_path: str = "",
) -> IDProModel:
    config = IDProConfig()
    config.encoder.name = encoder_name
    config.encoder.resolve()
    if structure_track:
        if config.encoder.backend != "esm3":
            raise ValueError(
                f"--structure-track requires an ESM3 encoder; got backend={config.encoder.backend!r}"
            )
        config.encoder.structure_track = True
        config.encoder.structure_manifest_path = structure_manifest_path
    config.llm.name = "qwen3.5-27b"
    config.llm.resolve()

    print(f"Encoder: {config.encoder.name}  dim={config.encoder.dim}  backend={config.encoder.backend}  structure_track={config.encoder.structure_track}")
    print(f"LLM:     {config.llm.name}  dim={config.llm.dim}")

    model = IDProModel(config)
    # Load encoder first (2.5GB)
    model.encoder.load(device)

    # Load LLM in bf16 (54 GB, fits on H100 80GB). QLoRA is NOT used here because
    # we need access to full-precision hidden states and aren't training.
    print("Loading LLM in bf16 (may take ~1 min)...")
    config.llm.use_qlora = False
    model.load_llm(device=device, dtype=torch.bfloat16)

    # Move bridge components to device + dtype
    dtype = torch.bfloat16
    model.adaptor = model.adaptor.to(device=device, dtype=dtype)
    model.projector = model.projector.to(device=device, dtype=dtype)
    model.evidence_head_pre = model.evidence_head_pre.to(device=device, dtype=dtype)
    model.evidence_head_post = model.evidence_head_post.to(device=device, dtype=dtype)
    model.protein_position = model.protein_position.to(device=device, dtype=dtype)
    model.protein_modality_embed.data = model.protein_modality_embed.data.to(device=device, dtype=dtype)
    model.prot_end_embed.data = model.prot_end_embed.data.to(device=device, dtype=dtype)
    return model


def load_checkpoint(model: IDProModel, ckpt_path: Path, device: str = "cuda"):
    tp = ckpt_path / "trainable.pt"
    ds = ckpt_path / "mp_rank_00_model_states.pt"
    if tp.exists():
        print(f"Loading trainable weights from {tp} ...")
        state = torch.load(tp, map_location=device, weights_only=False)
    elif ds.exists():
        # DeepSpeed ZeRO-2 checkpoint — full state_dict lives under 'module'.
        # load_module_state_dict in train_robust strips bnb 4-bit aux keys; we
        # only need to copy keys that exist in the freshly built model.
        print(f"Loading DeepSpeed ZeRO-2 weights from {ds} ...")
        state = torch.load(ds, map_location="cpu", weights_only=False)["module"]
    else:
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path} (looked for trainable.pt and "
            f"mp_rank_00_model_states.pt)"
        )
    missing, loaded = [], 0
    model_state = dict(model.named_parameters())
    for name, tensor in state.items():
        if name in model_state:
            try:
                model_state[name].data.copy_(tensor.to(device=device, dtype=model_state[name].dtype))
                loaded += 1
            except Exception as e:
                print(f"  skip {name}: {e}")
        else:
            missing.append(name)
    print(f"  loaded {loaded}/{len(state)} params")
    if missing:
        print(f"  not in model ({len(missing)}): {missing[:5]}...")
    model.eval()


# ---------------------------------------------------------------------------
# RAG: in-memory index via ESM C mean-pool
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_rag_index(model: IDProModel, reference_rows: List[dict], device: str, batch_size: int = 8) -> Tuple[np.ndarray, List[dict]]:
    """Encode reference proteins with ESM C mean-pool; return (embeddings, rows)."""
    print(f"Building RAG index over {len(reference_rows)} proteins (batch {batch_size})...")
    embs = []
    for i in range(0, len(reference_rows), batch_size):
        batch = reference_rows[i : i + batch_size]
        seqs = [r["sequence"][:MAX_SEQ_LEN] for r in batch]
        emb, mask = model.encoder.encode(seqs, device)  # (B, T, dim), (B, T)
        for j in range(len(batch)):
            n = int(mask[j].sum().item())
            pooled = emb[j, :n].mean(dim=0).float().cpu().numpy()
            embs.append(pooled)
        if (i // batch_size) % 25 == 0:
            print(f"  rag-index: {min(i + batch_size, len(reference_rows))}/{len(reference_rows)}")
    embs = np.stack(embs).astype(np.float32)
    # L2-normalize for cosine similarity
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    print(f"  rag index shape: {embs.shape}")
    return embs, reference_rows


def rag_retrieve(query_embedding: np.ndarray, index_embs: np.ndarray, index_rows: List[dict], k: int, exclude_self_acc: Optional[str] = None) -> str:
    """Return concatenated RAG description of top-k neighbors."""
    q = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    sims = index_embs @ q  # (N,)
    # If self-retrieval, exclude self
    if exclude_self_acc is not None:
        for i, r in enumerate(index_rows):
            if r["accession"] == exclude_self_acc:
                sims[i] = -1.0
    top = np.argsort(-sims)[:k]
    parts = []
    for rank, idx in enumerate(top):
        desc = index_rows[int(idx)].get("description", "").strip()
        if not desc:
            continue
        parts.append(f"[{rank+1}] {desc}")
    ctx = "\n".join(parts)
    return ctx[:RAG_CTX_MAX_CHARS]


@torch.no_grad()
def esmc_pool_one(model: IDProModel, sequence: str, device: str) -> np.ndarray:
    emb, mask = model.encoder.encode([sequence[:MAX_SEQ_LEN]], device)
    n = int(mask[0].sum().item())
    return emb[0, :n].mean(dim=0).float().cpu().numpy()


# ---------------------------------------------------------------------------
# Forward with hidden states — returns three pool views
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_views(
    model: IDProModel,
    sequence: str,
    question: str,
    rag_context: str,
    device: str,
    pdb_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build the combined input exactly as build_inputs does, then run LLM forward
    with output_hidden_states=True and pool three views.

    Returns a dict with tensors of shape (llm_dim,) float32 (CPU).
    """
    # 1. Encode protein (pass structure when ESM3 structure track is enabled)
    structures = [pdb_path] if pdb_path else None
    protein_tokens, protein_mask, _ = model.encode_protein([sequence], device, structures=structures)
    # 2. Build inputs WITH PromptEOL appended to the question
    full_question = question.strip() + PROMPTEOL_TEMPLATE
    inputs = model.build_inputs(
        protein_tokens,
        protein_mask,
        [full_question],
        rag_contexts=[rag_context] if rag_context else None,
        answers=None,
        device=device,
    )
    # 3. Compute span indices (matches build_inputs layout):
    #    [protein_tokens (n_prot)] [PROT_END (1)] [rag_text] [question_with_prompteol]
    n_prot = int(protein_mask[0].sum().item())
    # Recompute rag token count by tokenizing the rag string used by build_inputs
    if rag_context:
        rag_text = f"{model.config.rag_start_token} {rag_context} {model.config.rag_end_token} "
        n_rag = len(model.tokenizer.encode(rag_text, add_special_tokens=False))
    else:
        n_rag = 0
    q_ids = model.tokenizer.encode(full_question, add_special_tokens=False)
    n_q = len(q_ids)
    # Layout offsets
    protein_end = n_prot + 1  # after PROT_END
    q_start = protein_end + n_rag
    q_end = q_start + n_q
    total_len = q_end
    # Sanity check
    mask = inputs["attention_mask"][0]
    assert int(mask.sum().item()) == total_len, (
        f"length mismatch: computed {total_len} vs mask_sum {int(mask.sum().item())}"
    )
    # Position of PromptEOL colon = last real token (because template ends with ":")
    prompteol_pos = q_end - 1

    # 4. Forward through LLM
    outputs = model.llm(
        inputs_embeds=inputs["inputs_embeds"],
        attention_mask=inputs["attention_mask"],
        position_ids=inputs["position_ids"],
        output_hidden_states=True,
    )
    hidden = outputs.hidden_states  # tuple, length = 65 for 64-layer LLM

    # 5. Pool the three views
    h_mid = hidden[TAP_LAYER_MID][0]  # (total_len, llm_dim)
    h_last = hidden[TAP_LAYER_LAST][0]

    # Question span covers the original question PLUS the PromptEOL template.
    # For View B we want the question portion only (exclude PromptEOL), since
    # the PromptEOL colon IS View A. n_prompteol tokens = len of template.
    n_prompteol = len(model.tokenizer.encode(PROMPTEOL_TEMPLATE, add_special_tokens=False))
    q_only_end = q_end - n_prompteol
    q_only_start = q_start

    view_a = h_mid[prompteol_pos].float().cpu()           # PromptEOL colon @ L48
    view_b = h_mid[q_only_start:q_only_end].mean(dim=0).float().cpu()  # mean question @ L48
    view_c = h_last[prompteol_pos].float().cpu()          # last token @ L64

    return {
        "view_a_prompteol_l48": view_a,
        "view_b_question_mean_l48": view_b,
        "view_c_eos_l64": view_c,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_jsonl(p: Path) -> List[dict]:
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--which", choices=["reference", "benchmark", "dark"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="debug: only process first N")
    ap.add_argument("--rag-k", type=int, default=RAG_K)
    ap.add_argument(
        "--encoder",
        default="esmc-600m",
        choices=["esmc-600m", "esmc-300m", "esm2-650m", "esm2-3b", "esm3-1.4b"],
        help="Encoder preset; must match what the checkpoint was trained with",
    )
    ap.add_argument(
        "--structure-track",
        action="store_true",
        help="ESM3 only: populate the structure track from --structure-manifest",
    )
    ap.add_argument(
        "--structure-manifest",
        default="",
        help="Path to JSONL manifest with {\"accession\", \"pdb_path\"} per line",
    )
    args = ap.parse_args()

    device = args.device
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")

    # Load structure manifest if structure-track is on
    structure_lookup: Dict[str, str] = {}
    if args.structure_track:
        if not args.structure_manifest:
            raise SystemExit("ERROR: --structure-track requires --structure-manifest <jsonl>")
        manifest_path = Path(args.structure_manifest)
        if not manifest_path.exists():
            raise SystemExit(f"ERROR: structure manifest not found: {manifest_path}")
        with manifest_path.open() as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                acc = rec.get("accession")
                pdb = rec.get("pdb_path")
                if acc and pdb:
                    structure_lookup[acc] = pdb
        print(f"Structure manifest: {len(structure_lookup):,} accessions → PDB paths")

    ref_rows = load_jsonl(DATA_DIR / "reference.jsonl")
    bench_rows = load_jsonl(DATA_DIR / "benchmark.jsonl")
    dark_path = DATA_DIR / "dark.jsonl"
    dark_rows = load_jsonl(dark_path) if dark_path.exists() else []
    print(f"Loaded {len(ref_rows)} reference, {len(bench_rows)} benchmark, {len(dark_rows)} dark")

    # Target rows
    if args.which == "reference":
        rows = ref_rows
    elif args.which == "benchmark":
        rows = bench_rows
    elif args.which == "dark":
        rows = dark_rows
    else:
        raise ValueError(args.which)
    if args.limit:
        rows = rows[: args.limit]
    out_path = OUT_DIR / f"{args.which}_embeddings.pt"

    # Resume support
    results: Dict[str, Dict] = {}
    if out_path.exists():
        try:
            prev = torch.load(out_path, map_location="cpu", weights_only=False)
            if isinstance(prev, dict):
                results = prev
                print(f"Resuming: {len(results)} already done")
        except Exception as e:
            print(f"Could not resume: {e}")

    # Skip rows already done
    todo = [r for r in rows if r["accession"] not in results]
    if not todo:
        print("All rows already have embeddings.")
        return 0
    print(f"To process: {len(todo)} / {len(rows)}")

    # Build model
    model = build_model(
        device,
        encoder_name=args.encoder,
        structure_track=args.structure_track,
        structure_manifest_path=args.structure_manifest,
    )
    load_checkpoint(model, args.ckpt, device)

    # Build / load RAG index
    rag_path = OUT_DIR / "rag_index.npz"
    if rag_path.exists():
        print(f"Loading RAG index from {rag_path}")
        npz = np.load(rag_path, allow_pickle=True)
        all_embs = npz["embs"]
        all_ids = npz["ids"].tolist()
        # Align to reference set only (index was built from reference; later augmented
        # with benchmark for ESM C baseline — filter back down).
        id_to_row = {r["accession"]: r for r in ref_rows}
        keep = [(i, aid) for i, aid in enumerate(all_ids) if aid in id_to_row]
        rag_embs = np.stack([all_embs[i] for i, _ in keep])
        rag_rows = [id_to_row[aid] for _, aid in keep]
        print(f"  RAG source: {len(rag_rows)} / {len(all_ids)} indexed proteins (filtered to reference set)")
    else:
        rag_embs, rag_rows = build_rag_index(model, ref_rows, device)
        np.savez(rag_path, embs=rag_embs, ids=np.array([r["accession"] for r in rag_rows]))
        print(f"Saved RAG index to {rag_path}")

    # Main loop
    question = "What is the function of this protein?"
    save_every = 50
    t0 = time.time()
    for i, row in enumerate(todo):
        acc = row["accession"]
        seq = row["sequence"][:MAX_SEQ_LEN]

        # Build RAG context (exclude self-retrieval for reference rows)
        q_emb = esmc_pool_one(model, seq, device)
        ctx = rag_retrieve(
            q_emb, rag_embs, rag_rows, k=args.rag_k,
            exclude_self_acc=acc if args.which == "reference" else None,
        )

        pdb_path = structure_lookup.get(acc) if args.structure_track else None
        try:
            views = extract_views(model, seq, question, ctx, device, pdb_path=pdb_path)
        except Exception as e:
            print(f"  [error] {acc}: {e}")
            continue

        results[acc] = {
            "labels": row["labels"],
            "sequence_length": len(seq),
            "view_a_prompteol_l48": views["view_a_prompteol_l48"].to(torch.float16),
            "view_b_question_mean_l48": views["view_b_question_mean_l48"].to(torch.float16),
            "view_c_eos_l64": views["view_c_eos_l64"].to(torch.float16),
        }

        if (i + 1) % 5 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(todo) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(todo)}] acc={acc} rate={rate:.2f}/s eta={eta/60:.1f}min")

        if (i + 1) % save_every == 0:
            torch.save(results, out_path)
            print(f"  [save] {len(results)} proteins → {out_path}")

    torch.save(results, out_path)
    print(f"\nDone. {len(results)} proteins → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
