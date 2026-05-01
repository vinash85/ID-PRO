"""
Extract layer-48/64 hidden-state pool views for three IDPro ablations in one
pass per protein:

  A1  — per-residue input vs compressed (k=1 and k=32 tokens)
  A2  — RAG on vs off  (baseline vs empty RAG context)
  A3a — evidence-head logits (pre-LLM + post-LLM) for use as probe features

Per protein, we do up to 4 forward passes:
   (i)   baseline (full per-residue + RAG) with evidence-head logits captured
   (ii)  k=32 compressed protein tokens + RAG
   (iii) k=1  compressed protein tokens + RAG
   (iv)  full per-residue + NO-RAG

Outputs (to idpro/data/probe/embeddings/):
   ablation_{WHICH}_embeddings.pt  — dict: accession -> {
        view_a_prompteol_l48_base, view_b_question_mean_l48_base, view_c_eos_l64_base,
        view_a_prompteol_l48_k32,  view_b_question_mean_l48_k32,  view_c_eos_l64_k32,
        view_a_prompteol_l48_k1,   view_b_question_mean_l48_k1,   view_c_eos_l64_k1,
        view_a_prompteol_l48_norag, view_b_question_mean_l48_norag, view_c_eos_l64_norag,
        evidence_pre_mean_9d, evidence_post_mean_9d,
        labels, sequence_length
   }

Run (GPU 1 recommended):
    CUDA_VISIBLE_DEVICES=1 python scripts/extract_ablation_embeddings.py \
        --ckpt checkpoints/robust/stage4_step80000 --which reference
    CUDA_VISIBLE_DEVICES=1 python scripts/extract_ablation_embeddings.py \
        --ckpt checkpoints/robust/stage4_step80000 --which benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from idpro.eval.probes.extract_probe_embeddings import (  # noqa: E402
    build_model, load_checkpoint, esmc_pool_one, rag_retrieve,
    load_jsonl, DATA_DIR, OUT_DIR, MAX_SEQ_LEN, RAG_K, RAG_CTX_MAX_CHARS,
    PROMPTEOL_TEMPLATE, TAP_LAYER_MID, TAP_LAYER_LAST,
)
from idpro.config import IDProConfig  # noqa: E402
from idpro.model import IDProModel  # noqa: E402


# ---------------------------------------------------------------------------
# Build protein tokens with optional bucket-pool compression.
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_protein_compressed(
    model: IDProModel,
    sequence: str,
    device: str,
    k: int,
):
    """
    Replicates model.encode_protein but with a mean-pool compression step:
    the per-residue ESM C embeddings are bucketed into k equal-length groups,
    each group mean-pooled, BEFORE the adaptor+projector pipeline.

    k = 0 → no compression (equivalent to full per-residue path).
    k = 1 → single global mean-pool (one LLM token).
    k > 1 → k mean-pooled tokens (Perceiver-Resampler analog, no learned queries).
    """
    embeddings, mask = model.encoder.encode([sequence], device)
    # Dtype coherence
    target_dtype = next(model.adaptor.parameters()).dtype
    embeddings = embeddings.to(dtype=target_dtype)
    # (1, T, dim), (1, T)
    n = int(mask[0].sum().item())
    if k > 0 and k < n:
        # Bucket into k equal-length groups of the valid residues
        e = embeddings[0, :n]  # (n, dim)
        bucket_sizes = [n // k + (1 if i < (n % k) else 0) for i in range(k)]
        pooled = []
        offset = 0
        for bs in bucket_sizes:
            pooled.append(e[offset:offset + bs].mean(dim=0, keepdim=True))
            offset += bs
        emb_c = torch.cat(pooled, dim=0).unsqueeze(0)  # (1, k, dim)
        mask_c = torch.ones(1, k, dtype=mask.dtype, device=mask.device)
        embeddings, mask = emb_c, mask_c

    # Step 2: adaptor
    embeddings = model.adaptor(embeddings, mask)
    # Pre-LLM evidence head runs on the adaptor output; capture before projection
    pre_evidence_logits = model.evidence_head_pre(embeddings, mask)

    # Step 4: projector to LLM dim
    protein_tokens = model.projector(embeddings)
    protein_tokens = protein_tokens + model.protein_modality_embed
    protein_tokens = model.protein_position(protein_tokens, mask)
    return protein_tokens, mask, pre_evidence_logits


# ---------------------------------------------------------------------------
# Variant forward pass.
# ---------------------------------------------------------------------------


@torch.no_grad()
def forward_variant(
    model: IDProModel,
    sequence: str,
    question: str,
    rag_context: Optional[str],
    device: str,
    compress_k: int = 0,
    capture_evidence: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    One forward pass through LLM with the specified ablation knobs.
    Returns dict with three pool views, optionally plus evidence-head 9-d vectors.
    """
    # 1. Protein tokens (with optional compression)
    protein_tokens, protein_mask, pre_evidence = encode_protein_compressed(
        model, sequence, device, k=compress_k
    )
    # 2. Build the combined input
    full_question = question.strip() + PROMPTEOL_TEMPLATE
    inputs = model.build_inputs(
        protein_tokens,
        protein_mask,
        [full_question],
        rag_contexts=[rag_context] if rag_context else None,
        answers=None,
        device=device,
    )
    # Span offsets
    n_prot = int(protein_mask[0].sum().item())
    if rag_context:
        rag_text = f"{model.config.rag_start_token} {rag_context} {model.config.rag_end_token} "
        n_rag = len(model.tokenizer.encode(rag_text, add_special_tokens=False))
    else:
        n_rag = 0
    q_ids = model.tokenizer.encode(full_question, add_special_tokens=False)
    n_q = len(q_ids)
    protein_end = n_prot + 1
    q_start = protein_end + n_rag
    q_end = q_start + n_q
    prompteol_pos = q_end - 1

    outputs = model.llm(
        inputs_embeds=inputs["inputs_embeds"],
        attention_mask=inputs["attention_mask"],
        position_ids=inputs["position_ids"],
        output_hidden_states=True,
    )
    hidden = outputs.hidden_states
    h_mid = hidden[TAP_LAYER_MID][0]
    h_last = hidden[TAP_LAYER_LAST][0]

    n_prompteol = len(model.tokenizer.encode(PROMPTEOL_TEMPLATE, add_special_tokens=False))
    q_only_start = q_start
    q_only_end = q_end - n_prompteol

    view_a = h_mid[prompteol_pos].float().cpu()
    if q_only_end > q_only_start:
        view_b = h_mid[q_only_start:q_only_end].mean(dim=0).float().cpu()
    else:
        view_b = h_mid[q_only_start].float().cpu()
    view_c = h_last[prompteol_pos].float().cpu()

    out = {
        "view_a_prompteol_l48": view_a,
        "view_b_question_mean_l48": view_b,
        "view_c_eos_l64": view_c,
    }

    if capture_evidence:
        # Pre-LLM evidence: mean-pool across residues → 9-d
        # pre_evidence has shape (1, n_prot_or_k, num_labels)
        ev_pre = pre_evidence[0].mean(dim=0).float().cpu()  # (num_labels,)
        out["evidence_pre_mean_9d"] = ev_pre

        # Post-LLM evidence: run evidence_head_post on layer-48 protein-token positions
        # Protein tokens occupy positions [0, n_prot) in the combined sequence.
        # evidence_head_post expects (batch, seq_len, llm_dim) + mask.
        protein_hidden = h_mid[:n_prot].unsqueeze(0)  # (1, n_prot, 5120)
        protein_evidence_mask = torch.ones(1, n_prot, device=protein_hidden.device)
        post_logits = model.evidence_head_post(protein_hidden, protein_evidence_mask)
        # shape (1, n_prot, num_labels)
        ev_post = post_logits[0].mean(dim=0).float().cpu()
        out["evidence_post_mean_9d"] = ev_post

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--which", choices=["reference", "benchmark"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rag-k", type=int, default=RAG_K)
    ap.add_argument("--variants", default="k32,k1,norag,evidence",
                    help="comma list of variants to run beyond the baseline")
    ap.add_argument("--shard", default="0/1",
                    help="'i/N' — process only proteins whose index mod N == i. Use "
                         "to parallelize across multiple GPUs by launching N jobs with "
                         "i=0,...,N-1. Each writes to a separate _shard{i}.pt file.")
    args = ap.parse_args()

    try:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        assert 0 <= shard_i < shard_n
    except (ValueError, AssertionError):
        raise SystemExit(f"--shard must be 'i/N' with 0<=i<N, got {args.shard}")

    device = args.device
    print(f"Device: {device}")
    ref_rows = load_jsonl(DATA_DIR / "reference.jsonl")
    bench_rows = load_jsonl(DATA_DIR / "benchmark.jsonl")
    rows = ref_rows if args.which == "reference" else bench_rows
    if args.limit:
        rows = rows[:args.limit]
    # Shard: keep only proteins whose index mod shard_n == shard_i
    if shard_n > 1:
        rows = [r for i, r in enumerate(rows) if i % shard_n == shard_i]
        out_path = OUT_DIR / f"ablation_{args.which}_embeddings_shard{shard_i}of{shard_n}.pt"
        print(f"Shard {shard_i}/{shard_n}: {len(rows)} proteins")
    else:
        out_path = OUT_DIR / f"ablation_{args.which}_embeddings.pt"
    results: Dict[str, Dict] = {}
    if out_path.exists():
        try:
            prev = torch.load(out_path, map_location="cpu", weights_only=False)
            if isinstance(prev, dict):
                results = prev
                print(f"Resuming: {len(results)} already done")
        except Exception as e:
            print(f"Could not resume: {e}")

    wanted_variants = set(v.strip() for v in args.variants.split(","))
    todo = [r for r in rows if r["accession"] not in results]
    print(f"To process: {len(todo)} / {len(rows)};  variants = {sorted(wanted_variants)}")

    if not todo:
        print("Nothing to do.")
        return 0

    model = build_model(device)
    load_checkpoint(model, args.ckpt, device)

    # Load RAG index (restricted to reference set)
    rag_path = OUT_DIR / "rag_index.npz"
    npz = np.load(rag_path, allow_pickle=True)
    all_embs = npz["embs"]
    all_ids = npz["ids"].tolist()
    id_to_row = {r["accession"]: r for r in ref_rows}
    keep = [(i, aid) for i, aid in enumerate(all_ids) if aid in id_to_row]
    rag_embs = np.stack([all_embs[i] for i, _ in keep])
    rag_rows = [id_to_row[aid] for _, aid in keep]
    print(f"RAG index: {len(rag_rows)} proteins")

    question = "What is the function of this protein?"
    save_every = 50
    t0 = time.time()
    for i, row in enumerate(todo):
        acc = row["accession"]
        seq = row["sequence"][:MAX_SEQ_LEN]

        # Build RAG context (exclude self for reference rows)
        q_emb = esmc_pool_one(model, seq, device)
        ctx = rag_retrieve(
            q_emb, rag_embs, rag_rows, k=args.rag_k,
            exclude_self_acc=acc if args.which == "reference" else None,
        )

        rec = {
            "labels": row["labels"],
            "sequence_length": len(seq),
        }
        try:
            # Baseline (full per-residue + RAG) with evidence capture
            if "evidence" in wanted_variants:
                base = forward_variant(model, seq, question, ctx, device,
                                       compress_k=0, capture_evidence=True)
                rec["evidence_pre_mean_9d"] = base["evidence_pre_mean_9d"].to(torch.float16)
                rec["evidence_post_mean_9d"] = base["evidence_post_mean_9d"].to(torch.float16)

            if "k32" in wanted_variants:
                k32 = forward_variant(model, seq, question, ctx, device,
                                      compress_k=32, capture_evidence=False)
                for k, v in k32.items():
                    rec[f"{k}_k32"] = v.to(torch.float16)

            if "k1" in wanted_variants:
                k1 = forward_variant(model, seq, question, ctx, device,
                                     compress_k=1, capture_evidence=False)
                for k, v in k1.items():
                    rec[f"{k}_k1"] = v.to(torch.float16)

            if "norag" in wanted_variants:
                nr = forward_variant(model, seq, question, rag_context=None,
                                     device=device, compress_k=0, capture_evidence=False)
                for k, v in nr.items():
                    rec[f"{k}_norag"] = v.to(torch.float16)

        except Exception as e:
            print(f"  [error] {acc}: {e}")
            continue

        results[acc] = rec

        if (i + 1) % 5 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(todo) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(todo)}] rate={rate:.2f}/s  eta={eta/60:.1f}min", flush=True)

        if (i + 1) % save_every == 0:
            torch.save(results, out_path)

    torch.save(results, out_path)
    print(f"\nDone. {len(results)} proteins → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
