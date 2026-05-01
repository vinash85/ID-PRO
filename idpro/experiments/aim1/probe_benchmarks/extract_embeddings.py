"""
Build the embedding caches consumed by `run_probe.py` and `run_baselines.py`.

Three artifact kinds, all emitted under `EXTRACTED_EMBEDDINGS_DIR`
(= datasets/probe_data/extracted_embeddings/):

  {split}_embeddings.pt   — accession → {view_a_prompteol_l48, view_b_question_mean_l48,
                                          view_c_eos_l64, labels, sequence_length}
                            for split ∈ {reference, benchmark, dark}. These are the
                            three pool views off the frozen IDPro stack:
                              A — PromptEOL colon position, layer 48
                              B — mean over question token span, layer 48
                              C — last-position hidden state, layer 64
  rag_index.npz           — ESM C mean-pool vectors (L2-normalized) for the reference
                            (and optionally benchmark) proteins. Used as the RAG index
                            during view extraction AND as the ESM-C-only baseline view
                            in the probe sweep.
  uniprot_metadata_cache.jsonl  — written by run_baselines / conformal scripts; not here.

Inputs read from `PROBE_SPLITS_DIR` (= datasets/probe_data/probe_splits/):
  reference.jsonl, benchmark.jsonl, dark.jsonl, labels.json   (built by data_prep/)

CLI
---
# Build (or extend) the ESM C RAG index over reference + benchmark first:
python idpro/experiments/aim1/probe_benchmarks/extract_embeddings.py rag-index \
    --include benchmark

# Then the three IDPro view caches:
python idpro/experiments/aim1/probe_benchmarks/extract_embeddings.py views \
    --which reference --ckpt $IDPRO_RUNS_ROOT/checkpoints/stage4_step80000
python idpro/experiments/aim1/probe_benchmarks/extract_embeddings.py views \
    --which benchmark --ckpt $IDPRO_RUNS_ROOT/checkpoints/stage4_step80000
python idpro/experiments/aim1/probe_benchmarks/extract_embeddings.py views \
    --which dark      --ckpt $IDPRO_RUNS_ROOT/checkpoints/stage4_step80000

For the ESM3 structure ablation arms, add:
    --encoder esm3-1.4b --structure-track --structure-manifest <path.jsonl>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.config import IDProConfig  # noqa: E402
from idpro.model import IDProModel  # noqa: E402
from idpro.model.p2t.encoder import ProteinEncoder  # noqa: E402
from idpro.paths import EXTRACTED_EMBEDDINGS_DIR, PROBE_SPLITS_DIR  # noqa: E402

EXTRACTED_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Layer taps. Qwen3.5-27B has 64 layers → hidden_states has 65 entries (embedding + 64).
TAP_LAYER_MID = 48
TAP_LAYER_LAST = 64

PROMPTEOL_TEMPLATE = " In one word, the enzymatic function of this protein is:"
QUESTION = "What is the function of this protein?"

MAX_SEQ_LEN = 1000
RAG_K_DEFAULT = 5
RAG_CTX_MAX_CHARS = 1200


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _load_jsonl(p: Path) -> List[dict]:
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _split_path(which: str) -> Path:
    return PROBE_SPLITS_DIR / f"{which}.jsonl"


def _emb_path(which: str) -> Path:
    return EXTRACTED_EMBEDDINGS_DIR / f"{which}_embeddings.pt"


RAG_INDEX_PATH = EXTRACTED_EMBEDDINGS_DIR / "rag_index.npz"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _build_model(
    *,
    device: str,
    encoder_name: str,
    structure_track: bool,
    structure_manifest_path: str,
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
    model.encoder.load(device)

    print("Loading LLM in bf16 (~1 min)...")
    config.llm.use_qlora = False
    model.load_llm(device=device, dtype=torch.bfloat16)

    dtype = torch.bfloat16
    model.adaptor = model.adaptor.to(device=device, dtype=dtype)
    model.projector = model.projector.to(device=device, dtype=dtype)
    model.evidence_head_pre = model.evidence_head_pre.to(device=device, dtype=dtype)
    model.evidence_head_post = model.evidence_head_post.to(device=device, dtype=dtype)
    model.protein_position = model.protein_position.to(device=device, dtype=dtype)
    model.protein_modality_embed.data = model.protein_modality_embed.data.to(device=device, dtype=dtype)
    model.prot_end_embed.data = model.prot_end_embed.data.to(device=device, dtype=dtype)
    return model


def _load_checkpoint(model: IDProModel, ckpt_path: Path, device: str) -> None:
    tp = ckpt_path / "trainable.pt"
    ds = ckpt_path / "mp_rank_00_model_states.pt"
    if tp.exists():
        print(f"Loading trainable weights from {tp}")
        state = torch.load(tp, map_location=device, weights_only=False)
    elif ds.exists():
        print(f"Loading DeepSpeed ZeRO-2 weights from {ds}")
        state = torch.load(ds, map_location="cpu", weights_only=False)["module"]
    else:
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path} (looked for trainable.pt and mp_rank_00_model_states.pt)"
        )
    loaded = 0
    model_state = dict(model.named_parameters())
    for name, tensor in state.items():
        if name in model_state:
            try:
                model_state[name].data.copy_(tensor.to(device=device, dtype=model_state[name].dtype))
                loaded += 1
            except Exception as e:
                print(f"  skip {name}: {e}")
    print(f"  loaded {loaded}/{len(state)} params")
    model.eval()


# ---------------------------------------------------------------------------
# RAG index (ESM C mean-pool, L2-normalized)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _esmc_pool(encoder: ProteinEncoder, seqs: List[str], device: str) -> List[np.ndarray]:
    emb, mask = encoder.encode(seqs, device)
    out: List[np.ndarray] = []
    for i in range(len(seqs)):
        n = int(mask[i].sum().item())
        out.append(emb[i, :n].mean(dim=0).float().cpu().numpy().astype(np.float32))
    return out


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)


def _load_rag_index() -> Tuple[Dict[str, np.ndarray], int]:
    if not RAG_INDEX_PATH.exists():
        return {}, 0
    npz = np.load(RAG_INDEX_PATH, allow_pickle=True)
    ids = npz["ids"].tolist()
    embs = npz["embs"]
    return {acc: embs[i] for i, acc in enumerate(ids)}, embs.shape[1] if len(ids) else 0


def _save_rag_index(emb_by_acc: Dict[str, np.ndarray]) -> None:
    ids = list(emb_by_acc.keys())
    arr = _normalize_rows(np.stack([emb_by_acc[a] for a in ids]).astype(np.float32))
    np.savez(RAG_INDEX_PATH, embs=arr, ids=np.array(ids))
    print(f"  wrote {RAG_INDEX_PATH}  N={len(ids)}  dim={arr.shape[1]}")


def cmd_rag_index(args: argparse.Namespace) -> int:
    """Compute / extend the ESM C mean-pool index used as RAG source AND as the
    ESM-C-only probe baseline view."""
    device = args.device
    cfg = IDProConfig()
    cfg.encoder.name = "esmc-600m"
    cfg.encoder.resolve()
    encoder = ProteinEncoder(cfg.encoder)
    encoder.load(device)

    splits_to_index = ["reference"] + list(args.include or [])
    rows_by_split = {s: _load_jsonl(_split_path(s)) for s in splits_to_index}

    existing, _ = _load_rag_index()
    if existing:
        print(f"Loaded {len(existing)} existing ESM C embeddings from {RAG_INDEX_PATH.name}")

    bs = args.batch_size
    for split, rows in rows_by_split.items():
        todo = [r for r in rows if r["accession"] not in existing]
        print(f"Split {split}: {len(rows)} rows, need ESM C for {len(todo)}")
        for i in range(0, len(todo), bs):
            batch = todo[i : i + bs]
            seqs = [r["sequence"][:MAX_SEQ_LEN] for r in batch]
            pooled = _esmc_pool(encoder, seqs, device)
            for r, vec in zip(batch, pooled):
                existing[r["accession"]] = vec
            if (i // bs) % 5 == 0:
                print(f"  {split}: {min(i + bs, len(todo))}/{len(todo)}")

    _save_rag_index(existing)
    return 0


def _rag_retrieve(query_emb: np.ndarray, index_embs: np.ndarray, index_rows: List[dict],
                  k: int, exclude_self_acc: Optional[str]) -> str:
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    sims = index_embs @ q
    if exclude_self_acc is not None:
        for i, r in enumerate(index_rows):
            if r["accession"] == exclude_self_acc:
                sims[i] = -1.0
    top = np.argsort(-sims)[:k]
    parts = []
    for rank, idx in enumerate(top):
        desc = index_rows[int(idx)].get("description", "").strip()
        if desc:
            parts.append(f"[{rank+1}] {desc}")
    return "\n".join(parts)[:RAG_CTX_MAX_CHARS]


# ---------------------------------------------------------------------------
# IDPro view extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def _extract_views(
    model: IDProModel,
    sequence: str,
    rag_context: str,
    device: str,
    pdb_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    structures = [pdb_path] if pdb_path else None
    protein_tokens, protein_mask, _ = model.encode_protein([sequence], device, structures=structures)

    full_question = QUESTION.strip() + PROMPTEOL_TEMPLATE
    inputs = model.build_inputs(
        protein_tokens,
        protein_mask,
        [full_question],
        rag_contexts=[rag_context] if rag_context else None,
        answers=None,
        device=device,
    )
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
    mask = inputs["attention_mask"][0]
    assert int(mask.sum().item()) == q_end, (
        f"length mismatch: computed {q_end} vs mask_sum {int(mask.sum().item())}"
    )
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
    q_only_end = q_end - n_prompteol
    q_only_start = q_start

    return {
        "view_a_prompteol_l48": h_mid[prompteol_pos].float().cpu(),
        "view_b_question_mean_l48": h_mid[q_only_start:q_only_end].mean(dim=0).float().cpu(),
        "view_c_eos_l64": h_last[prompteol_pos].float().cpu(),
    }


def cmd_views(args: argparse.Namespace) -> int:
    device = args.device
    print(f"Device: {device}  ckpt: {args.ckpt}  which: {args.which}")

    structure_lookup: Dict[str, str] = {}
    if args.structure_track:
        if not args.structure_manifest:
            raise SystemExit("ERROR: --structure-track requires --structure-manifest")
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
        print(f"Structure manifest: {len(structure_lookup):,} accessions")

    rows = _load_jsonl(_split_path(args.which))
    ref_rows = _load_jsonl(_split_path("reference"))
    print(f"Loaded {len(rows)} {args.which} rows; reference set {len(ref_rows)} for RAG")
    if args.limit:
        rows = rows[: args.limit]

    out_path = _emb_path(args.which)
    results: Dict[str, Dict] = {}
    if out_path.exists():
        try:
            prev = torch.load(out_path, map_location="cpu", weights_only=False)
            if isinstance(prev, dict):
                results = prev
                print(f"Resuming: {len(results)} already done")
        except Exception as e:
            print(f"Could not resume: {e}")

    todo = [r for r in rows if r["accession"] not in results]
    if not todo:
        print("All rows already cached.")
        return 0
    print(f"To process: {len(todo)} / {len(rows)}")

    if not RAG_INDEX_PATH.exists():
        raise SystemExit(
            f"RAG index missing at {RAG_INDEX_PATH}. Run "
            f"`extract_embeddings.py rag-index` first."
        )
    npz = np.load(RAG_INDEX_PATH, allow_pickle=True)
    all_ids = npz["ids"].tolist()
    all_embs = npz["embs"]
    id_to_row = {r["accession"]: r for r in ref_rows}
    keep = [(i, aid) for i, aid in enumerate(all_ids) if aid in id_to_row]
    rag_embs = np.stack([all_embs[i] for i, _ in keep])
    rag_rows = [id_to_row[aid] for _, aid in keep]
    rag_lookup = {aid: all_embs[i] for i, aid in enumerate(all_ids)}
    print(f"  RAG source: {len(rag_rows)} reference proteins (filtered from {len(all_ids)})")

    model = _build_model(
        device=device,
        encoder_name=args.encoder,
        structure_track=args.structure_track,
        structure_manifest_path=args.structure_manifest,
    )
    _load_checkpoint(model, args.ckpt, device)

    save_every = 50
    t0 = time.time()
    for i, row in enumerate(todo):
        acc = row["accession"]
        seq = row["sequence"][:MAX_SEQ_LEN]

        if acc in rag_lookup:
            q_emb = rag_lookup[acc]
        else:
            q_emb = _esmc_pool(model.encoder, [seq], device)[0]
        ctx = _rag_retrieve(
            q_emb, rag_embs, rag_rows, k=args.rag_k,
            exclude_self_acc=acc if args.which == "reference" else None,
        )

        pdb_path = structure_lookup.get(acc) if args.structure_track else None
        try:
            views = _extract_views(model, seq, ctx, device, pdb_path=pdb_path)
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
            print(f"  [save] {len(results)} → {out_path}")

    torch.save(results, out_path)
    print(f"\nDone. {len(results)} proteins → {out_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rag = sub.add_parser("rag-index", help="Build / extend the ESM C mean-pool index")
    p_rag.add_argument("--include", nargs="*", default=["benchmark"],
                       help="Splits to index in addition to reference (default: benchmark)")
    p_rag.add_argument("--batch-size", type=int, default=16)
    p_rag.add_argument("--device", default="cuda")
    p_rag.set_defaults(func=cmd_rag_index)

    p_v = sub.add_parser("views", help="Extract IDPro 3-view caches for one split")
    p_v.add_argument("--which", choices=["reference", "benchmark", "dark"], required=True)
    p_v.add_argument("--ckpt", type=Path, required=True)
    p_v.add_argument("--device", default="cuda")
    p_v.add_argument("--limit", type=int, default=None)
    p_v.add_argument("--rag-k", type=int, default=RAG_K_DEFAULT)
    p_v.add_argument("--encoder", default="esmc-600m",
                     choices=["esmc-600m", "esmc-300m", "esm2-650m", "esm2-3b", "esm3-1.4b"])
    p_v.add_argument("--structure-track", action="store_true")
    p_v.add_argument("--structure-manifest", default="")
    p_v.set_defaults(func=cmd_views)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
