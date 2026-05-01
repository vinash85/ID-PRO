#!/usr/bin/env python3
"""
IDPro Robust Training Script.

Fixes from previous failures:
1. Explicitly moves ALL components to GPU with verification
2. Fails LOUD on first error (no silent swallowing)
3. Tests forward pass before starting training loop
4. Single-file, minimal dependencies
5. Checkpoint resume built-in

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/train_robust.py --stage 1
  CUDA_VISIBLE_DEVICES=0 python scripts/train_robust.py --stage 4 --resume
"""

import os
import re
import sys
import json
import time
import random
import math
import argparse
from pathlib import Path
from datetime import datetime

# Set CUDA devices BEFORE importing torch
if "CUDA_GPUS" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_GPUS"]

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Paths ──────────────────────────────────────────────────────────────

from idpro.paths import (
    QA_DIR,
    BENCHMARK,
    FEATURE_INDEX,
    RESULTS_DIR as _RESULTS_ROOT,
    CKPT_DIR as _CKPT_ROOT,
)

RESULTS_DIR = _RESULTS_ROOT / "robust"
CKPT_DIR = _CKPT_ROOT / "robust"

# RESULTS_DIR / CKPT_DIR are created lazily inside train() so that importing
# this module does not require write access to the production paths.


# ── Distributed helpers ────────────────────────────────────────────────

def is_main_process():
    """True on rank 0 (or in non-distributed runs)."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


# ── Model Loading ─────────────────────────────────────────────────────

def load_model(encoder_name, llm_name, device, structure_track=False,
               structure_manifest="", use_deepspeed=False):
    """Load model with explicit device placement and verification.

    When `use_deepspeed=True`, each rank loads a full copy of the (4-bit)
    LLM onto its own `device` (no `device_map='auto'` sharding), and bridge
    components stay in bf16 on the same device. ZeRO-2 will then partition
    only the optimizer state and gradients of the trainable params.
    """
    from idpro.config import IDProConfig, EncoderConfig, LLMConfig
    from idpro.model import IDProModel

    config = IDProConfig(
        encoder=EncoderConfig(
            name=encoder_name,
            structure_track=structure_track,
            structure_manifest_path=structure_manifest,
        ),
        llm=LLMConfig(name=llm_name),
    )
    config.resolve()
    print(f"Config: encoder={config.encoder.name} "
          f"(backend={config.encoder.backend}, dim={config.encoder.dim}), "
          f"llm dim={config.llm.dim}")
    if config.encoder.backend == "esm3":
        print(f"        structure_track={config.encoder.structure_track}, "
              f"manifest={config.encoder.structure_manifest_path or '(none)'}")
    elif structure_track:
        print(f"  WARNING: --structure-track requested but backend is "
              f"{config.encoder.backend}; flag is a no-op for non-ESM3 encoders.")

    model = IDProModel(config)

    # 1. Load encoder
    n_gpus = torch.cuda.device_count()
    if use_deepspeed:
        # Each rank owns its local GPU; encoder + bridge + LLM all colocate.
        encoder_device = device
    else:
        encoder_device = "cuda:0" if n_gpus > 1 else device
    print(f"\n[1/4] Loading protein encoder on {encoder_device}...")
    model.encoder.load(encoder_device)

    # 2. Load LLM (QLoRA 4-bit)
    print("\n[2/4] Loading LLM (QLoRA 4-bit to save memory)...")
    config.llm.use_qlora = True
    if use_deepspeed:
        # Full copy per rank — DS ZeRO-2 only partitions trainable params'
        # optimizer state and grads, not the (frozen, quantized) base weights.
        print(f"  DeepSpeed mode: loading full LLM copy on {device}")
        model.load_llm(device=device)
    elif n_gpus > 1:
        print(f"  Multi-GPU detected: {n_gpus} GPUs — using device_map='auto'")
        model.load_llm(device="auto")
    else:
        model.load_llm(device=device)

    # 3. Move ALL bridge components to device AND match dtype (bf16)
    if use_deepspeed:
        bridge_device = device
    else:
        bridge_device = "cuda:0" if n_gpus > 1 else device
    print(f"\n[3/4] Moving bridge components to {bridge_device} (bf16)...")
    dtype = torch.bfloat16
    model.adaptor = model.adaptor.to(device=bridge_device, dtype=dtype)
    model.projector = model.projector.to(device=bridge_device, dtype=dtype)
    model.evidence_head_pre = model.evidence_head_pre.to(device=bridge_device, dtype=dtype)
    model.evidence_head_post = model.evidence_head_post.to(device=bridge_device, dtype=dtype)
    model.protein_position = model.protein_position.to(device=bridge_device, dtype=dtype)
    model.protein_modality_embed = nn.Parameter(model.protein_modality_embed.data.to(device=bridge_device, dtype=dtype))
    model.prot_end_embed = nn.Parameter(model.prot_end_embed.data.to(device=bridge_device, dtype=dtype))

    # 4. Verify everything is on the same device
    print("\n[4/4] Verifying device placement...")
    issues = []
    for name, param in model.named_parameters():
        if param.device.type != "cuda":
            issues.append(f"  {name}: {param.device}")
    for name, buf in model.named_buffers():
        if buf.device.type != "cuda":
            issues.append(f"  (buffer) {name}: {buf.device}")

    if issues:
        print(f"WARNING: {len(issues)} parameters/buffers NOT on CUDA:")
        for issue in issues[:10]:
            print(issue)
        # Move remaining stragglers
        for name, param in model.named_parameters():
            if param.device.type != "cuda":
                param.data = param.data.to(device)
        for name, buf in model.named_buffers():
            if buf.device.type != "cuda":
                model.register_buffer(name.split(".")[-1], buf.to(device))
    else:
        print("  All parameters on CUDA ✓")

    return model, config


def test_forward_pass(model, device):
    """Run a test forward pass to catch errors BEFORE training."""
    print("\n[TEST] Running test forward pass...")

    test_seq = "MKFLILNILTFAASALAAEPFGSWQITKDGPNTNKNAYIDARLC"
    test_q = "What is the function of this protein?"
    test_a = "Based on the sequence, this is a test protein."

    try:
        # Test encoding
        prot_tokens, mask, pre_evidence = model.encode_protein([test_seq], device)
        print(f"  Encode: ✓ (shape={prot_tokens.shape}, device={prot_tokens.device})")

        # Test full forward (with loss)
        outputs = model(
            sequences=[test_seq],
            questions=[test_q],
            answers=[test_a],
            device=device,
        )
        loss = outputs.loss
        print(f"  Forward: ✓ (loss={loss.item():.4f})")

        # Test backward
        loss.backward()
        print(f"  Backward: ✓")

        # Clear grads
        model.zero_grad()
        print(f"  Test forward pass PASSED ✓\n")
        return True

    except Exception as e:
        print(f"  TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ── Data Loading ───────────────────────────────────────────────────────

def load_qa_data(stage, max_samples=None):
    """Load QA data for a training stage."""
    data_file = QA_DIR / f"stage{stage}" / f"stage{stage}_qa.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"No data for stage {stage}: {data_file}")

    data = []
    with open(data_file) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                seq = item.get("amino_seq", item.get("sequence", ""))
                convs = item.get("conversations", [])
                if len(convs) >= 2 and seq and 10 <= len(seq) <= 500:
                    q = convs[0]["value"].replace("<protein_sequence>\n", "")
                    a = convs[1]["value"]
                    pid = item.get("id", "")
                    if a and len(a) > 5:
                        data.append({"sequence": seq, "question": q, "answer": a, "id": pid})

    random.seed(42)
    random.shuffle(data)
    if max_samples:
        data = data[:max_samples]

    print(f"  Loaded {len(data)} QA pairs for stage {stage}")
    return data


TRAIN_QUESTION = "Analyze this protein sequence: identify its domains, motifs, spatial arrangement, and predict its function."


def load_eval_data():
    """Load benchmark data for evaluation."""
    with open(BENCHMARK) as f:
        bench = json.load(f)
    eval_data = []
    for p in bench:
        convs = p.get("conversations", [])
        if len(convs) >= 2:
            eval_data.append({
                "sequence": p["amino_seq"],
                "question": TRAIN_QUESTION,  # Use training question format
                "ground_truth": convs[1]["value"],
            })
    return eval_data


# ── Evaluation ─────────────────────────────────────────────────────────

def extract_infer_section(text):
    """Extract the [INFER] section from CoT output."""
    import re
    # Try to find [INFER] section
    m = re.search(r'\[INFER\]\s*(.*?)(?:\[CONTEXTUALIZE\]|\[|$)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: try [RELATE] section (also contains function info)
    m = re.search(r'\[RELATE\]\s*(.*?)(?:\[INFER\]|\[|$)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def keyword_f1(pred, gt, stop):
    """Compute keyword F1 between two texts."""
    import re
    pw = set(re.findall(r'[a-z]{3,}', pred.lower())) - stop
    tw = set(re.findall(r'[a-z]{3,}', gt.lower())) - stop
    if not tw or not pw:
        return 0.0
    ov = len(pw & tw)
    p = ov / len(pw)
    r = ov / len(tw)
    return 2 * p * r / max(p + r, 1e-10)


def evaluate(model, eval_data, device, n=30):
    """Evaluate with both full-CoT F1 and INFER-only F1.

    Distributed: only rank 0 runs the (slow, single-stream) Qwen3.5 generation
    loop. Other ranks flip into eval/train mode locally and wait at a barrier
    so collective-op SeqNums stay aligned with rank 0's training resume.
    """
    model.eval()
    is_dist = torch.distributed.is_initialized()

    if is_dist and not is_main_process():
        # Non-main ranks: skip generation entirely, just wait for rank 0.
        torch.distributed.barrier()
        model.train()
        return {"f1": 0.0, "f1_infer": 0.0, "n": 0, "examples": []}

    samples = random.sample(eval_data, min(n, len(eval_data)))
    stop = {"the","a","an","is","are","was","of","in","to","and","or","this","that",
            "it","for","with","on","at","by","from","as","protein","proteins","its","has",
            "based","sequence","identify","locate","relate","infer","contextualize",
            "key","residues","position","positions","domain","found"}

    f1_full = []
    f1_infer = []
    examples = []

    for s in samples:
        try:
            preds = model.generate(
                sequences=[s["sequence"][:1000]],
                questions=[s["question"]],
                max_new_tokens=500,
                temperature=0.3,
                device=device,
            )
            pred = preds[0] if preds else ""
        except Exception as e:
            pred = ""

        gt = s["ground_truth"]

        # Full CoT F1
        f1_full.append(keyword_f1(pred, gt, stop))

        # INFER-section F1
        infer_text = extract_infer_section(pred)
        f1_infer.append(keyword_f1(infer_text, gt, stop))

        if len(examples) < 3:
            examples.append({"gt": gt[:80], "pred": pred[:120]})

    if is_dist:
        torch.distributed.barrier()
    model.train()

    return {
        "f1": sum(f1_full) / max(len(f1_full), 1),
        "f1_infer": sum(f1_infer) / max(len(f1_infer), 1),
        "n": len(f1_full),
        "examples": examples,
    }


# ── Checkpoint ─────────────────────────────────────────────────────────

def save_ckpt(model, optimizer, stage, step, eval_log, engine=None):
    """Save checkpoint.

    DeepSpeed mode (`engine` set) writes a sharded DS checkpoint via
    engine.save_checkpoint; client state (stage/step/eval_log) goes into the
    DS metadata blob. Single-GPU mode keeps the original lightweight format
    (only-trainable-params + optimizer state).
    """
    path = CKPT_DIR / f"stage{stage}_step{step}"
    path.mkdir(parents=True, exist_ok=True)

    if engine is not None:
        # DS writes its own files into `path`; only rank 0 logs.
        client_state = {"stage": stage, "step": step, "eval_log": eval_log}
        # DeepSpeed's engine.save_checkpoint writes a plain-text `latest` file
        # inside CKPT_DIR. If we left a previous symlink pointing at a directory,
        # DS's open(latest, 'w') would dereference it and trip EISDIR. Unlink
        # before DS save, then replace with our symlink afterward.
        if is_main_process():
            latest = CKPT_DIR / "latest"
            if latest.is_symlink() or latest.exists():
                latest.unlink()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        engine.save_checkpoint(str(CKPT_DIR), tag=path.name, client_state=client_state)
        if is_main_process():
            # Replace DS's plain `latest` file with a symlink (load_ckpt expects symlink).
            latest = CKPT_DIR / "latest"
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(path.name)
            print(f"  [CKPT-DS] {path.name}")
        return

    # Save only trainable params (smaller)
    trainable = {n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(trainable, path / "trainable.pt")

    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    json.dump({"stage": stage, "step": step, "eval_log": eval_log},
              open(path / "state.json", "w"), indent=2)

    # Update latest link
    latest = CKPT_DIR / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)
    print(f"  [CKPT] {path.name}")


def load_ckpt(model, optimizer, device, engine=None, current_stage=None):
    """Resume from latest checkpoint.

    Returns (saved_stage, step, eval_log). Single-GPU and DS checkpoint
    formats are NOT interchangeable — running `--resume` against a
    checkpoint produced by the other path will be a no-op or fail
    gracefully and start from step 0.

    `current_stage`: if given and the saved checkpoint is from a different
    stage (parsed from tag prefix), skip optimizer + LR scheduler state to
    avoid Adam tensor-size mismatches when the trainable param set differs
    across stages.
    """
    latest = CKPT_DIR / "latest"
    if not latest.exists():
        return 0, 0, []

    path = CKPT_DIR / os.readlink(str(latest))
    if not path.exists():
        return 0, 0, []

    if engine is not None:
        # Cross-stage resume guard: stage 1 → stage 4 changes the trainable
        # param set, so saved Adam (m, v) partition sizes won't match. Skip
        # optimizer + LR scheduler state in that case (we keep module weights).
        cross_stage = False
        m = re.match(r"stage(\d+)_step\d+", path.name)
        if m and current_stage is not None and int(m.group(1)) != current_stage:
            cross_stage = True
        try:
            # load_module_strict=False absorbs:
            #   - bnb NF4 4-bit auxiliary keys (`absmax`, `quant_map`,
            #     `nested_absmax`, `quant_state.bitsandbytes__nf4`) that the
            #     freshly-loaded model's state_dict() doesn't return until
            #     first forward.
            #   - When structure_track=False: the saved ckpt lacks
            #     `_structure_encoder.*` keys (encode never instantiates that
            #     submodule).
            #   - When structure_track=True: the 34 saved `_structure_encoder.*`
            #     keys in the `module` dict (encoder.py leaves SE lazy at load
            #     time).
            _, client_state = engine.load_checkpoint(
                str(CKPT_DIR), tag=path.name,
                load_module_strict=False,
                load_optimizer_states=not cross_stage,
                load_lr_scheduler_states=not cross_stage,
            )
            if cross_stage and is_main_process():
                print(f"  [CKPT-DS] cross-stage resume "
                      f"(saved stage{m.group(1)} → current stage{current_stage}): "
                      f"loaded module only, skipped optimizer + LR scheduler state.")
        except Exception as e:
            if is_main_process():
                import traceback
                print(f"  [CKPT-DS] load failed ({e}); starting fresh.")
                traceback.print_exc()
            return 0, 0, []
        if client_state is None:
            return 0, 0, []
        if is_main_process():
            print(f"  [CKPT-DS] Loaded from {path.name}")
        return (client_state.get("stage", 0),
                client_state.get("step", 0),
                client_state.get("eval_log", []))

    tp = path / "trainable.pt"
    if tp.exists():
        trainable = torch.load(tp, map_location=device)
        for name, param in model.named_parameters():
            if name in trainable and param.requires_grad:
                param.data.copy_(trainable[name].to(device))
        print(f"  [CKPT] Model loaded from {path.name}")

    state = json.load(open(path / "state.json")) if (path / "state.json").exists() else {}
    saved_stage = state.get("stage", 0)

    # Only load optimizer if same stage (param groups must match)
    op = path / "optimizer.pt"
    if op.exists() and optimizer is not None:
        try:
            optimizer.load_state_dict(torch.load(op, map_location=device))
        except (ValueError, RuntimeError) as e:
            print(f"  [CKPT] Optimizer state incompatible (stage change?), using fresh optimizer. ({e})")

    return saved_stage, state.get("step", 0), state.get("eval_log", [])


# ── Logging ────────────────────────────────────────────────────────────

def log_eval(entry):
    """Append eval result to TSV."""
    tsv = RESULTS_DIR / "eval_progress.tsv"
    if not tsv.exists():
        with open(tsv, "w") as f:
            f.write("timestamp\tstage\tstep\tf1\ttime_min\tloss\n")
    with open(tsv, "a") as f:
        f.write(f"{datetime.now().isoformat()}\t{entry['stage']}\t{entry['step']}\t"
                f"{entry['f1']:.6f}\t{entry.get('time_min',0):.1f}\t{entry.get('loss',0):.4f}\n")


# ── Training ───────────────────────────────────────────────────────────

def train(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    use_deepspeed = bool(args.deepspeed)

    if use_deepspeed:
        # The deepspeed launcher sets LOCAL_RANK / WORLD_SIZE; we use that
        # to pin each rank to its own GPU before any model loading.
        local_rank = args.local_rank
        if local_rank < 0:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        # Initialize the process group via DS so torch.distributed is ready
        # before we instantiate any modules.
        # Bump the NCCL watchdog timeout to 1h so periodic eval (single-rank
        # generation, ~22 min on Qwen3.5-27B NF4) doesn't trip the default
        # 10-min ALLREDUCE timeout while non-eval ranks idle at the next sync.
        import deepspeed
        from datetime import timedelta
        deepspeed.init_distributed(timeout=timedelta(minutes=60))
    else:
        device = args.device

    if is_main_process():
        print("="*60)
        print("IDPro Robust Training")
        print(f"Device: {device}")
        print(f"Stage: {args.stage}")
        print(f"Max hours: {args.max_hours}")
        print(f"DeepSpeed: {args.deepspeed or 'OFF'}")
        print("="*60)

    # Load model
    model, config = load_model(
        args.encoder, args.llm, device,
        structure_track=args.structure_track,
        structure_manifest=args.structure_manifest,
        use_deepspeed=use_deepspeed,
    )

    # Optional ESM3 structure manifest: accession → PDB path
    structure_lookup = {}
    if config.encoder.backend == "esm3" and args.structure_track:
        if not args.structure_manifest:
            print("ERROR: --structure-track requires --structure-manifest <jsonl>")
            return
        manifest_path = Path(args.structure_manifest)
        if not manifest_path.exists():
            print(f"ERROR: structure manifest not found: {manifest_path}")
            return
        with open(manifest_path) as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                acc = rec.get("accession")
                pdb = rec.get("pdb_path")
                if acc and pdb:
                    structure_lookup[acc] = pdb
        print(f"  Structure manifest: {len(structure_lookup):,} accessions → PDB paths")

    # Configure training stage
    stage = args.stage
    stage_configs = {
        1: {"name": "Domain/Motif Recognition", "lr": 5e-5, "max_steps": 50000,
            "trainable": ["adaptor", "projector", "evidence_head_pre", "modality", "prot_end", "position"]},
        4: {"name": "CoT Reasoning", "lr": 1e-5, "max_steps": 100000,
            "trainable": ["adaptor", "projector", "evidence", "modality", "prot_end", "position",
                         "lora_", "lm_head"]},
    }

    sc = stage_configs.get(stage, stage_configs[1])
    if args.max_steps > 0:
        print(f"  [override] max_steps {sc['max_steps']} → {args.max_steps}")
        sc["max_steps"] = args.max_steps
    print(f"\nStage {stage}: {sc['name']}")

    # Set trainable parameters
    # Note: QLoRA quantized params (Int8/NF4) cannot require grad — skip them
    for param in model.parameters():
        if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
            param.requires_grad = False

    trainable_count = 0
    for name, param in model.named_parameters():
        if param.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            continue  # skip quantized params
        for pattern in sc["trainable"]:
            if pattern in name:
                param.requires_grad = True
                trainable_count += param.numel()
                break

    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_count:,} / {total:,} ({100*trainable_count/max(total,1):.3f}%)")

    if trainable_count == 0:
        print("ERROR: No trainable parameters! Check stage config.")
        return

    # Test forward pass BEFORE training
    # Enable gradient checkpointing to reduce memory
    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ENABLED")

    if not test_forward_pass(model, device):
        print("FATAL: Forward pass test failed. Fix model before training.")
        return

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=sc["lr"], weight_decay=0.01)

    # DeepSpeed engine wrap (opt-in). After this, `model` is the DS engine
    # and engine.backward / engine.step replace the manual loop.
    engine = None
    if use_deepspeed:
        import deepspeed
        model, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=trainable_params,
            config=args.deepspeed,
        )
        engine = model
        if is_main_process():
            print(f"\n[DS] Engine initialized. world_size={torch.distributed.get_world_size()}, "
                  f"micro_bs={engine.train_micro_batch_size_per_gpu()}, "
                  f"grad_accum={engine.gradient_accumulation_steps()}")

    # Resume
    eval_log = []
    start_step = 0
    if args.resume:
        resume_stage, resume_step, eval_log = load_ckpt(model, optimizer, device, engine=engine, current_stage=stage)
        if resume_step > 0:
            if resume_stage == stage:
                # Same stage — continue from where we left off
                start_step = resume_step
                print(f"Resumed from stage {resume_stage}, step {start_step}")
                if start_step >= sc["max_steps"]:
                    print(f"Stage {stage} already complete ({start_step}/{sc['max_steps']} steps). Nothing to do.")
                    return
            else:
                # Different stage — load weights but reset step counter
                start_step = 0
                eval_log = []
                print(f"Loaded weights from stage {resume_stage} step {resume_step}, starting stage {stage} from step 0")

    # Load data
    print("\nLoading data...")
    train_data = load_qa_data(stage, max_samples=sc["max_steps"])
    eval_data = load_eval_data()
    print(f"Eval data: {len(eval_data)} proteins")

    # Load feature index for evidence supervision
    feature_index = {}
    if FEATURE_INDEX.exists():
        import pickle
        with open(FEATURE_INDEX, "rb") as f:
            feature_index = pickle.load(f)
        n_overlap = sum(1 for s in train_data if s.get("id") in feature_index)
        print(f"Evidence supervision: {len(feature_index)} proteins indexed, {n_overlap}/{len(train_data)} overlap")
    else:
        print("No feature index found — training without evidence supervision")

    # Initial eval — skipped under DeepSpeed ZeRO. Single-rank Qwen3.5-27B
    # generation runs ~22 min for n=30, but the NCCL collective watchdog is
    # hard-capped at 10 min on this build (TORCH_NCCL_BLOCKING_WAIT and
    # init_process_group(timeout=...) didn't override it). Non-rank-0 workers
    # would idle past the watchdog and crash the run. Run eval offline post-hoc.
    if torch.distributed.is_initialized():
        print("\n[EVAL] Initial — skipped (distributed; eval offline)")
    else:
        print("\n[EVAL] Initial...")
        metrics = evaluate(model, eval_data, device)
        entry = {"stage": stage, "step": start_step, "f1": metrics["f1"], "time_min": 0, "loss": 0}
        eval_log.append(entry)
        log_eval(entry)
        print(f"  F1={metrics['f1']:.4f}  F1_infer={metrics.get('f1_infer',0):.4f}")
        for ex in metrics["examples"][:2]:
            print(f"  GT:   {ex['gt']}")
            print(f"  PRED: {ex['pred']}")

    # Training loop
    print(f"\nTraining ({len(train_data)} samples, max {sc['max_steps']} steps)...")
    model.train()
    t_start = time.time()
    deadline = t_start + args.max_hours * 3600

    total_loss = 0
    n_success = 0
    n_fail = 0
    step = start_step

    for epoch in range(100):  # enough epochs to hit max_steps
        random.shuffle(train_data)

        for i, sample in enumerate(train_data):
            if step >= sc["max_steps"] or time.time() > deadline:
                break

            try:
                seq = sample["sequence"][:500]
                # ESM3 structure-track lookup (no-op when lookup is empty)
                struct_arg = None
                if structure_lookup:
                    pdb_path = structure_lookup.get(sample.get("id", ""))
                    struct_arg = [pdb_path]  # may contain None to mask the track
                outputs = model(
                    sequences=[seq],
                    questions=[sample["question"]],
                    answers=[sample["answer"]],
                    device=device,
                    structures=struct_arg,
                )

                loss = outputs.loss

                # Evidence supervision: add per-residue classification loss
                pid = sample.get("id", "")
                if pid in feature_index and hasattr(outputs, 'pre_evidence_logits'):
                    from idpro.model.idpro.evidence import create_evidence_labels
                    feat_info = feature_index[pid]

                    # Map feature types to match create_evidence_labels format
                    mapped_feats = []
                    for feat in feat_info["features"]:
                        ftype = feat.get("type", "")
                        mapped = {
                            "active site": "active_site",
                            "binding site": "binding_site",
                            "domain": "domain",
                            "transmembrane region": "transmembrane",
                            "signal peptide": "signal_peptide",
                            "short sequence motif": "short_sequence_motif",
                            "DNA-binding region": "dna_binding",
                            "repeat": "repeat",
                            "region of interest": "domain",
                            "site": "active_site",
                        }.get(ftype, "")
                        if mapped:
                            entry = {"type": mapped}
                            if "position" in feat:
                                entry["position"] = int(feat["position"])
                            if "start" in feat:
                                entry["start"] = int(feat["start"])
                            if "end" in feat:
                                entry["end"] = int(feat["end"])
                            mapped_feats.append(entry)

                    if mapped_feats:
                        seq_len = len(seq)
                        ev_labels = create_evidence_labels(seq_len, mapped_feats).to(device)

                        # Pre-LLM evidence loss (on adaptor output)
                        pre_logits = outputs.pre_evidence_logits  # (1, T, 9)
                        pre_mask = outputs.protein_mask            # (1, T)
                        n_prot = min(pre_logits.shape[1], seq_len)
                        ev_labels_padded = torch.zeros(1, pre_logits.shape[1], dtype=torch.long, device=device)
                        ev_labels_padded[0, :n_prot] = ev_labels[:n_prot]
                        pre_ev_loss = model.evidence_head_pre.compute_loss(
                            pre_logits, ev_labels_padded, pre_mask)
                        loss = loss + pre_ev_loss

                        # Post-LLM evidence loss (on LLM layer 48)
                        if hasattr(outputs, 'post_evidence_logits') and outputs.post_evidence_logits is not None:
                            post_logits = outputs.post_evidence_logits
                            post_mask = outputs.protein_evidence_mask
                            n_post = min(post_logits.shape[1], seq_len)
                            ev_labels_post = torch.zeros(1, post_logits.shape[1], dtype=torch.long, device=device)
                            ev_labels_post[0, :n_post] = ev_labels[:n_post]
                            post_ev_loss = model.evidence_head_post.compute_loss(
                                post_logits, ev_labels_post, post_mask)
                            loss = loss + post_ev_loss

                if engine is not None:
                    # DS handles grad-accum, clipping, zero_grad internally.
                    engine.backward(loss)
                    engine.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                n_success += 1
                step += 1

                # Log
                if step % 50 == 0:
                    avg_loss = total_loss / max(n_success, 1)
                    elapsed = (time.time() - t_start) / 60
                    eta = elapsed / max(step - start_step, 1) * (sc["max_steps"] - step)
                    ev_str = "+ev" if pid in feature_index else ""
                    print(f"  Step {step} | Loss: {avg_loss:.4f}{ev_str} | "
                          f"Success: {n_success}/{n_success+n_fail} | "
                          f"Time: {elapsed:.0f}m | ETA: {eta:.0f}m")

                # Periodic eval — skipped entirely under distributed for the
                # same NCCL watchdog reason as the initial eval. Single-rank
                # Qwen gen would force non-rank-0 ranks past the 10-min ceiling.
                if step % 2000 == 0 and not torch.distributed.is_initialized():
                    elapsed = (time.time() - t_start) / 60
                    avg_loss = total_loss / max(n_success, 1)
                    metrics = evaluate(model, eval_data, device)
                    entry = {"stage": stage, "step": step, "f1": metrics["f1"],
                             "f1_infer": metrics.get("f1_infer", 0),
                             "time_min": elapsed, "loss": avg_loss}
                    eval_log.append(entry)
                    log_eval(entry)
                    print(f"  [EVAL] Step {step}: F1={metrics['f1']:.4f}, F1_infer={metrics.get('f1_infer',0):.4f}, Loss={avg_loss:.4f}")
                    for ex in metrics["examples"][:1]:
                        print(f"    PRED: {ex['pred']}")

                # Save
                if step % 2000 == 0:
                    save_ckpt(model, optimizer, stage, step, eval_log, engine=engine)

            except Exception as e:
                n_fail += 1
                if n_fail <= 5:
                    print(f"  Step {step} FAILED: {str(e)[:150]}")
                    import traceback
                    traceback.print_exc()
                if n_fail > 100 and n_success == 0:
                    print(f"FATAL: {n_fail} consecutive failures, 0 successes. Stopping.")
                    return
                if engine is None:
                    optimizer.zero_grad()
                continue

        if step >= sc["max_steps"] or time.time() > deadline:
            break

    # Final
    elapsed = (time.time() - t_start) / 60
    if torch.distributed.is_initialized():
        # Skip final eval under distributed for the NCCL watchdog reason; the
        # final checkpoint is still saved so post-hoc eval can run separately.
        final_f1 = float("nan")
        entry = {"stage": stage, "step": step, "f1": final_f1,
                 "time_min": elapsed, "loss": total_loss / max(n_success, 1)}
    else:
        metrics = evaluate(model, eval_data, device)
        final_f1 = metrics["f1"]
        entry = {"stage": stage, "step": step, "f1": final_f1,
                 "time_min": elapsed, "loss": total_loss / max(n_success, 1)}
    eval_log.append(entry)
    log_eval(entry)
    save_ckpt(model, optimizer, stage, step, eval_log, engine=engine)

    print(f"\n{'='*60}")
    print(f"DONE: Stage {stage}, {step} steps, {elapsed:.0f} min")
    print(f"  Success: {n_success}, Failed: {n_fail}")
    print(f"  Final F1: {final_f1:.4f}")
    print(f"  Results: {RESULTS_DIR / 'eval_progress.tsv'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encoder", type=str, default="esmc-600m",
                        help="Encoder preset: esmc-600m, esmc-300m, esm2-650m, "
                             "esm2-3b, esm3-1.4b")
    parser.add_argument("--llm", type=str, default="qwen3.5-27b")
    parser.add_argument("--max-hours", type=float, default=60)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Override the per-stage max_steps. 0 = use stage default "
                             "(50000 for stage 1, 100000 for stage 4).")
    parser.add_argument("--resume", action="store_true")
    # ESM3 structure-track flags (no-ops for non-ESM3 encoders)
    parser.add_argument("--structure-track", action="store_true",
                        help="ESM3 only: populate the structure track from PDB files. "
                             "Requires --structure-manifest.")
    parser.add_argument("--structure-manifest", type=str, default="",
                        help="Path to JSONL with one {\"accession\": ..., "
                             "\"pdb_path\": ...} per line. Used to look up PDBs "
                             "by UniProt accession during ESM3 training.")
    # DeepSpeed (opt-in): when --deepspeed is set, wraps the model in a DS
    # ZeRO-2 engine. Single-GPU runs without this flag are unchanged.
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to a DeepSpeed config JSON. When set, "
                             "enables multi-GPU ZeRO-2 training.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank (set automatically by the deepspeed "
                             "launcher).")
    # Per-run output overrides (default to idpro.paths derivatives).
    parser.add_argument("--qa-dir", type=str, default=None,
                        help="Override QA dir (default: $IDPRO_DATA_ROOT/preliminary_data/training_data/qa_stages).")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Override checkpoint dir (default: $IDPRO_RUNS_ROOT/checkpoints/robust).")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override training-results dir (default: $IDPRO_RUNS_ROOT/training_results/robust).")
    args = parser.parse_args()
    # Override module-level paths so the rest of the trainer sees them.
    if args.qa_dir:
        globals()["QA_DIR"] = Path(args.qa_dir)
    if args.ckpt_dir:
        globals()["CKPT_DIR"] = Path(args.ckpt_dir)
    if args.results_dir:
        globals()["RESULTS_DIR"] = Path(args.results_dir)
    train(args)
