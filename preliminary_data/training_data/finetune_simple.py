#!/usr/bin/env python3
"""
Simple LoRA fine-tune of P2T on microbial data.

Approach:
1. Load the already-merged P2T model (inference-ready)
2. Freeze ESM2 encoder + projector + resampler
3. Apply fresh LoRA to the LLaMA layers only
4. Train on microbial QA pairs using the P2T inference pathway
5. Save the new LoRA adapter

This avoids the ESM3 import issue in the LLaVA training code.
The key insight: we only need to adapt the LANGUAGE model to microbial
terminology — the protein encoder (ESM2) already handles microbial sequences.

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n protein2text_env python finetune_simple.py
"""

import os
import sys
import json
import time
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# P2T imports
P2T_DIR = Path("/data/asahu/projects/doe_genesis/Protein2Text")
sys.path.insert(0, str(P2T_DIR))
from evaluation.model.builder import load_pretrained_model
from evaluation.constants import DEFAULT_PROTEIN_SEQUENCE_TOKEN, PROTEIN_SEQUENCE_TOKEN_INDEX
from evaluation.inference_model import protein_sequence_tokenizer, eval_model
from evaluation.conversation import conv_templates

# Using partial unfreezing instead of LoRA to avoid PEFT wrapping issues with P2T's custom forward

# ── Config ─────────────────────────────────────────────────────────────
MODEL_PATH = str(P2T_DIR / "checkpoints/protein2text-llama3.1-8B-instruct-esm2-650M")
MODEL_BASE = "/data/ajararweh/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"
TRAIN_DATA = "/data/asahu/projects/doe_genesis/preliminary_data/training_data/uniprot/uniprot_combined_qa.json"
OUTPUT_DIR = "/data/asahu/projects/doe_genesis/preliminary_data/training_data/finetuned_microbial_p2t"

N_SAMPLES = 2000       # Proof of concept — small but enough to show improvement
N_EPOCHS = 1            # Single epoch for speed
LR = 5e-5
LORA_R = 16
LORA_ALPHA = 32
MAX_LEN = 512
EVAL_EVERY = 200        # Evaluate every N steps


def main():
    device = "cuda:1"
    t_start = time.time()

    print("=" * 60)
    print("IDPro: LoRA Fine-Tune P2T on Microbial Data")
    print("=" * 60)

    # ── Load Data ──────────────────────────────────────────────────────
    print("\nLoading data...")
    with open(TRAIN_DATA) as f:
        all_data = json.load(f)

    random.seed(42)
    random.shuffle(all_data)
    data = all_data[:N_SAMPLES]
    train_data = data[:int(0.9*len(data))]
    val_data = data[int(0.9*len(data)):]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # ── Load Model ─────────────────────────────────────────────────────
    print("\nLoading P2T model...")
    tokenizer, model, ctx_len = load_pretrained_model(
        MODEL_PATH, MODEL_BASE, "protein2text-llama3.1-8B-instruct-esm2-650M",
        device=device
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  Model loaded on {device}")

    # ── Freeze everything except a few layers ─────────────────────────
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only LLaMA's last 4 layers + lm_head for efficient fine-tuning
    # (simpler than LoRA, avoids PEFT wrapping issues with P2T's custom forward)
    unfrozen = 0
    for name, param in model.named_parameters():
        # Unfreeze last 4 transformer layers
        if any(f"layers.{i}." in name for i in range(28, 32)):
            param.requires_grad = True
            unfrozen += param.numel()
        # Unfreeze lm_head
        elif "lm_head" in name:
            param.requires_grad = True
            unfrozen += param.numel()
        # Unfreeze projector (to adapt to microbial embeddings)
        elif "mm_projector" in name:
            param.requires_grad = True
            unfrozen += param.numel()

    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Trainable: {unfrozen:,} / {total:,} ({100*unfrozen/total:.2f}%)")
    print(f"  Strategy: last 4 LLaMA layers + lm_head + projector")

    # ── Training Setup ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )

    model.train()
    model = model.bfloat16()

    training_log = []
    step = 0
    total_loss = 0

    # ── Training Loop ──────────────────────────────────────────────────
    print(f"\nTraining for {N_EPOCHS} epoch(s) on {len(train_data)} samples...")

    for epoch in range(N_EPOCHS):
        random.shuffle(train_data)

        for i, sample in enumerate(train_data):
            question = sample["conversations"][0]["value"].replace("<protein_sequence>\n", "")
            answer = sample["conversations"][1]["value"]
            amino_seq = sample["amino_seq"]

            # Build prompt: protein token + question + answer (plain template uses \n separator)
            qs = DEFAULT_PROTEIN_SEQUENCE_TOKEN + "\n" + question
            prompt = qs + "\n" + answer

            # Tokenize
            input_ids = protein_sequence_tokenizer(
                prompt, tokenizer, PROTEIN_SEQUENCE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            # Truncate
            if input_ids.shape[1] > MAX_LEN:
                input_ids = input_ids[:, :MAX_LEN]

            # Create labels (shift by 1 for causal LM)
            labels = input_ids.clone()
            # Mask the question part — only train on the answer
            question_prompt = qs + "\n"
            question_tokens = protein_sequence_tokenizer(
                question_prompt, tokenizer, PROTEIN_SEQUENCE_TOKEN_INDEX, return_tensors="pt"
            )
            q_len = min(len(question_tokens), labels.shape[1])
            labels[0, :q_len] = -100

            try:
                # Forward
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=input_ids, amino_seq=amino_seq, labels=labels)

                if hasattr(loss, 'loss'):
                    loss = loss.loss
                elif isinstance(loss, tuple):
                    loss = loss[0]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                step += 1

                if step % 50 == 0:
                    avg_loss = total_loss / step
                    elapsed = (time.time() - t_start) / 60
                    rate = step / max(elapsed, 0.01)
                    remaining = (len(train_data) * N_EPOCHS - step) / max(rate, 0.01)
                    print(f"  Step {step}/{len(train_data)*N_EPOCHS} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"Time: {elapsed:.0f}m | "
                          f"ETA: {remaining:.0f}m")

                # Periodic evaluation
                if step % EVAL_EVERY == 0:
                    eval_loss = evaluate(model, tokenizer, val_data[:20], device)
                    training_log.append({
                        "step": step,
                        "train_loss": total_loss / step,
                        "eval_loss": eval_loss,
                        "time_min": (time.time() - t_start) / 60,
                    })
                    print(f"  [EVAL] Step {step}: eval_loss={eval_loss:.4f}")
                    model.train()

            except Exception as e:
                if step < 5:
                    print(f"  Step {step} error: {str(e)[:200]}")
                optimizer.zero_grad()
                continue

    # ── Save Model ─────────────────────────────────────────────────────
    total_time = (time.time() - t_start) / 60
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save LoRA adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training log
    summary = {
        "n_train": len(train_data),
        "n_val": len(val_data),
        "epochs": N_EPOCHS,
        "total_steps": step,
        "final_avg_loss": total_loss / max(step, 1),
        "lora_rank": LORA_R,
        "lr": LR,
        "total_time_min": total_time,
        "training_log": training_log,
    }
    with open(os.path.join(OUTPUT_DIR, "training_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Steps: {step}")
    print(f"  Final loss: {total_loss/max(step,1):.4f}")
    print(f"  Time: {total_time:.0f} min")
    print(f"  Saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


def evaluate(model, tokenizer, val_data, device):
    """Quick evaluation on validation set."""
    model.eval()
    total_loss = 0
    n = 0

    with torch.no_grad():
        for sample in val_data:
            question = sample["conversations"][0]["value"].replace("<protein_sequence>\n", "")
            answer = sample["conversations"][1]["value"]
            amino_seq = sample["amino_seq"]

            qs = DEFAULT_PROTEIN_SEQUENCE_TOKEN + "\n" + question
            prompt = qs + "\n" + answer

            input_ids = protein_sequence_tokenizer(
                prompt, tokenizer, PROTEIN_SEQUENCE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            if input_ids.shape[1] > MAX_LEN:
                input_ids = input_ids[:, :MAX_LEN]

            labels = input_ids.clone()

            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=input_ids, amino_seq=amino_seq, labels=labels)
                if isinstance(loss, tuple):
                    loss = loss[0]
                if hasattr(loss, 'loss'):
                    loss = loss.loss
                total_loss += loss.item()
                n += 1
            except:
                continue

    return total_loss / max(n, 1)


if __name__ == "__main__":
    main()
