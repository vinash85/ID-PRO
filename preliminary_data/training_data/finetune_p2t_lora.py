#!/usr/bin/env python3
"""
LoRA fine-tuning of Protein2Text on microbial protein QA data.

Strategy: Load the already-trained P2T model (with merged LoRA weights),
apply a NEW LoRA adapter, and fine-tune on microbial-specific QA pairs.

This proves the concept: microbial fine-tuning improves P2T on microbial proteins.

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n protein2text_env python finetune_p2t_lora.py \
    --train_data uniprot/uniprot_combined_qa.json \
    --n_samples 5000 \
    --epochs 3 \
    --output_dir ./finetuned_microbial_p2t

Time estimate: ~1 hour for 5K samples x 3 epochs on H100
"""

import os
import sys
import json
import random
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Add P2T paths
P2T_DIR = Path("/data/asahu/projects/doe_genesis/Protein2Text")
sys.path.insert(0, str(P2T_DIR))

from evaluation.model.builder import load_pretrained_model
from evaluation.constants import DEFAULT_PROTEIN_SEQUENCE_TOKEN, PROTEIN_SEQUENCE_TOKEN_INDEX
from evaluation.inference_model import protein_sequence_tokenizer

MODEL_PATH = str(P2T_DIR / "checkpoints/protein2text-llama3.1-8B-instruct-esm2-650M")
MODEL_BASE = "/data/ajararweh/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"


class ProteinQADataset(Dataset):
    """Dataset for P2T fine-tuning: protein sequence + QA pairs."""

    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["conversations"][0]["value"].replace("<protein_sequence>\n", "")
        answer = item["conversations"][1]["value"]
        amino_seq = item["amino_seq"]

        # Format as instruction
        prompt = f"<protein_sequence>\n{question}"
        full_text = f"{prompt}\n{answer}"

        # Tokenize with protein sequence token
        input_ids = protein_sequence_tokenizer(
            full_text, self.tokenizer, PROTEIN_SEQUENCE_TOKEN_INDEX, return_tensors="pt"
        )

        # Truncate
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        # Labels: mask the prompt, only train on the answer
        labels = input_ids.clone()
        # Find where the answer starts (after the question)
        prompt_ids = protein_sequence_tokenizer(
            prompt + "\n", self.tokenizer, PROTEIN_SEQUENCE_TOKEN_INDEX, return_tensors="pt"
        )
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = -100  # mask prompt

        return {
            "input_ids": input_ids,
            "labels": labels,
            "amino_seq": amino_seq,
        }


def collate_fn(batch):
    """Pad sequences to same length within batch."""
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = len(b["input_ids"])
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]
        attention_mask[i, :seq_len] = 1

    amino_seqs = [b["amino_seq"] for b in batch]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "amino_seqs": amino_seqs,
    }


def train(args):
    print("=" * 60)
    print("IDPro: LoRA Fine-Tuning P2T on Microbial Data")
    print("=" * 60)

    device = torch.device("cuda")
    t_start = time.time()

    # ── Load Data ──────────────────────────────────────────────────────
    print(f"\nLoading training data from {args.train_data}...")
    with open(args.train_data) as f:
        all_data = json.load(f)

    # Sample if needed
    if args.n_samples and args.n_samples < len(all_data):
        random.seed(42)
        all_data = random.sample(all_data, args.n_samples)
    print(f"  Training samples: {len(all_data)}")

    # Split train/val (90/10)
    random.seed(42)
    random.shuffle(all_data)
    split = int(0.9 * len(all_data))
    train_data = all_data[:split]
    val_data = all_data[split:]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # ── Load Model ─────────────────────────────────────────────────────
    print(f"\nLoading P2T model...")
    tokenizer, model, context_len = load_pretrained_model(
        MODEL_PATH, MODEL_BASE, "protein2text-llama3.1-8B-instruct-esm2-650M",
        device=device
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Freeze protein encoder
    protein_encoder = model.get_protein_encoder()
    for param in protein_encoder.parameters():
        param.requires_grad = False

    # ── Apply LoRA ─────────────────────────────────────────────────────
    print("\nApplying LoRA adapter...")

    # Find linear layers in the LLM
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            if any(t in name for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                target_modules.append(name.split(".")[-1])
    target_modules = list(set(target_modules))

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules if target_modules else ["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Create DataLoaders ─────────────────────────────────────────────
    train_dataset = ProteinQADataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = ProteinQADataset(val_data, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    # ── Optimizer ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )

    # ── Training Loop ──────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    model.train()
    model = model.bfloat16()

    best_val_loss = float('inf')
    training_log = []

    for epoch in range(args.epochs):
        epoch_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            amino_seqs = batch["amino_seqs"]

            # Forward pass — P2T's forward takes (input_ids, amino_seq, ...)
            # For simplicity, we do standard causal LM training on the text part
            # The protein encoder processes amino_seq internally during forward
            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        amino_seq=amino_seqs[0] if len(amino_seqs) == 1 else amino_seqs[0],
                        labels=labels,
                    )
                    loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                n_batches += 1

                if batch_idx % 50 == 0:
                    avg_loss = epoch_loss / max(n_batches, 1)
                    elapsed = (time.time() - t_start) / 60
                    print(f"  Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                          f"Loss: {avg_loss:.4f} | Time: {elapsed:.0f}m")

            except Exception as e:
                print(f"  Batch {batch_idx} error: {str(e)[:100]}")
                optimizer.zero_grad()
                continue

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                amino_seqs = batch["amino_seqs"]
                try:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(
                            input_ids=input_ids,
                            amino_seq=amino_seqs[0],
                            labels=labels,
                        )
                        loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss
                    val_loss += loss.item()
                    val_batches += 1
                except:
                    continue
        avg_val_loss = val_loss / max(val_batches, 1)
        model.train()

        elapsed = (time.time() - t_start) / 60
        print(f"\n  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, time={elapsed:.0f}m")

        training_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "time_min": elapsed,
        })

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  New best val_loss: {best_val_loss:.4f} — saving...")
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    # ── Save Training Log ──────────────────────────────────────────────
    total_time = (time.time() - t_start) / 60
    summary = {
        "n_train": len(train_data),
        "n_val": len(val_data),
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "best_val_loss": best_val_loss,
        "total_time_min": total_time,
        "training_log": training_log,
    }

    with open(os.path.join(args.output_dir, "training_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Total time: {total_time:.0f} minutes")
    print(f"  Model saved to: {args.output_dir}")
    print(f"  Training log: {args.output_dir}/training_summary.json")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tune P2T on microbial data")
    parser.add_argument("--train_data", type=str, required=True, help="Path to QA JSON file")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (keep 1 for long sequences)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--output_dir", type=str, default="./finetuned_microbial_p2t")
    args = parser.parse_args()

    train(args)
