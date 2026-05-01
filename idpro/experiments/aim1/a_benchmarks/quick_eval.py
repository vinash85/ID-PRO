#!/usr/bin/env python3
"""Quick standalone inference to test eval fixes. Runs on a separate GPU."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_GPUS", "3")

import json, re, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from idpro.config import IDProConfig, EncoderConfig, LLMConfig
from idpro.model import IDProModel
from idpro.paths import CKPT_DIR as _CKPT_ROOT, BENCHMARK
import torch.nn as nn

CKPT_DIR = str(_CKPT_ROOT / "robust")
BENCHMARK = str(BENCHMARK)
TRAIN_Q = "Analyze this protein sequence: identify its domains, motifs, spatial arrangement, and predict its function."

device = "cuda"

# Load model
print("Loading model...")
config = IDProConfig(
    encoder=EncoderConfig(name="esmc-600m"),
    llm=LLMConfig(name="qwen3.5-27b"),
)
config.resolve()
model = IDProModel(config)
model.encoder.load(device)
config.llm.use_qlora = True
model.load_llm(device=device)

dtype = torch.bfloat16
model.adaptor = model.adaptor.to(device=device, dtype=dtype)
model.projector = model.projector.to(device=device, dtype=dtype)
model.evidence_head_pre = model.evidence_head_pre.to(device=device, dtype=dtype)
model.evidence_head_post = model.evidence_head_post.to(device=device, dtype=dtype)
model.protein_position = model.protein_position.to(device=device, dtype=dtype)
model.protein_modality_embed = nn.Parameter(model.protein_modality_embed.data.to(device=device, dtype=dtype))
model.prot_end_embed = nn.Parameter(model.prot_end_embed.data.to(device=device, dtype=dtype))

# Load checkpoint
latest = os.readlink(f"{CKPT_DIR}/latest")
tp = f"{CKPT_DIR}/{latest}/trainable.pt"
trainable = torch.load(tp, map_location=device)
for name, param in model.named_parameters():
    if name in trainable and param.requires_grad:
        param.data.copy_(trainable[name].to(device))
print(f"Loaded checkpoint: {latest}")

# Load eval data
with open(BENCHMARK) as f:
    bench = json.load(f)

stop = {"the","a","an","is","are","was","of","in","to","and","or","this","that",
        "it","for","with","on","at","by","from","as","protein","proteins","its","has",
        "based","sequence","identify","locate","relate","infer","contextualize",
        "key","residues","position","positions","domain","found"}

def extract_infer(text):
    m = re.search(r'\[INFER\]\s*(.*?)(?:\[CONTEXTUALIZE\]|\[|$)', text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r'\[RELATE\]\s*(.*?)(?:\[INFER\]|\[|$)', text, re.DOTALL)
    if m: return m.group(1).strip()
    return text

def kw_f1(pred, gt):
    pw = set(re.findall(r'[a-z]{3,}', pred.lower())) - stop
    tw = set(re.findall(r'[a-z]{3,}', gt.lower())) - stop
    if not tw or not pw: return 0.0
    ov = len(pw & tw)
    p, r = ov/len(pw), ov/len(tw)
    return 2*p*r / max(p+r, 1e-10)

# Run inference on 20 samples
model.eval()
import random
random.seed(42)
samples = random.sample(bench, 20)

f1_full, f1_infer = [], []
print(f"\nRunning inference on {len(samples)} proteins (max_new_tokens=500)...\n")

for i, p in enumerate(samples):
    seq = p["amino_seq"][:500]
    gt = p["conversations"][1]["value"]

    try:
        preds = model.generate(
            sequences=[seq],
            questions=[TRAIN_Q],
            max_new_tokens=500,
            temperature=0.3,
            device=device,
        )
        pred = preds[0] if preds else ""
    except Exception as e:
        print(f"  ERROR: {e}")
        pred = ""

    f1f = kw_f1(pred, gt)
    infer_text = extract_infer(pred)
    f1i = kw_f1(infer_text, gt)
    f1_full.append(f1f)
    f1_infer.append(f1i)

    print(f"[{i+1}/{len(samples)}] F1_full={f1f:.3f} F1_infer={f1i:.3f}")
    if i < 5:
        print(f"  GT:    {gt[:100]}")
        print(f"  PRED:  {pred[:200]}")
        print(f"  INFER: {infer_text[:150]}")
        print()

print(f"\n{'='*50}")
print(f"Results ({len(samples)} proteins, checkpoint {latest}):")
print(f"  F1 (full CoT): {sum(f1_full)/len(f1_full):.4f}")
print(f"  F1 (INFER):    {sum(f1_infer)/len(f1_infer):.4f}")
print(f"  Non-zero F1:   {sum(1 for f in f1_full if f > 0)}/{len(f1_full)}")
print(f"{'='*50}")
