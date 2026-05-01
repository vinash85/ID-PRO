#!/bin/bash
# Setup Qwen3.5-27B + DeepSpeed for multi-GPU IDPro training
# Run this in a fresh terminal / claude session

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/env.sh"
cd "$REPO_ROOT"

# IDPRO_HF_CACHE: where to download model weights. Defaults to $IDPRO_RUNS_ROOT/hf_cache.
export HF_CACHE="${IDPRO_HF_CACHE:-$IDPRO_RUNS_ROOT/hf_cache}"
mkdir -p "$HF_CACHE"

echo "============================================"
echo "IDPro: Setting up Qwen3.5-27B + Multi-GPU"
echo "============================================"

# 1. Create dedicated environment
echo ""
echo "[1/5] Creating conda environment..."
conda create -n idpro_env python=3.10 -y 2>/dev/null || echo "Env exists"
conda activate idpro_env 2>/dev/null || source activate idpro_env

# 2. Install dependencies
echo ""
echo "[2/5] Installing dependencies..."
pip install --upgrade pip
pip install torch>=2.1.2 transformers>=4.43.0 accelerate>=0.34.0 \
    peft>=0.13.0 deepspeed>=0.14.0 \
    sentencepiece einops einops_exts \
    datasets trl \
    numpy pyyaml requests tqdm \
    bitsandbytes  # for QLoRA fallback

# Install ESM C
pip install esm

# Install flash-attention (optional, for speed)
pip install flash-attn --no-build-isolation 2>/dev/null || echo "Flash attention not available"

# 3. Download Qwen3.5-27B
echo ""
echo "[3/5] Downloading Qwen3.5-27B model..."
python -c "
from huggingface_hub import snapshot_download
import os

model_id = 'Qwen/Qwen3.5-27B'
cache_dir = os.environ['HF_CACHE']
os.makedirs(cache_dir, exist_ok=True)

print(f'Downloading {model_id}...')
path = snapshot_download(
    model_id,
    cache_dir=cache_dir,
    ignore_patterns=['*.gguf', '*.onnx'],  # skip non-pytorch formats
)
print(f'Downloaded to: {path}')
"

# 4. Download ESM C 600M
echo ""
echo "[4/5] Downloading ESM C 600M..."
python -c "
from huggingface_hub import snapshot_download
import os

model_id = 'EvolutionaryScale/esmc-600m-2024-12'
cache_dir = os.environ['HF_CACHE']

print(f'Downloading {model_id}...')
path = snapshot_download(model_id, cache_dir=cache_dir)
print(f'Downloaded to: {path}')
"

# 5. Create DeepSpeed config
echo ""
echo "[5/5] Creating DeepSpeed config..."
cat > idpro/training/configs/ds_zero2.json << 'EOF'
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": false
}
EOF

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "To train:"
echo "  deepspeed --num_gpus=4 idpro/training/train.py \\"
echo "    --deepspeed idpro/training/configs/ds_zero2.json \\"
echo "    --encoder esmc-600m \\"
echo "    --llm qwen3.5-27b"
echo "============================================"
