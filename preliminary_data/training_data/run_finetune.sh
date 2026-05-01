#!/bin/bash
# IDPro: LoRA fine-tune P2T on microbial protein data
# Uses the original LLaVA training infrastructure that trained P2T
#
# Estimated time: ~30-60 min for 5K samples x 1 epoch on 1x H100
# This is a PROOF OF CONCEPT fine-tune to show improvement is possible.

export CUDA_VISIBLE_DEVICES=1
export WANDB_DISABLED=true
export PYTHONPATH="/data/ajararweh/Mutation2Text:$PYTHONPATH"

cd /data/ajararweh/Mutation2Text

conda run -n protein2text_env deepspeed --num_gpus=1 llava/train/train_mem_protein.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --protein_encoder facebook/esm2_t33_650M_UR50D \
    --version plain \
    --data_path /data/asahu/projects/doe_genesis/preliminary_data/training_data/uniprot/uniprot_combined_qa.json \
    --output_dir /data/asahu/projects/doe_genesis/preliminary_data/training_data/finetuned_microbial_p2t \
    --pretrain_mm_mlp_adapter /data/asahu/projects/doe_genesis/Protein2Text/checkpoints/protein2text-llama3.1-8B-instruct-esm2-650M/non_lora_trainables.bin \
    --pretrain_mm_resampler /data/asahu/projects/doe_genesis/Protein2Text/checkpoints/protein2text-llama3.1-8B-instruct-esm2-650M/non_lora_trainables.bin \
    --mm_projector_type mlp2x_gelu \
    --num_media_tokens 128 \
    --mm_use_resampler_gca resampler \
    --mm_protein_select_layer -2 \
    --mm_protein_select_feature patch \
    --bf16 True \
    --fp16 False \
    --tf32 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 8e-6 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
