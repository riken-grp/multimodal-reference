#!/usr/bin/env bash

set -euo pipefail

# settings on moss110
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
  --dataset_config configs/pretrain_ja.json \
  --ema \
  --text_encoder_type xlm-roberta-base \
  --backbone timm_tf_efficientnet_b3_ns \
  --lr_backbone 5e-5 \
  --freeze_text_encoder \
  --batch_size 2 \
  --epochs 1 \
  --lr_drop 100 \
  --output_dir ./result2 \
  --load ./pretrained_EB3_checkpoint.pth.zip
