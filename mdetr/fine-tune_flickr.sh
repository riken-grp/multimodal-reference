#!/usr/bin/env bash

set -euo pipefail

# settings on moss110
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
  --dataset_config configs/flickr.json \
  --ema \
  --text_encoder_type xlm-roberta-base \
  --backbone timm_tf_efficientnet_b3_ns \
  --lr_backbone 5e-5 \
  --freeze_text_encoder \
  --batch_size 1 \
  --epochs 1 \
  --output_dir ./result3 \
  --load ./result/pretrained_ja_flickr_mixed_1e.pth
