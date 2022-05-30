#!/usr/bin/env bash

set -euo pipefail

python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
  --dataset_config configs/pretrain_ja.json \
  --ema \
  --text_encoder_type xlm-roberta-base \
  --backbone timm_tf_efficientnet_b3_ns \
  --lr_backbone 5e-5 \
  --freeze_text_encoder \
  --batch_size 2 \
  --output_dir ./result2 \
  --load ./result2/pretrained_ja_flickr_1e.pth \
  --eval
