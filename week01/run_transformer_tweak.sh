#!/bin/bash
# contains the optimal hyperparameters for tweaked Transformer run

python main.py \
  --arch TransformerTweak \
  --epochs 10 \
  --batch_size 32 \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --beta1 0.9 --beta2 0.98 \
  --warmup_ratio 0.06 \
  --min_lr 1e-5 \
  --max_len 256 \
  --max_vocab_size 20000 \
  --min_freq 2 \
  --embed_size 256 \
  --n_layers 6 \
  --n_head 4 \
  --n_ff 1024 \
  --dropout 0.2 \
  --seed 0
