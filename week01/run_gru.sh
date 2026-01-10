#!/bin/bash
# contains the optimal hyperparameters for GRU run

python main.py \
  --arch GRU \
  --epochs 10 \
  --batch_size 64 \
  --lr 1e-3 \
  --weight_decay 0.0 \
  --beta1 0.9 --beta2 0.99 \
  --max_len 256 \
  --max_vocab_size 20000 \
  --min_freq 2 \
  --embed_size 200 \
  --hidden_size 256 \
  --n_layers 2 \
  --dropout 0.3 \
  --warmup_ratio 0.0 \
  --min_lr 0.0 \
  --seed 0