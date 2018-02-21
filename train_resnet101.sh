#!/usr/bin/env bash

mkdir checkpoints

python -u main.py \
    -shuffle \
    -train_record \
    -model resnet101 \
    -data_dir ./data \
    -save_path checkpoints \
    -output_classes 3755 \
    -n_epochs 150 \
    -learn_rate 0.003 \
    -pretrained ./pretrain/resnet101-5d3b4d8f.pth \
    -batch_size 64 \
    -workers 32 \
    -nGPU 1 \
2>&1 | tee train.log
