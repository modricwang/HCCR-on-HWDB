#!/usr/bin/env bash

mkdir checkpoints

python -u main.py \
    -shuffle \
    -train_record \
    -model resnet152 \
    -data_dir ./data \
    -save_path checkpoints \
    -color_classes 10 \
    -type_classes 7 \
    -n_epochs 150 \
    -learn_rate 0.003 \
    -pretrained ./pretrain/resnet152-b121ed2d.pth \
    -batch_size 64 \
    -workers 32 \
    -nGPU 1 \
2>&1 | tee train.log
