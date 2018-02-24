#!/usr/bin/env bash

mkdir checkpoints

python -u main.py \
    -shuffle \
    -train_record \
    -model resnet18 \
    -data_dir ./data \
    -save_path checkpoints \
    -output_classes 3755 \
    -n_epochs 150 \
    -learn_rate 0.01 \
    -pretrained ./pretrain/resnet18-5c106cde.pth \
    -batch_size 16 \
    -workers 0 \
    -nGPU 1 \
2>&1 | tee train.log
