#!/usr/bin/env bash
# Official Facebook DINO 100-epoch protocol used for the ImageNet, RealUW,
# Synthetic5 and controlled-100K ViT-S/16 experiments.

export DINO_ARCH=vit_small
export DINO_OPTIMIZER=adamw
export DINO_LR=0.0005
export DINO_WEIGHT_DECAY=0.04
export DINO_WEIGHT_DECAY_END=0.4
export DINO_WARMUP_EPOCHS=10
export DINO_MIN_LR=1e-6
export DINO_MOMENTUM_TEACHER=0.996
export DINO_FREEZE_LAST_LAYER=1
export DINO_GLOBAL_CROPS_SCALE='0.4 1'
export DINO_LOCAL_CROPS_NUMBER=8
export DINO_LOCAL_CROPS_SCALE='0.05 0.4'
