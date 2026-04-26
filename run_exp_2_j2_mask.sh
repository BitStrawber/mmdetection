#!/bin/bash

# j2_mask 训练脚本
# GPU: 0,1

WORK_DIR="work_dirs"
NUM_GPUS=2
GPU_IDS="0,1"
PORT=29502
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "========================================="
echo "j2_mask (ResNet-50 + torchvision)"
echo "========================================="
echo "GPU: $GPU_IDS"
echo ""

echo ">>> 启动: j2_mask"
mkdir -p $WORK_DIR/j2_mask

export PORT
CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
    configs/exp_2/mask-rcnn_r50_fpn_2x_ruod_j2_mask.py \
    $NUM_GPUS \
    --work-dir $WORK_DIR/j2_mask \
    2>&1 | tee "$LOG_DIR/j2_mask.log"

echo "<<< j2_mask 完成"