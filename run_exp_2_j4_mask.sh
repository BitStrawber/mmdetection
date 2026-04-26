#!/bin/bash

# j4_mask 训练脚本
# GPU: 4,5

WORK_DIR="work_dirs"
NUM_GPUS=2
GPU_IDS="4,5"
PORT=29506
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "========================================="
echo "j4_mask (ResNet-50 + DINO)"
echo "========================================="
echo "GPU: $GPU_IDS"
echo ""

echo ">>> 启动: j4_mask"
mkdir -p $WORK_DIR/j4_mask

export PORT
CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
    configs/exp_2/mask-rcnn_r50_dino_fpn_2x_ruod_j4_mask.py \
    $NUM_GPUS \
    --work-dir $WORK_DIR/j4_mask \
    2>&1 | tee "$LOG_DIR/j4_mask.log"

echo "<<< j4_mask 完成"