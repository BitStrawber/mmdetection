#!/bin/bash

# j2_mask 训练脚本 - 带GPU占位
# GPU: 0,1

WORK_DIR="work_dirs"
NUM_GPUS=2
GPU_IDS="0,1"
PORT=29502
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

# 占位函数
occupy_gpu() {
    local gpu=$1
    echo "开始占位 GPU $gpu..."
    # 后台占位
    CUDA_VISIBLE_DEVICES=$gpu nohup python -c "
import torch,time
for i in 0 1:
    x = torch.zeros(2000,2000,1000, device=f'cuda:{i}')
while True: time.sleep(60)
" > /dev/null 2>&1 &
}

echo "========================================="
echo "j2_mask (ResNet-50 + torchvision)"
echo "========================================="
echo "GPU: $GPU_IDS"
echo ""

# 占住GPU
occupy_gpu $GPU_IDS

echo ">>> 启动: j2_mask"
mkdir -p $WORK_DIR/j2_mask

export PORT
CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
    configs/exp_2/mask-rcnn_r50_fpn_2x_ruod_j2_mask.py \
    $NUM_GPUS \
    --work-dir $WORK_DIR/j2_mask \
    2>&1 | tee "$LOG_DIR/j2_mask.log"

echo "<<< j2_mask 完成"

# 释放占位
echo "释放GPU..."
pkill -f "torch.zeros"