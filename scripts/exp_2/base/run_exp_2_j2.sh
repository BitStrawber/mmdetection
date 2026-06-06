#!/bin/bash

# exp_2_j2 训练脚本 (ResNet-50 + torchvision)
# 自动顺序执行: cascade → mask
# GPU: 0,1

WORK_DIR="work_dirs"
NUM_GPUS=2
GPU_IDS="0,1"
PORT=29502
LOG_DIR="logs"
RESULT_DIR="results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

echo "========================================="
echo "exp_2_j2 (ResNet-50 + torchvision)"
echo "========================================="
echo "GPU: $GPU_IDS"
echo "任务: cascade → mask"
echo ""

run_task() {
    local name=$1
    local config=$2
    local log=$LOG_DIR/${name}.log
    
    echo ""
    echo ">>> 启动: $name"
    echo "GPU: $GPU_IDS"
    mkdir -p $WORK_DIR/$name
    
    export PORT
    CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
        configs/exp_2/$config \
        $NUM_GPUS \
        --work-dir $WORK_DIR/$name \
        --cfg-options model.init_cfg=None \
        2>&1 | tee "$log"
    
    echo "<<< $name 完成"
}

echo "开始时间: $(date)"

echo "===== Cascade Detection ====="
run_task "j2_det" "cascade-rcnn_r50_fpn_2x_ruod_j2.py"

echo "===== Mask ====="
run_task "j2_mask" "mask-rcnn_r50_fpn_2x_ruod_j2_mask.py"

echo "所有任务完成!"
echo "结束时间: $(date)"