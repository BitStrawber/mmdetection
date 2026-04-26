#!/bin/bash

# exp_2_j3 训练脚本 (ResNet-50 + DINO)
# 自动顺序执行: cascade → mask
# GPU: 4,5

CONFIG_DIR="configs/exp_2"
PORT=29520
WORK_DIR="work_dirs"
SCRIPT_LOG="train_j3.log"

declare -A PIDS

echo "========================================="
echo "exp_2_j3 (ResNet-50 + DINO)"
echo "========================================="
echo "GPU: 4,5"
echo "任务: cascade → mask"
echo ""

run_task() {
    local name=$1
    local config=$2
    local gpus=$3
    local log=$WORK_DIR/$name/training.log
    echo ""
    echo ">>> 启动: $name"
    echo "GPU: $gpus"
    mkdir -p $WORK_DIR/$name
    (CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=$PORT \
        tools/train.py $CONFIG_DIR/$config \
        --cfg-options work_dir=$WORK_DIR/$name > $log 2>&1) &
    PIDS[$name]=$!
    PORT=$((PORT+1))
}

wait_all() {
    for name in "${!PIDS[@]}"; do
        wait ${PIDS[$name]}
        echo "<<< $name 完成 (exit code: $?)"
    done
}

echo "开始时间: $(date)" | tee -a $SCRIPT_LOG

echo "===== Cascade Detection =====" | tee -a $SCRIPT_LOG
run_task "j4_det"    "cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py"    "4,5"
wait_all

echo "===== Mask =====" | tee -a $SCRIPT_LOG
run_task "j4_mask"  "mask-rcnn_r50_dino_fpn_2x_ruod_j4_mask.py"  "4,5"
wait_all

echo "所有任务完成!" | tee -a $SCRIPT_LOG
echo "结束时间: $(date)" | tee -a $SCRIPT_LOG