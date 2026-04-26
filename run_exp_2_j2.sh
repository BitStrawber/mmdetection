#!/bin/bash

# exp_2_j2 训练脚本 (ViT-Base + MAE)
# 自动顺序执行: cascade → mask
# GPU: 2,3

CONFIG_DIR="configs/exp_2"
PORT=29510
WORK_DIR="work_dirs"
SCRIPT_LOG="train_j2.log"
GPUS="2,3"

echo "========================================="
echo "exp_2_j2 (ViT-Base + MAE)"
echo "========================================="
echo "GPU: $GPUS"
echo "任务: cascade → mask"
echo ""

run_task() {
    local name=$1
    local config=$2
    local log=$WORK_DIR/$name/training.log
    echo ""
    echo ">>> 启动: $name"
    echo "GPU: $GPUS"
    mkdir -p $WORK_DIR/$name
    
    # 等待GPU空闲
    echo "等待GPU $GPUS 空闲..."
    while true; do
        available=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPUS | awk '{sum+=$1} END {print sum}')
        if [ "$available" -gt 20000 ]; then
            break
        fi
        sleep 5
    done
    echo "GPU $GPUS 可用，开始训练..."
    
    env CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=$PORT \
        tools/train.py $CONFIG_DIR/$config \
        --cfg-options work_dir=$WORK_DIR/$name > $log 2>&1 &
    local pid=$!
    PORT=$((PORT+1))
    wait $pid
    echo "<<< $name 完成 (exit code: $?)"
}

echo "开始时间: $(date)" | tee -a $SCRIPT_LOG

echo "===== Cascade Detection =====" | tee -a $SCRIPT_LOG
run_task "j3_det"    "cascade-rcnn_vit-base_mae_fpn_2x_ruod_j3.py"

echo "===== Mask =====" | tee -a $SCRIPT_LOG
run_task "j3_mask"  "mask-rcnn_vit-base_mae_fpn_2x_ruod_j3_mask.py"

echo "所有任务完成!" | tee -a $SCRIPT_LOG
echo "结束时间: $(date)" | tee -a $SCRIPT_LOG