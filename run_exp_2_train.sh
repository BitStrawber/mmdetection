#!/bin/bash

# exp_2 训练脚本
# 使用方法: bash run_exp_2_train.sh [task]
# Log保存在: work_dirs/<任务名>/

CONFIG_DIR="configs/exp_2"
PORT=29500
WORK_DIR="work_dirs"
SCRIPT_LOG="train_all.log"

declare -A PIDS

echo "========================================="
echo "exp_2 训练脚本"
echo "========================================="
echo "使用方法: bash run_exp_2_train.sh [task]"
echo ""
echo "可用任务: cascade mask all"
echo ""
echo "脚本Log: $SCRIPT_LOG"
echo "任务Log: $WORK_DIR/<任务名>/training.log"
echo ""

run_parallel() {
    local name=$1
    local config=$2
    local gpus=$3
    local log=$WORK_DIR/$name/training.log
    echo ""
    echo ">>> 启动: $name ($config)"
    echo "GPU: $gpus"
    echo "Log: $log"
    mkdir -p $WORK_DIR/$name
    (CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=$PORT \
        tools/train.py $CONFIG_DIR/$config \
        --cfg-options work_dir=$WORK_DIR/$name > $log 2>&1) &
    PIDS[$name]=$!
    echo "PID: ${PIDS[$name]}"
    PORT=$((PORT+1))
}

wait_all() {
    echo "等待所有任务完成..."
    for name in "${!PIDS[@]}"; do
        echo "等待 $name (PID: ${PIDS[$name]})"
        wait ${PIDS[$name]}
        echo "<<< $name 完成 (exit code: $?)"
    done
}

TASK=${1:-all}

echo "开始时间: $(date)" | tee -a $SCRIPT_LOG

case $TASK in
    cascade)
        echo "===== 第一批: Cascade Detection (并行) =====" | tee -a $SCRIPT_LOG
        run_parallel "j2_det"    "cascade-rcnn_r50_fpn_2x_ruod_j2.py"        "0,1"
        run_parallel "j3_det"    "cascade-rcnn_vit-base_mae_fpn_2x_ruod_j3.py"   "2,3"
        run_parallel "j4_det"    "cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py"    "4,5"
        wait_all
        echo "第一批完成!" | tee -a $SCRIPT_LOG ;;
    mask)
        echo "===== 第二批: Mask (并行) =====" | tee -a $SCRIPT_LOG
        run_parallel "j2_mask"  "mask-rcnn_r50_fpn_2x_ruod_j2_mask.py"     "0,1"
        run_parallel "j3_mask"  "mask-rcnn_vit-base_mae_fpn_2x_ruod_j3_mask.py" "2,3"
        run_parallel "j4_mask"  "mask-rcnn_r50_dino_fpn_2x_ruod_j4_mask.py"  "4,5"
        wait_all
        echo "第二批完成!" | tee -a $SCRIPT_LOG ;;
    all)
        echo "===== 第一批: Cascade Detection (并行) =====" | tee -a $SCRIPT_LOG
        run_parallel "j2_det"    "cascade-rcnn_r50_fpn_2x_ruod_j2.py"        "0,1"
        run_parallel "j3_det"    "cascade-rcnn_vit-base_mae_fpn_2x_ruod_j3.py"   "2,3"
        run_parallel "j4_det"    "cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py"    "4,5"
        wait_all
        echo "第一批完成!" | tee -a $SCRIPT_LOG
        
        echo "===== 第二批: Mask (并行) =====" | tee -a $SCRIPT_LOG
        run_parallel "j2_mask"  "mask-rcnn_r50_fpn_2x_ruod_j2_mask.py"     "0,1"
        run_parallel "j3_mask"  "mask-rcnn_vit-base_mae_fpn_2x_ruod_j3_mask.py" "2,3"
        run_parallel "j4_mask"  "mask-rcnn_r50_dino_fpn_2x_ruod_j4_mask.py"  "4,5"
        wait_all
        echo "第二批完成!" | tee -a $SCRIPT_LOG
        echo "所有任务完成!" | tee -a $SCRIPT_LOG ;;
    *) echo "未知任务: $TASK" | tee -a $SCRIPT_LOG ;;
esac

echo "结束时间: $(date)" | tee -a $SCRIPT_LOG