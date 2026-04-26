#!/bin/bash

# exp_2 自动运行脚本
# 监控并自动启动 j2 → j3 → j4
# 每个任务: cascade → mask

WORK_DIR="work_dirs"
LOG_DIR="logs"
RESULT_DIR="results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

declare -A GPU_ASSIGN=(
    ["j2"]="0,1"
    ["j3"]="2,3"
    ["j4"]="4,5"
)

declare -A CONFIGS=(
    ["j2_det"]="cascade-rcnn_r50_fpn_2x_ruod_j2.py"
    ["j2_mask"]="mask-rcnn_r50_fpn_2x_ruod_j2_mask.py"
    ["j3_det"]="cascade-rcnn_vit-base_mae_fpn_2x_ruod_j3.py"
    ["j3_mask"]="mask-rcnn_vit-base_mae_fpn_2x_ruod_j3_mask.py"
    ["j4_det"]="cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py"
    ["j4_mask"]="mask-rcnn_r50_dino_fpn_2x_ruod_j4_mask.py"
)

check_gpu_available() {
    local gpus=$1
    local available=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpus | awk '{sum+=$1} END {print sum}')
    echo $available
}

run_task() {
    local task=$1
    local config=${CONFIGS[$task]}
    local gpu=${GPU_ASSIGN[${task%%_*}]}
    local log=$LOG_DIR/${task}.log
    
    echo ">>> 启动: $task on GPU $gpu"
    mkdir -p $WORK_DIR/$task
    
    # 等待GPU空闲
    while true; do
        local free=$(check_gpu_available $gpu)
        if [ "$free" -gt 15000 ]; then
            break
        fi
        echo "等待GPU $gpu 空闲... (free: ${free}MB)"
        sleep 30
    done
    echo "GPU $gpu 可用，开始训练..."
    
    export PORT=29500
    CUDA_VISIBLE_DEVICES=$gpu bash tools/dist_train.sh \
        configs/exp_2/$config \
        2 \
        --work-dir $WORK_DIR/$task \
        --cfg-options model.init_cfg=None \
        2>&1 | tee "$log"
    
    echo "<<< $task 完成"
}

echo "========================================="
echo "exp_2 自动训练流程"
echo "========================================="
echo "开始时间: $(date)"

# J2
echo "===== J2 (GPU 0,1) =====" 
run_task "j2_det"
run_task "j2_mask"
echo "J2 完成!"

# J3
echo "===== J3 (GPU 2,3) ====="
run_task "j3_det"
run_task "j3_mask"
echo "J3 完成!"

# J4
echo "===== J4 (GPU 4,5) ====="
run_task "j4_det"
run_task "j4_mask"
echo "J4 完成!"

echo "========================================="
echo "全部完成! $(date)"
echo "========================================="