#!/bin/bash

# GPU Occupier - 监测并自动占用GPU
# 用法: bash run_exp_2_occupier.sh [j2|j3|j4|all]
# 例如: bash run_exp_2_occupier.sh j3

WORK_DIR="work_dirs"
LOG_DIR="logs"

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

get_gpu_free() {
    local gpus=$1
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpus | awk '{sum+=$1} END {print sum}'
}

wait_gpu() {
    local name=$1
    local gpu=${GPU_ASSIGN[$name]}
    local threshold=15000
    
    echo "[$name] 等待GPU $gpu 可用..."
    while true; do
        local free=$(get_gpu_free $gpu)
        if [ "$free" -gt $threshold ]; then
            echo "[$name] GPU $gpu 可用! (free: ${free}MB)"
            return 0
        fi
        echo "[$name] GPU $gpu 忙碌中... (free: ${free}MB, 需要: ${threshold}MB)"
        sleep 20
    done
}

occupy_and_run() {
    local task=$1
    local config=${CONFIGS[$task]}
    local name=${task%%_*}
    local gpu=${GPU_ASSIGN[$name]}
    local log=$LOG_DIR/${task}.log
    
    wait_gpu $name
    
    echo "[$task] 启动训练..."
    mkdir -p $WORK_DIR/$task
    
    export PORT=29500
    CUDA_VISIBLE_DEVICES=$gpu nohup bash tools/dist_train.sh \
        configs/exp_2/$config \
        2 \
        --work-dir $WORK_DIR/$task \
        --cfg-options model.init_cfg=None \
        > $log 2>&1 &
    
    echo "[$task] PID: $! 已启动 (GPU: $gpu)"
    echo "[$task] 日志: $log"
    
    # 等待训练完成
    wait
    echo "[$task] 完成!"
}

run_j() {
    local j=$1
    local name="j${j}"
    local gpu=${GPU_ASSIGN[$name]}
    
    echo "========================================="
    echo "运行 J${j} (GPU: $gpu)"
    echo "========================================="
    
    occupy_and_run "${name}_det"
    occupy_and_run "${name}_mask"
    
    echo "J${j} 全部完成!"
}

TASK=${1:-all}

echo "========================================="
echo "GPU Occupier - $(date)"
echo "========================================="
echo "任务: $TASK"
echo "可用: j2 j3 j4 all"
echo ""

case $TASK in
    j2) run_j 2 ;;
    j3) run_j 3 ;;
    j4) run_j 4 ;;
    all)
        run_j 2
        run_j 3
        run_j 4
        echo "所有任务完成!"
        ;;
    *) echo "未知任务: $TASK" ;;
esac

echo "结束: $(date)"