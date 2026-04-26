#!/bin/bash

# GPU Watcher - 后台监测并自动启动训练
# 用法: nohup bash run_exp_2_watcher.sh > watcher.log 2>&1 &

WORK_DIR="work_dirs"
LOG_DIR="logs"
declare -A GPU_ASSIGN=(["j2"]="0,1" ["j3"]="2,3" ["j4"]="4,5")
declare -A CONFIGS=(
    ["j2_det"]="cascade-rcnn_r50_fpn_2x_ruod_j2.py" ["j2_mask"]="mask-rcnn_r50_fpn_2x_ruod_j2_mask.py"
    ["j3_det"]="cascade-rcnn_vit-base_mae_fpn_2x_ruod_j3.py" ["j3_mask"]="mask-rcnn_vit-base_mae_fpn_2x_ruod_j3_mask.py"
    ["j4_det"]="cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py" ["j4_mask"]="mask-rcnn_r50_dino_fpn_2x_ruod_j4_mask.py"
)

check_task() {
    pgrep -f "configs/exp_2/.*${1}.*.py" >/dev/null 2>&1
}

get_gpu_free() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1 | awk '{sum+=$1} END {print sum}'
}

run_task() {
    local task=$1
    local name=${task%%_*}
    local gpu=${GPU_ASSIGN[$name]}
    echo "[$(date)] 启动 $task on GPU $gpu"
    mkdir -p $WORK_DIR/$task
    CUDA_VISIBLE_DEVICES=$gpu nohup bash tools/dist_train.sh configs/exp_2/${CONFIGS[$task]} 2 --work-dir $WORK_DIR/$task --cfg-options model.init_cfg=None > $LOG_DIR/${task}.log 2>&1 &
    wait
}

echo "[$(date)] GPU Watcher 启动"
mkdir -p "$LOG_DIR"

while true; do
    free_j2=$(get_gpu_free 0,1)
    free_j3=$(get_gpu_free 2,3)
    free_j4=$(get_gpu_free 4,5)
    
    echo "[$(date)] GPU: j2=$free_j2 j3=$free_j3 j4=$free_j4"
    
    [ $free_j2 -gt 15000 ] && ! check_task j2_det && run_task j2_det
    [ $free_j2 -gt 15000 ] && ! check_task j2_mask && run_task j2_mask
    [ $free_j3 -gt 15000 ] && ! check_task j3_det && run_task j3_det
    [ $free_j3 -gt 15000 ] && ! check_task j3_mask && run_task j3_mask
    [ $free_j4 -gt 15000 ] && ! check_task j4_det && run_task j4_det
    [ $free_j4 -gt 15000 ] && ! check_task j4_mask && run_task j4_mask
    
    sleep 60
done