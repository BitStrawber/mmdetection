#!/bin/bash

# exp_2 训练脚本
# 使用方法: bash run_exp_2_train.sh [task]
# Log保存在: work_dirs/<任务名>/

CONFIG_DIR="configs/exp_2"
PORT=29500
WORK_DIR="work_dirs"

echo "========================================="
echo "exp_2 训练脚本"
echo "========================================="
echo "使用方法: bash run_exp_2_train.sh [task]"
echo ""
echo "可用任务: j2_det j3_det j4_det j2_mask j3_mask j4_mask all"
echo ""
echo "Log保存在: $WORK_DIR/<任务名>/"
echo ""

TASK=${1:-all}

run_task() {
    local name=$1
    local config=$2
    local gpus=$3
    echo ""
    echo ">>> 开始: $name ($config)"
    echo "GPU: $gpus"
    echo "Log: $WORK_DIR/$name/training.log"
    CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=$PORT \
        tools/train.py $CONFIG_DIR/$config.py \
        --cfg-options work_dir=$WORK_DIR/$name 2>&1 | tee $WORK_DIR/$name/training.log
    PORT=$((PORT+1))
    echo "<<< 完成: $name"
}

case $TASK in
    j2_det)    run_task "j2_det"    "cascade-rcnn_r50_fpn_2x_ruod_j2.py"        "0,1" ;;
    j3_det)    run_task "j3_det"    "cascade-rcnn_vit-base_mae_fpn_2x_ruod_j3.py"   "2,3" ;;
    j4_det)    run_task "j4_det"    "cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py"    "4,5" ;;
    j2_mask)  run_task "j2_mask"  "mask-rcnn_r50_fpn_2x_ruod_j2_mask.py"     "0,1" ;;
    j3_mask)  run_task "j3_mask"  "mask-rcnn_vit-base_mae_fpn_2x_ruod_j3_mask.py" "2,3" ;;
    j4_mask)  run_task "j4_mask"  "mask-rcnn_r50_dino_fpn_2x_ruod_j4_mask.py"  "4,5" ;;
    all)
        run_task "j2_det"    "cascade-rcnn_r50_fpn_2x_ruod_j2.py"        "0,1"
        run_task "j3_det"    "cascade-rcnn_vit-base_mae_fpn_2x_ruod_j3.py"   "2,3"
        run_task "j4_det"    "cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py"    "4,5"
        run_task "j2_mask"  "mask-rcnn_r50_fpn_2x_ruod_j2_mask.py"     "0,1"
        run_task "j3_mask"  "mask-rcnn_vit-base_mae_fpn_2x_ruod_j3_mask.py" "2,3"
        run_task "j4_mask"  "mask-rcnn_r50_dino_fpn_2x_ruod_j4_mask.py"  "4,5"
        echo "所有任务完成!" ;;
    *) echo "未知任务: $TASK" ;;
esac