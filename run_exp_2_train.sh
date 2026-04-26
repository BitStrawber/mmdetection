#!/bin/bash

# exp_2 训练脚本
# 使用方法: bash run_exp_2_train.sh [task]

CONFIG_DIR="configs/exp_2"
PORT=29500

echo "========================================="
echo "exp_2 训练任务脚本"
echo "========================================="
echo ""
echo "使用方法:"
echo "  bash run_exp_2_train.sh [task]"
echo ""
echo "可用任务:"
echo "  j2_det    - J2 Detection (ResNet-50 Supervised)"
echo "  j3_det   - J3 Detection (ViT-Base MAE)"
echo "  j4_det   - J4 Detection (ResNet-50 DINO)"
echo "  j2_mask  - J2 Mask (ResNet-50 Supervised)"
echo "  j3_mask  - J3 Mask (ViT-Base MAE)"
echo "  j4_mask  - J4 Mask (ResNet-50 DINO)"
echo "  all      - 顺序运行所有任务"
echo ""

TASK=${1:-all}

run_j2_det() {
    echo "[J2 Detection] Cascade R-CNN (ResNet-50 Supervised)"
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT tools/train.py $CONFIG_DIR/cascade-rcnn_r50_fpn_2x_ruod_j2.py
}

run_j3_det() {
    echo "[J3 Detection] Cascade R-CNN (ViT-Base MAE)"
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((PORT+1)) tools/train.py $CONFIG_DIR/cascade-rcnn_vit-base_mae_fpn_2x_ruod_j3.py
}

run_j4_det() {
    echo "[J4 Detection] Cascade R-CNN (ResNet-50 DINO)"
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((PORT+2)) tools/train.py $CONFIG_DIR/cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py
}

run_j2_mask() {
    echo "[J2 Mask] Mask R-CNN (ResNet-50 Supervised)"
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((PORT+3)) tools/train.py $CONFIG_DIR/mask-rcnn_r50_fpn_2x_ruod_j2_mask.py
}

run_j3_mask() {
    echo "[J3 Mask] Mask R-CNN (ViT-Base MAE)"
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((PORT+4)) tools/train.py $CONFIG_DIR/mask-rcnn_vit-base_mae_fpn_2x_ruod_j3_mask.py
}

run_j4_mask() {
    echo "[J4 Mask] Mask R-CNN (ResNet-50 DINO)"
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((PORT+5)) tools/train.py $CONFIG_DIR/mask-rcnn_r50_dino_fpn_2x_ruod_j4_mask.py
}

case $TASK in
    j2_det) run_j2_det ;;
    j3_det) run_j3_det ;;
    j4_det) run_j4_det ;;
    j2_mask) run_j2_mask ;;
    j3_mask) run_j3_mask ;;
    j4_mask) run_j4_mask ;;
    all)
        echo "========== 开始所有任务 =========="
        run_j2_det
        run_j4_det
        run_j2_mask
        run_j4_mask
        run_j3_det
        run_j3_mask
        echo "========== 所有任务完成 =========="
        ;;
    *)
        echo "未知任务: $TASK"
        exit 1
        ;;
esac