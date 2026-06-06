#!/bin/bash

# J10 Two-stage: ImageNet(torchvision) → DFUI(s1:48ep) → RUOD(s2:24ep)
# GPU: 0,1

WORK_DIR="work_dirs"
NUM_GPUS=2
GPU_IDS="0,1"
PORT=29502
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "========================================="
echo "J10 Two-stage (ImageNet→DFUI→RUOD)"
echo "========================================="
echo "GPU: $GPU_IDS"
echo ""

# ===== Stage 1: DFUI训练48epoch =====
echo ">>> Stage 1: DFUI微调 (48 epoch)"
mkdir -p $WORK_DIR/j10_s1

export PORT
CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
    configs/exp_2/cascade-rcnn_r50_fpn_2x_dfui_j10_s1.py \
    $NUM_GPUS \
    --work-dir $WORK_DIR/j10_s1 \
    2>&1 | tee "$LOG_DIR/j10_s1.log"

echo "<<< Stage 1 完成"
echo "查找最佳checkpoint..."

# 获取最佳checkpoint路径
BEST_CKPT=$(ls -t $WORK_DIR/j10_s1/best_coco_bbox_mAP*.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    echo "错误: 未找到best checkpoint"
    exit 1
fi
echo "使用checkpoint: $BEST_CKPT"

# ===== Stage 2: RUOD训练 =====
echo ">>> Stage 2: RUOD微调 (24 epoch)"
mkdir -p $WORK_DIR/j10_s2

# 创建临时配置，加载Stage1的checkpoint
sed "s|load_from = 'work_dirs/j10_s1/best_coco_bbox_mAP_epoch_XX.pth'|load_from = '$BEST_CKPT'|" \
    configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_s2.py > configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_s2_temp.py

CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
    configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_s2_temp.py \
    $NUM_GPUS \
    --work-dir $WORK_DIR/j10_s2 \
    2>&1 | tee "$LOG_DIR/j10_s2.log"

rm -f configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_s2_temp.py

echo "<<< J10完成"
echo "结束时间: $(date)"