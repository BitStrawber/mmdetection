#!/bin/bash

# J10 RFTM: ImageNet → DFUI_NEW(48ep, RFTM特征增强) → Backbone+RFTM → RUOD(24ep)
#
# 参考论文: Learning Heavily-Degraded Prior for Underwater Object Detection
# 核心: 在ResNet layer1之后插入RFTM轻量模块，学习水下退化区域特征迁移
#
# GPU: 2,3 (单组实验)
# S1 LR: 0.00375 (2GPU batch=12)

WORK_DIR="work_dirs"
NUM_GPUS=2
GPU_IDS="2,3"
PORT=29520
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "========================================="
echo "J10 RFTM (ResNetWithRFTM + DFUI_NEW → RUOD)"
echo "========================================="
echo "GPU: $GPU_IDS"
echo ""

# ===== Stage 1: DFUI_NEW 预训练 (48 epoch, ResNetWithRFTM) =====
echo ">>> Stage 1: DFUI_NEW 预训练 (48 epoch, ResNetWithRFTM)"
mkdir -p $WORK_DIR/j10_rftm_s1

export PORT
CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
    configs/exp_2/cascade-rcnn_r50-rftm_fpn_2x_dfui_new_s1.py \
    $NUM_GPUS \
    --work-dir $WORK_DIR/j10_rftm_s1 \
    --cfg-options default_hooks.checkpoint.max_keep_ckpts=10 \
    2>&1 | tee "$LOG_DIR/j10_rftm_s1.log"

if [ $? -ne 0 ]; then
    echo "错误: Stage 1 训练失败"
    exit 1
fi
echo "<<< Stage 1 完成"

# ===== 提取Backbone+RFTM权重 =====
echo ">>> 提取Backbone+RFTM权重..."

BEST_CKPT=$(ls -t $WORK_DIR/j10_rftm_s1/best_coco_bbox_mAP*.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="$WORK_DIR/j10_rftm_s1/latest.pth"
fi
echo "使用checkpoint: $BEST_CKPT"

python tools/extract_backbone.py \
    --checkpoint "$BEST_CKPT" \
    --output "$WORK_DIR/j10_rftm_s1/backbone_only.pth"

if [ ! -f "$WORK_DIR/j10_rftm_s1/backbone_only.pth" ]; then
    echo "错误: Backbone提取失败"
    exit 1
fi
echo "<<< Backbone+RFTM权重已提取: $WORK_DIR/j10_rftm_s1/backbone_only.pth"

# ===== Stage 2: RUOD 微调 (24 epoch) =====
echo ">>> Stage 2: RUOD 微调 (24 epoch)"
mkdir -p $WORK_DIR/j10_rftm_s2

CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
    configs/exp_2/cascade-rcnn_r50-rftm_fpn_2x_ruod_s2.py \
    $NUM_GPUS \
    --work-dir $WORK_DIR/j10_rftm_s2 \
    --cfg-options \
        load_from="$WORK_DIR/j10_rftm_s1/backbone_only.pth" \
        default_hooks.checkpoint.max_keep_ckpts=10 \
    2>&1 | tee "$LOG_DIR/j10_rftm_s2.log"

if [ $? -ne 0 ]; then
    echo "错误: Stage 2 训练失败"
    exit 1
fi

echo "<<< J10 RFTM 完成"
echo "结束时间: $(date)"
