#!/bin/bash

# J10 Pretrain: ImageNet(torchvision) → DFUI_NEW(s1:96ep, 11类 预训练) → 提取backbone → RUOD(s2:24ep)
# 与J10 v2的区别:
#   1. S1使用调整后的LR (0.00375，适配2GPU batch=12)
#   2. S2加载backbone_only.pth（预训练方案），而非完整checkpoint
#   3. 只保留最佳模型和最近10个epoch的checkpoint
# GPU: 0,1

WORK_DIR="work_dirs"
NUM_GPUS=2
GPU_IDS="0,1"
PORT=29503
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "========================================="
echo "J10 Pretrain (ImageNet→DFUI_NEW→Backbone→RUOD)"
echo "========================================="
echo "GPU: $GPU_IDS"
echo ""

# ===== Stage 1: DFUI_NEW 预训练 (96 epoch, 11类) =====
echo ">>> Stage 1: DFUI_NEW 预训练 (96 epoch, 11类)"
mkdir -p $WORK_DIR/j10_pretrain_s1

export PORT
CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
    configs/exp_2/cascade-rcnn_r50_fpn_2x_merged_j10_s1.py \
    $NUM_GPUS \
    --work-dir $WORK_DIR/j10_pretrain_s1 \
    --cfg-options default_hooks.checkpoint.max_keep_ckpts=10 \
    2>&1 | tee "$LOG_DIR/j10_pretrain_s1.log"

if [ $? -ne 0 ]; then
    echo "错误: Stage 1 训练失败"
    exit 1
fi
echo "<<< Stage 1 完成"

# ===== 提取Backbone权重 =====
echo ">>> 提取Backbone权重..."

# 找最佳checkpoint
BEST_CKPT=$(ls -t $WORK_DIR/j10_pretrain_s1/best_coco_bbox_mAP*.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    # 如果没找到best，用最新checkpoint
    BEST_CKPT="$WORK_DIR/j10_pretrain_s1/latest.pth"
fi
echo "使用checkpoint: $BEST_CKPT"

PYTHONPATH="$(dirname "$0"):$PYTHONPATH" python tools/extract_backbone.py \
    --checkpoint "$BEST_CKPT" \
    --output "$WORK_DIR/j10_pretrain_s1/backbone_only.pth"

if [ ! -f "$WORK_DIR/j10_pretrain_s1/backbone_only.pth" ]; then
    echo "错误: Backbone提取失败"
    exit 1
fi
echo "<<< Backbone权重已提取: $WORK_DIR/j10_pretrain_s1/backbone_only.pth"

# ===== Stage 2: RUOD 微调 (24 epoch, 10类) =====
echo ">>> Stage 2: RUOD 微调 (24 epoch, 10类)"
mkdir -p $WORK_DIR/j10_pretrain_s2

BACKBONE_CKPT="$WORK_DIR/j10_pretrain_s1/backbone_only.pth"

# 使用sed替换load_from路径 -> 指向backbone_only.pth
sed "s|load_from = 'work_dirs/j10_v2_s1/best_coco_bbox_mAP_epoch_20.pth'|load_from = '$BACKBONE_CKPT'|" \
    configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_v2_s2.py > configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_pretrain_s2_temp.py

CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
    configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_pretrain_s2_temp.py \
    $NUM_GPUS \
    --work-dir $WORK_DIR/j10_pretrain_s2 \
    --cfg-options default_hooks.checkpoint.max_keep_ckpts=10 \
    2>&1 | tee "$LOG_DIR/j10_pretrain_s2.log"

rm -f configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_pretrain_s2_temp.py

if [ $? -ne 0 ]; then
    echo "错误: Stage 2 训练失败"
    exit 1
fi

echo "<<< J10 Pretrain 完成"
echo "结束时间: $(date)"
