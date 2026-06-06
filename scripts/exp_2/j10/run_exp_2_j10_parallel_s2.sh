#!/bin/bash

# J10 并行实验 - 仅 S2 阶段 (S1 已完成时使用)
# GPU分配: A(2,3)  B(4,5)  C(6,7)
#
# LR设置:
#   Exp A: 0.015  (base LR, 不缩放)
#   Exp B: 0.0075 (平方根缩放)
#   Exp C: 0.00375 (线性缩放)

WORK_DIR="work_dirs"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ======================== 实验定义 ========================
declare -A EXP_LR EXP_GPUS EXP_PORTS EXP_NAME

EXPS=("exp_a" "exp_b" "exp_c")
EXP_LR[exp_a]="0.015"
EXP_LR[exp_b]="0.0075"
EXP_LR[exp_c]="0.00375"
EXP_GPUS[exp_a]="2,3"
EXP_GPUS[exp_b]="4,5"
EXP_GPUS[exp_c]="6,7"
EXP_PORTS[exp_a]=29510
EXP_PORTS[exp_b]=29511
EXP_PORTS[exp_c]=29512
EXP_NAME[exp_a]="J10-A (LR=0.015)"
EXP_NAME[exp_b]="J10-B (LR=0.0075)"
EXP_NAME[exp_c]="J10-C (LR=0.00375)"

NUM_GPUS=2

# ======================== 提取Backbone ========================
echo "========================================="
echo "提取 Backbone 权重"
echo "========================================="

for exp in "${EXPS[@]}"; do
    name="${EXP_NAME[$exp]}"

    echo "[$name] 提取 backbone..."

    BEST_CKPT=$(ls -t "$WORK_DIR/${exp}_s1/best_coco_bbox_mAP"*.pth 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT="$WORK_DIR/${exp}_s1/latest.pth"
    fi

    python tools/extract_backbone.py \
        --checkpoint "$BEST_CKPT" \
        --output "$WORK_DIR/${exp}_s1/backbone_only.pth"

    if [ ! -f "$WORK_DIR/${exp}_s1/backbone_only.pth" ]; then
        echo "错误: [$name] Backbone提取失败"
        exit 1
    fi
    echo "[$name] backbone 已提取: $WORK_DIR/${exp}_s1/backbone_only.pth"
done
echo ""

# ======================== Stage 2: 并行微调 ========================
echo "========================================="
echo "J10 并行实验 - Stage 2: RUOD 微调"
echo "========================================="
echo ""

S2_PIDS=()
for exp in "${EXPS[@]}"; do
    lr="${EXP_LR[$exp]}"
    gpus="${EXP_GPUS[$exp]}"
    port="${EXP_PORTS[$exp]}"
    name="${EXP_NAME[$exp]}"
    backbone_ckpt="$WORK_DIR/${exp}_s1/backbone_only.pth"

    echo "启动 $name Stage 2..."

    (
        export PORT=$port
        mkdir -p "$WORK_DIR/${exp}_s2"

        # sed替换load_from -> 指向backbone_only.pth
        # 临时文件放在configs/exp_2/下，_base_的relative path才能正确解析
        sed "s|load_from = 'work_dirs/j10_v2_s1/best_coco_bbox_mAP_epoch_20.pth'|load_from = '$backbone_ckpt'|" \
            configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_v2_s2.py \
            > "configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_parallel_${exp}_s2_temp.py"

        CUDA_VISIBLE_DEVICES=$gpus bash tools/dist_train.sh \
            "configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_parallel_${exp}_s2_temp.py" \
            $NUM_GPUS \
            --work-dir "$WORK_DIR/${exp}_s2" \
            --cfg-options \
                optim_wrapper.optimizer.lr=$lr \
                default_hooks.checkpoint.max_keep_ckpts=10 \
        2>&1 | tee "$LOG_DIR/${exp}_s2.log"

        rm -f "configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_parallel_${exp}_s2_temp.py"

        echo "[$name] Stage 2 完成"
    ) &

    S2_PIDS+=($!)
done

# 等待所有 S2 完成
echo "等待所有 Stage 2 完成..."
for pid in "${S2_PIDS[@]}"; do
    wait $pid
done
echo "所有 Stage 2 完成！"
echo ""

# ======================== 结果汇总 ========================
echo "========================================="
echo "J10 并行实验完成"
echo "========================================="
echo ""

for exp in "${EXPS[@]}"; do
    lr="${EXP_LR[$exp]}"
    name="${EXP_NAME[$exp]}"

    s1_best=$(ls -t "$WORK_DIR/${exp}_s1/best_coco_bbox_mAP"*.pth 2>/dev/null | head -1)
    s2_best=$(ls -t "$WORK_DIR/${exp}_s2/best_coco_bbox_mAP"*.pth 2>/dev/null | head -1)

    echo "----------------------------------------"
    echo "$name"
    echo "  S1: $WORK_DIR/${exp}_s1/"
    echo "  S2: $WORK_DIR/${exp}_s2/"
    echo "  S1 best: $(basename ${s1_best:-'无'})"
    echo "  S2 best: $(basename ${s2_best:-'无'})"

    s2_map=$(grep -oP 'bbox_mAP: \K[0-9.]+' "$LOG_DIR/${exp}_s2.log" 2>/dev/null | tail -1)
    echo "  S2 mAP: ${s2_map:-'日志中未找到'}"
done

echo ""
echo "========================================="
echo "结束时间: $(date)"
echo "========================================="
