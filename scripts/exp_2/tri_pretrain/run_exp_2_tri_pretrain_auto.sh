#!/bin/bash
set -euo pipefail

# Auto launcher for Tri-pretrain experiments.
#
# Pipeline:
#   1. Build RealUW SSL imagefolder dataset once.
#   2. Run selected S1 self-supervised pretraining tasks sequentially.
#   3. Convert each S1 checkpoint to an MMDetection backbone checkpoint.
#   4. Run RUOD detection and UIIS10K mask downstream tasks in parallel for
#      each selected strategy.
#
# Example:
#   TASKS="j6 j7 j11 j13" \
#   S1_GPU_IDS=0,1,2,3,4,5,6,7 DET_GPU_IDS=0,1 MASK_GPU_IDS=2,3 \
#   bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_auto.sh
#
# J12 requires a verified SwinV2-Base masked-modeling pretrain config:
#   J12_CONFIG=/path/to/swinv2_base_masked_modeling_config.py TASKS="j12" bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

TASKS="${TASKS:-j6 j7 j11 j12 j13}"
RUN_S1="${RUN_S1:-1}"
RUN_CONVERT="${RUN_CONVERT:-1}"
RUN_S2="${RUN_S2:-1}"

S1_GPU_IDS="${S1_GPU_IDS:-0,1,2,3,4,5,6,7}"
DET_GPU_IDS="${DET_GPU_IDS:-0,1}"
MASK_GPU_IDS="${MASK_GPU_IDS:-2,3}"

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

S1_MAX_KEEP_CKPTS="${S1_MAX_KEEP_CKPTS:-3}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
CHECKPOINT_SAVE_BEST="${CHECKPOINT_SAVE_BEST:-coco/bbox_mAP}"
REALUW_SSL_ROOT="${REALUW_SSL_ROOT:-/media/HDD1/XCX/exp_2/REALUW_SSL}"
PRETRAIN_DIR="${PRETRAIN_DIR:-../pretrained_weights}"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"
WORK_ROOT="${WORK_ROOT:-work_dirs/tri_pretrain}"

mkdir -p "$LOG_DIR" "$WORK_ROOT" "$PRETRAIN_DIR"

task_enabled() {
    local target="$1"
    for task in $TASKS; do
        [ "$task" = "$target" ] && return 0
    done
    return 1
}

convert_one() {
    local exp_id="$1"
    local s1_name="$2"
    local default_ckpt="$3"
    local out_ckpt="$4"
    local ckpt="$default_ckpt"

    if [ ! -f "$ckpt" ]; then
        ckpt=$(ls -t "$WORK_ROOT/$s1_name"/*.pth 2>/dev/null | head -1 || true)
    fi
    if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
        echo "Error: no S1 checkpoint found for $exp_id in $WORK_ROOT/$s1_name"
        echo "Expected: $default_ckpt"
        return 1
    fi

    echo "Convert $exp_id S1 checkpoint:"
    echo "  input : $ckpt"
    echo "  output: $out_ckpt"
    python tools/convert_ssl_backbone_to_mmdet.py \
        --checkpoint "$ckpt" \
        --out "$out_ckpt" \
        2>&1 | tee "$LOG_DIR/${exp_id}_convert_backbone.log"
}

if [ "$RUN_S1" = "1" ]; then
    echo "========================================="
    echo "Run Tri-pretrain S1 tasks"
    echo "TASKS: $TASKS"
    echo "S1_GPU_IDS: $S1_GPU_IDS"
    echo "WAIT_FOR_GPUS: $WAIT_FOR_GPUS"
    echo "S1_MAX_KEEP_CKPTS: $S1_MAX_KEEP_CKPTS"
    echo "========================================="
    TASKS="$TASKS" \
    GPU_IDS="$S1_GPU_IDS" \
    REALUW_SSL_ROOT="$REALUW_SSL_ROOT" \
    WAIT_FOR_GPUS="$WAIT_FOR_GPUS" \
    GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
    GPU_MAX_UTIL="$GPU_MAX_UTIL" \
    GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
    GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
    S1_MAX_KEEP_CKPTS="$S1_MAX_KEEP_CKPTS" \
    LOG_DIR="$LOG_DIR" \
    WORK_ROOT="$WORK_ROOT" \
    bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_s1_all.sh"
fi

if [ "$RUN_CONVERT" = "1" ]; then
    echo "========================================="
    echo "Convert S1 checkpoints to MMDetection backbone checkpoints"
    echo "PRETRAIN_DIR: $PRETRAIN_DIR"
    echo "========================================="

    if task_enabled j6; then
        convert_one j6 j6_realuw_spark_resnet50 \
            "$WORK_ROOT/j6_realuw_spark_resnet50/resnet50_1kpretrained_timm_style.pth" \
            "$PRETRAIN_DIR/j6_realuw_spark_resnet50_backbone.pth"
    fi
    if task_enabled j7; then
        convert_one j7 j7_realuw_dino_resnet50 \
            "$WORK_ROOT/j7_realuw_dino_resnet50/checkpoint.pth" \
            "$PRETRAIN_DIR/j7_realuw_dino_resnet50_backbone.pth"
    fi
    if task_enabled j11; then
        convert_one j11 j11_realuw_mae_vit_base \
            "$WORK_ROOT/j11_realuw_mae_vit_base/latest.pth" \
            "$PRETRAIN_DIR/j11_realuw_mae_vit_base_backbone.pth"
    fi
    if task_enabled j12; then
        convert_one j12 j12_realuw_simmim_swinv2_base \
            "$WORK_ROOT/j12_realuw_simmim_swinv2_base/latest.pth" \
            "$PRETRAIN_DIR/j12_realuw_simmim_swinv2_base_backbone.pth"
    fi
    if task_enabled j13; then
        convert_one j13 j13_realuw_spark_convnextv2_tiny \
            "$WORK_ROOT/j13_realuw_spark_convnextv2_tiny/latest.pth" \
            "$PRETRAIN_DIR/j13_realuw_spark_convnextv2_tiny_backbone.pth"
    fi
fi

if [ "$RUN_S2" = "1" ]; then
    echo "========================================="
    echo "Run Tri-pretrain downstream tasks"
    echo "TASKS: $TASKS"
    echo "DET_GPU_IDS: $DET_GPU_IDS"
    echo "MASK_GPU_IDS: $MASK_GPU_IDS"
    echo "CHECKPOINT_SAVE_BEST: $CHECKPOINT_SAVE_BEST"
    echo "S2_MAX_KEEP_CKPTS: $MAX_KEEP_CKPTS"
    echo "========================================="
    TASKS="$TASKS" \
    DET_GPU_IDS="$DET_GPU_IDS" \
    MASK_GPU_IDS="$MASK_GPU_IDS" \
    WAIT_FOR_GPUS="$WAIT_FOR_GPUS" \
    GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
    GPU_MAX_UTIL="$GPU_MAX_UTIL" \
    GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
    GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
    MAX_KEEP_CKPTS="$MAX_KEEP_CKPTS" \
    CHECKPOINT_SAVE_BEST="$CHECKPOINT_SAVE_BEST" \
    PRETRAIN_DIR="$PRETRAIN_DIR" \
    LOG_DIR="$LOG_DIR" \
    WORK_DIR="$WORK_ROOT" \
    bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_downstream_all.sh"
fi

echo "========================================="
echo "Tri-pretrain auto pipeline finished: $(date)"
echo "logs: $LOG_DIR"
echo "work dirs: $WORK_ROOT"
echo "converted checkpoints: $PRETRAIN_DIR"
echo "========================================="
