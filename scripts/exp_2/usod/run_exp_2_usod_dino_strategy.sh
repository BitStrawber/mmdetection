#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# J10 DINO-route:
#   S1: DINO-pretrained ResNet-50 -> expanded DFUI source adaptation.
#   S2: RUOD fine-tuning with the extracted S1 ResNet backbone.
#
# This mirrors the MAE strategy entry, but keeps the backbone as ResNet-50 so it
# can load into the existing Cascade R-CNN J10 S2 config.

cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
GPU_IDS="${GPU_IDS:-4,5}"
PORT="${PORT:-29685}"
WORK_DIR="${WORK_DIR:-work_dirs/j10_dino_usod}"
LOG_DIR="${LOG_DIR:-logs/j10_dino_usod}"
EXP_NAME="${EXP_NAME:-j10_dino_r50_usod_easy}"

S1_CONFIG="${S1_CONFIG:-configs/exp_2/cascade-rcnn_r50_dino_fpn_2x_dfui_ruod_uiis_usod_easy_s1.py}"
S2_CONFIG="${S2_CONFIG:-configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_v2_s2.py}"

FROZEN_STAGES="${FROZEN_STAGES:-1}"
S1_LR="${S1_LR:-0.00375}"
S1_EPOCHS="${S1_EPOCHS:-48}"
S1_MILESTONES="${S1_MILESTONES:-[32,44]}"
S1_WEIGHT_DECAY="${S1_WEIGHT_DECAY:-0.0001}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
RUN_S2="${RUN_S2:-1}"

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

MERGED_ROOT="${MERGED_ROOT:-/media/HDD0/XCX/exp_2/DFUI_RUOD_UIIS_USOD_EASY}"
DINO_CKPT="${DINO_CKPT:-../pretrained_weights/dino_resnet50_pretrain.pth}"

mkdir -p "$LOG_DIR" "$WORK_DIR"

if [ ! -f "$MERGED_ROOT/annotations/instances_train.json" ] || \
   [ ! -f "$MERGED_ROOT/annotations/instances_val.json" ]; then
    echo "Error: merged expanded DFUI source not found under $MERGED_ROOT"
    echo "Run first:"
    echo "  bash scripts/exp_2/usod/run_exp_2_usod_easy_merge.sh"
    exit 1
fi

if [ ! -f "$DINO_CKPT" ]; then
    echo "Warning: expected DINO checkpoint not found: $DINO_CKPT"
    echo "The default S1 config uses ../pretrained_weights/dino_resnet50_pretrain.pth."
    echo "Set DINO_CKPT only for this warning, or edit/override S1_CONFIG if using another checkpoint."
fi

echo "========================================="
echo "J10 DINO-route ResNet transfer"
echo "========================================="
echo "GPU_IDS: $GPU_IDS"
echo "PORT: $PORT"
echo "EXP_NAME: $EXP_NAME"
echo "S1_CONFIG: $S1_CONFIG"
echo "S2_CONFIG: $S2_CONFIG"
echo "DINO_CKPT: $DINO_CKPT"
echo "FROZEN_STAGES: $FROZEN_STAGES"
echo "S1_LR: $S1_LR"
echo "S1_EPOCHS: $S1_EPOCHS"
echo "S1_MILESTONES: $S1_MILESTONES"
echo "S1_WEIGHT_DECAY: $S1_WEIGHT_DECAY"
echo "RUN_S2: $RUN_S2"
echo "========================================="

WORK_DIR="$WORK_DIR" \
LOG_DIR="$LOG_DIR" \
EXP_NAME="$EXP_NAME" \
GPU_IDS="$GPU_IDS" \
PORT="$PORT" \
S1_CONFIG="$S1_CONFIG" \
S2_CONFIG="$S2_CONFIG" \
FROZEN_STAGES="$FROZEN_STAGES" \
S1_LR="$S1_LR" \
S1_EPOCHS="$S1_EPOCHS" \
S1_MILESTONES="$S1_MILESTONES" \
S1_WEIGHT_DECAY="$S1_WEIGHT_DECAY" \
MAX_KEEP_CKPTS="$MAX_KEEP_CKPTS" \
RUN_S2="$RUN_S2" \
WAIT_FOR_GPUS="$WAIT_FOR_GPUS" \
GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
GPU_MAX_UTIL="$GPU_MAX_UTIL" \
GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
bash "$REPO_ROOT/scripts/exp_2/j10/run_exp_2_j10_scheme_c.sh" \
    2>&1 | tee "$LOG_DIR/${EXP_NAME}_launcher.log"

echo "========================================="
echo "J10 DINO-route done"
echo "Logs: $LOG_DIR"
echo "Work dir: $WORK_DIR"
echo "========================================="
