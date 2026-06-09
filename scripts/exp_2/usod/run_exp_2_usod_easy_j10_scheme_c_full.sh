#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Full USOD10K objectness expansion + J10 scheme C run.
#
# Workflow:
#   1) Run USOD10K A/B easy filtering at threshold 0.6.
#   2) Merge DFUI + RUOD_easy + UIIS_easy + USOD_easy into a 12-class source.
#   3) Run J10 scheme C:
#        S1 uses the 12-class DFUI_RUOD_UIIS_USOD_EASY config.
#        S2 stays unchanged and loads only the extracted S1 backbone.
#
# Prerequisite:
#   /media/HDD1/XCX/exp_2/USOD10K_DET/
#     images/
#     annotations/instances_trainval.json

cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
WORK_DIR="${WORK_DIR:-work_dirs/j10_scheme_c_usod}"
LOG_DIR="${LOG_DIR:-logs/j10_scheme_c_usod}"

USOD_ROOT="${USOD_ROOT:-/media/HDD1/XCX/exp_2/USOD10K_DET}"
USOD_ANN="${USOD_ANN:-$USOD_ROOT/annotations/instances_trainval.json}"
USOD_CROSS_DIR="${USOD_CROSS_DIR:-$USOD_ROOT/annotations/cross_split_det}"
USOD_GPU_IDS="${USOD_GPU_IDS:-2,3}"
USOD_NUM_GPUS="${USOD_NUM_GPUS:-2}"
USOD_THRESHOLD="${USOD_THRESHOLD:-0.6}"

FORCE_USOD_EASY="${FORCE_USOD_EASY:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"
RUN_J10="${RUN_J10:-1}"

EXP_NAME="${EXP_NAME:-j10_scheme_c_f1_lr00375_e48_usod_obj}"
GPU_IDS="${GPU_IDS:-2,3}"
PORT="${PORT:-29661}"
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

S1_CONFIG="${S1_CONFIG:-configs/exp_2/cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_usod_easy_j10_scheme_c_s1.py}"
S2_CONFIG="${S2_CONFIG:-configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_v2_s2.py}"

mkdir -p "$LOG_DIR"

echo "========================================="
echo "USOD10K easy + merge + J10 scheme C full pipeline"
echo "========================================="
echo "USOD_ROOT: $USOD_ROOT"
echo "USOD_ANN: $USOD_ANN"
echo "USOD_CROSS_DIR: $USOD_CROSS_DIR"
echo "USOD_GPU_IDS: $USOD_GPU_IDS"
echo "USOD_THRESHOLD: $USOD_THRESHOLD"
echo "FORCE_USOD_EASY: $FORCE_USOD_EASY"
echo "FORCE_MERGE: $FORCE_MERGE"
echo "RUN_J10: $RUN_J10"
echo "EXP_NAME: $EXP_NAME"
echo "GPU_IDS: $GPU_IDS"
echo "S1_CONFIG: $S1_CONFIG"
echo "S2_CONFIG: $S2_CONFIG"
echo "FROZEN_STAGES: $FROZEN_STAGES"
echo "S1_LR: $S1_LR"
echo "S1_EPOCHS: $S1_EPOCHS"
echo "S1_MILESTONES: $S1_MILESTONES"
echo "S1_WEIGHT_DECAY: $S1_WEIGHT_DECAY"
echo "========================================="

echo ">>> Step 1: USOD10K 60% easy filtering"
if [ "$FORCE_USOD_EASY" = "1" ] || [ ! -f "$USOD_CROSS_DIR/easy_merged.json" ]; then
    "$PYTHON" tools/usod10k_cross_easy.py \
        --step all \
        --data-root "$USOD_ROOT/" \
        --ann "$USOD_ANN" \
        --cross-dir "$USOD_CROSS_DIR" \
        --img-prefix images/ \
        --gpu-ids "$USOD_GPU_IDS" \
        --num-gpus "$USOD_NUM_GPUS" \
        --threshold "$USOD_THRESHOLD" \
        --log-dir "$LOG_DIR" \
        2>&1 | tee "$LOG_DIR/usod10k_cross_easy_full.log"
else
    echo "Skip USOD easy filtering, found: $USOD_CROSS_DIR/easy_merged.json"
fi

if [ ! -f "$USOD_CROSS_DIR/easy_merged.json" ]; then
    echo "Error: USOD easy annotation was not created: $USOD_CROSS_DIR/easy_merged.json"
    exit 1
fi

echo ">>> Step 2: Merge 12-class DFUI source"
MERGE_ARGS=()
if [ "$FORCE_MERGE" = "1" ]; then
    MERGE_ARGS+=(--overwrite)
fi

"$PYTHON" tools/merge_dfui_ruod_uiis_usod_easy.py \
    --usod-easy-img-dir "$USOD_ROOT/images" \
    --usod-easy-ann "$USOD_CROSS_DIR/easy_merged.json" \
    "${MERGE_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/dfui_ruod_uiis_usod_easy_merge.log"

if [ "$RUN_J10" = "0" ]; then
    echo "RUN_J10=0, stop after easy filtering and merge."
    exit 0
fi

echo ">>> Step 3: J10 scheme C S1 -> backbone -> S2"
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
echo "USOD J10 full pipeline finished"
echo "USOD easy: $USOD_CROSS_DIR/easy_merged.json"
echo "Merged root: /media/HDD0/XCX/exp_2/DFUI_RUOD_UIIS_USOD_EASY"
echo "J10 work dir: $WORK_DIR"
echo "J10 logs: $LOG_DIR"
echo "========================================="
