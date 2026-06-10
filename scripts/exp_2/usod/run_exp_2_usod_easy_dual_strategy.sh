#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Combined USOD expansion + two J10 strategies.
#
# Workflow:
#   1) GPU 2,3: USOD10K A/B cross filtering and DFUI merge.
#   2) GPU 2,3: original ResNet/Cascade scheme-C S1 -> RUOD S2.
#   3) GPU 4,5: J3-style MAE ViT S1 -> RUOD S2.
#
# The two strategy runs start in parallel after filtering/merge is finished.
# Set RUN_MAE=0 or RUN_RCNN=0 to run only one strategy.

cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

USOD_GPU_IDS="${USOD_GPU_IDS:-2,3}"
USOD_NUM_GPUS="${USOD_NUM_GPUS:-2}"
USOD_THRESHOLD="${USOD_THRESHOLD:-0.6}"
FORCE_USOD_EASY="${FORCE_USOD_EASY:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"

RUN_RCNN="${RUN_RCNN:-1}"
RUN_MAE="${RUN_MAE:-1}"

RCNN_GPU_IDS="${RCNN_GPU_IDS:-2,3}"
MAE_GPU_IDS="${MAE_GPU_IDS:-4,5}"

MASTER_LOG_DIR="${MASTER_LOG_DIR:-logs/j10_usod_dual_strategy}"
mkdir -p "$MASTER_LOG_DIR"

echo "========================================="
echo "USOD easy + dual J10 strategies"
echo "========================================="
echo "USOD_GPU_IDS: $USOD_GPU_IDS"
echo "USOD_THRESHOLD: $USOD_THRESHOLD"
echo "RUN_RCNN: $RUN_RCNN"
echo "RUN_MAE: $RUN_MAE"
echo "RCNN_GPU_IDS: $RCNN_GPU_IDS"
echo "MAE_GPU_IDS: $MAE_GPU_IDS"
echo "========================================="

echo ">>> Step 1: USOD easy filtering and DFUI merge on GPU $USOD_GPU_IDS"
PYTHON="$PYTHON" \
USOD_GPU_IDS="$USOD_GPU_IDS" \
USOD_NUM_GPUS="$USOD_NUM_GPUS" \
USOD_THRESHOLD="$USOD_THRESHOLD" \
FORCE_USOD_EASY="$FORCE_USOD_EASY" \
FORCE_MERGE="$FORCE_MERGE" \
bash "$REPO_ROOT/scripts/exp_2/usod/run_exp_2_usod_easy_merge.sh" \
    2>&1 | tee "$MASTER_LOG_DIR/usod_easy_merge.log"

PIDS=()
NAMES=()

if [ "$RUN_RCNN" = "1" ]; then
    echo ">>> Step 2: Original ResNet/Cascade scheme-C S1/S2 on GPU $RCNN_GPU_IDS"
    (
        WORK_DIR="${RCNN_WORK_DIR:-work_dirs/j10_scheme_c_usod}" \
        LOG_DIR="${RCNN_LOG_DIR:-logs/j10_scheme_c_usod}" \
        EXP_NAME="${RCNN_EXP_NAME:-j10_scheme_c_f1_lr00375_e48_usod_obj}" \
        GPU_IDS="$RCNN_GPU_IDS" \
        PORT="${RCNN_PORT:-29661}" \
        S1_CONFIG="${RCNN_S1_CONFIG:-configs/exp_2/cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_usod_easy_j10_scheme_c_s1.py}" \
        S2_CONFIG="${RCNN_S2_CONFIG:-configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_v2_s2.py}" \
        FROZEN_STAGES="${RCNN_FROZEN_STAGES:-1}" \
        S1_LR="${RCNN_S1_LR:-0.00375}" \
        S1_EPOCHS="${RCNN_S1_EPOCHS:-48}" \
        S1_MILESTONES="${RCNN_S1_MILESTONES:-[32,44]}" \
        S1_WEIGHT_DECAY="${RCNN_S1_WEIGHT_DECAY:-0.0001}" \
        MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}" \
        RUN_S2="${RCNN_RUN_S2:-1}" \
        WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}" \
        GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}" \
        GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}" \
        GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}" \
        GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}" \
        bash "$REPO_ROOT/scripts/exp_2/j10/run_exp_2_j10_scheme_c.sh" \
            2>&1 | tee "$MASTER_LOG_DIR/rcnn_strategy.log"
    ) &
    PIDS+=($!)
    NAMES+=("rcnn")
else
    echo "RUN_RCNN=$RUN_RCNN, skip original ResNet/Cascade strategy."
fi

if [ "$RUN_MAE" = "1" ]; then
    echo ">>> Step 3: MAE ViT strategy S1/S2 on GPU $MAE_GPU_IDS"
    (
        PYTHON="$PYTHON" \
        GPU_IDS="$MAE_GPU_IDS" \
        PORT="${MAE_PORT:-29675}" \
        WORK_DIR="${MAE_WORK_DIR:-work_dirs/j10_mae_usod}" \
        LOG_DIR="${MAE_LOG_DIR:-logs/j10_mae_usod}" \
        EXP_NAME="${MAE_EXP_NAME:-j10_mae_vitbase_usod_easy}" \
        S1_LR="${MAE_S1_LR:-0.0001}" \
        S1_EPOCHS="${MAE_S1_EPOCHS:-48}" \
        S2_LR="${MAE_S2_LR:-0.0001}" \
        S2_EPOCHS="${MAE_S2_EPOCHS:-100}" \
        MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}" \
        RUN_S2="${MAE_RUN_S2:-1}" \
        WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}" \
        GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}" \
        GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}" \
        GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}" \
        GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}" \
        bash "$REPO_ROOT/scripts/exp_2/usod/run_exp_2_usod_mae_strategy.sh" \
            2>&1 | tee "$MASTER_LOG_DIR/mae_strategy.log"
    ) &
    PIDS+=($!)
    NAMES+=("mae")
else
    echo "RUN_MAE=$RUN_MAE, skip MAE ViT strategy."
fi

if [ "${#PIDS[@]}" -gt 0 ]; then
    echo ">>> Waiting for strategy jobs: ${NAMES[*]}"
    failed=0
    for i in "${!PIDS[@]}"; do
        pid="${PIDS[$i]}"
        name="${NAMES[$i]}"
        if wait "$pid"; then
            echo "[$name] finished successfully."
        else
            echo "[$name] failed."
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "Error: at least one strategy failed."
        exit 1
    fi
fi

echo "========================================="
echo "Dual strategy pipeline finished"
echo "Master logs: $MASTER_LOG_DIR"
echo "RCNN logs: ${RCNN_LOG_DIR:-logs/j10_scheme_c_usod}"
echo "MAE logs: ${MAE_LOG_DIR:-logs/j10_mae_usod}"
echo "========================================="
