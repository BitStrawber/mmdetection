#!/bin/bash
set -euo pipefail

# Stagewise launcher for J6/J7/J11/J13.
#
# Order:
#   1. Run four S1 pretraining tasks sequentially on S1_GPU_IDS.
#   2. Convert all four S1 checkpoints to MMDetection backbone checkpoints.
#   3. Run four RUOD det tasks in parallel on GPU groups 01/23/45/67.
#   4. After all det tasks finish, run four UIIS10K mask tasks in parallel on
#      the same GPU groups.
#
# Example:
#   nohup bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_j6_j7_j11_j13_det_then_mask.sh \
#     > logs/tri_pretrain_j6_j7_j11_j13_stagewise_launcher.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

TASKS="${TASKS:-j6 j7 j11 j13}"
GPU_GROUPS="${GPU_GROUPS:-0,1 2,3 4,5 6,7}"
S1_GPU_IDS="${S1_GPU_IDS:-0,1,2,3,4,5,6,7}"

RUN_S1="${RUN_S1:-1}"
RUN_CONVERT="${RUN_CONVERT:-1}"
RUN_DET_STAGE="${RUN_DET_STAGE:-1}"
RUN_MASK_STAGE="${RUN_MASK_STAGE:-1}"

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
DET_BASE_PORT="${DET_BASE_PORT:-29800}"
MASK_BASE_PORT="${MASK_BASE_PORT:-29900}"

mkdir -p "$LOG_DIR" "$WORK_ROOT" "$PRETRAIN_DIR"

read -r -a task_array <<< "$TASKS"
read -r -a gpu_group_array <<< "$GPU_GROUPS"

if [ "${#task_array[@]}" -ne "${#gpu_group_array[@]}" ]; then
    echo "Error: TASKS count (${#task_array[@]}) must equal GPU_GROUPS count (${#gpu_group_array[@]})."
    echo "TASKS=$TASKS"
    echo "GPU_GROUPS=$GPU_GROUPS"
    exit 1
fi

run_s1_and_convert() {
    echo "========================================="
    echo "Stage 1/2: S1 pretraining and checkpoint conversion"
    echo "TASKS: $TASKS"
    echo "S1_GPU_IDS: $S1_GPU_IDS"
    echo "RUN_S1: $RUN_S1"
    echo "RUN_CONVERT: $RUN_CONVERT"
    echo "S1_MAX_KEEP_CKPTS: $S1_MAX_KEEP_CKPTS"
    echo "========================================="

    TASKS="$TASKS" \
    RUN_S1="$RUN_S1" \
    RUN_CONVERT="$RUN_CONVERT" \
    RUN_S2=0 \
    S1_GPU_IDS="$S1_GPU_IDS" \
    WAIT_FOR_GPUS="$WAIT_FOR_GPUS" \
    GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
    GPU_MAX_UTIL="$GPU_MAX_UTIL" \
    GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
    GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
    S1_MAX_KEEP_CKPTS="$S1_MAX_KEEP_CKPTS" \
    CHECKPOINT_SAVE_BEST="$CHECKPOINT_SAVE_BEST" \
    REALUW_SSL_ROOT="$REALUW_SSL_ROOT" \
    PRETRAIN_DIR="$PRETRAIN_DIR" \
    LOG_DIR="$LOG_DIR" \
    WORK_ROOT="$WORK_ROOT" \
    bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_auto.sh"
}

run_parallel_stage() {
    local stage_name="$1"
    local run_det="$2"
    local run_mask="$3"
    local base_port="$4"
    local pids=()
    local status=0
    local i=0

    echo "========================================="
    echo "Stage: $stage_name"
    echo "TASKS: $TASKS"
    echo "GPU_GROUPS: $GPU_GROUPS"
    echo "========================================="

    for task in "${task_array[@]}"; do
        local gpu_group="${gpu_group_array[$i]}"
        local port=$((base_port + i))
        echo "Launch $stage_name: EXP_ID=$task GPU=$gpu_group PORT=$port"
        (
            EXP_ID="$task" \
            RUN_DET="$run_det" \
            RUN_MASK="$run_mask" \
            DET_GPU_IDS="$gpu_group" \
            MASK_GPU_IDS="$gpu_group" \
            DET_PORT="$port" \
            MASK_PORT="$port" \
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
            bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_downstream_pair.sh"
        ) &
        pids+=("$!")
        i=$((i + 1))
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done

    if [ "$status" -ne 0 ]; then
        echo "Error: $stage_name failed."
        exit "$status"
    fi
    echo "$stage_name finished."
}

run_s1_and_convert

if [ "$RUN_DET_STAGE" = "1" ]; then
    run_parallel_stage "det" 1 0 "$DET_BASE_PORT"
fi

if [ "$RUN_MASK_STAGE" = "1" ]; then
    run_parallel_stage "mask" 0 1 "$MASK_BASE_PORT"
fi

echo "========================================="
echo "J6/J7/J11/J13 stagewise pipeline finished: $(date)"
echo "logs: $LOG_DIR"
echo "work dirs: $WORK_ROOT"
echo "========================================="
