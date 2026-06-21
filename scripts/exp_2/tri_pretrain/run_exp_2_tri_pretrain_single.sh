#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

EXP_ID="${EXP_ID:?EXP_ID is required, for example j6}"

GPU_IDS="${GPU_IDS:-0,1}"
PORT="${PORT:-29680}"
WORK_DIR="${WORK_DIR:-work_dirs/tri_pretrain}"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"
NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPU_IDS")

PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
CHECKPOINT_SAVE_BEST="${CHECKPOINT_SAVE_BEST:-coco/bbox_mAP}"
RUN_DET="${RUN_DET:-1}"
RUN_MASK="${RUN_MASK:-1}"
RUN_TEST="${RUN_TEST:-0}"
EXTRA_CFG_OPTIONS="${EXTRA_CFG_OPTIONS:-}"

DET_CONFIG="${DET_CONFIG:-}"
MASK_CONFIG="${MASK_CONFIG:-}"
if [ "$RUN_DET" = "1" ] && [ -z "$DET_CONFIG" ]; then
    echo "Error: DET_CONFIG is required when RUN_DET=1"
    exit 1
fi
if [ "$RUN_MASK" = "1" ] && [ -z "$MASK_CONFIG" ]; then
    echo "Error: MASK_CONFIG is required when RUN_MASK=1"
    exit 1
fi

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

query_gpu_state() {
    local gpu_id="$1"
    nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -F, -v id="$gpu_id" '
            {
                gsub(/[[:space:]]/, "", $1)
                gsub(/[[:space:]]/, "", $2)
                gsub(/[[:space:]]/, "", $3)
                if ($1 == id) {
                    print $2, $3
                    exit
                }
            }'
}

wait_msg() {
    if [ -w /dev/tty ]; then
        echo "$*" > /dev/tty
    else
        echo "$*" >&2
    fi
}

wait_for_gpus() {
    if [ "$WAIT_FOR_GPUS" != "1" ]; then
        wait_msg "WAIT_FOR_GPUS=$WAIT_FOR_GPUS, skip GPU idle waiting."
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        wait_msg "Warning: nvidia-smi not found, skip GPU idle waiting."
        return
    fi

    local idle_rounds=0
    local gpu_array=()
    IFS=',' read -r -a gpu_array <<< "$GPU_IDS"

    wait_msg "Waiting for GPU group [$GPU_IDS] to become idle..."
    wait_msg "Idle rule: memory.used <= ${GPU_MAX_MEM_MB}MB and utilization.gpu <= ${GPU_MAX_UTIL}% for ${GPU_IDLE_CHECKS} consecutive check(s)."

    while true; do
        local all_idle=1
        local status_parts=()
        for gpu in "${gpu_array[@]}"; do
            gpu="${gpu//[[:space:]]/}"
            [ -z "$gpu" ] && continue
            local state=""
            state=$(query_gpu_state "$gpu" || true)
            if [ -z "$state" ]; then
                status_parts+=("gpu${gpu}=not_found")
                all_idle=0
                continue
            fi
            local mem_used util
            read -r mem_used util <<< "$state"
            status_parts+=("gpu${gpu}=mem:${mem_used}MB,util:${util}%")
            if [ "$mem_used" -gt "$GPU_MAX_MEM_MB" ] || [ "$util" -gt "$GPU_MAX_UTIL" ]; then
                all_idle=0
            fi
        done

        wait_msg "GPU status: ${status_parts[*]}"
        if [ "$all_idle" -eq 1 ]; then
            idle_rounds=$((idle_rounds + 1))
            wait_msg "Idle check ${idle_rounds}/${GPU_IDLE_CHECKS} passed."
            if [ "$idle_rounds" -ge "$GPU_IDLE_CHECKS" ]; then
                wait_msg "GPU group [$GPU_IDS] is idle. Start training."
                return
            fi
        else
            idle_rounds=0
            wait_msg "GPU group [$GPU_IDS] is busy. Recheck after ${GPU_WAIT_INTERVAL}s."
        fi

        sleep "$GPU_WAIT_INTERVAL"
    done
}

run_task() {
    local task_name="$1"
    local config="$2"
    local work_dir="$WORK_DIR/${EXP_ID}_${task_name}"
    local log_file="$LOG_DIR/${EXP_ID}_${task_name}.log"
    local cfg_options=(
        default_hooks.checkpoint.max_keep_ckpts="$MAX_KEEP_CKPTS"
        default_hooks.checkpoint.save_best="$CHECKPOINT_SAVE_BEST"
    )

    if [ -n "$PRETRAIN_CKPT" ]; then
        cfg_options+=(model.backbone.init_cfg.checkpoint="$PRETRAIN_CKPT")
    fi

    if [ -n "$EXTRA_CFG_OPTIONS" ]; then
        # shellcheck disable=SC2206
        local extra_options=($EXTRA_CFG_OPTIONS)
        cfg_options+=("${extra_options[@]}")
    fi

    mkdir -p "$work_dir"
    echo ">>> ${EXP_ID} ${task_name}"
    echo "config: $config"
    echo "work_dir: $work_dir"
    echo "log: $log_file"

    CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
        "$config" \
        "$NUM_GPUS" \
        --work-dir "$work_dir" \
        --cfg-options "${cfg_options[@]}" \
        2>&1 | tee "$log_file"

    if [ "$RUN_TEST" = "1" ]; then
        local best_ckpt
        local test_log="$LOG_DIR/${EXP_ID}_${task_name}_test.log"
        best_ckpt=$(ls -t "$work_dir"/best_*.pth 2>/dev/null | head -1 || true)
        if [ -z "$best_ckpt" ]; then
            best_ckpt="$work_dir/latest.pth"
        fi
        if [ ! -f "$best_ckpt" ]; then
            echo "Error: no checkpoint found for test in $work_dir"
            exit 1
        fi

        echo ">>> ${EXP_ID} ${task_name} test"
        echo "checkpoint: $best_ckpt"
        echo "test log: $test_log"
        CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_test.sh \
            "$config" \
            "$best_ckpt" \
            "$NUM_GPUS" \
            --cfg-options "${cfg_options[@]}" \
            2>&1 | tee "$test_log"
    fi
}

mkdir -p "$LOG_DIR" "$WORK_DIR"
export PORT
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

if [ -n "$PRETRAIN_CKPT" ] && [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Warning: PRETRAIN_CKPT does not exist yet: $PRETRAIN_CKPT"
fi

wait_for_gpus

echo "========================================="
echo "Tri-pretrain downstream experiment: $EXP_ID"
echo "GPU_IDS: $GPU_IDS"
echo "NUM_GPUS: $NUM_GPUS"
echo "PORT: $PORT"
echo "DET_CONFIG: $DET_CONFIG"
echo "MASK_CONFIG: $MASK_CONFIG"
echo "PRETRAIN_CKPT: ${PRETRAIN_CKPT:-config default}"
echo "RUN_DET: $RUN_DET"
echo "RUN_MASK: $RUN_MASK"
echo "RUN_TEST: $RUN_TEST"
echo "========================================="

if [ "$RUN_DET" = "1" ]; then
    run_task det "$DET_CONFIG"
fi

if [ "$RUN_MASK" = "1" ]; then
    run_task mask "$MASK_CONFIG"
fi

echo "========================================="
echo "$EXP_ID done: $(date)"
echo "logs: $LOG_DIR/${EXP_ID}_det.log, $LOG_DIR/${EXP_ID}_mask.log"
echo "work_dirs: $WORK_DIR/${EXP_ID}_det, $WORK_DIR/${EXP_ID}_mask"
echo "========================================="
