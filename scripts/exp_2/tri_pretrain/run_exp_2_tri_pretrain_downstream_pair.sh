#!/bin/bash
set -euo pipefail

# Run downstream RUOD detection and UIIS10K mask tasks in parallel for one
# Tri-pretrain strategy. Each task uses an independent 2-GPU group by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

EXP_ID="${EXP_ID:?EXP_ID is required: j6, j7, j11, j12, or j13}"
DET_GPU_IDS="${DET_GPU_IDS:-0,1}"
MASK_GPU_IDS="${MASK_GPU_IDS:-2,3}"
DET_PORT="${DET_PORT:-29731}"
MASK_PORT="${MASK_PORT:-29732}"
WORK_DIR="${WORK_DIR:-work_dirs/tri_pretrain}"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
CHECKPOINT_SAVE_BEST="${CHECKPOINT_SAVE_BEST:-coco/bbox_mAP}"
RUN_DET="${RUN_DET:-1}"
RUN_MASK="${RUN_MASK:-1}"
RUN_TEST="${RUN_TEST:-0}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

PRETRAIN_DIR="${PRETRAIN_DIR:-../pretrained_weights}"

case "$EXP_ID" in
    j6)
        PRETRAIN_CKPT="${PRETRAIN_CKPT:-$PRETRAIN_DIR/j6_realuw_spark_resnet50_backbone.pth}"
        DET_CONFIG="${DET_CONFIG:-configs/exp_2/cascade-rcnn_r50_realuw-pretrain_fpn_2x_ruod_j6.py}"
        MASK_CONFIG="${MASK_CONFIG:-configs/exp_2/mask-rcnn_r50_realuw-pretrain_fpn_2x_uiis10k_j6_mask.py}"
        ;;
    j7)
        PRETRAIN_CKPT="${PRETRAIN_CKPT:-$PRETRAIN_DIR/j7_realuw_dino_resnet50_backbone.pth}"
        DET_CONFIG="${DET_CONFIG:-configs/exp_2/cascade-rcnn_r50_dino-realuw_fpn_2x_ruod_j7.py}"
        MASK_CONFIG="${MASK_CONFIG:-configs/exp_2/mask-rcnn_r50_dino-realuw_fpn_2x_uiis10k_j7_mask.py}"
        ;;
    j11)
        PRETRAIN_CKPT="${PRETRAIN_CKPT:-$PRETRAIN_DIR/j11_realuw_mae_vit_base_backbone.pth}"
        DET_CONFIG="${DET_CONFIG:-configs/exp_2/cascade-rcnn_vit-base_mae-realuw_fpn_2x_ruod_j11.py}"
        MASK_CONFIG="${MASK_CONFIG:-configs/exp_2/mask-rcnn_vit-base_mae-realuw_fpn_2x_uiis10k_j11_mask.py}"
        ;;
    j12)
        PRETRAIN_CKPT="${PRETRAIN_CKPT:-$PRETRAIN_DIR/j12_realuw_simmim_swinv2_base_backbone.pth}"
        DET_CONFIG="${DET_CONFIG:-configs/exp_2/cascade-rcnn_swinv2-base_mae-realuw_fpn_2x_ruod_j12.py}"
        MASK_CONFIG="${MASK_CONFIG:-configs/exp_2/mask-rcnn_swinv2-base_mae-realuw_fpn_2x_uiis10k_j12_mask.py}"
        ;;
    j13)
        PRETRAIN_CKPT="${PRETRAIN_CKPT:-$PRETRAIN_DIR/j13_realuw_spark_convnextv2_tiny_backbone.pth}"
        DET_CONFIG="${DET_CONFIG:-configs/exp_2/cascade-rcnn_convnext-tiny_mae-realuw_fpn_2x_ruod_j13.py}"
        MASK_CONFIG="${MASK_CONFIG:-configs/exp_2/mask-rcnn_convnext-tiny_mae-realuw_fpn_2x_uiis10k_j13_mask.py}"
        ;;
    *)
        echo "Error: unsupported EXP_ID=$EXP_ID"
        exit 1
        ;;
esac

mkdir -p "$WORK_DIR" "$LOG_DIR"

if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Error: PRETRAIN_CKPT not found: $PRETRAIN_CKPT"
    exit 1
fi

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

wait_for_gpu_group() {
    local gpu_ids="$1"
    local label="$2"

    if [ "$WAIT_FOR_GPUS" != "1" ]; then
        wait_msg "WAIT_FOR_GPUS=$WAIT_FOR_GPUS, skip GPU idle waiting for $label."
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        wait_msg "Warning: nvidia-smi not found, skip GPU idle waiting for $label."
        return
    fi

    local idle_rounds=0
    local gpu_array=()
    IFS=',' read -r -a gpu_array <<< "$gpu_ids"

    wait_msg "Waiting for GPU group [$gpu_ids] before $label..."
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
                wait_msg "GPU group [$gpu_ids] is idle. Start $label."
                return
            fi
        else
            idle_rounds=0
            wait_msg "GPU group [$gpu_ids] is busy. Recheck after ${GPU_WAIT_INTERVAL}s."
        fi

        sleep "$GPU_WAIT_INTERVAL"
    done
}

run_one() {
    local task_name="$1"
    local config="$2"
    local gpu_ids="$3"
    local port="$4"
    local num_gpus
    local work_dir="$WORK_DIR/${EXP_ID}_${task_name}"
    local log_file="$LOG_DIR/${EXP_ID}_${task_name}.log"

    num_gpus=$(awk -F, '{print NF}' <<< "$gpu_ids")
    mkdir -p "$work_dir"

    (
        wait_for_gpu_group "$gpu_ids" "${EXP_ID} ${task_name}"

        export PORT="$port"
        export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
        echo "========================================="
        echo "${EXP_ID} ${task_name}"
        echo "config: $config"
        echo "pretrain: $PRETRAIN_CKPT"
        echo "gpu_ids: $gpu_ids"
        echo "port: $port"
        echo "work_dir: $work_dir"
        echo "========================================="
        CUDA_VISIBLE_DEVICES="$gpu_ids" bash tools/dist_train.sh \
            "$config" \
            "$num_gpus" \
            --work-dir "$work_dir" \
            --cfg-options \
                model.backbone.init_cfg.checkpoint="$PRETRAIN_CKPT" \
                default_hooks.checkpoint.save_best="$CHECKPOINT_SAVE_BEST" \
                default_hooks.checkpoint.max_keep_ckpts="$MAX_KEEP_CKPTS" \
            2>&1 | tee "$log_file"

        if [ "$RUN_TEST" = "1" ]; then
            local best_ckpt
            best_ckpt=$(ls -t "$work_dir"/best_*.pth 2>/dev/null | head -1 || true)
            [ -z "$best_ckpt" ] && best_ckpt="$work_dir/latest.pth"
            CUDA_VISIBLE_DEVICES="$gpu_ids" bash tools/dist_test.sh \
                "$config" \
                "$best_ckpt" \
                "$num_gpus" \
                --cfg-options model.backbone.init_cfg.checkpoint="$PRETRAIN_CKPT" \
                2>&1 | tee "$LOG_DIR/${EXP_ID}_${task_name}_test.log"
        fi
    ) &
}

echo "Run downstream pair for $EXP_ID"
echo "DET:  $DET_GPU_IDS -> $DET_CONFIG"
echo "MASK: $MASK_GPU_IDS -> $MASK_CONFIG"
echo "PRETRAIN_CKPT: $PRETRAIN_CKPT"

pids=()
if [ "$RUN_DET" = "1" ]; then
    run_one det "$DET_CONFIG" "$DET_GPU_IDS" "$DET_PORT"
    pids+=("$!")
fi
if [ "$RUN_MASK" = "1" ]; then
    run_one mask "$MASK_CONFIG" "$MASK_GPU_IDS" "$MASK_PORT"
    pids+=("$!")
fi

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

exit "$status"
