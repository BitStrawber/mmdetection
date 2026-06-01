#!/bin/bash
set -euo pipefail

# J10 scheme C:
#   S1: Cascade-supervised underwater backbone adaptation.
#   S2: Keep the existing RUOD config unchanged, but load the extracted
#       backbone-only checkpoint from S1.
#
# Default center setting:
#   frozen_stages=2, lr=0.001875, epochs=48, milestones=[32,44]
#
# Common overrides:
#   GPU_IDS=2,3 bash run_exp_2_j10_scheme_c.sh
#   FROZEN_STAGES=3 S1_LR=0.001875 S1_EPOCHS=48 S1_MILESTONES='[32,44]' bash run_exp_2_j10_scheme_c.sh
#   RUN_S2=0 bash run_exp_2_j10_scheme_c.sh

WORK_DIR=${WORK_DIR:-work_dirs}
LOG_DIR=${LOG_DIR:-logs}
GPU_IDS=${GPU_IDS:-0,1}
PORT=${PORT:-29531}

S1_CONFIG=${S1_CONFIG:-configs/exp_2/cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_easy_j10_scheme_c_s1.py}
S2_CONFIG=${S2_CONFIG:-configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_v2_s2.py}

EXP_NAME=${EXP_NAME:-j10_scheme_c_f2_lr001875_e48}
S1_WORK_DIR=${S1_WORK_DIR:-$WORK_DIR/${EXP_NAME}_s1}
S2_WORK_DIR=${S2_WORK_DIR:-$WORK_DIR/${EXP_NAME}_s2}

FROZEN_STAGES=${FROZEN_STAGES:-2}
S1_LR=${S1_LR:-0.001875}
S1_EPOCHS=${S1_EPOCHS:-48}
S1_MILESTONES=${S1_MILESTONES:-[32,44]}
S1_WEIGHT_DECAY=${S1_WEIGHT_DECAY:-0.0001}
MAX_KEEP_CKPTS=${MAX_KEEP_CKPTS:-5}
RUN_S2=${RUN_S2:-1}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
GPU_MAX_MEM_MB=${GPU_MAX_MEM_MB:-3000}
GPU_MAX_UTIL=${GPU_MAX_UTIL:-10}
GPU_IDLE_CHECKS=${GPU_IDLE_CHECKS:-2}
GPU_WAIT_INTERVAL=${GPU_WAIT_INTERVAL:-30}

NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPU_IDS")
S1_LOG="$LOG_DIR/${EXP_NAME}_s1.log"
S2_LOG="$LOG_DIR/${EXP_NAME}_s2.log"
BACKBONE_CKPT="$S1_WORK_DIR/backbone_only.pth"
TEMP_S2_CONFIG="configs/exp_2/${EXP_NAME}_s2_temp.py"

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
            if [ -z "$gpu" ]; then
                continue
            fi
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

mkdir -p "$LOG_DIR" "$S1_WORK_DIR" "$S2_WORK_DIR"

export PORT
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}

wait_for_gpus

echo "========================================="
echo "J10 scheme C"
echo "========================================="
echo "GPU_IDS: $GPU_IDS"
echo "NUM_GPUS: $NUM_GPUS"
echo "PORT: $PORT"
echo "EXP_NAME: $EXP_NAME"
echo "S1_CONFIG: $S1_CONFIG"
echo "S2_CONFIG: $S2_CONFIG"
echo "S1_WORK_DIR: $S1_WORK_DIR"
echo "S2_WORK_DIR: $S2_WORK_DIR"
echo "FROZEN_STAGES: $FROZEN_STAGES"
echo "S1_LR: $S1_LR"
echo "S1_EPOCHS: $S1_EPOCHS"
echo "S1_MILESTONES: $S1_MILESTONES"
echo "S1_WEIGHT_DECAY: $S1_WEIGHT_DECAY"
echo "RUN_S2: $RUN_S2"
echo "WAIT_FOR_GPUS: $WAIT_FOR_GPUS"
echo "GPU_MAX_MEM_MB: $GPU_MAX_MEM_MB"
echo "GPU_MAX_UTIL: $GPU_MAX_UTIL"
echo "GPU_IDLE_CHECKS: $GPU_IDLE_CHECKS"
echo "GPU_WAIT_INTERVAL: $GPU_WAIT_INTERVAL"
echo "========================================="

echo ">>> Stage 1: Cascade-supervised backbone adaptation"
CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
    "$S1_CONFIG" \
    "$NUM_GPUS" \
    --work-dir "$S1_WORK_DIR" \
    --cfg-options \
        model.backbone.frozen_stages="$FROZEN_STAGES" \
        optim_wrapper.optimizer.lr="$S1_LR" \
        optim_wrapper.optimizer.weight_decay="$S1_WEIGHT_DECAY" \
        train_cfg.max_epochs="$S1_EPOCHS" \
        param_scheduler.1.end="$S1_EPOCHS" \
        param_scheduler.1.milestones="$S1_MILESTONES" \
        default_hooks.checkpoint.max_keep_ckpts="$MAX_KEEP_CKPTS" \
        default_hooks.checkpoint.save_best=coco/bbox_mAP \
    2>&1 | tee "$S1_LOG"

echo ">>> Extracting backbone-only checkpoint"
BEST_CKPT=$(ls -t "$S1_WORK_DIR"/best_coco_bbox_mAP*.pth 2>/dev/null | head -1 || true)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="$S1_WORK_DIR/latest.pth"
fi
if [ ! -f "$BEST_CKPT" ]; then
    echo "Error: no S1 checkpoint found in $S1_WORK_DIR"
    exit 1
fi
echo "Using S1 checkpoint: $BEST_CKPT"

PYTHONPATH="$(pwd):${PYTHONPATH:-}" python tools/extract_backbone_only.py \
    --checkpoint "$BEST_CKPT" \
    --output "$BACKBONE_CKPT"

if [ ! -f "$BACKBONE_CKPT" ]; then
    echo "Error: backbone checkpoint was not created: $BACKBONE_CKPT"
    exit 1
fi

if [ "$RUN_S2" = "0" ]; then
    echo "RUN_S2=0, stop after S1."
    echo "Backbone checkpoint: $BACKBONE_CKPT"
    exit 0
fi

echo ">>> Stage 2: RUOD fine-tuning with unchanged S2 config"
sed "s|load_from = 'work_dirs/j10_v2_s1/best_coco_bbox_mAP_epoch_20.pth'|load_from = '$BACKBONE_CKPT'|" \
    "$S2_CONFIG" > "$TEMP_S2_CONFIG"

CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
    "$TEMP_S2_CONFIG" \
    "$NUM_GPUS" \
    --work-dir "$S2_WORK_DIR" \
    --cfg-options default_hooks.checkpoint.max_keep_ckpts="$MAX_KEEP_CKPTS" \
    2>&1 | tee "$S2_LOG"

rm -f "$TEMP_S2_CONFIG"

echo "========================================="
echo "J10 scheme C done"
echo "S1 log: $S1_LOG"
echo "S2 log: $S2_LOG"
echo "Backbone checkpoint: $BACKBONE_CKPT"
echo "========================================="
