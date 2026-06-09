#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# J10 MAE-route:
#   S1: mae_pretrain_vit_base.pth -> expanded DFUI source detection adaptation.
#   S2: RUOD detection fine-tuning with the extracted S1 ViT backbone.
#
# This follows the previous J3 idea, but inserts an underwater S1 transfer stage
# before RUOD S2. Default GPUs are 4,5.

cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
GPU_IDS="${GPU_IDS:-4,5}"
PORT="${PORT:-29675}"
WORK_DIR="${WORK_DIR:-work_dirs/j10_mae_usod}"
LOG_DIR="${LOG_DIR:-logs/j10_mae_usod}"
EXP_NAME="${EXP_NAME:-j10_mae_vitbase_usod_easy}"

S1_CONFIG="${S1_CONFIG:-configs/exp_2/cascade-rcnn_vit-base_mae_fpn_2x_dfui_ruod_uiis_usod_easy_s1.py}"
S2_CONFIG="${S2_CONFIG:-configs/exp_2/cascade-rcnn_vit-base_mae_fpn_2x_ruod_j10_mae_s2.py}"

S1_WORK_DIR="${S1_WORK_DIR:-$WORK_DIR/${EXP_NAME}_s1}"
S2_WORK_DIR="${S2_WORK_DIR:-$WORK_DIR/${EXP_NAME}_s2}"
S1_LOG="$LOG_DIR/${EXP_NAME}_s1.log"
S2_LOG="$LOG_DIR/${EXP_NAME}_s2.log"
BACKBONE_CKPT="$S1_WORK_DIR/backbone_only.pth"

S1_LR="${S1_LR:-0.0001}"
S1_EPOCHS="${S1_EPOCHS:-48}"
S2_LR="${S2_LR:-0.0001}"
S2_EPOCHS="${S2_EPOCHS:-100}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
RUN_S2="${RUN_S2:-1}"

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

USOD_ROOT="${USOD_ROOT:-/media/HDD1/XCX/exp_2/USOD10K_DET}"
USOD_CROSS_DIR="${USOD_CROSS_DIR:-$USOD_ROOT/annotations/cross_split_det}"
MERGED_ROOT="${MERGED_ROOT:-/media/HDD0/XCX/exp_2/DFUI_RUOD_UIIS_USOD_EASY}"

NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPU_IDS")

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

mkdir -p "$LOG_DIR" "$S1_WORK_DIR" "$S2_WORK_DIR"

export PORT
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}

if [ ! -f "$MERGED_ROOT/annotations/instances_train.json" ] || [ ! -f "$MERGED_ROOT/annotations/instances_val.json" ]; then
    echo "Error: merged expanded DFUI source not found under $MERGED_ROOT"
    echo "Run first:"
    echo "  bash scripts/exp_2/usod/run_exp_2_usod_easy_merge.sh"
    exit 1
fi

if [ ! -f "../pretrained_weights/mae_pretrain_vit_base.pth" ]; then
    echo "Warning: expected MAE checkpoint not found: ../pretrained_weights/mae_pretrain_vit_base.pth"
    echo "The J3 config uses this relative path; training will fail unless it exists on the server."
fi

wait_for_gpus

echo "========================================="
echo "J10 MAE-route ViT transfer"
echo "========================================="
echo "GPU_IDS: $GPU_IDS"
echo "NUM_GPUS: $NUM_GPUS"
echo "EXP_NAME: $EXP_NAME"
echo "S1_CONFIG: $S1_CONFIG"
echo "S2_CONFIG: $S2_CONFIG"
echo "S1_WORK_DIR: $S1_WORK_DIR"
echo "S2_WORK_DIR: $S2_WORK_DIR"
echo "S1_LR: $S1_LR"
echo "S1_EPOCHS: $S1_EPOCHS"
echo "S2_LR: $S2_LR"
echo "S2_EPOCHS: $S2_EPOCHS"
echo "RUN_S2: $RUN_S2"
echo "========================================="

echo ">>> Stage 1: MAE-pretrained ViT transfer on expanded DFUI"
CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
    "$S1_CONFIG" \
    "$NUM_GPUS" \
    --work-dir "$S1_WORK_DIR" \
    --cfg-options \
        optim_wrapper.optimizer.lr="$S1_LR" \
        train_cfg.max_epochs="$S1_EPOCHS" \
        default_hooks.checkpoint.max_keep_ckpts="$MAX_KEEP_CKPTS" \
        default_hooks.checkpoint.save_best=coco/bbox_mAP \
    2>&1 | tee "$S1_LOG"

echo ">>> Extracting S1 backbone-only checkpoint"
BEST_CKPT=$(ls -t "$S1_WORK_DIR"/best_coco_bbox_mAP*.pth 2>/dev/null | head -1 || true)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="$S1_WORK_DIR/latest.pth"
fi
if [ ! -f "$BEST_CKPT" ]; then
    echo "Error: no S1 checkpoint found in $S1_WORK_DIR"
    exit 1
fi
echo "Using S1 checkpoint: $BEST_CKPT"

PYTHONPATH="$(pwd):${PYTHONPATH:-}" "$PYTHON" tools/extract_backbone_only.py \
    --checkpoint "$BEST_CKPT" \
    --output "$BACKBONE_CKPT"

if [ "$RUN_S2" = "0" ]; then
    echo "RUN_S2=0, stop after S1."
    echo "Backbone checkpoint: $BACKBONE_CKPT"
    exit 0
fi

echo ">>> Stage 2: RUOD fine-tuning with S1 ViT backbone"
CUDA_VISIBLE_DEVICES=$GPU_IDS bash tools/dist_train.sh \
    "$S2_CONFIG" \
    "$NUM_GPUS" \
    --work-dir "$S2_WORK_DIR" \
    --cfg-options \
        load_from="$BACKBONE_CKPT" \
        optim_wrapper.optimizer.lr="$S2_LR" \
        train_cfg.max_epochs="$S2_EPOCHS" \
        default_hooks.checkpoint.max_keep_ckpts="$MAX_KEEP_CKPTS" \
    2>&1 | tee "$S2_LOG"

echo "========================================="
echo "J10 MAE-route done"
echo "S1 log: $S1_LOG"
echo "S2 log: $S2_LOG"
echo "Backbone checkpoint: $BACKBONE_CKPT"
echo "========================================="
