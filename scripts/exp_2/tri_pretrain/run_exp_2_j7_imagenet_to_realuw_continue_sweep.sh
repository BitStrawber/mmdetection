#!/bin/bash
set -euo pipefail

# Sweep domain-adaptive DINO continued pretraining:
#   official ImageNet DINO ResNet-50 -> RealUW continue {10,20,30,40,50} epochs.
#
# Each run starts independently from the same ImageNet checkpoint. This makes the
# downstream RUOD comparison reflect total RealUW continue epochs, not cumulative
# reuse between sweep points.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

REALUW_SSL_ROOT="${REALUW_SSL_ROOT:-/media/SSD1/XCX/exp_2/REALUW}"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"
WORK_ROOT="${WORK_ROOT:-work_dirs/tri_pretrain}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-../pretrained_weights/dino_rn50_checkpoint.pth}"
INIT_URL="${INIT_URL:-https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_rn50_checkpoint.pth}"
DOWNLOAD_INIT="${DOWNLOAD_INIT:-1}"

EPOCH_LIST="${EPOCH_LIST:-10 20 30 40 50}"
LR_LIST="${LR_LIST:-0.003}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PORT_BASE="${PORT_BASE:-29740}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

TASK_CONFIG="${TASK_CONFIG:-configs/exp_2/tri_pretrain/s1_j7_dino_resnet50_realuw_continue_from_imagenet.sh}"

mkdir -p "$LOG_DIR" "$WORK_ROOT" "$(dirname "$INIT_CHECKPOINT")"

if [ ! -f "$INIT_CHECKPOINT" ]; then
    if [ "$DOWNLOAD_INIT" != "1" ]; then
        echo "Error: INIT_CHECKPOINT not found: $INIT_CHECKPOINT"
        echo "Set INIT_CHECKPOINT=/path/to/dino_rn50_checkpoint.pth or DOWNLOAD_INIT=1."
        exit 1
    fi
    echo "Downloading official ImageNet DINO RN50 checkpoint:"
    echo "  $INIT_URL"
    echo "to:"
    echo "  $INIT_CHECKPOINT"
    wget "$INIT_URL" -O "$INIT_CHECKPOINT"
fi

if [ ! -d "$REALUW_SSL_ROOT/imagefolder/train" ]; then
    echo "Error: RealUW ImageFolder train directory not found:"
    echo "  $REALUW_SSL_ROOT/imagefolder/train"
    echo "Set REALUW_SSL_ROOT=/path/to/REALUW with imagefolder/train."
    exit 1
fi

port_offset=0
for lr in $LR_LIST; do
    lr_tag="${lr//./p}"
    lr_tag="${lr_tag//-/_}"
    for epochs in $EPOCH_LIST; do
        name="j7_imagenet_dino_rn50_to_realuw_continue_${epochs}e_lr${lr_tag}"
        port=$((PORT_BASE + port_offset))
        port_offset=$((port_offset + 1))

        echo "========================================="
        echo "DINO continue sweep"
        echo "name: $name"
        echo "init_checkpoint: $INIT_CHECKPOINT"
        echo "realuw_root: $REALUW_SSL_ROOT"
        echo "epochs: $epochs"
        echo "lr: $lr"
        echo "gpu_ids: $GPU_IDS"
        echo "port: $port"
        echo "========================================="

        EXP_ID=j7 \
        TASK_CONFIG="$TASK_CONFIG" \
        REALUW_SSL_ROOT="$REALUW_SSL_ROOT" \
        BUILD_REALUW_SSL=0 \
        DINO_NAME="$name" \
        DINO_INIT_CHECKPOINT="$INIT_CHECKPOINT" \
        DINO_EPOCHS="$epochs" \
        DINO_LR="$lr" \
        GPU_IDS="$GPU_IDS" \
        PORT="$port" \
        LOG_DIR="$LOG_DIR" \
        WORK_ROOT="$WORK_ROOT" \
        WAIT_FOR_GPUS="$WAIT_FOR_GPUS" \
        GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
        GPU_MAX_UTIL="$GPU_MAX_UTIL" \
        GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
        GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
        bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_s1.sh"
    done
done
