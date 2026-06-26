#!/bin/bash
set -euo pipefail

# Materialize ImageNet-1K train class tar files to SSD ImageFolder format,
# then run the official facebookresearch/DINO ResNet-50 100e recipe.
#
# Reuses run_exp_2_imagenet_dino_vits_100e.sh for serial ImageNet extraction
# so both ImageNet ViT-S and ResNet-50 jobs share exactly the same SSD dataset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

IMAGENET_TAR_ROOT="${IMAGENET_TAR_ROOT:-/media/HDD0/XCX/IMAGENET}"
IMAGENET_SSL_ROOT="${IMAGENET_SSL_ROOT:-/media/SSD1/XCX/exp_2/IMAGENET1K}"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"

EXTRACT_IMAGENET="${EXTRACT_IMAGENET:-1}"
VERIFY_IMAGENET="${VERIFY_IMAGENET:-1}"
RUN_PRETRAIN="${RUN_PRETRAIN:-1}"

if [ "$EXTRACT_IMAGENET" = "1" ] || [ "$VERIFY_IMAGENET" = "1" ]; then
    IMAGENET_TAR_ROOT="$IMAGENET_TAR_ROOT" \
    IMAGENET_SSL_ROOT="$IMAGENET_SSL_ROOT" \
    LOG_DIR="$LOG_DIR" \
    EXTRACT_IMAGENET="$EXTRACT_IMAGENET" \
    VERIFY_IMAGENET="$VERIFY_IMAGENET" \
    RUN_PRETRAIN=0 \
    bash "$SCRIPT_DIR/run_exp_2_imagenet_dino_vits_100e.sh"
else
    echo "EXTRACT_IMAGENET=$EXTRACT_IMAGENET and VERIFY_IMAGENET=$VERIFY_IMAGENET, skip ImageNet materialization checks."
fi

if [ "$RUN_PRETRAIN" != "1" ]; then
    echo "RUN_PRETRAIN=$RUN_PRETRAIN, stop after ImageNet materialization."
    exit 0
fi

export EXP_ID="${EXP_ID:-j7}"
export TASK_CONFIG="${TASK_CONFIG:-configs/exp_2/tri_pretrain/s1_imagenet_dino_resnet50_100e.sh}"
export REALUW_SSL_ROOT="$IMAGENET_SSL_ROOT"
export BUILD_REALUW_SSL=0
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export PORT="${PORT:-29693}"
export WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
export GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
export GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
export GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
export GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_s1.sh"
