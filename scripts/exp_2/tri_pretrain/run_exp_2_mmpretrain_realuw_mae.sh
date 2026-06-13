#!/bin/bash
set -euo pipefail

# Run MMPreTrain MAE pretraining on the merged unlabeled REALUW_SSL dataset.
#
# Expected external repo:
#   MMPRETRAIN_DIR=/path/to/mmpretrain
#
# This script only handles the self-supervised pretraining stage. Downstream
# RUOD training should load the produced latest/best mmpretrain checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MMPRETRAIN_DIR="${MMPRETRAIN_DIR:-../mmpretrain}"
CONFIG="${CONFIG:-configs/exp_2/mmpretrain/realuw_ssl_mae_vit-base-p16_2xb128-amp-coslr-100e.py}"
WORK_DIR="${WORK_DIR:-work_dirs/tri_pretrain/realuw_mae_vit_base_mmpretrain}"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"
GPU_IDS="${GPU_IDS:-0,1}"
PORT="${PORT:-29711}"
REALUW_SSL_ROOT="${REALUW_SSL_ROOT:-/media/HDD1/XCX/exp_2/REALUW_SSL}"
BUILD_REALUW_SSL="${BUILD_REALUW_SSL:-1}"

NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPU_IDS")
mkdir -p "$LOG_DIR" "$WORK_DIR"

if [ "$BUILD_REALUW_SSL" = "1" ]; then
    python tools/build_realuw_ssl_dataset.py \
        --preset exp2_bbox20pct \
        --out-root "$REALUW_SSL_ROOT"
fi

if [ ! -f "$MMPRETRAIN_DIR/tools/train.py" ]; then
    echo "Error: mmpretrain tools/train.py not found: $MMPRETRAIN_DIR/tools/train.py"
    echo "Set MMPRETRAIN_DIR=/path/to/mmpretrain"
    exit 1
fi

export PORT
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

echo "========================================="
echo "MMPreTrain RealUW MAE"
echo "MMPRETRAIN_DIR: $MMPRETRAIN_DIR"
echo "CONFIG: $CONFIG"
echo "WORK_DIR: $WORK_DIR"
echo "GPU_IDS: $GPU_IDS"
echo "NUM_GPUS: $NUM_GPUS"
echo "REALUW_SSL_ROOT: $REALUW_SSL_ROOT"
echo "========================================="

CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH="$MMPRETRAIN_DIR:${PYTHONPATH:-}" \
python -m torch.distributed.launch \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$PORT" \
    "$MMPRETRAIN_DIR/tools/train.py" \
    "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher pytorch \
    2>&1 | tee "$LOG_DIR/realuw_mae_vit_base_mmpretrain.log"

echo "Done. Checkpoint dir: $WORK_DIR"
