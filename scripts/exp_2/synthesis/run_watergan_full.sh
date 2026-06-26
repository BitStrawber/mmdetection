#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_full/watergan}"
GPU="${GPU:-2}"
WATERGAN_EPOCH="${WATERGAN_EPOCH:-26}"
WATERGAN_BATCH_SIZE="${WATERGAN_BATCH_SIZE:-4}"
WATERGAN_TRAIN_SIZE="${WATERGAN_TRAIN_SIZE:-0}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "WaterGAN full prepare + train"
echo "========================================="
echo "SYN_ROOT:             ${SYN_ROOT}"
echo "WORK_ROOT:            ${WORK_ROOT}"
echo "GPU:                  ${GPU}"
echo "WATERGAN_EPOCH:       ${WATERGAN_EPOCH}"
echo "WATERGAN_BATCH_SIZE:  ${WATERGAN_BATCH_SIZE}"
echo "WATERGAN_TRAIN_SIZE:  ${WATERGAN_TRAIN_SIZE}"
echo "LOG_DIR:              ${LOG_DIR}"
echo "========================================="

MODE=full \
METHODS="watergan" \
SPLITS="train val" \
GPU="${GPU}" \
FULL_LIMIT=0 \
SYN_ROOT="${SYN_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh \
  2>&1 | tee "${LOG_DIR}/prepare.log"

DATA_NAME="imagenet_ruod_watergan_train_full_ssd"
DATA_ROOT="${WORK_ROOT}/watergan/datasets/${DATA_NAME}"

DATA_NAME="${DATA_NAME}" \
DATA_ROOT="${DATA_ROOT}" \
GPU="${GPU}" \
EPOCH="${WATERGAN_EPOCH}" \
BATCH_SIZE="${WATERGAN_BATCH_SIZE}" \
TRAIN_SIZE="${WATERGAN_TRAIN_SIZE}" \
bash scripts/exp_2/synthesis/run_watergan_train.sh \
  2>&1 | tee "${LOG_DIR}/train.log"

echo "WaterGAN full checkpoint/sample dirs are inside:"
echo "  /home/fcp/xcx/exp_2/syn/WaterGAN"
echo "Note: WaterGAN batch generation wrapper is not finalized yet."
