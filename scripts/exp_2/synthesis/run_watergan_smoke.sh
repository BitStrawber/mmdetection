#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_smoke/watergan}"
GPU="${GPU:-2}"
SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT:-100}"
SMOKE_VAL_LIMIT="${SMOKE_VAL_LIMIT:-30}"
WATERGAN_EPOCH="${WATERGAN_EPOCH:-2}"
WATERGAN_BATCH_SIZE="${WATERGAN_BATCH_SIZE:-4}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "WaterGAN smoke prepare + train"
echo "========================================="
echo "SYN_ROOT:          ${SYN_ROOT}"
echo "WORK_ROOT:         ${WORK_ROOT}"
echo "GPU:               ${GPU}"
echo "SMOKE_TRAIN_LIMIT: ${SMOKE_TRAIN_LIMIT}"
echo "SMOKE_VAL_LIMIT:   ${SMOKE_VAL_LIMIT}"
echo "WATERGAN_EPOCH:    ${WATERGAN_EPOCH}"
echo "WATERGAN_BATCH:    ${WATERGAN_BATCH_SIZE}"
echo "OMP_THREADS:       ${OMP_NUM_THREADS}"
echo "LOG_DIR:           ${LOG_DIR}"
echo "========================================="

MODE=smoke \
METHODS="watergan" \
SPLITS="train val" \
GPU="${GPU}" \
SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT}" \
SMOKE_VAL_LIMIT="${SMOKE_VAL_LIMIT}" \
SYN_ROOT="${SYN_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh \
  2>&1 | tee "${LOG_DIR}/prepare.log"

DATA_NAME="imagenet_ruod_watergan_train_smoke_ssd"
DATA_ROOT="${WORK_ROOT}/watergan/datasets/${DATA_NAME}"

DATA_NAME="${DATA_NAME}" \
DATA_ROOT="${DATA_ROOT}" \
GPU="${GPU}" \
EPOCH="${WATERGAN_EPOCH}" \
BATCH_SIZE="${WATERGAN_BATCH_SIZE}" \
TRAIN_SIZE="${SMOKE_TRAIN_LIMIT}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
bash scripts/exp_2/synthesis/run_watergan_train.sh \
  2>&1 | tee "${LOG_DIR}/train.log"

echo "WaterGAN smoke checkpoint/sample dirs are inside:"
echo "  /home/fcp/xcx/exp_2/syn/WaterGAN"
echo "Note: WaterGAN batch generation wrapper is not finalized yet."
