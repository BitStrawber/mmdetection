#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_full/watergan}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/watergan/source}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps/watergan}"
WATER_SOURCE="${WATER_SOURCE:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
GPU="${GPU:-2}"
WATERGAN_EPOCH="${WATERGAN_EPOCH:-26}"
WATERGAN_BATCH_SIZE="${WATERGAN_BATCH_SIZE:-8}"
WATERGAN_TRAIN_SIZE="${WATERGAN_TRAIN_SIZE:-50000}"
WATERGAN_AIR_PER_CLASS="${WATERGAN_AIR_PER_CLASS:-50}"
WATERGAN_WATER_REPEAT_TO="${WATERGAN_WATER_REPEAT_TO:-50000}"
WATERGAN_SAMPLE_SEED="${WATERGAN_SAMPLE_SEED:-2026}"
PREPARE_SPLITS="${PREPARE_SPLITS:-train}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "WaterGAN full prepare + train"
echo "========================================="
echo "SYN_ROOT:             ${SYN_ROOT}"
echo "WORK_ROOT:            ${WORK_ROOT}"
echo "SOURCE_ROOT:          ${SOURCE_ROOT}"
echo "DEPTH_ROOT:           ${DEPTH_ROOT}"
echo "WATER_SOURCE:         ${WATER_SOURCE}"
echo "GPU:                  ${GPU}"
echo "WATERGAN_EPOCH:       ${WATERGAN_EPOCH}"
echo "WATERGAN_BATCH_SIZE:  ${WATERGAN_BATCH_SIZE}"
echo "WATERGAN_TRAIN_SIZE:  ${WATERGAN_TRAIN_SIZE}"
echo "WATERGAN_AIR_PER_CLASS: ${WATERGAN_AIR_PER_CLASS}"
echo "WATERGAN_WATER_REPEAT_TO: ${WATERGAN_WATER_REPEAT_TO}"
echo "WATERGAN_SAMPLE_SEED: ${WATERGAN_SAMPLE_SEED}"
echo "PREPARE_SPLITS:       ${PREPARE_SPLITS}"
echo "OMP_THREADS:          ${OMP_NUM_THREADS}"
echo "LOG_DIR:              ${LOG_DIR}"
echo "========================================="

for split in ${PREPARE_SPLITS}; do
  if [[ "${split}" == "train" && "${WATERGAN_AIR_PER_CLASS}" != "0" ]]; then
    DATA_NAME="imagenet_ruod_watergan_train_balanced${WATERGAN_AIR_PER_CLASS}_ssd"
    AIR_PER_CLASS="${WATERGAN_AIR_PER_CLASS}"
    WATER_REPEAT_TO="${WATERGAN_WATER_REPEAT_TO}"
  else
    DATA_NAME="imagenet_ruod_watergan_${split}_full_ssd"
    AIR_PER_CLASS=0
    WATER_REPEAT_TO=0
  fi
  DATA_ROOT="${WORK_ROOT}/watergan/datasets/${DATA_NAME}"
  SOURCE_DIR="${SOURCE_ROOT}/${split}" \
  DEPTH_DIR="${DEPTH_ROOT}/${split}" \
  WATER_SOURCE="${WATER_SOURCE}" \
  DATA_NAME="${DATA_NAME}" \
  DATA_ROOT="${DATA_ROOT}" \
  SPLIT="${split}" \
  GPU="${GPU}" \
  RUN_DEPTH=0 \
  AIR_LIMIT=0 \
  WATER_LIMIT=0 \
  AIR_PER_CLASS="${AIR_PER_CLASS}" \
  WATER_REPEAT_TO="${WATER_REPEAT_TO}" \
  SAMPLE_SEED="${WATERGAN_SAMPLE_SEED}" \
  bash scripts/exp_2/synthesis/run_watergan_prepare_dataset.sh \
    2>&1 | tee "${LOG_DIR}/prepare_${split}.log"
done

if [[ "${WATERGAN_AIR_PER_CLASS}" != "0" ]]; then
  DATA_NAME="imagenet_ruod_watergan_train_balanced${WATERGAN_AIR_PER_CLASS}_ssd"
else
  DATA_NAME="imagenet_ruod_watergan_train_full_ssd"
fi
DATA_ROOT="${WORK_ROOT}/watergan/datasets/${DATA_NAME}"

DATA_NAME="${DATA_NAME}" \
DATA_ROOT="${DATA_ROOT}" \
GPU="${GPU}" \
EPOCH="${WATERGAN_EPOCH}" \
BATCH_SIZE="${WATERGAN_BATCH_SIZE}" \
TRAIN_SIZE="${WATERGAN_TRAIN_SIZE}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
bash scripts/exp_2/synthesis/run_watergan_train.sh \
  2>&1 | tee "${LOG_DIR}/train.log"

echo "WaterGAN full checkpoint/sample dirs are inside:"
echo "  /home/fcp/xcx/exp_2/syn/WaterGAN"
echo "Note: WaterGAN batch generation wrapper is not finalized yet."
