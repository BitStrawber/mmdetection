#!/usr/bin/env bash
set -euo pipefail

# Prepare full 250k ImageNet train data for WaterGAN, retrain WaterGAN, generate
# fake underwater images, and restore flat fake_*.png results to ImageNet-style
# class folders.
#
# Output organization:
#   /media/HDD1/XCX/exp_2/synthetic_imagenet/watergan/generated/train/<synset>/<image>.png
#
# Notes:
# - Original WaterGAN is not a true multi-GPU DDP implementation. Setting
#   GPU="5,6,7" exposes three GPUs to TensorFlow, but speedup mainly comes from
#   increasing BATCH_SIZE. Generation is kept single-process to avoid output
#   name collisions in the original code.
# - BATCH_SIZE defaults to 16 because 250000 is divisible by 16.
#
# Example:
#   conda activate /media/SSD1/conda_envs/watergan_tf1
#   GPU="5,6,7" BATCH_SIZE=16 bash scripts/exp_2/synthesis/run_watergan_full_train_generate_train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/watergan/source}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps/watergan}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
WATER_SOURCE="${WATER_SOURCE:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN}"

DATA_NAME="${DATA_NAME:-imagenet_ruod_watergan_train_full250k_ssd}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/watergan/datasets/${DATA_NAME}}"
SOURCE_DIR="${SOURCE_DIR:-${SOURCE_ROOT}/train}"
DEPTH_DIR="${DEPTH_DIR:-${DEPTH_ROOT}/train}"

GPU="${GPU:-5,6,7}"
EPOCH="${EPOCH:-26}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_SIZE="${TRAIN_SIZE:-250000}"
NUM_SAMPLES="${NUM_SAMPLES:-250000}"
SAVE_EPOCH="${SAVE_EPOCH:-1}"

AIR_WIDTH="${AIR_WIDTH:-640}"
AIR_HEIGHT="${AIR_HEIGHT:-480}"
WATER_WIDTH="${WATER_WIDTH:-1360}"
WATER_HEIGHT="${WATER_HEIGHT:-1024}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-64}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-48}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
BETA1="${BETA1:-0.5}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoint_${DATA_NAME}}"
SAMPLE_DIR="${SAMPLE_DIR:-samples_${DATA_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-${SYN_ROOT}/watergan/results/${DATA_NAME}_gpu567}"
RESTORE_DIR="${RESTORE_DIR:-${SYN_ROOT}/watergan/generated/train}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_full/watergan_full250k}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_RESTORE="${RUN_RESTORE:-1}"
RUN_DEPTH="${RUN_DEPTH:-0}"
AUTO_PATCH="${AUTO_PATCH:-1}"
RESET_RESULTS="${RESET_RESULTS:-1}"
RESET_RESTORE="${RESET_RESTORE:-1}"
RESET_CHECKPOINT="${RESET_CHECKPOINT:-0}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-12}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS

mkdir -p "${LOG_DIR}"

count_images() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo 0
    return
  fi
  find "${path}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l
}

count_flat() {
  local path="$1"
  local pattern="$2"
  if [[ ! -d "${path}" ]]; then
    echo 0
    return
  fi
  find "${path}" -maxdepth 1 -type f -name "${pattern}" | wc -l
}

check_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Error: ${label} not found: ${path}" >&2
    exit 1
  fi
}

cat <<EOF
=========================================
WaterGAN full 250k train + generate
=========================================
SYN_ROOT:        ${SYN_ROOT}
WORK_ROOT:       ${WORK_ROOT}
WATERGAN_DIR:    ${WATERGAN_DIR}
DATA_NAME:       ${DATA_NAME}
DATA_ROOT:       ${DATA_ROOT}
SOURCE_DIR:      ${SOURCE_DIR}
DEPTH_DIR:       ${DEPTH_DIR}
WATER_SOURCE:    ${WATER_SOURCE}
GPU:             ${GPU}
EPOCH:           ${EPOCH}
BATCH_SIZE:      ${BATCH_SIZE}
TRAIN_SIZE:      ${TRAIN_SIZE}
NUM_SAMPLES:     ${NUM_SAMPLES}
RESULTS_DIR:     ${RESULTS_DIR}
RESTORE_DIR:     ${RESTORE_DIR}
CHECKPOINT_DIR:  ${CHECKPOINT_DIR}
RUN_PREPARE:     ${RUN_PREPARE}
RUN_TRAIN:       ${RUN_TRAIN}
RUN_GENERATE:    ${RUN_GENERATE}
RUN_RESTORE:     ${RUN_RESTORE}
RESET_RESULTS:   ${RESET_RESULTS}
RESET_RESTORE:   ${RESET_RESTORE}
RESET_CHECKPOINT:${RESET_CHECKPOINT}
=========================================
EOF

check_dir "${WATERGAN_DIR}" "WATERGAN_DIR"
check_dir "${SOURCE_DIR}" "WaterGAN train source"
check_dir "${DEPTH_DIR}" "WaterGAN train depth"
check_dir "${WATER_SOURCE}" "RUOD water source"

if [[ "${AUTO_PATCH}" == "1" ]]; then
  WATERGAN_DIR="${WATERGAN_DIR}" bash scripts/exp_2/synthesis/patch_watergan_tf15_compat.sh \
    2>&1 | tee "${LOG_DIR}/patch.log"
fi

if [[ "${RUN_PREPARE}" == "1" ]]; then
  SOURCE_DIR="${SOURCE_DIR}" \
  DEPTH_DIR="${DEPTH_DIR}" \
  WATER_SOURCE="${WATER_SOURCE}" \
  DATA_NAME="${DATA_NAME}" \
  DATA_ROOT="${DATA_ROOT}" \
  SPLIT=train \
  GPU="${GPU%%,*}" \
  RUN_DEPTH="${RUN_DEPTH}" \
  AIR_LIMIT=0 \
  WATER_LIMIT=0 \
  AIR_PER_CLASS=0 \
  WATER_REPEAT_TO="${TRAIN_SIZE}" \
  AIR_WIDTH="${AIR_WIDTH}" \
  AIR_HEIGHT="${AIR_HEIGHT}" \
  WATER_WIDTH="${WATER_WIDTH}" \
  WATER_HEIGHT="${WATER_HEIGHT}" \
  OVERWRITE=1 \
  bash scripts/exp_2/synthesis/run_watergan_prepare_dataset.sh \
    2>&1 | tee "${LOG_DIR}/prepare_train_full250k.log"
fi

if [[ "${RESET_CHECKPOINT}" == "1" ]]; then
  echo "Reset checkpoint/sample dirs for ${DATA_NAME}"
  rm -rf "${WATERGAN_DIR}/${CHECKPOINT_DIR}" "${WATERGAN_DIR}/${SAMPLE_DIR}"
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  DATA_NAME="${DATA_NAME}" \
  DATA_ROOT="${DATA_ROOT}" \
  WATERGAN_DIR="${WATERGAN_DIR}" \
  GPU="${GPU}" \
  EPOCH="${EPOCH}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  TRAIN_SIZE="${TRAIN_SIZE}" \
  SAVE_EPOCH="${SAVE_EPOCH}" \
  AIR_WIDTH="${AIR_WIDTH}" \
  AIR_HEIGHT="${AIR_HEIGHT}" \
  WATER_WIDTH="${WATER_WIDTH}" \
  WATER_HEIGHT="${WATER_HEIGHT}" \
  OUTPUT_WIDTH="${OUTPUT_WIDTH}" \
  OUTPUT_HEIGHT="${OUTPUT_HEIGHT}" \
  LEARNING_RATE="${LEARNING_RATE}" \
  BETA1="${BETA1}" \
  CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  SAMPLE_DIR="${SAMPLE_DIR}" \
  RESULTS_DIR="${RESULTS_DIR}" \
  LOG_DIR="${SYN_ROOT}/watergan/logs" \
  AUTO_PATCH=0 \
  bash scripts/exp_2/synthesis/run_watergan_train.sh \
    2>&1 | tee "${LOG_DIR}/train_full250k.log"
fi

mkdir -p "${WATERGAN_DIR}/data" "${RESULTS_DIR}"
ln -sfn "${DATA_ROOT}/air_images" "${WATERGAN_DIR}/data/${DATA_NAME}_air_images"
ln -sfn "${DATA_ROOT}/air_depth" "${WATERGAN_DIR}/data/${DATA_NAME}_air_depth"
ln -sfn "${DATA_ROOT}/water_images" "${WATERGAN_DIR}/data/${DATA_NAME}_water_images"

if [[ "${RESET_RESULTS}" == "1" ]]; then
  echo "Reset WaterGAN flat results: ${RESULTS_DIR}"
  rm -f "${RESULTS_DIR}"/fake_*.png "${RESULTS_DIR}"/air_*.png "${RESULTS_DIR}"/depth_*.mat
fi

if [[ "${RUN_GENERATE}" == "1" ]]; then
  (
    cd "${WATERGAN_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU}" python mainmhl.py \
      --is_train=False \
      --water_dataset "${DATA_NAME}_water_images" \
      --air_dataset "${DATA_NAME}_air_images" \
      --depth_dataset "${DATA_NAME}_air_depth" \
      --checkpoint_dir "${CHECKPOINT_DIR}" \
      --sample_dir "${SAMPLE_DIR}" \
      --results_dir "${RESULTS_DIR}" \
      --num_samples "${NUM_SAMPLES}" \
      --train_size "${TRAIN_SIZE}" \
      --batch_size "${BATCH_SIZE}" \
      --input_height "${AIR_HEIGHT}" \
      --input_width "${AIR_WIDTH}" \
      --input_water_height "${WATER_HEIGHT}" \
      --input_water_width "${WATER_WIDTH}" \
      --output_height "${OUTPUT_HEIGHT}" \
      --output_width "${OUTPUT_WIDTH}"
  ) 2>&1 | tee "${LOG_DIR}/generate_full250k.log"
fi

if [[ "${RESET_RESTORE}" == "1" ]]; then
  echo "Reset WaterGAN restored train output: ${RESTORE_DIR}"
  rm -rf "${RESTORE_DIR}"
fi

if [[ "${RUN_RESTORE}" == "1" ]]; then
  python tools/restore_watergan_fake.py \
    --manifest "${DATA_ROOT}/watergan_air_manifest.jsonl" \
    --results-dir "${RESULTS_DIR}" \
    --out-dir "${RESTORE_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --overwrite \
    2>&1 | tee "${LOG_DIR}/restore_full250k.log"
fi

cat <<EOF
=========================================
WaterGAN full train pipeline done
=========================================
flat_fake:       $(count_flat "${RESULTS_DIR}" 'fake_*.png')
restored_images: $(count_images "${RESTORE_DIR}")
expected_train:  ${TRAIN_SIZE}
RESULTS_DIR:     ${RESULTS_DIR}
RESTORE_DIR:     ${RESTORE_DIR}
=========================================
EOF