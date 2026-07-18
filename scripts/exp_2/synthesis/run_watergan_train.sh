#!/usr/bin/env bash
set -euo pipefail

# Train the original WaterGAN MHL variant on prepared ImageNet/RUOD data.
# Run this inside the TensorFlow-1 WaterGAN environment:
#
#   conda activate /media/SSD1/conda_envs/watergan_tf1
#   DATA_NAME=imagenet_ruod_watergan_train_smoke GPU=2 EPOCH=2 BATCH_SIZE=4 \
#     bash scripts/exp_2/synthesis/run_watergan_train.sh
#
# The script links the prepared dataset into WaterGAN/data because the original
# code expects dataset names rather than arbitrary absolute paths.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN}"
DATA_NAME="${DATA_NAME:-imagenet_ruod_watergan_train_smoke}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/watergan/datasets/${DATA_NAME}}"
GPU="${GPU:-2}"

EPOCH="${EPOCH:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TRAIN_SIZE="${TRAIN_SIZE:-1000}"
AUTO_PATCH="${AUTO_PATCH:-1}"
SAVE_EPOCH="${SAVE_EPOCH:-1}"
AIR_WIDTH="${AIR_WIDTH:-640}"
AIR_HEIGHT="${AIR_HEIGHT:-480}"
WATER_WIDTH="${WATER_WIDTH:-1360}"
WATER_HEIGHT="${WATER_HEIGHT:-1024}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-64}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-48}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
BETA1="${BETA1:-0.5}"
WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS:-16}"
WATERGAN_LOG_EVERY="${WATERGAN_LOG_EVERY:-10}"
WATERGAN_THROTTLE_DIAGNOSTICS="${WATERGAN_THROTTLE_DIAGNOSTICS:-1}"
export WATERGAN_IO_WORKERS WATERGAN_LOG_EVERY WATERGAN_THROTTLE_DIAGNOSTICS

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoint_${DATA_NAME}}"
SAMPLE_DIR="${SAMPLE_DIR:-samples_${DATA_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-results_${DATA_NAME}}"
LOG_DIR="${LOG_DIR:-${SYN_ROOT}/watergan/logs}"

check_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Error: ${label} not found: ${path}" >&2
    exit 1
  fi
}

count_files() {
  local path="$1"
  find "${path}" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l | tr -d ' '
}

echo "========================================="
echo "Train WaterGAN ImageNet + RUOD"
echo "========================================="
echo "WATERGAN_DIR:   ${WATERGAN_DIR}"
echo "DATA_ROOT:      ${DATA_ROOT}"
echo "DATA_NAME:      ${DATA_NAME}"
echo "GPU:            ${GPU}"
echo "EPOCH:          ${EPOCH}"
echo "BATCH_SIZE:     ${BATCH_SIZE}"
echo "TRAIN_SIZE:     ${TRAIN_SIZE} (0 means auto from prepared data)"
echo "IO_WORKERS:     ${WATERGAN_IO_WORKERS}"
echo "LOG_EVERY:      ${WATERGAN_LOG_EVERY}"
echo "THROTTLE_DIAG:  ${WATERGAN_THROTTLE_DIAGNOSTICS}"
echo "AUTO_PATCH:     ${AUTO_PATCH}"
echo "AIR_SIZE:       ${AIR_WIDTH}x${AIR_HEIGHT}"
echo "WATER_SIZE:     ${WATER_WIDTH}x${WATER_HEIGHT}"
echo "OUTPUT_SIZE:    ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}"
echo "CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
echo "SAMPLE_DIR:     ${SAMPLE_DIR}"
echo "RESULTS_DIR:    ${RESULTS_DIR}"
echo "========================================="
echo

check_path "${WATERGAN_DIR}/mainmhl.py" "WaterGAN mainmhl.py"
check_path "${DATA_ROOT}/air_images" "prepared air_images"
check_path "${DATA_ROOT}/air_depth" "prepared air_depth"
check_path "${DATA_ROOT}/water_images" "prepared water_images"

if [[ "${AUTO_PATCH}" == "1" ]]; then
  WATERGAN_DIR="${WATERGAN_DIR}" bash scripts/exp_2/synthesis/patch_watergan_tf15_compat.sh
  echo
fi

air_count="$(count_files "${DATA_ROOT}/air_images")"
depth_count="$(count_files "${DATA_ROOT}/air_depth")"
water_count="$(count_files "${DATA_ROOT}/water_images")"

effective_train_size="${TRAIN_SIZE}"
if [[ "${TRAIN_SIZE}" == "0" ]]; then
  effective_train_size="${air_count}"
  if (( depth_count < effective_train_size )); then
    effective_train_size="${depth_count}"
  fi
  if (( water_count < effective_train_size )); then
    effective_train_size="${water_count}"
  fi
fi

if (( effective_train_size < BATCH_SIZE )); then
  echo "Error: effective_train_size (${effective_train_size}) is smaller than BATCH_SIZE (${BATCH_SIZE})." >&2
  echo "Check prepared data counts or lower BATCH_SIZE." >&2
  exit 1
fi

echo "Prepared dataset counts:"
echo "  air_images:   ${air_count}"
echo "  air_depth:    ${depth_count}"
echo "  water_images: ${water_count}"
echo "  effective_train_size: ${effective_train_size}"
echo

mkdir -p "${WATERGAN_DIR}/data" "${LOG_DIR}"

ln -sfn "${DATA_ROOT}/air_images" "${WATERGAN_DIR}/data/${DATA_NAME}_air_images"
ln -sfn "${DATA_ROOT}/air_depth" "${WATERGAN_DIR}/data/${DATA_NAME}_air_depth"
ln -sfn "${DATA_ROOT}/water_images" "${WATERGAN_DIR}/data/${DATA_NAME}_water_images"

echo "WaterGAN data links:"
echo "  ${WATERGAN_DIR}/data/${DATA_NAME}_air_images -> ${DATA_ROOT}/air_images"
echo "  ${WATERGAN_DIR}/data/${DATA_NAME}_air_depth -> ${DATA_ROOT}/air_depth"
echo "  ${WATERGAN_DIR}/data/${DATA_NAME}_water_images -> ${DATA_ROOT}/water_images"
echo "Training log: ${LOG_DIR}/${DATA_NAME}_train.log"
echo

(
  cd "${WATERGAN_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" python mainmhl.py \
    --water_dataset "${DATA_NAME}_water_images" \
    --air_dataset "${DATA_NAME}_air_images" \
    --depth_dataset "${DATA_NAME}_air_depth" \
    --epoch "${EPOCH}" \
    --train_size "${effective_train_size}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --beta1 "${BETA1}" \
    --input_height "${AIR_HEIGHT}" \
    --input_width "${AIR_WIDTH}" \
    --input_water_height "${WATER_HEIGHT}" \
    --input_water_width "${WATER_WIDTH}" \
    --output_height "${OUTPUT_HEIGHT}" \
    --output_width "${OUTPUT_WIDTH}" \
    --save_epoch "${SAVE_EPOCH}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --sample_dir "${SAMPLE_DIR}" \
    --results_dir "${RESULTS_DIR}"
) 2>&1 | tee "${LOG_DIR}/${DATA_NAME}_train.log"

echo
echo "Done."
echo "Checkpoint dir: ${WATERGAN_DIR}/${CHECKPOINT_DIR}"
echo "Sample dir:     ${WATERGAN_DIR}/${SAMPLE_DIR}"
echo "Results dir:    ${WATERGAN_DIR}/${RESULTS_DIR}"
