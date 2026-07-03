#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN}"
DATA_NAME="${DATA_NAME:-imagenet_ruod_watergan_train_smoke_ssd}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/watergan/datasets/${DATA_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_smoke/watergan_tiny_check}"

GPU="${GPU:-2}"
EPOCH="${EPOCH:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TRAIN_SIZE="${TRAIN_SIZE:-10}"
SAVE_EPOCH="${SAVE_EPOCH:-1}"
PREPARE_IF_MISSING="${PREPARE_IF_MISSING:-0}"
PREPARE_AIR_LIMIT="${PREPARE_AIR_LIMIT:-100}"
PREPARE_WATER_LIMIT="${PREPARE_WATER_LIMIT:-100}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoint_${DATA_NAME}}"
SAMPLE_DIR="${SAMPLE_DIR:-samples_${DATA_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-results_${DATA_NAME}}"

mkdir -p "${LOG_DIR}"

count_files() {
  local path="$1"
  find "${path}" -maxdepth 1 -type f 2>/dev/null | wc -l
}

check_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Error: ${label} not found: ${path}" >&2
    return 1
  fi
}

echo "========================================="
echo "WaterGAN tiny usability check"
echo "========================================="
echo "WATERGAN_DIR:       ${WATERGAN_DIR}"
echo "DATA_ROOT:          ${DATA_ROOT}"
echo "SOURCE_ROOT:        ${SOURCE_ROOT}"
echo "DATA_NAME:          ${DATA_NAME}"
echo "GPU:                ${GPU}"
echo "EPOCH:              ${EPOCH}"
echo "BATCH_SIZE:         ${BATCH_SIZE}"
echo "TRAIN_SIZE:         ${TRAIN_SIZE}"
echo "PREPARE_IF_MISSING: ${PREPARE_IF_MISSING}"
echo "LOG_DIR:            ${LOG_DIR}"
echo "CHECKPOINT_DIR:     ${CHECKPOINT_DIR}"
echo "SAMPLE_DIR:         ${SAMPLE_DIR}"
echo "RESULTS_DIR:        ${RESULTS_DIR}"
echo "========================================="

check_dir "${WATERGAN_DIR}" "WaterGAN repo"

echo
echo "Step 1/4: Patch WaterGAN TF1.15/Python3 compatibility"
WATERGAN_DIR="${WATERGAN_DIR}" \
bash scripts/exp_2/synthesis/patch_watergan_tf15_compat.sh \
  2>&1 | tee "${LOG_DIR}/patch.log"

if grep -RInE "batch_idxs\\s*=.* / config\\.batch_size" "${WATERGAN_DIR}"/*.py >/tmp/watergan_batch_divisions.txt 2>/dev/null; then
  echo "Error: Python2-style batch index divisions remain:" >&2
  cat /tmp/watergan_batch_divisions.txt >&2
  exit 1
fi

echo
echo "Step 2/4: Check or prepare tiny WaterGAN dataset"
if [[ ! -d "${DATA_ROOT}/air_images" || ! -d "${DATA_ROOT}/air_depth" || ! -d "${DATA_ROOT}/water_images" ]]; then
  if [[ "${PREPARE_IF_MISSING}" != "1" ]]; then
    echo "Error: prepared WaterGAN dataset is missing: ${DATA_ROOT}" >&2
    echo "Run this first in an environment with PyTorch/MegaDepth available, or set PREPARE_IF_MISSING=1:" >&2
    echo "  MODE=smoke METHODS=watergan bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh" >&2
    exit 1
  fi

  echo "Prepared dataset missing; building smoke WaterGAN dataset now."
  MODE=smoke \
  METHODS="watergan" \
  SPLITS="train val" \
  GPU="${GPU}" \
  SMOKE_TRAIN_LIMIT="${PREPARE_AIR_LIMIT}" \
  SMOKE_VAL_LIMIT=30 \
  SYN_ROOT="${SYN_ROOT}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  WORK_ROOT="${WORK_ROOT}" \
  bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh \
    2>&1 | tee "${LOG_DIR}/prepare.log"
fi

check_dir "${DATA_ROOT}/air_images" "prepared air_images"
check_dir "${DATA_ROOT}/air_depth" "prepared air_depth"
check_dir "${DATA_ROOT}/water_images" "prepared water_images"

air_count="$(count_files "${DATA_ROOT}/air_images")"
depth_count="$(count_files "${DATA_ROOT}/air_depth")"
water_count="$(count_files "${DATA_ROOT}/water_images")"
echo "Prepared dataset counts:"
echo "  air_images:   ${air_count}"
echo "  air_depth:    ${depth_count}"
echo "  water_images: ${water_count}"

if [[ "${air_count}" -lt "${TRAIN_SIZE}" || "${depth_count}" -lt "${TRAIN_SIZE}" || "${water_count}" -lt "${BATCH_SIZE}" ]]; then
  echo "Error: dataset is too small for TRAIN_SIZE=${TRAIN_SIZE}, BATCH_SIZE=${BATCH_SIZE}" >&2
  exit 1
fi

echo
echo "Step 3/4: Run WaterGAN tiny training"
DATA_NAME="${DATA_NAME}" \
DATA_ROOT="${DATA_ROOT}" \
GPU="${GPU}" \
EPOCH="${EPOCH}" \
BATCH_SIZE="${BATCH_SIZE}" \
TRAIN_SIZE="${TRAIN_SIZE}" \
SAVE_EPOCH="${SAVE_EPOCH}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
SAMPLE_DIR="${SAMPLE_DIR}" \
RESULTS_DIR="${RESULTS_DIR}" \
bash scripts/exp_2/synthesis/run_watergan_train.sh \
  2>&1 | tee "${LOG_DIR}/train.log"

echo
echo "Step 4/4: Verify WaterGAN outputs"
checkpoint_path="${WATERGAN_DIR}/${CHECKPOINT_DIR}"
sample_path="${WATERGAN_DIR}/${SAMPLE_DIR}"
results_path="${WATERGAN_DIR}/${RESULTS_DIR}"

echo "Checkpoint path: ${checkpoint_path}"
echo "Sample path:     ${sample_path}"
echo "Results path:    ${results_path}"

checkpoint_files="$(find "${checkpoint_path}" -type f 2>/dev/null | wc -l || true)"
sample_files="$(find "${sample_path}" -type f 2>/dev/null | wc -l || true)"
result_files="$(find "${results_path}" -type f 2>/dev/null | wc -l || true)"

echo "Output file counts:"
echo "  checkpoint files: ${checkpoint_files}"
echo "  sample files:     ${sample_files}"
echo "  result files:     ${result_files}"

if [[ "${checkpoint_files}" -eq 0 && "${sample_files}" -eq 0 && "${result_files}" -eq 0 ]]; then
  echo "Error: WaterGAN command exited but no checkpoint/sample/result files were found." >&2
  exit 1
fi

echo
echo "WaterGAN tiny usability check completed."
echo "Logs:       ${LOG_DIR}"
echo "Checkpoint: ${checkpoint_path}"
echo "Samples:    ${sample_path}"
echo "Results:    ${results_path}"
