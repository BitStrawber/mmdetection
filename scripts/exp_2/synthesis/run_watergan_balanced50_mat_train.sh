#!/usr/bin/env bash
set -euo pipefail

# Rebuild the class-balanced 50k WaterGAN dataset and train the MHL model for
# 26 epochs. The MAT files use the original single "depth" variable.
#
# Default run:
#   conda activate /media/SSD1/conda_envs/watergan_tf1
#   bash scripts/exp_2/synthesis/run_watergan_balanced50_mat_train.sh
#
# Prepare only:
#   RUN_TRAIN=0 bash scripts/exp_2/synthesis/run_watergan_balanced50_mat_train.sh
#
# Train an already prepared dataset:
#   RUN_PREPARE=0 bash scripts/exp_2/synthesis/run_watergan_balanced50_mat_train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN}"

DATA_NAME="${DATA_NAME:-imagenet_ruod_watergan_train_balanced50_mat_ssd}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/watergan/datasets/${DATA_NAME}}"
SOURCE_DIR="${SOURCE_DIR:-${SOURCE_ROOT}/watergan/source/train}"
DEPTH_DIR="${DEPTH_DIR:-/media/SSD1/XCX/exp_2/depthanything_v2_maps/watergan/train}"
WATER_SOURCE="${WATER_SOURCE:-/media/SSD1/XCX/exp_2/RUOD/coco/train}"

GPU="${GPU:-2}"
EPOCH="${EPOCH:-26}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_SIZE="${TRAIN_SIZE:-50000}"
AIR_PER_CLASS="${AIR_PER_CLASS:-50}"
WATER_REPEAT_TO="${WATER_REPEAT_TO:-50000}"
SAMPLE_SEED="${SAMPLE_SEED:-2026}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
SAVE_EPOCH="${SAVE_EPOCH:-25}"
PREPARE_WORKERS="${PREPARE_WORKERS:-16}"
WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS:-16}"
WATERGAN_LOG_EVERY="${WATERGAN_LOG_EVERY:-10}"
WATERGAN_THROTTLE_DIAGNOSTICS="${WATERGAN_THROTTLE_DIAGNOSTICS:-1}"
WATERGAN_MAX_TO_KEEP="${WATERGAN_MAX_TO_KEEP:-5}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RESUME_PREPARE="${RESUME_PREPARE:-1}"
VERIFY_EXISTING="${VERIFY_EXISTING:-1}"
OVERWRITE_PREPARE="${OVERWRITE_PREPARE:-0}"
AUTO_PATCH="${AUTO_PATCH:-0}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoint_${DATA_NAME}_bs${BATCH_SIZE}_e${EPOCH}}"
SAMPLE_DIR="${SAMPLE_DIR:-samples_${DATA_NAME}_bs${BATCH_SIZE}_e${EPOCH}}"
RESULTS_DIR="${RESULTS_DIR:-${SYN_ROOT}/watergan/results/${DATA_NAME}_bs${BATCH_SIZE}_e${EPOCH}_gpu${GPU}}"
LOG_DIR="${LOG_DIR:-${SYN_ROOT}/watergan/logs}"

count_entries() {
  local path="$1"
  find "${path}" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null |
    wc -l |
    tr -d ' '
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Error: ${label} not found: ${path}" >&2
    exit 1
  fi
}

echo "========================================="
echo "WaterGAN balanced50 MAT train"
echo "========================================="
echo "DATA_NAME:          ${DATA_NAME}"
echo "DATA_ROOT:          ${DATA_ROOT}"
echo "SOURCE_DIR:         ${SOURCE_DIR}"
echo "DEPTH_DIR:          ${DEPTH_DIR}"
echo "WATER_SOURCE:       ${WATER_SOURCE}"
echo "WATERGAN_DIR:       ${WATERGAN_DIR}"
echo "GPU:                ${GPU}"
echo "EPOCH:              ${EPOCH}"
echo "BATCH_SIZE:         ${BATCH_SIZE}"
echo "TRAIN_SIZE:         ${TRAIN_SIZE}"
echo "AIR_PER_CLASS:      ${AIR_PER_CLASS}"
echo "WATER_REPEAT_TO:    ${WATER_REPEAT_TO}"
echo "SAMPLE_SEED:        ${SAMPLE_SEED}"
echo "DEPTH_FORMAT:       mat"
echo "MAT_LAYOUT:         official"
echo "PREPARE_WORKERS:    ${PREPARE_WORKERS}"
echo "IO_WORKERS:         ${WATERGAN_IO_WORKERS}"
echo "MAX_TO_KEEP:        ${WATERGAN_MAX_TO_KEEP}"
echo "RUN_PREPARE:        ${RUN_PREPARE}"
echo "RUN_TRAIN:          ${RUN_TRAIN}"
echo "RESUME_PREPARE:     ${RESUME_PREPARE}"
echo "OVERWRITE_PREPARE:  ${OVERWRITE_PREPARE}"
echo "CHECKPOINT_DIR:     ${WATERGAN_DIR}/${CHECKPOINT_DIR}"
echo "LOG:                ${LOG_DIR}/${DATA_NAME}_train.log"
echo "========================================="

require_path "${SOURCE_DIR}" "ImageNet WaterGAN source"
require_path "${DEPTH_DIR}" "precomputed depth source"
require_path "${WATER_SOURCE}" "SSD RUOD source"
require_path "${WATERGAN_DIR}/mainmhl.py" "WaterGAN main entry"

if [[ "${RUN_PREPARE}" == "1" ]]; then
  SOURCE_ROOT="${SOURCE_ROOT}" \
  WORK_ROOT="${WORK_ROOT}" \
  SOURCE_DIR="${SOURCE_DIR}" \
  DEPTH_DIR="${DEPTH_DIR}" \
  WATER_SOURCE="${WATER_SOURCE}" \
  DATA_NAME="${DATA_NAME}" \
  DATA_ROOT="${DATA_ROOT}" \
  SPLIT=train \
  GPU="${GPU}" \
  RUN_DEPTH=0 \
  AIR_LIMIT=0 \
  WATER_LIMIT=0 \
  AIR_PER_CLASS="${AIR_PER_CLASS}" \
  WATER_REPEAT_TO="${WATER_REPEAT_TO}" \
  SAMPLE_SEED="${SAMPLE_SEED}" \
  DEPTH_FORMAT=mat \
  MAT_LAYOUT=official \
  NUM_WORKERS="${PREPARE_WORKERS}" \
  RESUME="${RESUME_PREPARE}" \
  VERIFY_EXISTING="${VERIFY_EXISTING}" \
  OVERWRITE="${OVERWRITE_PREPARE}" \
  bash scripts/exp_2/synthesis/run_watergan_prepare_dataset.sh
fi

require_path "${DATA_ROOT}/air_images" "prepared air_images"
require_path "${DATA_ROOT}/air_depth" "prepared air_depth"
require_path "${DATA_ROOT}/water_images" "prepared water_images"
require_path "${DATA_ROOT}/watergan_air_manifest.jsonl" "prepared manifest"

air_count="$(count_entries "${DATA_ROOT}/air_images")"
depth_count="$(count_entries "${DATA_ROOT}/air_depth")"
water_count="$(count_entries "${DATA_ROOT}/water_images")"
manifest_count="$(
  wc -l < "${DATA_ROOT}/watergan_air_manifest.jsonl" 2>/dev/null ||
    echo 0
)"
manifest_count="$(echo "${manifest_count}" | tr -d ' ')"

echo
echo "Prepared dataset counts:"
printf '  air_images:   %s / %s\n' "${air_count}" "${TRAIN_SIZE}"
printf '  air_depth:    %s / %s\n' "${depth_count}" "${TRAIN_SIZE}"
printf '  water_images: %s / %s\n' "${water_count}" "${TRAIN_SIZE}"
printf '  manifest:     %s / %s\n' "${manifest_count}" "${TRAIN_SIZE}"

if [[ "${air_count}" != "${TRAIN_SIZE}" ||
      "${depth_count}" != "${TRAIN_SIZE}" ||
      "${water_count}" != "${TRAIN_SIZE}" ||
      "${manifest_count}" != "${TRAIN_SIZE}" ]]; then
  echo "Error: balanced50 preparation is incomplete; training was not started." >&2
  exit 1
fi

python - \
  "${DATA_ROOT}/air_depth" \
  "${DATA_ROOT}/watergan_air_manifest.jsonl" \
  "${AIR_PER_CLASS}" \
  "${TRAIN_SIZE}" <<'PY'
from collections import Counter
import json
from pathlib import Path
import sys

from scipy.io import loadmat

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
per_class = int(sys.argv[3])
expected_total = int(sys.argv[4])

sample = next(root.glob('*.mat'))
values = loadmat(str(sample))
keys = sorted(key for key in values if not key.startswith('__'))
if keys != ['depth']:
    raise RuntimeError(
        'Expected official MAT keys ["depth"], got {} in {}'.format(keys, sample)
    )
print('MAT layout check: OK ({}, keys={})'.format(sample, keys))

counts = Counter()
with manifest.open('r', encoding='utf-8') as handle:
    for line in handle:
        if line.strip():
            counts[json.loads(line)['synset']] += 1

if sum(counts.values()) != expected_total:
    raise RuntimeError(
        'Manifest has {} rows, expected {}'.format(
            sum(counts.values()), expected_total
        )
    )
bad = {name: count for name, count in counts.items() if count != per_class}
expected_classes = expected_total // per_class
if len(counts) != expected_classes or bad:
    raise RuntimeError(
        'Expected {} classes x {} images; got {} classes and {} bad counts'.format(
            expected_classes, per_class, len(counts), len(bad)
        )
    )
print(
    'Class balance check: OK (classes={}, per_class={}, total={})'.format(
        len(counts), per_class, sum(counts.values())
    )
)
PY

if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo
  echo "Preparation complete; RUN_TRAIN=${RUN_TRAIN}, so training was skipped."
  exit 0
fi

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

DATA_NAME="${DATA_NAME}" \
DATA_ROOT="${DATA_ROOT}" \
SYN_ROOT="${SYN_ROOT}" \
WATERGAN_DIR="${WATERGAN_DIR}" \
GPU="${GPU}" \
EPOCH="${EPOCH}" \
BATCH_SIZE="${BATCH_SIZE}" \
TRAIN_SIZE="${TRAIN_SIZE}" \
NUM_SAMPLES="${NUM_SAMPLES}" \
SAVE_EPOCH="${SAVE_EPOCH}" \
AUTO_PATCH="${AUTO_PATCH}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
SAMPLE_DIR="${SAMPLE_DIR}" \
RESULTS_DIR="${RESULTS_DIR}" \
LOG_DIR="${LOG_DIR}" \
WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS}" \
WATERGAN_LOG_EVERY="${WATERGAN_LOG_EVERY}" \
WATERGAN_THROTTLE_DIAGNOSTICS="${WATERGAN_THROTTLE_DIAGNOSTICS}" \
WATERGAN_MAX_TO_KEEP="${WATERGAN_MAX_TO_KEEP}" \
PYTHONUNBUFFERED=1 \
bash scripts/exp_2/synthesis/run_watergan_train.sh

echo
echo "Training finished."
echo "Checkpoint: ${WATERGAN_DIR}/${CHECKPOINT_DIR}"
echo "Log:        ${LOG_DIR}/${DATA_NAME}_train.log"
