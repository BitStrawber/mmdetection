#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_smoke/cut}"
GPU="${GPU:-2}"
GPU_IDS="${GPU_IDS:-${GPU}}"
SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT:-100}"
SMOKE_VAL_LIMIT="${SMOKE_VAL_LIMIT:-30}"
CUT_EPOCHS="${CUT_EPOCHS:-2}"
CUT_BATCH_SIZE="${CUT_BATCH_SIZE:-1}"
CUT_NUM_THREADS="${CUT_NUM_THREADS:-8}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "CUT smoke train + generate"
echo "========================================="
echo "SYN_ROOT:          ${SYN_ROOT}"
echo "SOURCE_ROOT:       ${SOURCE_ROOT}"
echo "WORK_ROOT:         ${WORK_ROOT}"
echo "GPU:               ${GPU}"
echo "GPU_IDS:           ${GPU_IDS}"
echo "SMOKE_TRAIN_LIMIT: ${SMOKE_TRAIN_LIMIT}"
echo "SMOKE_VAL_LIMIT:   ${SMOKE_VAL_LIMIT}"
echo "CUT_EPOCHS:        ${CUT_EPOCHS}"
echo "CUT_BATCH_SIZE:    ${CUT_BATCH_SIZE}"
echo "CUT_NUM_THREADS:   ${CUT_NUM_THREADS}"
echo "LOG_DIR:           ${LOG_DIR}"
echo "========================================="

MODE=smoke \
METHODS="cut" \
SPLITS="train val" \
GPU="${GPU}" \
SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT}" \
SMOKE_VAL_LIMIT="${SMOKE_VAL_LIMIT}" \
SYN_ROOT="${SYN_ROOT}" \
SOURCE_ROOT="${SOURCE_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh \
  2>&1 | tee "${LOG_DIR}/prepare.log"

DATA_NAME="imagenet_ruod_cut_smoke_ssd"
DATA_ROOT="${WORK_ROOT}/cut/datasets/${DATA_NAME}"

DATA_NAME="${DATA_NAME}" \
DATA_ROOT="${DATA_ROOT}" \
EXP_NAME="${DATA_NAME}" \
GPU_IDS="${GPU_IDS}" \
BATCH_SIZE="${CUT_BATCH_SIZE}" \
NUM_THREADS="${CUT_NUM_THREADS}" \
N_EPOCHS="${CUT_EPOCHS}" \
N_EPOCHS_DECAY=0 \
SAVE_EPOCH_FREQ=1 \
bash scripts/exp_2/synthesis/run_cut_train.sh \
  2>&1 | tee "${LOG_DIR}/train.log"

DATA_NAME="${DATA_NAME}" \
DATA_ROOT="${DATA_ROOT}" \
EXP_NAME="${DATA_NAME}" \
SPLIT=val \
GPU_IDS="${GPU_IDS}" \
NUM_TEST="${SMOKE_VAL_LIMIT}" \
RESULTS_ROOT="${WORK_ROOT}/cut/results/${DATA_NAME}_val" \
RESTORE_DIR="${WORK_ROOT}/cut/generated/val" \
MANIFEST="${DATA_ROOT}/manifests/testA_manifest.jsonl" \
bash scripts/exp_2/synthesis/run_cut_generate.sh \
  2>&1 | tee "${LOG_DIR}/generate_val.log"

echo "CUT smoke output:"
echo "  ${WORK_ROOT}/cut/generated/val"
