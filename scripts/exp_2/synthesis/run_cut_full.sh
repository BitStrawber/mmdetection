#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
RUOD_REF_SRC="${RUOD_REF_SRC:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_full/cut}"
GPU="${GPU:-2}"
GPU_IDS="${GPU_IDS:-${GPU}}"
CUT_EPOCHS="${CUT_EPOCHS:-100}"
CUT_EPOCHS_DECAY="${CUT_EPOCHS_DECAY:-100}"
CUT_BATCH_SIZE="${CUT_BATCH_SIZE:-1}"
CUT_NUM_THREADS="${CUT_NUM_THREADS:-12}"
CUT_NUM_TEST="${CUT_NUM_TEST:-100000000}"
COPY_MODE="${COPY_MODE:-copy}"
RESET_OUTPUTS="${RESET_OUTPUTS:-0}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "CUT full train + generate"
echo "========================================="
echo "SYN_ROOT:         ${SYN_ROOT}"
echo "SOURCE_ROOT:      ${SOURCE_ROOT}"
echo "WORK_ROOT:        ${WORK_ROOT}"
echo "RUOD_REF_SRC:     ${RUOD_REF_SRC}"
echo "GPU:              ${GPU}"
echo "GPU_IDS:          ${GPU_IDS}"
echo "CUT_EPOCHS:       ${CUT_EPOCHS}"
echo "CUT_DECAY:        ${CUT_EPOCHS_DECAY}"
echo "CUT_BATCH_SIZE:   ${CUT_BATCH_SIZE}"
echo "CUT_NUM_THREADS:  ${CUT_NUM_THREADS}"
echo "CUT_NUM_TEST:     ${CUT_NUM_TEST}"
echo "LOG_DIR:          ${LOG_DIR}"
echo "RESET_OUTPUTS:    ${RESET_OUTPUTS}"
echo "========================================="

if [[ "${RESET_OUTPUTS}" == "1" ]]; then
  echo "Reset CUT outputs and intermediate results"
  rm -rf \
    "${SYN_ROOT}/cut/generated/train" \
    "${SYN_ROOT}/cut/generated/val" \
    "${WORK_ROOT}/cut/results/imagenet_ruod_cut_full_ssd_train" \
    "${WORK_ROOT}/cut/results/imagenet_ruod_cut_full_ssd_val" \
    "${WORK_ROOT}/cut/results/imagenet_ruod_cut_full_ssd_train_as_test_train"
fi
MODE=full \
METHODS="cut" \
SPLITS="train val" \
GPU="${GPU}" \
FULL_LIMIT=0 \
SYN_ROOT="${SYN_ROOT}" \
SOURCE_ROOT="${SOURCE_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
RUOD_REF_SRC="${RUOD_REF_SRC}" \
COPY_MODE="${COPY_MODE}" \
bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh \
  2>&1 | tee "${LOG_DIR}/prepare.log"

DATA_NAME="imagenet_ruod_cut_full_ssd"
DATA_ROOT="${WORK_ROOT}/cut/datasets/${DATA_NAME}"

DATA_NAME="${DATA_NAME}" \
DATA_ROOT="${DATA_ROOT}" \
EXP_NAME="${DATA_NAME}" \
GPU_IDS="${GPU_IDS}" \
BATCH_SIZE="${CUT_BATCH_SIZE}" \
NUM_THREADS="${CUT_NUM_THREADS}" \
N_EPOCHS="${CUT_EPOCHS}" \
N_EPOCHS_DECAY="${CUT_EPOCHS_DECAY}" \
SAVE_EPOCH_FREQ=10 \
bash scripts/exp_2/synthesis/run_cut_train.sh \
  2>&1 | tee "${LOG_DIR}/train.log"

DATA_NAME="${DATA_NAME}" \
DATA_ROOT="${DATA_ROOT}" \
EXP_NAME="${DATA_NAME}" \
SPLIT=val \
GPU_IDS="${GPU_IDS}" \
NUM_TEST="${CUT_NUM_TEST}" \
RESULTS_ROOT="${WORK_ROOT}/cut/results/${DATA_NAME}_val" \
RESTORE_DIR="${SYN_ROOT}/cut/generated/val" \
MANIFEST="${DATA_ROOT}/manifests/testA_manifest.jsonl" \
bash scripts/exp_2/synthesis/run_cut_generate.sh \
  2>&1 | tee "${LOG_DIR}/generate_val.log"

TRAIN_GEN_NAME="${DATA_NAME}_train_as_test"
TRAIN_GEN_ROOT="${WORK_ROOT}/cut/datasets/${TRAIN_GEN_NAME}"

DATA_NAME="${TRAIN_GEN_NAME}" \
DATA_ROOT="${TRAIN_GEN_ROOT}" \
TRAIN_A_SOURCE="${WORK_ROOT}/sources/cut/train" \
TEST_A_SOURCE="${WORK_ROOT}/sources/cut/train" \
TRAIN_B_SOURCE="${RUOD_REF_SRC}" \
TEST_B_SOURCE="${RUOD_REF_SRC}" \
TRAIN_A_LIMIT=0 TEST_A_LIMIT=0 TRAIN_B_LIMIT=0 TEST_B_LIMIT=1000 \
LINK_MODE="${COPY_MODE}" \
OVERWRITE=1 \
bash scripts/exp_2/synthesis/run_cut_prepare_dataset.sh \
  2>&1 | tee "${LOG_DIR}/prepare_train_as_test.log"

DATA_NAME="${TRAIN_GEN_NAME}" \
DATA_ROOT="${TRAIN_GEN_ROOT}" \
EXP_NAME="imagenet_ruod_cut_full_ssd" \
SPLIT=train \
GPU_IDS="${GPU_IDS}" \
NUM_TEST="${CUT_NUM_TEST}" \
RESULTS_ROOT="${WORK_ROOT}/cut/results/${DATA_NAME}_train" \
RESTORE_DIR="${SYN_ROOT}/cut/generated/train" \
MANIFEST="${TRAIN_GEN_ROOT}/manifests/testA_manifest.jsonl" \
bash scripts/exp_2/synthesis/run_cut_generate.sh \
  2>&1 | tee "${LOG_DIR}/generate_train.log"

echo "CUT full outputs:"
echo "  ${SYN_ROOT}/cut/generated/train"
echo "  ${SYN_ROOT}/cut/generated/val"
